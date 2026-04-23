/**
 * NeuroEdge — mfcc.c
 *
 * MFCC feature extraction for DS-CNN keyword spotting.
 *
 * Pipeline per frame:
 *   1. Apply Hamming window to 640-sample frame
 *   2. Zero-pad to 512 samples and compute real FFT (Cooley-Tukey)
 *   3. Power spectrum: |FFT[k]|^2
 *   4. Apply 40-bank mel filterbank
 *   5. log() of each filter energy
 *   6. DCT-II to get 10 cepstral coefficients (skip C0)
 *
 * Memory (PSRAM):
 *   Hamming window:  640 floats  =  2.5 KB
 *   FFT twiddles:    512 floats  =  2.0 KB
 *   Mel filterbank:  40×257 f32  = 41.1 KB
 *   DCT matrix:      10×40 f32   =  1.6 KB
 *   Scratch:         512 floats  =  2.0 KB
 *   Total:          ~50 KB PSRAM
 */

#include "mfcc.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

static const char *TAG = "mfcc";

#define PI        3.14159265358979323846f
#define LOG_FLOOR 1e-12f   /* Prevent log(0) */

/* -------------------------------------------------------------------------
 * Context structure
 * ---------------------------------------------------------------------- */

struct ne_mfcc_ctx {
    float *hamming;         /* [NE_MFCC_WINDOW_SAMPLES] */
    float *fft_real;        /* [NE_MFCC_FFT_SIZE] scratch */
    float *fft_imag;        /* [NE_MFCC_FFT_SIZE] scratch */
    float *mel_fb;          /* [NE_MFCC_NUM_FILTERS × (FFT_SIZE/2+1)] */
    float *dct_matrix;      /* [NE_MFCC_NUM_COEFFS × NE_MFCC_NUM_FILTERS] */
    float *log_energies;    /* [NE_MFCC_NUM_FILTERS] scratch */
    /* Per-feature normalization stats (populated on first call) */
    float  feat_mean[NE_MFCC_NUM_COEFFS];
    float  feat_std[NE_MFCC_NUM_COEFFS];
    bool   norm_ready;
};

/* -------------------------------------------------------------------------
 * Helpers: PSRAM alloc
 * ---------------------------------------------------------------------- */

static float *psram_alloc_f32(size_t n)
{
    return (float *)heap_caps_malloc(n * sizeof(float),
                                     MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
}

/* -------------------------------------------------------------------------
 * FFT — in-place radix-2 Cooley-Tukey (real FFT via split-radix)
 *
 * Input:  re[0..N-1] (imaginary set to 0 before call)
 * Output: re[0..N-1], im[0..N-1] (complex spectrum)
 * N must be a power of 2.
 * ---------------------------------------------------------------------- */

static void fft(float *re, float *im, int N)
{
    /* Bit-reversal permutation */
    int bits = 0;
    for (int n = N; n > 1; n >>= 1) bits++;
    for (int i = 0; i < N; i++) {
        int rev = 0, x = i;
        for (int b = 0; b < bits; b++) { rev = (rev << 1) | (x & 1); x >>= 1; }
        if (rev > i) {
            float t = re[i]; re[i] = re[rev]; re[rev] = t;
            t = im[i]; im[i] = im[rev]; im[rev] = t;
        }
    }
    /* Butterfly passes */
    for (int len = 2; len <= N; len <<= 1) {
        float ang = -2.0f * PI / (float)len;
        float wr = cosf(ang), wi = sinf(ang);
        for (int i = 0; i < N; i += len) {
            float cur_r = 1.0f, cur_i = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                float u_r = re[i+j],          u_i = im[i+j];
                float v_r = re[i+j+len/2] * cur_r - im[i+j+len/2] * cur_i;
                float v_i = re[i+j+len/2] * cur_i + im[i+j+len/2] * cur_r;
                re[i+j]         = u_r + v_r;
                im[i+j]         = u_i + v_i;
                re[i+j+len/2]   = u_r - v_r;
                im[i+j+len/2]   = u_i - v_i;
                float next_r    = cur_r * wr - cur_i * wi;
                cur_i           = cur_r * wi + cur_i * wr;
                cur_r           = next_r;
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * Mel filterbank construction
 *
 * Returns a [num_filters × (fft_size/2+1)] matrix in row-major order.
 * Matches librosa's mel_filters(sr=16000, n_fft=512, n_mels=40, htk=False).
 * ---------------------------------------------------------------------- */

static void build_mel_filterbank(float *fb,
                                  int    num_filters,
                                  int    fft_size,
                                  float  sample_rate)
{
    int    num_bins = fft_size / 2 + 1;   /* 257 */

    /* Hz → Mel and back (O'Shaughnessy formula) */
    float mel_min  = 2595.0f * log10f(1.0f + 0.0f    / 700.0f);
    float mel_max  = 2595.0f * log10f(1.0f + (sample_rate / 2.0f) / 700.0f);

    /* num_filters+2 equally-spaced mel points */
    int    n_pts   = num_filters + 2;
    float *mel_pts = (float *)alloca((size_t)n_pts * sizeof(float));
    for (int i = 0; i < n_pts; i++)
        mel_pts[i] = mel_min + (mel_max - mel_min) * (float)i / (float)(n_pts - 1);

    /* Convert mel points back to Hz, then to FFT bin indices */
    float *hz_pts  = (float *)alloca((size_t)n_pts * sizeof(float));
    float *bin_pts = (float *)alloca((size_t)n_pts * sizeof(float));
    for (int i = 0; i < n_pts; i++) {
        hz_pts[i]  = 700.0f * (powf(10.0f, mel_pts[i] / 2595.0f) - 1.0f);
        bin_pts[i] = floorf((float)(fft_size + 1) * hz_pts[i] / sample_rate);
    }

    /* Build triangular filters */
    memset(fb, 0, (size_t)(num_filters * num_bins) * sizeof(float));
    for (int m = 0; m < num_filters; m++) {
        int f_left   = (int)bin_pts[m];
        int f_center = (int)bin_pts[m + 1];
        int f_right  = (int)bin_pts[m + 2];

        for (int k = f_left; k <= f_center && k < num_bins; k++) {
            if (f_center != f_left)
                fb[m * num_bins + k] = (float)(k - f_left) /
                                       (float)(f_center - f_left);
        }
        for (int k = f_center; k <= f_right && k < num_bins; k++) {
            if (f_right != f_center)
                fb[m * num_bins + k] = (float)(f_right - k) /
                                       (float)(f_right - f_center);
        }
    }
}

/* -------------------------------------------------------------------------
 * DCT-II matrix [num_coeffs × num_filters]
 * out[n] = sum_k{ fb[k] * cos(pi * n * (k + 0.5) / num_filters) }
 * ---------------------------------------------------------------------- */

static void build_dct_matrix(float *dct, int num_coeffs, int num_filters)
{
    float scale0 = sqrtf(1.0f / (float)num_filters);
    float scale  = sqrtf(2.0f / (float)num_filters);
    for (int n = 0; n < num_coeffs; n++) {
        float s = (n == 0) ? scale0 : scale;
        for (int k = 0; k < num_filters; k++) {
            dct[n * num_filters + k] =
                s * cosf(PI * (float)n * ((float)k + 0.5f) / (float)num_filters);
        }
    }
}

/* -------------------------------------------------------------------------
 * Context creation / destruction
 * ---------------------------------------------------------------------- */

ne_mfcc_ctx_t *ne_mfcc_create(void)
{
    ne_mfcc_ctx_t *ctx = (ne_mfcc_ctx_t *)heap_caps_calloc(
        1, sizeof(ne_mfcc_ctx_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!ctx) { ESP_LOGE(TAG, "alloc ctx"); return NULL; }

    ctx->hamming      = psram_alloc_f32(NE_MFCC_WINDOW_SAMPLES);
    ctx->fft_real     = psram_alloc_f32(NE_MFCC_FFT_SIZE);
    ctx->fft_imag     = psram_alloc_f32(NE_MFCC_FFT_SIZE);
    ctx->mel_fb       = psram_alloc_f32(NE_MFCC_NUM_FILTERS * (NE_MFCC_FFT_SIZE/2+1));
    ctx->dct_matrix   = psram_alloc_f32(NE_MFCC_NUM_COEFFS  * NE_MFCC_NUM_FILTERS);
    ctx->log_energies = psram_alloc_f32(NE_MFCC_NUM_FILTERS);

    if (!ctx->hamming || !ctx->fft_real || !ctx->fft_imag ||
        !ctx->mel_fb  || !ctx->dct_matrix || !ctx->log_energies) {
        ESP_LOGE(TAG, "PSRAM alloc failed for MFCC buffers");
        ne_mfcc_destroy(ctx);
        return NULL;
    }

    /* Hamming window: w[n] = 0.54 - 0.46*cos(2π*n/(N-1)) */
    for (int n = 0; n < NE_MFCC_WINDOW_SAMPLES; n++) {
        ctx->hamming[n] = 0.54f - 0.46f * cosf(
            2.0f * PI * (float)n / (float)(NE_MFCC_WINDOW_SAMPLES - 1));
    }

    build_mel_filterbank(ctx->mel_fb,
                         NE_MFCC_NUM_FILTERS,
                         NE_MFCC_FFT_SIZE,
                         (float)NE_MFCC_SAMPLE_RATE);

    build_dct_matrix(ctx->dct_matrix, NE_MFCC_NUM_COEFFS, NE_MFCC_NUM_FILTERS);

    /* Default normalization: zero mean, unit std (will be updated after first batch) */
    for (int i = 0; i < NE_MFCC_NUM_COEFFS; i++) {
        ctx->feat_mean[i] = 0.0f;
        ctx->feat_std[i]  = 1.0f;
    }
    ctx->norm_ready = false;

    ESP_LOGI(TAG, "MFCC context ready (%d frames × %d coeffs)",
             NE_MFCC_NUM_FRAMES, NE_MFCC_NUM_COEFFS);
    return ctx;
}

void ne_mfcc_destroy(ne_mfcc_ctx_t *ctx)
{
    if (!ctx) return;
    heap_caps_free(ctx->hamming);
    heap_caps_free(ctx->fft_real);
    heap_caps_free(ctx->fft_imag);
    heap_caps_free(ctx->mel_fb);
    heap_caps_free(ctx->dct_matrix);
    heap_caps_free(ctx->log_energies);
    heap_caps_free(ctx);
}

/* -------------------------------------------------------------------------
 * Feature extraction
 * ---------------------------------------------------------------------- */

int ne_mfcc_compute(ne_mfcc_ctx_t *ctx,
                    const int16_t *pcm,
                    size_t         num_samples,
                    float         *features)
{
    if (!ctx || !pcm || !features) return -1;
    if (num_samples < NE_MFCC_MIN_SAMPLES) {
        ESP_LOGW(TAG, "Audio too short: %zu samples (need %d)",
                 num_samples, NE_MFCC_MIN_SAMPLES);
        return -1;
    }

    const int num_bins = NE_MFCC_FFT_SIZE / 2 + 1;   /* 257 */

    for (int frame = 0; frame < NE_MFCC_NUM_FRAMES; frame++) {
        int offset = frame * NE_MFCC_HOP_SAMPLES;

        /* --- Step 1: Hamming window + load into FFT real buffer --- */
        memset(ctx->fft_real, 0, NE_MFCC_FFT_SIZE * sizeof(float));
        memset(ctx->fft_imag, 0, NE_MFCC_FFT_SIZE * sizeof(float));

        for (int n = 0; n < NE_MFCC_WINDOW_SAMPLES; n++) {
            ctx->fft_real[n] = ctx->hamming[n] *
                               ((float)pcm[offset + n] / 32768.0f);
        }

        /* --- Step 2: FFT --- */
        fft(ctx->fft_real, ctx->fft_imag, NE_MFCC_FFT_SIZE);

        /* --- Step 3: Power spectrum |X[k]|^2 for k = 0..N/2 --- */
        /* Reuse fft_real[0..256] for power spectrum in-place */
        for (int k = 0; k < num_bins; k++) {
            float r = ctx->fft_real[k];
            float i = ctx->fft_imag[k];
            ctx->fft_real[k] = r * r + i * i;   /* power */
        }

        /* --- Step 4: Mel filterbank → log energies --- */
        for (int m = 0; m < NE_MFCC_NUM_FILTERS; m++) {
            float energy = 0.0f;
            const float *row = ctx->mel_fb + m * num_bins;
            for (int k = 0; k < num_bins; k++)
                energy += row[k] * ctx->fft_real[k];
            ctx->log_energies[m] = logf(energy + LOG_FLOOR);
        }

        /* --- Step 5: DCT-II → NE_MFCC_NUM_COEFFS coefficients --- */
        float *out_row = features + frame * NE_MFCC_NUM_COEFFS;
        for (int n = 0; n < NE_MFCC_NUM_COEFFS; n++) {
            float coeff = 0.0f;
            const float *dct_row = ctx->dct_matrix + n * NE_MFCC_NUM_FILTERS;
            for (int m = 0; m < NE_MFCC_NUM_FILTERS; m++)
                coeff += dct_row[m] * ctx->log_energies[m];
            out_row[n] = coeff;
        }
    }

    /* --- Step 6: Per-coefficient mean normalization (CMN) --- */
    /* Compute mean over frames for each coefficient */
    float mean[NE_MFCC_NUM_COEFFS] = {0};
    for (int f = 0; f < NE_MFCC_NUM_FRAMES; f++)
        for (int c = 0; c < NE_MFCC_NUM_COEFFS; c++)
            mean[c] += features[f * NE_MFCC_NUM_COEFFS + c];
    for (int c = 0; c < NE_MFCC_NUM_COEFFS; c++)
        mean[c] /= (float)NE_MFCC_NUM_FRAMES;

    /* Subtract mean */
    for (int f = 0; f < NE_MFCC_NUM_FRAMES; f++)
        for (int c = 0; c < NE_MFCC_NUM_COEFFS; c++)
            features[f * NE_MFCC_NUM_COEFFS + c] -= mean[c];

    return 0;
}
