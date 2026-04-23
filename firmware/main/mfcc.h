/**
 * NeuroEdge — mfcc.h
 *
 * Mel-Frequency Cepstral Coefficients (MFCC) for keyword spotting.
 *
 * Configuration matches the DS-CNN model trained on Google Speech Commands:
 *   Sample rate:    16 000 Hz
 *   Window size:    640 samples (40 ms)
 *   Hop size:       320 samples (20 ms)
 *   FFT size:       512
 *   Mel filterbanks: 40
 *   MFCC coeffs:    10  (coefficients 1-10, skip C0)
 *   Frames:         49  (1 second of audio)
 *   Input tensor:   [1, 49, 10, 1] float32
 *
 * Output layout: row-major float array, dims [NE_MFCC_FRAMES × NE_MFCC_COEFFS]
 * All floats are normalized to zero-mean, unit-variance (per-feature).
 */

#ifndef NE_MFCC_H
#define NE_MFCC_H

#include <stdint.h>
#include <stddef.h>

/* Parameters must match the trained DS-CNN model exactly */
#define NE_MFCC_SAMPLE_RATE     16000
#define NE_MFCC_WINDOW_SAMPLES  640     /* 40 ms */
#define NE_MFCC_HOP_SAMPLES     320     /* 20 ms */
#define NE_MFCC_FFT_SIZE        512
#define NE_MFCC_NUM_FILTERS     40      /* Mel filterbank banks */
#define NE_MFCC_NUM_COEFFS      10      /* Coefficients per frame */
#define NE_MFCC_NUM_FRAMES      49      /* Frames in 1 second */
#define NE_MFCC_FEATURE_SIZE    (NE_MFCC_NUM_FRAMES * NE_MFCC_NUM_COEFFS)  /* 490 */

/* Minimum audio buffer length needed (samples) */
#define NE_MFCC_MIN_SAMPLES     ((NE_MFCC_NUM_FRAMES - 1) * NE_MFCC_HOP_SAMPLES \
                                 + NE_MFCC_WINDOW_SAMPLES)    /* 15680 @ 16 kHz */

/**
 * Opaque MFCC context — allocated once, reused across calls.
 * Holds precomputed Hamming window, mel filterbank, and DCT matrix.
 */
typedef struct ne_mfcc_ctx ne_mfcc_ctx_t;

/**
 * Create an MFCC context. Allocates ~80 KB in PSRAM.
 * Returns NULL on allocation failure.
 */
ne_mfcc_ctx_t *ne_mfcc_create(void);

/**
 * Free an MFCC context.
 */
void ne_mfcc_destroy(ne_mfcc_ctx_t *ctx);

/**
 * Extract MFCC features from a 16-bit PCM audio buffer.
 *
 * @param ctx         MFCC context (from ne_mfcc_create)
 * @param pcm         16-bit PCM samples at NE_MFCC_SAMPLE_RATE Hz, mono
 * @param num_samples Number of samples (must be >= NE_MFCC_MIN_SAMPLES)
 * @param features    Output: float array of size NE_MFCC_FEATURE_SIZE
 *                    Layout: features[frame * NE_MFCC_NUM_COEFFS + coeff]
 * @return            0 on success, -1 on error
 */
int ne_mfcc_compute(ne_mfcc_ctx_t *ctx,
                    const int16_t *pcm,
                    size_t         num_samples,
                    float         *features);

#endif /* NE_MFCC_H */
