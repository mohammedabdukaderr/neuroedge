/**
 * NeuroEdge — model_runner.c  (Day 3-6: llama.cpp + TFLite Micro)
 *
 * Architecture:
 *   LLM path (TinyLlama / any GGUF):
 *     1. model_loader_open() — mmap GGUF from flash partition
 *     2. llama_model_load_from_memory() — load into PSRAM
 *     3. llama_new_context_with_model() — inference context
 *     4. Per-request: tokenize → eval → sample → detokenize
 *
 *   TFLite path (DS-CNN KWS / MobileNet):
 *     1. model_loader_open() — mmap TFLite flatbuffer from flash
 *     2. kws_init() / tflite_classify_init() — allocate tensor arena in PSRAM
 *     3. Per-request (KWS): ne_mfcc_compute() → kws_infer()
 *     4. Per-request (classify): preprocess → interpreter->Invoke()
 *
 * Memory budget on ESP32-S3 with 8 MB PSRAM:
 *   - GGUF model weights (mmap'd from flash, zero PSRAM): ~0
 *   - llama context (KV cache + compute): ~3-4 MB PSRAM
 *   - ggml scratch buffer: 4 MB PSRAM
 *   - Remaining for FW + OS: ~1-2 MB internal SRAM
 *
 * Thread model: single-threaded (called only from uart_server task).
 */

#include "model_runner.h"
#include "model_loader.h"
#include "mfcc.h"
#include "kws.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* llama.cpp headers — available after git clone into components/llama_cpp */
#if defined(CONFIG_NE_USE_LLAMA_CPP) || defined(NE_LLAMA_CPP_AVAILABLE)
#  include "llama.h"
#  include "common.h"
#  include "sampling.h"
#  define LLAMA_AVAILABLE 1
#else
#  define LLAMA_AVAILABLE 0
   /* Forward-declare minimal stubs so this file compiles without llama.cpp */
   typedef void llama_model;
   typedef void llama_context;
   typedef int  llama_token;
#endif

static const char *TAG = "model_runner";

/* -------------------------------------------------------------------------
 * Configuration constants
 * ---------------------------------------------------------------------- */

#define LLM_N_CTX           512     /* Context window (tokens) — balance mem vs quality */
#define LLM_N_BATCH         32      /* Batch size for prompt processing */
#define LLM_N_THREADS       1       /* Single-threaded on ESP32-S3 */
#define LLM_N_PREDICT       128     /* Max generated tokens per request */
#define LLM_TEMP            0.8f    /* Sampling temperature */
#define LLM_TOP_K           40      /* Top-K sampling */
#define LLM_TOP_P           0.95f   /* Top-P (nucleus) sampling */
#define LLM_REPEAT_PENALTY  1.1f    /* Repetition penalty */

/* Tokens that terminate generation early */
static const char * const STOP_TOKENS[] = {
    "\n\n", "User:", "Human:", "</s>", "[/INST]", NULL
};

/* -------------------------------------------------------------------------
 * State
 * ---------------------------------------------------------------------- */

typedef struct {
    bool                initialized;
    ne_model_id_t       active_model;
    ne_model_region_t   region;         /* Mmap'd flash region (primary model) */
    ne_model_region_t   kws_region;     /* Mmap'd flash region (KWS model)     */
    ne_mfcc_ctx_t      *mfcc_ctx;       /* MFCC feature extractor              */
    float              *mfcc_features;  /* [NE_MFCC_FEATURE_SIZE] in PSRAM     */

#if LLAMA_AVAILABLE
    llama_model        *llm_model;
    llama_context      *llm_ctx;
    struct llama_sampler *llm_sampler;
#endif
} model_runner_state_t;

static model_runner_state_t s_state = { 0 };

/* -------------------------------------------------------------------------
 * Memory helpers
 * ---------------------------------------------------------------------- */

static void log_psram_stats(const char *label)
{
    size_t free_psram    = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    ESP_LOGI(TAG, "[%s] PSRAM free: %zu KB, internal free: %zu KB",
             label, free_psram / 1024, free_internal / 1024);
}

/* -------------------------------------------------------------------------
 * LLM init / deinit
 * ---------------------------------------------------------------------- */

#if LLAMA_AVAILABLE

static esp_err_t llm_init(const ne_model_region_t *region)
{
    log_psram_stats("before llm_init");

    /* ggml backend init — CPU only, no GPU */
    llama_backend_init();

    /* Model params — use mmap pointer directly */
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers  = 0;       /* CPU only */
    mparams.use_mmap      = true;    /* Use our flash mmap */
    mparams.use_mlock     = false;   /* Cannot mlock on ESP32 */
    mparams.vocab_only    = false;

    /*
     * llama_load_model_from_memory() — loads a GGUF model from an in-memory
     * buffer. Available in llama.cpp builds since mid-2024.
     *
     * The mmap_ptr points directly to GGUF data in the flash address space.
     * llama.cpp reads tensors directly; weights stay in flash (XIP).
     * Only the KV cache and compute buffers land in PSRAM.
     */
    s_state.llm_model = llama_model_load_from_memory(
        (const uint8_t *)region->mmap_ptr,
        (size_t)region->data_size,
        &mparams);

    if (!s_state.llm_model) {
        ESP_LOGE(TAG, "llama_model_load_from_memory failed — check PSRAM");
        llama_backend_free();
        return ESP_ERR_NO_MEM;
    }

    /* Context params */
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = LLM_N_CTX;
    cparams.n_batch    = LLM_N_BATCH;
    cparams.n_threads  = LLM_N_THREADS;
    cparams.type_k     = GGML_TYPE_F16;
    cparams.type_v     = GGML_TYPE_F16;
    cparams.flash_attn = false;   /* Not available on CPU backend */

    s_state.llm_ctx = llama_new_context_with_model(s_state.llm_model, cparams);
    if (!s_state.llm_ctx) {
        ESP_LOGE(TAG, "llama_new_context_with_model failed — out of PSRAM");
        llama_model_free(s_state.llm_model);
        s_state.llm_model = NULL;
        llama_backend_free();
        return ESP_ERR_NO_MEM;
    }

    /* Sampler chain: temperature + top-k + top-p + repetition penalty */
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    s_state.llm_sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(s_state.llm_sampler,
        llama_sampler_init_temp(LLM_TEMP));
    llama_sampler_chain_add(s_state.llm_sampler,
        llama_sampler_init_top_k(LLM_TOP_K));
    llama_sampler_chain_add(s_state.llm_sampler,
        llama_sampler_init_top_p(LLM_TOP_P, 1));
    llama_sampler_chain_add(s_state.llm_sampler,
        llama_sampler_init_penalties(
            llama_model_n_vocab(s_state.llm_model),
            llama_token_eos(s_state.llm_model),
            llama_token_nl(s_state.llm_model),
            64,                /* Last N tokens for penalty */
            LLM_REPEAT_PENALTY,
            0.0f, 0.0f, false, false));

    log_psram_stats("after llm_init");
    ESP_LOGI(TAG, "LLM ready: %s  ctx=%d  batch=%d  threads=%d",
             region->model_id, LLM_N_CTX, LLM_N_BATCH, LLM_N_THREADS);
    return ESP_OK;
}

static void llm_deinit(void)
{
    if (s_state.llm_sampler) {
        llama_sampler_free(s_state.llm_sampler);
        s_state.llm_sampler = NULL;
    }
    if (s_state.llm_ctx) {
        llama_free(s_state.llm_ctx);
        s_state.llm_ctx = NULL;
    }
    if (s_state.llm_model) {
        llama_model_free(s_state.llm_model);
        s_state.llm_model = NULL;
    }
    llama_backend_free();
    ESP_LOGI(TAG, "LLM deinitialized");
}

/* -------------------------------------------------------------------------
 * LLM inference
 * ---------------------------------------------------------------------- */

static bool should_stop_generation(const char *text, size_t len)
{
    for (int i = 0; STOP_TOKENS[i]; i++) {
        size_t slen = strlen(STOP_TOKENS[i]);
        if (len >= slen && memcmp(text + len - slen, STOP_TOKENS[i], slen) == 0)
            return true;
    }
    return false;
}

static esp_err_t llm_generate(const char    *prompt,
                               uint32_t       max_tokens,
                               mr_llm_result_t *result)
{
    if (!s_state.llm_ctx || !s_state.llm_model) return ESP_ERR_INVALID_STATE;

    int64_t t_start = esp_timer_get_time();

    /* Build instruct-format prompt */
    char full_prompt[MR_MAX_PROMPT_LEN + 64];
    snprintf(full_prompt, sizeof(full_prompt),
             "<|system|>\nYou are NeuroEdge, a concise embedded AI assistant.\n"
             "<|user|>\n%s\n<|assistant|>\n", prompt);

    /* Tokenize */
    const int vocab_size = llama_model_n_vocab(s_state.llm_model);
    const int max_tokens_in = LLM_N_CTX - (int)max_tokens - 4;

    llama_token *tokens = (llama_token *)heap_caps_malloc(
        (size_t)(max_tokens_in + 4) * sizeof(llama_token),
        MALLOC_CAP_SPIRAM);
    if (!tokens) return ESP_ERR_NO_MEM;

    int n_tokens = llama_tokenize(s_state.llm_model,
                                  full_prompt, (int)strlen(full_prompt),
                                  tokens, max_tokens_in,
                                  true,   /* add_special (BOS) */
                                  false); /* parse_special */
    if (n_tokens < 0) {
        heap_caps_free(tokens);
        ESP_LOGE(TAG, "Tokenize failed (prompt too long?)");
        return ESP_ERR_INVALID_SIZE;
    }

    /* Create batch and evaluate prompt */
    llama_batch batch = llama_batch_get_one(tokens, n_tokens);

    if (llama_decode(s_state.llm_ctx, batch) != 0) {
        heap_caps_free(tokens);
        ESP_LOGE(TAG, "llama_decode (prompt) failed");
        return ESP_FAIL;
    }
    heap_caps_free(tokens);

    /* Generation loop */
    memset(result->text, 0, MR_MAX_RESPONSE_LEN);
    size_t text_len   = 0;
    int    n_generated = 0;
    bool   stopped    = false;

    /* Single-token buffer for decode */
    llama_token out_token[1];

    for (int i = 0; i < (int)max_tokens && !stopped; i++) {
        /* Sample next token */
        out_token[0] = llama_sampler_sample(s_state.llm_sampler, s_state.llm_ctx, -1);

        if (out_token[0] == llama_token_eos(s_state.llm_model)) break;

        /* Convert token to string */
        char piece[32];
        int  piece_len = llama_token_to_piece(s_state.llm_model,
                                              out_token[0],
                                              piece, sizeof(piece) - 1,
                                              0, false);
        if (piece_len < 0) piece_len = 0;
        piece[piece_len] = '\0';

        /* Append to result */
        if (text_len + (size_t)piece_len < MR_MAX_RESPONSE_LEN - 1) {
            memcpy(result->text + text_len, piece, (size_t)piece_len);
            text_len += (size_t)piece_len;
            result->text[text_len] = '\0';
        } else {
            break;  /* Output buffer full */
        }

        /* Check stop tokens */
        if (should_stop_generation(result->text, text_len)) {
            /* Trim the stop token from output */
            for (int s = 0; STOP_TOKENS[s]; s++) {
                size_t slen = strlen(STOP_TOKENS[s]);
                if (text_len >= slen &&
                    memcmp(result->text + text_len - slen, STOP_TOKENS[s], slen) == 0) {
                    text_len -= slen;
                    result->text[text_len] = '\0';
                }
            }
            stopped = true;
            break;
        }

        /* Feed token back for next step */
        llama_batch next_batch = llama_batch_get_one(out_token, 1);
        if (llama_decode(s_state.llm_ctx, next_batch) != 0) {
            ESP_LOGW(TAG, "llama_decode (token %d) failed — stopping", i);
            break;
        }
        n_generated++;
    }

    /* Reset KV cache for next request */
    llama_kv_cache_clear(s_state.llm_ctx);
    llama_sampler_reset(s_state.llm_sampler);

    /* Trim trailing whitespace */
    while (text_len > 0 && (result->text[text_len-1] == ' ' ||
                              result->text[text_len-1] == '\n' ||
                              result->text[text_len-1] == '\r')) {
        result->text[--text_len] = '\0';
    }

    result->latency_ms = (uint32_t)((esp_timer_get_time() - t_start) / 1000ULL);
    ESP_LOGI(TAG, "Generated %d tokens in %u ms (%.1f tok/s)",
             n_generated, result->latency_ms,
             n_generated > 0
                 ? (float)n_generated / ((float)result->latency_ms / 1000.0f)
                 : 0.0f);

    return ESP_OK;
}

#endif /* LLAMA_AVAILABLE */

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

esp_err_t model_runner_init(void)
{
    if (s_state.initialized) {
        ESP_LOGW(TAG, "Already initialized");
        return ESP_OK;
    }

    log_psram_stats("startup");

    /* Open the primary model partition */
    esp_err_t err = model_loader_open("model_0", &s_state.region);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "model_0 partition open failed (%s) — running without model",
                 esp_err_to_name(err));
        /* Continue in degraded mode — commands will return errors */
        s_state.active_model = MODEL_NONE;
        s_state.initialized  = true;
        return ESP_OK;
    }

    /* Route to appropriate backend based on model type */
    switch (s_state.region.type) {
        case MODEL_TYPE_GGUF:
#if LLAMA_AVAILABLE
            err = llm_init(&s_state.region);
            if (err != ESP_OK) {
                model_loader_close(&s_state.region);
                return err;
            }
            s_state.active_model = MODEL_TINYLLAMA;
#else
            ESP_LOGW(TAG, "GGUF model found but llama.cpp not compiled in");
            ESP_LOGW(TAG, "Add CONFIG_NE_USE_LLAMA_CPP=y to sdkconfig");
            s_state.active_model = MODEL_NONE;
#endif
            break;

        case MODEL_TYPE_TFLITE:
            /* Determine if this is a KWS or classifier model by model_id */
            if (strncmp(s_state.region.model_id, "dscnn", 5) == 0) {
                /* DS-CNN keyword spotting */
                s_state.mfcc_ctx = ne_mfcc_create();
                if (!s_state.mfcc_ctx) {
                    model_loader_close(&s_state.region);
                    return ESP_ERR_NO_MEM;
                }

                s_state.mfcc_features = (float *)heap_caps_malloc(
                    NE_MFCC_FEATURE_SIZE * sizeof(float),
                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
                if (!s_state.mfcc_features) {
                    ne_mfcc_destroy(s_state.mfcc_ctx);
                    s_state.mfcc_ctx = NULL;
                    model_loader_close(&s_state.region);
                    return ESP_ERR_NO_MEM;
                }

                kws_config_t kws_cfg = { .detection_threshold = KWS_DEFAULT_THRESHOLD };
                err = kws_init(s_state.region.mmap_ptr,
                               s_state.region.data_size,
                               &kws_cfg);
                if (err != ESP_OK) {
                    heap_caps_free(s_state.mfcc_features);
                    s_state.mfcc_features = NULL;
                    ne_mfcc_destroy(s_state.mfcc_ctx);
                    s_state.mfcc_ctx = NULL;
                    model_loader_close(&s_state.region);
                    return err;
                }
                s_state.active_model = MODEL_DSCNN_KWS;
            } else {
                /* MobileNet or other TFLite classifier — TODO Day 6 */
                ESP_LOGW(TAG, "MobileNet classify not yet implemented");
                s_state.active_model = MODEL_MOBILENET;
            }
            break;

        default:
            ESP_LOGE(TAG, "Unknown model type %u", s_state.region.type);
            model_loader_close(&s_state.region);
            return ESP_ERR_NOT_SUPPORTED;
    }

    s_state.initialized = true;
    ESP_LOGI(TAG, "Model runner ready: %s (%s)",
             s_state.region.model_id,
             model_loader_type_name(s_state.region.type));
    return ESP_OK;
}

void model_runner_deinit(void)
{
    if (!s_state.initialized) return;

#if LLAMA_AVAILABLE
    llm_deinit();
#endif
    kws_deinit();

    if (s_state.mfcc_features) {
        heap_caps_free(s_state.mfcc_features);
        s_state.mfcc_features = NULL;
    }
    if (s_state.mfcc_ctx) {
        ne_mfcc_destroy(s_state.mfcc_ctx);
        s_state.mfcc_ctx = NULL;
    }

    model_loader_close(&s_state.region);
    model_loader_close(&s_state.kws_region);
    s_state.active_model = MODEL_NONE;
    s_state.initialized  = false;
    ESP_LOGI(TAG, "Model runner deinitialized");
}

esp_err_t model_runner_set_model(ne_model_id_t model_id)
{
    if (model_id == MODEL_NONE || model_id > MODEL_DSCNN_KWS)
        return ESP_ERR_NOT_FOUND;

    /* Determine which partition to load from */
    const char *partition = "model_0";
    if (model_id == MODEL_DSCNN_KWS) partition = "model_0";  /* Adjust per layout */

    /* Deinit current */
#if LLAMA_AVAILABLE
    llm_deinit();
#endif
    model_loader_close(&s_state.region);
    s_state.active_model = MODEL_NONE;

    /* Load new */
    esp_err_t err = model_loader_open(partition, &s_state.region);
    if (err != ESP_OK) return err;

#if LLAMA_AVAILABLE
    if (s_state.region.type == MODEL_TYPE_GGUF) {
        err = llm_init(&s_state.region);
        if (err != ESP_OK) {
            model_loader_close(&s_state.region);
            return err;
        }
    }
#endif

    s_state.active_model = model_id;
    ESP_LOGI(TAG, "Switched to model: %s", model_runner_active_name());
    return ESP_OK;
}

const char *model_runner_active_name(void)
{
    if (s_state.region.model_id[0] != '\0')
        return s_state.region.model_id;

    switch (s_state.active_model) {
        case MODEL_TINYLLAMA: return "tinyllama-q4";
        case MODEL_MOBILENET: return "mobilenet-v3";
        case MODEL_DSCNN_KWS: return "dscnn-kws";
        default:              return "none";
    }
}

esp_err_t model_runner_ask(const char      *prompt,
                           uint32_t         max_tokens,
                           mr_llm_result_t *result)
{
    if (!s_state.initialized) return ESP_ERR_INVALID_STATE;
    if (!prompt || !result)   return ESP_ERR_INVALID_ARG;

#if LLAMA_AVAILABLE
    if (s_state.llm_ctx) {
        return llm_generate(prompt, max_tokens, result);
    }
#endif

    /* Fallback: model not loaded */
    snprintf(result->text, MR_MAX_RESPONSE_LEN,
             "[no model loaded — flash a GGUF model to partition model_0]");
    result->latency_ms = 0;
    return ESP_OK;
}

esp_err_t model_runner_classify(const uint8_t        *data,
                                size_t                data_len,
                                mr_classify_result_t *result)
{
    if (!s_state.initialized) return ESP_ERR_INVALID_STATE;
    if (!data || !result)     return ESP_ERR_INVALID_ARG;

    /* TODO(tflite Day 5-6): route to TFLite interpreter */
    snprintf(result->label, MR_MAX_LABEL_LEN, "unknown");
    result->confidence = 0.0f;
    result->latency_ms = 0;
    return ESP_OK;
}

esp_err_t model_runner_detect_keyword(const int16_t  *pcm,
                                      size_t          num_samples,
                                      mr_kws_result_t *result)
{
    if (!s_state.initialized) return ESP_ERR_INVALID_STATE;
    if (!pcm || !result)      return ESP_ERR_INVALID_ARG;

    if (!s_state.mfcc_ctx || !s_state.mfcc_features) {
        /* KWS model not loaded — guide the user */
        ESP_LOGW(TAG, "KWS not initialized — flash dscnn-kws model to model_0");
        result->detected   = false;
        result->keyword[0] = '\0';
        result->confidence = 0.0f;
        result->latency_ms = 0;
        return ESP_OK;
    }

    int64_t t_start = esp_timer_get_time();

    /* Step 1: Extract MFCC features */
    int rc = ne_mfcc_compute(s_state.mfcc_ctx, pcm, num_samples,
                              s_state.mfcc_features);
    if (rc != 0) {
        ESP_LOGE(TAG, "MFCC extraction failed");
        return ESP_ERR_INVALID_SIZE;
    }

    /* Step 2: DS-CNN inference */
    kws_result_t kws_out = {0};
    esp_err_t err = kws_infer(s_state.mfcc_features, &kws_out);
    if (err != ESP_OK) return err;

    /* Map kws_result_t → mr_kws_result_t */
    result->detected   = kws_out.detected;
    result->confidence = kws_out.confidence;
    result->latency_ms = (uint32_t)((esp_timer_get_time() - t_start) / 1000ULL);
    strncpy(result->keyword, kws_out.keyword, MR_MAX_LABEL_LEN - 1);
    result->keyword[MR_MAX_LABEL_LEN - 1] = '\0';

    return ESP_OK;
}
