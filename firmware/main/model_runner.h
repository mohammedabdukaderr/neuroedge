/**
 * NeuroEdge — model_runner.h
 *
 * Abstracts inference backends: llama.cpp (LLM), TFLite Micro (classifier /
 * keyword spotting). The UART server calls these functions; the rest of the
 * firmware does not need to know which backend is active.
 */

#ifndef NE_MODEL_RUNNER_H
#define NE_MODEL_RUNNER_H

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

/* Maximum sizes */
#define MR_MAX_PROMPT_LEN    1024
#define MR_MAX_RESPONSE_LEN  2048
#define MR_MAX_LABEL_LEN     64
#define MR_MAX_DATA_LEN      512

/* Model identifiers */
typedef enum {
    MODEL_NONE       = 0,
    MODEL_TINYLLAMA  = 1,   /* TinyLlama 1.1B Q4_K_M via llama.cpp     */
    MODEL_MOBILENET  = 2,   /* MobileNet v3 via TFLite Micro            */
    MODEL_DSCNN_KWS  = 3,   /* DS-CNN keyword spotting via TFLite Micro */
} ne_model_id_t;

/* Result structures */
typedef struct {
    char     text[MR_MAX_RESPONSE_LEN];
    uint32_t latency_ms;
} mr_llm_result_t;

typedef struct {
    char     label[MR_MAX_LABEL_LEN];
    float    confidence;
    uint32_t latency_ms;
} mr_classify_result_t;

typedef struct {
    bool     detected;
    char     keyword[MR_MAX_LABEL_LEN];
    float    confidence;
    uint32_t latency_ms;
} mr_kws_result_t;

/**
 * Initialize the model runner subsystem.
 * Loads the model(s) stored in flash into memory.
 * Must be called once at firmware startup.
 *
 * @return ESP_OK on success
 */
esp_err_t model_runner_init(void);

/**
 * Switch the active LLM/classifier model at runtime.
 *
 * @param model_id   One of the ne_model_id_t values
 * @return           ESP_OK, or ESP_ERR_NOT_FOUND if model not flashed
 */
esp_err_t model_runner_set_model(ne_model_id_t model_id);

/**
 * Run LLM inference (TinyLlama).
 *
 * @param prompt      Null-terminated input string
 * @param max_tokens  Maximum tokens to generate
 * @param result      Output filled on success
 * @return            ESP_OK on success
 */
esp_err_t model_runner_ask(const char    *prompt,
                           uint32_t       max_tokens,
                           mr_llm_result_t *result);

/**
 * Run TFLite Micro classifier on raw sensor data.
 *
 * @param data        Raw byte buffer
 * @param data_len    Number of bytes
 * @param result      Output filled on success
 * @return            ESP_OK on success
 */
esp_err_t model_runner_classify(const uint8_t       *data,
                                size_t               data_len,
                                mr_classify_result_t *result);

/**
 * Run DS-CNN keyword spotting on a PCM audio buffer.
 *
 * @param pcm         16-bit PCM samples, 16 kHz mono
 * @param num_samples Number of samples
 * @param result      Output filled on success
 * @return            ESP_OK on success
 */
esp_err_t model_runner_detect_keyword(const int16_t  *pcm,
                                      size_t          num_samples,
                                      mr_kws_result_t *result);

/**
 * Return the name of the currently loaded model (for the info command).
 */
const char *model_runner_active_name(void);

/**
 * Free all model resources. Called before model_runner_set_model() or at
 * shutdown.
 */
void model_runner_deinit(void);

#endif /* NE_MODEL_RUNNER_H */
