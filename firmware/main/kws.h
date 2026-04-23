/**
 * NeuroEdge — kws.h
 *
 * Keyword Spotting (KWS) using DS-CNN model via TFLite Micro.
 *
 * Supported keywords (Google Speech Commands v2 subset):
 *   yes, no, up, down, left, right, on, off, stop, go
 *   + silence, unknown
 *
 * Input to model:  [1, 49, 10, 1] float32 (MFCC features)
 * Output of model: [1, 12] float32 (softmax probabilities)
 *
 * Detection threshold: 0.85 (configurable in kws_config_t)
 */

#ifndef NE_KWS_H
#define NE_KWS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "esp_err.h"

/* Keywords supported by the DS-CNN model */
#define KWS_NUM_CLASSES  12
#define KWS_LABEL_LEN    16

static const char * const KWS_LABELS[KWS_NUM_CLASSES] = {
    "silence", "unknown",
    "yes", "no", "up", "down",
    "left", "right", "on", "off",
    "stop", "go",
};

/* Detection threshold — probability above this is a positive detection */
#define KWS_DEFAULT_THRESHOLD  0.85f

/* Tensor arena size in PSRAM for DS-CNN (~100 KB) */
#define KWS_TENSOR_ARENA_SIZE  (120 * 1024)

typedef struct {
    float detection_threshold;  /* 0.0–1.0 (default: KWS_DEFAULT_THRESHOLD) */
} kws_config_t;

typedef struct {
    bool     detected;
    char     keyword[KWS_LABEL_LEN];
    int      class_idx;
    float    confidence;
    uint32_t latency_ms;
} kws_result_t;

/**
 * Initialize the KWS engine.
 *
 * @param model_data  Pointer to TFLite flatbuffer data (from flash mmap)
 * @param model_size  Size in bytes
 * @param config      Configuration, or NULL for defaults
 * @return            ESP_OK on success
 */
esp_err_t kws_init(const void        *model_data,
                   size_t             model_size,
                   const kws_config_t *config);

/**
 * Run keyword detection on pre-computed MFCC features.
 *
 * @param features  Float array [NE_MFCC_FEATURE_SIZE] from ne_mfcc_compute()
 * @param result    Output detection result
 * @return          ESP_OK on success
 */
esp_err_t kws_infer(const float *features, kws_result_t *result);

/**
 * Free all KWS resources.
 */
void kws_deinit(void);

#endif /* NE_KWS_H */
