/**
 * NeuroEdge — kws.cc  (C++ because TFLite Micro API is C++)
 *
 * Keyword spotting with DS-CNN via TFLite Micro.
 * Called from model_runner.c via extern "C" linkage.
 *
 * TFLite Micro pipeline:
 *   1. Load flatbuffer model from flash mmap pointer
 *   2. Build MicroMutableOpResolver with required ops for DS-CNN:
 *      Conv2D, DepthwiseConv2D, FullyConnected, AveragePool, Reshape, Softmax
 *   3. Allocate tensor arena from PSRAM
 *   4. Per-inference: copy MFCC features → input tensor → Invoke() → read output
 */

#include "kws.h"
#include "mfcc.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#if defined(NE_TFLITE_AVAILABLE)

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define TFLITE_AVAILABLE 1

#else
/* Stubs for when TFLite Micro is not yet cloned */
#define TFLITE_AVAILABLE 0
#endif

static const char *TAG = "kws";

/* -------------------------------------------------------------------------
 * State
 * ---------------------------------------------------------------------- */

struct KwsState {
    float    threshold;
    bool     initialized;

#if TFLITE_AVAILABLE
    const tflite::Model                              *model;
    tflite::MicroMutableOpResolver<8>                resolver;
    tflite::MicroInterpreter                         *interpreter;
    TfLiteTensor                                     *input;
    TfLiteTensor                                     *output;
    uint8_t                                          *arena;
#endif
};

static KwsState s_kws = {};

/* -------------------------------------------------------------------------
 * Init
 * ---------------------------------------------------------------------- */

extern "C"
esp_err_t kws_init(const void        *model_data,
                   size_t             model_size,
                   const kws_config_t *config)
{
    if (s_kws.initialized) {
        ESP_LOGW(TAG, "Already initialized");
        return ESP_OK;
    }

    s_kws.threshold = config ? config->detection_threshold
                             : KWS_DEFAULT_THRESHOLD;

#if TFLITE_AVAILABLE
    /* Allocate tensor arena in PSRAM */
    s_kws.arena = (uint8_t *)heap_caps_malloc(KWS_TENSOR_ARENA_SIZE,
                                               MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_kws.arena) {
        ESP_LOGE(TAG, "Cannot allocate %d B tensor arena in PSRAM",
                 KWS_TENSOR_ARENA_SIZE);
        return ESP_ERR_NO_MEM;
    }

    /* Load model from mmap'd pointer */
    s_kws.model = tflite::GetModel(model_data);
    if (s_kws.model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "TFLite schema version mismatch: model=%u, runtime=%u",
                 s_kws.model->version(), TFLITE_SCHEMA_VERSION);
        heap_caps_free(s_kws.arena);
        s_kws.arena = nullptr;
        return ESP_ERR_NOT_SUPPORTED;
    }

    /* Register only the ops used by DS-CNN (minimizes flash footprint) */
    s_kws.resolver.AddConv2D();
    s_kws.resolver.AddDepthwiseConv2D();
    s_kws.resolver.AddFullyConnected();
    s_kws.resolver.AddAveragePool2D();
    s_kws.resolver.AddReshape();
    s_kws.resolver.AddSoftmax();
    s_kws.resolver.AddQuantize();
    s_kws.resolver.AddDequantize();

    /* Create interpreter */
    s_kws.interpreter = new tflite::MicroInterpreter(
        s_kws.model, s_kws.resolver,
        s_kws.arena, KWS_TENSOR_ARENA_SIZE);

    TfLiteStatus alloc_status = s_kws.interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed — arena too small?");
        delete s_kws.interpreter;
        s_kws.interpreter = nullptr;
        heap_caps_free(s_kws.arena);
        s_kws.arena = nullptr;
        return ESP_ERR_NO_MEM;
    }

    /* Validate input tensor shape: [1, 49, 10, 1] */
    s_kws.input  = s_kws.interpreter->input(0);
    s_kws.output = s_kws.interpreter->output(0);

    bool shape_ok = (s_kws.input->dims->size == 4          &&
                     s_kws.input->dims->data[0] == 1       &&
                     s_kws.input->dims->data[1] == NE_MFCC_NUM_FRAMES &&
                     s_kws.input->dims->data[2] == NE_MFCC_NUM_COEFFS &&
                     s_kws.input->dims->data[3] == 1);
    if (!shape_ok) {
        ESP_LOGE(TAG, "Input tensor shape mismatch: expected [1,%d,%d,1]",
                 NE_MFCC_NUM_FRAMES, NE_MFCC_NUM_COEFFS);
        delete s_kws.interpreter;
        s_kws.interpreter = nullptr;
        heap_caps_free(s_kws.arena);
        s_kws.arena = nullptr;
        return ESP_ERR_INVALID_ARG;
    }

    if (s_kws.output->dims->data[1] != KWS_NUM_CLASSES) {
        ESP_LOGE(TAG, "Output tensor has %d classes, expected %d",
                 s_kws.output->dims->data[1], KWS_NUM_CLASSES);
        delete s_kws.interpreter;
        s_kws.interpreter = nullptr;
        heap_caps_free(s_kws.arena);
        s_kws.arena = nullptr;
        return ESP_ERR_INVALID_ARG;
    }

    ESP_LOGI(TAG, "DS-CNN ready: threshold=%.2f arena=%d KB",
             s_kws.threshold, KWS_TENSOR_ARENA_SIZE / 1024);

#else /* !TFLITE_AVAILABLE */
    ESP_LOGW(TAG, "TFLite Micro not compiled — KWS in stub mode");
    ESP_LOGW(TAG, "Clone tflite-micro into components/tflite_micro/tflite-micro");
#endif

    s_kws.initialized = true;
    return ESP_OK;
}

/* -------------------------------------------------------------------------
 * Inference
 * ---------------------------------------------------------------------- */

extern "C"
esp_err_t kws_infer(const float *features, kws_result_t *result)
{
    if (!s_kws.initialized) return ESP_ERR_INVALID_STATE;
    if (!features || !result) return ESP_ERR_INVALID_ARG;

    int64_t t0 = esp_timer_get_time();

#if TFLITE_AVAILABLE
    if (!s_kws.interpreter) {
        /* TFLite not available — stub output */
        result->detected   = false;
        result->keyword[0] = '\0';
        result->class_idx  = 0;
        result->confidence = 0.0f;
        result->latency_ms = 0;
        return ESP_OK;
    }

    /* Copy features into input tensor */
    float *input_data = s_kws.input->data.f;
    memcpy(input_data, features, NE_MFCC_FEATURE_SIZE * sizeof(float));

    /* Run inference */
    TfLiteStatus status = s_kws.interpreter->Invoke();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() failed");
        return ESP_FAIL;
    }

    /* Find argmax in output probabilities */
    const float *probs = s_kws.output->data.f;
    int   best_idx  = 0;
    float best_prob = probs[0];
    for (int i = 1; i < KWS_NUM_CLASSES; i++) {
        if (probs[i] > best_prob) { best_prob = probs[i]; best_idx = i; }
    }

    result->class_idx  = best_idx;
    result->confidence = best_prob;
    result->latency_ms = (uint32_t)((esp_timer_get_time() - t0) / 1000ULL);

    /* "silence" (0) and "unknown" (1) are not treated as positive detections */
    if (best_idx >= 2 && best_prob >= s_kws.threshold) {
        result->detected = true;
        strncpy(result->keyword, KWS_LABELS[best_idx], KWS_LABEL_LEN - 1);
        result->keyword[KWS_LABEL_LEN - 1] = '\0';
        ESP_LOGI(TAG, "Detected '%s' (%.3f) in %u ms",
                 result->keyword, best_prob, result->latency_ms);
    } else {
        result->detected   = false;
        result->keyword[0] = '\0';
    }

#else /* stub */
    result->detected   = false;
    result->keyword[0] = '\0';
    result->class_idx  = 0;
    result->confidence = 0.0f;
    result->latency_ms = (uint32_t)((esp_timer_get_time() - t0) / 1000ULL);
#endif

    return ESP_OK;
}

/* -------------------------------------------------------------------------
 * Deinit
 * ---------------------------------------------------------------------- */

extern "C"
void kws_deinit(void)
{
    if (!s_kws.initialized) return;
#if TFLITE_AVAILABLE
    if (s_kws.interpreter) { delete s_kws.interpreter; s_kws.interpreter = nullptr; }
    if (s_kws.arena)        { heap_caps_free(s_kws.arena); s_kws.arena = nullptr; }
    s_kws.model = nullptr;
#endif
    s_kws.initialized = false;
    ESP_LOGI(TAG, "KWS deinitialized");
}
