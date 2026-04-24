/**
 * NeuroEdge — ggml_esp32_config.h
 *
 * Compile-time configuration for ggml / llama.cpp on ESP32 (Xtensa LX6).
 * Include this file as the GGML_CONFIG_FILE to override ggml's defaults.
 *
 * Key constraints on ESP32 (Xtensa LX6):
 *   - No x86 SIMD (SSE/AVX/NEON) — CPU-only quantized matmul
 *   - Internal SRAM: 520 KB — avoid for large tensors
 *   - PSRAM (SPIRAM, quad): 4 MB on WROVER boards, none on WROOM
 *   - Flash: 4 MB (model_0 partition: 1.5 MB — TFLite only)
 *   - No pthreads by default; use single-threaded mode
 */

#ifndef GGML_ESP32_CONFIG_H
#define GGML_ESP32_CONFIG_H

/* Disable all SIMD backends — Xtensa LX6 has none */
#define GGML_NO_METAL        1
#define GGML_NO_ACCELERATE   1
#define GGML_NO_OPENMP       1
#define GGML_NO_LLAMAFILE    1
#define GGML_NO_CUDA         1
#define GGML_NO_VULKAN       1
#define GGML_NO_KOMPUTE      1
#define GGML_NO_SYCL         1

/* CPU-only path */
#define GGML_USE_CPU         1

/* Memory — use PSRAM if available, fall back to internal heap */
#include "esp_heap_caps.h"
#ifdef CONFIG_SPIRAM
#define GGML_MALLOC(size)       heap_caps_malloc((size), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
#define GGML_CALLOC(n, size)    heap_caps_calloc((n), (size), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
#define GGML_FREE(ptr)          heap_caps_free(ptr)
#define GGML_REALLOC(ptr, size) heap_caps_realloc((ptr), (size), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
#else
#define GGML_MALLOC(size)       malloc(size)
#define GGML_CALLOC(n, size)    calloc((n), (size))
#define GGML_FREE(ptr)          free(ptr)
#define GGML_REALLOC(ptr, size) realloc((ptr), (size))
#endif

/* Reduce static buffer sizes */
#define GGML_MAX_CONTEXTS   4
#define GGML_MAX_SRC        8
#define GGML_MAX_DIMS       4
#define GGML_MAX_NODES      2048
#define GGML_MAX_PARAMS     1024

/* Scratch buffer — reduced for 4 MB PSRAM / no-PSRAM boards */
#ifdef CONFIG_SPIRAM
#define GGML_SCRATCH_SIZE   (2UL * 1024UL * 1024UL)   /* 2 MB with PSRAM */
#else
#define GGML_SCRATCH_SIZE   (128UL * 1024UL)           /* 128 KB internal only */
#endif

#define GGML_MEM_ALIGN      16

/* Threading: single-threaded on FreeRTOS */
#define GGML_N_THREADS_DEFAULT  1
typedef int pthread_t;

#define GGML_NO_FILE_IO      0

#endif /* GGML_ESP32_CONFIG_H */
