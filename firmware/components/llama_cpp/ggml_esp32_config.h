/**
 * NeuroEdge — ggml_esp32_config.h
 *
 * Compile-time configuration for ggml / llama.cpp on ESP32-S3.
 * Include this file as the GGML_CONFIG_FILE to override ggml's defaults.
 *
 * Key constraints on ESP32-S3 (Xtensa LX7):
 *   - No x86 SIMD (SSE/AVX/NEON) — CPU-only quantized matmul
 *   - Internal SRAM: 512 KB — never use for large tensors
 *   - PSRAM (SPIRAM, octal): 8 MB — all model tensors go here
 *   - Flash execution (XIP): allowed but slow; model loaded into PSRAM
 *   - No pthreads by default; use single-threaded mode
 *   - Stack depth: keep llama inference below 32 KB stack usage
 */

#ifndef GGML_ESP32_CONFIG_H
#define GGML_ESP32_CONFIG_H

/* Disable all SIMD backends — Xtensa LX7 has none */
#define GGML_NO_METAL        1
#define GGML_NO_ACCELERATE   1
#define GGML_NO_OPENMP       1   /* Enable with care if FreeRTOS pthreads available */
#define GGML_NO_LLAMAFILE    1
#define GGML_NO_CUDA         1
#define GGML_NO_VULKAN       1
#define GGML_NO_KOMPUTE      1
#define GGML_NO_SYCL         1

/* CPU-only path */
#define GGML_USE_CPU         1

/* Memory — route all large allocs to PSRAM */
#include "esp_heap_caps.h"
#define GGML_MALLOC(size)       heap_caps_malloc((size), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
#define GGML_CALLOC(n, size)    heap_caps_calloc((n), (size), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
#define GGML_FREE(ptr)          heap_caps_free(ptr)
#define GGML_REALLOC(ptr, size) heap_caps_realloc((ptr), (size), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)

/* Reduce static buffer sizes to fit in PSRAM */
#define GGML_MAX_CONTEXTS   4
#define GGML_MAX_SRC        8
#define GGML_MAX_DIMS       4
#define GGML_MAX_NODES      4096    /* Reduce from 8192 for memory */
#define GGML_MAX_PARAMS     2048

/* Scratch buffer: goes in PSRAM */
#define GGML_SCRATCH_SIZE   (4UL * 1024UL * 1024UL)   /* 4 MB scratch */

/* Computation buffer: PSRAM */
#define GGML_MEM_ALIGN      16

/* Threading: single-threaded on bare-metal FreeRTOS */
#define GGML_N_THREADS_DEFAULT  1
typedef int pthread_t;              /* Stub — not used with n_threads=1 */

/* Disable file I/O where possible (model loaded from partition directly) */
#define GGML_NO_FILE_IO      0      /* We implement custom mmap via partition */

#endif /* GGML_ESP32_CONFIG_H */
