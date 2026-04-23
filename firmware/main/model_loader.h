/**
 * NeuroEdge — model_loader.h
 *
 * Loads AI model data from an ESP32 flash partition into PSRAM.
 * Provides a FILE*-compatible interface so llama.cpp can open the
 * model with llama_load_model_from_file() using a virtual path.
 *
 * Flash partition layout:
 *   Partition "model_0" — primary model (DS-CNN, MobileNet, or GGUF)
 *   Partition "model_1" — secondary slot
 *
 * Memory strategy:
 *   - Model tensors: PSRAM (MALLOC_CAP_SPIRAM)
 *   - Compute buffers: PSRAM
 *   - KV cache: PSRAM
 *   - Firmware + task stacks: internal SRAM
 */

#ifndef NE_MODEL_LOADER_H
#define NE_MODEL_LOADER_H

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"
#include "esp_partition.h"

/* Virtual path prefix — llama.cpp "opens" this to trigger flash reads */
#define MODEL_FLASH_PATH_0   "ne://model_0"
#define MODEL_FLASH_PATH_1   "ne://model_1"

/* Model header stored at start of flash partition */
#define MODEL_MAGIC          0x4E454D4C   /* "NEML" */
#define MODEL_HEADER_VERSION 1

typedef struct __attribute__((packed)) {
    uint32_t magic;          /* MODEL_MAGIC */
    uint8_t  version;        /* Header version */
    uint8_t  model_type;     /* 1=GGUF/llama, 2=TFLite */
    uint16_t reserved;
    uint32_t data_size;      /* Size of model data after header (bytes) */
    uint32_t crc32;          /* CRC32 of the data blob */
    char     model_id[32];   /* e.g. "tinyllama-q4" */
    uint8_t  padding[16];    /* Pad to 64 bytes total */
} ne_model_header_t;

_Static_assert(sizeof(ne_model_header_t) == 64, "Header must be 64 bytes");

typedef enum {
    MODEL_TYPE_GGUF   = 1,
    MODEL_TYPE_TFLITE = 2,
} ne_model_type_t;

/* Opaque handle for a loaded model region */
typedef struct ne_model_region {
    const esp_partition_t *partition;
    uint32_t               data_offset;  /* Offset past the header */
    uint32_t               data_size;
    ne_model_type_t        type;
    char                   model_id[32];
    void                  *mmap_handle;  /* esp_partition_mmap handle */
    const void            *mmap_ptr;     /* Mapped address in virtual addr space */
} ne_model_region_t;

/**
 * Validate and map a model partition into the virtual address space.
 *
 * Uses esp_partition_mmap — the partition data is mapped read-only into
 * IRAM/DRAM virtual space. llama.cpp receives a pointer and reads tensors
 * directly from flash via the MMU (no copy, zero PSRAM use for weights).
 *
 * @param partition_name  "model_0" or "model_1"
 * @param out_region      Filled on success
 * @return                ESP_OK, or error code
 */
esp_err_t model_loader_open(const char *partition_name,
                             ne_model_region_t *out_region);

/**
 * Unmap and release a model region.
 */
void model_loader_close(ne_model_region_t *region);

/**
 * Return a human-readable string for the type of model in a region.
 */
const char *model_loader_type_name(ne_model_type_t type);

/**
 * Verify CRC32 of model data. Slow (reads whole partition) — use only at
 * flash time or on-demand diagnostics, not in the hot path.
 *
 * @return ESP_OK if CRC matches, ESP_ERR_INVALID_CRC otherwise
 */
esp_err_t model_loader_verify_crc(const ne_model_region_t *region);

#endif /* NE_MODEL_LOADER_H */
