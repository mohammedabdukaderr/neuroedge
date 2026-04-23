/**
 * NeuroEdge — model_loader.c
 *
 * Validates and memory-maps AI model data from an ESP32 flash partition.
 * Uses esp_partition_mmap() so the data is accessible as a plain pointer —
 * no copy into PSRAM needed. llama.cpp's GGUF loader reads tensors directly
 * from flash via the MMU.
 */

#include "model_loader.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_rom_crc.h"
#include <string.h>

static const char *TAG = "model_loader";

esp_err_t model_loader_open(const char *partition_name,
                             ne_model_region_t *out_region)
{
    if (!partition_name || !out_region) return ESP_ERR_INVALID_ARG;
    memset(out_region, 0, sizeof(*out_region));

    /* Locate the partition by name (type=data, subtype=any) */
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, partition_name);

    if (!part) {
        ESP_LOGE(TAG, "Partition '%s' not found", partition_name);
        return ESP_ERR_NOT_FOUND;
    }

    /* Read and validate the 64-byte header */
    ne_model_header_t hdr;
    esp_err_t err = esp_partition_read(part, 0, &hdr, sizeof(hdr));
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read header from '%s': %s",
                 partition_name, esp_err_to_name(err));
        return err;
    }

    if (hdr.magic != MODEL_MAGIC) {
        ESP_LOGE(TAG, "Bad magic in '%s': 0x%08X (expected 0x%08X)",
                 partition_name, hdr.magic, MODEL_MAGIC);
        return ESP_ERR_INVALID_ARG;
    }

    if (hdr.version != MODEL_HEADER_VERSION) {
        ESP_LOGE(TAG, "Unsupported header version %u in '%s'",
                 hdr.version, partition_name);
        return ESP_ERR_NOT_SUPPORTED;
    }

    if (hdr.data_size == 0 || hdr.data_size > part->size - sizeof(hdr)) {
        ESP_LOGE(TAG, "Invalid data_size %u in '%s'", hdr.data_size, partition_name);
        return ESP_ERR_INVALID_SIZE;
    }

    ESP_LOGI(TAG, "Opening model '%s' from partition '%s' (%u bytes, type=%u)",
             hdr.model_id, partition_name, hdr.data_size, hdr.model_type);

    /* Memory-map the data region (skipping the 64-byte header) */
    esp_partition_mmap_handle_t mmap_handle;
    const void *mmap_ptr = NULL;

    err = esp_partition_mmap(part,
                             sizeof(hdr),     /* Offset: skip header */
                             hdr.data_size,
                             ESP_PARTITION_MMAP_DATA,
                             &mmap_ptr,
                             &mmap_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "mmap failed for '%s': %s",
                 partition_name, esp_err_to_name(err));
        return err;
    }

    /* Fill output */
    out_region->partition    = part;
    out_region->data_offset  = sizeof(hdr);
    out_region->data_size    = hdr.data_size;
    out_region->type         = (ne_model_type_t)hdr.model_type;
    out_region->mmap_handle  = (void *)(uintptr_t)mmap_handle;
    out_region->mmap_ptr     = mmap_ptr;
    memcpy(out_region->model_id, hdr.model_id, sizeof(out_region->model_id));
    out_region->model_id[sizeof(out_region->model_id) - 1] = '\0';

    ESP_LOGI(TAG, "Model mapped at %p (%u bytes)", mmap_ptr, hdr.data_size);
    return ESP_OK;
}

void model_loader_close(ne_model_region_t *region)
{
    if (!region || !region->mmap_handle) return;

    esp_partition_munmap((esp_partition_mmap_handle_t)(uintptr_t)region->mmap_handle);
    memset(region, 0, sizeof(*region));
    ESP_LOGD(TAG, "Model region closed");
}

const char *model_loader_type_name(ne_model_type_t type)
{
    switch (type) {
        case MODEL_TYPE_GGUF:   return "GGUF (llama.cpp)";
        case MODEL_TYPE_TFLITE: return "TFLite Micro";
        default:                return "Unknown";
    }
}

esp_err_t model_loader_verify_crc(const ne_model_region_t *region)
{
    if (!region || !region->partition) return ESP_ERR_INVALID_ARG;

    /* Read the stored CRC from the header */
    ne_model_header_t hdr;
    esp_err_t err = esp_partition_read(region->partition, 0, &hdr, sizeof(hdr));
    if (err != ESP_OK) return err;

    /* Compute CRC32 over the mapped data */
    uint32_t computed = esp_rom_crc32_le(0, (const uint8_t *)region->mmap_ptr,
                                          region->data_size);

    if (computed != hdr.crc32) {
        ESP_LOGE(TAG, "CRC mismatch: computed=0x%08X stored=0x%08X",
                 computed, hdr.crc32);
        return ESP_ERR_INVALID_CRC;
    }

    ESP_LOGI(TAG, "Model CRC OK (0x%08X)", computed);
    return ESP_OK;
}
