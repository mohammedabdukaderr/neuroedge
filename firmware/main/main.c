/**
 * NeuroEdge — main.c
 *
 * Firmware entry point for the ESP32-S3 NeuroEdge module.
 * Boot sequence:
 *   1. Log startup banner
 *   2. Initialize model runner (loads model from flash)
 *   3. Start UART server (begins accepting commands)
 *   4. Main task exits (FreeRTOS continues running)
 */

#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "model_runner.h"
#include "uart_server.h"

static const char *TAG = "main";

void app_main(void)
{
    ESP_LOGI(TAG, "NeuroEdge firmware v1.0.0 starting");
    ESP_LOGI(TAG, "Chip: %s, cores: %d, flash: %dMB",
             CONFIG_IDF_TARGET,
             CONFIG_FREERTOS_UNICORE ? 1 : 2,
             CONFIG_ESPTOOLPY_FLASHSIZE_16MB ? 16 : 4);

    /* NVS — required by some ESP-IDF components */
    esp_err_t nvs_err = nvs_flash_init();
    if (nvs_err == ESP_ERR_NVS_NO_FREE_PAGES ||
        nvs_err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_LOGW(TAG, "NVS erase and reinit");
        nvs_flash_erase();
        nvs_flash_init();
    }

    /* Model runner — loads AI model(s) into PSRAM/SRAM */
    esp_err_t err = model_runner_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Model runner init failed: %s — rebooting in 3s",
                 esp_err_to_name(err));
        vTaskDelay(pdMS_TO_TICKS(3000));
        esp_restart();
    }

    /* UART server — starts accepting commands from host */
    err = uart_server_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "UART server init failed: %s — rebooting in 3s",
                 esp_err_to_name(err));
        vTaskDelay(pdMS_TO_TICKS(3000));
        esp_restart();
    }

    ESP_LOGI(TAG, "NeuroEdge ready. Awaiting commands on UART%d.", NE_UART_PORT);
    /* app_main returning is fine — FreeRTOS tasks continue */
}
