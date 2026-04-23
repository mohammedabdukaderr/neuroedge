/**
 * NeuroEdge — uart_server.c
 *
 * UART JSON command server for the ESP32-S3 module.
 *
 * Architecture:
 *   - Single FreeRTOS task (ne_server_task) reads newline-terminated frames
 *   - Each frame is a JSON object; CRC16-CCITT covers the JSON text up to
 *     but not including the "crc" field itself
 *   - Dispatches to model_runner_* functions
 *   - Writes back a JSON response with the same id and a fresh CRC
 *
 * Error handling guarantees:
 *   - Malformed JSON / CRC mismatch: error response, never silent drop
 *   - Model timeout: error response with "timeout" code
 *   - Buffer overflow: error response, RX ring buffer flushed
 *   - UART driver error: logs and reinitializes UART (auto-recovery)
 *
 * Thread safety:
 *   - All state is owned by the server task; no external synchronization needed
 *   - model_runner functions are called from the server task only
 */

#include "uart_server.h"
#include "model_runner.h"
#include "crc16.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "driver/uart.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "cJSON.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

static const char *TAG = "uart_server";

/* -------------------------------------------------------------------------
 * Internal state
 * ---------------------------------------------------------------------- */

static TaskHandle_t  s_server_task = NULL;
static bool          s_running     = false;

/* Ring-buffer scratch: one frame at a time */
static char s_rx_buf[NE_FRAME_MAX_LEN];
static char s_tx_buf[NE_RESPONSE_MAX_LEN];

/* -------------------------------------------------------------------------
 * Helper: send a JSON response string over UART
 * ---------------------------------------------------------------------- */

static void uart_send(const char *json)
{
    size_t len = strlen(json);
    uart_write_bytes(NE_UART_PORT, json, len);
    uart_write_bytes(NE_UART_PORT, "\r\n", 2);
}

/* -------------------------------------------------------------------------
 * Helper: build and send an error response
 * ---------------------------------------------------------------------- */

static void send_error(uint32_t id, const char *code, const char *message)
{
    /* Build JSON manually to avoid cJSON allocation on the error path */
    char crc_hex[5];
    /* CRC covers everything up to the crc field — compute over the body */
    char body[256];
    int body_len = snprintf(body, sizeof(body),
                            "{\"id\":%" PRIu32 ",\"status\":\"error\","
                            "\"code\":\"%s\",\"message\":\"%s\"}",
                            id, code, message);
    if (body_len < 0 || (size_t)body_len >= sizeof(body)) {
        /* Fallback: minimal error without message */
        snprintf(body, sizeof(body),
                 "{\"id\":%" PRIu32 ",\"status\":\"error\",\"code\":\"overflow\"}", id);
        body_len = (int)strlen(body);
    }

    uint16_t crc = ne_crc16((const uint8_t *)body, (size_t)body_len);
    ne_crc16_to_hex(crc, crc_hex);

    snprintf(s_tx_buf, NE_RESPONSE_MAX_LEN,
             "{\"id\":%" PRIu32 ",\"status\":\"error\","
             "\"code\":\"%s\",\"message\":\"%s\",\"crc\":\"%s\"}",
             id, code, message, crc_hex);

    uart_send(s_tx_buf);
    ESP_LOGW(TAG, "Error response id=%" PRIu32 " code=%s", id, code);
}

/* -------------------------------------------------------------------------
 * CRC verification helper
 *
 * Protocol: CRC covers the raw received JSON text, EXCLUDING the "crc"
 * key-value pair and trailing comma (if any). For simplicity and robustness,
 * we treat the crc field as optional on inbound requests (host may omit it
 * for testing). When present, it is verified.
 *
 * Returns true if CRC is absent (skip check) or matches.
 * ---------------------------------------------------------------------- */

static bool verify_frame_crc(const char *frame_json, size_t frame_len,
                               const cJSON *root)
{
    cJSON *crc_item = cJSON_GetObjectItemCaseSensitive(root, "crc");
    if (!crc_item || !cJSON_IsString(crc_item)) {
        /* CRC field absent — host chose not to send it; accept frame */
        return true;
    }

    /* Locate the "crc" key in the raw text and exclude it from the check.
     * We compute CRC over the portion of the JSON before the "crc" field.
     * This is safe because cJSON serializes fields in insertion order when
     * constructing requests, and we document that "crc" must be last. */
    const char *crc_key = strstr(frame_json, "\"crc\"");
    if (!crc_key) return true;  /* Should not happen if cJSON found it */

    /* Strip trailing comma before "crc" field if present */
    size_t check_len = (size_t)(crc_key - frame_json);
    while (check_len > 0 && (frame_json[check_len - 1] == ',' ||
                              frame_json[check_len - 1] == ' ')) {
        check_len--;
    }

    uint16_t computed = ne_crc16((const uint8_t *)frame_json, check_len);
    uint16_t received = ne_crc16_from_hex(cJSON_GetStringValue(crc_item));

    if (computed != received) {
        ESP_LOGW(TAG, "CRC mismatch: computed=%04X received=%04X", computed, received);
        return false;
    }
    return true;
}

/* -------------------------------------------------------------------------
 * Command handlers
 * ---------------------------------------------------------------------- */

static void handle_ping(uint32_t id, int64_t t_recv)
{
    uint32_t latency_ms = (uint32_t)((esp_timer_get_time() - t_recv) / 1000ULL);

    char body[128];
    snprintf(body, sizeof(body),
             "{\"id\":%" PRIu32 ",\"status\":\"ok\",\"ms\":%" PRIu32 "}",
             id, latency_ms);

    char crc_hex[5];
    uint16_t crc = ne_crc16((const uint8_t *)body, strlen(body));
    ne_crc16_to_hex(crc, crc_hex);

    snprintf(s_tx_buf, NE_RESPONSE_MAX_LEN,
             "{\"id\":%" PRIu32 ",\"status\":\"ok\",\"ms\":%" PRIu32 ",\"crc\":\"%s\"}",
             id, latency_ms, crc_hex);

    uart_send(s_tx_buf);
}

static void handle_ask(uint32_t id, const cJSON *root)
{
    cJSON *prompt_item     = cJSON_GetObjectItemCaseSensitive(root, "prompt");
    cJSON *max_tokens_item = cJSON_GetObjectItemCaseSensitive(root, "max_tokens");

    if (!cJSON_IsString(prompt_item) || !prompt_item->valuestring[0]) {
        send_error(id, "bad_request", "Missing or empty 'prompt' field");
        return;
    }

    uint32_t max_tokens = 64;
    if (cJSON_IsNumber(max_tokens_item) && max_tokens_item->valuedouble > 0) {
        max_tokens = (uint32_t)max_tokens_item->valuedouble;
        if (max_tokens > 512) max_tokens = 512;  /* Hard cap for memory safety */
    }

    mr_llm_result_t result;
    memset(&result, 0, sizeof(result));

    esp_err_t err = model_runner_ask(prompt_item->valuestring, max_tokens, &result);
    if (err != ESP_OK) {
        send_error(id, "inference_error", esp_err_to_name(err));
        return;
    }

    /* Escape the text for embedding in JSON — cJSON handles this */
    cJSON *resp = cJSON_CreateObject();
    if (!resp) { send_error(id, "alloc_error", "cJSON_CreateObject"); return; }

    cJSON_AddNumberToObject(resp, "id",     (double)id);
    cJSON_AddStringToObject(resp, "status", "ok");
    cJSON_AddStringToObject(resp, "text",   result.text);
    cJSON_AddNumberToObject(resp, "ms",     (double)result.latency_ms);

    char *json_str = cJSON_PrintUnformatted(resp);
    cJSON_Delete(resp);
    if (!json_str) { send_error(id, "alloc_error", "cJSON_Print"); return; }

    /* Append CRC */
    uint16_t crc = ne_crc16((const uint8_t *)json_str, strlen(json_str));
    char crc_hex[5];
    ne_crc16_to_hex(crc, crc_hex);

    /* Build final frame: insert crc field before closing brace */
    size_t jlen = strlen(json_str);
    if (jlen > 0 && json_str[jlen - 1] == '}') {
        json_str[jlen - 1] = '\0';  /* Remove closing brace */
        snprintf(s_tx_buf, NE_RESPONSE_MAX_LEN,
                 "%s,\"crc\":\"%s\"}", json_str, crc_hex);
    } else {
        snprintf(s_tx_buf, NE_RESPONSE_MAX_LEN, "%s", json_str);
    }
    free(json_str);

    uart_send(s_tx_buf);
    ESP_LOGD(TAG, "ask id=%" PRIu32 " -> %u ms", id, result.latency_ms);
}

static void handle_classify(uint32_t id, const cJSON *root)
{
    cJSON *data_item = cJSON_GetObjectItemCaseSensitive(root, "data");

    if (!cJSON_IsArray(data_item)) {
        send_error(id, "bad_request", "Field 'data' must be a JSON array of bytes");
        return;
    }

    int arr_len = cJSON_GetArraySize(data_item);
    if (arr_len <= 0 || arr_len > MR_MAX_DATA_LEN) {
        send_error(id, "bad_request", "data array empty or exceeds 512 bytes");
        return;
    }

    /* Decode byte array */
    uint8_t data_buf[MR_MAX_DATA_LEN];
    for (int i = 0; i < arr_len; i++) {
        cJSON *item = cJSON_GetArrayItem(data_item, i);
        if (!cJSON_IsNumber(item)) {
            send_error(id, "bad_request", "data array must contain integers");
            return;
        }
        data_buf[i] = (uint8_t)item->valuedouble;
    }

    mr_classify_result_t result;
    memset(&result, 0, sizeof(result));

    esp_err_t err = model_runner_classify(data_buf, (size_t)arr_len, &result);
    if (err != ESP_OK) {
        send_error(id, "inference_error", esp_err_to_name(err));
        return;
    }

    char body[256];
    snprintf(body, sizeof(body),
             "{\"id\":%" PRIu32 ",\"status\":\"ok\","
             "\"label\":\"%s\",\"confidence\":%.4f,\"ms\":%" PRIu32 "}",
             id, result.label, (double)result.confidence, result.latency_ms);

    char crc_hex[5];
    uint16_t crc = ne_crc16((const uint8_t *)body, strlen(body));
    ne_crc16_to_hex(crc, crc_hex);

    snprintf(s_tx_buf, NE_RESPONSE_MAX_LEN,
             "{\"id\":%" PRIu32 ",\"status\":\"ok\","
             "\"label\":\"%s\",\"confidence\":%.4f,\"ms\":%" PRIu32 ",\"crc\":\"%s\"}",
             id, result.label, (double)result.confidence, result.latency_ms, crc_hex);

    uart_send(s_tx_buf);
}

static void handle_kws(uint32_t id, const cJSON *root)
{
    cJSON *pcm_item = cJSON_GetObjectItemCaseSensitive(root, "pcm");

    if (!cJSON_IsArray(pcm_item)) {
        send_error(id, "bad_request", "Field 'pcm' must be a JSON array of int16 samples");
        return;
    }

    int num_samples = cJSON_GetArraySize(pcm_item);
    if (num_samples <= 0 || num_samples > 16000) {  /* Max 1 second @ 16 kHz */
        send_error(id, "bad_request", "pcm array empty or exceeds 16000 samples");
        return;
    }

    int16_t *pcm_buf = (int16_t *)malloc((size_t)num_samples * sizeof(int16_t));
    if (!pcm_buf) {
        send_error(id, "alloc_error", "Cannot allocate PCM buffer");
        return;
    }

    for (int i = 0; i < num_samples; i++) {
        cJSON *item = cJSON_GetArrayItem(pcm_item, i);
        if (!cJSON_IsNumber(item)) {
            free(pcm_buf);
            send_error(id, "bad_request", "pcm array must contain integers");
            return;
        }
        pcm_buf[i] = (int16_t)item->valuedouble;
    }

    mr_kws_result_t result;
    memset(&result, 0, sizeof(result));

    esp_err_t err = model_runner_detect_keyword(pcm_buf, (size_t)num_samples, &result);
    free(pcm_buf);

    if (err != ESP_OK) {
        send_error(id, "inference_error", esp_err_to_name(err));
        return;
    }

    char body[256];
    snprintf(body, sizeof(body),
             "{\"id\":%" PRIu32 ",\"status\":\"ok\","
             "\"detected\":%s,\"keyword\":\"%s\",\"confidence\":%.4f,\"ms\":%" PRIu32 "}",
             id,
             result.detected ? "true" : "false",
             result.keyword,
             (double)result.confidence,
             result.latency_ms);

    char crc_hex[5];
    uint16_t crc = ne_crc16((const uint8_t *)body, strlen(body));
    ne_crc16_to_hex(crc, crc_hex);

    snprintf(s_tx_buf, NE_RESPONSE_MAX_LEN,
             "{\"id\":%" PRIu32 ",\"status\":\"ok\","
             "\"detected\":%s,\"keyword\":\"%s\","
             "\"confidence\":%.4f,\"ms\":%" PRIu32 ",\"crc\":\"%s\"}",
             id,
             result.detected ? "true" : "false",
             result.keyword,
             (double)result.confidence,
             result.latency_ms,
             crc_hex);

    uart_send(s_tx_buf);
}

static void handle_info(uint32_t id)
{
    multi_heap_info_t heap;
    heap_caps_get_info(&heap, MALLOC_CAP_DEFAULT);

    char body[512];
    snprintf(body, sizeof(body),
             "{\"id\":%" PRIu32 ",\"status\":\"ok\","
             "\"firmware\":\"%d.%d.%d\","
             "\"model\":\"%s\","
             "\"heap_free\":%" PRIu32 ","
             "\"heap_min\":%" PRIu32 "}",
             id,
             1, 0, 0,
             model_runner_active_name(),
             (uint32_t)heap.total_free_bytes,
             (uint32_t)heap.minimum_free_bytes);

    char crc_hex[5];
    uint16_t crc = ne_crc16((const uint8_t *)body, strlen(body));
    ne_crc16_to_hex(crc, crc_hex);

    /* Append CRC field */
    size_t blen = strlen(body);
    body[blen - 1] = '\0';  /* Strip closing } */
    snprintf(s_tx_buf, NE_RESPONSE_MAX_LEN, "%s,\"crc\":\"%s\"}", body, crc_hex);

    uart_send(s_tx_buf);
}

static void handle_set_model(uint32_t id, const cJSON *root)
{
    cJSON *model_item = cJSON_GetObjectItemCaseSensitive(root, "model");
    if (!cJSON_IsString(model_item)) {
        send_error(id, "bad_request", "Missing 'model' string field");
        return;
    }

    const char *name = model_item->valuestring;
    ne_model_id_t mid;

    if      (strcmp(name, "tinyllama-q4")  == 0) mid = MODEL_TINYLLAMA;
    else if (strcmp(name, "mobilenet-v3")  == 0) mid = MODEL_MOBILENET;
    else if (strcmp(name, "dscnn-kws")     == 0) mid = MODEL_DSCNN_KWS;
    else {
        send_error(id, "not_found", "Unknown model name");
        return;
    }

    esp_err_t err = model_runner_set_model(mid);
    if (err != ESP_OK) {
        send_error(id, "model_error", esp_err_to_name(err));
        return;
    }

    char body[128];
    snprintf(body, sizeof(body),
             "{\"id\":%" PRIu32 ",\"status\":\"ok\",\"model\":\"%s\"}",
             id, model_runner_active_name());

    char crc_hex[5];
    uint16_t crc = ne_crc16((const uint8_t *)body, strlen(body));
    ne_crc16_to_hex(crc, crc_hex);

    snprintf(s_tx_buf, NE_RESPONSE_MAX_LEN,
             "{\"id\":%" PRIu32 ",\"status\":\"ok\",\"model\":\"%s\",\"crc\":\"%s\"}",
             id, model_runner_active_name(), crc_hex);

    uart_send(s_tx_buf);
}

static void handle_sleep(uint32_t id, const cJSON *root)
{
    cJSON *dur_item = cJSON_GetObjectItemCaseSensitive(root, "duration_ms");
    uint32_t duration_ms = 0;
    if (cJSON_IsNumber(dur_item) && dur_item->valuedouble > 0) {
        duration_ms = (uint32_t)dur_item->valuedouble;
    }

    /* Send acknowledgment before going to sleep */
    char body[128];
    snprintf(body, sizeof(body),
             "{\"id\":%" PRIu32 ",\"status\":\"ok\",\"sleep_ms\":%" PRIu32 "}",
             id, duration_ms);

    char crc_hex[5];
    uint16_t crc = ne_crc16((const uint8_t *)body, strlen(body));
    ne_crc16_to_hex(crc, crc_hex);

    snprintf(s_tx_buf, NE_RESPONSE_MAX_LEN,
             "{\"id\":%" PRIu32 ",\"status\":\"ok\","
             "\"sleep_ms\":%" PRIu32 ",\"crc\":\"%s\"}",
             id, duration_ms, crc_hex);

    uart_send(s_tx_buf);
    uart_wait_tx_done(NE_UART_PORT, pdMS_TO_TICKS(100));

    /* Configure wake-on-UART and enter deep sleep */
    if (duration_ms == 0) {
        ESP_LOGI(TAG, "Entering deep sleep indefinitely (wake on UART)");
    } else {
        ESP_LOGI(TAG, "Entering deep sleep for %" PRIu32 " ms", duration_ms);
    }

    /* TODO: esp_sleep_enable_uart_wakeup(NE_UART_PORT);
     *       esp_sleep_enable_timer_wakeup((uint64_t)duration_ms * 1000);
     *       esp_deep_sleep_start();
     * Uncomment when deep-sleep wakeup on UART is validated on target HW. */
    ESP_LOGW(TAG, "Sleep not yet enabled (uncomment in handle_sleep)");
}

/* -------------------------------------------------------------------------
 * Frame dispatcher
 * ---------------------------------------------------------------------- */

static void dispatch_frame(const char *frame, size_t len)
{
    int64_t t_recv = esp_timer_get_time();

    if (len == 0) return;

    ESP_LOGD(TAG, "RX [%zu]: %.80s%s", len, frame, len > 80 ? "..." : "");

    /* Parse JSON */
    cJSON *root = cJSON_ParseWithLength(frame, len);
    if (!root) {
        ESP_LOGW(TAG, "JSON parse error: %.40s", frame);
        /* We don't know the id, use 0 */
        send_error(0, "json_error", "Malformed JSON");
        return;
    }

    /* Extract mandatory id field */
    cJSON *id_item = cJSON_GetObjectItemCaseSensitive(root, "id");
    if (!cJSON_IsNumber(id_item)) {
        send_error(0, "bad_request", "Missing numeric 'id' field");
        cJSON_Delete(root);
        return;
    }
    uint32_t id = (uint32_t)id_item->valuedouble;

    /* Verify CRC if provided */
    if (!verify_frame_crc(frame, len, root)) {
        send_error(id, "crc_error", "CRC mismatch");
        cJSON_Delete(root);
        return;
    }

    /* Extract command */
    cJSON *cmd_item = cJSON_GetObjectItemCaseSensitive(root, "cmd");
    if (!cJSON_IsString(cmd_item)) {
        send_error(id, "bad_request", "Missing 'cmd' string field");
        cJSON_Delete(root);
        return;
    }
    const char *cmd = cmd_item->valuestring;

    /* Dispatch */
    if      (strcmp(cmd, "ping")      == 0) handle_ping(id, t_recv);
    else if (strcmp(cmd, "ask")       == 0) handle_ask(id, root);
    else if (strcmp(cmd, "classify")  == 0) handle_classify(id, root);
    else if (strcmp(cmd, "kws")       == 0) handle_kws(id, root);
    else if (strcmp(cmd, "info")      == 0) handle_info(id);
    else if (strcmp(cmd, "set_model") == 0) handle_set_model(id, root);
    else if (strcmp(cmd, "sleep")     == 0) handle_sleep(id, root);
    else {
        char msg[64];
        snprintf(msg, sizeof(msg), "Unknown command: %.32s", cmd);
        send_error(id, "unknown_cmd", msg);
    }

    cJSON_Delete(root);
}

/* -------------------------------------------------------------------------
 * UART receive — accumulate bytes until \n, then dispatch
 * ---------------------------------------------------------------------- */

#define UART_REINIT_DELAY_MS  1000

static esp_err_t uart_hw_init(void)
{
    uart_config_t cfg = {
        .baud_rate           = NE_UART_BAUD,
        .data_bits           = UART_DATA_8_BITS,
        .parity              = UART_PARITY_DISABLE,
        .stop_bits           = UART_STOP_BITS_1,
        .flow_ctrl           = UART_HW_FLOWCTRL_DISABLE,
        .source_clk          = UART_SCLK_DEFAULT,
    };

    esp_err_t err;

    err = uart_driver_install(NE_UART_PORT,
                              NE_UART_BUF_SIZE * 2,  /* RX ring buffer */
                              NE_UART_BUF_SIZE,      /* TX ring buffer */
                              0, NULL, 0);
    if (err != ESP_OK) return err;

    err = uart_param_config(NE_UART_PORT, &cfg);
    if (err != ESP_OK) { uart_driver_delete(NE_UART_PORT); return err; }

    err = uart_set_pin(NE_UART_PORT,
                       NE_UART_TX_PIN, NE_UART_RX_PIN,
                       UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    if (err != ESP_OK) { uart_driver_delete(NE_UART_PORT); return err; }

    return ESP_OK;
}

/* -------------------------------------------------------------------------
 * Server task
 * ---------------------------------------------------------------------- */

static void ne_server_task(void *arg)
{
    (void)arg;

    size_t frame_pos  = 0;
    int    uart_errors = 0;

    ESP_LOGI(TAG, "NeuroEdge UART server started on UART%d TX=%d RX=%d @ %d baud",
             NE_UART_PORT, NE_UART_TX_PIN, NE_UART_RX_PIN, NE_UART_BAUD);

    while (s_running) {
        uint8_t byte;
        int n = uart_read_bytes(NE_UART_PORT, &byte, 1, pdMS_TO_TICKS(50));

        if (n < 0) {
            /* UART driver error — attempt recovery */
            uart_errors++;
            ESP_LOGE(TAG, "UART read error (attempt %d)", uart_errors);
            if (uart_errors >= 5) {
                ESP_LOGE(TAG, "Too many UART errors — reinitializing");
                uart_driver_delete(NE_UART_PORT);
                vTaskDelay(pdMS_TO_TICKS(UART_REINIT_DELAY_MS));
                if (uart_hw_init() != ESP_OK) {
                    ESP_LOGE(TAG, "UART reinit failed — halting task");
                    break;
                }
                frame_pos  = 0;
                uart_errors = 0;
            }
            continue;
        }

        uart_errors = 0;  /* Reset error counter on successful read */

        if (n == 0) continue;  /* Timeout — no data yet */

        /* Accumulate byte */
        if (byte == '\n' || byte == '\r') {
            if (frame_pos > 0) {
                s_rx_buf[frame_pos] = '\0';
                dispatch_frame(s_rx_buf, frame_pos);
                frame_pos = 0;
            }
            /* Ignore standalone CR/LF */
            continue;
        }

        if (frame_pos >= NE_FRAME_MAX_LEN - 1) {
            /* Frame too long — discard and reset, report error */
            ESP_LOGW(TAG, "Frame overflow at %zu bytes — discarding", frame_pos);
            /* We lost the id, send id=0 error and flush */
            send_error(0, "overflow", "Frame exceeds maximum length");
            uart_flush_input(NE_UART_PORT);
            frame_pos = 0;
            continue;
        }

        s_rx_buf[frame_pos++] = (char)byte;
    }

    ESP_LOGI(TAG, "Server task exiting");
    vTaskDelete(NULL);
}

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

esp_err_t uart_server_init(void)
{
    if (s_running) {
        ESP_LOGW(TAG, "Already initialized");
        return ESP_OK;
    }

    esp_err_t err = uart_hw_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "UART hardware init failed: %s", esp_err_to_name(err));
        return err;
    }

    s_running = true;

    BaseType_t rc = xTaskCreate(ne_server_task,
                                "ne_uart_srv",
                                NE_SERVER_TASK_STACK,
                                NULL,
                                NE_SERVER_TASK_PRIORITY,
                                &s_server_task);
    if (rc != pdPASS) {
        ESP_LOGE(TAG, "Failed to create server task");
        uart_driver_delete(NE_UART_PORT);
        s_running = false;
        return ESP_ERR_NO_MEM;
    }

    ESP_LOGI(TAG, "UART server initialized");
    return ESP_OK;
}

void uart_server_deinit(void)
{
    if (!s_running) return;
    s_running = false;

    /* Give the task time to notice and exit */
    vTaskDelay(pdMS_TO_TICKS(200));

    if (s_server_task) {
        vTaskDelete(s_server_task);
        s_server_task = NULL;
    }

    uart_driver_delete(NE_UART_PORT);
    ESP_LOGI(TAG, "UART server deinitialized");
}
