/**
 * NeuroEdge — uart_server.h
 *
 * UART JSON command server running on the ESP32-S3 module.
 * Listens for commands from the host MCU, dispatches to model_runner,
 * and writes back JSON responses.
 *
 * Protocol summary:
 *   Frame format: <JSON object>\r\n
 *   Request:  {"id":<uint32>,"cmd":"<cmd>","crc":"<4hex>",...fields}
 *   Response: {"id":<uint32>,"status":"ok"|"error","crc":"<4hex>",...fields}
 *
 *   Supported commands:
 *     ping       — latency probe
 *     ask        — LLM text generation
 *     classify   — sensor/image classification
 *     kws        — keyword spotting
 *     info       — firmware + resource info
 *     set_model  — switch active model
 *     sleep      — enter deep sleep
 */

#ifndef NE_UART_SERVER_H
#define NE_UART_SERVER_H

#include "esp_err.h"
#include "driver/uart.h"

/* Hardware configuration — override via sdkconfig / Kconfig */
#ifndef NE_UART_PORT
#define NE_UART_PORT       UART_NUM_1
#endif
#ifndef NE_UART_TX_PIN
#define NE_UART_TX_PIN     17
#endif
#ifndef NE_UART_RX_PIN
#define NE_UART_RX_PIN     18
#endif
#ifndef NE_UART_BAUD
#define NE_UART_BAUD       115200
#endif

/* FreeRTOS task settings */
#define NE_SERVER_TASK_STACK    8192
#define NE_SERVER_TASK_PRIORITY 5

/* Frame settings */
#define NE_FRAME_MAX_LEN        (2048 + 256)   /* JSON payload + overhead */
#define NE_RESPONSE_MAX_LEN     (NE_FRAME_MAX_LEN)

/**
 * Initialize UART hardware and start the server task.
 * Must be called after model_runner_init().
 *
 * @return ESP_OK on success
 */
esp_err_t uart_server_init(void);

/**
 * Stop the server task and free UART resources.
 */
void uart_server_deinit(void);

#endif /* NE_UART_SERVER_H */
