/**
 * NeuroEdge Host SDK — neuroedge.h
 *
 * Unified API for communicating with a NeuroEdge module over UART.
 * Supports ESP32, STM32, nRF52, Arduino, and any platform with a UART.
 *
 * Typical usage:
 *   ne_config_t cfg = NE_DEFAULT_CONFIG("/dev/ttyUSB0");
 *   ne_handle_t *ne = ne_init(&cfg);
 *   char response[256];
 *   ne_ask(ne, "Is this heartbeat abnormal?", response, sizeof(response));
 *   ne_close(ne);
 *
 * Protocol: UART 115200 baud, JSON envelope, CRC16-CCITT, newline-terminated.
 *   Request:  {"id":1,"cmd":"ask","prompt":"...","max_tokens":64}\r\n
 *   Response: {"id":1,"status":"ok","text":"...","ms":240,"crc":"ABCD"}\r\n
 *
 * MIT License — https://github.com/neuroedge/neuroedge
 */

#ifndef NEUROEDGE_H
#define NEUROEDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* -------------------------------------------------------------------------
 * Version
 * ---------------------------------------------------------------------- */
#define NE_VERSION_MAJOR  1
#define NE_VERSION_MINOR  0
#define NE_VERSION_PATCH  0
#define NE_VERSION_STRING "1.0.0"

/* -------------------------------------------------------------------------
 * Limits & defaults
 * ---------------------------------------------------------------------- */
#define NE_MAX_PROMPT_LEN       1024
#define NE_MAX_RESPONSE_LEN     2048
#define NE_MAX_LABEL_LEN        64
#define NE_MAX_SENSOR_BYTES     512
#define NE_DEFAULT_TIMEOUT_MS   5000
#define NE_DEFAULT_BAUD         115200
#define NE_MAX_RETRIES          3
#define NE_RECONNECT_DELAY_MS   500
#define NE_UART_BUF_SIZE        4096

/* -------------------------------------------------------------------------
 * Error codes
 * ---------------------------------------------------------------------- */
typedef enum {
    NE_OK               =  0,   /* Success */
    NE_ERR_TIMEOUT      = -1,   /* No response within timeout */
    NE_ERR_UART         = -2,   /* UART open / write / read failure */
    NE_ERR_JSON         = -3,   /* Malformed JSON in response */
    NE_ERR_CRC          = -4,   /* CRC mismatch — data corruption */
    NE_ERR_OVERFLOW     = -5,   /* Response larger than output buffer */
    NE_ERR_MODULE       = -6,   /* Module returned status:"error" */
    NE_ERR_NOT_INIT     = -7,   /* ne_init() not called or failed */
    NE_ERR_BUSY         = -8,   /* Module busy / previous cmd pending */
    NE_ERR_PARAM        = -9,   /* Invalid parameter */
    NE_ERR_NO_MODEL     = -10,  /* Requested model not loaded on module */
    NE_ERR_ALLOC        = -11,  /* Memory allocation failure */
} ne_error_t;

/* -------------------------------------------------------------------------
 * Log levels (set via ne_config_t.log_level)
 * ---------------------------------------------------------------------- */
typedef enum {
    NE_LOG_NONE    = 0,
    NE_LOG_ERROR   = 1,
    NE_LOG_WARN    = 2,
    NE_LOG_INFO    = 3,
    NE_LOG_DEBUG   = 4,
} ne_log_level_t;

/* -------------------------------------------------------------------------
 * Platform UART abstraction
 * Applications on bare-metal targets implement this interface.
 * On POSIX (Linux/macOS) and Arduino, built-in backends are provided.
 * ---------------------------------------------------------------------- */
typedef struct ne_uart_ops {
    /**
     * Open the UART port. Return 0 on success, negative on failure.
     * @param port   Port string, e.g. "/dev/ttyUSB0" or "COM3"
     * @param baud   Baud rate, e.g. 115200
     * @param ctx    Opaque context pointer stored in ne_handle_t
     */
    int  (*open)(const char *port, uint32_t baud, void **ctx);

    /**
     * Write bytes. Return number of bytes written, negative on error.
     */
    int  (*write)(void *ctx, const uint8_t *buf, size_t len);

    /**
     * Read up to len bytes, waiting at most timeout_ms milliseconds.
     * Return number of bytes read (0 = timeout), negative on error.
     */
    int  (*read)(void *ctx, uint8_t *buf, size_t len, uint32_t timeout_ms);

    /**
     * Flush TX and RX buffers.
     */
    void (*flush)(void *ctx);

    /**
     * Close the port and free resources.
     */
    void (*close)(void *ctx);
} ne_uart_ops_t;

/* -------------------------------------------------------------------------
 * Configuration
 * ---------------------------------------------------------------------- */
typedef struct ne_config {
    const char      *port;          /* UART port, e.g. "/dev/ttyUSB0" */
    uint32_t         baud;          /* Baud rate (default: 115200)     */
    uint32_t         timeout_ms;    /* Per-command timeout in ms       */
    uint8_t          max_retries;   /* Auto-retry on timeout/CRC error */
    ne_log_level_t   log_level;     /* Verbosity of built-in logger    */
    ne_uart_ops_t   *uart_ops;      /* NULL = use built-in POSIX/Win   */

    /**
     * Optional log callback. If NULL, logs go to stderr.
     * @param level   Log level of this message
     * @param msg     Null-terminated log string (no trailing newline)
     * @param user    User-supplied pointer from log_user_data
     */
    void (*log_cb)(ne_log_level_t level, const char *msg, void *user);
    void  *log_user_data;
} ne_config_t;

/** Convenience macro for default config on a given port */
#define NE_DEFAULT_CONFIG(uart_port) {          \
    .port          = (uart_port),               \
    .baud          = NE_DEFAULT_BAUD,           \
    .timeout_ms    = NE_DEFAULT_TIMEOUT_MS,     \
    .max_retries   = NE_MAX_RETRIES,            \
    .log_level     = NE_LOG_WARN,               \
    .uart_ops      = NULL,                      \
    .log_cb        = NULL,                      \
    .log_user_data = NULL,                      \
}

/* -------------------------------------------------------------------------
 * Opaque handle returned by ne_init()
 * ---------------------------------------------------------------------- */
typedef struct ne_handle ne_handle_t;

/* -------------------------------------------------------------------------
 * Module info (returned by ne_get_info)
 * ---------------------------------------------------------------------- */
typedef struct ne_module_info {
    char     firmware_version[32];
    char     loaded_model[64];      /* Currently active model name      */
    uint32_t flash_total_kb;
    uint32_t flash_free_kb;
    uint32_t heap_free_bytes;
    float    temperature_c;         /* ESP32 internal sensor (approx)   */
} ne_module_info_t;

/* -------------------------------------------------------------------------
 * Keyword detection result
 * ---------------------------------------------------------------------- */
typedef struct ne_keyword_result {
    bool     detected;
    char     keyword[NE_MAX_LABEL_LEN];
    float    confidence;            /* 0.0–1.0 */
    uint32_t latency_ms;
} ne_keyword_result_t;

/* -------------------------------------------------------------------------
 * Classification result
 * ---------------------------------------------------------------------- */
typedef struct ne_classify_result {
    char     label[NE_MAX_LABEL_LEN];
    float    confidence;            /* 0.0–1.0 */
    uint32_t latency_ms;
} ne_classify_result_t;

/* -------------------------------------------------------------------------
 * Core API
 * ---------------------------------------------------------------------- */

/**
 * Initialize the NeuroEdge SDK and open the UART connection.
 *
 * Allocates internal state, opens the port, and sends a ping to verify
 * the module is alive. Returns a handle on success, NULL on failure.
 * Caller must call ne_close() when done.
 *
 * @param config  Pointer to a filled ne_config_t. Must remain valid until
 *                ne_close() is called.
 * @return        Pointer to opaque handle, or NULL on failure.
 */
ne_handle_t *ne_init(const ne_config_t *config);

/**
 * Close the UART connection and free all resources.
 * After this call, the handle is invalid.
 */
void ne_close(ne_handle_t *ne);

/**
 * Send a natural-language question to the on-module LLM and receive a
 * text answer. Blocks until the response arrives or timeout expires.
 *
 * @param ne          Handle from ne_init()
 * @param prompt      Null-terminated question string (<= NE_MAX_PROMPT_LEN)
 * @param response    Output buffer for the answer text
 * @param resp_size   Size of the response buffer in bytes
 * @return            NE_OK on success, negative ne_error_t on failure
 */
ne_error_t ne_ask(ne_handle_t *ne,
                  const char  *prompt,
                  char        *response,
                  size_t       resp_size);

/**
 * Classify a raw sensor/data buffer using the on-module classifier.
 *
 * @param ne          Handle from ne_init()
 * @param data        Raw byte buffer (sensor readings, image, etc.)
 * @param data_len    Length of data in bytes (<= NE_MAX_SENSOR_BYTES)
 * @param result      Output: filled with label and confidence score
 * @return            NE_OK on success, negative ne_error_t on failure
 */
ne_error_t ne_classify(ne_handle_t          *ne,
                       const uint8_t        *data,
                       size_t                data_len,
                       ne_classify_result_t *result);

/**
 * Run keyword spotting on a PCM audio buffer.
 *
 * @param ne          Handle from ne_init()
 * @param pcm         16-bit PCM audio samples, 16 kHz mono
 * @param num_samples Number of samples in the buffer
 * @param result      Output: detection result
 * @return            NE_OK on success, negative ne_error_t on failure
 */
ne_error_t ne_detect_keyword(ne_handle_t          *ne,
                             const int16_t        *pcm,
                             size_t                num_samples,
                             ne_keyword_result_t  *result);

/**
 * Query module firmware version, loaded model, and resource usage.
 *
 * @param ne    Handle from ne_init()
 * @param info  Output: module information struct
 * @return      NE_OK on success, negative ne_error_t on failure
 */
ne_error_t ne_get_info(ne_handle_t *ne, ne_module_info_t *info);

/**
 * Send a ping and measure round-trip latency.
 *
 * @param ne        Handle from ne_init()
 * @param latency   Output: round-trip time in milliseconds
 * @return          NE_OK on success, negative ne_error_t on failure
 */
ne_error_t ne_ping(ne_handle_t *ne, uint32_t *latency_ms);

/**
 * Request the module to enter deep sleep. It will wake on next UART byte.
 *
 * @param ne           Handle from ne_init()
 * @param duration_ms  Sleep duration in ms. 0 = sleep until woken by UART.
 * @return             NE_OK on success, negative ne_error_t on failure
 */
ne_error_t ne_sleep(ne_handle_t *ne, uint32_t duration_ms);

/**
 * Convert an error code to a human-readable string.
 */
const char *ne_strerror(ne_error_t err);

/* -------------------------------------------------------------------------
 * Advanced / optional API
 * ---------------------------------------------------------------------- */

/**
 * Send a raw JSON command and receive a raw JSON response.
 * Useful for custom commands or future protocol extensions.
 *
 * @param ne            Handle from ne_init()
 * @param json_cmd      Null-terminated JSON object string (no newline)
 * @param json_resp     Output buffer for the JSON response
 * @param resp_size     Size of json_resp in bytes
 * @return              NE_OK on success, negative ne_error_t on failure
 */
ne_error_t ne_raw_command(ne_handle_t *ne,
                          const char  *json_cmd,
                          char        *json_resp,
                          size_t       resp_size);

/**
 * Set the active model on the module (model must be flashed).
 *
 * @param ne          Handle from ne_init()
 * @param model_name  Model identifier, e.g. "tinyllama-q4" or "dscnn-kws"
 * @return            NE_OK on success, NE_ERR_NO_MODEL if not found
 */
ne_error_t ne_set_model(ne_handle_t *ne, const char *model_name);

#ifdef __cplusplus
}
#endif

#endif /* NEUROEDGE_H */
