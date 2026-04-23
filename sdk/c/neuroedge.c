/**
 * NeuroEdge Host SDK — neuroedge.c
 *
 * POSIX implementation (Linux, macOS, Windows via MinGW).
 * On bare-metal targets (STM32, Arduino) link neuroedge_platform_*.c instead
 * and set ne_config_t.uart_ops to your platform driver.
 *
 * Thread safety: Each ne_handle_t is single-threaded. Protect with a mutex
 * if you call ne_* from multiple threads with the same handle.
 */

#include "neuroedge.h"

/* POSIX includes — guarded so the header stays clean for cross-compile */
#if defined(__linux__) || defined(__APPLE__) || defined(__unix__)
#  define NE_POSIX 1
#  include <termios.h>
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/time.h>
#  include <sys/select.h>
#  include <errno.h>
#elif defined(_WIN32)
#  define NE_WINDOWS 1
#  include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>

/* -------------------------------------------------------------------------
 * CRC16-CCITT (inline — avoids depending on firmware crc16.c) */

static uint16_t sdk_crc16(const uint8_t *data, size_t len)
{
    uint16_t crc = 0xFFFF;
    while (len--) {
        uint8_t byte = *data++;
        for (int i = 0; i < 8; i++) {
            if ((crc >> 15) ^ (byte >> 7)) crc = (uint16_t)((crc << 1) ^ 0x1021);
            else                            crc = (uint16_t)(crc << 1);
            byte <<= 1;
        }
    }
    return crc;
}

/* -------------------------------------------------------------------------
 * Time helpers */

static uint64_t ms_now(void)
{
#if defined(NE_POSIX)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
#elif defined(NE_WINDOWS)
    return (uint64_t)GetTickCount64();
#else
    return 0;
#endif
}

/* -------------------------------------------------------------------------
 * POSIX UART backend */

#if defined(NE_POSIX)

static int posix_open(const char *port, uint32_t baud, void **ctx)
{
    int fd = open(port, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) return -1;

    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(fd, &tty) != 0) { close(fd); return -1; }

    speed_t speed;
    switch (baud) {
        case 9600:   speed = B9600;   break;
        case 19200:  speed = B19200;  break;
        case 38400:  speed = B38400;  break;
        case 57600:  speed = B57600;  break;
        case 115200: speed = B115200; break;
        case 230400: speed = B230400; break;
        default:     speed = B115200; break;
    }
    cfsetispeed(&tty, speed);
    cfsetospeed(&tty, speed);

    tty.c_cflag  = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag |= CLOCAL | CREAD;
    tty.c_cflag &= ~(PARENB | PARODD | CSTOPB | CRTSCTS);
    tty.c_iflag  = IGNBRK;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY | ICRNL | INLCR);
    tty.c_oflag  = 0;
    tty.c_lflag  = 0;
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 0;

    tcsetattr(fd, TCSANOW, &tty);
    tcflush(fd, TCIOFLUSH);

    int *fdp = (int *)malloc(sizeof(int));
    if (!fdp) { close(fd); return -1; }
    *fdp = fd;
    *ctx = fdp;
    return 0;
}

static int posix_write(void *ctx, const uint8_t *buf, size_t len)
{
    int fd = *(int *)ctx;
    ssize_t n = write(fd, buf, len);
    return (n < 0) ? -1 : (int)n;
}

static int posix_read(void *ctx, uint8_t *buf, size_t len, uint32_t timeout_ms)
{
    int fd = *(int *)ctx;
    uint64_t deadline = ms_now() + timeout_ms;
    size_t total = 0;

    while (total < len) {
        uint64_t now = ms_now();
        if (now >= deadline) break;

        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);

        struct timeval tv;
        uint64_t rem = deadline - now;
        tv.tv_sec  = (time_t)(rem / 1000);
        tv.tv_usec = (suseconds_t)((rem % 1000) * 1000);

        int rc = select(fd + 1, &rfds, NULL, NULL, &tv);
        if (rc <= 0) break;  /* Timeout or error */

        ssize_t n = read(fd, buf + total, len - total);
        if (n < 0) {
            if (errno == EAGAIN || errno == EINTR) continue;
            return -1;
        }
        total += (size_t)n;
    }
    return (int)total;
}

static void posix_flush(void *ctx)
{
    int fd = *(int *)ctx;
    tcflush(fd, TCIOFLUSH);
}

static void posix_close(void *ctx)
{
    if (!ctx) return;
    int fd = *(int *)ctx;
    close(fd);
    free(ctx);
}

static ne_uart_ops_t s_posix_ops = {
    .open  = posix_open,
    .write = posix_write,
    .read  = posix_read,
    .flush = posix_flush,
    .close = posix_close,
};

#endif /* NE_POSIX */

/* -------------------------------------------------------------------------
 * Handle structure */

#define NE_HANDLE_MAGIC 0x4E455247UL  /* "NERG" */

struct ne_handle {
    uint32_t          magic;
    ne_config_t       cfg;
    ne_uart_ops_t    *ops;
    void             *uart_ctx;
    uint32_t          next_id;
    char              rx_buf[NE_UART_BUF_SIZE];
    char              tx_buf[NE_UART_BUF_SIZE];
};

/* -------------------------------------------------------------------------
 * Logging */

static void ne_log(const ne_handle_t *ne, ne_log_level_t level,
                   const char *fmt, ...)
{
    if (!ne || level > ne->cfg.log_level) return;

    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);

    if (ne->cfg.log_cb) {
        ne->cfg.log_cb(level, msg, ne->cfg.log_user_data);
    } else {
        static const char *level_str[] = {"", "ERROR", "WARN", "INFO", "DEBUG"};
        fprintf(stderr, "[NE %s] %s\n",
                level < 5 ? level_str[level] : "?", msg);
    }
}

/* -------------------------------------------------------------------------
 * Frame I/O */

/* Read bytes until we get a '\n', or timeout. */
static ne_error_t read_line(ne_handle_t *ne, uint32_t timeout_ms)
{
    size_t   pos      = 0;
    uint64_t deadline = ms_now() + timeout_ms;

    while (pos < sizeof(ne->rx_buf) - 1) {
        uint64_t now = ms_now();
        if (now >= deadline) return NE_ERR_TIMEOUT;

        uint32_t rem = (uint32_t)(deadline - now);
        uint8_t byte;
        int n = ne->ops->read(ne->uart_ctx, &byte, 1, rem < 10 ? rem : 10);

        if (n < 0)  return NE_ERR_UART;
        if (n == 0) continue;  /* partial timeout — loop */

        if (byte == '\n') {
            /* Trim trailing '\r' if present */
            if (pos > 0 && ne->rx_buf[pos - 1] == '\r') pos--;
            ne->rx_buf[pos] = '\0';
            return NE_OK;
        }
        if (byte == '\r') continue;  /* Skip bare CR */

        ne->rx_buf[pos++] = (char)byte;
    }
    return NE_ERR_OVERFLOW;
}

/* Send a JSON command line (appends \r\n). */
static ne_error_t send_frame(ne_handle_t *ne, const char *json)
{
    size_t len = strlen(json);
    int n = ne->ops->write(ne->uart_ctx, (const uint8_t *)json, len);
    if (n != (int)len) return NE_ERR_UART;
    n = ne->ops->write(ne->uart_ctx, (const uint8_t *)"\r\n", 2);
    if (n != 2) return NE_ERR_UART;
    return NE_OK;
}

/* -------------------------------------------------------------------------
 * Simple JSON field extractors (no dependency on cJSON in the SDK) */

/* Extract a string value from key in a flat JSON object.
 * Writes to out_buf (size out_size), returns true on success. */
static bool json_get_string(const char *json, const char *key,
                             char *out_buf, size_t out_size)
{
    /* Search for "key":" */
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\":\"", key);
    const char *p = strstr(json, needle);
    if (!p) return false;
    p += strlen(needle);

    size_t i = 0;
    while (*p && *p != '"' && i < out_size - 1) {
        if (*p == '\\' && *(p + 1)) { p++; }  /* Skip escape */
        out_buf[i++] = *p++;
    }
    out_buf[i] = '\0';
    return (*p == '"');
}

/* Extract a number value from key. Returns true on success. */
static bool json_get_number(const char *json, const char *key, double *out)
{
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    const char *p = strstr(json, needle);
    if (!p) return false;
    p += strlen(needle);
    while (*p == ' ') p++;
    char *end;
    *out = strtod(p, &end);
    return (end > p);
}

/* Extract a bool value (true/false). Returns true on success. */
static bool json_get_bool(const char *json, const char *key, bool *out)
{
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    const char *p = strstr(json, needle);
    if (!p) return false;
    p += strlen(needle);
    while (*p == ' ') p++;
    if (strncmp(p, "true",  4) == 0) { *out = true;  return true; }
    if (strncmp(p, "false", 5) == 0) { *out = false; return true; }
    return false;
}

/* -------------------------------------------------------------------------
 * Command execution with retry */

typedef struct {
    const char *cmd_json;       /* Formatted command */
    char       *resp_buf;       /* Output: raw JSON response */
    size_t      resp_size;
} ne_txn_t;

static ne_error_t execute(ne_handle_t *ne, const ne_txn_t *txn)
{
    ne_error_t err = NE_ERR_TIMEOUT;

    for (int attempt = 0; attempt <= ne->cfg.max_retries; attempt++) {
        if (attempt > 0) {
            ne_log(ne, NE_LOG_WARN, "Retry %d/%d", attempt, ne->cfg.max_retries);
            ne->ops->flush(ne->uart_ctx);

            /* Re-open on persistent UART errors */
            if (err == NE_ERR_UART) {
                ne_log(ne, NE_LOG_WARN, "Attempting UART reconnect");
                ne->ops->close(ne->uart_ctx);
                ne->uart_ctx = NULL;
                int rc = ne->ops->open(ne->cfg.port, ne->cfg.baud, &ne->uart_ctx);
                if (rc != 0) {
                    ne_log(ne, NE_LOG_ERROR, "Reconnect failed");
                    continue;
                }
            }
        }

        err = send_frame(ne, txn->cmd_json);
        if (err != NE_OK) continue;

        err = read_line(ne, ne->cfg.timeout_ms);
        if (err != NE_OK) continue;

        /* Copy raw response to caller */
        size_t rlen = strlen(ne->rx_buf);
        if (rlen >= txn->resp_size) { err = NE_ERR_OVERFLOW; continue; }
        memcpy(txn->resp_buf, ne->rx_buf, rlen + 1);

        /* Check status field */
        char status[16];
        if (!json_get_string(ne->rx_buf, "status", status, sizeof(status))) {
            err = NE_ERR_JSON;
            continue;
        }
        if (strcmp(status, "ok") != 0) {
            ne_log(ne, NE_LOG_WARN, "Module error: %s", ne->rx_buf);
            err = NE_ERR_MODULE;
            break;  /* No retry on module-side errors */
        }

        /* Verify CRC if present in response */
        char crc_hex[8];
        if (json_get_string(ne->rx_buf, "crc", crc_hex, sizeof(crc_hex))) {
            const char *crc_pos = strstr(ne->rx_buf, "\"crc\"");
            if (crc_pos) {
                size_t check_len = (size_t)(crc_pos - ne->rx_buf);
                while (check_len > 0 && (ne->rx_buf[check_len-1] == ',' ||
                                         ne->rx_buf[check_len-1] == ' '))
                    check_len--;
                uint16_t computed = sdk_crc16((const uint8_t *)ne->rx_buf, check_len);
                uint16_t received = (uint16_t)strtoul(crc_hex, NULL, 16);
                if (computed != received) {
                    ne_log(ne, NE_LOG_WARN, "Response CRC mismatch %04X vs %04X",
                           computed, received);
                    err = NE_ERR_CRC;
                    continue;
                }
            }
        }

        return NE_OK;
    }

    ne_log(ne, NE_LOG_ERROR, "Command failed after %d attempts: %s",
           ne->cfg.max_retries + 1, ne_strerror(err));
    return err;
}

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

ne_handle_t *ne_init(const ne_config_t *config)
{
    if (!config || !config->port) return NULL;

    ne_handle_t *ne = (ne_handle_t *)calloc(1, sizeof(ne_handle_t));
    if (!ne) return NULL;

    ne->magic   = NE_HANDLE_MAGIC;
    ne->cfg     = *config;
    ne->next_id = 1;

    /* Select UART backend */
#if defined(NE_POSIX)
    ne->ops = config->uart_ops ? config->uart_ops : &s_posix_ops;
#else
    ne->ops = config->uart_ops;
    if (!ne->ops) {
        fprintf(stderr, "[NE ERROR] No UART backend — set uart_ops in config\n");
        free(ne);
        return NULL;
    }
#endif

    /* Apply defaults */
    if (ne->cfg.baud == 0)        ne->cfg.baud        = NE_DEFAULT_BAUD;
    if (ne->cfg.timeout_ms == 0)  ne->cfg.timeout_ms  = NE_DEFAULT_TIMEOUT_MS;
    if (ne->cfg.max_retries == 0) ne->cfg.max_retries = NE_MAX_RETRIES;

    int rc = ne->ops->open(ne->cfg.port, ne->cfg.baud, &ne->uart_ctx);
    if (rc != 0) {
        ne_log(ne, NE_LOG_ERROR, "Failed to open port: %s", ne->cfg.port);
        free(ne);
        return NULL;
    }

    /* Ping to confirm module is alive */
    uint32_t latency;
    ne_error_t err = ne_ping(ne, &latency);
    if (err != NE_OK) {
        ne_log(ne, NE_LOG_ERROR, "Module not responding on %s (err=%s)",
               ne->cfg.port, ne_strerror(err));
        ne->ops->close(ne->uart_ctx);
        free(ne);
        return NULL;
    }

    ne_log(ne, NE_LOG_INFO, "Connected to NeuroEdge module on %s (ping %u ms)",
           ne->cfg.port, latency);
    return ne;
}

void ne_close(ne_handle_t *ne)
{
    if (!ne || ne->magic != NE_HANDLE_MAGIC) return;
    ne->ops->close(ne->uart_ctx);
    ne->magic = 0;
    free(ne);
}

ne_error_t ne_ping(ne_handle_t *ne, uint32_t *latency_ms)
{
    if (!ne || ne->magic != NE_HANDLE_MAGIC) return NE_ERR_NOT_INIT;
    if (!latency_ms) return NE_ERR_PARAM;

    uint32_t id = ne->next_id++;
    uint64_t t0 = ms_now();

    snprintf(ne->tx_buf, sizeof(ne->tx_buf),
             "{\"id\":%" PRIu32 ",\"cmd\":\"ping\"}", id);

    char resp[256];
    ne_txn_t txn = { ne->tx_buf, resp, sizeof(resp) };
    ne_error_t err = execute(ne, &txn);
    if (err != NE_OK) return err;

    *latency_ms = (uint32_t)(ms_now() - t0);
    return NE_OK;
}

ne_error_t ne_ask(ne_handle_t *ne,
                  const char  *prompt,
                  char        *response,
                  size_t       resp_size)
{
    if (!ne || ne->magic != NE_HANDLE_MAGIC) return NE_ERR_NOT_INIT;
    if (!prompt || !response || resp_size == 0) return NE_ERR_PARAM;
    if (strlen(prompt) > NE_MAX_PROMPT_LEN) return NE_ERR_PARAM;

    uint32_t id = ne->next_id++;

    /* Escape double-quotes in the prompt */
    char escaped[NE_MAX_PROMPT_LEN * 2];
    size_t ei = 0;
    for (const char *p = prompt; *p && ei < sizeof(escaped) - 2; p++) {
        if (*p == '"' || *p == '\\') escaped[ei++] = '\\';
        escaped[ei++] = *p;
    }
    escaped[ei] = '\0';

    snprintf(ne->tx_buf, sizeof(ne->tx_buf),
             "{\"id\":%" PRIu32 ",\"cmd\":\"ask\","
             "\"prompt\":\"%s\",\"max_tokens\":64}",
             id, escaped);

    char raw_resp[NE_MAX_RESPONSE_LEN];
    ne_txn_t txn = { ne->tx_buf, raw_resp, sizeof(raw_resp) };
    ne_error_t err = execute(ne, &txn);
    if (err != NE_OK) return err;

    if (!json_get_string(raw_resp, "text", response, resp_size)) {
        return NE_ERR_JSON;
    }
    return NE_OK;
}

ne_error_t ne_classify(ne_handle_t          *ne,
                       const uint8_t        *data,
                       size_t                data_len,
                       ne_classify_result_t *result)
{
    if (!ne || ne->magic != NE_HANDLE_MAGIC) return NE_ERR_NOT_INIT;
    if (!data || !result || data_len == 0)    return NE_ERR_PARAM;
    if (data_len > NE_MAX_SENSOR_BYTES)       return NE_ERR_PARAM;

    uint32_t id = ne->next_id++;

    /* Encode data as JSON byte array */
    char arr_buf[NE_MAX_SENSOR_BYTES * 5];  /* "255," per byte */
    size_t pos = 0;
    arr_buf[pos++] = '[';
    for (size_t i = 0; i < data_len; i++) {
        pos += (size_t)snprintf(arr_buf + pos, sizeof(arr_buf) - pos,
                                "%u%s", data[i], i + 1 < data_len ? "," : "");
    }
    arr_buf[pos++] = ']';
    arr_buf[pos]   = '\0';

    snprintf(ne->tx_buf, sizeof(ne->tx_buf),
             "{\"id\":%" PRIu32 ",\"cmd\":\"classify\",\"data\":%s}",
             id, arr_buf);

    char raw_resp[512];
    ne_txn_t txn = { ne->tx_buf, raw_resp, sizeof(raw_resp) };
    ne_error_t err = execute(ne, &txn);
    if (err != NE_OK) return err;

    if (!json_get_string(raw_resp, "label", result->label, sizeof(result->label)))
        return NE_ERR_JSON;

    double conf = 0.0, ms = 0.0;
    json_get_number(raw_resp, "confidence", &conf);
    json_get_number(raw_resp, "ms", &ms);
    result->confidence  = (float)conf;
    result->latency_ms  = (uint32_t)ms;
    return NE_OK;
}

ne_error_t ne_detect_keyword(ne_handle_t          *ne,
                             const int16_t        *pcm,
                             size_t                num_samples,
                             ne_keyword_result_t  *result)
{
    if (!ne || ne->magic != NE_HANDLE_MAGIC) return NE_ERR_NOT_INIT;
    if (!pcm || !result || num_samples == 0) return NE_ERR_PARAM;

    /* PCM is sent as JSON array — limit to 1 second @ 16 kHz */
    if (num_samples > 16000) num_samples = 16000;

    uint32_t id = ne->next_id++;

    /* Build PCM array — max 16000 samples * 7 chars = ~112 KB; use heap */
    size_t arr_cap = num_samples * 8 + 8;
    char  *arr_buf = (char *)malloc(arr_cap);
    if (!arr_buf) return NE_ERR_ALLOC;

    size_t pos = 0;
    arr_buf[pos++] = '[';
    for (size_t i = 0; i < num_samples; i++) {
        pos += (size_t)snprintf(arr_buf + pos, arr_cap - pos,
                                "%d%s", pcm[i], i + 1 < num_samples ? "," : "");
    }
    arr_buf[pos++] = ']';
    arr_buf[pos]   = '\0';

    /* Build the full command in tx_buf; may be large, use dynamic if needed */
    size_t cmd_needed = pos + 64;
    char  *cmd_buf = ne->tx_buf;
    char  *dyn_buf = NULL;
    if (cmd_needed > sizeof(ne->tx_buf)) {
        dyn_buf = (char *)malloc(cmd_needed);
        if (!dyn_buf) { free(arr_buf); return NE_ERR_ALLOC; }
        cmd_buf = dyn_buf;
    }

    snprintf(cmd_buf, cmd_needed,
             "{\"id\":%" PRIu32 ",\"cmd\":\"kws\",\"pcm\":%s}",
             id, arr_buf);
    free(arr_buf);

    char raw_resp[512];
    ne_txn_t txn = { cmd_buf, raw_resp, sizeof(raw_resp) };
    ne_error_t err = execute(ne, &txn);
    free(dyn_buf);
    if (err != NE_OK) return err;

    json_get_bool(raw_resp,   "detected",   &result->detected);
    json_get_string(raw_resp, "keyword",     result->keyword, sizeof(result->keyword));
    double conf = 0.0, ms = 0.0;
    json_get_number(raw_resp, "confidence", &conf);
    json_get_number(raw_resp, "ms",         &ms);
    result->confidence  = (float)conf;
    result->latency_ms  = (uint32_t)ms;
    return NE_OK;
}

ne_error_t ne_get_info(ne_handle_t *ne, ne_module_info_t *info)
{
    if (!ne || ne->magic != NE_HANDLE_MAGIC) return NE_ERR_NOT_INIT;
    if (!info) return NE_ERR_PARAM;

    uint32_t id = ne->next_id++;
    snprintf(ne->tx_buf, sizeof(ne->tx_buf),
             "{\"id\":%" PRIu32 ",\"cmd\":\"info\"}", id);

    char raw_resp[512];
    ne_txn_t txn = { ne->tx_buf, raw_resp, sizeof(raw_resp) };
    ne_error_t err = execute(ne, &txn);
    if (err != NE_OK) return err;

    json_get_string(raw_resp, "firmware",  info->firmware_version, sizeof(info->firmware_version));
    json_get_string(raw_resp, "model",     info->loaded_model,     sizeof(info->loaded_model));
    double hf = 0.0, hm = 0.0;
    json_get_number(raw_resp, "heap_free", &hf);
    json_get_number(raw_resp, "heap_min",  &hm);
    info->heap_free_bytes = (uint32_t)hf;
    return NE_OK;
}

ne_error_t ne_sleep(ne_handle_t *ne, uint32_t duration_ms)
{
    if (!ne || ne->magic != NE_HANDLE_MAGIC) return NE_ERR_NOT_INIT;

    uint32_t id = ne->next_id++;
    snprintf(ne->tx_buf, sizeof(ne->tx_buf),
             "{\"id\":%" PRIu32 ",\"cmd\":\"sleep\",\"duration_ms\":%" PRIu32 "}",
             id, duration_ms);

    char raw_resp[256];
    ne_txn_t txn = { ne->tx_buf, raw_resp, sizeof(raw_resp) };
    return execute(ne, &txn);
}

ne_error_t ne_raw_command(ne_handle_t *ne,
                          const char  *json_cmd,
                          char        *json_resp,
                          size_t       resp_size)
{
    if (!ne || ne->magic != NE_HANDLE_MAGIC) return NE_ERR_NOT_INIT;
    if (!json_cmd || !json_resp || resp_size == 0) return NE_ERR_PARAM;

    ne_txn_t txn = { json_cmd, json_resp, resp_size };
    return execute(ne, &txn);
}

ne_error_t ne_set_model(ne_handle_t *ne, const char *model_name)
{
    if (!ne || ne->magic != NE_HANDLE_MAGIC) return NE_ERR_NOT_INIT;
    if (!model_name) return NE_ERR_PARAM;

    uint32_t id = ne->next_id++;
    snprintf(ne->tx_buf, sizeof(ne->tx_buf),
             "{\"id\":%" PRIu32 ",\"cmd\":\"set_model\",\"model\":\"%s\"}",
             id, model_name);

    char raw_resp[256];
    ne_txn_t txn = { ne->tx_buf, raw_resp, sizeof(raw_resp) };
    return execute(ne, &txn);
}

const char *ne_strerror(ne_error_t err)
{
    switch (err) {
        case NE_OK:           return "OK";
        case NE_ERR_TIMEOUT:  return "Timeout";
        case NE_ERR_UART:     return "UART error";
        case NE_ERR_JSON:     return "JSON parse error";
        case NE_ERR_CRC:      return "CRC mismatch";
        case NE_ERR_OVERFLOW: return "Buffer overflow";
        case NE_ERR_MODULE:   return "Module returned error";
        case NE_ERR_NOT_INIT: return "Not initialized";
        case NE_ERR_BUSY:     return "Module busy";
        case NE_ERR_PARAM:    return "Invalid parameter";
        case NE_ERR_NO_MODEL: return "Model not found";
        case NE_ERR_ALLOC:    return "Memory allocation failed";
        default:              return "Unknown error";
    }
}
