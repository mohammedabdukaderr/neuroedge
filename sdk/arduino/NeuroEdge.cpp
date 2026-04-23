/**
 * NeuroEdge Arduino Library — NeuroEdge.cpp
 */

#include "NeuroEdge.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* -------------------------------------------------------------------------
 * CRC16-CCITT */

uint16_t NeuroEdge::crc16(const uint8_t *data, size_t len)
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
 * Constructor */

NeuroEdge::NeuroEdge()
    : _serial(nullptr), _timeout_ms(5000), _next_id(1), _initialized(false)
{
    _rx_buf[0] = '\0';
    _tx_buf[0] = '\0';
}

/* -------------------------------------------------------------------------
 * begin() */

ne_err_t NeuroEdge::begin(HardwareSerial &serial, uint32_t timeout_ms)
{
    _serial      = &serial;
    _timeout_ms  = timeout_ms;
    _next_id     = 1;
    _initialized = false;

    /* Flush any garbage in the buffer */
    while (_serial->available()) _serial->read();

    /* Ping the module */
    uint32_t latency;
    ne_err_t err = ping(latency);
    if (err != NE_OK) return err;

    _initialized = true;
    return NE_OK;
}

/* -------------------------------------------------------------------------
 * I/O helpers */

ne_err_t NeuroEdge::sendFrame(const char *json)
{
    if (!_serial) return NE_ERR_NOT_INIT;
    size_t written = _serial->print(json);
    written += _serial->print("\r\n");
    return (written > 2) ? NE_OK : NE_ERR_UART;
}

ne_err_t NeuroEdge::readLine(uint32_t timeout_ms)
{
    if (!_serial) return NE_ERR_NOT_INIT;

    unsigned long deadline = millis() + timeout_ms;
    size_t pos = 0;

    while (millis() < deadline) {
        if (!_serial->available()) {
            delay(1);
            continue;
        }
        char c = (char)_serial->read();
        if (c == '\n') {
            if (pos > 0 && _rx_buf[pos - 1] == '\r') pos--;
            _rx_buf[pos] = '\0';
            return NE_OK;
        }
        if (c == '\r') continue;
        if (pos >= sizeof(_rx_buf) - 1) {
            _rx_buf[pos] = '\0';
            return NE_ERR_OVERFLOW;
        }
        _rx_buf[pos++] = c;
    }
    return NE_ERR_TIMEOUT;
}

ne_err_t NeuroEdge::executeCommand(const char *cmd,
                                    char *resp_buf, size_t resp_size)
{
    const int max_retries = 3;

    for (int attempt = 0; attempt < max_retries; attempt++) {
        if (attempt > 0) {
            while (_serial->available()) _serial->read();
        }

        ne_err_t err = sendFrame(cmd);
        if (err != NE_OK) continue;

        err = readLine(_timeout_ms);
        if (err != NE_OK) continue;

        /* Validate status */
        char status[16];
        if (!jsonGetString(_rx_buf, "status", status, sizeof(status)))
            return NE_ERR_JSON;
        if (strcmp(status, "ok") != 0)
            return NE_ERR_MODULE;

        /* Verify CRC if present */
        char crc_hex[8];
        if (jsonGetString(_rx_buf, "crc", crc_hex, sizeof(crc_hex))) {
            const char *crc_pos = strstr(_rx_buf, "\"crc\"");
            if (crc_pos) {
                size_t check_len = (size_t)(crc_pos - _rx_buf);
                while (check_len > 0 &&
                       (_rx_buf[check_len-1] == ',' || _rx_buf[check_len-1] == ' '))
                    check_len--;
                uint16_t computed = crc16((const uint8_t *)_rx_buf, check_len);
                uint16_t received = (uint16_t)strtoul(crc_hex, nullptr, 16);
                if (computed != received) { continue; }  /* CRC error, retry */
            }
        }

        size_t rlen = strlen(_rx_buf);
        if (rlen >= resp_size) return NE_ERR_OVERFLOW;
        memcpy(resp_buf, _rx_buf, rlen + 1);
        return NE_OK;
    }
    return NE_ERR_TIMEOUT;
}

/* -------------------------------------------------------------------------
 * Minimal JSON field extractors */

bool NeuroEdge::jsonGetString(const char *json, const char *key,
                               char *out, size_t out_size)
{
    char needle[48];
    snprintf(needle, sizeof(needle), "\"%s\":\"", key);
    const char *p = strstr(json, needle);
    if (!p) return false;
    p += strlen(needle);
    size_t i = 0;
    while (*p && *p != '"' && i < out_size - 1) {
        if (*p == '\\' && *(p+1)) p++;
        out[i++] = *p++;
    }
    out[i] = '\0';
    return (*p == '"');
}

bool NeuroEdge::jsonGetFloat(const char *json, const char *key, float &out)
{
    char needle[48];
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    const char *p = strstr(json, needle);
    if (!p) return false;
    p += strlen(needle);
    while (*p == ' ') p++;
    char *end;
    out = strtof(p, &end);
    return (end > p);
}

bool NeuroEdge::jsonGetBool(const char *json, const char *key, bool &out)
{
    char needle[48];
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    const char *p = strstr(json, needle);
    if (!p) return false;
    p += strlen(needle);
    while (*p == ' ') p++;
    if (strncmp(p, "true",  4) == 0) { out = true;  return true; }
    if (strncmp(p, "false", 5) == 0) { out = false; return true; }
    return false;
}

/* -------------------------------------------------------------------------
 * Public API */

ne_err_t NeuroEdge::ping(uint32_t &latency_ms)
{
    uint32_t id = _next_id++;
    snprintf(_tx_buf, sizeof(_tx_buf),
             "{\"id\":%lu,\"cmd\":\"ping\"}", (unsigned long)id);

    unsigned long t0 = millis();
    char resp[256];
    ne_err_t err = executeCommand(_tx_buf, resp, sizeof(resp));
    latency_ms = (uint32_t)(millis() - t0);
    return err;
}

ne_err_t NeuroEdge::ask(const char *prompt, char *response, size_t resp_size)
{
    if (!_initialized && _next_id > 1) return NE_ERR_NOT_INIT;
    if (!prompt || !response || resp_size == 0) return NE_ERR_PARAM;

    /* Escape prompt */
    char esc[512];
    size_t ei = 0;
    for (const char *p = prompt; *p && ei < sizeof(esc) - 2; p++) {
        if (*p == '"' || *p == '\\') esc[ei++] = '\\';
        esc[ei++] = *p;
    }
    esc[ei] = '\0';

    uint32_t id = _next_id++;
    snprintf(_tx_buf, sizeof(_tx_buf),
             "{\"id\":%lu,\"cmd\":\"ask\",\"prompt\":\"%s\",\"max_tokens\":64}",
             (unsigned long)id, esc);

    char raw[2048];
    ne_err_t err = executeCommand(_tx_buf, raw, sizeof(raw));
    if (err != NE_OK) return err;

    if (!jsonGetString(raw, "text", response, resp_size))
        return NE_ERR_JSON;
    return NE_OK;
}

ne_err_t NeuroEdge::ask(const char *prompt, String &result)
{
    char buf[512];
    ne_err_t err = ask(prompt, buf, sizeof(buf));
    if (err == NE_OK) result = String(buf);
    return err;
}

ne_err_t NeuroEdge::classify(const uint8_t *data, size_t data_len,
                              NeClassifyResult &result)
{
    if (!data || data_len == 0 || data_len > 512) return NE_ERR_PARAM;

    uint32_t id = _next_id++;

    /* Build data array in tx_buf — keep small to fit in stack */
    String cmd = "{\"id\":";
    cmd += (unsigned long)id;
    cmd += ",\"cmd\":\"classify\",\"data\":[";
    for (size_t i = 0; i < data_len; i++) {
        cmd += data[i];
        if (i + 1 < data_len) cmd += ',';
    }
    cmd += "]}";

    char raw[512];
    ne_err_t err = executeCommand(cmd.c_str(), raw, sizeof(raw));
    if (err != NE_OK) return err;

    jsonGetString(raw, "label", result.label, sizeof(result.label));
    float conf = 0.0f, ms = 0.0f;
    jsonGetFloat(raw, "confidence", conf);
    jsonGetFloat(raw, "ms",         ms);
    result.confidence = conf;
    result.latency_ms = (uint32_t)ms;
    return NE_OK;
}

ne_err_t NeuroEdge::detectKeyword(const int16_t *pcm, size_t num_samples,
                                   NeKeywordResult &result)
{
    if (!pcm || num_samples == 0) return NE_ERR_PARAM;
    if (num_samples > 16000) num_samples = 16000;

    uint32_t id = _next_id++;

    /* Build PCM command using String to handle heap allocation gracefully */
    String cmd;
    cmd.reserve((int)(num_samples * 7 + 64));
    cmd  = "{\"id\":";
    cmd += (unsigned long)id;
    cmd += ",\"cmd\":\"kws\",\"pcm\":[";
    for (size_t i = 0; i < num_samples; i++) {
        cmd += pcm[i];
        if (i + 1 < num_samples) cmd += ',';
    }
    cmd += "]}";

    char raw[512];
    ne_err_t err = executeCommand(cmd.c_str(), raw, sizeof(raw));
    if (err != NE_OK) return err;

    bool det = false;
    jsonGetBool(raw,   "detected",   det);
    jsonGetString(raw, "keyword",    result.keyword, sizeof(result.keyword));
    float conf = 0.0f, ms = 0.0f;
    jsonGetFloat(raw, "confidence",  conf);
    jsonGetFloat(raw, "ms",          ms);
    result.detected   = det;
    result.confidence = conf;
    result.latency_ms = (uint32_t)ms;
    return NE_OK;
}

ne_err_t NeuroEdge::sleep(uint32_t duration_ms)
{
    uint32_t id = _next_id++;
    snprintf(_tx_buf, sizeof(_tx_buf),
             "{\"id\":%lu,\"cmd\":\"sleep\",\"duration_ms\":%lu}",
             (unsigned long)id, (unsigned long)duration_ms);
    char resp[256];
    return executeCommand(_tx_buf, resp, sizeof(resp));
}

ne_err_t NeuroEdge::setModel(const char *modelName)
{
    if (!modelName) return NE_ERR_PARAM;
    uint32_t id = _next_id++;
    snprintf(_tx_buf, sizeof(_tx_buf),
             "{\"id\":%lu,\"cmd\":\"set_model\",\"model\":\"%s\"}",
             (unsigned long)id, modelName);
    char resp[256];
    return executeCommand(_tx_buf, resp, sizeof(resp));
}

bool NeuroEdge::isConnected()
{
    uint32_t lat;
    return (ping(lat) == NE_OK);
}

const char *NeuroEdge::errorString(ne_err_t err)
{
    switch (err) {
        case NE_OK:            return "OK";
        case NE_ERR_TIMEOUT:   return "Timeout";
        case NE_ERR_UART:      return "UART error";
        case NE_ERR_JSON:      return "JSON error";
        case NE_ERR_CRC:       return "CRC mismatch";
        case NE_ERR_OVERFLOW:  return "Buffer overflow";
        case NE_ERR_MODULE:    return "Module error";
        case NE_ERR_NOT_INIT:  return "Not initialized";
        case NE_ERR_PARAM:     return "Invalid parameter";
        default:               return "Unknown error";
    }
}
