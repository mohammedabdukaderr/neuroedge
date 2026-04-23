/**
 * NeuroEdge Arduino Library
 *
 * Wraps the NeuroEdge C SDK for use in Arduino sketches.
 * Works on any Arduino board with a hardware serial port:
 *   Arduino Uno R4, ESP32, STM32duino, nRF52840, Teensy, etc.
 *
 * Minimal usage (10 lines or fewer):
 *
 *   #include <NeuroEdge.h>
 *   NeuroEdge ne;
 *   void setup() {
 *     Serial1.begin(115200);
 *     ne.begin(Serial1);
 *   }
 *   void loop() {
 *     char answer[128];
 *     if (ne.ask("Is this heartbeat abnormal?", answer, sizeof(answer)) == NE_OK)
 *       Serial.println(answer);
 *     delay(5000);
 *   }
 */

#ifndef NEUROEDGE_ARDUINO_H
#define NEUROEDGE_ARDUINO_H

#include <Arduino.h>
#include <stdint.h>

/* -------------------------------------------------------------------------
 * Error codes (mirrored from neuroedge.h for Arduino convenience) */

typedef enum {
    NE_OK            =  0,
    NE_ERR_TIMEOUT   = -1,
    NE_ERR_UART      = -2,
    NE_ERR_JSON      = -3,
    NE_ERR_CRC       = -4,
    NE_ERR_OVERFLOW  = -5,
    NE_ERR_MODULE    = -6,
    NE_ERR_NOT_INIT  = -7,
    NE_ERR_PARAM     = -9,
} ne_err_t;

/* -------------------------------------------------------------------------
 * Keyword detection result */

struct NeKeywordResult {
    bool     detected;
    char     keyword[32];
    float    confidence;
    uint32_t latency_ms;
};

/* -------------------------------------------------------------------------
 * Classification result */

struct NeClassifyResult {
    char     label[32];
    float    confidence;
    uint32_t latency_ms;
};

/* -------------------------------------------------------------------------
 * NeuroEdge class */

class NeuroEdge {
public:
    /**
     * Default constructor. Call begin() before any other method.
     */
    NeuroEdge();

    /**
     * Initialize with a HardwareSerial port (Serial1, Serial2, …).
     *
     * @param serial      Hardware serial port wired to the NeuroEdge module
     * @param timeout_ms  Per-command timeout in ms (default: 5000)
     * @return            NE_OK if the module responds to a ping, error otherwise
     */
    ne_err_t begin(HardwareSerial &serial, uint32_t timeout_ms = 5000);

    /**
     * Send a natural-language question to the on-module LLM.
     *
     * @param prompt    Question string
     * @param response  Output buffer for the answer
     * @param resp_size Size of the response buffer
     * @return          NE_OK on success
     */
    ne_err_t ask(const char *prompt, char *response, size_t resp_size);

    /** String overload — writes result into a String object. */
    ne_err_t ask(const char *prompt, String &result);

    /**
     * Classify a raw byte buffer (sensor data, image, etc.).
     *
     * @param data      Pointer to byte array
     * @param data_len  Number of bytes (max 512)
     * @param result    Output classification result
     * @return          NE_OK on success
     */
    ne_err_t classify(const uint8_t *data, size_t data_len,
                      NeClassifyResult &result);

    /**
     * Run keyword spotting on a PCM audio buffer (16-bit, 16 kHz, mono).
     *
     * @param pcm         Audio samples
     * @param num_samples Number of samples
     * @param result      Output detection result
     * @return            NE_OK on success
     */
    ne_err_t detectKeyword(const int16_t *pcm, size_t num_samples,
                           NeKeywordResult &result);

    /**
     * Measure round-trip latency to the module in ms.
     *
     * @param latency_ms  Output: measured latency
     * @return            NE_OK on success
     */
    ne_err_t ping(uint32_t &latency_ms);

    /**
     * Put the module into deep sleep.
     *
     * @param duration_ms  Sleep time in ms. 0 = sleep until next UART byte.
     * @return             NE_OK on success
     */
    ne_err_t sleep(uint32_t duration_ms = 0);

    /**
     * Switch the active AI model on the module.
     *
     * @param modelName  "tinyllama-q4", "mobilenet-v3", or "dscnn-kws"
     * @return           NE_OK on success
     */
    ne_err_t setModel(const char *modelName);

    /** Return a human-readable string for an error code. */
    static const char *errorString(ne_err_t err);

    /** Return true if the module is connected and responding. */
    bool isConnected();

private:
    HardwareSerial *_serial;
    uint32_t        _timeout_ms;
    uint32_t        _next_id;
    bool            _initialized;

    char _rx_buf[2048];
    char _tx_buf[1024];

    /* Internal helpers */
    ne_err_t  sendFrame(const char *json);
    ne_err_t  readLine(uint32_t timeout_ms);
    ne_err_t  executeCommand(const char *cmd, char *resp_buf, size_t resp_size);
    bool      jsonGetString(const char *json, const char *key,
                            char *out, size_t out_size);
    bool      jsonGetFloat(const char *json, const char *key, float &out);
    bool      jsonGetBool(const char *json, const char *key, bool &out);
    uint16_t  crc16(const uint8_t *data, size_t len);
};

#endif /* NEUROEDGE_ARDUINO_H */
