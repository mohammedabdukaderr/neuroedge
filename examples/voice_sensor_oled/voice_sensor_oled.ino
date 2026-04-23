/**
 * NeuroEdge Demo — Offline Voice-Controlled Sensor Display
 *
 * Hardware:
 *   - Arduino Uno R4 WiFi (or ESP32, STM32, nRF52840)
 *   - NeuroEdge module on Serial1 (TX=D1, RX=D0)
 *   - I2S MEMS microphone (e.g. INMP441) on pins D4/D5/D6
 *   - SSD1306 128×64 OLED on I2C (SDA=A4, SCL=A5)
 *   - Any analog sensor on A0 (temperature, pressure, etc.)
 *
 * What it does:
 *   - Listens continuously for keywords: "read", "status", "reset"
 *   - On "read":   reads sensor, asks NeuroEdge if value is abnormal
 *   - On "status": shows last reading on OLED
 *   - On "reset":  clears history
 *   - All processing is 100% offline — no WiFi, no cloud
 *
 * Install libraries (Arduino IDE):
 *   Tools → Manage Libraries →
 *     Adafruit SSD1306 + Adafruit GFX
 *     driver_i2s (or use PDM.h on Nano 33 BLE)
 *
 * Wiring:
 *   NeuroEdge TX → Arduino Serial1 RX (D0)
 *   NeuroEdge RX → Arduino Serial1 TX (D1)
 *   NeuroEdge GND → Arduino GND
 *   NeuroEdge 3.3V → Arduino 3.3V
 */

#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <NeuroEdge.h>

/* -------------------------------------------------------------------------
 * Configuration */

#define OLED_WIDTH    128
#define OLED_HEIGHT    64
#define OLED_RESET     -1
#define OLED_I2C_ADDR 0x3C

#define SENSOR_PIN    A0
#define MIC_SAMPLE_RATE  16000
#define MIC_BUF_SAMPLES  16000   /* 1 second of audio */

/* -------------------------------------------------------------------------
 * Globals */

Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire, OLED_RESET);
NeuroEdge        ne;

int16_t  mic_buf[MIC_BUF_SAMPLES];
float    last_sensor_value = 0.0f;
char     last_analysis[128] = "–";
bool     alert_active       = false;

/* -------------------------------------------------------------------------
 * Microphone capture (PDM — works on Nano 33 BLE, nRF52840)
 * For I2S MEMS: replace with I2S.read() calls */

#if defined(ARDUINO_ARCH_NRF52840) || defined(ARDUINO_ARDUINO_NANO33BLE)
#  include <PDM.h>
   static volatile bool mic_done = false;
   static void onPDMdata() {
       int available = PDM.available();
       if (available > 0)
           PDM.read(mic_buf, min((int)sizeof(mic_buf), available));
       mic_done = true;
   }
   static bool capture_audio() {
       mic_done = false;
       PDM.begin(1, MIC_SAMPLE_RATE);
       PDM.onReceive(onPDMdata);
       unsigned long t0 = millis();
       while (!mic_done && millis() - t0 < 2000) delay(1);
       PDM.end();
       return mic_done;
   }
#else
   /* Stub for boards without PDM/I2S — fills with silence for testing */
   static bool capture_audio() {
       memset(mic_buf, 0, sizeof(mic_buf));
       delay(200);   /* Simulate capture time */
       return true;
   }
#endif

/* -------------------------------------------------------------------------
 * OLED helpers */

static void oled_clear_show(const char *line1,
                             const char *line2 = nullptr,
                             const char *line3 = nullptr)
{
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println(line1);
    if (line2) { display.setCursor(0, 16); display.println(line2); }
    if (line3) { display.setCursor(0, 32); display.println(line3); }
    display.display();
}

static void oled_status(float sensor_val, bool is_alert,
                         const char *analysis)
{
    char val_str[32], alert_str[32];
    snprintf(val_str,   sizeof(val_str),   "Sensor: %.1f", sensor_val);
    snprintf(alert_str, sizeof(alert_str), "%s", is_alert ? "!! ALERT !!" : "OK");

    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);

    /* Title */
    display.setCursor(0, 0);
    display.print("NeuroEdge");

    /* Sensor value */
    display.setCursor(0, 12);
    display.print(val_str);

    /* Status */
    display.setTextSize(is_alert ? 2 : 1);
    display.setCursor(0, 28);
    display.print(alert_str);

    /* Analysis (small text, wrap) */
    display.setTextSize(1);
    display.setCursor(0, 50);
    char trunc[22];
    strncpy(trunc, analysis, 21);
    trunc[21] = '\0';
    display.print(trunc);

    display.display();
}

/* -------------------------------------------------------------------------
 * Sensor reading */

static float read_sensor()
{
    /* Simple analog sensor — scale 0-1023 → 0.0–100.0
     * For a real temperature sensor: apply calibration here */
    int raw = analogRead(SENSOR_PIN);
    return (float)raw / 1023.0f * 100.0f;
}

/* -------------------------------------------------------------------------
 * Core logic */

static void do_read_and_analyze()
{
    oled_clear_show("Reading sensor...");

    last_sensor_value = read_sensor();

    /* Ask NeuroEdge to analyze the value */
    char prompt[128], response[128];
    snprintf(prompt, sizeof(prompt),
             "Sensor reading is %.1f (scale 0-100). Is this abnormal? "
             "Answer in 5 words or fewer.",
             last_sensor_value);

    ne_err_t err = ne.ask(prompt, response, sizeof(response));

    if (err == NE_OK) {
        strncpy(last_analysis, response, sizeof(last_analysis) - 1);
        last_analysis[sizeof(last_analysis) - 1] = '\0';

        /* Simple heuristic: if response contains alarm words → alert */
        alert_active = (strstr(response, "abnormal") != nullptr ||
                        strstr(response, "high")     != nullptr ||
                        strstr(response, "low")      != nullptr ||
                        strstr(response, "critical")  != nullptr);
    } else {
        snprintf(last_analysis, sizeof(last_analysis),
                 "err:%s", NeuroEdge::errorString(err));
        alert_active = false;
    }

    oled_status(last_sensor_value, alert_active, last_analysis);
    Serial.printf("[READ] sensor=%.1f alert=%d analysis=%s\n",
                  last_sensor_value, alert_active, last_analysis);
}

static void do_status()
{
    oled_status(last_sensor_value, alert_active, last_analysis);
    Serial.println("[STATUS] Displayed last reading");
}

static void do_reset()
{
    last_sensor_value = 0.0f;
    strncpy(last_analysis, "–", sizeof(last_analysis));
    alert_active = false;
    oled_clear_show("Reset", "History cleared");
    Serial.println("[RESET] Cleared");
}

/* -------------------------------------------------------------------------
 * Keyword → action */

static void dispatch_keyword(const char *keyword)
{
    Serial.printf("[KWS] Detected: %s\n", keyword);

    if      (strcmp(keyword, "read")   == 0) do_read_and_analyze();
    else if (strcmp(keyword, "status") == 0) do_status();
    else if (strcmp(keyword, "stop")   == 0) do_reset();
    else if (strcmp(keyword, "go")     == 0) do_read_and_analyze();
}

/* -------------------------------------------------------------------------
 * Arduino setup */

void setup()
{
    Serial.begin(115200);
    while (!Serial && millis() < 2000);
    Serial.println("NeuroEdge Voice Sensor Demo");

    /* OLED init */
    Wire.begin();
    if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_I2C_ADDR)) {
        Serial.println("ERROR: SSD1306 not found");
        while (true) delay(1000);
    }
    oled_clear_show("NeuroEdge", "Connecting...");

    /* NeuroEdge init */
    Serial1.begin(115200);
    ne_err_t err = ne.begin(Serial1, 8000);
    if (err != NE_OK) {
        char msg[64];
        snprintf(msg, sizeof(msg), "Module error: %s", NeuroEdge::errorString(err));
        oled_clear_show("ERROR", msg);
        Serial.printf("NeuroEdge init failed: %s\n", NeuroEdge::errorString(err));
        while (true) delay(1000);
    }

    /* Load KWS model */
    err = ne.setModel("dscnn-kws");
    if (err != NE_OK) {
        Serial.printf("WARNING: setModel failed (%s) — using default\n",
                      NeuroEdge::errorString(err));
    }

    oled_clear_show("Ready!", "Say: read/status", "       /stop/go");
    Serial.println("Ready. Say a keyword.");
}

/* -------------------------------------------------------------------------
 * Arduino loop */

void loop()
{
    /* Capture 1 second of audio */
    if (!capture_audio()) {
        Serial.println("Mic capture failed");
        delay(100);
        return;
    }

    /* Run keyword detection */
    NeKeywordResult kws;
    ne_err_t err = ne.detectKeyword(mic_buf, MIC_BUF_SAMPLES, kws);

    if (err != NE_OK) {
        Serial.printf("KWS error: %s\n", NeuroEdge::errorString(err));
        delay(100);
        return;
    }

    if (kws.detected) {
        dispatch_keyword(kws.keyword);
        /* Brief pause after detection to avoid double-trigger */
        delay(500);
    }
    /* No detected keyword: loop immediately (continuous listening) */
}
