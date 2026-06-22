/**
 * amg8833_udp_stream.ino
 * =========================================================
 * ESP32 + AMG8833 UDP IR Streamer
 *
 * Sends IR frames using the SAME protocol expected by your
 * Python ws_bridge.py backend.
 *
 * Packet format (145 bytes):
 *
 *   [4]  MAGIC      = "ES32"
 *   [1]  TYPE       = 0x01 (IR)
 *   [4]  SEQ
 *   [8]  TIMESTAMP (microseconds)
 *   [128] 64 x int16_t temperatures (*100)
 *
 * Total = 145 bytes
 *
 * Compatible with:
 *   parse_ir() inside ws_bridge.py
 *
 * Wiring:
 *   AMG8833 SDA -> GPIO15
 *   AMG8833 SCL -> GPIO14
 *   VCC         -> 3.3V
 *   GND         -> GND
 *   AD0         -> GND  (0x69)
 */

 #include <WiFi.h>
 #include <WiFiUdp.h>
 #include <Wire.h>
 #include <Adafruit_AMG88xx.h>
 #include "esp_timer.h"
 
 // ─────────────────────────────────────────────
 // CONFIG
 // ─────────────────────────────────────────────
 const char* WIFI_SSID     = "LairofKwtaghn";
 const char* WIFI_PASSWORD = "kmbs6485";
 
 // Your backend PC IP
 const char* UDP_DEST_IP = "172.19.134.209";
 const uint16_t UDP_PORT = 5005;
 
 // I2C
 const int PIN_SDA = 15;
 const int PIN_SCL = 14;
 
 // AMG8833 address
 const uint8_t AMG_ADDR = 0x69;
 
 // Protocol constants
 const uint32_t MAGIC  = 0x45533332; // "ES32"
 const uint8_t  TYPE_IR = 0x01;
 
 // ─────────────────────────────────────────────
 // GLOBALS
 // ─────────────────────────────────────────────
 Adafruit_AMG88xx amg;
 WiFiUDP udp;
 
 static float pixelBufA[64];
 static float pixelBufB[64];
 
 static float* writeBuf = pixelBufA;
 static float* readBuf  = pixelBufB;
 
 static volatile bool newFrame = false;
 static uint32_t frameCounter  = 0;
 
 SemaphoreHandle_t bufMutex;
 
 // ─────────────────────────────────────────────
 // UDP TASK (Core 0)
 // ─────────────────────────────────────────────
 void udpTask(void* pvParameters) {
 
   const size_t PKT_SIZE = 145;
 
   uint8_t pkt[PKT_SIZE];
 
   for (;;) {
 
     if (!newFrame) {
       vTaskDelay(1);
       continue;
     }
 
     float localPixels[64];
     uint32_t localFrame;
 
     // Copy shared frame safely
     if (xSemaphoreTake(bufMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
 
       memcpy(localPixels, readBuf, sizeof(localPixels));
 
       localFrame = frameCounter;
 
       newFrame = false;
 
       xSemaphoreGive(bufMutex);
 
     } else {
       continue;
     }
 
     // ─────────────────────────────────────────
     // Build packet
     // ─────────────────────────────────────────
     uint8_t* p = pkt;
 
     // MAGIC (big endian)
     uint32_t magic_be = htonl(MAGIC);
     memcpy(p, &magic_be, 4);
     p += 4;
 
     // TYPE
     *p++ = TYPE_IR;
 
     // SEQ
     uint32_t seq_be = htonl(localFrame);
     memcpy(p, &seq_be, 4);
     p += 4;
 
     // TIMESTAMP (microseconds)
     uint64_t ts = esp_timer_get_time();
 
     uint64_t ts_be =
       ((uint64_t)htonl((uint32_t)(ts & 0xFFFFFFFF)) << 32) |
       htonl((uint32_t)(ts >> 32));
 
     memcpy(p, &ts_be, 8);
     p += 8;
 
     // 64 temperatures as int16_t *100
     for (int i = 0; i < 64; i++) {
 
       int16_t temp = (int16_t)(localPixels[i] * 100.0f);
 
       int16_t temp_be = htons(temp);
 
       memcpy(p, &temp_be, 2);
 
       p += 2;
     }
 
     // ─────────────────────────────────────────
     // Send packet
     // ─────────────────────────────────────────
     udp.beginPacket(UDP_DEST_IP, UDP_PORT);
     udp.write(pkt, PKT_SIZE);
     udp.endPacket();
   }
 }
 
 // ─────────────────────────────────────────────
 // SETUP
 // ─────────────────────────────────────────────
 void setup() {
 
   Serial.begin(115200);
 
   delay(300);
 
   Serial.println();
   Serial.println("[AMG8833] Starting IR streamer...");
 
   // I2C
   Wire.begin(PIN_SDA, PIN_SCL);
   Wire.setClock(400000);
 
   // Sensor init
   if (!amg.begin(AMG_ADDR)) {
 
     Serial.println("[ERROR] AMG8833 not found!");
 
     while (true) {
       delay(1000);
     }
   }
 
   Serial.println("[AMG8833] Sensor OK");
 
   // WiFi
   WiFi.mode(WIFI_STA);
 
   Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
 
   WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
 
   while (WiFi.status() != WL_CONNECTED) {
 
     delay(300);
     Serial.print(".");
   }
 
   Serial.println();
 
   Serial.printf(
     "[WiFi] Connected | ESP32 IP: %s\n",
     WiFi.localIP().toString().c_str()
   );
 
   // UDP
   udp.begin(UDP_PORT);
 
   // Mutex
   bufMutex = xSemaphoreCreateMutex();
 
   configASSERT(bufMutex);
 
   // UDP sender task on Core 0
   xTaskCreatePinnedToCore(
     udpTask,
     "udpTask",
     4096,
     NULL,
     2,
     NULL,
     0
   );
 
   Serial.printf(
     "[UDP] Streaming IR to %s:%d\n",
     UDP_DEST_IP,
     UDP_PORT
   );
 }
 
 // ─────────────────────────────────────────────
 // SENSOR LOOP (Core 1)
 // ─────────────────────────────────────────────
 void loop() {
 
   float tempBuf[64];
 
   // Read sensor
   amg.readPixels(tempBuf);
 
   // Swap buffers safely
   if (xSemaphoreTake(bufMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
 
     memcpy(writeBuf, tempBuf, sizeof(tempBuf));
 
     float* tmp = writeBuf;
     writeBuf   = readBuf;
     readBuf    = tmp;
 
     frameCounter++;
 
     newFrame = true;
 
     xSemaphoreGive(bufMutex);
   }
 
   // AMG8833 max practical rate ≈ 10 FPS
   vTaskDelay(pdMS_TO_TICKS(80));
 }