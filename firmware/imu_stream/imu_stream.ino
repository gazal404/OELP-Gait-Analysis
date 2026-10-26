// imu_stream.ino - two MPU6050 (thigh + shank) on one ESP32, 100 Hz CSV stream.
//
//   USE_WIFI 1 : ESP32 makes its own WiFi network. Laptop joins it and reads TCP port 5000.
//                Use this for walking (nothing tied to the laptop).
//   USE_WIFI 0 : Same lines over USB serial at 921600 baud. Use this for bench tests.
//
// Wiring (both sensors share the same 4 wires):
//   VCC -> 3V3,  GND -> GND,  SDA -> GPIO21,  SCL -> GPIO22
//   Thigh sensor: AD0 left alone / to GND  -> address 0x68
//   Shank sensor: AD0 wired to its own VCC -> address 0x69
//
// One line per sample:
//   n,t_us, thigh ax,ay,az (g), gx,gy,gz (deg/s), shank ax,ay,az (g), gx,gy,gz (deg/s)
//   n    = sample counter (a jump in n means a dropped sample)
//   t_us = microseconds since the stream started
//
// LED (GPIO2): slow blink = waiting for the laptop, solid = streaming,
//              fast blink = a sensor was not found (see Serial Monitor at 115200).

#define USE_WIFI 0

#include <Wire.h>
#if USE_WIFI
#include <WiFi.h>
#endif

const int SDA_PIN = 21;
const int SCL_PIN = 22;
const int LED_PIN = 2;

const uint8_t ADDR_THIGH = 0x68;
const uint8_t ADDR_SHANK = 0x69;

const unsigned long PERIOD_US = 10000;   // 10,000 us = 100 Hz

// +-8 g and +-1000 deg/s: big enough that heel strikes and fast shank swings do not clip
const float ACC_LSB_PER_G    = 4096.0f;
const float GYRO_LSB_PER_DPS = 32.8f;

#if USE_WIFI
const char* AP_SSID = "gait-esp32";
const char* AP_PASS = "gait12345";       // at least 8 characters
const uint16_t PORT = 5000;
WiFiServer server(PORT);
WiFiClient client;
#endif

bool streaming = false;
unsigned long nextTick = 0;
unsigned long t0 = 0;
unsigned long sampleNo = 0;

bool writeReg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

bool initImu(uint8_t addr) {
  if (!writeReg(addr, 0x6B, 0x00)) return false;   // wake up (clear sleep bit)
  delay(50);
  writeReg(addr, 0x1A, 0x03);                      // digital low-pass ~44 Hz
  writeReg(addr, 0x1B, 0x10);                      // gyro  +-1000 deg/s
  writeReg(addr, 0x1C, 0x10);                      // accel +-8 g
  return true;
}

// out = ax, ay, az (g), gx, gy, gz (deg/s). Returns false if the read failed.
bool readImu(uint8_t addr, float out[6]) {
  Wire.beginTransmission(addr);
  Wire.write(0x3B);                                // first accel register
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(addr, (uint8_t)14) != 14) return false;

  int16_t raw[7];                                  // ax ay az temp gx gy gz
  for (int i = 0; i < 7; i++) {
    uint8_t hi = Wire.read();                      // read high byte first, on its own line
    uint8_t lo = Wire.read();
    raw[i] = (int16_t)((hi << 8) | lo);
  }
  out[0] = raw[0] / ACC_LSB_PER_G;
  out[1] = raw[1] / ACC_LSB_PER_G;
  out[2] = raw[2] / ACC_LSB_PER_G;
  out[3] = raw[4] / GYRO_LSB_PER_DPS;              // raw[3] is temperature, skipped
  out[4] = raw[5] / GYRO_LSB_PER_DPS;
  out[5] = raw[6] / GYRO_LSB_PER_DPS;
  return true;
}

void startStream() {
  streaming = true;
  t0 = micros();
  nextTick = t0;
  sampleNo = 0;
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
#if USE_WIFI
  Serial.begin(115200);
#else
  Serial.begin(921600);
#endif

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);       // 100 kHz is safer over the long shank wire
  Wire.setTimeOut(50);         // a loose wire must not freeze the board

  bool okThigh = initImu(ADDR_THIGH);
  bool okShank = initImu(ADDR_SHANK);
  if (!okThigh || !okShank) {
    while (true) {
      Serial.print("ERROR: sensor not found:");
      if (!okThigh) Serial.print(" thigh(0x68)");
      if (!okShank) Serial.print(" shank(0x69)");
      Serial.println();
      for (int i = 0; i < 5; i++) {
        digitalWrite(LED_PIN, HIGH); delay(100);
        digitalWrite(LED_PIN, LOW);  delay(100);
      }
    }
  }

#if USE_WIFI
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
  WiFi.setSleep(false);
  server.begin();
  Serial.print("WiFi network: ");
  Serial.println(AP_SSID);
  Serial.print("IP address: ");
  Serial.println(WiFi.softAPIP());
#else
  startStream();
#endif
}

void loop() {
#if USE_WIFI
  if (!streaming) {
    digitalWrite(LED_PIN, (millis() / 500) % 2);   // slow blink while waiting
    WiFiClient c = server.available();
    if (c) {
      client = c;
      client.setNoDelay(true);
      startStream();
    }
    return;
  }
  if (!client.connected()) {                        // laptop closed the connection
    client.stop();
    streaming = false;
    return;
  }
#endif
  digitalWrite(LED_PIN, HIGH);

  while ((long)(micros() - nextTick) < 0) { yield(); }
  nextTick += PERIOD_US;

  unsigned long n = sampleNo++;
  unsigned long tUs = micros() - t0;

  float a[6], b[6];
  if (!readImu(ADDR_THIGH, a) || !readImu(ADDR_SHANK, b)) return;   // a skipped sample shows up as a gap in n

  char buf[200];
  int len = snprintf(buf, sizeof(buf),
                     "%lu,%lu,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f\n",
                     n, tUs,
                     a[0], a[1], a[2], a[3], a[4], a[5],
                     b[0], b[1], b[2], b[3], b[4], b[5]);
#if USE_WIFI
  client.write((const uint8_t*)buf, len);
#else
  Serial.write((const uint8_t*)buf, len);
#endif
}
