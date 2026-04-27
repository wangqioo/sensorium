/**
 * FSR 触觉阵列 Arduino 固件
 *
 * 读取多路 FSR 传感器的 ADC 值，打包成帧通过串口发给 Jetson。
 *
 * 硬件接线：
 *   FSR 传感器一端接 5V，另一端接 ADC 引脚并通过分压电阻（10kΩ）下拉到 GND。
 *   压力越大 → FSR 阻值越小 → ADC 电压越高 → ADC 读值越大。
 *
 * 帧格式（与 sensorium/drivers/tactile.py 约定一致）：
 *   [0xAA][len_hi][len_lo][sensor_0_hi][sensor_0_lo]...[checksum]
 *   len = N_SENSORS * 2（字节数）
 *   sensor_x：10bit ADC 值，big-endian，范围 0-1023
 *   checksum：len 之后所有字节的异或
 *
 * 支持的 Arduino 型号：Mega 2560（16个模拟口）/ Due / Zero（多路复用扩展）
 *
 * 波特率：921600（与 TactileDriver 一致）
 * 采样率：100Hz（每 10ms 发一帧）
 */

// ——— 配置 ———
#define N_SENSORS     16      // 实际连接的 FSR 数量（最多 Arduino 模拟口数量）
#define SAMPLE_HZ     100     // 采样频率（Hz）
#define BAUD_RATE     921600
#define FRAME_MAGIC   0xAA

// 模拟引脚映射（按身体区域排列）
// 修改此数组以匹配实际接线
const uint8_t SENSOR_PINS[N_SENSORS] = {
  A0,  A1,  A2,  A3,   // 头部（4个）
  A4,  A5,  A6,  A7,   // 背部上（4个）
  A8,  A9,  A10, A11,  // 背部下（4个）
  A12, A13, A14, A15,  // 腹部（4个）
};

// ——— 全局变量 ———
const uint16_t SAMPLE_INTERVAL_MS = 1000 / SAMPLE_HZ;
uint32_t last_sample_ms = 0;
uint16_t readings[N_SENSORS];

// ——— 发送帧 ———
void send_frame() {
  uint8_t len_hi = 0;
  uint8_t len_lo = (uint8_t)(N_SENSORS * 2);
  uint8_t checksum = len_lo;  // 从 len 字节开始异或

  // 帧头
  Serial.write(FRAME_MAGIC);
  Serial.write(len_hi);
  Serial.write(len_lo);

  // 传感器数据（big-endian）
  for (int i = 0; i < N_SENSORS; i++) {
    uint8_t hi = (readings[i] >> 8) & 0x03;  // 高 2bit（10bit ADC）
    uint8_t lo = readings[i] & 0xFF;          // 低 8bit
    Serial.write(hi);
    Serial.write(lo);
    checksum ^= hi;
    checksum ^= lo;
  }

  // 校验和
  Serial.write(checksum);
}

// ——— 初始化 ———
void setup() {
  Serial.begin(BAUD_RATE);

  // 配置模拟引脚为输入
  for (int i = 0; i < N_SENSORS; i++) {
    pinMode(SENSOR_PINS[i], INPUT);
  }

  // 等待串口稳定
  delay(100);
}

// ——— 主循环 ———
void loop() {
  uint32_t now = millis();

  if (now - last_sample_ms >= SAMPLE_INTERVAL_MS) {
    last_sample_ms = now;

    // 读取所有传感器
    for (int i = 0; i < N_SENSORS; i++) {
      readings[i] = analogRead(SENSOR_PINS[i]);
    }

    // 发送帧
    send_frame();
  }

  // 处理来自主机的命令（预留，目前只接收不发送命令）
  while (Serial.available()) {
    Serial.read();  // 丢弃未知命令
  }
}
