#include <Wire.h>
#include "NineAxesMotion.h"

NineAxesMotion sensor;
#define SENSOR_ADDRESS 0x28 // I2C address of BNO055
#define INT_PIN 2

int ledPin = 13; // Status LED pin
int intPin = 2;  // Interrupt pin
int serialBaud = 9600;
int motionThreshold_yes = 2;
int duration_yes = 10; // 80ms @62.5Hz
int motionThreshold_no = 5;
int duration_no = 1; // 8ms @62.5Hz
byte lastError = 0;
bool interruptsEnabled = true;
bool humanReadable = true;
bool inMotion = false;
bool interrupted = false;

// Defining the serial protocol below
#define SERIAL_CONNCHECK 0x01
#define SERIAL_STATUS 0x02
#define SERIAL_OK 0x03
#define SERIAL_FAULT 0x04
#define SERIAL_ENABLE_INTERRUPTS 0x05
#define SERIAL_DISABLE_INTERRUPTS 0x06
#define SERIAL_INTERRUPTS_ENABLED 0x07
#define SERIAL_READ_QUATERNION 0x0e
#define SERIAL_READ_EULER 0x0f
#define SERIAL_RESET 0x10
#define SERIAL_MOTION_INTERRUPT 0x11
#define SERIAL_NO_MOTION_INTERRUPT 0x12
#define SERIAL_TOGGLE_HUMAN_READABLE 0x13
#define SERIAL_DATA_BEGIN 0xae
#define SERIAL_DATA_END 0xaf
#define SERIAL_DATA_TRUE 0x01
#define SERIAL_DATA_FALSE 0x00

void setup() {
  // put your setup code here, to run once:
  Serial.begin(serialBaud);
  I2C.begin();

  Serial.println("Please wait, initialization in progress.");
  sensor.initSensor();
  sensor.setOperationMode(OPERATION_MODE_IMUPLUS);
  sensor.setUpdateMode(AUTO);

  attachInterrupt(digitalPinToInterrupt(INT_PIN), motionISR, RISING);
  sensor.resetInterrupt();
  sensor.accelInterrupts(ENABLE, ENABLE, ENABLE);
  sensor.enableAnyMotion(motionThreshold_yes, duration_yes);
  Serial.println("Initialization complete.");
}

void printHex(byte data) {
  char buf[3];
  snprintf(buf, 3, "%02x", data);
  Serial.print(buf);
}

// Print out numbers etc. as per memory layout (little-endian)
void printHexBuf(uint8_t* buf, int len) {
  for (int i = 0; i < len; ++i, ++buf) {
    printHex(*buf);
  }
}

void sendData(uint8_t* buf, int len) {
  Serial.write(SERIAL_DATA_BEGIN);
  printHexBuf(buf, len);
  Serial.write(SERIAL_DATA_END);
}

void setInterrupts(bool state) {
  if (state) {
    interruptsEnabled = true;
    enableInterrupts();
  } else {
    disableInterrupts();
    interruptsEnabled = false;
  }
}

void processSerial(byte instruction) {
  switch(instruction) {
    case SERIAL_CONNCHECK:
    Serial.write(SERIAL_OK);
    Serial.println("> RESP OK TO CONNCHECK");
    break;
    case SERIAL_STATUS:
    if (lastError == 0) {
      Serial.write(SERIAL_OK);
      if (humanReadable)
        Serial.println("> RESP OK TO STATUS");
    } else {
      Serial.write(SERIAL_FAULT);
      sendData(&lastError, 1);
      if (humanReadable) {
        Serial.print("> RESP FAULT TO STATUS");
        Serial.print(lastError, DEC);
        Serial.println("");
      }
    }
    break;
    case SERIAL_OK:
    if (humanReadable) Serial.println("> RECV OK");
    break;
    case SERIAL_FAULT:
    if (humanReadable) Serial.println("> RECV FAULT");
    break;
    case SERIAL_ENABLE_INTERRUPTS:
    setInterrupts(true);
    Serial.write(SERIAL_OK);
    if (humanReadable)
      Serial.println("> RESP OK TO ENABLE INTERRUPTS");
    break;
    case SERIAL_DISABLE_INTERRUPTS:
    setInterrupts(false);
    Serial.write(SERIAL_OK);
    if (humanReadable)
      Serial.println("> RESP OK TO ENABLE INTERRUPTS");
    break;
    case SERIAL_INTERRUPTS_ENABLED:
    if (interruptsEnabled == true) {
      sendData(SERIAL_DATA_TRUE, 1);
    } else {
      sendData(SERIAL_DATA_FALSE, 1);
    }
    if (humanReadable) {
      Serial.print("> RESP INTERRUPTS ENABLED = ");
      if (interruptsEnabled) {
        Serial.println("TRUE");
      } else {
        Serial.println("FALSE");
      }
    }
    break;
    case SERIAL_READ_QUATERNION: {
      int16_t buf[4];
      sensor.readQuaternion(buf[0], buf[1], buf[2], buf[3]);
      sendData(reinterpret_cast<uint8_t*>(buf), 4 * sizeof(int16_t));
      if (humanReadable) {
      Serial.print("> RESP QUATERNION [");
      Serial.print(buf[0], DEC);
      Serial.print(", ");
      Serial.print(buf[1], DEC);
      Serial.print(", ");
      Serial.print(buf[2], DEC);
      Serial.print(", ");
      Serial.print(buf[3], DEC);
      Serial.println("]");
      }
      break;
    }
    case SERIAL_READ_EULER: {
      float buf[3];
      buf[0] = sensor.readEulerHeading();
      buf[1] = sensor.readEulerRoll();
      buf[2] = sensor.readEulerPitch();
      sendData(reinterpret_cast<uint8_t*>(buf), 3 * sizeof(float));
      if (humanReadable) {
      Serial.print("> RESP EULER HRP [");
      Serial.print(buf[0]);
      Serial.print(", ");
      Serial.print(buf[1]);
      Serial.print(", ");
      Serial.print(buf[2]);
      Serial.println("]");
      }
      break;
    }
    case SERIAL_RESET:
    sensor.resetSensor(SENSOR_ADDRESS);
    Serial.write(SERIAL_OK);
    if (humanReadable)
      Serial.println("> RESP OK TO RESET");
    case SERIAL_TOGGLE_HUMAN_READABLE:
    humanReadable = (! humanReadable);
    Serial.write(SERIAL_OK);
    if (humanReadable)
      Serial.println("> RESP OK TO TOGGLE HUMAN READABLE");
    break;
    default:
    if (humanReadable) {
      Serial.print("> IGNORED BYTE ");
      Serial.print(instruction, DEC);
    }
  }
}

void serialEvent() {
  while (Serial.available() > 0) {
    byte instruction = Serial.read();
    processSerial(instruction);
  }
}

void motionISR() {
  interrupted = true;
}

void disableInterrupts() {
  sensor.resetInterrupt();
  sensor.disableSlowNoMotion();
  sensor.disableAnyMotion();
  inMotion = false;
  interrupted = false;
}

void enableInterrupts() {
  sensor.resetInterrupt();
  sensor.disableSlowNoMotion();
  sensor.enableAnyMotion(motionThreshold_yes, duration_yes);
  inMotion = false;
  interrupted = false;
}

void loop() {
  if (interrupted) {
    interrupted = false;
    if (!inMotion) {
      Serial.write(SERIAL_MOTION_INTERRUPT);
      if (humanReadable) {
        Serial.println("> ENCOUNTERED MOTION INTERRUPT");
      }
      sensor.resetInterrupt();
      sensor.disableAnyMotion();
      sensor.enableSlowNoMotion(motionThreshold_no, duration_no, NO_MOTION);
      inMotion = true;
    } else {
      Serial.write(SERIAL_NO_MOTION_INTERRUPT);
      if (humanReadable) {
        Serial.println("> ENCOUNTERED NO-MOTION INTERRUPT");
      }
      sensor.resetInterrupt();
      sensor.disableSlowNoMotion();
      sensor.enableAnyMotion(motionThreshold_yes, duration_yes);
      inMotion = false;
    }
  }
}
