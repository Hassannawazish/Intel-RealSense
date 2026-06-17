// Servo-based door actuator controlled over USB serial.
#include <Servo.h>

const int SERVO_PIN = 9;
const int LOCKED_ANGLE = 0;
const int UNLOCKED_ANGLE = 90;

Servo doorServo;
String inputBuffer = "";

void setup() {
  doorServo.attach(SERVO_PIN);
  doorServo.write(LOCKED_ANGLE);
  Serial.begin(9600);
}

void applyCommand(const String& command) {
  if (command == "ON") {
    doorServo.write(UNLOCKED_ANGLE);
    Serial.println("SERVO_UNLOCKED");
    return;
  }

  if (command == "OFF") {
    doorServo.write(LOCKED_ANGLE);
    Serial.println("SERVO_LOCKED");
  }
}

void loop() {
  while (Serial.available() > 0) {
    char incoming = (char)Serial.read();
    if (incoming == '\n' || incoming == '\r') {
      if (inputBuffer.length() > 0) {
        inputBuffer.trim();
        applyCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += incoming;
    }
  }
}
