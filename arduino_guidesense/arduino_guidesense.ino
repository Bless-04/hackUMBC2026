/*
 * arduino_guidesense.ino — Arduino Breadboard Controller
 * ======================================================
 * Built for Freshman #1 (Computer Engineering)
 *
 * Responsibilities on Breadboard:
 *   1. Hardware Status Indicators (LEDs on Breadboard):
 *      - Pin 2: Green LED  -> SILENT state (Path clear)
 *      - Pin 3: Yellow LED -> INFORMATIVE state (Object detected at mid-range)
 *      - Pin 4: Red LED    -> URGENT state (Hazard detected!)
 *
 *   2. Optional Manual Proximity / Sensitivity Dial (Potentiometer):
 *      - Pin A0: Potentiometer center wiper pin
 *      - Sends distance readings "D:<cm>\n" over USB serial
 *      - Allows live manual demonstration of distance zones to judges!
 *
 * Wiring (Breadboard to Arduino):
 *   - Green LED  -> Pin 2 (through 220-330 ohm resistor to GND)
 *   - Yellow LED -> Pin 3 (through 220-330 ohm resistor to GND)
 *   - Red LED    -> Pin 4 (through 220-330 ohm resistor to GND)
 *   - (Optional) Potentiometer -> Left: 5V, Right: GND, Middle: A0
 */

const int PIN_LED_GREEN  = 2;
const int PIN_LED_YELLOW = 3;
const int PIN_LED_RED    = 4;
const int PIN_POT        = A0;

unsigned long lastSendTime = 0;
const unsigned long SEND_INTERVAL_MS = 100; // 10 Hz report rate

void setup() {
  Serial.begin(9600);

  pinMode(PIN_LED_GREEN, OUTPUT);
  pinMode(PIN_LED_YELLOW, OUTPUT);
  pinMode(PIN_LED_RED, OUTPUT);

  // Default state: SILENT (Green ON)
  setStateSilent();
}

void loop() {
  // 1. Read commands from Raspberry Pi
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "STATE:SILENT" || cmd == "S") {
      setStateSilent();
    } else if (cmd == "STATE:INFORMATIVE" || cmd == "I") {
      setStateInformative();
    } else if (cmd == "STATE:URGENT" || cmd == "U" || cmd == "BUZZ_ON") {
      setStateUrgent();
    }
  }

  // 2. Read potentiometer and transmit simulated distance (if wired)
  unsigned long now = millis();
  if (now - lastSendTime >= SEND_INTERVAL_MS) {
    lastSendTime = now;
    int potValue = analogRead(PIN_POT); // 0 to 1023
    // Map 0-1023 to 30 cm - 350 cm
    int cm = map(potValue, 0, 1023, 30, 350);

    // Send formatted line expected by serial_reader.py:
    Serial.print("D:");
    Serial.println(cm);
  }
}

void setStateSilent() {
  digitalWrite(PIN_LED_GREEN, HIGH);
  digitalWrite(PIN_LED_YELLOW, LOW);
  digitalWrite(PIN_LED_RED, LOW);
}

void setStateInformative() {
  digitalWrite(PIN_LED_GREEN, LOW);
  digitalWrite(PIN_LED_YELLOW, HIGH);
  digitalWrite(PIN_LED_RED, LOW);
}

void setStateUrgent() {
  digitalWrite(PIN_LED_GREEN, LOW);
  digitalWrite(PIN_LED_YELLOW, LOW);
  digitalWrite(PIN_LED_RED, HIGH);
}
