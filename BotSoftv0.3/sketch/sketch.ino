/*
 * FlyRobot MCU sketch for Arduino UNO Q.
 *
 * Provides Bridge RPC functions that the Python connectome calls:
 *   - read_imu: roll angle from Modulino Movement
 *   - read_thermal: 8x8 AMG8833 thermal frame (packed bytes)
 *   - read_limit: limit switch state
 *   - set_servo: PCA9685 servo angle
 *   - update_display: render face + telemetry on ST7789V LCD
 *
 * Safety: if no Bridge call arrives for 500ms, all servos neutral.
 */

#include <Arduino_RouterBridge.h>
#include <Wire.h>
#include <Arduino_Modulino.h>
#include <Adafruit_AMG8833.h>
#include <Adafruit_PWMServoDriver.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>

// ---------- pin config (adjust to your wiring) ----------
#define LIMIT_PIN       2
#define TFT_CS          10
#define TFT_DC          7
#define TFT_RST         8
#define TFT_BL          9

// ---------- objects ----------
ModulinoMovement movement;
Adafruit_AMG8833 thermal;
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);
Adafruit_ST7789 tft = Adafruit_ST7789(TFT_CS, TFT_DC, TFT_RST);

// ---------- watchdog ----------
unsigned long last_bridge_call = 0;
const unsigned long WATCHDOG_MS = 500;

// ---------- display state ----------
uint8_t last_face_id = 0;
uint8_t last_state_id = 0;
float last_j[4] = {90, 90, 90, 90};
float last_steer = 0, last_servo = 90;
int last_pokes = 0, last_trips = 0;

// ---------- faces (drawn as strings) ----------
const char* FACES[] = {
    "( -.- )", "( <.< )", "( >.> )", "( o.o )",
    "( O.O )", "( o.O )", "( ^.^ )", "( x.x )"
};
const char* STATES[] = {
    "IDLE", "STEERING", "TOUCH", "THERMAL", "TRIPPED"
};

// =============================================================
// Bridge RPC functions (registered with provide_safe)
// =============================================================

bool set_servo(int channel, float degrees) {
    last_bridge_call = millis();
    if (channel < 0 || channel > 15) return false;

    // PCA9685: 50 Hz, pulse range ~150-600 for 0-180 deg
    // Adjust SERVOMIN/SERVOMAX for your specific servos
    const int SERVOMIN = 150;
    const int SERVOMAX = 600;
    float clamped = constrain(degrees, 0.0, 180.0);
    int pulse = SERVOMIN + (int)((SERVOMAX - SERVOMIN) * (clamped / 180.0));
    pwm.setPWM(channel, 0, pulse);
    return true;
}

void read_imu() {
    // Returns via Bridge: [roll]
    last_bridge_call = millis();
    movement.update();
    float roll = movement.getRoll();
    // Return as a MsgPack array
    Bridge.call("return_imu", roll);  // This won't work -- see note below
}

// NOTE: Bridge RPC return values are sent as the function's return.
// For simple values, just return them directly.

float read_imu_value() {
    last_bridge_call = millis();
    movement.update();
    return movement.getRoll();
}

// For arrays, we return a vector which MsgPack serializes
std::vector<uint8_t> read_thermal() {
    last_bridge_call = millis();
    std::vector<uint8_t> packed(64);
    for (int i = 0; i < 64; i++) {
        float c = thermal.readPixel(i);
        // Pack: (C + 10) * 5, clamped to 0-255
        int v = (int)((c + 10.0) * 5.0);
        packed[i] = (uint8_t)constrain(v, 0, 255);
    }
    return packed;
}

bool read_limit() {
    last_bridge_call = millis();
    return digitalRead(LIMIT_PIN) == LOW;
}

bool update_display(uint8_t face_id, uint8_t state_id,
                    float j0, float j1, float j2, float j3,
                    float steer, float servo,
                    int pokes, int trips) {
    last_bridge_call = millis();

    // Only redraw if something changed meaningfully
    if (face_id == last_face_id &&
        state_id == last_state_id &&
        abs(j0 - last_j[0]) < 0.5 &&
        abs(j1 - last_j[1]) < 0.5 &&
        abs(j2 - last_j[2]) < 0.5 &&
        abs(j3 - last_j[3]) < 0.5) {
        return true;
    }

    last_face_id = face_id;
    last_state_id = state_id;
    last_j[0] = j0; last_j[1] = j1; last_j[2] = j2; last_j[3] = j3;
    last_steer = steer; last_servo = servo;
    last_pokes = pokes; last_trips = trips;

    render_display();
    return true;
}

// =============================================================
// Display rendering
// =============================================================

void render_display() {
    tft.fillScreen(ST77XX_BLACK);
    tft.setTextWrap(false);

    // Face (large, centered top)
    tft.setTextSize(4);
    tft.setTextColor(ST77XX_YELLOW);
    int face_w = strlen(FACES[last_face_id]) * 24;  // ~24px per char at size 4
    tft.setCursor((240 - face_w) / 2, 20);
    tft.print(FACES[last_face_id]);

    // State label
    tft.setTextSize(2);
    tft.setTextColor(ST77XX_CYAN);
    tft.setCursor(10, 80);
    tft.print("STATE: ");
    tft.print(STATES[last_state_id]);

    // Joint angles (4 bars)
    tft.setTextSize(1);
    tft.setTextColor(ST77XX_WHITE);
    const char* joint_names[] = {"Lsh", "Lel", "Rsh", "Rel"};
    float joints[] = {last_j[0], last_j[1], last_j[2], last_j[3]};
    for (int i = 0; i < 4; i++) {
        int y = 120 + i * 40;
        tft.setCursor(10, y);
        tft.print(joint_names[i]);
        tft.setCursor(60, y);
        tft.print(joints[i], 1);

        // Bar graph: 90 deg is center, +/- 90 range
        int bar_x = 120;
        int bar_w = (int)((joints[i] - 0) / 180.0 * 100);
        tft.drawRect(bar_x, y, 100, 12, ST77XX_WHITE);
        tft.fillRect(bar_x, y, bar_w, 12, ST77XX_GREEN);
    }

    // Telemetry bottom
    tft.setTextColor(ST77XX_MAGENTA);
    tft.setCursor(10, 290);
    tft.print("steer=");
    tft.print(last_steer, 2);
    tft.setCursor(140, 290);
    tft.print("sv=");
    tft.print(last_servo, 0);
    tft.setCursor(10, 305);
    tft.print("pokes=");
    tft.print(last_pokes);
    tft.setCursor(140, 305);
    tft.print("trips=");
    tft.print(last_trips);
}

// =============================================================
// setup / loop
// =============================================================

void setup() {
    Monitor.begin();
    Bridge.begin();

    // I2C
    Wire.begin();
    Wire1.begin();  // Qwiic port for Modulino

    // Modulino Movement
    Modulino.begin(Wire1);
    movement.begin();

    // AMG8833
    if (!thermal.begin(0x68)) {
        Monitor.println("AMG8833 not found");
    }

    // PCA9685
    pwm.begin();
    pwm.setOscillatorFrequency(27000000);
    pwm.setPWMFreq(50);  // standard servo frequency

    // Limit switch
    pinMode(LIMIT_PIN, INPUT_PULLUP);

    // LCD
    pinMode(TFT_BL, OUTPUT);
    digitalWrite(TFT_BL, HIGH);  // backlight on
    tft.init(240, 320);
    tft.setRotation(0);
    tft.fillScreen(ST77XX_BLACK);

    // Register RPC functions
    Bridge.provide_safe("read_imu", read_imu_value);
    Bridge.provide_safe("read_thermal", read_thermal);
    Bridge.provide_safe("read_limit", read_limit);
    Bridge.provide_safe("set_servo", set_servo);
    Bridge.provide_safe("update_display", update_display);

    last_bridge_call = millis();
    Monitor.println("FlyRobot MCU ready");
}

void loop() {
    // Watchdog: if Python hasn't called us in 500ms, neutral all servos
    if (millis() - last_bridge_call > WATCHDOG_MS) {
        for (int i = 0; i < 16; i++) {
            pwm.setPWM(i, 0, 375);  // ~90 deg neutral
        }
    }
    delay(10);
}
