# main.py — v0.3, sensor-driven live loop for Arduino UNO Q
"""
FlyRobot live loop.

No synthetic stimuli. The connectome runs continuously; its inputs are
whatever the physical sensors report:
  - IMU roll      -> haltere drive
  - limit switch  -> tactile group (onset pulse)
  - thermal frame -> display only (does not drive the connectome)

Servos on channels 0-4, LCD updated at 10 Hz.

Usage:
    python main.py
"""

import sys
import time
import numpy as np
from pathlib import Path

from hardware.hardware import HardwareSensors
from brain.connectome import ConnectomeBrain
from control.fly_body import FlyBody

# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
TICK_HZ = 100
DT = 1.0 / TICK_HZ
WARMUP_TICKS = 1000
BASELINE_TICKS = 200

DN_TYPES_PER_JOINT = [
    ["DNg36_a", "DNg08",   "DNge138"],   # left shoulder
    ["DNpe004", "DNpe008", "DNge033"],   # left elbow
    ["DNge138", "DNg41",   "DNpe032"],   # right shoulder
    ["DNpe032", "DNg11",   "DNge127"],   # right elbow
]

HALTERE_BASE = 0.175
HALTERE_SPAN = 0.125
HALTERE_CLAMP = 0.15
HALTERE_TOTAL = HALTERE_BASE * 204   # approximate; overwritten at runtime

K_SERVO = 150.0
SERVO_CLAMP = 60.0

DISPLAY_HZ = 10.0
DISPLAY_DT = 1.0 / DISPLAY_HZ

TOUCH_ONSET = 1.0
TOUCH_HELD = 0.4
TOUCH_PULSE_S = 0.2   # how long the onset stays high


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def imu_to_drive(roll_deg):
    d = max(-1.0, min(1.0, roll_deg / 45.0)) * HALTERE_SPAN
    dl = max(HALTERE_BASE - HALTERE_CLAMP,
             min(HALTERE_BASE + HALTERE_CLAMP, HALTERE_BASE - d))
    dr = max(HALTERE_BASE - HALTERE_CLAMP,
             min(HALTERE_BASE + HALTERE_CLAMP, HALTERE_BASE + d))
    return dl, dr


def scaled_drive(idx, raw, haltere_total, cap=0.30):
    if len(idx) == 0:
        return 0.0
    return min(raw * haltere_total / len(idx), cap)


def warmup_and_baseline(brain, body):
    brain.reset(seed=0)
    for _ in range(WARMUP_TICKS):
        brain.step(HALTERE_BASE, HALTERE_BASE)
    body.capture_baseline(brain, n_ticks=BASELINE_TICKS)
    for h in body.hists:
        h.clear()
    return brain.steering()


def make_face(tripped, touch, thermal_peak, roll):
    """Reflect the current sensor state in the face."""
    if tripped:
        return "( x.x )"
    if touch:
        return "( o.o )"
    if thermal_peak > 35.0:
        return "( O.O )"
    if abs(roll) > 30.0:
        return "( <.< )" if roll < 0 else "( >.> )"
    return "( -.- )"


def state_label(tripped, touch, thermal_peak, roll):
    if tripped:
        return "TRIPPED"
    if touch:
        return "TOUCH"
    if thermal_peak > 35.0:
        return "THERMAL"
    if abs(roll) > 30.0:
        return "STEERING"
    return "IDLE"


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    print("[boot] connecting to MCU via Bridge ...")
    sensors = HardwareSensors()
    print("[boot] Bridge connected")

    print("[boot] loading connectome ...")
    brain = ConnectomeBrain(Path("flybrain"), gain=0.003)
    body = FlyBody(brain, DN_TYPES_PER_JOINT, center=90.0,
                   gain=40.0, smooth=0.20)

    steer_baseline = warmup_and_baseline(brain, body)
    print(f"[brain] steering baseline = {steer_baseline:+.4f}")

    haltere_total = HALTERE_BASE * (len(brain.left_idx) + len(brain.right_idx))

    # send a neutral command on boot
    sensors.set_servos([90.0, 90.0, 90.0, 90.0, 90.0])
    sensors.set_display("( -.- )", "BOOT", {
        "j0": 90.0, "j1": 90.0, "j2": 90.0, "j3": 90.0,
        "steer": 0.0, "servo": 90.0, "pokes": 0, "trips": 0,
    })

    t_start = time.perf_counter()
    last_display = 0.0
    prev_touch = False
    touch_until = 0.0
    trip_count = 0
    tick_count = 0

    print("[loop] running. Ctrl-C to stop.")

    try:
        while True:
            t0 = time.perf_counter()
            t_sec = t0 - t_start

            # --- read sensors ---
            try:
                reading = sensors.read()
            except Exception as e:
                print(f"[sensor] read failed: {e}", file=sys.stderr)
                time.sleep(DT)
                continue

            roll = float(reading.get("roll_deg", 0.0))
            touch_now = bool(reading.get("touch_left", False))
            thermal_frame = reading.get("thermal_frame", [25.0] * 64)
            thermal_peak = max(thermal_frame) if thermal_frame else 25.0
            thermal_min = min(thermal_frame) if thermal_frame else 25.0

            # --- IMU -> haltere ---
            dl, dr = imu_to_drive(roll)

            # --- limit switch -> tactile onset pulse ---
            extras = {}
            if touch_now and not prev_touch:
                touch_until = t_sec + TOUCH_PULSE_S
            if t_sec < touch_until:
                extras["tactile"] = (
                    brain.tactile,
                    scaled_drive(brain.tactile, TOUCH_ONSET, haltere_total),
                )
            prev_touch = touch_now

            # thermal is NOT wired to the connectome (it trips the network).
            # It appears in the display only.

            # --- tick connectome ---
            tick = brain.step(dl, dr, extras=extras)
            body.step(brain)

            # --- trip handling ---
            if tick["tripped"]:
                trip_count += 1
                sensors.set_display("( x.x )", "TRIPPED", {
                    "j0": float(body.angles[0]), "j1": float(body.angles[1]),
                    "j2": float(body.angles[2]), "j3": float(body.angles[3]),
                    "steer": 0.0, "servo": 90.0,
                    "pokes": 0, "trips": trip_count,
                })
                steer_baseline = warmup_and_baseline(brain, body)
                touch_until = 0.0
                prev_touch = False
                continue

            # --- steering + servo commands ---
            steer = brain.steering()
            servo = 90 + max(-SERVO_CLAMP,
                             min(SERVO_CLAMP, K_SERVO * (steer - steer_baseline)))

            sensors.set_servos([
                servo,
                float(body.angles[0]), float(body.angles[1]),
                float(body.angles[2]), float(body.angles[3]),
            ])

            # --- display (10 Hz) ---
            if t_sec - last_display >= DISPLAY_DT:
                face = make_face(tick["tripped"], touch_now,
                                 thermal_peak, roll)
                state = state_label(tick["tripped"], touch_now,
                                    thermal_peak, roll)
                telemetry = {
                    "j0": float(body.angles[0]), "j1": float(body.angles[1]),
                    "j2": float(body.angles[2]), "j3": float(body.angles[3]),
                    "steer": float(steer), "servo": float(servo),
                    "pokes": int(thermal_peak * 10),  # repurpose: thermal peak
                    "trips": trip_count,
                }
                try:
                    sensors.set_display(face, state, telemetry)
                except Exception as e:
                    print(f"[display] {e}", file=sys.stderr)
                last_display = t_sec

            # --- log every 10 ticks ---
            tick_count += 1
            if tick_count % 10 == 0:
                j = body.angles
                print(f"t={t_sec:7.2f} roll={roll:+6.1f} "
                      f"touch={int(touch_now)} thermal={thermal_peak:5.1f} "
                      f"steer={steer:+.3f} servo={servo:5.1f} "
                      f"joints=[{j[0]:6.1f} {j[1]:6.1f} "
                      f"{j[2]:6.1f} {j[3]:6.1f}]")

            # --- fixed tick rate ---
            elapsed = time.perf_counter() - t0
            if elapsed < DT:
                time.sleep(DT - elapsed)

    except KeyboardInterrupt:
        print("\n[loop] interrupted")
    except Exception as e:
        print(f"\n[loop] crashed: {e}", file=sys.stderr)
        raise
    finally:
        try:
            sensors.set_servos([90.0, 90.0, 90.0, 90.0, 90.0])
            sensors.set_display("( -.- )", "OFF", {
                "j0": 90.0, "j1": 90.0, "j2": 90.0, "j3": 90.0,
                "steer": 0.0, "servo": 90.0, "pokes": 0, "trips": trip_count,
            })
        except Exception:
            pass

    print(f"Closed. Trips: {trip_count}")


if __name__ == "__main__":
    main()
