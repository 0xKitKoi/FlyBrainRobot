# main.py
import time, math
import numpy as np
from pathlib import Path
from sensors.simulator import SimulatedSensors
from brain.connectome import ConnectomeBrain
from control.fly_body import FlyBody

TICK_HZ = 100
DT = 1.0 / TICK_HZ
DN_TYPES = ["DNg36_a", "DNpe004", "DNge138", "DNpe032"]


def imu_to_drive(roll_deg, base=0.175, span=0.125, clamp=0.15):
    d = max(-1.0, min(1.0, roll_deg / 45.0)) * span
    return (max(base-clamp, min(base+clamp, base-d)),
            max(base-clamp, min(base+clamp, base+d)))


def scaled_drive(idx, raw, haltere_total, cap=0.30):
    if len(idx) == 0:
        return 0.0
    return min(raw * haltere_total / len(idx), cap)


def touch_to_drive(pressed, prev_pressed):
    """Two-state model: strong onset, weaker held."""
    if pressed and not prev_pressed:
        return 1.0
    if pressed:
        return 0.4
    return 0.0


def thermal_to_drive(warmth):
    """warmth in [-1, 1]. positive -> hot, negative -> cool."""
    if warmth > 0:
        return ("hot", warmth)
    return ("cool", -warmth)


def main():
    sensors = SimulatedSensors()
    brain = ConnectomeBrain(Path("flybrain"), gain=0.003)
    body = FlyBody(brain, DN_TYPES, center=90.0, gain=40.0, smooth=0.20)

    brain.reset(seed=0)
    for _ in range(1000):
        brain.step(0.175, 0.175)
    body.capture_baseline(brain, n_ticks=200)

    haltere_total = 0.175 * (len(brain.left_idx) + len(brain.right_idx))
    t_start = time.perf_counter()
    prev_switch = False

    try:
        while True:
            t0 = time.perf_counter()
            t_sec = t0 - t_start

            reading = sensors.read()

            # scripted roll for bench, or real IMU roll
            roll = reading.get("roll_deg", 45.0 * math.sin(2 * math.pi * 0.05 * t_sec))
            dl, dr = imu_to_drive(roll)

            # --- build extras from all sensors ---
            extras = {}

            # 1. tactile: limit switch -> tactile group
            switch = reading.get("touch_left", False) or reading.get("touch_right", False)
            touch = touch_to_drive(switch, prev_switch)
            prev_switch = switch
            if touch > 0:
                extras["touch"] = (brain.tactile,
                                   scaled_drive(brain.tactile, touch, haltere_total))

            # 2. thermal: scalar warmth -> thermo hot / cool
            warmth = reading.get("thermal_intensity", 0.0)
            if reading.get("thermal_hot", False):
                warmth = max(warmth, 0.5)   # any hot flag gives at least 0.5
            else:
                warmth = 0.0
            if warmth > 0:
                extras["thermo_hot"] = (brain.thermo_hot,
                    scaled_drive(brain.thermo_hot, warmth, haltere_total))

            # 3. servo feedback -> ascending group
            fb = (np.mean(body.angles) - 90.0) / 60.0
            extras["feedback"] = (brain.ascending,
                scaled_drive(brain.ascending, fb, haltere_total))

            # --- tick brain ---
            tick = brain.step(dl, dr, extras=extras)
            body.step(brain)

            # --- servo from steering ---
            steer = brain.steering()
            servo = 90 + max(-60, min(60, 150 * (steer - 0.10)))

            # --- log ---
            if brain.t % 10 == 0:
                j = body.angles
                print(f"t={t_sec:6.2f} roll={roll:+6.1f} sw={int(switch)} "
                      f"warm={warmth:+.2f} steer={steer:+.3f} "
                      f"joints=[{j[0]:.0f} {j[1]:.0f} {j[2]:.0f} {j[3]:.0f}] "
                      f"{'TRIP' if tick['tripped'] else ''}")

            elapsed = time.perf_counter() - t0
            if elapsed < DT:
                time.sleep(DT - elapsed)
    except KeyboardInterrupt:
        pass
    print("Closed.")


if __name__ == "__main__":
    main()