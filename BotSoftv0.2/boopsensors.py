# poke_sensor.py
"""
Sensor-driven pokes. The limit switch and the thermal sensor trigger
injections into the connectome. Each real sensor event becomes a "poke."
"""

import time, math
import numpy as np
from pathlib import Path
from brain.connectome import ConnectomeBrain
from control.fly_body import FlyBody

DN_TYPES = ["DNg36_a", "DNpe004", "DNge138", "DNpe032"]
TICK_HZ = 100
DT = 1.0 / TICK_HZ
WARMUP_TICKS = 1000
BASELINE_TICKS = 200

# limit switch -> tactile group
TOUCH_ONSET_DRIVE = 0.8
TOUCH_ONSET_MS = 200       # ms

# thermal -> 3 possible channels depending on frame
THERMAL_PULSE_DRIVE = 0.10
THERMAL_PULSE_MS = 200
THERMAL_MAG_THRESHOLD = 0.2


def build_groups(brain):
    a = brain.annotations
    return [
        ("haltere_L",    brain.left_idx),
        ("haltere_R",    brain.right_idx),
        ("tactile",      brain.tactile),
        ("thermo_hot",   brain.thermo_hot),
        ("thermo_cool",  brain.thermo_cool),
        ("visual_L",     brain.vis_L),
        ("visual_R",     brain.vis_R),
        ("ascending",    brain.ascending),
    ]


def warmup_and_baseline(brain, body):
    brain.reset(seed=0)
    for _ in range(WARMUP_TICKS):
        brain.step(0.175, 0.175)
    body.capture_baseline(brain, n_ticks=BASELINE_TICKS)
    for h in body.hists:
        h.clear()


def analyze_thermal_frame(frame):
    """8x8 temperature array in C. Returns (direction, magnitude)."""
    bg = np.median(frame)
    delta = frame - bg
    if delta.max() < 3.0:
        return 0.0, 0.0
    weights = np.clip(delta, 0, None)
    total = weights.sum()
    if total < 1e-6:
        return 0.0, 0.0
    xs = np.arange(8)[None, :].repeat(8, axis=0)
    cx = (weights * xs).sum() / total
    direction = (cx - 3.5) / 3.5
    magnitude = min(1.0, (delta.max() - 3.0) / 15.0)
    return float(direction), float(magnitude)


def main():
    brain = ConnectomeBrain(Path("flybrain"), gain=0.003)
    body = FlyBody(brain, DN_TYPES, center=90.0, gain=40.0, smooth=0.20)
    warmup_and_baseline(brain, body)

    # in real code, this comes from SimulatedSensors or the Modulino
    # here we script a sequence for bench testing
    def read_sensors(t):
        touch = (5 < t < 5.3) or (12 < t < 12.2) or (20 < t < 20.5)
        # thermal hot spot sweeps left to right over 20s, then back
        cx = 3.5 + 3.0 * math.sin(2 * math.pi * t / 20.0)
        if 8 < t < 28:
            peak = 30.0 + 5.0 * math.sin(2 * math.pi * (t - 8) / 5.0)
        else:
            peak = 25.0
        return touch, (cx, peak)

    haltere_total = 0.175 * (len(brain.left_idx) + len(brain.right_idx))

    def scaled(idx, val, cap=0.30):
        if len(idx) == 0:
            return 0.0
        return min(val * haltere_total / len(idx), cap)

    touch_prev = False
    touch_pulse_end = 0.0
    thermal_pulse_end = 0.0
    thermal_prev_mag = 0.0

    t_start = time.perf_counter()
    last_log = 0.0
    total_pokes = 0
    total_trips = 0

    try:
        while True:
            t0 = time.perf_counter()
            t_sec = t0 - t_start

            touch, thermal_frame_raw = read_sensors(t_sec)

            # --- touch handling ---
            if touch and not touch_prev:
                touch_pulse_end = t_sec + TOUCH_ONSET_MS / 1000.0
                total_pokes += 1
                print(f"t={t_sec:6.2f}  TOUCH  -> tactile")
            touch_prev = touch

            # --- thermal handling ---
            # synth an 8x8 frame for bench testing
            ys, xs = np.mgrid[0:8, 0:8]
            cx, peak = thermal_frame_raw
            frame = 25.0 + (peak - 25.0) * np.exp(
                -((xs - cx)**2 + (ys - 3.5)**2) / (2 * 1.5**2))
            direction, magnitude = analyze_thermal_frame(frame)

            # pulse the thermal channel on *onset* of heat only
            if magnitude > THERMAL_MAG_THRESHOLD and thermal_prev_mag <= THERMAL_MAG_THRESHOLD:
                thermal_pulse_end = t_sec + THERMAL_PULSE_MS / 1000.0
                total_pokes += 1
                side = "R" if direction > 0 else "L"
                print(f"t={t_sec:6.2f}  THERMAL  dir={direction:+.2f} "
                      f"mag={magnitude:.2f}  -> avoid, drive haltere_{side}")
            thermal_prev_mag = magnitude

            # --- build extras ---
            extras = {}
            if t_sec < touch_pulse_end:
                extras["tactile"] = (brain.tactile,
                                     scaled(brain.tactile, TOUCH_ONSET_DRIVE))
            if t_sec < thermal_pulse_end:
                extras["thermo_hot"] = (brain.thermo_hot,
                                        scaled(brain.thermo_hot, THERMAL_PULSE_DRIVE))
            # sustained thermal avoidance: drive the opposite haltere side
            if magnitude > THERMAL_MAG_THRESHOLD:
                avoid = 0.4 * magnitude
                if direction > 0:
                    extras["haltere_L"] = (brain.left_idx,
                                           scaled(brain.left_idx, avoid))
                else:
                    extras["haltere_R"] = (brain.right_idx,
                                           scaled(brain.right_idx, avoid))

            tick = brain.step(0.175, 0.175, extras=extras)
            body.step(brain)

            if tick["tripped"]:
                total_trips += 1
                print(f"t={t_sec:6.2f}  TRIP #{total_trips}  resetting")
                warmup_and_baseline(brain, body)
                touch_pulse_end = 0.0
                thermal_pulse_end = 0.0
                thermal_prev_mag = 0.0
                continue

            if t_sec - last_log >= 0.1:
                j = body.angles
                print(f"  t={t_sec:6.2f}  mag={magnitude:.2f}  "
                      f"joints=[{j[0]:6.1f} {j[1]:6.1f} {j[2]:6.1f} {j[3]:6.1f}]")
                last_log = t_sec

            elapsed = time.perf_counter() - t0
            if elapsed < DT:
                time.sleep(DT - elapsed)
    except KeyboardInterrupt:
        pass
    print(f"\nClosed. Pokes: {total_pokes}  Trips: {total_trips}")


if __name__ == "__main__":
    main()