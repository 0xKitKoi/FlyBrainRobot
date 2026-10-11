"""
Multi-sensor simulation harness.

Three input groups injected into the MaleCNS connectome:
  - IMU roll  -> haltere neurons (left/right)
  - Thermal   -> TRN_VP2 (hot) / TRN_VP3a+b (cool), scalar warmth
  - Touch     -> SNta bristle group, two-state (onset 1.0, held 0.4)

Output: 80 selected descending neurons -> steering value.
Logs every tick to CSV. Prints a per-scenario summary.
"""

import csv
from pathlib import Path
import numpy as np

from brain.connectome import ConnectomeBrain

TICK_HZ = 100
DT = 1.0 / TICK_HZ
WARMUP_TICKS = 1000
LOG_PATH = Path("multisensor_log.csv")

# drive envelopes (validated for haltere; starting guesses for the others)
HALTERE_BASE = 0.175
HALTERE_SPAN = 0.125
HALTERE_CLAMP = 0.15

THERMAL_GAIN = 0.30   # max drive into TRN group at full warmth
TOUCH_ONSET = 1.00
TOUCH_HELD = 0.40


def imu_to_drive(roll_deg):
    d = max(-1.0, min(1.0, roll_deg / 45.0)) * HALTERE_SPAN
    dl = max(HALTERE_BASE - HALTERE_CLAMP,
             min(HALTERE_BASE + HALTERE_CLAMP, HALTERE_BASE - d))
    dr = max(HALTERE_BASE - HALTERE_CLAMP,
             min(HALTERE_BASE + HALTERE_CLAMP, HALTERE_BASE + d))
    return dl, dr


def thermal_to_drives(warmth):
    """warmth in [-1, +1]: positive = warmer than background, negative = cooler."""
    hot = max(0.0, warmth) * THERMAL_GAIN
    cool = max(0.0, -warmth) * THERMAL_GAIN
    return hot, cool


def touch_to_drive(switch_now, switch_prev):
    if switch_now and not switch_prev:
        return TOUCH_ONSET
    if switch_now:
        return TOUCH_HELD
    return 0.0


def scaled_drive(idx, raw_drive, haltere_total, max_per_neuron=0.30):
    """Per-neuron drive so this group's total current matches haltere,
    but capped so small groups don't produce synchronized volleys."""
    if len(idx) == 0:
        return 0.0
    d = raw_drive * haltere_total / len(idx)
    return min(d, max_per_neuron)


# ----------------------------------------------------------------------
# scenarios
# ----------------------------------------------------------------------
# Each entry: (t_start, t_end, {"roll":deg, "warmth":-1..1, "switch":bool})
SCENARIOS = {
    "baseline": [
        (0, 30, {"roll": 0.0, "warmth": 0.0, "switch": False}),
    ],
    "imu_only": [
        (0, 10, {"roll":   0.0, "warmth": 0.0, "switch": False}),
        (10, 25, {"roll": -45.0, "warmth": 0.0, "switch": False}),
        (25, 40, {"roll": +45.0, "warmth": 0.0, "switch": False}),
        (40, 50, {"roll":   0.0, "warmth": 0.0, "switch": False}),
    ],
    "thermal_only": [
        (0, 10, {"roll": 0.0, "warmth":  0.0, "switch": False}),
        (10, 25, {"roll": 0.0, "warmth": +1.0, "switch": False}),  # warm hand
        (25, 35, {"roll": 0.0, "warmth": -1.0, "switch": False}),  # cold object
        (35, 45, {"roll": 0.0, "warmth":  0.0, "switch": False}),
    ],
    "touch_only": [
        (0, 10, {"roll": 0.0, "warmth": 0.0, "switch": False}),
        (10, 12, {"roll": 0.0, "warmth": 0.0, "switch": True}),   # brief press
        (12, 20, {"roll": 0.0, "warmth": 0.0, "switch": False}),
        (20, 25, {"roll": 0.0, "warmth": 0.0, "switch": True}),   # sustained press
        (25, 35, {"roll": 0.0, "warmth": 0.0, "switch": False}),
    ],
    "imu_plus_thermal": [
        (0, 10,  {"roll":   0.0, "warmth": 0.0, "switch": False}),
        (10, 25, {"roll": -45.0, "warmth": 0.0, "switch": False}),  # tilt alone
        (25, 40, {"roll": -45.0, "warmth": +1.0, "switch": False}), # tilt + warm
        (40, 55, {"roll":   0.0, "warmth": +1.0, "switch": False}), # warm alone
        (55, 65, {"roll":   0.0, "warmth": 0.0, "switch": False}),
    ],
    "all_three": [
        (0, 10,  {"roll":   0.0, "warmth":  0.0, "switch": False}),
        (10, 20, {"roll": -30.0, "warmth":  0.0, "switch": False}),
        (20, 30, {"roll": -30.0, "warmth": +1.0, "switch": False}),
        (30, 40, {"roll": -30.0, "warmth": +1.0, "switch": True}),
        (40, 50, {"roll":   0.0, "warmth":  0.0, "switch": True}),
        (50, 60, {"roll": +30.0, "warmth": -1.0, "switch": False}),
        (60, 70, {"roll":   0.0, "warmth":  0.0, "switch": False}),
    ],
    "visual_only": [
    (0, 10,  {"roll": 0, "warmth": 0, "switch": False, "lum_L": 0.5, "lum_R": 0.5}),
    (10, 25, {"roll": 0, "warmth": 0, "switch": False, "lum_L": 1.0, "lum_R": 0.1}),  # bright left
    (25, 40, {"roll": 0, "warmth": 0, "switch": False, "lum_L": 0.1, "lum_R": 1.0}),  # bright right
    (40, 50, {"roll": 0, "warmth": 0, "switch": False, "lum_L": 0.5, "lum_R": 0.5}),
    ],
}


# ----------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------
def run_scenario(brain, name, schedule, baseline):
    """Run one scenario, return list of per-tick log rows."""
    brain.reset(seed=0)
    for _ in range(WARMUP_TICKS):
        brain.step(HALTERE_BASE, HALTERE_BASE)
    # flush the readout deque
    for _ in range(600):
        brain.step(HALTERE_BASE, HALTERE_BASE)

    rows = []
    switch_prev = False
    t_total = schedule[-1][1]

    t = 0.0
    while t < t_total:
        # find active phase
        phase = None
        for t0, t1, vals in schedule:
            if t0 <= t < t1:
                phase = vals
                break
        if phase is None:
            t += DT
            continue

        dl, dr = imu_to_drive(phase["roll"])
        hot, cool = thermal_to_drives(phase["warmth"])
        touch = touch_to_drive(phase["switch"], switch_prev)
        switch_prev = phase["switch"]

        haltere_total = HALTERE_BASE * (len(brain.left_idx) + len(brain.right_idx))

        extras = {}
        if hot > 0:
            extras["thermo_hot"] = (
                brain.thermo_hot,
                scaled_drive(brain.thermo_hot, hot, haltere_total))
        if cool > 0:
            extras["thermo_cool"] = (
                brain.thermo_cool,
                scaled_drive(brain.thermo_cool, cool, haltere_total))
        if touch > 0:
            extras["tactile"] = (
                brain.tactile,
                scaled_drive(brain.tactile, touch, haltere_total))
        
        # new scenario field: "lum_L" and "lum_R" in [0, 1]
        if "lum_L" in phase:
            haltere_total = HALTERE_BASE * (len(brain.left_idx) + len(brain.right_idx))
            if phase["lum_L"] > 0:
                extras["vis_L"] = (brain.vis_L,
                                scaled_drive(brain.vis_L, phase["lum_L"], haltere_total))
            if phase["lum_R"] > 0:
                extras["vis_R"] = (brain.vis_R,
                                scaled_drive(brain.vis_R, phase["lum_R"], haltere_total))

        tick = brain.step(dl, dr, extras=extras)
        rows.append({
            "scenario": name,
            "t": round(t, 4),
            "roll": phase["roll"],
            "warmth": phase["warmth"],
            "switch": int(phase["switch"]),
            "hot": round(hot, 4),
            "cool": round(cool, 4),
            "touch": round(touch, 4),
            "steering": round(brain.steering(), 6),
            "dn_left": tick["dn_left"],
            "dn_right": tick["dn_right"],
            "total": tick["total"],
            "tripped": int(tick["tripped"]),
        })
        t += DT

    return rows


def summarize(name, rows, baseline):
    steers = np.array([r["steering"] for r in rows])
    trips = sum(r["tripped"] for r in rows)
    print(f"\n=== {name} ===")
    print(f"  ticks: {len(rows)}   trips: {trips}")
    print(f"  steering min/mean/max: "
          f"{steers.min():+.3f} / {steers.mean():+.3f} / {steers.max():+.3f}")
    print(f"  delta vs baseline: "
          f"{steers.min()-baseline:+.3f} .. {steers.max()-baseline:+.3f}")
    # bucket by phase for quick reading
    last_phase = None
    bucket = []
    for r in rows:
        key = (r["roll"], r["warmth"], r["switch"])
        if key != last_phase:
            if bucket:
                b = np.array(bucket)
                print(f"  roll={last_phase[0]:+6.1f} warm={last_phase[1]:+.1f} "
                      f"sw={last_phase[2]}  steer={b.mean():+.3f} "
                      f"(n={len(b)})")
            bucket = []
            last_phase = key
        bucket.append(r["steering"])
    if bucket:
        b = np.array(bucket)
        print(f"  roll={last_phase[0]:+6.1f} warm={last_phase[1]:+.1f} "
              f"sw={last_phase[2]}  steer={b.mean():+.3f} (n={len(b)})")


def main():
    brain = ConnectomeBrain(Path("flybrain"), gain=0.003)

    # capture baseline
    brain.reset(seed=0)
    for _ in range(WARMUP_TICKS):
        brain.step(HALTERE_BASE, HALTERE_BASE)
    for _ in range(600):
        brain.step(HALTERE_BASE, HALTERE_BASE)
    baseline = brain.steering()
    print(f"[sim] baseline steering = {baseline:+.4f}")

    all_rows = []
    for name, schedule in SCENARIOS.items():
        rows = run_scenario(brain, name, schedule, baseline)
        summarize(name, rows, baseline)
        all_rows.extend(rows)

    with LOG_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[sim] wrote {len(all_rows)} rows to {LOG_PATH}")


if __name__ == "__main__":
    main()