# main.py
"""
Random-stimulus demo loop.

- Randomly stimulates several sensory groups (like the poke lab, but live)
- Prints a "face" panel to the terminal showing current stimulus + joints
- Logs every tick in the poke-log format to run.log, so visualize_poke.py
  can turn it into a gif.

Usage:
    python main.py                # logs to run.log
    python main.py my_run.log     # logs to my_run.log
"""

import sys
import time
import random
import numpy as np
from pathlib import Path

from sensors.simulator import SimulatedSensors
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

# random poke schedule
POKE_INTERVAL_S = (1.5, 4.0)
POKE_DURATION_S = (0.2, 0.9)
POKE_DRIVE = (0.05, 0.40)

LOG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("run.log")

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def scaled_drive(idx, raw, haltere_total, cap=0.30):
    if len(idx) == 0:
        return 0.0
    return min(raw * haltere_total / len(idx), cap)


def warmup_and_baseline(brain, body):
    brain.reset(seed=0)
    for _ in range(WARMUP_TICKS):
        brain.step(0.175, 0.175)
    body.capture_baseline(brain, n_ticks=BASELINE_TICKS)
    for h in body.hists:
        h.clear()
    return brain.steering()


# ----------------------------------------------------------------------
# face rendering
# ----------------------------------------------------------------------
def make_face(active_pokes, joint_angles, tripped):
    """Return a single-line ASCII face that reflects the current stimulus."""
    if tripped:
        return "( x.x )"

    # if multiple pokes are active, prefer the first
    name = None
    if active_pokes:
        name = active_pokes[0][0]

    face_map = {
        "haltere_L":  "( <.< )",
        "haltere_R":  "( >.> )",
        "tactile":    "( o.o )",
        "thermo_hot": "( O.O )",
        "thermo_cool":"( o.O )",
        "visual_L":   "( O.o )",
        "visual_R":   "( o.O )",
        "ascending":  "( ^.^ )",
        "bristle":    "( -.- )",
    }
    if name is None:
        return "( -.- )"
    if name.startswith("bristle_"):
        return face_map["bristle"]
    return face_map.get(name, "( o.o )")


def render_status(active_pokes, joint_angles, steer, servo, tripped,
                  t_sec, poke_count, trip_count):
    """Return a multi-line string for the terminal panel."""
    face = make_face(active_pokes, joint_angles, tripped)

    # top stimulus line
    if active_pokes:
        # show up to 2 simultaneous pokes
        stim = "  ".join(f"{n}({d:.2f})" for n, d in active_pokes[:2])
    else:
        stim = "idle"

    lines = []
    lines.append("+--------------------------------------+")
    lines.append(f"|  {face:<32}  |")
    lines.append(f"|  t = {t_sec:6.2f} s                        |")
    lines.append(f"|  stimulus: {stim:<26} |")
    lines.append(f"|  steer = {steer:+.3f}   servo = {servo:5.1f}     |")
    lines.append(f"|  joints: "
                 f"Lsh={joint_angles[0]:5.1f} Lel={joint_angles[1]:5.1f} |")
    lines.append(f"|          "
                 f"Rsh={joint_angles[2]:5.1f} Rel={joint_angles[3]:5.1f} |")
    lines.append(f"|  pokes: {poke_count:4d}   trips: {trip_count:3d}        |")
    if tripped:
        lines.append("|  *** NETWORK TRIPPED ***             |")
    lines.append("+--------------------------------------+")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    rng = random.Random(1234)

    sensors = SimulatedSensors()
    brain = ConnectomeBrain(Path("flybrain"), gain=0.003)
    body = FlyBody(brain, DN_TYPES_PER_JOINT, center=90.0,
                   gain=40.0, smooth=0.20)

    steer_baseline = warmup_and_baseline(brain, body)
    print(f"[brain] steering baseline = {steer_baseline:+.4f}")

    haltere_total = 0.175 * (len(brain.left_idx) + len(brain.right_idx))

    # choose which groups to poke at random
    poke_groups = [
        ("haltere_L",   brain.left_idx),
        ("haltere_R",   brain.right_idx),
        ("tactile",     brain.tactile),
        ("thermo_hot",  brain.thermo_hot),
        ("thermo_cool", brain.thermo_cool),
        ("visual_L",    brain.vis_L),
        ("visual_R",    brain.vis_R),
        ("ascending",   brain.ascending),
    ]
    poke_groups = [(n, idx) for n, idx in poke_groups if len(idx) > 0]

    # open log in poke format
    log_f = LOG_PATH.open("w")
    print(f"[loop] logging to {LOG_PATH}", file=sys.stderr)
    print("[loop] press Ctrl-C to stop", file=sys.stderr)

    # state machine
    next_poke_t = 1.0
    active = {}     # name -> (idx, drive, end_time)
    poke_count = 0
    trip_count = 0

    t_start = time.perf_counter()
    last_print = 0.0

    try:
        while True:
            t0 = time.perf_counter()
            t_sec = t0 - t_start

            # --- start a new poke ---
            if t_sec >= next_poke_t and len(active) < 2:
                name, idx = rng.choice(poke_groups)
                drive = rng.uniform(*POKE_DRIVE)
                dur = rng.uniform(*POKE_DURATION_S)
                active[name] = (idx, drive, t_sec + dur)
                next_poke_t = t_sec + rng.uniform(*POKE_INTERVAL_S)
                poke_count += 1
                log_f.write(f"t={t_sec:6.2f}  POKE #{poke_count:3d}  "
                            f"{name:16s} drive={drive:.2f} dur={dur:.2f}\n")

            # --- expire finished pokes ---
            expired = [n for n, (_, _, end) in active.items() if t_sec >= end]
            for n in expired:
                log_f.write(f"t={t_sec:6.2f}  end\n")
                del active[n]

            # --- build extras ---
            extras = {}
            active_list = []
            for name, (idx, drive, _) in active.items():
                val = scaled_drive(idx, drive, haltere_total)
                extras[name] = (idx, val)
                active_list.append((name, drive))

            # --- tick brain ---
            tick = brain.step(0.175, 0.175, extras=extras)
            body.step(brain)

            # --- trip handling ---
            if tick["tripped"]:
                trip_count += 1
                log_f.write(f"t={t_sec:6.2f}  TRIP #{trip_count}  -- resetting\n")
                log_f.flush()
                steer_baseline = warmup_and_baseline(brain, body)
                active.clear()
                next_poke_t = t_sec + 2.0
                continue

            steer = brain.steering()
            servo = 90 + max(-60, min(60, 150 * (steer - steer_baseline)))

            # --- log joints every tick ---
            j = body.angles
            total = tick["total"]
            log_f.write(f"t={t_sec:6.2f} {'active':16s} "
                        f"joints=[{j[0]:6.1f} {j[1]:6.1f} "
                        f"{j[2]:6.1f} {j[3]:6.1f}] total={total:5d}\n")

            # --- terminal panel at 5 Hz ---
            if t_sec - last_print >= 0.2:
                panel = render_status(active_list, j, steer, servo,
                                      False, t_sec, poke_count, trip_count)
                # ANSI clear screen + home, then draw
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.write(panel + "\n")
                sys.stdout.flush()
                last_print = t_sec

            # --- fixed tick rate ---
            elapsed = time.perf_counter() - t0
            if elapsed < DT:
                time.sleep(DT - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        log_f.close()

    print(f"\nClosed. Pokes: {poke_count}  Trips: {trip_count}")
    print(f"Log written to {LOG_PATH}")
    print(f"To render: python visualize_poke.py {LOG_PATH}")


if __name__ == "__main__":
    main()