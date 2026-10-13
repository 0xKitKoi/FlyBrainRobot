# poke_random_auto.py
"""
Auto-random pokes. No keyboard. Every 2-6 seconds, poke a random group
at a random intensity for a random duration. Trip auto-resets.
"""

import time
import random
import numpy as np
from pathlib import Path
from brain.connectome import ConnectomeBrain
from control.fly_body import FlyBody


DN_TYPES = ["DNg36_a", "DNpe004", "DNge138", "DNpe032"]
TICK_HZ = 100
DT = 1.0 / TICK_HZ
WARMUP_TICKS = 1000
BASELINE_TICKS = 200

#POKE_INTERVAL_S = (0.0, 2.0)
#POKE_DURATION_S = (0.5, 1.5)#(0.2, 1.5)
#POKE_DRIVE = (0.05, 0.6)
POKE_INTERVAL_S = (0.1, 0.6)     # rapid fire
POKE_DURATION_S = (0.1, 0.4)     # brief, so no sustained cascade
POKE_DRIVE = (0.3, 0.7)          # strong enough to move the joints



def build_groups(brain):
    a = brain.annotations
    groups = [
        ("haltere_L",    brain.left_idx),
        ("haltere_R",    brain.right_idx),
        ("tactile",      brain.tactile),
        ("thermo_hot",   brain.thermo_hot),
        ("thermo_cool",  brain.thermo_cool),
        ("visual_L",     brain.vis_L),
        ("visual_R",     brain.vis_R),
        ("ascending",    brain.ascending),
        ("DNg08",        brain.to_idx(a[a["type"] == "DNg08"]["bodyId"])),
        ("DNpe008",      brain.to_idx(a[a["type"] == "DNpe008"]["bodyId"])),
        ("cb_intrinsic", brain.to_idx(a[a["superclass"] == "cb_intrinsic"]["bodyId"])),
        ("vnc_intrinsic",brain.to_idx(a[a["superclass"] == "vnc_intrinsic"]["bodyId"])),
    ]
    tactile = a[a["class"] == "mechanosensory_tactile"]
    for t in ["SNta29", "SNta37", "SNta20", "SNta38"]:
        idx = brain.to_idx(tactile[tactile["type"] == t]["bodyId"])
        if len(idx):
            groups.append((f"bristle_{t}", idx))
    return [(n, idx) for n, idx in groups if len(idx) > 0]


def warmup_and_baseline(brain, body):
    brain.reset(seed=int(time.time() * 1000) & 0xffffffff)
    for _ in range(WARMUP_TICKS):
        brain.step(0.175, 0.175)
    body.capture_baseline(brain, n_ticks=BASELINE_TICKS)
    for h in body.hists:
        h.clear()


def main():
    rng = random.Random(int(time.time()))
    brain = ConnectomeBrain(Path("flybrain"), gain=0.003)
    body = FlyBody(brain, DN_TYPES, center=90.0, gain=40.0, smooth=0.20)

    warmup_and_baseline(brain, body)
    groups = build_groups(brain)

    print(f"[poke] {len(groups)} groups")
    for n, idx in groups:
        print(f"  {n:16s} {len(idx):6d} cells")
    print()

    haltere_total = 0.175 * (len(brain.left_idx) + len(brain.right_idx))

    def scaled(idx, val, cap=0.30):
        if len(idx) == 0:
            return 0.0
        return min(val * haltere_total / len(idx), cap)

    next_poke_t = 1.0
    poke_end_t = None
    current = None
    current_drive = 0.0
    total_pokes = 0
    total_trips = 0

    t_start = time.perf_counter()
    last_log = 0.0

    try:
        while True:
            t0 = time.perf_counter()
            t_sec = t0 - t_start

            if poke_end_t is None and t_sec >= next_poke_t:
                name, idx = rng.choice(groups)
                drive = rng.uniform(*POKE_DRIVE)
                dur = rng.uniform(*POKE_DURATION_S)
                current = (name, idx)
                current_drive = drive
                poke_end_t = t_sec + dur
                next_poke_t = poke_end_t + rng.uniform(*POKE_INTERVAL_S)
                total_pokes += 1
                print(f"t={t_sec:6.2f}  POKE #{total_pokes:3d}  {name:16s} "
                      f"drive={drive:.2f} dur={dur:.2f}")

            if poke_end_t is not None and t_sec >= poke_end_t:
                print(f"t={t_sec:6.2f}  end")
                poke_end_t = None
                current = None
                current_drive = 0.0

            extras = {}
            if current is not None:
                name, idx = current
                extras[name] = (idx, scaled(idx, current_drive))

            tick = brain.step(0.175, 0.175, extras=extras)
            body.step(brain)

            if tick["tripped"]:
                total_trips += 1
                print(f"t={t_sec:6.2f}  TRIP #{total_trips}  -- resetting")
                warmup_and_baseline(brain, body)
                poke_end_t = None
                current = None
                current_drive = 0.0
                next_poke_t = t_sec + 2.0
                continue

            if t_sec - last_log >= 0.1:
                j = body.angles
                active = current[0] if current else "idle"
                print(f"  t={t_sec:6.2f} {active:16s} "
                      f"joints=[{j[0]:6.1f} {j[1]:6.1f} {j[2]:6.1f} {j[3]:6.1f}]  "
                      f"total={tick['total']:4d}")
                last_log = t_sec

            elapsed = time.perf_counter() - t0
            if elapsed < DT:
                time.sleep(DT - elapsed)
    except KeyboardInterrupt:
        pass
    print(f"\nClosed. Pokes: {total_pokes}  Trips: {total_trips}")


if __name__ == "__main__":
    main()