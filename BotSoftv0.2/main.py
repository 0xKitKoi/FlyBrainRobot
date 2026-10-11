import time, math
import numpy as np
from pathlib import Path
from sensors.simulator import SimulatedSensors
from brain.connectome import ConnectomeBrain
from control.fly_body import FlyBody
from control.face import render_status

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


def main():
    sensors = SimulatedSensors()
    brain = ConnectomeBrain(Path("flybrain"), gain=0.003)
    #body = FlyBody(brain, DN_TYPES, center=90.0, gain=8.0, smooth=0.08)
    body = FlyBody(brain, DN_TYPES, center=90.0, gain=40.0, smooth=0.20)

    # warmup
    brain.reset(seed=0)
    for _ in range(1000):
        brain.step(0.175, 0.175)
    body.capture_baseline(brain, n_ticks=200)

    haltere_total = 0.175 * (len(brain.left_idx) + len(brain.right_idx))
    t_start = time.perf_counter()

    try:
        while True:
            t0 = time.perf_counter()
            t_sec = t0 - t_start

            reading = sensors.read()
            roll = 45.0 * math.sin(2 * math.pi * 0.05 * t_sec)
            dl, dr = imu_to_drive(roll)

            fb_norm = (np.mean(body.angles) - 90.0) / 60.0
            extras = {"feedback": (brain.ascending,
                                   scaled_drive(brain.ascending, fb_norm,
                                                haltere_total))}

            tick = brain.step(dl, dr, extras=extras)
            angles = body.step(brain)

            steer = brain.steering()
            servo = 90 + max(-60, min(60, 150 * (steer - 0.10)))

            cmd = body.command(servo)

            if brain.t % 10 == 0:
                print(f"t={t_sec:6.2f} roll={roll:+6.1f} "
                      f"steer={steer:+.3f} servo={servo:5.1f}  "
                      f"joints=[{angles[0]:.0f} {angles[1]:.0f} "
                      f"{angles[2]:.0f} {angles[3]:.0f}]  "
                      f"{'TRIP' if tick['tripped'] else ''}")

            elapsed = time.perf_counter() - t0
            if elapsed < DT:
                time.sleep(DT - elapsed)
    except KeyboardInterrupt:
        pass
    print("Closed.")


if __name__ == "__main__":
    main()