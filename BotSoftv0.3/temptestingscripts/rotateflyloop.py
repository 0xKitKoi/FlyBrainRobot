"""
FlyRobot live loop.

Two modes:
  - 'sine'  : scripted +/-45 deg sine roll at 0.05 Hz. For bench testing the
              full pipeline without an IMU. Default.
  - 'imu'   : read roll from sensors.read()["roll_deg"]. Use once the
              Modulino Movement is wired in and SimulatedSensors is replaced
              (or extended) with a real backend that fills roll_deg.

Tick rate is fixed at TICK_HZ. The connectome's decay constant assumes a
consistent tick interval, so don't change this without re-validating.
"""

import math
import time
from pathlib import Path

from sensors.simulator import SimulatedSensors
from brain.connectome import ConnectomeBrain
from control.behavior import BehaviorController
from control.motor_controller import MotorController
from control.face import render_status

# ----------------------------------------------------------------------
# tunables
# ----------------------------------------------------------------------
TICK_HZ = 100
DT = 1.0 / TICK_HZ

MODE = "sine"          # "sine" or "imu"
SINE_HZ = 0.05         # full sweep period = 20 s
SINE_AMP_DEG = 45.0    # matches the validated +/-45 deg range

WARMUP_TICKS = 1000    # discard network transient before capturing baseline
FLUSH_TICKS = 600      # extra flush before baseline capture, matches test.py

K_SERVO = 150.0        # deg per unit steering, validated
SERVO_CLAMP = 60.0     # deg, soft limit around 90
SERVO_CENTER = 90.0

PRINT_EVERY = 10       # throttle terminal output (ticks)


def imu_to_drive(roll_deg, base=0.175, span=0.125, clamp=0.15):
    """Map roll angle (deg) to left/right haltere drive.

    Validated range: roll +/-45 deg -> drive (0.30,0.05)..(0.05,0.30).
    Outside +/-45 deg the input saturates; the network was not validated
    beyond that and the drive is clamped to the operating envelope.
    """
    d = max(-1.0, min(1.0, roll_deg / 45.0)) * span
    dl = max(base - clamp, min(base + clamp, base - d))
    dr = max(base - clamp, min(base + clamp, base + d))
    return dl, dr


def steering_to_servo(steer, baseline):
    """Map steering to servo angle, clamped."""
    delta = K_SERVO * (steer - baseline)
    delta = max(-SERVO_CLAMP, min(SERVO_CLAMP, delta))
    return SERVO_CENTER + delta


def capture_baseline(brain):
    """Warm up the network, flush the readout window, then measure."""
    brain.reset(seed=0)
    for _ in range(WARMUP_TICKS):
        brain.step(0.175, 0.175)
    for _ in range(FLUSH_TICKS):
        brain.step(0.175, 0.175)
    return brain.steering()


def main():
    sensors = SimulatedSensors()
    behavior = BehaviorController()
    motors = MotorController()
    brain = ConnectomeBrain(Path(__file__).resolve().parent / "flybrain",
                            gain=0.003)

    baseline = capture_baseline(brain)
    print(f"[brain] baseline steering = {baseline:+.4f}")
    print(f"[loop]  mode={MODE}  tick={TICK_HZ} Hz  "
          f"K_SERVO={K_SERVO}  clamp=+/-{SERVO_CLAMP} deg")

    t_start = time.perf_counter()
    last_tripped = False

    try:
        while True:
            t0 = time.perf_counter()
            t_sec = t0 - t_start

            reading = sensors.read()

            if MODE == "sine":
                roll = SINE_AMP_DEG * math.sin(2 * math.pi * SINE_HZ * t_sec)
                reading["roll_deg"] = roll
            else:
                roll = reading.get("roll_deg", 0.0)

            dl, dr = imu_to_drive(roll)

            tick = brain.step(dl, dr)
            tripped = tick["tripped"]

            if tripped:
                reading["brain_steering"] = 0.0
                reading["brain_tripped"] = True
            else:
                reading["brain_steering"] = brain.steering()
                reading["brain_tripped"] = False

            state = behavior.update(reading)
            cmd = motors.command_for(state)

            # servo is written only when the brain is healthy. while tripped
            # the motor controller's default pose (servo center) holds.
            if not tripped:
                cmd["servo_deg"] = steering_to_servo(
                    reading["brain_steering"], baseline)

            # throttled one-line status
            if brain.t % PRINT_EVERY == 0:
                tag = "TRIP" if tripped else "    "
                servo = cmd.get("servo_deg", SERVO_CENTER)
                print(f"t={t_sec:7.2f}  roll={roll:+6.1f}  "
                      f"drive=({dl:.3f},{dr:.3f})  "
                      f"steer={reading['brain_steering']:+.3f}  "
                      f"servo={servo:5.1f}  {tag}")

            # one-shot notice on trip, one-shot notice on recovery
            if tripped and not last_tripped:
                print("[safety] runaway detected; servo pinned at center. "
                      "Restart or call brain.reset() to clear.")
            last_tripped = tripped

            # fixed tick rate
            elapsed = time.perf_counter() - t0
            if elapsed < DT:
                time.sleep(DT - elapsed)
    except KeyboardInterrupt:
        print("\n[loop] interrupted.")
    finally:
        print("Closed. No physical hardware was controlled.")


if __name__ == "__main__":
    main()
