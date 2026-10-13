from pathlib import Path
from sensors.simulator import SimulatedSensors
from brain.connectome import ConnectomeBrain
from control.behavior import BehaviorController
from control.motor_controller import MotorController
from control.face import render_status
import random

def main():
    sensors = SimulatedSensors()
    behavior = BehaviorController()
    motors = MotorController()
    brain = None
    print("FlyRobot simulator: l/r/b=touch, h=heat, m=motion, n=neutral, status, q=quit")
    while True:
        try:
            cmd = input("\nfly> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd in ("q", "quit", "exit"):
            break
        elif cmd == "l":
            sensors.reset(); sensors.set_touch(True, False)
        elif cmd == "r":
            sensors.reset(); sensors.set_touch(False, True)
        elif cmd == "b":
            sensors.reset(); sensors.set_touch(True, True)
        elif cmd == "h":
            #sensors.reset(); sensors.set_thermal(True, 0.85)
            sensors.reset(); sensors.set_thermal(True, random.random())
        elif cmd == "m":
            sensors.reset(); sensors.set_motion((0.2, 0.7, 0.1), (-0.1, 0.4, 0.2))
        elif cmd == "n":
            sensors.reset()
        elif cmd == "status":
            reading = sensors.read()
            state = behavior.update(reading)
            print(render_status(reading, state, motors.command_for(state)))
            continue
        elif not cmd:
            continue
        else:
            print("Unknown command."); continue

        reading = sensors.read()
        # The existing connectome model only receives left/right drive values.
        # These are an experimental proxy, not a validated sensor-to-neuron mapping.
        if reading["touch_left"] or reading["touch_right"] or reading["motion_left"] != (0.0,0.0,0.0) or reading["motion_right"] != (0.0,0.0,0.0):
            if brain is None:
                try:
                    brain = ConnectomeBrain(Path(__file__).resolve().parent / "flybrain")
                except Exception as exc:
                    print(f"Connectome data/model unavailable: {exc}")
                    print("Continuing with sensor/behavior simulation only.")
                    brain = False
            if brain:
                try:
                    dl = 0.30 if reading["touch_left"] else 0.05
                    dr = 0.30 if reading["touch_right"] else 0.05
                    reading.update(brain.run_episode(dl, dr, seed=1))
                except Exception as exc:
                    reading["brain_error"] = str(exc)
        state = behavior.update(reading)
        print(render_status(reading, state, motors.command_for(state)))
    print("Closed. No physical hardware was controlled.")

if __name__ == "__main__":
    main()
