"""
Hardware sensor interface for the Arduino UNO Q.

Replaces SimulatedSensors when running on real hardware. All sensor reads
and motor commands go through the Arduino Bridge RPC layer.
"""

from arduino.router_bridge import Bridge

class HardwareSensors:
    """
    Talks to the MCU sketch over Bridge RPC.

    Methods match SimulatedSensors.read() so main.py can swap between them.
    """

    def __init__(self):
        self.bridge = Bridge()
        self.bridge.connect()
        self.bridge.wait_connected(timeout=5)

    def read(self):
        """
        Read all sensors from the MCU in one batch.

        Returns a dict matching SimulatedSensors.read()'s interface.
        """
        # IMU (Modulino Movement) - roll, pitch, yaw
        imu = self.bridge.call("read_imu")
        roll = imu[0] if imu else 0.0

        # Thermal array - 64 bytes, each 0-255 representing (C+10)*5
        thermal_bytes = self.bridge.call("read_thermal")
        if thermal_bytes and len(thermal_bytes) == 64:
            # Decode back to Celsius
            thermal_frame = [(b / 5.0) - 10.0 for b in thermal_bytes]
        else:
            thermal_frame = [25.0] * 64

        # Limit switch
        switch = self.bridge.call("read_limit")

        return {
            "roll_deg": roll,
            "thermal_frame": thermal_frame,
            "touch_left": switch,
            "touch_right": False,
            "thermal_hot": max(thermal_frame) > 35.0,
            "thermal_intensity": min(1.0, (max(thermal_frame) - 30.0) / 15.0),
            "motion_left": (0.0, 0.0, 0.0),
            "motion_right": (0.0, 0.0, 0.0),
        }

    def set_servo(self, channel, degrees):
        """Set a servo angle on the PCA9685. channel 0-15."""
        self.bridge.call("set_servo", channel, degrees)

    def set_display(self, face, state, telemetry):
        """
        Send display data to the MCU for rendering on the ST7789V.
        face: str, e.g. '( o.o )'
        state: str, e.g. 'IDLE' or 'STEERING'
        telemetry: dict with joint angles, steering, etc.
        """
        # Pack into a compact tuple to stay under 256 bytes
        # Format: (face_id, state_id, j0, j1, j2, j3, steer, servo, pokes, trips)
        face_id = self._face_to_id(face)
        state_id = self._state_to_id(state)
        self.bridge.call(
            "update_display",
            face_id, state_id,
            telemetry.get("j0", 90.0),
            telemetry.get("j1", 90.0),
            telemetry.get("j2", 90.0),
            telemetry.get("j3", 90.0),
            telemetry.get("steer", 0.0),
            telemetry.get("servo", 90.0),
            telemetry.get("pokes", 0),
            telemetry.get("trips", 0),
        )

    def _face_to_id(self, face):
        """Map face string to a compact ID for the MCU."""
        faces = {
            "( -.- )": 0,   # idle
            "( <.< )": 1,   # haltere L
            "( >.> )": 2,   # haltere R
            "( o.o )": 3,   # tactile
            "( O.O )": 4,   # thermo hot
            "( o.O )": 5,   # thermo cool
            "( ^.^ )": 6,   # ascending
            "( x.x )": 7,   # tripped
        }
        return faces.get(face, 0)

    def _state_to_id(self, state):
        """Map state string to a compact ID."""
        states = {
            "IDLE": 0, "STEERING": 1, "TOUCH": 2,
            "THERMAL": 3, "TRIPPED": 4,
        }
        return states.get(state, 0)
