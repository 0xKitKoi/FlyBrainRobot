# Hardware integration boundary (not implemented)
Use the UNO Q Linux MPU for Python/connectome processing and the STM32 MCU for time-critical I/O. Future MCU adapter will read roller switches, both MPU-6050s, and AMG8833, drive PCA9685/servos, and update the LCD after its exact controller is confirmed. Verify current UNO Q pin/API documentation before wiring.

Suggested Linux-to-MCU message:
`{"seq":12,"state":"LEFT_CONTACT","servos":{"left_shoulder":120,"left_elbow":65,"right_shoulder":90,"right_elbow":90}}`

MCU must clamp angles, reject stale commands, and move to a safe pose if commands time out. Use an external regulated servo supply and common ground; do not power servos from GPIO.





With K_SERVO = 150:
steering	delta	servo
−0.095 (left-heavy)	−33.7°	56.3°
0.130 (balanced)	0°	90°
0.306 (right-heavy)	+26.4°	116.4°

That uses about half your available ±60° clamp. You could raise K_SERVO to ~250 to use the full range, but I'd start at 150 and raise it only if the servo feels sluggish. The reason: your input contrast (0.30 vs 0.05) is the validated extreme. Real IMU roll will rarely hit that, so most of the time you'll be operating in the ±0.05 steering band, which at K=150 gives ±7.5° of servo motion. That's subtle. At K=250 it's ±12.5°. For a robot that's supposed to look like a fly reflex, subtle is probably right. For a demo, you might want more.


K_SERVO = 150.0          # degrees per unit steering, tune from trace
STEER_CLAMP = 60.0       # degrees
baseline = 0.130         # measured, or captured live at rest

def steering_to_servo(steer):
    delta = K_SERVO * (steer - baseline)
    return 90.0 + max(-STEER_CLAMP, min(STEER_CLAMP, delta))







At roll = 0: (0.175, 0.175) → balanced → steering +0.130 → servo 90°.
At roll = +45° (tilt right): d = +0.125 → dl = 0.05, dr = 0.30 → right-heavy → steering +0.306 → servo 116°.
At roll = −45° (tilt left): d = −0.125 → dl = 0.30, dr = 0.05 → left-heavy → steering −0.095 → servo 56°.

So the servo leans with the tilt. If you want a self-righting reflex (servo leans against the tilt), flip the sign of d in imu_to_drive. Flies do both depending on the reflex, so pick whichever looks right on the bench.

One caution: the relationship between roll and steering is not linear. Your two validated conditions are at roll ±45° (the extremes of the input span), and balanced at 0°. In between, the network may not interpolate linearly. Before you trust the servo mapping, run the live contrast at roll = −22.5°, 0°, +22.5° and confirm the steering values are roughly halfway between the extremes. If they're not, you have a nonlinearity and you'll want a lookup table rather than a linear K_SERVO.


def imu_to_drive(roll_deg, base=0.175, span=0.125, clamp=0.15):
    d = max(-1.0, min(1.0, roll_deg / 45.0)) * span
    dl = max(base - clamp, min(base + clamp, base - d))
    dr = max(base - clamp, min(base + clamp, base + d))
    return dl, dr
