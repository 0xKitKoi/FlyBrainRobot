# Hardware integration boundary (not implemented)
Use the UNO Q Linux MPU for Python/connectome processing and the STM32 MCU for time-critical I/O. Future MCU adapter will read roller switches, both MPU-6050s, and AMG8833, drive PCA9685/servos, and update the LCD after its exact controller is confirmed. Verify current UNO Q pin/API documentation before wiring.

Suggested Linux-to-MCU message:
`{"seq":12,"state":"LEFT_CONTACT","servos":{"left_shoulder":120,"left_elbow":65,"right_shoulder":90,"right_elbow":90}}`

MCU must clamp angles, reject stale commands, and move to a safe pose if commands time out. Use an external regulated servo supply and common ground; do not power servos from GPIO.
