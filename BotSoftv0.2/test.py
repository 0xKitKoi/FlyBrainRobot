from pathlib import Path
import numpy as np
from brain.connectome import ConnectomeBrain


"""
b = ConnectomeBrain(Path("flybrain"))
b.reset(seed=0)
for _ in range(1000):
    b.step(0.175, 0.175)
baseline = b.steering()
print(f"baseline = {baseline:+.4f}")

def imu_to_drive(roll_deg, base=0.175, span=0.125, clamp=0.15):
    d = max(-1.0, min(1.0, roll_deg / 45.0)) * span
    return (max(base-clamp, min(base+clamp, base-d)),
            max(base-clamp, min(base+clamp, base+d)))

for roll in [-45, -22.5, 0, 22.5, 45]:
    dl, dr = imu_to_drive(roll)
    b.dn_L_hist.clear(); b.dn_R_hist.clear()
    for _ in range(600):
        b.step(dl, dr)
    vals = []
    for _ in range(500):
        b.step(dl, dr)
        vals.append(b.steering())
    steer = float(np.mean(vals[-200:]))
    servo = 90 + 150 * (steer - baseline)
    print(f"roll {roll:+6.1f}  drive ({dl:.3f},{dr:.3f})  steering {steer:+.3f}  servo {servo:6.1f}")

"""

"""
b = ConnectomeBrain(Path("flybrain"), gain=0.004)
b.reset(seed=0)
for i in range(2000):
    t = b.step(0.175, 0.175)
    if t["tripped"]:
        print(f"tripped at tick {i}")
        break
else:
    print("did not trip in 2000 ticks -- raise gain or lower threshold")
print("tripped flag:", b.tripped)
b.reset(seed=0)
print("after reset:", b.tripped)
"""


import time
from pathlib import Path
from brain.connectome import ConnectomeBrain
b = ConnectomeBrain(Path("flybrain"), gain=0.003)
b.reset(seed=0)
for _ in range(1000):
    b.step(0.175, 0.175)
N = 5000
t0 = time.perf_counter()
for _ in range(N):
    b.step(0.175, 0.175)
dt = time.perf_counter() - t0
print(f"{N/dt:.0f} ticks/s ({1000*dt/N:.3f} ms/tick)")
