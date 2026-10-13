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
"""


"""
import pyarrow.feather as feather
from pathlib import Path

a = feather.read_table(
    Path("flybrain/body-annotations-male-cns-v1.0-minconf-0.5.feather")
).to_pandas()

for pattern in ["thermo", "TPN", "VP", "tactile", "bristle", "campaniform"]:
    hits = a[a.apply(lambda r: pattern.lower() in str(r).lower(), axis=1)]
    print(f"{pattern}: {len(hits)}")
    if len(hits):
        print(hits[["class", "subclass", "type"]].value_counts().head(5))

"""

"""
import pyarrow.feather as feather
from pathlib import Path

a = feather.read_table(
    Path("flybrain/body-annotations-male-cns-v1.0-minconf-0.5.feather")
).to_pandas()

# thermal: print every row whose type/class/subclass contains thermo or TPN
thermo = a[a.apply(lambda r: any(
    s in str(r.get(c, "")) for c in ["class","subclass","type"]
    for s in ["thermo","Thermo","TPN","tpv","TPV"]
), axis=1)]
print("THERMAL:")
print(thermo[["bodyId","class","subclass","type"]].to_string())

# tactile: aggregate by type
tact = a[a["class"] == "mechanosensory_tactile"]
print("\nTACTILE by type:")
print(tact["type"].value_counts().head(20))

"""



"""
DN activity census.

Step 1: list DN types by cell count (find large populations)
Step 2: log per-type spike activity at rest (find types that actually fire)
"""

from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.feather as feather

from brain.connectome import ConnectomeBrain


WARMUP_TICKS = 1000
LOG_TICKS = 500


def main():
    # ---- connectome ----
    b = ConnectomeBrain(Path("flybrain"), gain=0.003)
    dn = b.annotations[b.annotations["superclass"] == "descending_neuron"]

    # ---- Step 1: type census ----
    print("=" * 60)
    print("STEP 1: descending neuron types by cell count")
    print("=" * 60)
    counts = dn["type"].value_counts()
    print(counts.head(40).to_string())
    print()
    print(f"total DN types: {len(counts)}")
    print(f"types with >=10 cells: {(counts >= 10).sum()}")
    print(f"types with >=20 cells: {(counts >= 20).sum()}")
    print(f"types with >=50 cells: {(counts >= 50).sum()}")

    # ---- Step 2: per-type activity at rest ----
    print()
    print("=" * 60)
    print("STEP 2: per-type spike activity at balanced drive")
    print("=" * 60)

    b.reset(seed=0)
    for _ in range(WARMUP_TICKS):
        b.step(0.175, 0.175)

    # precompute index arrays for each type to avoid recomputation
    type_idx = {}
    for t in dn["type"].unique():
        idx = b.to_idx(dn[dn["type"] == t]["bodyId"])
        if len(idx):
            type_idx[t] = idx

    spike_counts = defaultdict(int)
    for _ in range(LOG_TICKS):
        b.step(0.175, 0.175)
        for t, idx in type_idx.items():
            spike_counts[t] += int(b.spikes[idx].sum())

    # sort by activity
    ranked = sorted(spike_counts.items(), key=lambda x: -x[1])

    print(f"{'type':<20} {'cells':>6} {'spikes':>8} {'/tick':>8} {'/cell/tick':>12}")
    print("-" * 60)
    for t, c in ranked[:40]:
        cells = len(type_idx[t])
        per_tick = c / LOG_TICKS
        per_cell = per_tick / cells
        print(f"{t:<20} {cells:>6} {c:>8} {per_tick:>8.3f} {per_cell:>12.4f}")

    print()
    print(f"active types (>=1 spike in {LOG_TICKS} ticks): "
          f"{sum(1 for _, c in ranked if c > 0)}")
    print(f"total DN types tracked: {len(ranked)}")


if __name__ == "__main__":
    main()