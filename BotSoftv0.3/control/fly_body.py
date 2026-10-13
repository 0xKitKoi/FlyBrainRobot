# control/fly_body.py
from collections import deque
import numpy as np

WINDOW = 200

class FlyBody:
    JOINT_NAMES = ("left_shoulder", "left_elbow",
                   "right_shoulder", "right_elbow")

    def __init__(self, brain, dn_types_per_joint, center=90.0,
                 gain=40.0, smooth=0.20):
        self.center = center
        self.gain = gain
        self.smooth = smooth
        self.dn_idx = []
        self.dn_names = []

        for i, types in enumerate(dn_types_per_joint):
            parts = []
            names = []
            for t in types:
                mask = ((brain.annotations["superclass"] == "descending_neuron")
                        & (brain.annotations["type"] == t))
                idx = brain.to_idx(brain.annotations[mask]["bodyId"])
                if len(idx):
                    parts.append(idx)
                    names.append(t)
            if parts:
                cat = np.concatenate(parts)
                print(f"[body] joint {i} ({self.JOINT_NAMES[i]}): "
                      f"{len(cat)} cells from {names}")
                self.dn_idx.append(cat)
                self.dn_names.append(names)
            else:
                print(f"[body] joint {i}: empty")
                self.dn_idx.append(np.array([], dtype=int))
                self.dn_names.append([])

        n = len(dn_types_per_joint)
        self.hists = [deque(maxlen=WINDOW) for _ in range(n)]
        self.baseline = np.zeros(n, dtype=np.float32)
        self.angles = np.full(n, center, dtype=np.float32)

    def read_dn_counts(self, brain):
        return np.array(
            [int(brain.spikes[idx].sum()) for idx in self.dn_idx],
            dtype=np.float32,
        )

    def capture_baseline(self, brain, n_ticks=200):
        counts = []
        for _ in range(n_ticks):
            counts.append(self.read_dn_counts(brain))
        self.baseline = np.mean(counts, axis=0)
        print(f"[body] baseline DN rates: {self.baseline}")

    def step(self, brain):
        counts = self.read_dn_counts(brain)
        for i, c in enumerate(counts):
            self.hists[i].append(c)

        target = np.zeros(len(self.dn_idx), dtype=np.float32)
        for i in range(len(self.dn_idx)):
            mean = np.mean(self.hists[i]) if len(self.hists[i]) else 0.0
            target[i] = self.center + self.gain * (mean - self.baseline[i])

        self.angles = self.angles + self.smooth * (target - self.angles)
        return self.angles

    def command(self, servo_deg):
        cmd = {"servo_deg": float(servo_deg)}
        for i, name in enumerate(self.JOINT_NAMES):
            cmd[name] = float(self.angles[i])
        return cmd