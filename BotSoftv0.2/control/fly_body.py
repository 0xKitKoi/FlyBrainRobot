# control/fly_body.py
from collections import deque
import numpy as np

WINDOW = 100 #200   # ticks, same as brain window

class FlyBody:
    """Maps 4 descending neuron groups to 4 joints, with smoothing
    and feedback."""

    JOINT_NAMES = ("left_shoulder", "left_elbow",
                   "right_shoulder", "right_elbow")

    def __init__(self, brain, dn_types, center=90.0, gain=40.0, smooth=0.20): #  center=90.0, gain=8.0, smooth=0.08):
        """
        brain: ConnectomeBrain instance
        dn_types: list of 4 type-name strings, one per joint
        center: neutral joint angle
        gain: degrees per unit of normalized DN rate
        smooth: EMA coefficient (lower = smoother)
        """
        self.center = center
        self.gain = gain
        self.smooth = smooth
        self.dn_idx = []
        self.dn_names = []
        for t in dn_types:
            mask = (brain.annotations["superclass"] == "descending_neuron") & \
                (brain.annotations["type"] == t)
            idx = brain.to_idx(brain.annotations[mask]["bodyId"])
            print(f"[body] {t}: {len(idx)} cells")
            self.dn_idx.append(idx)
            self.dn_names.append(t)

        # spike history per joint
        self.hists = [deque(maxlen=WINDOW) for _ in range(4)]
        # baseline per joint, captured after warmup
        self.baseline = np.zeros(4, dtype=np.float32)
        # current smoothed joint angles
        self.angles = np.full(4, center, dtype=np.float32)

    def read_dn_counts(self, brain):
        """Return array of 4 spike counts from the last tick."""
        return np.array([int(brain.spikes[idx].sum()) for idx in self.dn_idx],
                        dtype=np.float32)

    def capture_baseline(self, brain, n_ticks=200):
        """Average DN rates over n_ticks at rest."""
        counts = []
        for _ in range(n_ticks):
            counts.append(self.read_dn_counts(brain))
        self.baseline = np.mean(counts, axis=0)
        print(f"[body] baseline DN rates: {self.baseline}")

    def step(self, brain):
        """One tick: read DNs, smooth, update angles."""
        counts = self.read_dn_counts(brain)
        for i, c in enumerate(counts):
            self.hists[i].append(c)

        # normalized rate per joint: (current window mean - baseline) 
        target = np.zeros(4, dtype=np.float32)
        for i in range(4):
            mean = np.mean(self.hists[i]) if len(self.hists[i]) else 0.0
            target[i] = self.center + self.gain * (mean - self.baseline[i])

        # EMA smoothing
        self.angles = self.angles + self.smooth * (target - self.angles)
        return self.angles

    def command(self, servo_deg):
        """Full command dict for the motor layer."""
        return {
            "servo_deg": float(servo_deg),
            "left_shoulder": float(self.angles[0]),
            "left_elbow": float(self.angles[1]),
            "right_shoulder": float(self.angles[2]),
            "right_elbow": float(self.angles[3]),
            "feedback": float(np.mean(self.angles) - self.center),
        }
