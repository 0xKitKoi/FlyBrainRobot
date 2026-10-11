from pathlib import Path
import numpy as np
import scipy.sparse as sp
import pyarrow.feather as feather
from collections import deque


class ConnectomeBrain:
    """Persistent-state connectome model for live robot use.

    Keeps run_episode() as a compatibility shim for offline scripts.
    """

    # Tunables -- expose these so main.py can log them
    DEFAULT_GAIN = 0.003
    DEFAULT_DECAY = 0.95
    DEFAULT_THRESH = 1.0
    DEFAULT_JITTER = 0.3
    WINDOW_MS = 200
    #RUNAWAY_SPIKES_PER_TICK = 400     # tune from rest/balanced numbers
    #RUNAWAY_TRIP_TICKS = 50           # ~50 ms at 1 kHz, 5 ticks at 100 Hz
    RUNAWAY_SPIKES_PER_TICK = 800     # total spikes, backstop
    RUNAWAY_DN_PER_TICK = 30          # selected DN spikes, primary
    RUNAWAY_TRIP_TICKS = 50           # debounce


    def __init__(self, data_dir, gain=None):
        d = Path(data_dir)
        self.ids = np.load(d / "ids.npy")
        indptr = np.load(d / "indptr.npy")
        indices = np.load(d / "indices.npy")
        data = np.load(d / "weights.npy")
        self.n = len(self.ids)

        self.annotations = feather.read_table(
            d / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
        ).to_pandas()
        nt = feather.read_table(
            d / "body-neurotransmitters-male-cns-v1.0.feather",
            columns=["body", "consensus_nt"],
        ).to_pandas()
        self.annotations = self.annotations.set_index("bodyId", drop=False)

        sign = np.ones(self.n, dtype=np.float32)
        inhib = nt[nt.consensus_nt.isin(["gaba", "histamine", "glutamate"])].body
        sign[self.to_idx(inhib)] = -1.0
        data = data * np.repeat(sign, np.diff(indptr))
        self.M = sp.csr_matrix((data, indices, indptr), shape=(self.n, self.n))

        self.out_idx = self.to_idx(
            self.annotations[self.annotations.superclass == "descending_neuron"].bodyId
        )
        h = self.annotations[self.annotations.subclass == "haltere"]
        side_col = "rootSide" if "rootSide" in h.columns else "somaSide"
        self.left_idx = self.to_idx(h[h[side_col].isin(["L", "left", "Left"])].bodyId)
        self.right_idx = self.to_idx(h[h[side_col].isin(["R", "right", "Right"])].bodyId)

        # --- thermosensory (antennal thermoreceptors) ---
        thermo = self.annotations[self.annotations["class"] == "thermosensory"]
        self.thermo_hot = self.to_idx(
            thermo[thermo["type"] == "TRN_VP2"]["bodyId"])
        self.thermo_cool = self.to_idx(
            thermo[thermo["type"].isin(["TRN_VP3a", "TRN_VP3b"])]["bodyId"])
        self.thermo_other = self.to_idx(
            thermo[thermo["type"] == "TRN_VP1m"]["bodyId"])

        # --- tactile (mechanosensory bristles, top 6 types) ---
        tact_types = ["SNta02,SNta09", "SNta29", "SNta37",
                      "SNta20", "SNta38", "SNta21"]
        self.tactile = self.to_idx(
            self.annotations[self.annotations["type"].isin(tact_types)]["bodyId"])

        print(f"[brain] thermo hot={len(self.thermo_hot)} "
              f"cool={len(self.thermo_cool)} other={len(self.thermo_other)}")
        print(f"[brain] tactile={len(self.tactile)}")



        self.gain = gain if gain is not None else self.DEFAULT_GAIN
        

        self.gain = gain if gain is not None else self.DEFAULT_GAIN

        # Decoder: K=40 most discriminating DNs per side, positions into out_idx.
        self.left_dn, self.right_dn = self._train_decoder()

        # Absolute indices into self.spikes for the live readout.
        self.dn_L = self.out_idx[self.left_dn]
        self.dn_R = self.out_idx[self.right_dn]

        # Coverage log: full DN groups by side (somaSide on DNs, not rootSide).
        dn = self.annotations[self.annotations.superclass == "descending_neuron"]
        side_col_dn = "somaSide" if "somaSide" in dn.columns else "rootSide"
        dn_side = dn.set_index("bodyId")[side_col_dn].reindex(self.ids[self.out_idx]).values
        self.dn_L_full = np.isin(dn_side, ["L", "left", "Left"])
        self.dn_R_full = np.isin(dn_side, ["R", "right", "Right"])
        print(f"[brain] selected DN groups: L={len(self.dn_L)} R={len(self.dn_R)} "
              f"(of full L={self.dn_L_full.sum()} R={self.dn_R_full.sum()} "
              f"of {len(self.out_idx)} descending)")

        self.reset(seed=0)
        
        # --- thermosensory (antennal thermoreceptors) ---
        thermo = self.annotations[self.annotations["class"] == "thermosensory"]
        self.thermo_hot = self.to_idx(
            thermo[thermo["type"] == "TRN_VP2"]["bodyId"])
        self.thermo_cool = self.to_idx(
            thermo[thermo["type"].isin(["TRN_VP3a", "TRN_VP3b"])]["bodyId"])
        self.thermo_other = self.to_idx(
            thermo[thermo["type"] == "TRN_VP1m"]["bodyId"])

        # --- tactile (mechanosensory bristles, top 6 types) ---
        tact_types = ["SNta02,SNta09", "SNta29", "SNta37",
                      "SNta20", "SNta38", "SNta21"]
        self.tactile = self.to_idx(
            self.annotations[self.annotations["type"].isin(tact_types)]["bodyId"])

        print(f"[brain] thermo hot={len(self.thermo_hot)} "
              f"cool={len(self.thermo_cool)} other={len(self.thermo_other)}")
        print(f"[brain] tactile={len(self.tactile)}")

        # in ConnectomeBrain.__init__, after tactile:
        vp = self.annotations[self.annotations["type"] == "R1-R6"]
        self.vis_L = self.to_idx(vp[vp["rootSide"] == "L"]["bodyId"])
        self.vis_R = self.to_idx(vp[vp["rootSide"] == "R"]["bodyId"])
        print(f"[brain] visual L={len(self.vis_L)} R={len(self.vis_R)}")

        # in ConnectomeBrain.__init__, add:
        asc = self.annotations[self.annotations["superclass"] == "ascending_neuron"]
        self.ascending = self.to_idx(asc["bodyId"])
        print(f"[brain] ascending: {len(self.ascending)}")


    # ------------------------------------------------------------------
    # index helper
    # ------------------------------------------------------------------
    def to_idx(self, body_ids):
        body_ids = np.asarray(body_ids)
        pos = np.searchsorted(self.ids, body_ids)
        valid = pos < self.n
        matched = np.zeros(len(pos), dtype=bool)
        matched[valid] = self.ids[pos[valid]] == body_ids[valid]
        return pos[matched]

    # ------------------------------------------------------------------
    # state management
    # ------------------------------------------------------------------
    def reset(self, seed=0):
        self.v = np.zeros(self.n, dtype=np.float32)
        self.spikes = np.zeros(self.n, dtype=bool)
        self.rng = np.random.default_rng(seed)
        self.dn_L_hist = deque(maxlen=self.WINDOW_MS)
        self.dn_R_hist = deque(maxlen=self.WINDOW_MS)
        self.tripped = False
        self._trip_count = 0
        self.t = 0

    # ------------------------------------------------------------------
    # one tick
    # ------------------------------------------------------------------

    def step(self, drive_l, drive_r, extras=None, gain=None):
        """One tick. extras = {idx_array: drive_value} for additional input groups."""
        g = gain if gain is not None else self.gain

        if self.tripped:
            return self._empty_return(tripped=True)

        fired = np.flatnonzero(self.spikes)
        if len(fired):
            self.v += np.asarray(self.M[fired].sum(axis=0)).ravel() * g
        self.v *= self.DEFAULT_DECAY

        if len(self.left_idx):
            self.v[self.left_idx] += drive_l * (
                1 + self.DEFAULT_JITTER * self.rng.standard_normal(len(self.left_idx)))
        if len(self.right_idx):
            self.v[self.right_idx] += drive_r * (
                1 + self.DEFAULT_JITTER * self.rng.standard_normal(len(self.right_idx)))

        if extras:
            for name, (idx, val) in extras.items():
                if len(idx) and val != 0.0:
                    self.v[idx] += val * (
                        1 + self.DEFAULT_JITTER
                        * self.rng.standard_normal(len(idx)))

        self.spikes = self.v >= self.DEFAULT_THRESH
        self.v[self.spikes] = 0.0
        self.t += 1

        n_spikes = int(self.spikes.sum())
        dn_L = int(self.spikes[self.dn_L].sum())
        dn_R = int(self.spikes[self.dn_R].sum())
        dn_total = dn_L + dn_R

        over = (n_spikes > self.RUNAWAY_SPIKES_PER_TICK
                or dn_total > self.RUNAWAY_DN_PER_TICK)
        if over:
            self._trip_count += 1
            if self._trip_count >= self.RUNAWAY_TRIP_TICKS:
                self.v[:] = 0.0
                self.spikes[:] = False
                self.tripped = True
                self.dn_L_hist.clear()
                self.dn_R_hist.clear()
                return self._empty_return(tripped=True)
        else:
            self._trip_count = 0

        self.dn_L_hist.append(dn_L)
        self.dn_R_hist.append(dn_R)

        return {"dn_left": dn_L, "dn_right": dn_R,
                "total": n_spikes, "tripped": False}






    def _empty_return(self, tripped):
        return {"dn_left": 0, "dn_right": 0, "total": 0, "tripped": tripped}

    # ------------------------------------------------------------------
    # windowed readout
    # ------------------------------------------------------------------
    def steering(self):
        """Normalized steering in roughly [-1, +1]. Positive = right-heavy output.

        Sign convention matches the offline script: (R - L) / (R + L + 1).
        Returns 0.0 while tripped.
        """
        if self.tripped:
            return 0.0
        L = sum(self.dn_L_hist)
        R = sum(self.dn_R_hist)
        # testing:
        if L+R==0:
            return 0.0
        return (R - L) / (R + L + 1)

    # ------------------------------------------------------------------
    # offline compatibility
    # ------------------------------------------------------------------
    def _run_raw(self, dl, dr, seed, steps=500, decay=0.95, thresh=1.0, jitter=0.3):
        """Unchanged from your original. Used only by _train_decoder and run_episode."""
        rng = np.random.default_rng(seed)
        v = np.zeros(self.n, dtype=np.float32)
        spikes = np.zeros(self.n, dtype=bool)
        counts = np.zeros(len(self.out_idx), dtype=np.float32)
        total = 0
        for _ in range(steps):
            fired = np.flatnonzero(spikes)
            if len(fired):
                v += np.asarray(self.M[fired].sum(axis=0)).ravel() * self.gain
            v *= decay
            if len(self.left_idx):
                v[self.left_idx] += dl * (1 + jitter * rng.standard_normal(len(self.left_idx)))
            if len(self.right_idx):
                v[self.right_idx] += dr * (1 + jitter * rng.standard_normal(len(self.right_idx)))
            spikes = v >= thresh
            v[spikes] = 0.0
            total += int(spikes.sum())
            counts += spikes[self.out_idx]
            if total > 5e6:
                raise RuntimeError("Connectome activity runaway; reduce gain.")
        return counts

    def _train_decoder(self):
        left = np.array([self._run_raw(0.30, 0.05, s) for s in range(10)])
        right = np.array([self._run_raw(0.05, 0.30, s) for s in range(10)])
        t = (left.mean(0) - right.mean(0)) / np.sqrt(
            left.var(0, ddof=1) / len(left) + right.var(0, ddof=1) / len(right) + 1e-9
        )
        k = min(40, max(1, len(t) // 2))
        return np.argsort(-t)[:k], np.argsort(t)[:k]

    def run_episode(self, drive_l, drive_r, seed=0):
        """Offline shim. Unchanged semantics, so your validation scripts still work."""
        counts = self._run_raw(drive_l, drive_r, seed)
        left = int(counts[self.left_dn].sum())
        right = int(counts[self.right_dn].sum())
        return {
            "brain_steering": right - left,
            "brain_left_activity": left,
            "brain_right_activity": right,
            "total_descending_spikes": int(counts.sum()),
        }
