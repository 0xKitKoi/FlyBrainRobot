from pathlib import Path
import numpy as np
import scipy.sparse as sp
import pyarrow.feather as feather

class ConnectomeBrain:
    """Episode-based refactor of firstspikingsim.py; simplified research model."""
    def __init__(self, data_dir):
        d = Path(data_dir)
        self.ids = np.load(d / "ids.npy")
        indptr = np.load(d / "indptr.npy")
        indices = np.load(d / "indices.npy")
        data = np.load(d / "weights.npy")
        self.n = len(self.ids)
        self.annotations = feather.read_table(d / "body-annotations-male-cns-v1.0-minconf-0.5.feather").to_pandas()
        nt = feather.read_table(d / "body-neurotransmitters-male-cns-v1.0.feather", columns=["body","consensus_nt"]).to_pandas()
        self.annotations = self.annotations.set_index("bodyId", drop=False)
        sign = np.ones(self.n, dtype=np.float32)
        inhib = nt[nt.consensus_nt.isin(["gaba","histamine","glutamate"])].body
        sign[self.to_idx(inhib)] = -1.0
        data = data * np.repeat(sign, np.diff(indptr))
        self.M = sp.csr_matrix((data, indices, indptr), shape=(self.n,self.n))
        self.out_idx = self.to_idx(self.annotations[self.annotations.superclass == "descending_neuron"].bodyId)
        h = self.annotations[self.annotations.subclass == "haltere"]
        side_col = "rootSide" if "rootSide" in h.columns else "somaSide"
        self.left_idx = self.to_idx(h[h[side_col].isin(["L","left","Left"])].bodyId)
        self.right_idx = self.to_idx(h[h[side_col].isin(["R","right","Right"])].bodyId)
        self.gain = 0.003
        self.left_dn, self.right_dn = self._train_decoder()

    def to_idx(self, body_ids):
        body_ids = np.asarray(body_ids)
        pos = np.searchsorted(self.ids, body_ids)
        valid = pos < self.n
        matched = np.zeros(len(pos), dtype=bool)
        matched[valid] = self.ids[pos[valid]] == body_ids[valid]
        return pos[matched]

    def _run_raw(self, dl, dr, seed, steps=500, decay=0.95, thresh=1.0, jitter=0.3):
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
                v[self.left_idx] += dl * (1 + jitter*rng.standard_normal(len(self.left_idx)))
            if len(self.right_idx):
                v[self.right_idx] += dr * (1 + jitter*rng.standard_normal(len(self.right_idx)))
            spikes = v >= thresh
            v[spikes] = 0.0
            total += int(spikes.sum())
            counts += spikes[self.out_idx]
            if total > 5e6:
                raise RuntimeError("Connectome activity runaway; reduce gain.")
        return counts

    def _train_decoder(self):
        left = np.array([self._run_raw(0.30,0.05,s) for s in range(10)])
        right = np.array([self._run_raw(0.05,0.30,s) for s in range(10)])
        t = (left.mean(0)-right.mean(0)) / np.sqrt(left.var(0,ddof=1)/len(left)+right.var(0,ddof=1)/len(right)+1e-9)
        k = min(40, max(1, len(t)//2))
        return np.argsort(-t)[:k], np.argsort(t)[:k]

    def run_episode(self, drive_l, drive_r, seed=0):
        counts = self._run_raw(drive_l, drive_r, seed)
        left = int(counts[self.left_dn].sum())
        right = int(counts[self.right_dn].sum())
        return {"brain_steering": right-left, "brain_left_activity": left,
                "brain_right_activity": right, "total_descending_spikes": int(counts.sum())}
