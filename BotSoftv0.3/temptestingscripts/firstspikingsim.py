import time
import numpy as np
import scipy.sparse as sp
import pyarrow.feather as feather

D = "./flybrain/"
ids = np.load(D + "ids.npy")
indptr = np.load(D + "indptr.npy")
indices = np.load(D + "indices.npy")
data = np.load(D + "weights.npy")
n = len(ids)

# --- labels -> index space ---
a = feather.read_table(D + "body-annotations-male-cns-v1.0-minconf-0.5.feather",
                       columns=["bodyId", "superclass", "class", "type", "subclass", "entryNerve"]).to_pandas()
nt = feather.read_table(D + "body-neurotransmitters-male-cns-v1.0.feather",
                        columns=["body", "consensus_nt"]).to_pandas()

def to_idx(body_ids):
    body_ids = np.asarray(body_ids)
    pos = np.searchsorted(ids, body_ids)
    pos[pos >= n] = 0
    ok = ids[pos] == body_ids
    return pos[ok]

# --- sign per presynaptic neuron ---
sign = np.ones(n, dtype=np.float32)
nt_map = nt.set_index("body").consensus_nt
inhib = nt[nt.consensus_nt.isin(["gaba", "histamine", "glutamate"])].body
sign[to_idx(inhib)] = -1.0
data = data * np.repeat(sign, np.diff(indptr))   # apply sign to every outgoing edge

M = sp.csr_matrix((data, indices, indptr), shape=(n, n))  # row = pre

# --- pick inputs and outputs ---
halt = a[(a.entryNerve == "DMetaN") & a.superclass.str.contains("sensory", na=False)]
print("DMetaN sensory neurons:", len(halt))
print(halt[["type", "class", "subclass"]].value_counts().head(10))  # CHECK these look like haltere neurons
in_idx = to_idx(halt.bodyId)
out_idx = to_idx(a[a.superclass == "descending_neuron"].bodyId)
print("inputs:", len(in_idx), " outputs (descending):", len(out_idx))


# haltere-only inputs
in_idx = to_idx(a[(a.subclass == "haltere")].bodyId)
print("haltere inputs:", len(in_idx))

def run(gain, drive, steps=300, decay=0.95, thresh=1.0, seed=0):
    v = np.zeros(n, dtype=np.float32)
    spikes = np.zeros(n, dtype=bool)
    total = 0
    out_counts = np.zeros(len(out_idx))
    t0 = time.time()
    for t in range(steps):
        fired = np.flatnonzero(spikes)
        if len(fired):
            v += np.asarray(M[fired].sum(axis=0)).ravel() * gain
        v *= decay
        v[in_idx] += drive
        spikes = v >= thresh
        v[spikes] = 0.0
        total += spikes.sum()
        out_counts += spikes[out_idx]
        if total > 5e6:                     # bail out of runaway early
            return None
    el = time.time() - t0
    sec = steps / 1000
    return dict(gain=gain, drive=drive,
                rate_hz=total / n / sec,
                desc_active=int((out_counts > 0).sum()),
                speed_x=sec / el)

for gain in [0.02, 0.01, 0.005, 0.0025]:
    for drive in [0.1, 0.3]:
        print(run(gain, drive) or f"gain={gain} drive={drive}: RUNAWAY")




a = feather.read_table(D + "body-annotations-male-cns-v1.0-minconf-0.5.feather",
                       columns=["bodyId", "superclass", "class", "type", "subclass",
                                "entryNerve", "somaSide", "rootSide"]).to_pandas()
h = a[a.subclass == "haltere"]
print(h.somaSide.value_counts(dropna=False))
print(h.rootSide.value_counts(dropna=False))


SIDE = "rootSide"   # or "somaSide", whichever printed clean L/R values
left_idx  = to_idx(h[h[SIDE].isin(["L", "left", "Left"])].bodyId)
right_idx = to_idx(h[h[SIDE].isin(["R", "right", "Right"])].bodyId)
print("left:", len(left_idx), "right:", len(right_idx))

def run2(gain, drive_L, drive_R, steps=500, decay=0.95, thresh=1.0):
    v = np.zeros(n, dtype=np.float32)
    spikes = np.zeros(n, dtype=bool)
    total, out_counts = 0, np.zeros(len(out_idx))
    for t in range(steps):
        fired = np.flatnonzero(spikes)
        if len(fired):
            v += np.asarray(M[fired].sum(axis=0)).ravel() * gain
        v *= decay
        v[left_idx] += drive_L
        v[right_idx] += drive_R
        spikes = v >= thresh
        v[spikes] = 0.0
        total += spikes.sum()
        out_counts += spikes[out_idx]
        if total > 5e6:
            return None
    return total / n / (steps / 1000), out_counts

# 1. finer gain sweep
for gain in [0.003, 0.0035, 0.004]:
    r = run2(gain, 0.3, 0.3)
    print(gain, "RUNAWAY" if r is None else f"{r[0]:.1f} Hz, desc active {(r[1]>0).sum()}")

# 2. does left vs right matter? (use the best stable gain from above)
G = 0.0025
rest = run2(G, 0.0, 0.0)
left = run2(G, 0.3, 0.05)
right = run2(G, 0.05, 0.3)
for name, r in [("rest", rest), ("left-heavy", left), ("right-heavy", right)]:
    print(name, f"{r[0]:.2f} Hz, desc active {(r[1]>0).sum()}")

diff = left[1] - right[1]
print("descending neurons preferring left:", (diff > 0).sum(),
      " preferring right:", (diff < 0).sum())



a2 = feather.read_table(D + "body-annotations-male-cns-v1.0-minconf-0.5.feather",
                        columns=["bodyId", "superclass", "somaSide", "rootSide"]).to_pandas()
dn = a2[a2.superclass == "descending_neuron"].set_index("bodyId")
print(dn.somaSide.value_counts(dropna=False))
print(dn.rootSide.value_counts(dropna=False))


DN_SIDE = "somaSide"   # or "rootSide"
side = dn[DN_SIDE].reindex(ids[out_idx]).values
dn_L = np.isin(side, ["L", "left", "Left"])
dn_R = np.isin(side, ["R", "right", "Right"])
print("DN left:", dn_L.sum(), "DN right:", dn_R.sum())

def run3(gain, drive_L, drive_R, seed, steps=500, decay=0.95, thresh=1.0, jitter=0.3):
    rng = np.random.default_rng(seed)
    v = np.zeros(n, dtype=np.float32)
    spikes = np.zeros(n, dtype=bool)
    out_counts = np.zeros(len(out_idx))
    total = 0
    for t in range(steps):
        fired = np.flatnonzero(spikes)
        if len(fired):
            v += np.asarray(M[fired].sum(axis=0)).ravel() * gain
        v *= decay
        v[left_idx]  += drive_L * (1 + jitter * rng.standard_normal(len(left_idx)))
        v[right_idx] += drive_R * (1 + jitter * rng.standard_normal(len(right_idx)))
        spikes = v >= thresh
        v[spikes] = 0.0
        total += spikes.sum()
        out_counts += spikes[out_idx]
        if total > 5e6:
            return None
    return out_counts

G = 0.003
conds = {"balanced": (0.175, 0.175), "left-heavy": (0.3, 0.05), "right-heavy": (0.05, 0.3)}
results = {}
for name, (dl, dr) in conds.items():
    runs = [run3(G, dl, dr, seed) for seed in range(5)]
    if any(r is None for r in runs):
        print(name, "RUNAWAY"); continue
    # steering signal per seed: (right-side DN spikes - left-side DN spikes) / total
    s = [(r[dn_R].sum() - r[dn_L].sum()) / (r.sum() + 1) for r in runs]
    results[name] = np.array(s)
    print(f"{name:12s} total DN spikes {np.mean([r.sum() for r in runs]):.0f}"
          f"   steering {np.mean(s):+.3f} ± {np.std(s):.3f}")



G = 0.003
NS = 20
runs_L = [run3(G, 0.30, 0.05, s)        for s in range(NS)]
runs_R = [run3(G, 0.05, 0.30, s + 1000) for s in range(NS)]
runs_B = [run3(G, 0.175, 0.175, s + 2000) for s in range(NS)]
assert all(r is not None for r in runs_L + runs_R + runs_B), "runaway - lower G"
LH, RH, BAL = map(np.array, (runs_L, runs_R, runs_B))   # shape (NS, n_dn)

tr, te = slice(0, 10), slice(10, 20)

def tstat(a, b):
    return (a.mean(0) - b.mean(0)) / np.sqrt(a.var(0, ddof=1)/len(a) + b.var(0, ddof=1)/len(b) + 1e-9)

t = tstat(LH[tr], RH[tr])            # positive = prefers left-heavy input
K = 40
left_dn  = np.argsort(-t)[:K]
right_dn = np.argsort(t)[:K]

def steer(c):                        # positive = steer right
    return c[:, right_dn].sum(1) - c[:, left_dn].sum(1)

sL, sR, sB = steer(LH[te]), steer(RH[te]), steer(BAL[te])
print(f"left-heavy  {sL.mean():+6.1f} ± {sL.std():.1f}")
print(f"balanced    {sB.mean():+6.1f} ± {sB.std():.1f}")
print(f"right-heavy {sR.mean():+6.1f} ± {sR.std():.1f}")
pooled = np.sqrt((sL.var() + sR.var()) / 2)
print("separation (d'):", (sR.mean() - sL.mean()) / pooled)
print("DN types picked (left group):", a.set_index("bodyId").loc[ids[out_idx][left_dn], "type"].value_counts().head(8).to_dict())
print("DN types picked (right group):", a.set_index("bodyId").loc[ids[out_idx][right_dn], "type"].value_counts().head(8).to_dict())
