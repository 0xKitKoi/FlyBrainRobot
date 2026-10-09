import numpy as np
import pandas as pd
import pyarrow.feather as feather
import scipy.sparse as sp

D = "./flybrain/"
W = D + "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"
A = D + "body-annotations-male-cns-v1.0-minconf-0.5.feather"
N = D + "body-neurotransmitters-male-cns-v1.0.feather"

# --- weights: numeric columns only ---
w = feather.read_table(W, columns=["body_pre", "body_post", "weight"]).to_pandas()
ids = np.unique(np.concatenate([w.body_pre.values, w.body_post.values]))
print("neurons in edges:", len(ids), " edges:", len(w))
print("weight min/median/max:", w.weight.min(), w.weight.median(), w.weight.max())

pre = np.searchsorted(ids, w.body_pre.values).astype(np.int32)
post = np.searchsorted(ids, w.body_post.values).astype(np.int32)
wt = w.weight.values.astype(np.float32)
del w

m = sp.csr_matrix((wt, (pre, post)), shape=(len(ids), len(ids)))  # row = presynaptic
m.indices = m.indices.astype(np.int32)
m.indptr = m.indptr.astype(np.int32)
print("CSR size: %.0f MB" % ((m.data.nbytes + m.indices.nbytes + m.indptr.nbytes) / 1e6))
deg = np.diff(m.indptr)
print("out-degree mean/max:", deg.mean(), deg.max())

np.save(D + "ids.npy", ids)
np.save(D + "indptr.npy", m.indptr)
np.save(D + "indices.npy", m.indices)
np.save(D + "weights.npy", m.data)

# --- labels: who is sensory / motor? ---
a = feather.read_table(A, columns=["bodyId", "superclass", "class", "subclass", "type",
                                   "somaSide", "entryNerve", "exitNerve"]).to_pandas()
print("\nannotation rows:", len(a))
print("\nSUPERCLASS:\n", a.superclass.value_counts(dropna=False).head(20))
print("\nCLASS:\n", a["class"].value_counts(dropna=False).head(30))
print("\nentryNerve (sample):\n", a.entryNerve.value_counts().head(10))
print("\nexitNerve (sample):\n", a.exitNerve.value_counts().head(10))

# --- neurotransmitters ---
n = feather.read_table(N, columns=["body", "consensus_nt", "predicted_nt"]).to_pandas()
n = n[n.body.isin(ids)]
print("\nCONSENSUS NT (neurons in network):\n", n.consensus_nt.value_counts(dropna=False))
