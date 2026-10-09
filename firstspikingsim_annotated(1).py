# firstspikingsim_annotated.py
# Annotated walkthrough of the original firstspikingsim.py.
# This file is intended for reading/learning, not as a new validated robot controller.
# The comments explain what each statement does and, where useful, why it is there.

import time  # Used to measure how long a simulation takes.
import numpy as np  # NumPy provides fast arrays and numerical operations.
import scipy.sparse as sp  # SciPy sparse matrices store the mostly-empty connectome efficiently.
import pyarrow.feather as feather  # Reads the Feather-format annotation tables.

D = "./flybrain/"  # Folder containing the downloaded fly connectome files.
ids = np.load(D + "ids.npy")  # Load neuron/body IDs in the row/column order used by the connectome.
indptr = np.load(D + "indptr.npy")  # CSR sparse-matrix row boundaries: where each neuron's outgoing edges begin/end.
indices = np.load(D + "indices.npy")  # Target-neuron column index for every stored connection.
data = np.load(D + "weights.npy")  # Weight/strength for every stored connection.
n = len(ids)  # Number of neurons represented by the IDs array; used for matrix dimensions.

# --- labels -> index space ---
# The annotation files use body IDs, while the matrix uses compact integer positions.
# These tables let us select neurons by biological labels, then map them into matrix positions.
a = feather.read_table(
    D + "body-annotations-male-cns-v1.0-minconf-0.5.feather",  # Annotation dataset path.
    columns=["bodyId", "superclass", "class", "type", "subclass", "entryNerve"]  # Read only fields this section needs.
).to_pandas()  # Convert the Arrow table to a pandas DataFrame for filtering/grouping.
nt = feather.read_table(
    D + "body-neurotransmitters-male-cns-v1.0.feather",  # Neurotransmitter annotation dataset path.
    columns=["body", "consensus_nt"]  # Need each neuron ID and its consensus transmitter label.
).to_pandas()  # Convert the table to pandas so we can filter it.

def to_idx(body_ids):  # Convert biological body IDs into positions in the connectome arrays.
    body_ids = np.asarray(body_ids)  # Ensure the input is a NumPy array, even if a pandas Series was passed.
    pos = np.searchsorted(ids, body_ids)  # Find where each ID would appear in sorted ids; this assumes ids is sorted.
    pos[pos >= n] = 0  # Prevent out-of-range indexing for IDs larger than all IDs; these will fail the check below.
    ok = ids[pos] == body_ids  # Keep only IDs that exactly match a dataset ID; missing IDs are discarded.
    return pos[ok]  # Return matrix positions for IDs that were found.

# --- sign per presynaptic neuron ---
# A connection's sign is assigned according to the transmitter annotation of its source neuron.
sign = np.ones(n, dtype=np.float32)  # Start by treating every neuron's outgoing connections as positive/excitatory.
nt_map = nt.set_index("body").consensus_nt  # Make a body-ID-indexed transmitter Series (currently not used later).
inhib = nt[nt.consensus_nt.isin(["gaba", "histamine", "glutamate"])].body  # Select IDs with these transmitter labels.
# IMPORTANT BIOLOGICAL CAVEAT: treating all three labels as inhibitory is a simplifying assumption.
# Glutamate can be excitatory or inhibitory depending on receptor/circuit; transmitter labels alone do not
# reliably determine synaptic sign in every neuron.
sign[to_idx(inhib)] = -1.0  # Mark the selected neurons as having negative outgoing weights.
data = data * np.repeat(sign, np.diff(indptr))  # Repeat each source neuron's sign once per outgoing edge, then sign its weights.

M = sp.csr_matrix((data, indices, indptr), shape=(n, n))  # Build sparse connectivity matrix; rows are presynaptic neurons.

# --- pick inputs and outputs ---
halt = a[(a.entryNerve == "DMetaN") & a.superclass.str.contains("sensory", na=False)]  # Select sensory neurons entering via DMetaN.
print("DMetaN sensory neurons:", len(halt))  # Report how many annotations match that filter.
print(halt[["type", "class", "subclass"]].value_counts().head(10))  # Show the ten most common label combinations for inspection.
# The comment in the original says to check that these resemble haltere neurons.
in_idx = to_idx(halt.bodyId)  # Convert selected sensory body IDs to connectome positions.
out_idx = to_idx(a[a.superclass == "descending_neuron"].bodyId)  # Select all annotated descending neurons and map them to positions.
print("inputs:", len(in_idx), " outputs (descending):", len(out_idx))  # Print input and output population sizes.

# haltere-only inputs
# This replaces the earlier in_idx selection: subsequent functions use this narrower set.
in_idx = to_idx(a[(a.subclass == "haltere")].bodyId)  # Select neurons annotated with subclass exactly "haltere".
print("haltere inputs:", len(in_idx))  # Report how many of those IDs exist in the connectome.

def run(gain, drive, steps=300, decay=0.95, thresh=1.0, seed=0):  # Run one simulation with uniform input to all selected input neurons.
    v = np.zeros(n, dtype=np.float32)  # Each neuron's accumulated activation/membrane-like value starts at zero.
    spikes = np.zeros(n, dtype=bool)  # Boolean array indicating which neurons fired on the previous time step.
    total = 0  # Count all neuron firing events across all steps.
    out_counts = np.zeros(len(out_idx))  # Count firing events for each descending neuron.
    t0 = time.time()  # Start timing the simulation.
    for t in range(steps):  # Advance the simulation for the requested number of discrete time steps.
        fired = np.flatnonzero(spikes)  # Find indices of neurons that fired in the previous step.
        if len(fired):  # Only do the matrix calculation if at least one neuron fired.
            v += np.asarray(M[fired].sum(axis=0)).ravel() * gain  # Sum outgoing effects from fired neurons into target neurons, scaled by gain.
        v *= decay  # Leak/decay every neuron's accumulated activation; 0.95 retains 95% each step.
        v[in_idx] += drive  # Inject the same artificial input amount into each selected haltere neuron each step.
        spikes = v >= thresh  # Neurons whose activation reaches threshold fire this step.
        v[spikes] = 0.0  # Reset firing neurons' activation to zero (a simplified reset rule).
        total += spikes.sum()  # Add this step's number of spikes to the total spike count.
        out_counts += spikes[out_idx]  # Add this step's spikes for descending neurons; bools add as 0/1.
        if total > 5e6:  # Safety stop if the simulation generates an extremely large number of spikes.
            return None  # Signal that this parameter combination ran away.
    el = time.time() - t0  # Measure elapsed wall-clock time.
    sec = steps / 1000  # Assume each simulation step represents 1 millisecond; 1000 steps = 1 second.
    return dict(gain=gain, drive=drive,  # Return parameters so the printed result identifies the experiment.
                rate_hz=total / n / sec,  # Average spikes per neuron per simulated second, not the rate of one neuron.
                desc_active=int((out_counts > 0).sum()),  # Number of descending neurons that fired at least once.
                speed_x=sec / el)  # Simulated seconds per real second; >1 means faster than real time.

for gain in [0.02, 0.01, 0.005, 0.0025]:  # Try progressively weaker synaptic scaling.
    for drive in [0.1, 0.3]:  # Try two strengths of constant sensory input.
        print(run(gain, drive) or f"gain={gain} drive={drive}: RUNAWAY")  # Print metrics, or a runaway warning if run() returned None.

# Re-read annotations with extra fields needed to split haltere neurons by side.
a = feather.read_table(
    D + "body-annotations-male-cns-v1.0-minconf-0.5.feather",  # Same annotation file.
    columns=["bodyId", "superclass", "class", "type", "subclass", "entryNerve", "somaSide", "rootSide"]  # Add side labels.
).to_pandas()  # Convert to pandas again; this overwrites the earlier variable a with the wider table.
h = a[a.subclass == "haltere"]  # Keep only haltere neurons.
print(h.somaSide.value_counts(dropna=False))  # Count soma-side labels, including missing values.
print(h.rootSide.value_counts(dropna=False))  # Count root-side labels, including missing values.

SIDE = "rootSide"  # Choose which metadata field identifies left/right; rootSide has usable L/R values in this output.
left_idx  = to_idx(h[h[SIDE].isin(["L", "left", "Left"])].bodyId)  # Map left-side haltere IDs to matrix positions.
right_idx = to_idx(h[h[SIDE].isin(["R", "right", "Right"])].bodyId)  # Map right-side haltere IDs to matrix positions.
print("left:", len(left_idx), "right:", len(right_idx))  # Print the number of usable neurons on each side.

def run2(gain, drive_L, drive_R, steps=500, decay=0.95, thresh=1.0):  # Simulate separate input strengths on left and right.
    v = np.zeros(n, dtype=np.float32)  # Reset all neuron activation values for this trial.
    spikes = np.zeros(n, dtype=bool)  # Reset the previous-step spike state.
    total, out_counts = 0, np.zeros(len(out_idx))  # Initialize total spikes and per-descending-neuron spike counts.
    for t in range(steps):  # Run the requested number of discrete time steps.
        fired = np.flatnonzero(spikes)  # Find neurons that fired in the previous step.
        if len(fired):  # If there are spikes to propagate...
            v += np.asarray(M[fired].sum(axis=0)).ravel() * gain  # ...sum their connection effects into targets and scale them.
        v *= decay  # Apply activation leakage.
        v[left_idx] += drive_L  # Apply the left-side sensory drive on every step.
        v[right_idx] += drive_R  # Apply the right-side sensory drive on every step.
        spikes = v >= thresh  # Decide which neurons fire this step.
        v[spikes] = 0.0  # Reset the activation of neurons that fired.
        total += spikes.sum()  # Count all spikes.
        out_counts += spikes[out_idx]  # Count spikes for each descending neuron.
        if total > 5e6:  # Detect an unstable/high-activity run.
            return None  # Return failure marker rather than continuing.
    return total / n / (steps / 1000), out_counts  # Return average network firing rate and per-DN spike counts.

# 1. finer gain sweep
for gain in [0.003, 0.0035, 0.004]:  # Test gains near the apparent transition to higher activity.
    r = run2(gain, 0.3, 0.3)  # Drive both sides equally at 0.3.
    print(gain, "RUNAWAY" if r is None else f"{r[0]:.1f} Hz, desc active {(r[1]>0).sum()}")  # Print network rate and number of DNs active.

# 2. does left vs right matter? (use the best stable gain from above)
G = 0.0025  # Choose a lower gain for the left/right comparison.
rest = run2(G, 0.0, 0.0)  # No injected sensory drive; baseline/control trial.
left = run2(G, 0.3, 0.05)  # Strong left input and weak right input.
right = run2(G, 0.05, 0.3)  # Weak left input and strong right input.
for name, r in [("rest", rest), ("left-heavy", left), ("right-heavy", right)]:  # Iterate over the three conditions.
    print(name, f"{r[0]:.2f} Hz, desc active {(r[1]>0).sum()}")  # Report network rate and active descending-neuron count.

diff = left[1] - right[1]  # For each DN, subtract its right-heavy spike count from its left-heavy count.
print("descending neurons preferring left:", (diff > 0).sum(),  # Count DNs with more spikes during left-heavy input.
      " preferring right:", (diff < 0).sum())  # Count DNs with more spikes during right-heavy input.
# A "preference" here only means a difference in this particular simplified simulation, not proven biological tuning.

# Read side annotations for descending neurons.
a2 = feather.read_table(
    D + "body-annotations-male-cns-v1.0-minconf-0.5.feather",  # Annotation file path.
    columns=["bodyId", "superclass", "somaSide", "rootSide"]  # Only columns used for this check.
).to_pandas()  # Convert to pandas.
dn = a2[a2.superclass == "descending_neuron"].set_index("bodyId")  # Keep descending neurons and use bodyId as the index.
print(dn.somaSide.value_counts(dropna=False))  # Count descending-neuron soma-side annotations, including missing values.
print(dn.rootSide.value_counts(dropna=False))  # Count descending-neuron root-side annotations, including missing values.

DN_SIDE = "somaSide"  # Use soma side because this dataset has L/R labels there for descending neurons.
side = dn[DN_SIDE].reindex(ids[out_idx]).values  # Align side labels to exactly the same order as out_idx.
dn_L = np.isin(side, ["L", "left", "Left"])  # Boolean mask: which output-array positions are left-side DNs.
dn_R = np.isin(side, ["R", "right", "Right"])  # Boolean mask: which output-array positions are right-side DNs.
print("DN left:", dn_L.sum(), "DN right:", dn_R.sum())  # Print how many output neurons were classified on each side.

def run3(gain, drive_L, drive_R, seed, steps=500, decay=0.95, thresh=1.0, jitter=0.3):  # Same model, but with randomized input variation.
    rng = np.random.default_rng(seed)  # Create a reproducible random-number generator for this trial.
    v = np.zeros(n, dtype=np.float32)  # Start all neuron activations at zero.
    spikes = np.zeros(n, dtype=bool)  # Start with no previous-step spikes.
    out_counts = np.zeros(len(out_idx))  # Count spikes per descending neuron.
    total = 0  # Count total network spikes; used only for runaway detection here.
    for t in range(steps):  # Advance one time step at a time.
        fired = np.flatnonzero(spikes)  # Find neurons that fired on the previous step.
        if len(fired):  # If any neurons fired...
            v += np.asarray(M[fired].sum(axis=0)).ravel() * gain  # ...propagate their signed connection weights to targets.
        v *= decay  # Leak activation toward zero.
        v[left_idx]  += drive_L * (1 + jitter * rng.standard_normal(len(left_idx)))  # Add noisy left sensory drive.
        v[right_idx] += drive_R * (1 + jitter * rng.standard_normal(len(right_idx)))  # Add noisy right sensory drive.
        spikes = v >= thresh  # Fire neurons whose activation reaches threshold.
        v[spikes] = 0.0  # Reset activation after firing.
        total += spikes.sum()  # Accumulate all network spikes.
        out_counts += spikes[out_idx]  # Accumulate each descending neuron's spike count.
        if total > 5e6:  # Stop pathological/high-activity simulations.
            return None  # Mark this run as unstable.
    return out_counts  # Return only descending-neuron spike counts; the caller analyzes those.

G = 0.003  # Set the gain for repeated noisy trials.
conds = {"balanced": (0.175, 0.175), "left-heavy": (0.3, 0.05), "right-heavy": (0.05, 0.3)}  # Define the three input conditions.
results = {}  # Store a summary array for each condition.
for name, (dl, dr) in conds.items():  # Iterate over condition name and its left/right drive values.
    runs = [run3(G, dl, dr, seed) for seed in range(5)]  # Repeat five times with different random noise sequences.
    if any(r is None for r in runs):  # Check whether any trial hit the runaway limit.
        print(name, "RUNAWAY"); continue  # Report failure and skip the rest of this condition.
    # Steering score: (right-side DN spikes - left-side DN spikes) / (all DN spikes + 1).
    # The +1 prevents division by zero if no descending neuron spikes.
    s = [(r[dn_R].sum() - r[dn_L].sum()) / (r.sum() + 1) for r in runs]  # Calculate one normalized side-balance score per trial.
    results[name] = np.array(s)  # Save the five scores as a NumPy array.
    print(f"{name:12s} total DN spikes {np.mean([r.sum() for r in runs]):.0f}"  # Print mean total descending-neuron spikes.
          f"   steering {np.mean(s):+.3f} ± {np.std(s):.3f}")  # Print average score and its standard deviation.
# This first steering score uses all left/right DNs; its sign depends on the chosen convention above.

G = 0.003  # Keep the gain fixed for a larger experiment.
NS = 20  # Number of independent random trials per input condition.
runs_L = [run3(G, 0.30, 0.05, s)        for s in range(NS)]  # Twenty left-heavy trials with seeds 0..19.
runs_R = [run3(G, 0.05, 0.30, s + 1000) for s in range(NS)]  # Twenty right-heavy trials with a separate seed range.
runs_B = [run3(G, 0.175, 0.175, s + 2000) for s in range(NS)]  # Twenty balanced trials with another seed range.
assert all(r is not None for r in runs_L + runs_R + runs_B), "runaway - lower G"  # Stop with an error if any trial was unstable.
LH, RH, BAL = map(np.array, (runs_L, runs_R, runs_B))  # Convert lists to arrays; expected shape is (NS, number of DNs).

tr, te = slice(0, 10), slice(10, 20)  # Split each condition into 10 discovery trials and 10 held-out test trials.

def tstat(a, b):  # Compute a per-neuron statistic comparing two sets of trials.
    return (a.mean(0) - b.mean(0)) / np.sqrt(a.var(0, ddof=1)/len(a) + b.var(0, ddof=1)/len(b) + 1e-9)  # Difference in means divided by estimated standard error; epsilon avoids division by zero.

t = tstat(LH[tr], RH[tr])  # Compare left-heavy vs right-heavy training trials for each DN; positive means higher in left-heavy.
K = 40  # Number of candidate DNs to select for each group.
left_dn  = np.argsort(-t)[:K]  # Select 40 DNs with the largest positive training statistic (left-heavy-preferring candidates).
right_dn = np.argsort(t)[:K]  # Select 40 DNs with the most negative training statistic (right-heavy-preferring candidates).

def steer(c):  # Convert each trial into a simple steering score using the selected DN groups.
    return c[:, right_dn].sum(1) - c[:, left_dn].sum(1)  # Right-group spikes minus left-group spikes; positive means "steer right" by this chosen convention.

sL, sR, sB = steer(LH[te]), steer(RH[te]), steer(BAL[te])  # Evaluate the score on held-out trials not used to select the groups.
print(f"left-heavy  {sL.mean():+6.1f} ± {sL.std():.1f}")  # Print mean and standard deviation for held-out left-heavy trials.
print(f"balanced    {sB.mean():+6.1f} ± {sB.std():.1f}")  # Print the same metrics for balanced trials.
print(f"right-heavy {sR.mean():+6.1f} ± {sR.std():.1f}")  # Print the same metrics for held-out right-heavy trials.
pooled = np.sqrt((sL.var() + sR.var()) / 2)  # Pool the two conditions' score variances into one scale estimate.
print("separation (d'):", (sR.mean() - sL.mean()) / pooled)  # Standardized difference between right-heavy and left-heavy scores; large magnitude means strong separation in these trials.
print("DN types picked (left group):", a.set_index("bodyId").loc[ids[out_idx][left_dn], "type"].value_counts().head(8).to_dict())  # Show common annotated types among selected left-preferring DNs.
print("DN types picked (right group):", a.set_index("bodyId").loc[ids[out_idx][right_dn], "type"].value_counts().head(8).to_dict())  # Show common annotated types among selected right-preferring DNs.

# Important limitations before hardware deployment:
# 1. This is a simplified threshold-and-leak model, not a detailed biological neuron model.
# 2. The time step is assumed to be 1 ms for reporting, but the simulation is not calibrated to biological time.
# 3. Every selected input neuron receives a constant drive every step; this is not yet a calibrated sensor encoding.
# 4. Transmitter-based signs are approximate, especially the blanket inhibitory treatment of glutamate.
# 5. Descending-neuron firing counts are not yet mapped to specific muscles, actuators, or safe motor commands.
# 6. The model needs validation, normalization, and a deliberate output decoder before it controls physical hardware.
