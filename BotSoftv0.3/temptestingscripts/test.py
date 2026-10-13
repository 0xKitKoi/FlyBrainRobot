import numpy as np
from brain.connectome import ConnectomeBrain
from pathlib import Path

b = ConnectomeBrain(Path("flybrain"))

def run_cond(dl, dr, warm=600, ticks=1000):
    b.dn_L_hist.clear(); b.dn_R_hist.clear()
    for _ in range(warm):
        b.step(dl, dr)
    vals, dns, tots = [], [], []
    for _ in range(ticks):
        t = b.step(dl, dr)
        vals.append(b.steering())
        dns.append(t["dn_left"] + t["dn_right"])
        tots.append(t["total"])
    vals, dns, tots = map(np.array, (vals, dns, tots))
    return vals[-500:].mean(), vals[-500:].std(), dns.mean(), tots.mean()

for name, (dl, dr) in [("left-heavy", (0.30, 0.05)),
                       ("balanced",   (0.175, 0.175)),
                       ("right-heavy",(0.05, 0.30))]:
    m, s, dn, tot = run_cond(dl, dr)
    print(f"{name:12s} steering {m:+.3f} ± {s:.3f}   dn/tick {dn:.2f}   total/tick {tot:.1f}")
