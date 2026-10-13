# visualize_poke.py
"""
Render a poke log as a joint trajectory plot and an animated stick figure.

Usage:
    python visualize_poke.py <logfile>

Reads lines like:
  t=  3.66 DNg08            joints=[ 126.1   95.9   55.1   94.8]  total= 335
  t=  1.01  POKE #  1  thermo_hot       drive=0.40 dur=0.23
  t=  1.66  TRIP #1  -- resetting

Outputs:
  poke_plot.png   -- joint angles over time, with poke/trip markers
  poke_anim.gif   -- stick-figure animation
"""

import re
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# lines that carry joint data
JOINT_RE = re.compile(
    r"t=\s*(?P<t>[\d.]+)\s+\S+\s+"
    r"joints=\[\s*(?P<j0>-?[\d.]+)\s+(?P<j1>-?[\d.]+)\s+"
    r"(?P<j2>-?[\d.]+)\s+(?P<j3>-?[\d.]+)\]\s+total=\s*(?P<tot>\d+)"
)
# lines that announce a poke
POKE_RE = re.compile(
    r"t=\s*(?P<t>[\d.]+)\s+POKE\s+#\s*(?P<n>\d+)\s+(?P<name>\S+)\s+"
    r"drive=(?P<d>[\d.]+)\s+dur=(?P<dur>[\d.]+)"
)
# lines that announce a trip
TRIP_RE = re.compile(r"t=\s*(?P<t>[\d.]+)\s+TRIP")


def parse(path):
    times, joints, totals = [], [], []
    pokes, trips = [], []
    for line in Path(path).read_text().splitlines():
        m = JOINT_RE.search(line)
        if m:
            times.append(float(m["t"]))
            joints.append([float(m[f"j{i}"]) for i in range(4)])
            totals.append(int(m["tot"]))
            continue
        m = POKE_RE.search(line)
        if m:
            pokes.append((float(m["t"]), m["name"], float(m["d"])))
            continue
        m = TRIP_RE.search(line)
        if m:
            trips.append(float(m["t"]))
    if not times:
        raise SystemExit("no joint lines found -- is this a poke log?")
    return (np.array(times), np.array(joints),
            np.array(totals), pokes, trips)


def draw_arm(ax, joint_angles, t, active=None):
    """Draw a four-joint stick figure."""
    ax.clear()
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # torso
    ax.plot([0, 0], [0.8, 1.6], color="k", lw=3)

    def seg(origin, angle_deg, length, sign):
        a = np.radians(90.0 - angle_deg)
        dx = sign * length * np.sin(a)
        dy = -length * np.cos(a)
        return (origin[0] + dx, origin[1] + dy)

    # left arm (blue)
    l_sh = (0, 1.6)
    l_el = seg(l_sh, joint_angles[0], 0.8, -1)
    l_ha = seg(l_el, joint_angles[1], 0.8, -1)
    ax.plot([l_sh[0], l_el[0]], [l_sh[1], l_el[1]],
            color="tab:blue", lw=5, solid_capstyle="round")
    ax.plot([l_el[0], l_ha[0]], [l_el[1], l_ha[1]],
            color="tab:blue", lw=5, solid_capstyle="round")
    ax.scatter([l_ha[0]], [l_ha[1]], color="tab:blue", s=60, zorder=5)

    # right arm (red)
    r_sh = (0, 1.6)
    r_el = seg(r_sh, joint_angles[2], 0.8, +1)
    r_ha = seg(r_el, joint_angles[3], 0.8, +1)
    ax.plot([r_sh[0], r_el[0]], [r_sh[1], r_el[1]],
            color="tab:red", lw=5, solid_capstyle="round")
    ax.plot([r_el[0], r_ha[0]], [r_el[1], r_ha[1]],
            color="tab:red", lw=5, solid_capstyle="round")
    ax.scatter([r_ha[0]], [r_ha[1]], color="tab:red", s=60, zorder=5)

    title = f"t = {t:.1f} s"
    if active:
        title += f"   |  {active}"
    ax.set_title(title, fontsize=11)


def main():
    if len(sys.argv) < 2:
        print("usage: python visualize_poke.py <logfile>", file=sys.stderr)
        sys.exit(2)

    t, joints, totals, pokes, trips = parse(sys.argv[1])
    print(f"parsed {len(t)} ticks, t = {t[0]:.2f}..{t[-1]:.2f} s")
    print(f"pokes: {len(pokes)}   trips: {len(trips)}")

    # ---- static plot ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    names = ["L shoulder", "L elbow", "R shoulder", "R elbow"]
    colors = ["tab:blue", "tab:cyan", "tab:red", "tab:orange"]

    for i in range(4):
        axes[0].plot(t, joints[:, i], color=colors[i], label=names[i], lw=1.2)
    for tp, name, drive in pokes:
        axes[0].axvline(tp, color="grey", alpha=0.3, lw=0.8)
        axes[0].annotate(name, (tp, axes[0].get_ylim()[1]),
                         rotation=90, fontsize=6, va="top",
                         color="grey")
    for tt in trips:
        axes[0].axvspan(tt, tt + 1.5, color="red", alpha=0.15)
    axes[0].set_ylabel("joint angle (deg)")
    axes[0].legend(loc="upper left", ncol=4, fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Joint angles over time (grey lines = pokes, red = trip+reset)")

    axes[1].plot(t, totals, color="k", lw=0.8)
    axes[1].set_ylabel("spikes/tick")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig("poke_plot.png", dpi=120)
    print("wrote poke_plot.png")

    # ---- animation ----
    duration = t[-1] - t[0]
    fps = 15
    n_frames = int(duration * fps)
    frame_times = np.linspace(t[0], t[-1], n_frames)
    frame_idx = np.searchsorted(t, frame_times)

    # precompute active poke label per frame
    def active_at(ft):
        active = None
        for tp, name, drive in pokes:
            if tp <= ft <= tp + 0.5:   # rough window; actual duration is in log
                active = f"{name} d={drive:.2f}"
                break
        return active

    fig2, ax2 = plt.subplots(figsize=(5, 5))

    def update(i):
        idx = frame_idx[i]
        ft = frame_times[i]
        draw_arm(ax2, joints[idx], ft, active=active_at(ft))
        return []

    print(f"rendering {n_frames} frames at {fps} fps...")
    anim = FuncAnimation(fig2, update, frames=n_frames, interval=1000 // fps)
    anim.save("poke_anim.gif", writer=PillowWriter(fps=fps))
    print("wrote poke_anim.gif")


if __name__ == "__main__":
    main()