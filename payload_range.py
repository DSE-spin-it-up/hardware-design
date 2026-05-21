import numpy as np
from matplotlib import pyplot as plt

from pipeline import loop, config, reporting
from pipeline.sensitivity import clone_config
from sizing.wing import g0

cfg = clone_config(sizing=config.SIZING, propulsion=config.PROPULSION)
result = loop.run_pipeline(cfg)

m = result.masses["total"]
m_p = config.SIZING.m_payload
n_drones = config.SIZING.n_drones
m_b = result.masses["battery"]
E = result.propulsion.E_total
LD = result.cl_req / result.cd_full_buildup
R = config.SIZING.R

eta = (R * g0 * (m + m_p / n_drones)) / (E * LD * m_b)

R_lst = []
p_lst = []

for p in np.arange(m_p - 10):
    p_lst.append(p)
    R_lst.append(eta * E * LD * m_b / (1000 * g0 * (m + (p + 10) / n_drones)))

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(7, 4.5))

# Main line
ax.plot(R_lst, p_lst, color="#2563EB", linewidth=2)

# Reference lines at the design point
ax.axvline(R / 1000, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
ax.axhline(m_p - 10, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)

# Design-point marker
ax.scatter([R / 1000], [m_p - 10], color="#2563EB", zorder=5,
           label=f"Design point  ({R/1000:.0f} km, {m_p - 10:.1f} kg)")

# Axes labels & title
ax.set_xlabel("Range  [km]", labelpad=8)
ax.set_ylabel("Payload mass  [kg]", labelpad=8)
ax.set_title("Payload–Range Trade-off", fontsize=13, fontweight="bold", pad=12)

# Tight axis limits with small padding
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

ax.legend(frameon=False, fontsize=10)
fig.tight_layout()
plt.show()