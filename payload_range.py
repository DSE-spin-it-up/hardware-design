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

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(R_lst, p_lst, color="C0", linewidth=2)
ax.hlines(50, 0, 19.45, colors="C0", linewidth=2)

R_empty = eta * E * LD * m_b / (1000 * g0 * (m + (0 + 10) / n_drones))

ax.axvline(R / 1000, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.axhline(m_p - 10, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.scatter([R / 1000], [m_p - 10], color="C3", zorder=5,
           label=f"Design point  ({R / 1000:.0f} km, {m_p - 10:.1f} kg)")

ax.axvline(R_empty, color="C2", linewidth=0.8, linestyle="--", alpha=0.5)
ax.scatter([R_empty], [0], color="C2", zorder=5, marker="D",
           label=f"Empty payload  ({R_empty:.0f} km)")

ax.set_xlabel("Range  [km]", fontsize=13)
ax.set_ylabel("Payload mass  [kg]", fontsize=13)
ax.tick_params(labelsize=12)

ax.set_xlim(left=20)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
ax.legend(loc="best", fontsize=11)

fig.tight_layout()
plt.show()