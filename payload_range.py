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
# Styled to match the sensitivity-analysis figures (pipeline.sensitivity.
# plot_dual_axis): figsize (8, 5), default Matplotlib spines, the C0/C3 palette,
# units in square brackets, grid at alpha 0.3, and no embedded title (the report
# caption carries it). Legend placed automatically with loc="best".
fig, ax = plt.subplots(figsize=(8, 5))

# Continuous analytic relation, so a plain C0 line — no per-point markers, unlike
# the discrete one-factor-at-a-time sweeps that use "o-"/"s--".
ax.plot(R_lst, p_lst, color="C0", linewidth=2)

# Design-point guides and marker (same operating-point convention as the scissor
# plot: muted dashed crosshairs plus a contrasting C3 marker).
ax.axvline(R / 1000, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.axhline(m_p - 10, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.scatter([R / 1000], [m_p - 10], color="C3", zorder=5,
           label=f"Design point  ({R / 1000:.0f} km, {m_p - 10:.1f} kg)")

ax.set_xlabel("Range  [km]")
ax.set_ylabel("Payload mass  [kg]")
# ax.set_title("Payload–range trade-off")

ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
ax.legend(loc="best")

fig.tight_layout()
plt.show()