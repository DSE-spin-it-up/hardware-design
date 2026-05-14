# hardware-design

Conceptual sizing pipeline for a fixed-wing cargo drone. A single design loop
in `pipeline/loop.py` runs the wing → propulsion → fuselage → drag → mass →
wing-area chain to convergence, then reports the result and plots polars.

All design knobs live in `config.yaml`. Edit values there and re-run
`python main.py` — no Python edits required for a parameter sweep.

---

## Quick start

```bash
# install dependencies (uv recommended)
uv sync

# run the full pipeline
uv run python main.py

# run any single sub-solver standalone (see "Running individual modules" below)
uv run python -m sizing.wing
```

Requirements:
- Python ≥ 3.13 (see `pyproject.toml`)
- `xfoil` on `$PATH` (used by `aerodynamics/airfoil_polar.py`)
- numpy, scipy, matplotlib, pandas, pyyaml (handled by `uv sync`)

---

## Repository layout

```
hardware-design/
├── main.py                 # entry point — wires loop → reporting → plots
├── config.yaml             # all design knobs (edit this, not Python)
├── pipeline/               # convergence loop, plots, reporting, config loader
│   ├── config.py           # loads config.yaml into module-level constants
│   ├── loop.py             # PipelineResult + run_pipeline (the iteration)
│   ├── helpers.py          # airfoil resolution, drag/LLT wrappers, polar sweep
│   ├── plots.py            # plot_convergence, plot_drone_ld
│   └── reporting.py        # console summary blocks
├── sizing/
│   ├── wing.py             # initial wing + tail sizing (geometry, atmosphere, CL)
│   ├── fuselage.py         # fuselage box dimensions around battery + airfoil
│   └── aileron.py          # aileron span search to meet roll-rate requirement
├── propulsion/
│   ├── sizing.py           # cruise + climb + VTOL prop sizing → battery, motor
│   └── propeller_solver.py # RPM walk against a propeller-table CSV
├── aerodynamics/
│   ├── airfoil_polar.py    # XFOIL wrapper, cached 2-D viscous polar
│   ├── airfoil_geometry.py # geometry from .dat (thickness, area, centroid)
│   ├── llt.py              # Prandtl lifting-line solver
│   └── drag_buildup.py     # Raymer CD0 buildup for wing + tail + fuselage
├── structures/
│   ├── rods.py             # CFRP rod wall thickness for deflection + stress
│   └── materials.py        # CFRP, EPP, CF-PLA, PLA, Wood material objects
├── weights/
│   ├── mass.py             # total mass + CG x-position
│   └── part_materials.py   # which material each part is made of
├── airfoils/               # .dat coordinate files (MH112, NACA4412, NACA0010, …)
└── data/                   # propeller performance tables (e.g. 20x10E_performance.csv)
```

---

## Architecture

### Sub-solver interface

Every sub-solver in `sizing/`, `propulsion/`, `structures/`, `aerodynamics/`,
and `weights/` follows the same shape:

- an `*Inputs` dataclass (default values + a `__init__` populated from
  `config.yaml`)
- a `run(...)` function that takes the upstream result(s) + its `*Inputs`
  and returns an `*Result` dataclass
- a `summary(result)` printer for the console report
- a `__main__` block that smoke-tests the module standalone

`config.yaml` is mapped onto these dataclasses by `pipeline/config.py`:
each top-level key in the YAML matches the constructor of one `*Inputs`
class. Adding a new knob therefore means adding a field to both the
dataclass and the YAML — `pipeline/config.py` itself rarely needs
edits.

### `PipelineResult`

`pipeline/loop.py` returns a single `PipelineResult` bundling every
sub-result plus the convergence history (CD0, mass, Sw, CG per
iteration). `pipeline/reporting.py` and `pipeline/plots.py` are
read-only consumers of this object.

### Airfoils and propellers

- `aerodynamics/airfoil_polar.py` shells out to XFOIL once per
  `(airfoil, Re, M, α-range)`, fits `Cl_α` and `α_L0`, and caches the
  full polar to `~/.cache/wing_sizing/airfoil_polar/`. Subsequent calls
  with the same key are free.
- `propulsion/propeller_solver.py` walks RPM against a propeller
  performance CSV (`data/*.csv`). `solve(...)` handles cruise/climb
  (advance ratio set by forward speed); `vtolsolve(...)` handles hover
  at J = 0. The propeller diameter can be parsed from the CSV filename
  (`20x10E_performance.csv` → 20 in) or set explicitly in `config.yaml`.

---

## The iterative loop

`pipeline/loop.py` runs the loop. One pass through the design is:

```
   wing.run  →  propulsion.run  →  fuselage.run  →  drag_buildup.run
        ↑                                                   │
        │                                                   ▼
   aileron.run ◄── rods.run  ◄──  weights.mass / cg  ◄──  CD0
        │
        ▼
   (loop body)
        │  feed back into SizingInputs:
        │     • Cd0     ← drag.CD0
        │     • b       ← span that puts op-CL at max L/D (if sw_closure)
        │     • m_drone ← masses["total"]                  (if mass_closure)
        └─► re-run wing.run with the updated inputs
```

The loop iterates until every active criterion is below tolerance:

| Criterion        | Always-on | Active when           | Tolerance     |
| ---------------- | --------- | --------------------- | ------------- |
| `|ΔCD0|`         | yes       | —                     | `cd0_tol`     |
| `|Δm_drone|`     | no        | `mass_closure: true`  | `mass_tol`    |
| `|ΔSw|`          | no        | `sw_closure: true`    | `sw_tol`      |

A second, faster propeller-solver iteration is nested inside step
"propulsion.run": for each segment (cruise, climb, hover) it walks RPM
against the propeller table until the table thrust matches the required
thrust within `thrust_tol`. The XFOIL polar lookup at step "drag /
LLT" is *not* iterative — the polar is computed once at the initial
sizing Re and reused, since the Re drift across iterations is small.

The two closures:

- **`mass_closure`** — `false` holds `m_drone_empty` at the requirement
  (the sizing target); `true` feeds the buildup mass back in, so the
  loop converges to a *self-consistent* empty mass instead of a target.
- **`sw_closure`** — `false` holds `b`/`AR` (and therefore `Sw`) at the
  config values; `true` resizes `Sw` each pass so the operating CL sits
  at the max L/D of the drone+payload polar. `AR` is held, so
  `b = sqrt(AR · Sw)` floats.

After convergence the loop runs one final pass so every sub-result
matches the converged sizing, then runs the LLT at the required CL and
sweeps α to build the wing drag polar shown in `plot_drone_ld`.

---

## Running `main.py`

```bash
uv run python main.py
```

Output sequence:
1. Airfoil + polar header lines.
2. Per-iteration log (`iter NN: CD0=… m_drone=… Sw=… …`).
3. `plot_convergence` window — 4 panes: CD0, mass, Sw, CG vs. iteration.
4. `print_main_summary` — sizing, propulsion, fuselage, control surfaces,
   structure, masses, CG, drag buildup, LLT at required CL, α-sweep.
5. `plot_drone_ld` window — L/D vs CL for drone-only and drone+payload.
6. `print_final_drag` — full-buildup CD breakdown.

To change anything, edit `config.yaml` and re-run — no source edits.

---

## Running individual modules

Every sub-solver module has a `__main__` block that runs the solver with
default upstream inputs and prints its `summary()`. Because they use
absolute imports (`from sizing.wing import SizingInputs`), they must be
invoked **as modules from the repo root**:

```bash
# Sizing
uv run python -m sizing.wing            # atmosphere, wing+tail geometry, CL/CD
uv run python -m sizing.fuselage        # fuselage box around the battery
uv run python -m sizing.aileron         # aileron span for the roll-rate spec

# Propulsion
uv run python -m propulsion.sizing      # cruise + climb + VTOL → battery + motor
uv run python propulsion/propeller_solver.py   # standalone RPM walk (uses its own constants)

# Aerodynamics
uv run python -m aerodynamics.airfoil_polar 2412 --Re 4e5 --alpha -5 15 0.5 --cache
uv run python -m aerodynamics.airfoil_polar airfoils/NACA4412.dat --Re 4e5 --cache
uv run python -m aerodynamics.airfoil_geometry airfoils/MH112.dat
uv run python -m aerodynamics.drag_buildup     # CD0 buildup with default sizing
uv run python -m aerodynamics.llt 2412 --AR 7.5 --b 3.0 --CL 0.6 --plot

# Structures
uv run python -m structures.rods        # wing + tail rod sizing

# Pipeline pieces (rarely useful standalone — main.py drives them)
uv run python -m pipeline.loop          # no __main__ block; use main.py instead
```

Notes:
- `propeller_solver.py` has its own top-level constants (`CSV_FILE`,
  `TARGET_THRUST`, …) and runs directly with `python propulsion/propeller_solver.py`.
  All other modules pull from sizing defaults.
- `airfoil_polar.py` and `llt.py` take CLI arguments via `argparse`; run
  with `--help` for the full list.
- The standalone runs use the dataclass defaults from each `*Inputs`,
  not `config.yaml`. Only `main.py` reads `config.yaml`.
- Caching for XFOIL: pass `--cache` (or `use_cache=True` from Python)
  to reuse polars across runs. The pipeline always uses the cache.

---

## Editing `config.yaml`

The file is grouped by sub-solver — each block matches one `*Inputs`
dataclass. The most common edits:

- **Mission / payload**: `sizing.m_payload`, `sizing.R`, `sizing.V_cruise`,
  `sizing.h_cruise`, `sizing.n_drones`.
- **Wing planform**: `sizing.b`, `sizing.AR`, `sizing.lam` (with
  `sw_closure: true`, `b` is overwritten by the loop and `AR` is held).
- **Airfoils**: top-level `airfoil` (wing) and `tail_airfoil` —
  either a `.dat` path under `airfoils/` or a 4/5-digit NACA string.
- **Propeller**: `propulsion.csv_prop` (path under `data/`) and
  `propulsion.D_prop` (or leave `null` to parse the diameter from the
  filename).
- **Loop**: `mass_closure`, `sw_closure`, `n_iter_max`, the three `*_tol`.

The header comment at the top of `config.yaml` explains the closures
and tolerances in detail.
