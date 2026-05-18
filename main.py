"""Design pipeline entry point.

Edit the design knobs in `config.yaml` and run `python main.py`. Pipeline:
  1. Initial sizing with a guessed wing Cd0
  2. Battery + propulsion sizing
  3. Re-estimate Cd0 from fuselage (PLACEHOLDER — teammate)
  4. Re-run sizing + propulsion with the refined Cd0
  5. Pick airfoil → LLT solve at the required wing CL
  6. Wing CL/CD vs CL sweep; check whether operating CL sits near max L/D
  7. Full-buildup CD / final L/D (PLACEHOLDER — teammate)
"""
from __future__ import annotations

from pipeline import config, loop, plots, reporting


def main() -> None:
    result = loop.run_pipeline(config)
    plots.plot_convergence(
        result.cd0_history, result.mass_history, result.sw_history, result.cg_history,
        result.cd0_guess, result.m_drone_guess, result.sw_guess,
    )
    reporting.print_main_summary(result)
    plots.plot_cg_side_view(
        sizing=result.sizing,
        fus=result.fus,
        struct=result.struct,
        cg=result.cg,
        masses=result.masses,
        airfoil_path=result.airfoil,
    )
    plots.plot_drone_ld(
        result.cl_sweep, result.cd_drone_sweep, result.cd_full_sweep,
        CL_op=result.cl_req, airfoil_name=result.polar.name,
    )
    plots.plot_scissor(result.scissor)
    reporting.print_final_drag(result)


if __name__ == "__main__":
    main()
