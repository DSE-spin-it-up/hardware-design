"""Hinge moment for a plain-flap control surface.

Reference
---------
Márquez et al., "Control surface design for radio-controlled aircraft",
Revista Facultad de Ingeniería, Universidad de Antioquia, No. 104, 2022.
Equations 37–38.

    Chi  = Ch0 + Ch_alpha·α + Ch_delta·δ          [Eq. 38]
    H    = ½·ρ·V²·S_control·c_control·Chi         [Eq. 37]

The three hinge-moment coefficients (Ch0, Ch_alpha, Ch_delta) are
designer inputs with conservative plain-flap defaults.  A future
extension can replace the defaults with values computed from thin
airfoil theory (NACA TR-688 / Roskam Part VI) using only the chord
ratio cf/c, which is already available for every surface.
"""
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HingeMomentInputs:
    """Hinge-moment coefficients for one control surface.

    Defaults are typical values for a plain flap (symmetric airfoil,
    no aerodynamic balance).  Override with XFLR5 / thin-airfoil-theory
    values when available.

    Ch0      : hinge moment at α = 0, δ = 0  [-]
    Ch_alpha : dCh/dα                         [1/rad]  (negative for plain flap)
    Ch_delta : dCh/dδ                         [1/rad]  (negative for plain flap)
    """
    Ch0:      float = 0.00   # zero for symmetric unloaded surface
    Ch_alpha: float = -0.10  # typical plain flap [1/rad]
    Ch_delta: float = -0.30  # typical plain flap [1/rad]


@dataclass
class HingeMomentResult:
    """Output of a single hinge-moment evaluation."""
    Chi:  float   # hinge-moment coefficient [-]
    H:    float   # hinge moment [N·m]
    # inputs echoed for traceability
    alpha:   float   # angle of attack used [rad]
    delta:   float   # control deflection used [rad]
    q:       float   # dynamic pressure used [Pa]
    S_ctrl:  float   # control surface planform area [m²]
    c_ctrl:  float   # control surface chord [m]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute(
    alpha:   float,
    delta:   float,
    q:       float,
    S_ctrl:  float,
    c_ctrl:  float,
    inputs:  HingeMomentInputs | None = None,
) -> HingeMomentResult:
    """Compute the aerodynamic hinge moment for a plain-flap surface.

    Parameters
    ----------
    alpha   : angle of attack of the surface [rad]
    delta   : control-surface deflection (positive convention per surface) [rad]
    q       : dynamic pressure ½·ρ·V²  [Pa]
    S_ctrl  : control-surface planform area  [m²]
    c_ctrl  : control-surface chord          [m]
    inputs  : HingeMomentInputs; defaults used when None

    Returns
    -------
    HingeMomentResult with Chi and H.
    """
    if inputs is None:
        inputs = HingeMomentInputs()
    i = inputs

    Chi = i.Ch0 + i.Ch_alpha * alpha + i.Ch_delta * delta
    H   = q * S_ctrl * c_ctrl * Chi          # ½ρV² is already in q

    return HingeMomentResult(
        Chi    = Chi,
        H      = H,
        alpha  = alpha,
        delta  = delta,
        q      = q,
        S_ctrl = S_ctrl,
        c_ctrl = c_ctrl,
    )