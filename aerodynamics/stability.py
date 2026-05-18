
from aerodynamics.airfoil_polar import AirfoilPolar
from aerodynamics.llt import WingGeometry
from sizing.wing import SizingResult

L_tail=SizingResult.L_tail
ARt=

def calculate_CM_wing(inputs : AirfoilPolar, wing_geometry : WingGeometry, ) -> float:
    CM_wing = inputs.Cm * wing_geometry.AR / (2 + wing_geometry.AR)

def calculate_CL_tail(inputs : AirfoilPolar, tail_geometry : WingGeometry) -> float:
    CL_tail = CM
    return CL_tail



    
