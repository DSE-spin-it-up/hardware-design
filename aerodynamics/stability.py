
from aerodynamics.airfoil_polar import AirfoilPolar
from aerodynamics.llt import WingGeometry


def calculate_CM_wing(inputs : AirfoilPolar, wing_geometry : WingGeometry) -> float:
    CM_wing = inputs.Cm * wing_geometry.AR / (2 + wing_geometry.AR)



    
