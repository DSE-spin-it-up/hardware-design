class Material:
    def __init__(self, E, rho, Y, s_t, s_c = None):
        self.E = E  # Pa
        self.rho = rho  # kg/m^3
        self.Y = Y  # Pa
        self.s_t = s_t  # Pa
        self.s_c = s_c

    def mass(self, volume: float) -> float:
        """Compute mass from a volume using this material density."""
        return self.rho * volume

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rho={self.rho:.1f} kg/m^3)"

class CFRP(Material):  # https://www.researchgate.net/publication/342938721_Experimental_Investigation_of_Reinforced_Concrete_Beam_with_Openings_Strengthened_Using_FRP_Sheets_under_Cyclic_Load
    def __init__(self):
        super().__init__(
            E=230e9,
            rho=1.72e3,
            Y=3400e6,
            s_t=3400e6,
            s_c=600e6
        )

class EPP(Material):  # https://www.foambymail.com/polypropylene-foam-sheet.html?srsltid=AfmBOorBTKilq9ebl6LLzYihjew-KWV77s8RLA2MVI6mPx15FnjyM8NA
    def __init__(self):
        super().__init__(
            E=5*262e3,
            rho=20.824002386148,
            Y=262e3,
            s_t=262e3
        )

class CF_PLA(Material):  # https://www.iemai3d.com/wp-content/uploads/2020/12/CF-PLA_TDS.pdf
    def __init__(self):
        super().__init__(
            E=4950e6,
            rho=1.29e3,
            Y=48e6,
            s_t=48e6
        )

class PLA(Material):  # https://www.sciencedirect.com/science/article/pii/S0169409X16302058#s0010
    def __init__(self):
        super().__init__(
            E=3.5e9,
            rho=1.252e3,
            Y=70e6,
            s_t=59e6,
        )

class Wood(Material):  # https://www.matweb.com/search/datasheet.aspx?matguid=1e56abdf98904f2ca53bff4bd1250cab&ckck=1
    def __init__(self):
        super().__init__(
            E=8.14e9,
            rho=360,
            Y=20.7e6,
            s_t=1.59e6,
            s_c=2.28e6
        )

class Glass_Fiber(Material):  # https://www.matweb.com/search/datasheet.aspx?MatGUID=8f9003366c9044bdb91bcd86e1fa6e42
    def __init__(self):
        super().__init__(
            E=68.9e9,
            rho=200/1000,
            Y=3310e6,
            s_t=3310e6
        )