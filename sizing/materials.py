class Material:
    def __init__(self, E, rho, Y, s_t):
        self.E = E
        self.rho = rho
        self.Y = Y
        self.s_t = s_t

class CFRP(Material):  # https://www.researchgate.net/publication/342938721_Experimental_Investigation_of_Reinforced_Concrete_Beam_with_Openings_Strengthened_Using_FRP_Sheets_under_Cyclic_Load
    def __init__(self):
        super().__init__(
            E=230e9,
            rho=1.72e3,
            Y=3400e6,
            s_t=3400e6
        )

class EPP(Material):  # https://www.foambymail.com/polypropylene-foam-sheet.html?srsltid=AfmBOorBTKilq9ebl6LLzYihjew-KWV77s8RLA2MVI6mPx15FnjyM8NA
    def __init__(self):
        super().__init__(
            E=230e9,
            rho=20.824002386148,
            Y=262e3,
            s_t=262e3
        )

class CF_PLA(Material):  # https://www.iemai3d.com/wp-content/uploads/2020/12/CF-PLA_TDS.pdf
    def __init__(self):
        super().__init__(
            E=4950e6,
            rho=1.29e3,
            Y= 48e6,
            s_t = 48e6
        )