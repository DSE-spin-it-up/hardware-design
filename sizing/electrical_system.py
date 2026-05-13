from dataclasses import dataclass, fields

from sizing import initial_sizing
from sizing.initial_sizing import SizingResult, g0


@dataclass
class ElectricalInputs:
    # Propulsion
    n_props: int = 2             # Number of propellers [-]
    eff_motor: float = 0.8       # Motor efficiency [-]
    eff_prop: float = 0.7        # Non-ideal propeller efficiency [-]
    T_W_to: float = 2.0          # Thrust-to-weight ratio at take-off [-]
    J: float = 0.4               # Advance ratio [-]
    C_t: float = 0.04            # Thrust coefficient [-]
    D_prop: float = 20 * 0.0254  # Propeller diameter [m]
    prop_mass: float = 0.05      # Mass of a single propeller [kg]
    # Battery
    n_cells: int = 6
    voltage_cell: float = 3.7    # [V]
    battery_density: float = 250 # [Wh/L]
    eff_battery: float = 0.95    # Battery discharge efficiency [-]


@dataclass
class ElectricalResult:
    inputs: ElectricalInputs
    voltage_battery: float
    # Propulsion
    thrust_cruise: float
    thrust_to: float
    P_required: float
    thrust_prop: float
    thrust_total_prop: float
    C_p: float
    P_shaft: float
    KV: float
    motor_mass: float        # [kg] per motor
    total_motor_mass: float  # [kg]
    total_prop_mass: float   # [kg] all propellers combined
    # Energy & Battery
    I_battery: float
    P_battery: float
    E_cruise: float          # [J] electrical energy delivered during cruise
    E_battery: float         # [J] energy the battery must store (E_cruise / eff_battery)
    battery_mass: float      # [kg]
    battery_volume: float    # [m^3]


def motor_mass_from_kv(KV: float) -> float:
    """Linear motor-mass fit from KV rating; returns kg."""
    if KV < 100 or KV > 2000:
        raise ValueError(f"KV={KV} out of fit range [100, 2000]")
    if KV < 500:
        slope = (900 - 300) / (500 - 100)
        mass_g = 900 - (KV - 100) * slope
    else:
        slope = (300 - 50) / (2000 - 500)
        mass_g = 300 - (KV - 500) * slope
    return mass_g / 1000


def battery_mass_from_energy(E_cruise: float, n_cells: int, voltage_cell: float) -> float:
    """Battery mass [kg] from required cruise energy [J]."""
    voltage = n_cells * voltage_cell
    C = E_cruise * 1000 / (3600 * voltage)
    if n_cells == 6:
        a, b = 0.3988, 0.8810
    else:
        raise NotImplementedError(f"battery_mass fit not available for n_cells={n_cells}")
    mass_g = a * (C ** b)
    return mass_g / 1000


def battery_volume_from_energy(E_cruise: float, battery_density: float) -> float:
    """Battery volume [m^3] from required cruise energy [J].

    battery_density is in [Wh/L], so volume_L = E_Wh / battery_density.
    """
    volume_L = (E_cruise / 3600) / battery_density
    return volume_L / 1000


def run(
    sizing: SizingResult,
    inputs: ElectricalInputs | None = None,
) -> ElectricalResult:
    if inputs is None:
        inputs = ElectricalInputs()
    i = inputs
    s = sizing

    voltage_battery = i.n_cells * i.voltage_cell

    # Propulsion
    thrust_cruise = s.D
    thrust_to = i.T_W_to * (s.inputs.m_drone_empty * g0)
    P_required = s.D * s.inputs.V_cruise
    thrust_prop = i.C_t * s.rho * (s.inputs.V_cruise * i.D_prop / i.J) ** 2
    thrust_total_prop = thrust_prop * i.n_props
    C_p = (i.C_t / i.eff_prop) * i.J
    P_shaft = C_p * s.rho * ((s.inputs.V_cruise / i.J) ** 3) * (i.D_prop ** 2)
    KV = 60 * s.inputs.V_cruise / (i.J * i.D_prop * voltage_battery)

    motor_mass = motor_mass_from_kv(KV)
    total_motor_mass = motor_mass * i.n_props
    total_prop_mass = i.prop_mass * i.n_props

    # Energy & Battery
    I_battery = (P_shaft * i.n_props) / voltage_battery
    P_battery = voltage_battery * I_battery
    E_cruise = P_battery * s.t_cruise
    # Battery must store more than it delivers: account for discharge efficiency.
    E_battery = E_cruise / i.eff_battery
    battery_mass = battery_mass_from_energy(E_battery, i.n_cells, i.voltage_cell)
    battery_volume = battery_volume_from_energy(E_battery, i.battery_density)

    return ElectricalResult(
        inputs=inputs,
        voltage_battery=voltage_battery,
        thrust_cruise=thrust_cruise,
        thrust_to=thrust_to,
        P_required=P_required,
        thrust_prop=thrust_prop,
        thrust_total_prop=thrust_total_prop,
        C_p=C_p,
        P_shaft=P_shaft,
        KV=KV,
        motor_mass=motor_mass,
        total_motor_mass=total_motor_mass,
        total_prop_mass=total_prop_mass,
        I_battery=I_battery,
        P_battery=P_battery,
        E_cruise=E_cruise,
        E_battery=E_battery,
        battery_mass=battery_mass,
        battery_volume=battery_volume,
    )


def summary(r: ElectricalResult) -> None:
    if r.thrust_total_prop > r.thrust_cruise:
        print(f"Thrust margin of {r.thrust_total_prop - r.thrust_cruise:.2f} [N] available.")
    else:
        print(f"Additional {r.thrust_cruise - r.thrust_total_prop:.2f} [N] required")

    print("\n--- Propulsion ---")
    print(f"  Cruise thrust        : {r.thrust_cruise:.2f}  N")
    print(f"  Total prop. thrust   : {r.thrust_total_prop:.2f}  N")
    print(f"  Take-off thrust      : {r.thrust_to:.2f}  N")
    print(f"  Cruise power required: {r.P_required:.2f}  W")
    print(f"  Number of propellers : {r.inputs.n_props}")
    print(f"  Propeller Diameter   : {r.inputs.D_prop:.2f}  m")
    print(f"  Propeller Mass (each): {r.inputs.prop_mass:.3f}  kg")
    print(f"  Total Propeller Mass : {r.total_prop_mass:.3f}  kg")
    print(f"  Shaft Power          : {r.P_shaft:.2f}  W")
    print(f"  KV Motor             : {r.KV:.2f}  RPM/V")
    print(f"  Battery Current      : {r.I_battery:.2f}  A")
    print(f"  Battery Mass         : {r.battery_mass:.2f}  kg")
    print(f"  Total Motor Mass     : {r.total_motor_mass:.2f}  kg")

    print("\n--- Energy ---")
    print(f"  Cruise energy        : {r.E_cruise / 3600:.2f}  Wh")
    print(f"  Battery efficiency   : {r.inputs.eff_battery:.3f}")
    print(f"  Battery stored energy: {r.E_battery / 3600:.2f}  Wh")
    print(f"  Battery volume       : {r.battery_volume * 1000:.2f}  L")


# Default run at module load — exposes inputs+result fields as module attributes.
_default = run(initial_sizing._default)
for _f in fields(ElectricalInputs):
    globals()[_f.name] = getattr(_default.inputs, _f.name)
for _f in fields(ElectricalResult):
    if _f.name != "inputs":
        globals()[_f.name] = getattr(_default, _f.name)


if __name__ == "__main__":
    summary(_default)
