"""
fh_corollaries_cas.py

Flux Holography – Corollaries and Structural Scales (CAS 5)

This module collects *derived* consequences of the FH backbone
(EAL, Flux Law, UTL, UAL, Θ, k_SEG) into one place, matching and
recycling the content of the original CAS 1 and CAS 2:

- Bekenstein entropy shift
- Entropic inertia F = m a
- 8π inertia identity (Compton step → 8π ℓ_P²)
- Thermal / Compton / Planck scale family (χ_th = 1/π)
- Cosmological structural scales:
    - Hubble “quantum” E = ħ H / 2
    - Minimum inertial mass m_min ~ ħ H / (2 c²)
    - Maximum cosmic mass m_max = c³ / (2 G H)
    - Planck crossover mass (geometric mean)
    - Dark-energy ratio ~ (H ℓ_P / c)²
- Tick counter and entropy–time duality
- GSL inequality scaffold
- Mass–time dictionary τ_M = k_SEG M and Schwarzschild tick t*(M)
- (MODEL) entropy-arrow rate  Ṡ ≈ (π/2) k_B H

All of these are *corollaries*, not postulates.

IMPORTANT (integral vs product X):

  The *fundamental* FH flux is defined as the line integral

      X = ∫ t*(dE − Ω_H dJ − Φ_H dQ_el)

  (or X = A/k_SEG in the exact 1-form sector on stationary horizons).

  Any local use of a product E t* in this module is a
  single-channel, fixed-tick *toy* within that broader integral
  framework and must NOT be treated as the global definition of X.
"""

import sympy as sp

# ---------------------------------------------------------------------
# Constants and basic scales
# ---------------------------------------------------------------------

G, c, hbar, kB = sp.symbols("G c hbar k_B", positive=True, real=True, nonzero=True)
pi = sp.pi

ellP2 = G*hbar/c**3
ellP  = sp.sqrt(ellP2)

kSEG = 4*pi*G/c**3

# ---------------------------------------------------------------------
# 1. Scale families: thermal, Compton, Planck (Consistency A)
# ---------------------------------------------------------------------

# thermal
T_th = sp.symbols("T_th", positive=True)     # generic temperature [K]
t_th = hbar/(pi*kB*T_th)                     # thermal tick [s] from UTL
lam_th = c*t_th                              # thermal length [m]

# Compton
m = sp.symbols("m", positive=True)           # mass [kg]
lam_C = hbar/(m*c)                           # Compton length [m]
t_C   = lam_C/c                              # Compton time [s]

# Planck
t_P = ellP/c                                 # Planck time [s]
T_P = hbar/(kB*t_P)                          # Planck temperature [K]

# dimensionless combo χ_th = k_B T t_th / ħ = 1/π
chi_th = sp.simplify((kB*T_th*t_th)/hbar)
chi_th_minus_1_over_pi = sp.simplify(chi_th - 1/pi)

# ---------------------------------------------------------------------
# 2. Bekenstein shift and entropic inertia (FH III / ConsC)
# ---------------------------------------------------------------------

E_sym, dx = sp.symbols("E_sym dx", positive=True)

def bekenstein_shift(E: sp.Expr, delta_x: sp.Expr) -> sp.Expr:
    """
    ID: BEKENSTEIN_SHIFT
    Role: DERIVED

    Bekenstein entropy shift (near-horizon, single channel):

        ΔS = (2π k_B /(ħ c)) E Δx
    """
    return (2*pi*kB/(hbar*c))*E*delta_x

# Unruh temperature
a = sp.symbols("a", positive=True)
T_Unruh = hbar*a/(2*pi*c*kB)

# Entropic inertia: F = T ΔS/Δx with ΔS = 2π k_B, Δx = ħ/(m c)
delta_x_inertia = hbar/(m*c)
delta_S_inertia = 2*pi*kB
F_entropic = sp.simplify(T_Unruh*delta_S_inertia/delta_x_inertia)
F_entropic_minus_ma = sp.simplify(F_entropic - m*a)

# ---------------------------------------------------------------------
# 3. 8π inertia identity (Compton step → 8π ℓ_P²)
# ---------------------------------------------------------------------

def eight_pi_inertia_deltaA(m_sym: sp.Expr) -> sp.Expr:
    """
    ID: EIGHT_PI_INERTIA
    Role: DERIVED (toy single-channel sector)

    Single-channel *toy* realization of the FH Flux Law near the Compton scale.

    In a local inertial patch we take:
      - an effective energy jump    ΔE = m c²,
      - a local reversible tick     t*_step = 2 ħ/(m c²)
        (two Compton times, consistent with UTL once T is chosen),

    and plug these into the local Flux Law:

        ΔA = k_SEG t*_step ΔE.

    Using k_SEG = 4πG/c³ and ℓ_P² = Għ/c³ this gives

        ΔA = 8π ℓ_P²,

    independent of m. Conceptually: one Compton-scale inertial “step”
    sweeps 8π Planck pixels in this toy model.

    IMPORTANT:
      - This does NOT redefine the global flux X.
      - The fundamental FH flux remains the integral
            X = ∫ t*(dE − Ω dJ − Φ dQ)
        on horizons; here we are using a fixed-tick, single-channel
        proxy with t*_step in a local inertial sector.
    """
    t_star_step = 2*hbar/(m_sym*c**2)   # local reversible tick for this toy step
    deltaE = m_sym*c**2
    deltaA = kSEG * t_star_step * deltaE
    return sp.simplify(deltaA)

eight_pi_deltaA = eight_pi_inertia_deltaA(m)
eight_pi_ratio = sp.simplify(eight_pi_deltaA/(8*pi*ellP2))

# ---------------------------------------------------------------------
# 4. Cosmological structural scales (ConsB / ConsD)
# ---------------------------------------------------------------------

H = sp.symbols("H", positive=True)  # Hubble parameter [1/s]

def rho_crit_mass(H_sym: sp.Expr) -> sp.Expr:
    """
    ID: RHO_CRIT_MASS
    Role: ASSUMED

    ρ_crit = 3 H² /(8π G)   [kg/m³]
    """
    return 3*H_sym**2/(8*pi*G)

def rho_crit_energy(H_sym: sp.Expr) -> sp.Expr:
    """
    ID: RHO_CRIT_ENERGY
    Role: DERIVED

    ε_crit = ρ_crit c²      [J/m³]
    """
    return rho_crit_mass(H_sym)*c**2

rho_m = rho_crit_mass(H)
rho_e = rho_crit_energy(H)
rho_energy_vs_mass_ratio = sp.simplify(rho_e/(rho_m*c**2))

# Hubble horizon quantum
def hubble_horizon_quantum(H_sym: sp.Expr) -> sp.Expr:
    """
    ID: HUBBLE_QUANTUM
    Role: DERIVED

    Hubble horizon "quantum":

        E_H = ħ H / 2   [J]
    """
    return hbar*H_sym/2

E_H = hubble_horizon_quantum(H)
E_H_minus_def = sp.simplify(E_H - hbar*H/2)

# Minimum inertial mass, maximum cosmic mass, Planck crossover
def min_inertial_mass(H_sym: sp.Expr) -> sp.Expr:
    """
    ID: M_MIN
    Role: DERIVED

    Minimum inertial mass:

        m_min ≈ ħ H /(2 c²)
    """
    return hbar*H_sym/(2*c**2)

def max_cosmic_mass(H_sym: sp.Expr) -> sp.Expr:
    """
    ID: M_MAX
    Role: DERIVED

    Maximum cosmic mass:

        m_max = c³ /(2 G H)
    """
    return c**3/(2*G*H_sym)

def planck_mass_crossover(H_sym: sp.Expr) -> sp.Expr:
    """
    ID: M_CROSS
    Role: DERIVED

    Planck crossover mass:

        m_cross = sqrt(m_min m_max)
    """
    m_min = min_inertial_mass(H_sym)
    m_max = max_cosmic_mass(H_sym)
    return sp.sqrt(m_min*m_max)

m_min_H = min_inertial_mass(H)
m_max_H = max_cosmic_mass(H)
m_cross_H = planck_mass_crossover(H)

# Dark-energy ratio (dimensionless)
def dark_energy_ratio(H_sym: sp.Expr) -> sp.Expr:
    """
    ID: DARK_ENERGY_RATIO
    Role: DERIVED

    Structural dark-energy ratio:

        ρ_eff / ρ_P ~ (H ℓ_P / c)²
    """
    return (H_sym*ellP/c)**2

dark_ratio = dark_energy_ratio(H)
dark_ratio_minus_def = sp.simplify(dark_ratio - (H*ellP/c)**2)

# ---------------------------------------------------------------------
# 5. Tick counter, entropy-time duality, Landauer mapping, GSL
# ---------------------------------------------------------------------

S = sp.symbols("S", positive=True)

def tick_counter_from_entropy(S_sym: sp.Expr) -> sp.Expr:
    """
    ID: TICK_COUNTER
    Role: DERIVED

    Tick counter from entropy:

        N = S /(π k_B)
    """
    return sp.simplify(S_sym/(pi*kB))

N_ticks = tick_counter_from_entropy(S)

def entropy_time_duality_per_tick():
    """
    ID: ENTROPY_TIME_DUALITY
    Role: DERIVED

    Each reversible tick quantum in FH carries:

        action quantum      = ħ
        entropy quantum     = π k_B

    The pair (ħ, π k_B) is the basic action–entropy packet
    associated with one unit of X in the EAL normalization.
    """
    return hbar, pi*kB

# GSL scaffold
S_hor_i, S_hor_f = sp.symbols("S_hor_i S_hor_f", real=True)
S_mat_i, S_mat_f = sp.symbols("S_mat_i S_mat_f", real=True)

GSL_ineq = sp.Ge(S_hor_f + S_mat_f, S_hor_i + S_mat_i)

# ---------------------------------------------------------------------
# 6. Mass–time dictionary and entropy-arrow (FH-VI extensions)
# ---------------------------------------------------------------------

def tau_M_from_mass(M_sym: sp.Expr) -> sp.Expr:
    """
    ID: TAU_M_FROM_MASS
    Role: DERIVED

    Mass–time dictionary (SEG form):

        τ_M = k_SEG * M

    τ_M is a time scale associated with mass M via the
    same coupling that appears in the Flux Law. This is a
    convenient helper; the fundamental tick still comes
    from UTL or the horizon kinematics.
    """
    return kSEG * M_sym

def t_star_schwarzschild_from_mass(M_sym: sp.Expr) -> sp.Expr:
    """
    ID: T_STAR_SCHW_FROM_MASS
    Role: DERIVED

    Schwarzschild calibrated tick from mass:

        t*(M) = 8 G M / c³

    which is also

        t*(M) = (2/π) τ_M

    once τ_M = k_SEG M with k_SEG = 4πG/c³ is used.
    """
    tau_M = tau_M_from_mass(M_sym)
    return sp.simplify((2/pi)*tau_M)

def entropy_arrow_rate(H_sym: sp.Expr) -> sp.Expr:
    """
    ID: ENTROPY_ARROW_RATE
    Role: MODEL

    MODEL entropy-arrow rate (cosmological coarse-grain):

        Ṡ ≈ (π/2) k_B H

    This is a phenomenological mapping from H to a global
    entropy production rate. It is NOT a postulate of FH
    and should be treated as a modeling ansatz in cosmological
    applications.
    """
    return (pi/2)*kB*H_sym

# ---------------------------------------------------------------------
# 7. Corollary verification
# ---------------------------------------------------------------------

def verify_corollaries():
    """
    Returns key -> expression that should simplify to 0 or 1.
    """
    checks = {}

    # scale family
    checks["CHI_TH_MINUS_1_OVER_PI"] = chi_th_minus_1_over_pi

    # entropic inertia
    checks["F_ENTROPIC_MINUS_MA"] = F_entropic_minus_ma

    # 8π inertia identity
    checks["EIGHT_PI_RATIO_MINUS_1"] = sp.simplify(eight_pi_ratio - 1)

    # cosmology: ε_crit vs ρ_crit
    checks["RHO_ENERGY_VS_MASS_MINUS_1"] = sp.simplify(rho_energy_vs_mass_ratio - 1)

    # Hubble quantum
    checks["HUBBLE_QUANTUM_DEF"] = E_H_minus_def

    # dark-energy ratio definition (structural)
    checks["DARK_ENERGY_RATIO_DEF"] = dark_ratio_minus_def

    # mass–time dictionary consistency (nonzero, but dimensional check is implicit)
    # We do not expect these to be 0; they are helpers, so we don't include
    # tau_M_from_mass or entropy_arrow_rate here.

    return checks


def all_corollary_checks_pass() -> bool:
    """
    True iff all symbolic checks simplify to exactly 0 as expected.
    """
    checks = verify_corollaries()
    ok = True
    for key, expr in checks.items():
        ok = ok and (sp.simplify(expr) == 0)
    return ok


if __name__ == "__main__":
    print("=== Flux Holography – CAS 5 (Corollaries) ===")
    checks = verify_corollaries()
    for k, v in checks.items():
        print(f"{k:32s}: {sp.simplify(v)}")
    print()
    print("ALL COROLLARY CHECKS PASS:", all_corollary_checks_pass())