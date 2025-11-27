"""
fh_horizons_cosmo_cas.py

Flux Holography – Horizons & Cosmology CAS.

This module attaches the Flux Holography (FH) core constitutive structure
to concrete spacetime families:

  - Schwarzschild horizons
  - de Sitter / cosmological horizons
  - FRW cosmology and effective/critical densities

It is designed for LLMs and symbolic tools that already know the FH core
(fh_core_cas.py). It provides:

  * Exact identities such as
      X(M) = ∫ t*(M) dE(M) = A(M)/k_SEG
    for Schwarzschild: the primitive flux is defined by the line integral
    and, on stationary exact families, equals A/k_SEG.

  * Bekenstein–Hawking entropy checks on black-hole and de Sitter horizons.

  * Cosmological identities:
      ρ_eff = 3 H² c² /(8π G)  (horizon effective energy density),
      Λ   = 3 H² / c²         (pure de Sitter),
    and their mapping to standard critical density language.

  * Convenience helpers (all using the calibrated tick t* = 2c/κ or its
    FRW/Rindler realizations):
      - FRW tick:    t*_FRW(H)      = 2/H
      - Rindler tick t*_Rindler(x)  = 2 x / c
      - Rindler T:   T_Rindler(x)   = ħ c / (2π x k_B)
      - FRW accel:   a¨/a = - (4πG/3)(ρ + 3p/c²) + Λ c²/3
      - BH triad:    consistency of κ, r_s, t* for Schwarzschild

Conventions:
  - We keep G, c, ħ, k_B explicit (imported from fh_core_cas).
  - We distinguish clearly:
      rho_mass   : mass density [kg/m³]
      rho_energy : energy density [J/m³]
      rho_eff    : effective horizon energy density [J/m³]
"""

from typing import Dict

import sympy as sp

from fh_core_cas import (
    Role,
    EquationMeta,
    G,
    c,
    hbar,
    kB,
    pi,
    ellP2,
    kSEG,
    S_BH_standard,
)

# ---------------------------------------------------------------------------
# NOTE ON FLUX X IN THIS MODULE
# ---------------------------------------------------------------------------
#
# The FUNDAMENTAL FH flux is defined globally as the line integral
#
#   X  =  ∫ t* (dE − Ω dJ − Φ dQ)
#
# over an admissible, quasi-stationary horizon variation, with a fixed
# screen and time-flow. In the exact 1-form sector of stationary families
# (Schwarzschild, Kerr–Newman, de Sitter / FRW horizons), the horizon
# first law and the calibrated tick t* = 2c/κ imply
#
#   t* (dE − Ω dJ − Φ dQ) = (1 / k_SEG) dA
#   ⇒ X = A / k_SEG + const.
#
# In this CAS:
#
#   - schwarzschild_flux_X_from_integral(M) computes X(M) from the integral
#     definition ∫ t*(M') dE(M') using the calibrated t*(M).
#   - schwarzschild_flux_X_from_area(M) computes A(M)/k_SEG.
#   - schwarzschild_X_identity_expr() checks that
#         X(M) (integral) − A(M)/k_SEG = 0
#     which is the explicit realization of the exact-1-form statement
#     for the Schwarzschild family.
#
# The FRW and de Sitter helpers use the same calibrated tick dictionary:
#   - de Sitter/GH:     k_B T = ħ H / (2π)
#   - UTL:              T t* = ħ/(π k_B) ⇒ t* = 2/H
#   - Rindler:          T(x) = ħ c/(2π x k_B), t*(x) = 2x/c.
#
# No naive global product X = E t* is assumed here; wherever a product
# appears it is either:
#   (i)   the explicit evaluation of the integral on a stationary family
#         where t* is constant along the path, or
#   (ii)  a convenience dictionary derived from the UTL and GH/Unruh input.
#

# ---------------------------------------------------------------------------
# Symbols shared across sectors
# ---------------------------------------------------------------------------

# Mass, radius, Hubble parameter, cosmological constant, pressure, distance
M = sp.symbols("M", positive=True, nonzero=True)          # mass parameter
r = sp.symbols("r", positive=True, nonzero=True)          # radius
H = sp.symbols("H", positive=True)                        # Hubble rate
Lambda = sp.symbols("Lambda", real=True)                  # cosmological constant
p = sp.symbols("p", real=True)                            # pressure
x = sp.symbols("x", positive=True)                        # proper distance (Rindler)

# Densities
rho_mass = sp.symbols("rho_mass", real=True)              # [kg/m³]
rho_energy = sp.symbols("rho_energy", real=True)          # [J/m³]


# ---------------------------------------------------------------------------
# 1. Schwarzschild horizon module
# ---------------------------------------------------------------------------

def schwarzschild_radius(M_sym: sp.Expr) -> sp.Expr:
    """
    ID: SCHW_RADIUS
    Role: ASSUMED (standard GR result)

    Schwarzschild radius:
        r_s = 2 G M / c²
    """
    return 2 * G * M_sym / c**2


def schwarzschild_area(M_sym: sp.Expr) -> sp.Expr:
    """
    ID: SCHW_AREA
    Role: ASSUMED

    Horizon area:
        A(M) = 4π r_s²
    """
    r_s = schwarzschild_radius(M_sym)
    return 4 * pi * r_s**2


def schwarzschild_surface_gravity(M_sym: sp.Expr) -> sp.Expr:
    """
    ID: SCHW_KAPPA
    Role: ASSUMED

    Surface gravity:
        κ(M) = c⁴ / (4 G M)
    """
    return c**4 / (4 * G * M_sym)


def schwarzschild_hawking_temperature(M_sym: sp.Expr) -> sp.Expr:
    """
    ID: SCHW_T_HAWKING
    Role: DERIVED (from κ + Hawking relation)

    Hawking temperature (KMS / Unruh–Hawking):

        k_B T = ħ κ /(2π c)
    """
    kappa = schwarzschild_surface_gravity(M_sym)
    return hbar * kappa / (2 * pi * c * kB)


def schwarzschild_tick_from_mass(M_sym: sp.Expr) -> sp.Expr:
    """
    ID: SCHW_TICK
    Role: DERIVED

    Reversible horizon tick for Schwarzschild, using the calibrated tick:

        t*(M) = 2c/κ = 8 G M / c³

    This is the same t* that appears in the integral flux

        X = ∫ t*(M') dE(M')
    """
    kappa = schwarzschild_surface_gravity(M_sym)
    return 2 * c / kappa


def schwarzschild_flux_X_from_integral(M_sym: sp.Expr) -> sp.Expr:
    """
    ID: SCHW_X_INTEGRAL
    Role: DERIVED

    Integral definition of the primitive flux for Schwarzschild:

        E(M)   = M c²
        t*(M)  = 8 G M / c³
        X(M)   = ∫_0^M t*(M') dE(M')

    This is the explicit evaluation of the global flux definition
    X = ∫ t* dE on the 1-parameter Schwarzschild family with fixed tick
    dictionary t*(M).
    """
    Mprime = sp.symbols("Mprime", positive=True)
    t_star_M = schwarzschild_tick_from_mass(Mprime)
    dE_dM = c**2
    integrand = t_star_M * dE_dM
    X_int = sp.integrate(integrand, (Mprime, 0, M_sym))
    return sp.simplify(X_int)


def schwarzschild_flux_X_from_area(M_sym: sp.Expr) -> sp.Expr:
    """
    ID: SCHW_X_FROM_AREA
    Role: DERIVED

    Area-based flux using the exact 1-form identity on stationary families:

        X = A(M) / k_SEG
    """
    A_M = schwarzschild_area(M_sym)
    return sp.simplify(A_M / kSEG)


def schwarzschild_X_identity_expr() -> sp.Expr:
    """
    Returns:
        X_int(M) - A(M)/k_SEG
    which should simplify to 0.

    This encodes the exact-1-form statement ω = d(A/k_SEG) for the
    Schwarzschild family when evaluated with the calibrated tick.
    """
    X_int = schwarzschild_flux_X_from_integral(M)
    X_area = schwarzschild_flux_X_from_area(M)
    return sp.simplify(X_int - X_area)


def schwarzschild_S_BH_via_FH(M_sym: sp.Expr) -> sp.Expr:
    """
    ID: SCHW_S_BH_FH
    Role: DERIVED

    Bekenstein–Hawking entropy for Schwarzschild via FH:

        S = S_BH_standard(A(M))
    """
    A_M = schwarzschild_area(M_sym)
    return sp.simplify(S_BH_standard(A_M))


def schwarzschild_S_BH_ratio_expr() -> sp.Expr:
    """
    Returns the symbolic entropy per area ratio for Schwarzschild:
        (S_BH / A) - (k_B / (4 ℓ_P²))
    which should simplify to 0.
    """
    A_M = schwarzschild_area(M)
    S_M = schwarzschild_S_BH_via_FH(M)
    ratio = sp.simplify(S_M / A_M)
    return sp.simplify(ratio - kB / (4 * ellP2))


def komar_energy_from_kappa_area(
    kappa_sym: sp.Expr,
    A_sym_local: sp.Expr,
    mode: str = "Komar1959",
) -> sp.Expr:
    """
    ID: KOMAR_ENERGY
    Role: ASSUMED / DERIVED (standard GR, two conventions)

      mode = "Komar1959": E = c² κ A /(8π G)   (half Smarr for Schwarzschild)
      mode = "ADM"      : E = c² κ A /(4π G)   (equals Smarr)

    FH observables depend only on variations, so this normalization
    ambiguity does not affect δE_eff or δX.
    """
    if mode == "Komar1959":
        denom = 8 * pi * G
    elif mode == "ADM":
        denom = 4 * pi * G
    else:
        raise ValueError('mode must be "Komar1959" or "ADM".')
    return c**2 * kappa_sym * A_sym_local / denom


def smarr_energy_from_T_and_S(T_sym: sp.Expr, S_sym: sp.Expr) -> sp.Expr:
    """
    ID: SMARR_SCHW
    Role: DERIVED

    Smarr relation for Schwarzschild:
        E = 2 T S
    """
    return 2 * T_sym * S_sym


def schwarzschild_komar_smarr_ratios_exprs() -> Dict[str, sp.Expr]:
    """
    ID: SCHW_KOMAR_SMARR_RATIOS
    Role: DERIVED

    Returns ratios:
      ratio_1959 = E_K(1/8π) / E_Smarr  → 1/2
      ratio_ADM  = E_K(1/4π) / E_Smarr  → 1
    expressed symbolically in terms of κ and A.
    """
    kappa = sp.symbols("kappa", positive=True)
    A_loc = sp.symbols("A_loc", positive=True)

    T_loc = hbar * kappa / (2 * pi * c * kB)
    S_loc = S_BH_standard(A_loc)
    E_smarr = smarr_energy_from_T_and_S(T_loc, S_loc)

    E_K_1959 = komar_energy_from_kappa_area(kappa, A_loc, mode="Komar1959")
    E_K_ADM = komar_energy_from_kappa_area(kappa, A_loc, mode="ADM")

    ratio_1959 = sp.simplify(E_K_1959 / E_smarr)
    ratio_ADM = sp.simplify(E_K_ADM / E_smarr)

    return {
        "ratio_komar1959_vs_smarr": sp.simplify(ratio_1959),
        "ratio_komarADM_vs_smarr": sp.simplify(ratio_ADM),
    }


def schwarzschild_bh_triad_of_unity(kappa_sym: sp.Expr, r_s_sym: sp.Expr) -> Dict[str, sp.Expr]:
    """
    ID: SCHW_BH_TRIAD
    Role: DERIVED

    Consistency triad for Schwarzschild (calibrated tick dictionary):

      - Surface gravity from radius:
          κ_from_r_s = c² / (2 r_s)
      - Tick definitions:
          t*_κ      = 2c / κ
          t*_r_s    = 4 r_s / c

    In standard Schwarzschild,
      κ = κ_from_r_s,
      t*(κ) = t*(r_s),

    so the ratios go to 1 once κ = c²/(2 r_s) is imposed:

      ratio_kappa = κ / κ_from_r_s → 1
      ratio_t     = t*_κ / t*_r_s → 1
    """
    t_star_kappa = 2 * c / kappa_sym
    kappa_from_rs = c**2 / (2 * r_s_sym)
    t_star_from_rs = 4 * r_s_sym / c

    ratio_kappa = sp.simplify(kappa_sym / kappa_from_rs)
    ratio_t = sp.simplify(t_star_kappa / t_star_from_rs)

    return {
        "t_star_kappa":   t_star_kappa,
        "t_star_from_rs": t_star_from_rs,
        "kappa_from_rs":  kappa_from_rs,
        "ratio_kappa":    ratio_kappa,
        "ratio_t_star":   ratio_t,
    }


# ---------------------------------------------------------------------------
# 2. de Sitter / cosmological horizon module
# ---------------------------------------------------------------------------

def de_sitter_radius(H_sym: sp.Expr) -> sp.Expr:
    """
    ID: DS_RADIUS
    Role: ASSUMED

    de Sitter horizon radius:
        R_H = c / H
    """
    return c / H_sym


def de_sitter_area(H_sym: sp.Expr) -> sp.Expr:
    """
    ID: DS_AREA
    Role: DERIVED

    Horizon area:
        A_H = 4π R_H²
    """
    R_H = de_sitter_radius(H_sym)
    return 4 * pi * R_H**2


def de_sitter_temperature(H_sym: sp.Expr) -> sp.Expr:
    """
    ID: DS_T
    Role: ASSUMED / DERIVED

    Gibbons–Hawking temperature:
        k_B T = ħ H /(2π)
    """
    return hbar * H_sym / (2 * pi * kB)


def de_sitter_entropy_via_BH(H_sym: sp.Expr) -> sp.Expr:
    """
    ID: DS_S_BH
    Role: DERIVED

    de Sitter horizon entropy via BH formula:
        S = k_B A_H / (4 ℓ_P²)
    """
    A_H = de_sitter_area(H_sym)
    return S_BH_standard(A_H)


def de_sitter_entropy_ratio_expr() -> sp.Expr:
    """
    Returns:
        (S_dS / A_H) - (k_B / (4 ℓ_P²))
    which should simplify to 0.
    """
    A_H = de_sitter_area(H)
    S_H = de_sitter_entropy_via_BH(H)
    ratio = sp.simplify(S_H / A_H)
    return sp.simplify(ratio - kB / (4 * ellP2))


def de_sitter_lambda_from_radius(R_sym: sp.Expr) -> sp.Expr:
    """
    ID: DS_LAMBDA_FROM_R
    Role: ASSUMED (geometry)

    For pure de Sitter, the cosmological constant is:
        Λ = 3 / R_H²
    """
    return 3 / R_sym**2


def de_sitter_lambda_H_relation_expr() -> sp.Expr:
    """
    ID: DS_LAMBDA_H_RELATION
    Role: DERIVED

    For pure de Sitter (R_H = c/H), one expects:
        Λ = 3 H² / c²

    We return:
        Λ_from_R - 3 H² / c²
    which should simplify to 0.
    """
    R_H = de_sitter_radius(H)
    Lambda_from_R = de_sitter_lambda_from_radius(R_H)
    expr = sp.simplify(Lambda_from_R - 3 * H**2 / c**2)
    return expr


# ---------------------------------------------------------------------------
# 3. FRW cosmology and effective density
# ---------------------------------------------------------------------------

def rho_crit_mass_from_H(H_sym: sp.Expr) -> sp.Expr:
    """
    ID: FRW_RHO_CRIT_MASS
    Role: ASSUMED (standard FRW result)

    Critical mass density:
        ρ_crit = 3 H² /(8π G)    [kg/m³]
    """
    return 3 * H_sym**2 / (8 * pi * G)


def rho_crit_energy_from_H(H_sym: sp.Expr) -> sp.Expr:
    """
    ID: FRW_RHO_CRIT_ENERGY
    Role: DERIVED

    Critical energy density:
        ε_crit = ρ_crit c² = 3 H² c² /(8π G)   [J/m³]
    """
    return rho_crit_mass_from_H(H_sym) * c**2


def rho_eff_horizon_from_H(H_sym: sp.Expr) -> sp.Expr:
    """
    ID: FRW_RHO_EFF_HORIZON
    Role: DERIVED (FH interpretation)

    Horizon-effective energy density in FH:
        ρ_eff = 3 H² c² /(8π G)
    """
    return 3 * H_sym**2 * c**2 / (8 * pi * G)


def frw_rho_eff_vs_crit_expr() -> sp.Expr:
    """
    ID: FRW_RHO_MATCH_EXPR
    Role: DERIVED

    Compares FH effective energy density with standard critical energy
    density:

        ρ_eff(H) - ε_crit(H)
    should simplify to 0 under the FH/FRW identification.
    """
    rho_eff = rho_eff_horizon_from_H(H)
    rho_crit_E = rho_crit_energy_from_H(H)
    return sp.simplify(rho_eff - rho_crit_E)


def friedmann_first_equation_expr(
    H_sym: sp.Expr,
    rho_mass_sym: sp.Expr,
    Lambda_sym: sp.Expr,
) -> sp.Expr:
    """
    ID: FRW_FIRST_EQUATION
    Role: ASSUMED

    First Friedmann equation (flat k=0 case shown explicitly):

        H² = (8π G/3) ρ_mass + Λ c²/3

    We return:
        H² - [(8π G/3) ρ_mass + (Λ c²/3)]
    which vanishes when the equation holds.
    """
    rhs = (8 * pi * G / 3) * rho_mass_sym + (Lambda_sym * c**2 / 3)
    return sp.simplify(H_sym**2 - rhs)


def friedmann_accel_equation_rhs(
    rho_sym: sp.Expr,
    p_sym: sp.Expr,
    Lambda_sym: sp.Expr,
) -> sp.Expr:
    """
    ID: FRW_ACCEL_EQUATION
    Role: ASSUMED

    Friedmann acceleration equation (k = 0):

        a¨/a = - (4πG/3)(ρ + 3p/c²) + Λ c²/3

    This function returns the right-hand side a¨/a as a SymPy expression.
    """
    return -(4 * pi * G / 3) * (rho_sym + 3 * p_sym / c**2) + (Lambda_sym * c**2 / 3)


# ---------------------------------------------------------------------------
# 4. Convenience tick/temperature helpers (FRW, Rindler)
# ---------------------------------------------------------------------------

def frw_tick_from_H(H_sym: sp.Expr) -> sp.Expr:
    """
    ID: FRW_TICK_FROM_H
    Role: DERIVED

    FRW / cosmological reversible tick (FH + GH consistency):

        t*_FRW(H) = 2 / H

    Derived from:
      - Gibbons–Hawking T(H) for de Sitter / apparent horizon:
            k_B T = ħ H /(2π)
      - Universal Tick Law t* = ħ/(π k_B T)

    so that T t* = ħ/(π k_B) ≡ Θ holds.
    """
    return 2 / H_sym


def rindler_tick_from_x(x_sym: sp.Expr) -> sp.Expr:
    """
    ID: RINDLER_TICK_FROM_X
    Role: DERIVED

    Local Rindler reversible tick (radar distance x):

        t*_Rindler(x) = 2 x / c

    x is the proper distance from the observer to the Rindler horizon.
    """
    return 2 * x_sym / c


def rindler_temperature_from_x(x_sym: sp.Expr) -> sp.Expr:
    """
    ID: RINDLER_T_FROM_X
    Role: DERIVED

    Local Rindler temperature as a function of distance x:

        T_Rindler(x) = ħ c / (2π x k_B)

    This is equivalent to the Unruh temperature with a = c²/x.
    """
    return (hbar * c) / (2 * pi * x_sym * kB)


# ---------------------------------------------------------------------------
# Identity verifier for this module
# ---------------------------------------------------------------------------

def verify_horizons_cosmo_identities() -> Dict[str, sp.Expr]:
    """
    Returns a dict of ID -> expression that should simplify
    to 0 or to simple constants (e.g. 1/2, 1) for this module.
    """
    checks: Dict[str, sp.Expr] = {}

    # Schwarzschild flux identity
    checks["SCHW_X_FROM_AREA"] = sp.simplify(schwarzschild_X_identity_expr())

    # Schwarzschild BH entropy ratio
    checks["SCHW_S_BH_RATIO"] = sp.simplify(schwarzschild_S_BH_ratio_expr())

    # Komar vs Smarr ratios (should be 1/2 and 1)
    komar_ratios = schwarzschild_komar_smarr_ratios_exprs()
    checks["SCHW_KOMAR1959_VS_SMARR"] = komar_ratios["ratio_komar1959_vs_smarr"]
    checks["SCHW_KOMARADM_VS_SMARR"] = komar_ratios["ratio_komarADM_vs_smarr"]

    # de Sitter BH entropy ratio
    checks["DS_S_BH_RATIO"] = sp.simplify(de_sitter_entropy_ratio_expr())

    # de Sitter Λ–H relation
    checks["DS_LAMBDA_H_RELATION"] = sp.simplify(de_sitter_lambda_H_relation_expr())

    # FRW: ρ_eff vs critical energy density
    checks["FRW_RHO_EFF_EQ_CRIT"] = sp.simplify(frw_rho_eff_vs_crit_expr())

    return checks


def all_horizons_cosmo_checks_pass() -> bool:
    """
    True iff all horizon/cosmology identity expressions are consistent
    with their expected values (0 or, for Komar ratios, {1/2, 1}).
    """
    checks = verify_horizons_cosmo_identities()

    # Komar ratios are expected to be 1/2 and 1; others should be 0.
    ok = True
    for key, expr in checks.items():
        if key == "SCHW_KOMAR1959_VS_SMARR":
            ok = ok and (sp.simplify(expr - sp.Rational(1, 2)) == 0)
        elif key == "SCHW_KOMARADM_VS_SMARR":
            ok = ok and (sp.simplify(expr - 1) == 0)
        else:
            ok = ok and (sp.simplify(expr) == 0)
    return ok


# ---------------------------------------------------------------------------
# Metadata index for this module
# ---------------------------------------------------------------------------

HORIZONS_COSMO_META: Dict[str, EquationMeta] = {
    "SCHW_RADIUS": EquationMeta(
        id="SCHW_RADIUS",
        role=Role.ASSUMED,
        description="Schwarzschild radius r_s = 2 G M / c².",
        papers=["FH-I", "Integral", "ConsA"],
        depends_on=[],
    ),
    "SCHW_AREA": EquationMeta(
        id="SCHW_AREA",
        role=Role.ASSUMED,
        description="Schwarzschild horizon area A = 4π r_s².",
        papers=["FH-I", "FH-V", "ConsA"],
        depends_on=["SCHW_RADIUS"],
    ),
    "SCHW_KAPPA": EquationMeta(
        id="SCHW_KAPPA",
        role=Role.ASSUMED,
        description="Schwarzschild surface gravity κ = c⁴/(4 G M).",
        papers=["FH-I", "ConsC"],
        depends_on=[],
    ),
    "SCHW_T_HAWKING": EquationMeta(
        id="SCHW_T_HAWKING",
        role=Role.DERIVED,
        description="Schwarzschild Hawking temperature from κ.",
        papers=["FH-I", "FH-IV"],
        depends_on=["SCHW_KAPPA"],
    ),
    "SCHW_TICK": EquationMeta(
        id="SCHW_TICK",
        role=Role.DERIVED,
        description="Schwarzschild reversible tick t* = 2c/κ = 8GM/c³.",
        papers=["FH-IV", "Integral"],
        depends_on=["SCHW_KAPPA"],
    ),
    "SCHW_X_INTEGRAL": EquationMeta(
        id="SCHW_X_INTEGRAL",
        role=Role.DERIVED,
        description="Integral definition of X(M) = ∫ t*(M) dE(M).",
        papers=["Integral"],
        depends_on=["SCHW_TICK"],
    ),
    "SCHW_X_FROM_AREA": EquationMeta(
        id="SCHW_X_FROM_AREA",
        role=Role.DERIVED,
        description="Schwarzschild identity X(M) = A(M)/k_SEG.",
        papers=["Integral", "Gauss-kSEG"],
        depends_on=["SCHW_X_INTEGRAL", "SCHW_AREA"],
    ),
    "SCHW_S_BH_FH": EquationMeta(
        id="SCHW_S_BH_FH",
        role=Role.DERIVED,
        description="Schwarzschild BH entropy via FH constitutive structure.",
        papers=["FH-V", "ConsA"],
        depends_on=["SCHW_AREA"],
    ),
    "SCHW_S_BH_RATIO": EquationMeta(
        id="SCHW_S_BH_RATIO",
        role=Role.DERIVED,
        description="Checks S_BH/A = k_B/(4 ℓ_P²) for Schwarzschild.",
        papers=["FH-V", "ConsA"],
        depends_on=["SCHW_S_BH_FH"],
    ),
    "KOMAR_ENERGY": EquationMeta(
        id="KOMAR_ENERGY",
        role=Role.ASSUMED,
        description="Komar energy with 1/8π or 1/4π convention.",
        papers=["ConsC", "Energy"],
        depends_on=[],
    ),
    "SMARR_SCHW": EquationMeta(
        id="SMARR_SCHW",
        role=Role.DERIVED,
        description="Smarr energy E = 2 T S for Schwarzschild.",
        papers=["FH-I", "ConsC"],
        depends_on=[],
    ),
    "SCHW_KOMAR_SMARR_RATIOS": EquationMeta(
        id="SCHW_KOMAR_SMARR_RATIOS",
        role=Role.DERIVED,
        description="Ratios E_K(1/8π)/E_Smarr = 1/2, E_K(1/4π)/E_Smarr = 1.",
        papers=["ConsC", "Energy"],
        depends_on=["KOMAR_ENERGY", "SMARR_SCHW"],
    ),
    "SCHW_BH_TRIAD": EquationMeta(
        id="SCHW_BH_TRIAD",
        role=Role.DERIVED,
        description="Schwarzschild BH triad: κ = c²/(2r_s) and t*(κ) = t*(r_s) = 4r_s/c.",
        papers=["Integral", "FH-IV"],
        depends_on=["SCHW_KAPPA", "SCHW_RADIUS", "SCHW_TICK"],
    ),
    "DS_RADIUS": EquationMeta(
        id="DS_RADIUS",
        role=Role.ASSUMED,
        description="de Sitter horizon radius R_H = c/H.",
        papers=["ConsD", "kSEG-FRW"],
        depends_on=[],
    ),
    "DS_AREA": EquationMeta(
        id="DS_AREA",
        role=Role.DERIVED,
        description="de Sitter horizon area A_H = 4π R_H².",
        papers=["ConsD", "kSEG-FRW"],
        depends_on=["DS_RADIUS"],
    ),
    "DS_T": EquationMeta(
        id="DS_T",
        role=Role.ASSUMED,
        description="Gibbons–Hawking temperature k_B T = ħH/(2π).",
        papers=["ConsD", "FH-IV"],
        depends_on=[],
    ),
    "DS_S_BH": EquationMeta(
        id="DS_S_BH",
        role=Role.DERIVED,
        description="de Sitter BH entropy S = k_B A_H /(4 ℓ_P²).",
        papers=["ConsD", "FH-V"],
        depends_on=["DS_AREA"],
    ),
    "DS_S_BH_RATIO": EquationMeta(
        id="DS_S_BH_RATIO",
        role=Role.DERIVED,
        description="Checks S_dS/A_H = k_B/(4 ℓ_P²).",
        papers=["ConsD", "FH-V"],
        depends_on=["DS_S_BH"],
    ),
    "DS_LAMBDA_FROM_R": EquationMeta(
        id="DS_LAMBDA_FROM_R",
        role=Role.ASSUMED,
        description="For pure de Sitter: Λ = 3 / R_H².",
        papers=["ConsD"],
        depends_on=[],
    ),
    "DS_LAMBDA_H_RELATION": EquationMeta(
        id="DS_LAMBDA_H_RELATION",
        role=Role.DERIVED,
        description="Checks Λ = 3 H² / c² using R_H = c/H.",
        papers=["ConsD"],
        depends_on=["DS_RADIUS", "DS_LAMBDA_FROM_R"],
    ),
    "FRW_RHO_CRIT_MASS": EquationMeta(
        id="FRW_RHO_CRIT_MASS",
        role=Role.ASSUMED,
        description="Critical mass density ρ_crit = 3 H² /(8π G).",
        papers=["kSEG-FRW", "ConsD"],
        depends_on=[],
    ),
    "FRW_RHO_CRIT_ENERGY": EquationMeta(
        id="FRW_RHO_CRIT_ENERGY",
        role=Role.DERIVED,
        description="Critical energy density ε_crit = ρ_crit c².",
        papers=["kSEG-FRW", "ConsD"],
        depends_on=["FRW_RHO_CRIT_MASS"],
    ),
    "FRW_RHO_EFF_HORIZON": EquationMeta(
        id="FRW_RHO_EFF_HORIZON",
        role=Role.DERIVED,
        description="FH horizon-effective energy density ρ_eff = 3H² c² /(8π G).",
        papers=["ConsD", "kSEG-FRW"],
        depends_on=[],
    ),
    "FRW_RHO_MATCH_EXPR": EquationMeta(
        id="FRW_RHO_MATCH_EXPR",
        role=Role.DERIVED,
        description="Checks ρ_eff(H) = ε_crit(H).",
        papers=["ConsD", "kSEG-FRW"],
        depends_on=["FRW_RHO_CRIT_ENERGY", "FRW_RHO_EFF_HORIZON"],
    ),
    "FRW_FIRST_EQUATION": EquationMeta(
        id="FRW_FIRST_EQUATION",
        role=Role.ASSUMED,
        description="Flat FRW first Friedmann equation H² = (8πG/3)ρ + Λ c²/3.",
        papers=["kSEG-FRW", "ConsD"],
        depends_on=[],
    ),
    "FRW_ACCEL_EQUATION": EquationMeta(
        id="FRW_ACCEL_EQUATION",
        role=Role.ASSUMED,
        description="Friedmann acceleration equation a¨/a = -(4πG/3)(ρ+3p/c²)+Λc²/3.",
        papers=["kSEG-FRW", "ConsD"],
        depends_on=[],
    ),
    "FRW_TICK_FROM_H": EquationMeta(
        id="FRW_TICK_FROM_H",
        role=Role.DERIVED,
        description="FRW reversible tick t*_FRW(H) = 2/H (GH + UTL).",
        papers=["ConsD", "FH-IV"],
        depends_on=["DS_T"],
    ),
    "RINDLER_TICK_FROM_X": EquationMeta(
        id="RINDLER_TICK_FROM_X",
        role=Role.DERIVED,
        description="Local Rindler tick t*_Rindler(x) = 2x/c.",
        papers=["FH0", "FH-II", "ConsC"],
        depends_on=[],
    ),
    "RINDLER_T_FROM_X": EquationMeta(
        id="RINDLER_T_FROM_X",
        role=Role.DERIVED,
        description="Rindler temperature T_Rindler(x) = ħc/(2πx k_B) via a=c²/x.",
        papers=["FH0", "FH-II", "ConsC"],
        depends_on=[],
    ),
}


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Flux Holography – Horizons & Cosmology CAS: identity checks")
    checks = verify_horizons_cosmo_identities()
    for id_key, expr in checks.items():
        print(f"  {id_key:24s}: {sp.simplify(expr)}")
    print()
    print(f"ALL HORIZONS/COSMO CHECKS PASS: {all_horizons_cosmo_checks_pass()}")