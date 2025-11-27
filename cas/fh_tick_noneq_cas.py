"""
fh_tick_noneq_cas.py

Flux Holography – Tick Sector, Θ, Log Corrections, and Planckian Bound.

This CAS module encodes the *non-equilibrium / tick-sector* layer of Flux
Holography (FH), aligned with:

  - Theta v1+ (thermotemporal constant and tick calibration)
  - Consistency Check E (logarithmic corrections)
  - FH VI v2+ (tick-sector dynamics and Planckian bound)

It assumes the core FH constitutive backbone from fh_core_cas.py
(Entropy–Action Law, Flux Law, UTL/UAL), and focuses on:

  * The thermotemporal constant Θ:
        Θ = ħ/(π k_B),   T t* = Θ.

  * Log-corrected entropy:
        S(A) = k_B A/(4 ℓ_P²) + α k_B ln(A/ℓ_P²),
    and the induced finite-area deformation ε(A) of the effective
    area susceptibility A/S(A).

  * Tick renormalization:
        t_eff = t* (1 + δ(A)),  with δ(A) ≃ 4 α ℓ_P² / A
    interpreted as a tick-sector correction (k_SEG frozen).

  * Tick-sector dynamics:
        dε/dt = -Γ ε
    with directional rate ceiling γ(Ω) ≤ 1/t* and geometric 4π factor,
    yielding the Planckian relaxation bound:
        τ_min = t*/(4π) = ħ/(4π² k_B T)   (for α→1).

The goal is to give LLMs and symbolic engines a clean, constants-explicit
representation of the tick layer: Θ, ε(A), and τ_min(T).

IMPORTANT (integral vs product X):

  This module *does not* define the primitive flux X. The fundamental FH
  definition is:

      X = ∫ t*(dE − Ω_H dJ − Φ_H dQ_el)

  (or X = A/k_SEG in the exact 1-form sector on stationary horizons,
   as implemented in fh_horizons_cosmo_cas.py).

  All constructions here work at the level of the *tick* t*(T) and the
  deformation ε(A); any appearance of local products like t* ΔE in other
  CAS modules should be read as single-channel, fixed-tick *coordinates*
  within that global integral framework, not as a redefinition of X.
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
)

# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

T = sp.symbols("T", positive=True)                    # temperature [K]
A = sp.symbols("A", positive=True)                    # horizon area [m²]
alpha_log = sp.symbols("alpha_log", real=True)        # log-correction coeff
t_star = sp.symbols("t_star", positive=True)          # tick [s]
eps = sp.symbols("eps", real=True)                    # small deformation ε
Gamma = sp.symbols("Gamma", positive=True)            # relaxation rate [1/s]
tau = sp.symbols("tau", positive=True)                # relaxation time [s]
D = sp.symbols("D", positive=True)                    # noise strength (OU)
alpha_pf = sp.symbols("alpha_pf", positive=True)      # domain-flip factor (0<α_pf≤1)


# ---------------------------------------------------------------------------
# 1. Thermotemporal constant and Universal Tick Law
# ---------------------------------------------------------------------------

def thermotemporal_constant_sym() -> sp.Expr:
    """
    ID: THETA_CONST
    Role: DERIVED

    Thermotemporal constant:
        Θ = ħ/(π k_B)   [K·s].

    This is the same Θ that appears in the equilibrium relation
        T t* = Θ

    and in the integral flux formulation via the calibrated tick
    t* extracted from Hawking/Unruh/Gibbons–Hawking kinematics.
    """
    return hbar / (pi * kB)


def tick_from_temperature(T_sym: sp.Expr) -> sp.Expr:
    """
    ID: UTL_TICK_FROM_T
    Role: DERIVED

    Universal Tick Law (UTL):
        t* = ħ/(π k_B T).

    This is the *calibrated* reversible tick inferred from the
    combination of:
      - KMS / Hawking–Unruh temperature,
      - the definition of t* as a reversible crossing interval, and
      - the thermotemporal constant Θ.
    """
    return hbar / (pi * kB * T_sym)


def utl_identity_expr() -> sp.Expr:
    """
    ID: UTL_THETA_ID
    Role: DERIVED

    Checks:
        T t*(T) - Θ  →  0.
    """
    Theta = thermotemporal_constant_sym()
    t_sym = tick_from_temperature(T)
    return sp.simplify(T * t_sym - Theta)


# ---------------------------------------------------------------------------
# 2. Log-corrected entropy and ε(A)
# ---------------------------------------------------------------------------

def S_log_entropy(A_sym: sp.Expr, alpha_sym: sp.Expr = alpha_log) -> sp.Expr:
    """
    ID: LOG_S_ENTROPY
    Role: DERIVED (from ConsE + Theta notes)

    Log-corrected entropy:
        S(A) = k_B A/(4 ℓ_P²) + α k_B ln(A/ℓ_P²).

    Here α is a dimensionless coefficient encoding microscopic
    corrections; in the large-area limit A >> ℓ_P² one recovers
    the Bekenstein–Hawking term as the leading piece.
    """
    return kB * A_sym / (4 * ellP2) + alpha_sym * kB * sp.log(A_sym / ellP2)


def area_over_entropy(A_sym: sp.Expr, alpha_sym: sp.Expr = alpha_log) -> sp.Expr:
    """
    ID: LOG_A_OVER_S
    Role: DERIVED

    Returns:
        A / S(A)
    for the log-corrected entropy S(A).
    """
    S_A = S_log_entropy(A_sym, alpha_sym)
    return sp.simplify(A_sym / S_A)


def epsilon_from_log_correction(
    A_sym: sp.Expr,
    alpha_sym: sp.Expr = alpha_log,
) -> sp.Expr:
    """
    ID: LOG_EPSILON_FROM_S
    Role: DERIVED

    Defines ε(A) via:
        A / S(A) = (4 ℓ_P² / k_B) [1 - ε(A)].

    Solving:
        ε(A) = 1 - (k_B A) / [4 ℓ_P² S(A)].
    """
    S_A = S_log_entropy(A_sym, alpha_sym)
    epsilon_A = sp.simplify(1 - (kB * A_sym) / (4 * ellP2 * S_A))
    return epsilon_A


def epsilon_large_A_series(
    alpha_sym: sp.Expr = alpha_log,
    order: int = 2,
) -> sp.Expr:
    """
    ID: LOG_EPSILON_LARGE_A_SERIES
    Role: DERIVED

    Large-area expansion of ε(A) for A >> ℓ_P².

    Uses x = ℓ_P² / A as small parameter and returns ε(A) expanded
    to the requested order in x (default up to O(x)).
    """
    x = sp.symbols("x", positive=True)
    eps_A = epsilon_from_log_correction(A, alpha_sym)

    # Substitute A = ℓ_P²/x (large A ↔ small x), expand, then map back.
    expr_in_x = sp.simplify(eps_A.subs({A: ellP2 / x}))
    series_x = sp.series(expr_in_x, x, 0, order).removeO()
    series_A = sp.simplify(series_x.subs({x: ellP2 / A}))
    return series_A


def epsilon_large_A_limit_expr(alpha_sym: sp.Expr = alpha_log) -> sp.Expr:
    """
    ID: LOG_EPSILON_LARGE_A_LIMIT
    Role: DERIVED

    Checks:
        lim_{A→∞} ε(A)  →  0.
    """
    eps_A = epsilon_from_log_correction(A, alpha_sym)
    return sp.simplify(sp.limit(eps_A, A, sp.oo))


# ---------------------------------------------------------------------------
# 3. Tick renormalization from logarithmic corrections
# ---------------------------------------------------------------------------

def tick_delta_symbolic(A_sym: sp.Expr, alpha_sym: sp.Expr = alpha_log) -> sp.Expr:
    """
    ID: LOG_TICK_DELTA
    Role: MODEL

    Effective tick-sector deformation extracted from S(A):

      Motivated by Theta/ConsE,
        δ(A) ≃ 4 α ℓ_P² / A   (no explicit log),
      interpreted as a finite-area correction to the tick sector
      (k_SEG frozen).

    This is a *model* identification: the microscopic log corrections
    in S(A) are packaged into a macroscopic tick renormalization, while
    the constitutive relation A(X) = k_SEG X is kept rigid.
    """
    return 4 * alpha_sym * ellP2 / A_sym


def tick_renormalized_log(
    A_sym: sp.Expr,
    t_star_sym: sp.Expr,
    alpha_sym: sp.Expr = alpha_log,
) -> sp.Expr:
    """
    ID: LOG_TICK_RENORM
    Role: MODEL

    Tick renormalization:
        t_eff(A) = t* [1 + δ(A)],

    with:
        δ(A) = 4 α ℓ_P² / A.

    In FH-frozen interpretation this is purely a tick-sector effect:
    k_SEG itself does not run.
    """
    delta_A = tick_delta_symbolic(A_sym, alpha_sym)
    return sp.simplify(t_star_sym * (1 + delta_A))


# ---------------------------------------------------------------------------
# 4. OU / relaxational scaffold (ε as internal variable)
# ---------------------------------------------------------------------------

def ou_equilibrium_variance(D_sym: sp.Expr, Gamma_sym: sp.Expr) -> sp.Expr:
    """
    ID: OU_EPS_VARIANCE
    Role: MODEL

    Ornstein–Uhlenbeck equilibrium variance for ε:

        dε/dt = -Γ ε + noise  ⇒  ⟨ε²⟩_eq = D / (2 Γ).

    This is a generic stochastic scaffold for ε(t) interpreted as an
    internal nonequilibrium variable; it does not introduce new FH
    postulates, just a convenient kinetic wrapper.
    """
    return sp.simplify(D_sym / (2 * Gamma_sym))


# ---------------------------------------------------------------------------
# 5. Directional rate ceiling and Planckian bound
# ---------------------------------------------------------------------------

def gamma_ceiling_per_direction(t_star_sym: sp.Expr) -> sp.Expr:
    """
    ID: DIR_RATE_CEILING
    Role: MODEL

    Per-direction tick ceiling (lemma-level statement):

        γ(Ω) ≤ 1 / t*.

    In the Nov 18 integrability/kinetics notes this bound is derived
    from the requirement that no direction can flip faster than one
    reversible tick t*, once the calibrated tick is fixed by UTL +
    horizon kinematics.
    """
    return 1 / t_star_sym


def relaxation_rate_bound_from_tick(
    t_star_sym: sp.Expr,
    alpha_sym: sp.Expr = alpha_pf,
) -> sp.Expr:
    """
    ID: RELAX_RATE_BOUND
    Role: MODEL

    Coarse-grained relaxation rate from directional bound:

        Γ = α ∫_{S²} γ(Ω) dΩ ≤ α (4π) / t*.

    This function returns:
        Γ_max = 4π α / t*.
    """
    return 4 * pi * alpha_sym / t_star_sym


def tau_min_from_tick(
    t_star_sym: sp.Expr,
    alpha_sym: sp.Expr = alpha_pf,
) -> sp.Expr:
    """
    ID: TAU_MIN_FROM_TICK
    Role: MODEL

    Relaxation time bound from Γ_max:

        τ = 1/Γ ≥ t* / (4π α).

    We return:
        τ_min(t*, α) = t* / (4π α).
    """
    Gamma_max = relaxation_rate_bound_from_tick(t_star_sym, alpha_sym)
    return sp.simplify(1 / Gamma_max)


def tau_min_from_temperature(
    T_sym: sp.Expr,
    alpha_sym: sp.Expr = alpha_pf,
) -> sp.Expr:
    """
    ID: TAU_MIN_FROM_T
    Role: MODEL / DERIVED

    Using UTL, t*(T) = ħ/(π k_B T), the bound becomes:

        τ_min(T, α) = t*(T)/(4π α)
                    = ħ / (4 π² α k_B T).

    This function returns τ_min(T, α) in closed form.
    """
    t_sym = tick_from_temperature(T_sym)
    tau_min = tau_min_from_tick(t_sym, alpha_sym)
    return sp.simplify(tau_min)


def tau_min_planckian_alpha1_expr() -> sp.Expr:
    """
    ID: TAU_MIN_ALPHA1_PLANCKIAN
    Role: DERIVED

    Specializing to the maximally efficient limit α → 1 gives:

        τ_min(T) = ħ / (4 π² k_B T).

    We return:
        τ_min(T, α=1) - ħ/(4 π² k_B T),
    which should simplify to 0.
    """
    tau_min_alpha1 = tau_min_from_temperature(T, alpha_sym=1)
    return sp.simplify(tau_min_alpha1 - hbar / (4 * pi**2 * kB * T))


# ---------------------------------------------------------------------------
# 6. Identity verifier for this module
# ---------------------------------------------------------------------------

def verify_tick_noneq_identities() -> Dict[str, sp.Expr]:
    """
    Returns a dict of ID -> expression capturing the central
    tick/Θ/log/Planckian identities that should simplify to 0.
    """
    checks: Dict[str, sp.Expr] = {}

    # UTL + Θ
    checks["UTL_THETA_ID"] = sp.simplify(utl_identity_expr())

    # ε(A) → 0 at large area
    checks["LOG_EPSILON_LARGE_A_LIMIT"] = sp.simplify(
        epsilon_large_A_limit_expr(alpha_sym=alpha_log)
    )

    # τ_min(α=1) vs explicit Planckian form
    checks["TAU_MIN_ALPHA1_PLANCKIAN"] = sp.simplify(
        tau_min_planckian_alpha1_expr()
    )

    return checks


def all_tick_noneq_checks_pass() -> bool:
    """
    True iff all tick/Θ/log/Planckian identity expressions simplify to 0.
    """
    checks = verify_tick_noneq_identities()
    ok = True
    for key, expr in checks.items():
        ok = ok and (sp.simplify(expr) == 0)
    return ok


# ---------------------------------------------------------------------------
# 7. Metadata index for this module
# ---------------------------------------------------------------------------

TICK_NONEQ_META: Dict[str, EquationMeta] = {
    "THETA_CONST": EquationMeta(
        id="THETA_CONST",
        role=Role.DERIVED,
        description="Thermotemporal constant Θ = ħ/(π k_B).",
        papers=["FH-IV", "Theta"],
        depends_on=[],
    ),
    "UTL_TICK_FROM_T": EquationMeta(
        id="UTL_TICK_FROM_T",
        role=Role.DERIVED,
        description="Universal Tick Law t* = ħ/(π k_B T).",
        papers=["FH-IV", "Theta"],
        depends_on=["THETA_CONST"],
    ),
    "UTL_THETA_ID": EquationMeta(
        id="UTL_THETA_ID",
        role=Role.DERIVED,
        description="Checks T t* = Θ.",
        papers=["FH-IV", "Theta"],
        depends_on=["THETA_CONST", "UTL_TICK_FROM_T"],
    ),
    "LOG_S_ENTROPY": EquationMeta(
        id="LOG_S_ENTROPY",
        role=Role.DERIVED,
        description="Log-corrected entropy S(A) = k_B A/(4ℓ_P²) + α k_B ln(A/ℓ_P²).",
        papers=["ConsE", "Theta"],
        depends_on=[],
    ),
    "LOG_A_OVER_S": EquationMeta(
        id="LOG_A_OVER_S",
        role=Role.DERIVED,
        description="Area susceptibility A/S(A) with log correction.",
        papers=["ConsE", "Theta"],
        depends_on=["LOG_S_ENTROPY"],
    ),
    "LOG_EPSILON_FROM_S": EquationMeta(
        id="LOG_EPSILON_FROM_S",
        role=Role.DERIVED,
        description="Defines ε(A) via A/S(A) = (4ℓ_P²/k_B)(1 − ε(A)).",
        papers=["ConsE", "Theta"],
        depends_on=["LOG_S_ENTROPY"],
    ),
    "LOG_EPSILON_LARGE_A_SERIES": EquationMeta(
        id="LOG_EPSILON_LARGE_A_SERIES",
        role=Role.DERIVED,
        description="Large-area expansion of ε(A) for A >> ℓ_P².",
        papers=["ConsE", "Theta"],
        depends_on=["LOG_EPSILON_FROM_S"],
    ),
    "LOG_EPSILON_LARGE_A_LIMIT": EquationMeta(
        id="LOG_EPSILON_LARGE_A_LIMIT",
        role=Role.DERIVED,
        description="Checks lim_{A→∞} ε(A) = 0.",
        papers=["ConsE", "Theta"],
        depends_on=["LOG_EPSILON_FROM_S"],
    ),
    "LOG_TICK_DELTA": EquationMeta(
        id="LOG_TICK_DELTA",
        role=Role.MODEL,
        description="Model tick-sector deformation δ(A) ≃ 4 α ℓ_P² / A.",
        papers=["Theta", "FH-VI"],
        depends_on=[],
    ),
    "LOG_TICK_RENORM": EquationMeta(
        id="LOG_TICK_RENORM",
        role=Role.MODEL,
        description="Tick renormalization t_eff = t*(1 + δ(A)) with δ(A) from log corrections.",
        papers=["Theta", "FH-VI"],
        depends_on=["LOG_TICK_DELTA"],
    ),
    "OU_EPS_VARIANCE": EquationMeta(
        id="OU_EPS_VARIANCE",
        role=Role.MODEL,
        description="OU equilibrium variance ⟨ε²⟩_eq = D/(2Γ).",
        papers=["FH-VI"],
        depends_on=[],
    ),
    "DIR_RATE_CEILING": EquationMeta(
        id="DIR_RATE_CEILING",
        role=Role.MODEL,
        description="Per-direction ceiling γ(Ω) ≤ 1/t*.",
        papers=["FH-VI"],
        depends_on=[],
    ),
    "RELAX_RATE_BOUND": EquationMeta(
        id="RELAX_RATE_BOUND",
        role=Role.MODEL,
        description="Relaxation rate bound Γ ≤ 4π α / t* from directional ceiling.",
        papers=["FH-VI"],
        depends_on=["DIR_RATE_CEILING"],
    ),
    "TAU_MIN_FROM_TICK": EquationMeta(
        id="TAU_MIN_FROM_TICK",
        role=Role.MODEL,
        description="τ_min(t*, α) = t*/(4π α) from Γ_max.",
        papers=["FH-VI"],
        depends_on=["RELAX_RATE_BOUND"],
    ),
    "TAU_MIN_FROM_T": EquationMeta(
        id="TAU_MIN_FROM_T",
        role=Role.DERIVED,
        description="τ_min(T, α) = ħ/(4π² α k_B T) using UTL.",
        papers=["FH-VI", "Theta"],
        depends_on=["TAU_MIN_FROM_TICK", "UTL_TICK_FROM_T"],
    ),
    "TAU_MIN_ALPHA1_PLANCKIAN": EquationMeta(
        id="TAU_MIN_ALPHA1_PLANCKIAN",
        role=Role.DERIVED,
        description="Checks Planckian bound τ_min = ħ/(4π² k_B T) for α=1.",
        papers=["FH-VI"],
        depends_on=["TAU_MIN_FROM_T"],
    ),
}


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Flux Holography – Tick/Θ/Log/Planckian CAS: identity checks")
    checks = verify_tick_noneq_identities()
    for id_key, expr in checks.items():
        print(f"  {id_key:32s}: {sp.simplify(expr)}")
    print()
    print(f"ALL TICK/NONEQ CHECKS PASS: {all_tick_noneq_checks_pass()}")