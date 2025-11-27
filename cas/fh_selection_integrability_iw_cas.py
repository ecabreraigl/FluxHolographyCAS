"""
fh_selection_integrability_iw_cas.py

Flux Holography – CAS 4
Selection Principle, Horizon 1-form Integrability, Iyer–Wald Invariance.

This module encodes the *structural guardrails* of Flux Holography (FH):

  - Selection principle:
      Compares the Entropy–Action Law (EAL) coefficient in Einstein gravity
      to a toy higher-curvature deformation λ_fR = λ_E (1 + α_curv R),
      and imposes horizon universality (same λ on all horizons)
      as a selection rule ⇒ α_curv = 0 ⇒ Einstein.

  - Integrability of the horizon heat 1-form:
      ω = t*(dE − Ω_H dJ − Φ_H dQ_el)
    in a generic state-space chart (E, J, Q). FH applies its constitutive
    closure only on families where dω = 0, i.e. where ω is an exact 1-form.

  - Iyer–Wald energy-normalization invariance:
      Ẽ = E + B with δB = 0 on the admissible variation space leaves
      δX invariant, so X, S(X), A(X) do not depend on the choice of
      Hamiltonian zero-point.

  - Constitutive forms:
      S(X) = (π k_B/ħ) X,
      A(X) = k_SEG X.

IMPORTANT (integral vs local ΔX):

  The *global* FH definition of the primitive flux is:

      X = ∫ t*(dE − Ω_H dJ − Φ_H dQ_el),

  or equivalently X = A/k_SEG in the exact 1-form sector on stationary
  horizons (see fh_horizons_cosmo_cas.py and the Integral Law note).

  In this file we use a local single-channel coordinate

      δX_loc = t_var δE

  only as a *toy* representation for Iyer–Wald invariance. It must be
  understood as a local coordinate in the integral framework, NOT as a
  replacement for the integral definition of X.
"""

import sympy as sp
from typing import Dict

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

# ---------------------------------------------------------------------
# A. Selection Principle: toy Wald-entropy deformation
# ---------------------------------------------------------------------

# curvature scalar (symbolic)
curv = sp.symbols("curv", real=True)

# dimensionless parameter controlling higher-curvature deformation
alpha_curv = sp.symbols("alpha_curv", real=True)

# EAL coefficient in Einstein gravity (constant; in FH λ_E = π k_B/ħ)
lambda_E = sp.symbols("lambda_E", positive=True, real=True, nonzero=True)

# EAL coefficient in a toy f(R)-like deformation
lambda_fR = lambda_E * (1 + alpha_curv * curv)

# symbolic statements of the two EAL variants
deltaS, t_star, deltaE_eff = sp.symbols("deltaS t_star deltaE_eff", real=True)

EAL_Einstein = sp.Eq(deltaS, lambda_E * t_star * deltaE_eff)
EAL_fR = sp.Eq(deltaS, lambda_fR * t_star * deltaE_eff)

# horizon universality condition (FH selection requirement):
#   FH demands a *horizon-independent* EAL coefficient,
#   so λ_fR − λ_E must vanish identically ⇒ α_curv = 0.
selection_condition = sp.simplify(lambda_fR - lambda_E)

# ---------------------------------------------------------------------
# B. Integrability of the horizon 1-form ω
# ---------------------------------------------------------------------

# state-space coordinates
E_var, J_var, Q_var = sp.symbols("E_var J_var Q_var", real=True)

# arbitrary symbolic functions (generic horizon family):
t_fun = sp.Function("t")(E_var, J_var, Q_var)        # tick t*(E,J,Q)
Omega_f = sp.Function("Omega")(E_var, J_var, Q_var)  # horizon angular velocity
Phi_f = sp.Function("Phi")(E_var, J_var, Q_var)      # electric potential

# 1-form components in (E, J, Q):
#   ω = ω_E dE + ω_J dJ + ω_Q dQ
omega_E = t_fun
omega_J = -t_fun * Omega_f
omega_Q = -t_fun * Phi_f

# exterior derivative components encoding dω = 0 (Maxwell-type conditions)
cond_EJ = sp.simplify(sp.diff(omega_E, J_var) - sp.diff(omega_J, E_var))
cond_EQ = sp.simplify(sp.diff(omega_E, Q_var) - sp.diff(omega_Q, E_var))
cond_JQ = sp.simplify(sp.diff(omega_J, Q_var) - sp.diff(omega_Q, J_var))

integrability_conditions: Dict[str, sp.Expr] = {
    "EJ": cond_EJ,
    "EQ": cond_EQ,
    "JQ": cond_JQ,
}

# ---------------------------------------------------------------------
# C. Iyer–Wald invariance of X, S(X), A(X)
# ---------------------------------------------------------------------

# B is an arbitrary boundary term in the Hamiltonian
B = sp.symbols("B", real=True)

# δB is the variation of B; FH restricts to δB = 0 on admissible variations
dB = sp.symbols("dB", real=True)

# local tick and energy variation (single-channel toy)
t_var = sp.symbols("t_var", real=True)
dE = sp.symbols("dE", real=True)

# Local single-channel representation of δX in an admissible chart:
#   δX_loc     = t_var δE
#   δX_loc^IW  = t_var (δE + δB)
#
# With the Iyer–Wald constraint δB = 0, one must have δX_loc^IW = δX_loc.
dX_no_shift = t_var * dE
dX_shift = t_var * (dE + dB)

# invariance condition: impose δB = 0 and check δX_shift − δX_no_shift = 0
invariance_check = sp.simplify(dX_shift.subs({dB: 0}) - dX_no_shift)

# ---------------------------------------------------------------------
# D. Constitutive forms S(X), A(X)
# ---------------------------------------------------------------------

X = sp.symbols("X", real=True)

# FH constitutive relations (rank-1 closure):
S_of_X = (pi * kB / hbar) * X
A_of_X = kSEG * X

# ---------------------------------------------------------------------
# E. Metadata index for this module
# ---------------------------------------------------------------------

CAS4_META: Dict[str, EquationMeta] = {
    "EAL_SELECTION_TOY": EquationMeta(
        id="EAL_SELECTION_TOY",
        role=Role.DERIVED,
        description=(
            "Toy f(R)-like deformation λ_fR = λ_E (1 + α_curv R) and "
            "selection condition λ_fR − λ_E = 0 ⇒ α_curv = 0 (Einstein)."
        ),
        papers=["FH-IV", "Selection", "WaldEntropy"],
        depends_on=[],
    ),
    "OMEGA_INTEGRABILITY": EquationMeta(
        id="OMEGA_INTEGRABILITY",
        role=Role.DERIVED,
        description=(
            "Integrability conditions for ω = t*(dE − Ω dJ − Φ dQ) in (E,J,Q) "
            "coordinates: dω = 0 ⇔ Maxwell-type relations on t, Ω, Φ."
        ),
        papers=["Integral", "IntegrabilityNote"],
        depends_on=[],
    ),
    "IYER_WALD_INVARIANCE": EquationMeta(
        id="IYER_WALD_INVARIANCE",
        role=Role.DERIVED,
        description=(
            "Local single-channel toy for Iyer–Wald invariance: "
            "δX_loc = t δE invariant under E→E+B with δB=0."
        ),
        papers=["Energy", "IyerWald"],
        depends_on=[],
    ),
    "CONSTITUTIVE_SX": EquationMeta(
        id="CONSTITUTIVE_SX",
        role=Role.DERIVED,
        description="FH constitutive relation S(X) = (π k_B/ħ) X.",
        papers=["FH-V", "Closure"],
        depends_on=[],
    ),
    "CONSTITUTIVE_AX": EquationMeta(
        id="CONSTITUTIVE_AX",
        role=Role.DERIVED,
        description="FH constitutive relation A(X) = k_SEG X.",
        papers=["FH-V", "Closure"],
        depends_on=[],
    ),
}

# ---------------------------------------------------------------------
# F. Script entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=== CAS 4 — Selection, Integrability, Iyer–Wald Invariance ===")
    print("Universality condition (Einstein selection): λ_fR − λ_E =", selection_condition)
    print("\nIntegrability conditions dω (should vanish on FH horizon families):")
    for k, v in integrability_conditions.items():
        print(f"  dω condition {k}: {v}")
    print("\nIyer–Wald invariance (should be 0 when δB = 0):", invariance_check)
    print("\nConstitutive forms:")
    print("  S(X) =", S_of_X)
    print("  A(X) =", A_of_X)