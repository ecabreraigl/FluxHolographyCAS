#!/usr/bin/env python3
"""
Flux Holography — Master CAS Backbone (Single-File Version)

This file is an executable, machine-readable "mini paper" for the
backbone of Flux Holography (FH). It is designed so that:

  * A human can read:
      - this docstring,
      - the metadata registry,
      - and the printed summary
    to understand what is being checked.

  * A symbolic engine (SymPy) can:
      - verify that key derived relations simplify to 0,
      - confirm the algebraic closure of the backbone.

  * An LLM can:
      - parse the PAPERS registry (titles + DOIs),
      - traverse EQUATIONS to see which identities exist,
      - inspect which identities are postulates vs derived vs model.

What is encoded here?

  - Fundamental constants: G, c, ħ, k_B.
  - Derived constants:
        k_SEG = 4π G / c^3,
        ℓ_P^2 = G ħ / c^3,
        Θ = ħ / (π k_B).
  - Constitutive laws:
        S(X) = (π k_B / ħ) X,
        A(X) = k_SEG X.
  - Universal Area Law (UAL):
        A / S = 4 ℓ_P^2 / k_B.
  - Integral flux identity:
        X = A / k_SEG (for stationary horizons).
  - Bekenstein–Hawking entropy:
        S = k_B A / (4 ℓ_P^2).
  - Thermotemporal constant:
        T t* = Θ = ħ / (π k_B).
  - Tick-sector Planckian bound (FH VI model):
        τ_min = ħ / (4 π^2 k_B T).
  - Friedmann critical density as an equation of state:
        ρ_eff = 3 H^2 c^2 / (8 π G).

This is a *backbone* CAS, not full GR. Detailed curvature, Raychaudhuri,
and full horizon families are handled in the separate FH papers and your
larger CAS suite. Here we just encode and verify the core algebra.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import sympy as sp


# ---------------------------------------------------------------------------
# Paper registry (titles + DOIs) so identities can point to literature.
# ---------------------------------------------------------------------------


class PaperKind(str, Enum):
    PUBLISHED = "published"


@dataclass
class PaperRef:
    key: str
    title: str
    kind: PaperKind
    doi: Optional[str]
    url: Optional[str]


PAPERS: Dict[str, PaperRef] = {
    # Backbone FH series
    "FH0": PaperRef(
        key="FH0",
        title="Flux Holography 0: The Entropy–Action Axiom",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17211916",
        url=None,
    ),
    "FH-I": PaperRef(
        key="FH-I",
        title="Flux Holography I: A Constants-Explicit Equation of State for Spacetime",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17078630",
        url=None,
    ),
    "FH-II": PaperRef(
        key="FH-II",
        title="Flux Holography II: The Entropy–Action Law of Spacetime",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17079019",
        url=None,
    ),
    "FH-III": PaperRef(
        key="FH-III",
        title="Flux Holography III: From the Entropy–Action Law to Inertia",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17079223",
        url=None,
    ),
    "FH-IV": PaperRef(
        key="FH-IV",
        title="Flux Holography IV: The Universal Tick Law",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17184495",
        url=None,
    ),
    "FH-V": PaperRef(
        key="FH-V",
        title="Flux Holography V: The Universal Area Law",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17291257",
        url=None,
    ),
    "FH-VI": PaperRef(
        key="FH-VI",
        title="Flux Holography VI: Tick-Sector Dynamics and the Planckian Relaxation Bound",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17677166",
        url=None,
    ),
    # Closure / response algebra
    "Closure": PaperRef(
        key="Closure",
        title="Constitutive Closure of Flux Holography: The Response Algebra of Spacetime",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17294759",
        url=None,
    ),
    # Integral / 1-form structure
    "Integrability-Exact1Form": PaperRef(
        key="Integrability-Exact1Form",
        title="Flux Holography: Integrability and the Exact 1-Form",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17675434",
        url=None,
    ),
    "Integral": PaperRef(
        key="Integral",
        title="Flux Holography: The Integral Constitutive Law of Spacetime",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17675715",
        url=None,
    ),
    # Energy normalization
    "Energy": PaperRef(
        key="Energy",
        title="Energy Normalization Theorem for Flux Holography",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17675922",
        url=None,
    ),
    # Theta / thermotemporal constant
    "Theta": PaperRef(
        key="Theta",
        title="The Thermotemporal Constant: The Heartbeat of Flux Holography",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17676017",
        url=None,
    ),
    # EAL from Clausius/causality/quantum structure
    "EAL-FP": PaperRef(
        key="EAL-FP",
        title="Flux Holography: The Entropy–Action Law from Clausius, Causality, and Quantum Structure",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17676150",
        url=None,
    ),
    # Consistency checks A–E
    "ConsE": PaperRef(
        key="ConsE",
        title="Consistency Checks E: Logarithmic Corrections from the Entropy–Action Law",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17143644",
        url=None,
    ),
    "ConsD": PaperRef(
        key="ConsD",
        title="Consistency Checks D: Cosmological Horizon Density from the Entropy–Action Law",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17143382",
        url=None,
    ),
    "ConsC": PaperRef(
        key="ConsC",
        title=(
            "Consistency Checks C: Komar Energy, Raychaudhuri–Clausius Relation, "
            "Verlinde's Entropic Force, and the Generalized Second Law"
        ),
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17143270",
        url=None,
    ),
    "ConsB": PaperRef(
        key="ConsB",
        title="Consistency Checks B: Structural Predictions from the Entropy-Action Law",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17142838",
        url=None,
    ),
    "ConsA": PaperRef(
        key="ConsA",
        title="Consistency Checks A: Thermal, Compton, and Planck Families from the Entropy-Action Law",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17142743",
        url=None,
    ),
    # Selection principle
    "EAL-Selection": PaperRef(
        key="EAL-Selection",
        title="The Entropy–Action Law: A Constants-Explicit Selection Principle for Gravity",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17143922",
        url=None,
    ),
    # kSEG / equation of state family
    "Pixel-kSEG": PaperRef(
        key="Pixel-kSEG",
        title=(
            "Pixel-Flip Gravity kSEG Equation of State: "
            "A Constants-Explicit Thermodynamic Framework for Spacetime Dynamics"
        ),
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.17009385",
        url=None,
    ),
    "BH-kSEG": PaperRef(
        key="BH-kSEG",
        title="Black Hole Mechanics as an Equation of State in kSEG form",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.16989655",
        url=None,
    ),
    "kSEG-FRW": PaperRef(
        key="kSEG-FRW",
        title="Friedmann Equations as an Equation of State in kSEG_Form",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.16988020",
        url=None,
    ),
    "Gauss-kSEG": PaperRef(
        key="Gauss-kSEG",
        title="From Gauss to Horizons: The Exact Flux Identity Behind kSEG",
        kind=PaperKind.PUBLISHED,
        doi="10.5281/zenodo.16967581",
        url=None,
    ),
}


# ---------------------------------------------------------------------------
# Equation metadata + identity registry.
# ---------------------------------------------------------------------------


class Sector(str, Enum):
    CORE = "core"
    HORIZONS = "horizons_cosmo"
    TICK = "tick_noneq"
    SELECTION = "selection_integrability"
    COROLLARIES = "corollaries"


class Role(str, Enum):
    ASSUMED = "assumed"
    POSTULATE = "postulate"
    DERIVED_BACKBONE = "derived_backbone"
    DERIVED_COROLLARY = "derived_corollary"
    MODEL_DERIVED = "model_derived"
    CHECK = "consistency_check"


@dataclass
class EquationMeta:
    id: str
    title: str
    sector: Sector
    role: Role
    description: str
    papers: List[str]
    depends_on: List[str]
    latex_ref: Optional[str]
    backbone: bool


@dataclass
class IdentityRecord:
    id: str
    expr: sp.Expr
    simplified: sp.Expr
    meta: EquationMeta


EQUATIONS: Dict[str, EquationMeta] = {}
IDENTITIES: Dict[str, IdentityRecord] = {}


def register_equation(meta: EquationMeta) -> None:
    if meta.id in EQUATIONS:
        raise ValueError(f"Duplicate equation id: {meta.id}")
    EQUATIONS[meta.id] = meta


def _simplify_expr(expr: sp.Expr) -> sp.Expr:
    try:
        return sp.simplify(expr)
    except Exception:
        return expr


def register_identity(eq_id: str, expr: sp.Expr) -> None:
    if eq_id not in EQUATIONS:
        raise KeyError(f"Equation id not in EQUATIONS: {eq_id}")
    simplified = _simplify_expr(expr)
    IDENTITIES[eq_id] = IdentityRecord(
        id=eq_id,
        expr=expr,
        simplified=simplified,
        meta=EQUATIONS[eq_id],
    )


def _expr_to_str(expr: sp.Expr) -> str:
    try:
        return str(expr)
    except Exception:
        return repr(expr)


# ---------------------------------------------------------------------------
# SymPy symbols and backbone definitions.
# ---------------------------------------------------------------------------

# Fundamental constants
G, c, hbar, k_B = sp.symbols("G c hbar k_B", positive=True)

# Backbone flux and horizon variables
X = sp.symbols("X", real=True)            # primitive flux scalar
A_hor = sp.symbols("A_hor", real=True)    # horizon area used in some identities
T, t_star = sp.symbols("T t_star", positive=True)
tau_min = sp.symbols("tau_min", positive=True)
H = sp.symbols("H", real=True)

# Define FRW effective density directly as the standard Friedmann expression.
rho_eff = 3 * H**2 * c**2 / (8 * sp.pi * G)

# Derived constants as expressions
K_SEG = 4 * sp.pi * G / c**3
L_P2 = G * hbar / c**3
THETA = hbar / (sp.pi * k_B)


def S_of_X(x: sp.Expr) -> sp.Expr:
    return sp.pi * k_B * x / hbar


def A_of_X(x: sp.Expr) -> sp.Expr:
    return K_SEG * x


# ---------------------------------------------------------------------------
# Populate EQUATIONS metadata (backbone subset; extensible).
# ---------------------------------------------------------------------------


def _populate_equations() -> None:
    # Postulates: EAL, area flux. These are not checked as residuals here.
    register_equation(
        EquationMeta(
            id="EAL_CORE",
            title="Entropy–Action Law (local single-channel form)",
            sector=Sector.CORE,
            role=Role.POSTULATE,
            description="δS = (π k_B / ħ) t* δE_eff for a calibrated horizon channel.",
            papers=["FH0", "FH-II", "EAL-FP"],
            depends_on=[],
            latex_ref="Manifesto Eq. (EAL)",
            backbone=True,
        )
    )

    register_equation(
        EquationMeta(
            id="AREA_FLUX_CORE",
            title="Area flux law (local flux coordinate)",
            sector=Sector.CORE,
            role=Role.POSTULATE,
            description="δA = k_SEG t* δE_eff; area responds to the same flux scalar X.",
            papers=["FH-I", "FH-II", "Gauss-kSEG"],
            depends_on=["EAL_CORE"],
            latex_ref=None,
            backbone=True,
        )
    )

    # Definitions / structural identities
    register_equation(
        EquationMeta(
            id="K_SEG_DEF",
            title="Definition of k_SEG",
            sector=Sector.CORE,
            role=Role.DERIVED_BACKBONE,
            description="k_SEG = 4π G / c^3 from Komar / first-law structure.",
            papers=["FH-I", "Gauss-kSEG"],
            depends_on=[],
            latex_ref=None,
            backbone=True,
        )
    )

    register_equation(
        EquationMeta(
            id="L_P2_DEF",
            title="Definition of Planck area ℓ_P^2",
            sector=Sector.CORE,
            role=Role.ASSUMED,
            description="ℓ_P^2 = G ħ / c^3.",
            papers=["FH0"],
            depends_on=[],
            latex_ref=None,
            backbone=True,
        )
    )

    register_equation(
        EquationMeta(
            id="S_OF_X_CORE",
            title="Constitutive law S(X)",
            sector=Sector.CORE,
            role=Role.DERIVED_BACKBONE,
            description="S(X) = (π k_B / ħ) X from integrated EAL.",
            papers=["FH-II", "FH-V"],
            depends_on=["EAL_CORE"],
            latex_ref=None,
            backbone=True,
        )
    )

    register_equation(
        EquationMeta(
            id="A_OF_X_CORE",
            title="Constitutive law A(X)",
            sector=Sector.CORE,
            role=Role.DERIVED_BACKBONE,
            description="A(X) = k_SEG X from area flux law.",
            papers=["FH-I", "FH-V"],
            depends_on=["AREA_FLUX_CORE", "K_SEG_DEF"],
            latex_ref=None,
            backbone=True,
        )
    )

    register_equation(
        EquationMeta(
            id="UAL_CORE",
            title="Universal Area Law (UAL)",
            sector=Sector.CORE,
            role=Role.DERIVED_BACKBONE,
            description="A / S = 4 ℓ_P^2 / k_B for equilibrium horizons.",
            papers=["FH-V", "ConsB"],
            depends_on=["S_OF_X_CORE", "A_OF_X_CORE", "L_P2_DEF", "K_SEG_DEF"],
            latex_ref=None,
            backbone=True,
        )
    )

    register_equation(
        EquationMeta(
            id="X_EQ_A_OVER_KSEG",
            title="Integral flux identity X = A / k_SEG",
            sector=Sector.HORIZONS,
            role=Role.DERIVED_BACKBONE,
            description="For stationary horizons, X equals A / k_SEG via exact 1-form ω.",
            papers=["Integral", "Integrability-Exact1Form", "Gauss-kSEG"],
            depends_on=["K_SEG_DEF"],
            latex_ref=None,
            backbone=True,
        )
    )

    register_equation(
        EquationMeta(
            id="BH_ENTROPY_CORE",
            title="Bekenstein–Hawking entropy from FH backbone",
            sector=Sector.HORIZONS,
            role=Role.DERIVED_BACKBONE,
            description="S = k_B A / (4 ℓ_P^2) from S(X) and X = A / k_SEG.",
            papers=["FH-II", "FH-V", "Integral", "ConsA"],
            depends_on=["S_OF_X_CORE", "X_EQ_A_OVER_KSEG", "L_P2_DEF", "K_SEG_DEF"],
            latex_ref=None,
            backbone=True,
        )
    )

    register_equation(
        EquationMeta(
            id="THETA_DEF",
            title="Thermotemporal constant Θ",
            sector=Sector.TICK,
            role=Role.DERIVED_BACKBONE,
            description="Θ = ħ / (π k_B), the equilibrium value of T t*.",
            papers=["FH-IV", "Theta"],
            depends_on=[],
            latex_ref=None,
            backbone=True,
        )
    )

    register_equation(
        EquationMeta(
            id="TAU_MIN_ALPHA1_PLANCKIAN",
            title="Planckian relaxation time bound (τ_min)",
            sector=Sector.TICK,
            role=Role.MODEL_DERIVED,
            description="τ_min = ħ / (4 π^2 k_B T) in the FH VI domain-flip model.",
            papers=["FH-VI", "Theta"],
            depends_on=["THETA_DEF"],
            latex_ref=None,
            backbone=True,
        )
    )

    register_equation(
        EquationMeta(
            id="FRW_RHO_EFF_EQ_CRIT",
            title="Effective cosmological density (Friedmann EOS)",
            sector=Sector.COROLLARIES,
            role=Role.DERIVED_COROLLARY,
            description="ρ_eff = 3 H^2 c^2 / (8 π G) as an equation of state.",
            papers=["ConsD", "kSEG-FRW"],
            depends_on=[],
            latex_ref=None,
            backbone=False,
        )
    )


# ---------------------------------------------------------------------------
# Construct residuals for identities that are actually checked here.
# ---------------------------------------------------------------------------


def _populate_identities() -> None:
    S_expr = S_of_X(X)
    A_expr = A_of_X(X)

    # k_SEG definition
    res_kseg = K_SEG - 4 * sp.pi * G / c**3
    register_identity("K_SEG_DEF", res_kseg)

    # ℓ_P^2 definition
    res_lp2 = L_P2 - G * hbar / c**3
    register_identity("L_P2_DEF", res_lp2)

    # S(X) constitutive law
    res_S_of_X = S_expr - sp.pi * k_B * X / hbar
    register_identity("S_OF_X_CORE", res_S_of_X)

    # A(X) constitutive law
    res_A_of_X = A_expr - K_SEG * X
    register_identity("A_OF_X_CORE", res_A_of_X)

    # UAL: A/S - 4 ℓ_P^2 / k_B
    ratio_expr = A_expr / S_expr
    res_ual = ratio_expr - 4 * L_P2 / k_B
    register_identity("UAL_CORE", res_ual)

    # X = A / k_SEG ⇔ A_of_X(A/k_SEG) = A
    res_x_eq = A_of_X(A_hor / K_SEG) - A_hor
    register_identity("X_EQ_A_OVER_KSEG", res_x_eq)

    # BH entropy: S(X = A/k_SEG) - k_B A / (4 ℓ_P^2)
    S_from_A = S_of_X(A_hor / K_SEG)
    res_bh = S_from_A - k_B * A_hor / (4 * L_P2)
    register_identity("BH_ENTROPY_CORE", res_bh)

    # Θ definition: Θ - ħ/(π k_B)
    res_theta = THETA - hbar / (sp.pi * k_B)
    register_identity("THETA_DEF", res_theta)

    # τ_min from Θ and T: τ_min - ħ / (4 π^2 k_B T)
    tau_expr = THETA / (4 * sp.pi * T)
    res_tau = tau_expr - hbar / (4 * sp.pi**2 * k_B * T)
    register_identity("TAU_MIN_ALPHA1_PLANCKIAN", res_tau)

    # FRW effective density: ρ_eff - 3 H^2 c^2 / (8 π G)
    res_frw = rho_eff - 3 * H**2 * c**2 / (8 * sp.pi * G)
    register_identity("FRW_RHO_EFF_EQ_CRIT", res_frw)


# ---------------------------------------------------------------------------
# Runner and CLI.
# ---------------------------------------------------------------------------


def run_all_checks() -> Dict[str, Any]:
    """Run all registered identities and return a machine-friendly summary."""
    summary: Dict[str, Any] = {
        "all_pass": True,
        "by_sector": {},
        "identities": {},
        "papers": {},
    }

    # Group identities by sector
    sector_ids: Dict[Sector, List[str]] = {s: [] for s in Sector}
    for rec in IDENTITIES.values():
        sector_ids[rec.meta.sector].append(rec.id)

    by_sector: Dict[str, Dict[str, Any]] = {}
    global_pass = True

    for sector, ids in sector_ids.items():
        sector_failed: List[str] = []
        for eq_id in ids:
            rec = IDENTITIES[eq_id]
            is_zero = bool(_simplify_expr(rec.simplified) == 0)
            if not is_zero:
                sector_failed.append(eq_id)
        all_pass_sector = len(sector_failed) == 0
        if not all_pass_sector:
            global_pass = False
        by_sector[sector.value] = {
            "all_pass": all_pass_sector,
            "failed": sector_failed,
            "count": len(ids),
        }

    # Build identity details
    identities_view: Dict[str, Any] = {}
    for eq_id, rec in IDENTITIES.items():
        m = rec.meta
        identities_view[eq_id] = {
            "sector": m.sector.value,
            "role": m.role.value,
            "title": m.title,
            "description": m.description,
            "papers": m.papers,
            "depends_on": m.depends_on,
            "backbone": m.backbone,
            "residual": _expr_to_str(rec.expr),
            "residual_simplified": _expr_to_str(rec.simplified),
        }

    # Paper info
    papers_view: Dict[str, Any] = {
        key: {"title": p.title, "doi": p.doi} for key, p in PAPERS.items()
    }

    summary["all_pass"] = global_pass
    summary["by_sector"] = by_sector
    summary["identities"] = identities_view
    summary["papers"] = papers_view
    return summary


def _print_human_summary(summary: Dict[str, Any]) -> None:
    print("Flux Holography — Master CAS Backbone Summary\n")
    print(f"Global all_pass: {summary['all_pass']}\n")

    print("[Sectors]")
    print("---------")
    for sector_name, info in summary["by_sector"].items():
        print(
            f"{sector_name}: all_pass={info['all_pass']}, "
            f"count={info['count']}, failed={info['failed']}"
        )
    print()

    print("[Identities]")
    print("------------")
    for eq_id, data in summary["identities"].items():
        print(f"{eq_id}:")
        print(f"  sector     : {data['sector']}")
        print(f"  role       : {data['role']}")
        print(f"  title      : {data['title']}")
        print(f"  backbone   : {data['backbone']}")
        if data["papers"]:
            print(f"  papers     : {', '.join(data['papers'])}")
        if data["depends_on"]:
            print(f"  depends_on : {', '.join(data['depends_on'])}")
        print(f"  residual   : {data['residual_simplified']}")
        print()

    print("[Papers]")
    print("--------")
    for key, p in PAPERS.items():
        print(f"{key}: {p.title} (doi:{p.doi})")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Flux Holography master CAS backbone. "
            "Runs symbolic checks for core FH identities."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary instead of human-readable text.",
    )
    args = parser.parse_args(argv)

    # Wire up metadata and identities
    _populate_equations()
    _populate_identities()

    summary = run_all_checks()

    if args.json:
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        _print_human_summary(summary)


if __name__ == "__main__":
    main()
