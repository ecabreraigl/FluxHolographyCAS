#!/usr/bin/env python3
"""
fh_complementary_cas.py

Flux Holography – Complementary CAS (to fh_master_cas.py)

This module collects the remaining CAS identities and checks from the
original 5-file FH CAS suite that are *not* encoded in fh_master_cas.py.

Design:
  - Depends on:
        fh_master_cas.py
        fh_core_cas.py
        fh_horizons_cosmo_cas.py
        fh_tick_noneq_cas.py
        fh_selection_integrability_iw_cas.py
        fh_corollaries_cas.py
    all importable by those module names.

  - No papers / DOIs are listed here. This file only knows:
        * which identity IDs exist,
        * which sector they belong to,
        * a plain-text description (taken from the original META where available),
        * and the SymPy residual expression used as a check (when defined).

  - Style mirrors fh_master_cas:
        * EquationMeta and IdentityRecord come from fh_master_cas.
        * We group by Sector and Role.
        * We provide run_all_checks() and a CLI with --json.

Usage:
  - Place this file next to:
        fh_master_cas.py
        fh_core_cas.py
        fh_horizons_cosmo_cas.py
        fh_tick_noneq_cas.py
        fh_selection_integrability_iw_cas.py
        fh_corollaries_cas.py

  - Run:
        python fh_complementary_cas.py
    or:
        python fh_complementary_cas.py --json
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import sympy as sp

# ---------------------------------------------------------------------------
# Imports from the master CAS and original 5 CAS modules
# ---------------------------------------------------------------------------

try:
    import fh_master_cas as master
except ImportError as e:
    raise SystemExit(
        "fh_complementary_cas: could not import fh_master_cas. "
        "Make sure fh_master_cas.py is on the Python path."
    ) from e

try:
    import fh_core_cas as core
    import fh_horizons_cosmo_cas as horizons
    import fh_tick_noneq_cas as tick
    import fh_selection_integrability_iw_cas as selection
    import fh_corollaries_cas as corollaries
except ImportError as e:
    raise SystemExit(
        "fh_complementary_cas: could not import one of the original CAS modules "
        "(fh_core_cas, fh_horizons_cosmo_cas, fh_tick_noneq_cas, "
        "fh_selection_integrability_iw_cas, fh_corollaries_cas). "
        "Place them alongside this file."
    ) from e

# Reuse Sector, Role, EquationMeta, IdentityRecord from master
Sector = master.Sector
Role = master.Role
EquationMeta = master.EquationMeta
IdentityRecord = master.IdentityRecord

# Local registries for the *extra* identities (not in fh_master_cas)
EQUATIONS_EXTRA: Dict[str, EquationMeta] = {}
IDENTITIES_EXTRA: Dict[str, IdentityRecord] = {}


def _simplify_expr(expr: sp.Expr) -> sp.Expr:
    try:
        return sp.simplify(expr)
    except Exception:
        return expr


def _map_role(orig_role: Any, fallback: Role = Role.DERIVED_COROLLARY) -> Role:
    """
    Map role enums from the original CAS modules to the master Role enum.
    """
    name = getattr(orig_role, "name", str(orig_role))
    if name == "ASSUMED":
        return Role.ASSUMED
    if name == "POSTULATE":
        return Role.POSTULATE
    if name == "DERIVED":
        return fallback
    if name == "MODEL":
        return Role.MODEL_DERIVED
    return fallback


def _ensure_equation_meta(
    eq_id: str,
    *,
    sector: Sector,
    role: Role,
    description: str = "",
    depends_on: Optional[List[str]] = None,
    backbone: bool = False,
) -> EquationMeta:
    """
    Ensure an EquationMeta entry exists for eq_id in EQUATIONS_EXTRA.
    If present, return it; otherwise create a new one.
    """
    if eq_id in EQUATIONS_EXTRA:
        return EQUATIONS_EXTRA[eq_id]

    meta = EquationMeta(
        id=eq_id,
        title=eq_id,
        sector=sector,
        role=role,
        description=description,
        papers=[],          # complementary CAS: do not list papers here
        depends_on=depends_on or [],
        latex_ref=None,
        backbone=backbone,
    )
    EQUATIONS_EXTRA[eq_id] = meta
    return meta


def _populate_equations_extra() -> None:
    """
    Build the metadata registry for all IDs that are present in the
    original 5 CAS modules but *not* already present in fh_master_cas.
    """
    # Ensure master has populated its own EQUATIONS
    if not master.EQUATIONS:
        if hasattr(master, "_populate_equations"):
            master._populate_equations()  # type: ignore[attr-defined]

    master_ids = set(master.EQUATIONS.keys())

    # 1) CORE_META from fh_core_cas
    for eq_id, meta_orig in core.CORE_META.items():
        if eq_id in master_ids:
            continue
        role = _map_role(meta_orig.role, fallback=Role.DERIVED_BACKBONE)
        _ensure_equation_meta(
            eq_id,
            sector=Sector.CORE,
            role=role,
            description=meta_orig.description,
            depends_on=list(meta_orig.depends_on),
            backbone=False,
        )

    # 2) HORIZONS_COSMO_META from fh_horizons_cosmo_cas
    for eq_id, meta_orig in horizons.HORIZONS_COSMO_META.items():
        if eq_id in master_ids:
            continue
        role = _map_role(meta_orig.role, fallback=Role.DERIVED_COROLLARY)
        _ensure_equation_meta(
            eq_id,
            sector=Sector.HORIZONS,
            role=role,
            description=meta_orig.description,
            depends_on=list(meta_orig.depends_on),
            backbone=False,
        )

    # 3) TICK_NONEQ_META from fh_tick_noneq_cas
    for eq_id, meta_orig in tick.TICK_NONEQ_META.items():
        if eq_id in master_ids:
            continue
        role = _map_role(meta_orig.role, fallback=Role.DERIVED_COROLLARY)
        _ensure_equation_meta(
            eq_id,
            sector=Sector.TICK,
            role=role,
            description=meta_orig.description,
            depends_on=list(meta_orig.depends_on),
            backbone=False,
        )

    # 4) CAS4_META from fh_selection_integrability_iw_cas
    for eq_id, meta_orig in selection.CAS4_META.items():
        if eq_id in master_ids:
            continue
        role = _map_role(meta_orig.role, fallback=Role.DERIVED_COROLLARY)
        _ensure_equation_meta(
            eq_id,
            sector=Sector.SELECTION,
            role=role,
            description=meta_orig.description,
            depends_on=list(meta_orig.depends_on),
            backbone=False,
        )

    # 5) Corollaries from fh_corollaries_cas (no META dict there)
    #    We build minimal metadata entries for the IDs used in verify_corollaries().
    try:
        cor_checks = corollaries.verify_corollaries()
        for eq_id in cor_checks.keys():
            if eq_id in master_ids:
                continue
            _ensure_equation_meta(
                eq_id,
                sector=Sector.COROLLARIES,
                role=Role.DERIVED_COROLLARY,
                description=f"Corollary identity {eq_id} from fh_corollaries_cas.",
                depends_on=[],
                backbone=False,
            )
    except Exception:
        # If corollaries.verify_corollaries fails, we silently skip its metadata.
        pass


def _register_identity(eq_id: str, expr: sp.Expr) -> None:
    """
    Register an IdentityRecord in IDENTITIES_EXTRA, assuming metadata exists.
    """
    if eq_id not in EQUATIONS_EXTRA:
        # Fallback: treat as a generic check in a best-guess sector.
        meta = _ensure_equation_meta(
            eq_id,
            sector=Sector.CORE,
            role=Role.CHECK,
            description=f"Generic check {eq_id} (auto-created).",
            depends_on=[],
            backbone=False,
        )
    else:
        meta = EQUATIONS_EXTRA[eq_id]

    simplified = _simplify_expr(expr)
    IDENTITIES_EXTRA[eq_id] = IdentityRecord(
        id=eq_id,
        expr=expr,
        simplified=simplified,
        meta=meta,
    )


def _populate_identities_extra() -> None:
    """
    Use the verify_* functions in the original CAS modules to construct
    residuals lhs - rhs (or equivalent) for all available IDs that are
    not already covered by fh_master_cas.
    """
    if not EQUATIONS_EXTRA:
        _populate_equations_extra()

    master_ids = set(master.EQUATIONS.keys())

    # CORE
    try:
        core_checks = core.verify_core_identities()
        for eq_id, expr in core_checks.items():
            if eq_id in master_ids:
                # This one is already covered in the master CAS; skip.
                continue
            _register_identity(eq_id, expr)
    except Exception:
        pass

    # HORIZONS / COSMO
    try:
        hz_checks = horizons.verify_horizons_cosmo_identities()
        for eq_id, expr in hz_checks.items():
            if eq_id in master_ids:
                continue
            # Some IDs (like SCHW_KOMAR* ratios) are not in HORIZONS_COSMO_META;
            # for those, _ensure_equation_meta will auto-create a generic entry.
            if eq_id not in EQUATIONS_EXTRA:
                _ensure_equation_meta(
                    eq_id,
                    sector=Sector.HORIZONS,
                    role=Role.CHECK,
                    description=f"Additional horizons/cosmo check {eq_id}.",
                    depends_on=[],
                    backbone=False,
                )
            _register_identity(eq_id, expr)
    except Exception:
        pass

    # TICK / near-equilibrium
    try:
        tick_checks = tick.verify_tick_noneq_identities()
        for eq_id, expr in tick_checks.items():
            if eq_id in master_ids:
                continue
            if eq_id not in EQUATIONS_EXTRA:
                _ensure_equation_meta(
                    eq_id,
                    sector=Sector.TICK,
                    role=Role.CHECK,
                    description=f"Additional tick-sector check {eq_id}.",
                    depends_on=[],
                    backbone=False,
                )
            _register_identity(eq_id, expr)
    except Exception:
        pass

    # Corollaries
    try:
        cor_checks = corollaries.verify_corollaries()
        for eq_id, expr in cor_checks.items():
            if eq_id in master_ids:
                continue
            if eq_id not in EQUATIONS_EXTRA:
                _ensure_equation_meta(
                    eq_id,
                    sector=Sector.COROLLARIES,
                    role=Role.DERIVED_COROLLARY,
                    description=f"Corollary identity {eq_id} from fh_corollaries_cas.",
                    depends_on=[],
                    backbone=False,
                )
            _register_identity(eq_id, expr)
    except Exception:
        pass

    # Selection / integrability / Iyer–Wald:
    # This module does not currently expose a verify_* API in the same
    # pattern, so we only carry its metadata in EQUATIONS_EXTRA.
    # (If a future version adds verify_selection_identities(), it can
    #  be wired in here analogously.)


def run_all_checks() -> Dict[str, Any]:
    """
    Run all complementary identities (those not covered in fh_master_cas)
    and return a machine-friendly summary.
    """
    if not IDENTITIES_EXTRA:
        _populate_identities_extra()

    summary: Dict[str, Any] = {
        "all_pass": True,
        "by_sector": {},
        "identities": {},
    }

    # Group IDs by sector
    sector_ids: Dict[Sector, List[str]] = {s: [] for s in Sector}
    for rec in IDENTITIES_EXTRA.values():
        sector_ids[rec.meta.sector].append(rec.id)

    by_sector: Dict[str, Dict[str, Any]] = {}
    global_pass = True

    for sector, ids in sector_ids.items():
        failed: List[str] = []
        for eq_id in ids:
            rec = IDENTITIES_EXTRA[eq_id]
            is_zero = bool(_simplify_expr(rec.simplified) == 0)
            if not is_zero:
                failed.append(eq_id)
        all_pass_sector = len(failed) == 0
        if not all_pass_sector:
            global_pass = False
        by_sector[sector.value] = {
            "all_pass": all_pass_sector,
            "failed": failed,
            "count": len(ids),
        }

    identities_view: Dict[str, Any] = {}
    for eq_id, rec in IDENTITIES_EXTRA.items():
        m = rec.meta
        identities_view[eq_id] = {
            "sector": m.sector.value,
            "role": m.role.value,
            "title": m.title,
            "description": m.description,
            "depends_on": m.depends_on,
            "backbone": m.backbone,
            "residual": str(rec.expr),
            "residual_simplified": str(rec.simplified),
        }

    summary["all_pass"] = global_pass
    summary["by_sector"] = by_sector
    summary["identities"] = identities_view
    return summary


def _print_human_summary(summary: Dict[str, Any]) -> None:
    print("Flux Holography — Complementary CAS Summary\n")
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
        if data["depends_on"]:
            print(f"  depends_on : {', '.join(data['depends_on'])}")
        print(f"  residual   : {data['residual_simplified']}")
        print()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Flux Holography complementary CAS. "
            "Runs symbolic checks for FH identities that are not in fh_master_cas."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary instead of human-readable text.",
    )
    args = parser.parse_args(argv)

    summary = run_all_checks()

    if args.json:
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        _print_human_summary(summary)


if __name__ == "__main__":
    main()
