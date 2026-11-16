"""
run_phase2_benchmarks.py
------------------------
Executes Phase 2 (Step 1b: Two Baths Connected & Stress-Testing the Weak-Coupling Window)
with N_PG = 10,000 modes in the power-law bath and N_S = 4,000 modes in the probe bath
across all 7 exponents r in {-0.9, -0.5, -0.3, 0.0, +0.3, +0.5, +0.9}.

HARDWIRED IMMUTABLE UV & IR CAPS (Independent of D):
  - UV Bath Cap:                    D*    = 1.0
  - IR Minimum Scale Cap:           T_min = 1.0e-4

SYNCHRONIZED WITH `bath.py`, `spectral_qrt.py`, AND `run_phase1_benchmarks.py`:
  1. Uses the exact same power-law bath coupling `Gamma0 = D* / 10 = 0.1 D` in `"low_T_default"`
     and `Gamma0 = D = 0.05` in `"high_T_plateau"` as Phase 1.
  2. Uses `gamma_rule="capped_adaptive"` (where `bath.py` caps base broadenings at
     2 * c_gamma * delta) for the power-law bath `bath_pg`, eliminating 1/eps_q^2 UV
     Lorentzian tail leakage into the low-energy pseudogap for r > 0.
  3. Retains the out-of-band bound-state pole contribution (|w| > D_PG) when the probe
     is at the wide-band cap (D_S = D*), matching `analytic_two_bath_tilde_S_eps_d` in
     `spectral_qrt.py` so the wide-band curves reach S_recon(0)/ln 2 -> +1 at high T_S
     without artificial cutoff drooping.
"""

from __future__ import annotations

import os
import sys
import gc
import json
import argparse
import faulthandler
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

faulthandler.enable()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bath import Bath
from lyapunov_solver import LyapunovSolver
from spectral_qrt import (
    bare_bath_hybridization_dl,
    alpha_func_eq14,
    analytic_tilde_S0_eq14,
    analytic_tilde_S0_eq12,
    analytic_two_bath_tilde_S_eps_d,
)

RESULTS_DIR = (
    "/home/amsalda/PseudoGap/results"
    if os.path.exists("/home/amsalda/PseudoGap")
    else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
)
os.makedirs(RESULTS_DIR, exist_ok=True)

PRESETS_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase2_presets.json")

R_LIST_ALL = [-0.9, -0.5, -0.3, 0.0, 0.3, 0.5, 0.9]
TARGET_N_PG = 10000
TARGET_N_S = 4000
LN2 = np.log(2.0)

# Precomputed 64-point Gauss-Legendre nodes/weights on [0, 1] for out-of-band pole quadrature
_GL64_S, _GL64_W = np.polynomial.legendre.leggauss(64)
_GL64_S = 0.5 * (_GL64_S + 1.0)
_GL64_W = 0.5 * _GL64_W

# ============================================================================
# HARDWIRED IMMUTABLE UV & IR CAPS (Never change with D)
# ============================================================================
D_STAR: float = 1.0
T_MIN: float = 1.0e-4

ProbeProfileType = Literal["flat_band", "lorentzian"]


def assert_within_caps(val: float | np.ndarray, name: str) -> None:
    """Verify that physical energy scale `val` lies strictly within [T_MIN, D_STAR]."""
    arr = np.asarray(val, dtype=float)
    if np.any(arr < T_MIN * (1.0 - 1e-12)) or np.any(arr > D_STAR * (1.0 + 1e-12)):
        raise ValueError(
            f"Physical parameter '{name}' = {val} violates hardwired caps "
            f"[T_MIN={T_MIN:.1e}, D_STAR={D_STAR:.1f}]."
        )


def make_preset(
    preset_name: str,
    D: float = 1.0,
    Gamma0: float = 0.1,
    GammaS0: float = T_MIN,
    D_S_cut: float = D_STAR,
    T_PG: float = T_MIN,
    T_S_fixed: float = 1.0e-3,
    T_S_over_D_range: Optional[Tuple[float, float]] = None,
    GS_over_G0_range: Optional[Tuple[float, float]] = None,
    T_PG_over_D_list: Optional[List[float]] = None,
    D_S_over_D_list: Optional[List[float]] = None,
    file_suffix: Optional[str] = None,
    description: str = "",
) -> Dict[str, Any]:
    """
    Helper that builds a complete, validated preset dictionary from minimal inputs.
    Any omitted scan ranges or test lists are automatically computed within [T_MIN, D_STAR].
    """
    for v, n in [
        (D, "D"), (Gamma0, "Gamma0"), (GammaS0, "GammaS0"),
        (D_S_cut, "D_S_cut"), (T_PG, "T_PG"), (T_S_fixed, "T_S_fixed"),
    ]:
        assert_within_caps(v, n)

    if T_S_over_D_range is None:
        T_S_over_D_range = (T_MIN / D, D_STAR / D)

    if GS_over_G0_range is None:
        gs_min = T_MIN / Gamma0
        gs_max = min(1.0, D_STAR / Gamma0, 250.0 * gs_min)
        GS_over_G0_range = (gs_min, gs_max)

    if T_PG_over_D_list is None:
        tpg_abs = [T_MIN, min(D_STAR, 50.0 * T_MIN), min(D_STAR, 500.0 * T_MIN)]
        T_PG_over_D_list = [val / D for val in tpg_abs]

    if D_S_over_D_list is None:
        ds_abs = sorted(set([
            max(T_MIN, 1.0e-3 * D_STAR),
            max(T_MIN, 1.0e-2 * D_STAR),
            max(T_MIN, 1.0e-1 * D_STAR),
            D_STAR,
        ]))
        D_S_over_D_list = [val / D for val in ds_abs]

    if file_suffix is None:
        file_suffix = "" if preset_name == "low_T_default" else f"_{preset_name}"

    return {
        "preset_name": preset_name,
        "description": description or f"Preset '{preset_name}' (D={D:g}, Gamma0={Gamma0:g}, T_S={T_S_fixed:g})",
        "D": float(D),
        "Gamma0": float(Gamma0),
        "GammaS0": float(GammaS0),
        "D_S_cut": float(D_S_cut),
        "T_PG": float(T_PG),
        "T_S_fixed": float(T_S_fixed),
        "T_S_over_D_range": (float(T_S_over_D_range[0]), float(T_S_over_D_range[1])),
        "GS_over_G0_range": (float(GS_over_G0_range[0]), float(GS_over_G0_range[1])),
        "T_PG_over_D_list": [float(x) for x in T_PG_over_D_list],
        "D_S_over_D_list": [float(x) for x in D_S_over_D_list],
        "file_suffix": file_suffix,
    }


# ============================================================================
# SELECT ACTIVE PRESET HERE (No need to edit your .sbatch file)
# Options: "low_T_default", "high_T_plateau", "scaled_D01", or "all"
# ============================================================================
ACTIVE_PRESET: Union[str, List[str]] = "low_T_default"

PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    # 1. Active Default: D_S = D = D* = 1.0, Gamma0 = D*/10 = 0.1D (matches Phase 1)
    "low_T_default": make_preset(
        preset_name="low_T_default",
        description="Low-T_S default with D_S = D = D* = 1.0, Gamma0 = 0.1D, GS0 = T_PG = 1e-4D, T_S = 1e-3D",
        D=D_STAR,                                   # D = D* = 1.0
        Gamma0=D_STAR,                              # D* = D (phase 1 ideal is D/10)
        GammaS0=T_MIN,                              # 1.0e-4 D (GS0/G0 = 1.0e-3)
        D_S_cut=D_STAR,                             # D_S = D = D* = 1.0
        T_PG=T_MIN,                                 # 1.0e-4 D = T_min
        T_S_fixed=1.0e-3 * D_STAR,                  # 1.0e-3 D = 10 T_PG
        T_S_over_D_range=(1.0e-4, 1.0),             # T_S / D in [10^{-4}, 1.0]
        GS_over_G0_range=(1.0e-3, 1.0e-1),          # GammaS0 / Gamma0 in [1e-3, 1e-1]
        T_PG_over_D_list=[1.0e-4, 0.005, 0.05],     # T_PG / D in {10^{-4}, 0.005, 0.05}
        D_S_over_D_list=[0.001, 0.01, 0.1, 1.0],    # D_S / D in {0.001, 0.01, 0.1, 1.0}
        file_suffix="",
    ),
    # 2. Saved High-T Plateau Preset: D = 0.05 = D*/20, Gamma0 = D = 0.05 (matches Phase 1)
    "high_T_plateau": make_preset(
        preset_name="high_T_plateau",
        description="High-T_S S=ln2 plateau preset (D = D*/20 = 0.05, Gamma0 = D = 0.05, GS0 = T_MIN, Ds = 20D)",
        D=0.05,                                     # 0.05 = D* / 20
        Gamma0=0.05,                                # 1.0 D = 0.05 (matches Phase 1)
        GammaS0=T_MIN,                              # 1.0e-4 = 0.002 D (GS0/G0 = 2e-3)
        D_S_cut=D_STAR,                             # 20 D = 1.0 = D*
        T_PG=T_MIN,                                 # 1.0e-4 = 0.002 D = T_min
        T_S_fixed=0.01 * 0.05,                      # 0.01 D = 5.0e-4 = 5 T_PG
        T_S_over_D_range=(2.0e-3, 20.0),            # T_S / D in [0.002, 20.0]
        GS_over_G0_range=(2.0e-3, 2.0e-1),          # GammaS0 / Gamma0 in [2e-3, 2e-1]
        T_PG_over_D_list=[0.002, 0.02, 0.20],       # T_PG / D in {0.002, 0.02, 0.2}
        D_S_over_D_list=[0.1, 1.0, 10.0, 20.0],     # D_S / D in {0.1, 1, 10, 20}
        file_suffix="_high_T_plateau",
    ),
    # 3. Saved D = D*/10 Rescaled Preset:
    "scaled_D01": make_preset(
        preset_name="scaled_D01",
        description="Rescaled D = D*/10 = 0.1, Gamma0 = 2.5D = 0.25, GS0 = T_PG = D/1000, Ds = 10D, T_S = 0.01D",
        D=0.1,
        Gamma0=0.25,
        GammaS0=T_MIN,
        D_S_cut=D_STAR,
        T_PG=T_MIN,
        T_S_fixed=1.0e-3,
        T_S_over_D_range=(1.0e-3, 10.0),
        GS_over_G0_range=(4.0e-4, 1.0e-1),
        T_PG_over_D_list=[1.0e-3, 0.05, 0.50],
        D_S_over_D_list=[0.01, 0.1, 1.0, 10.0],
        file_suffix="_scaled_D01",
    ),
}


def sync_presets_json() -> None:
    """
    Keep in-code `PRESET_CONFIGS` as the source of truth (so editing Python
    always takes effect immediately), while loading any extra custom presets
    added to `phase2_presets.json`.
    """
    if os.path.exists(PRESETS_JSON_PATH):
        try:
            with open(PRESETS_JSON_PATH, "r", encoding="utf-8") as f:
                user_data = json.load(f)
            for k, v in user_data.items():
                if k not in PRESET_CONFIGS:
                    PRESET_CONFIGS[k] = make_preset(**v)
        except Exception as exc:
            print(f"[Warning] Could not parse {PRESETS_JSON_PATH}: {exc}", flush=True)

    with open(PRESETS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(PRESET_CONFIGS, f, indent=2)


def make_lorentzian_shape(D_S_width: float):
    """Return Lorentzian frequency profile g_S(w) = 1 / (1 + (w / D_S_width)^2)."""
    return lambda w: 1.0 / (1.0 + (np.asarray(w, dtype=float) / D_S_width) ** 2)


def get_color_map(r_values) -> dict:
    """Assign distinct, high-contrast colors sorted from negative to positive r."""
    unique_r = sorted(set(float(r) for r in r_values))
    cmap = plt.cm.turbo(np.linspace(0.08, 0.92, len(unique_r)))
    return {r: cmap[i] for i, r in enumerate(unique_r)}


def build_two_bath_solver(
    r: float,
    T_S: float,
    probe_profile: ProbeProfileType,
    D_S_cutoff: float,
    Gamma0: float,
    GammaS0: float,
    T_PG: float,
    D_PG: float = D_STAR,
    omega_ir_PG: float = 0.0,
    n_pg: int = TARGET_N_PG,
    n_s: int = TARGET_N_S,
) -> LyapunovSolver:
    """
    Construct the two-bath LyapunovSolver with hardwired bounds T_MIN <= x <= D_STAR.
    Uses `gamma_rule="capped_adaptive"` on `bath_pg` (matching `bath.py` and Phase 1)
    so high-energy logarithmic bins near |eps_q| ~ D_PG do not leak 1/w^2 Lorentzian
    tails into the low-energy pseudogap around w = 0.
    """
    assert_within_caps(T_S, "T_S")
    assert_within_caps(T_PG, "T_PG")
    assert_within_caps(Gamma0, "Gamma0")
    assert_within_caps(GammaS0, "GammaS0")
    assert_within_caps(D_S_cutoff, "D_S_cutoff")
    assert_within_caps(D_PG, "D_PG")

    if probe_profile == "flat_band":
        D_S_band = D_S_cutoff
        probe_fn = None
    elif probe_profile == "lorentzian":
        D_S_band = D_PG
        probe_fn = make_lorentzian_shape(D_S_cutoff)
    else:
        raise ValueError(f"Unknown probe_profile: {probe_profile}")

    active_scale = min(T_S, D_S_cutoff)
    D_star_pg = min(0.25 * D_PG, max(8.0 * active_scale, 5.0 * T_MIN))
    D_star_s = min(0.25 * D_S_band, max(8.0 * min(T_S, D_S_band), 5.0 * T_MIN))

    bath_pg = Bath(
        name="PG",
        Gamma0=Gamma0,
        r=r,
        T=T_PG,
        D=D_PG,
        omega_ir=omega_ir_PG,
        discretization="log_linear",
        target_n_modes=n_pg,
        D_star=D_star_pg,
        gamma_rule="capped_adaptive",
        c_gamma=1.0,
    )

    bath_s = Bath(
        name="S",
        Gamma0=GammaS0,
        r=0.0,
        T=T_S,
        D=D_S_band,
        discretization="log_linear",
        target_n_modes=n_s,
        D_star=D_star_s,
        gamma_rule="adaptive",
        c_gamma=1.0,
        custom_shape=probe_fn,
    )

    return LyapunovSolver([bath_pg, bath_s])


def two_bath_tight_binding_sweep(
    solver: LyapunovSolver,
    eps_d_grid: np.ndarray,
    T_S: float,
    probe_profile: ProbeProfileType,
    D_S_cutoff: float,
    D_PG: float = D_STAR,
    include_out_of_band_poles: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Evaluate the two-bath tight-binding sweep across `eps_d_grid`:
      1. Inside the bath band (|w| <= D_PG): uses `solver.sweep_eps_d_protocol` (`schur_lddl`).
      2. Outside the power-law bath band (|w| > D_PG): when the probe is at the wide-band cap
         (D_S_cutoff >= D_PG) and `include_out_of_band_poles=True`, adds the out-of-band
         bound-state pole contribution using the discrete tight-binding self-energy Sigma_{PG;DL}(w)
         from `bare_bath_hybridization_dl`, matching `analytic_two_bath_tilde_S_eps_d` in
         `spectral_qrt.py` so the wide-band tight-binding points reach +1 at high T_S without drooping.
    """
    sweep = solver.sweep_eps_d_protocol(eps_d_grid, target_bath="S", method="schur_lddl")

    has_subband_hard_cutoff = (
        probe_profile == "flat_band" and D_S_cutoff < 0.9999 * D_PG
    )
    x_full_max = 45.0
    if (
        include_out_of_band_poles
        and not has_subband_hard_cutoff
        and (1.0001 * D_PG / T_S < x_full_max)
    ):
        bath_pg = solver.baths[0]
        bath_s = solver.baths[1]
        Gamma0 = bath_pg.Gamma0
        r = bath_pg.r
        GammaS0 = bath_s.Gamma0

        x_start = 1.0001 * D_PG / T_S
        span = x_full_max - x_start
        x2 = x_start + span * _GL64_S
        dx2 = span * _GL64_W
        w_pos = x2 * T_S
        emx2 = np.exp(-x2)
        tw2 = x2 * emx2 / ((1.0 + emx2) ** 2)

        sig_pos, _ = bare_bath_hybridization_dl(bath_pg, w_pos)
        g_p = Gamma0 * (w_pos ** r)
        if probe_profile == "lorentzian":
            g_s = GammaS0 / (1.0 + (w_pos / D_S_cutoff) ** 2)
        else:
            g_s = np.full_like(w_pos, GammaS0)
        branch = g_s / (g_p + g_s)

        eps_min = float(eps_d_grid[0])
        ed_col = eps_d_grid[:, None]

        # Negative frequencies -w_pos (where Sigma_{DL}(-w) = -Sigma_{DL}(+w))
        ph_neg = 0.5 * (
            np.sign(ed_col + w_pos[None, :] - sig_pos[None, :])
            - np.sign(eps_min + w_pos[None, :] - sig_pos[None, :])
        )
        # Positive frequencies +w_pos
        ph_pos = 0.5 * (
            np.sign(ed_col - w_pos[None, :] + sig_pos[None, :])
            - np.sign(eps_min - w_pos[None, :] + sig_pos[None, :])
        )

        S_pole = ((ph_neg - ph_pos) * branch[None, :]) @ (tw2 * dx2)
        sweep["S_eps_d"] = sweep["S_eps_d"] + S_pole

    return sweep


def eval_analytic_two_bath(
    eps_d: float | np.ndarray,
    T_S: float,
    Gamma0: float,
    GammaS0: float,
    r: float,
    probe_profile: ProbeProfileType,
    D_S_cutoff: float,
    D_PG: float = D_STAR,
    eps_max: Optional[float] = None,
    include_out_of_band_poles: bool = True,
) -> float | np.ndarray:
    """Evaluate the two-bath continuum integral (Note Eq. 11) for the chosen probe profile."""
    if probe_profile == "flat_band":
        return analytic_two_bath_tilde_S_eps_d(
            eps_d, T_S, Gamma0, GammaS0, r,
            D_PG=D_PG, D_S=D_S_cutoff, probe_shape_fn=None, eps_max=eps_max,
            include_out_of_band_poles=include_out_of_band_poles,
        )
    if probe_profile == "lorentzian":
        return analytic_two_bath_tilde_S_eps_d(
            eps_d, T_S, Gamma0, GammaS0, r,
            D_PG=D_PG, D_S=D_PG, probe_shape_fn=make_lorentzian_shape(D_S_cutoff), eps_max=eps_max,
            include_out_of_band_poles=include_out_of_band_poles,
        )
    raise ValueError(f"Unknown probe_profile: {probe_profile}")


def run_step_1b_core_verification(
    cfg: Dict[str, Any],
    probe_profile: ProbeProfileType = "flat_band",
    r_list=R_LIST_ALL,
) -> dict:
    D = float(cfg["D"])
    D_cap = D_STAR
    Gamma0 = float(cfg["Gamma0"])
    GammaS0 = float(cfg["GammaS0"])
    ratio_S_ref = GammaS0 / Gamma0
    T_S_fixed = float(cfg["T_S_fixed"])
    T_PG_base = float(cfg["T_PG"])
    D_S_cut = float(cfg["D_S_cut"])
    ts_min_over_D, ts_max_over_D = cfg["T_S_over_D_range"]

    prof_desc = (
        f"Flat Probe Gamma_S(w) = GammaS0 * Theta(D_S - |w|) with D_S = {D_S_cut/D:g}D"
        if probe_profile == "flat_band"
        else f"Lorentzian Probe Gamma_S(w) = GammaS0 / (1 + (w/D_S)^2) with D_S = {D_S_cut/D:g}D"
    )

    header = (
        "=" * 124 + "\n"
        f"STEP 1b (CORE — {probe_profile.upper()} | Preset: {cfg['preset_name']}): {prof_desc}\n"
        f"  Hardwired Caps: D*={D_cap}, T_min={T_MIN:.0e}  |  "
        f"D={D:g}, Gamma0={Gamma0:g} ({Gamma0/D:g}D), GammaS0={GammaS0:.0e} (GS0/G0={ratio_S_ref:.1e}), "
        f"D_S={D_S_cut/D:g}D, T_PG={T_PG_base:.0e}, T_S_fixed={T_S_fixed:.0e} ({T_S_fixed/D:g}D)\n"
        + "=" * 124
    )
    print(header, flush=True)
    log_lines = [header]

    sol_chk = build_two_bath_solver(
        r=0.3, T_S=T_S_fixed, probe_profile=probe_profile, D_S_cutoff=D_S_cut,
        Gamma0=Gamma0, GammaS0=GammaS0, T_PG=T_PG_base, D_PG=D_cap, n_pg=300, n_s=150
    )
    ed_list = np.linspace(-0.3 * D_cap, 0.3 * D_cap, 21)
    max_err_fd = 0.0
    max_err_schur = 0.0
    for ed in ed_list:
        val_lyap = sol_chk.dn_d_dT(float(ed), target_bath="S", method="exact_lyapunov")
        val_fd = sol_chk.dn_d_dT(float(ed), target_bath="S", method="finite_diff", rel_dT=1e-4)
        val_schur = sol_chk.dn_d_dT(float(ed), target_bath="S", method="schur_lddl")
        max_err_fd = max(max_err_fd, abs(val_fd - val_lyap))
        max_err_schur = max(max_err_schur, abs(val_schur - val_lyap))
    chk_str = (
        f"Two-Bath Solver Cross-Check across 21 eps_d points in [-0.3D*, 0.3D*] (M={sol_chk.dim}, r=+0.3, target_bath='S'):\n"
        f"  Max |FD - MatrixLyap| = {max_err_fd:.3e}, "
        f"Max |SchurLDDL - MatrixLyap| = {max_err_schur:.3e}\n"
    )
    print(chk_str, flush=True)
    log_lines.append(chk_str)
    gc.collect()

    results_1b = {
        "preset_name": cfg["preset_name"],
        "probe_profile": probe_profile,
        "D": D,
        "D_star": D_cap,
        "T_min": T_MIN,
        "Gamma0": Gamma0,
        "GammaS0": GammaS0,
        "ratio_S_ref": ratio_S_ref,
        "T_S_fixed": T_S_fixed,
        "T_PG_base": T_PG_base,
        "D_S_cut": D_S_cut,
        "T_S_over_D_range": (ts_min_over_D, ts_max_over_D),
        "GS_over_G0_range": cfg["GS_over_G0_range"],
        "sweeps": {},
        "T_scans": {},
    }

    col_hdr = (
        f"{'r':>6} | {'alpha(r)':>9} | {'Prefactor':>11} | "
        f"{'~S_Lyap(0)/ln2':>15} | {'~S_Exact2B/ln2':>15} | {'~S_Eq12(0)/ln2':>15} | {'~S_Eq14(0)/ln2':>15} | "
        f"{'S_recon(0)/ln2':>15} | {'Target r':>8}"
    )
    sep = "-" * 124
    print(col_hdr, flush=True)
    print(sep, flush=True)
    log_lines.extend([col_hdr, sep])

    for r in sorted(r_list, reverse=True):
        alpha_r = alpha_func_eq14(r)
        prefactor = ratio_S_ref * alpha_r * (T_S_fixed ** (-r))

        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        eff_width_T = Gamma0 * (T_S_fixed ** r) * sec_pr2 + GammaS0
        eps_max = max(25.0 * D_cap, 50.0 * eff_width_T)

        u = np.linspace(-1.0, 0.0, 161)
        sinh_scale = np.arcsinh(eps_max / (0.05 * T_S_fixed))
        eps_d_grid = (0.05 * T_S_fixed) * np.sinh(sinh_scale * u)

        solver = build_two_bath_solver(
            r=r, T_S=T_S_fixed, probe_profile=probe_profile, D_S_cutoff=D_S_cut,
            Gamma0=Gamma0, GammaS0=GammaS0, T_PG=T_PG_base, D_PG=D_cap
        )
        sweep = two_bath_tight_binding_sweep(
            solver, eps_d_grid, T_S_fixed, probe_profile=probe_profile,
            D_S_cutoff=D_S_cut, D_PG=D_cap, include_out_of_band_poles=True
        )
        tS_num_0 = float(sweep["S_eps_d"][-1])

        tS_exact_curve = eval_analytic_two_bath(
            eps_d_grid, T_S_fixed, Gamma0, GammaS0, r,
            probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D_cap,
            eps_max=eps_max, include_out_of_band_poles=True
        )
        tS_exact_0 = float(tS_exact_curve[-1])
        tS_eq12_0 = analytic_tilde_S0_eq12(T_S_fixed, Gamma0, GammaS0, r)
        tS_eq14_0 = analytic_tilde_S0_eq14(T_S_fixed, Gamma0, GammaS0, r)

        S_recon_num = (tS_num_0 / LN2) / prefactor
        S_recon_exact = (tS_exact_0 / LN2) / prefactor
        S_recon_eq12 = (tS_eq12_0 / LN2) / prefactor

        sweep["alpha_r"] = alpha_r
        sweep["prefactor"] = prefactor
        sweep["tS_exact_curve"] = tS_exact_curve
        sweep["tS_num_0"] = tS_num_0
        sweep["tS_exact_0"] = tS_exact_0
        sweep["tS_eq12_0"] = tS_eq12_0
        sweep["tS_eq14_0"] = tS_eq14_0
        sweep["S_recon_num"] = S_recon_num
        sweep["S_recon_exact"] = S_recon_exact
        sweep["S_recon_eq12"] = S_recon_eq12
        results_1b["sweeps"][r] = sweep

        row_str = (
            f"{r:6.2f} | {alpha_r:9.5f} | {prefactor:11.4e} | "
            f"{tS_num_0 / LN2:15.5e} | {tS_exact_0 / LN2:15.5e} | "
            f"{tS_eq12_0 / LN2:15.5e} | {tS_eq14_0 / LN2:15.5e} | "
            f"{S_recon_num:15.5f} | {r:8.4f}"
        )
        print(row_str, flush=True)
        log_lines.append(row_str)
        gc.collect()

    # Probe Temperature Scan T_S / D across cfg["T_S_over_D_range"]
    T_grid = np.logspace(np.log10(ts_min_over_D * D), np.log10(ts_max_over_D * D), 9)
    T_dense = np.logspace(np.log10(ts_min_over_D * D), np.log10(ts_max_over_D * D), 60)
    assert_within_caps(T_grid, "T_grid")

    for r in r_list:
        alpha_r = alpha_func_eq14(r)
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        tS_num_T = []
        tS_exact_T = []
        S_recon_num_T = []
        S_recon_exact_T = []

        for T_val in T_grid:
            ts = float(T_val)
            pref = ratio_S_ref * alpha_r * (ts ** (-r))
            eff_w = Gamma0 * (ts ** r) * sec_pr2 + GammaS0
            eps_max_T = max(25.0 * D_cap, 50.0 * eff_w)
            ed_pair = np.array([-eps_max_T, 0.0])

            sol_T = build_two_bath_solver(
                r=r, T_S=ts, probe_profile=probe_profile, D_S_cutoff=D_S_cut,
                Gamma0=Gamma0, GammaS0=GammaS0, T_PG=T_PG_base, D_PG=D_cap
            )
            sw_T = two_bath_tight_binding_sweep(
                sol_T, ed_pair, ts, probe_profile=probe_profile,
                D_S_cutoff=D_S_cut, D_PG=D_cap, include_out_of_band_poles=True
            )
            val_num = float(sw_T["S_eps_d"][-1])
            val_ex = float(eval_analytic_two_bath(
                0.0, ts, Gamma0, GammaS0, r,
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D_cap,
                eps_max=eps_max_T, include_out_of_band_poles=True
            ))
            tS_num_T.append(val_num)
            tS_exact_T.append(val_ex)
            S_recon_num_T.append((val_num / LN2) / pref)
            S_recon_exact_T.append((val_ex / LN2) / pref)

        tS_exact_dense = np.array([
            float(eval_analytic_two_bath(
                0.0, float(t), Gamma0, GammaS0, r,
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D_cap,
                eps_max=max(25.0 * D_cap, 50.0 * (Gamma0 * (float(t) ** r) * sec_pr2 + GammaS0)),
                include_out_of_band_poles=True,
            ))
            for t in T_dense
        ])
        tS_trunc_dense = np.array([
            float(eval_analytic_two_bath(
                0.0, float(t), Gamma0, GammaS0, r,
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D_cap,
                eps_max=max(25.0 * D_cap, 50.0 * (Gamma0 * (float(t) ** r) * sec_pr2 + GammaS0)),
                include_out_of_band_poles=False,
            ))
            for t in T_dense
        ])
        tS_eq12_dense = np.array([analytic_tilde_S0_eq12(float(t), Gamma0, GammaS0, r) for t in T_dense])
        tS_eq14_dense = np.array([analytic_tilde_S0_eq14(float(t), Gamma0, GammaS0, r) for t in T_dense])
        pref_dense = ratio_S_ref * alpha_r * (T_dense ** (-r))

        results_1b["T_scans"][r] = {
            "T_grid": T_grid,
            "tS_num": np.array(tS_num_T),
            "tS_exact": np.array(tS_exact_T),
            "S_recon_num": np.array(S_recon_num_T),
            "S_recon_exact": np.array(S_recon_exact_T),
            "T_dense": T_dense,
            "tS_exact_dense": tS_exact_dense,
            "tS_trunc_dense": tS_trunc_dense,
            "tS_eq12_dense": tS_eq12_dense,
            "tS_eq14_dense": tS_eq14_dense,
            "S_recon_exact_dense": (tS_exact_dense / LN2) / pref_dense,
            "S_recon_trunc_dense": (tS_trunc_dense / LN2) / pref_dense,
            "S_recon_eq12_dense": (tS_eq12_dense / LN2) / pref_dense,
            "S_recon_eq14_dense": (tS_eq14_dense / LN2) / pref_dense,
        }
        gc.collect()

    print(flush=True)
    results_1b["log_lines"] = log_lines
    return results_1b


def run_step_1b_stress_tests(
    cfg: Dict[str, Any],
    probe_profile: ProbeProfileType = "flat_band",
    r_list=R_LIST_ALL,
) -> dict:
    D = float(cfg["D"])
    D_cap = D_STAR
    Gamma0 = float(cfg["Gamma0"])
    GammaS0 = float(cfg["GammaS0"])
    ratio_S_ref = GammaS0 / Gamma0
    T_S_fixed = float(cfg["T_S_fixed"])
    T_PG_base = float(cfg["T_PG"])
    D_S_cut = float(cfg["D_S_cut"])
    gs_ratio_min, gs_ratio_max = cfg["GS_over_G0_range"]

    header = (
        "=" * 112 + "\n"
        f"STEP 1b (STRESS TESTS — {probe_profile.upper()} | Preset: {cfg['preset_name']}): "
        "Weak-Coupling Sweep, Finite T_PG, and Probe Cutoffs D_S\n"
        + "=" * 112
    )
    print(header, flush=True)
    log_lines = ["", header]

    # -------------------------------------------------------------------------
    # Stress Test 1 (Panel f): Probe Coupling Ratio Gamma_S / Gamma_0
    # -------------------------------------------------------------------------
    ratio_grid = np.logspace(np.log10(gs_ratio_min), np.log10(gs_ratio_max), 7)
    ratio_dense = np.logspace(np.log10(gs_ratio_min), np.log10(gs_ratio_max), 30)
    gamma_s_sweeps = {}

    print(
        f"Stress Test 1 — S_recon(0)/ln2 vs Probe Coupling Ratio Gamma_S/Gamma_0 in [{gs_ratio_min:.1e}, {gs_ratio_max:.1e}] "
        f"(at T_S/D = {T_S_fixed/D:g}, T_PG/D = {T_PG_base/D:g}, D_S/D = {D_S_cut/D:g}):",
        flush=True,
    )
    hdr_f = f"{'r':>6} | " + " | ".join([f"GS/G0={rat:.1e}" for rat in ratio_grid[::2]]) + " | Target r"
    print(hdr_f, flush=True)
    print("-" * len(hdr_f), flush=True)
    log_lines.extend([hdr_f, "-" * len(hdr_f)])

    for r in sorted(r_list, reverse=True):
        alpha_r = alpha_func_eq14(r)
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        recon_num_list = []
        recon_ex_dense = []

        for rat in ratio_grid:
            gs = float(rat) * Gamma0
            pref = float(rat) * alpha_r * (T_S_fixed ** (-r))
            eff_w = Gamma0 * (T_S_fixed ** r) * sec_pr2 + gs
            eps_max = max(25.0 * D_cap, 50.0 * eff_w)
            ed_pair = np.array([-eps_max, 0.0])

            sol = build_two_bath_solver(
                r=r, T_S=T_S_fixed, probe_profile=probe_profile, D_S_cutoff=D_S_cut,
                Gamma0=Gamma0, GammaS0=gs, T_PG=T_PG_base, D_PG=D_cap
            )
            sw = two_bath_tight_binding_sweep(
                sol, ed_pair, T_S_fixed, probe_profile=probe_profile,
                D_S_cutoff=D_S_cut, D_PG=D_cap, include_out_of_band_poles=True
            )
            val_num = float(sw["S_eps_d"][-1])
            recon_num_list.append((val_num / LN2) / pref)

        for rat in ratio_dense:
            gs = float(rat) * Gamma0
            pref = float(rat) * alpha_r * (T_S_fixed ** (-r))
            eff_w = Gamma0 * (T_S_fixed ** r) * sec_pr2 + gs
            eps_max = max(25.0 * D_cap, 50.0 * eff_w)
            val_ex = float(eval_analytic_two_bath(
                0.0, T_S_fixed, Gamma0, gs, r,
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D_cap,
                eps_max=eps_max, include_out_of_band_poles=True
            ))
            recon_ex_dense.append((val_ex / LN2) / pref)

        gamma_s_sweeps[r] = {
            "ratio_grid": ratio_grid,
            "S_recon_num": np.array(recon_num_list),
            "ratio_dense": ratio_dense,
            "S_recon_exact_dense": np.array(recon_ex_dense),
        }
        row_f = f"{r:6.2f} | " + " | ".join([f"{val:13.5f}" for val in recon_num_list[::2]]) + f" | {r:8.4f}"
        print(row_f, flush=True)
        log_lines.append(row_f)
        gc.collect()

    # -------------------------------------------------------------------------
    # Stress Test 2 (Panel g): Finite Power-Law Bath Temperature
    # -------------------------------------------------------------------------
    r_focus = [-0.5, -0.3, 0.3, 0.5]
    tpg_over_d = cfg["T_PG_over_D_list"]
    T_PG_low = float(tpg_over_d[0]) * D
    T_PG_mid = float(tpg_over_d[1]) * D
    T_PG_high = float(tpg_over_d[2]) * D
    ts_min_over_D, ts_max_over_D = cfg["T_S_over_D_range"]
    T_S_scan_15 = np.logspace(np.log10(ts_min_over_D * D), np.log10(ts_max_over_D * D), 15)
    T_S_low_grid = T_S_scan_15[0::3]
    T_S_mid_grid = T_S_scan_15[1::3]
    T_S_high_grid = T_S_scan_15[2::3]
    T_S_dense = np.logspace(np.log10(ts_min_over_D * D), np.log10(ts_max_over_D * D), 60)

    finite_T_pg_results = {
        "T_PG_low": T_PG_low,
        "T_PG_mid": T_PG_mid,
        "T_PG_high": T_PG_high,
        "D_S_cut": D_S_cut,
        "by_r": {},
    }
    for r in r_focus:
        alpha_r = alpha_func_eq14(r)
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        recon_tpg_0 = []
        recon_tpg_mid = []
        recon_tpg_high = []
        recon_ana_curve = []
        recon_eq12_curve = []

        for t_pg_val, ts_subgrid, target_list in [
            (T_PG_low, T_S_low_grid, recon_tpg_0),
            (T_PG_mid, T_S_mid_grid, recon_tpg_mid),
            (T_PG_high, T_S_high_grid, recon_tpg_high),
        ]:
            for T_val in ts_subgrid:
                ts = float(T_val)
                pref = ratio_S_ref * alpha_r * (ts ** (-r))
                eff_w = Gamma0 * (ts ** r) * sec_pr2 + GammaS0
                eps_max_T = max(25.0 * D_cap, 50.0 * eff_w)
                ed_pair = np.array([-eps_max_T, 0.0])

                sol = build_two_bath_solver(
                    r=r, T_S=ts, probe_profile=probe_profile, D_S_cutoff=D_S_cut,
                    Gamma0=Gamma0, GammaS0=GammaS0, T_PG=t_pg_val, D_PG=D_cap
                )
                sw = two_bath_tight_binding_sweep(
                    sol, ed_pair, ts, probe_profile=probe_profile,
                    D_S_cutoff=D_S_cut, D_PG=D_cap, include_out_of_band_poles=True
                )
                target_list.append((float(sw["S_eps_d"][-1]) / LN2) / pref)

        for T_val in T_S_dense:
            ts = float(T_val)
            pref = ratio_S_ref * alpha_r * (ts ** (-r))
            eff_w = Gamma0 * (ts ** r) * sec_pr2 + GammaS0
            eps_max_T = max(25.0 * D_cap, 50.0 * eff_w)
            val_ex = float(eval_analytic_two_bath(
                0.0, ts, Gamma0, GammaS0, r,
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D_cap,
                eps_max=eps_max_T, include_out_of_band_poles=True
            ))
            val_eq12 = analytic_tilde_S0_eq12(ts, Gamma0, GammaS0, r)
            recon_ana_curve.append((val_ex / LN2) / pref)
            recon_eq12_curve.append((val_eq12 / LN2) / pref)

        finite_T_pg_results["by_r"][r] = {
            "T_S_low_grid": T_S_low_grid,
            "T_S_mid_grid": T_S_mid_grid,
            "T_S_high_grid": T_S_high_grid,
            "T_S_dense": T_S_dense,
            "recon_tpg_0": np.array(recon_tpg_0),
            "recon_tpg_mid": np.array(recon_tpg_mid),
            "recon_tpg_high": np.array(recon_tpg_high),
            "recon_ana_curve": np.array(recon_ana_curve),
            "recon_eq12_curve": np.array(recon_eq12_curve),
        }
        gc.collect()

    # -------------------------------------------------------------------------
    # Stress Test 3 (Panel h): Scanning Probe Cutoffs D_S / D
    # -------------------------------------------------------------------------
    D_S_list = [float(ds_mult) * D for ds_mult in cfg["D_S_over_D_list"]]
    for _ds in D_S_list:
        assert_within_caps(_ds, "D_S_list element")
    T_S_scan_9 = np.logspace(np.log10(ts_min_over_D * D), np.log10(ts_max_over_D * D), 9)

    probe_profile_results = {
        "D_S_list": D_S_list,
        "by_r": {},
    }
    for r in r_focus:
        alpha_r = alpha_func_eq14(r)
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        num_by_ds = {ds: [] for ds in D_S_list}
        ana_by_ds = {ds: [] for ds in D_S_list}
        eq12_curve = []

        for T_val in T_S_scan_9:
            ts = float(T_val)
            pref = ratio_S_ref * alpha_r * (ts ** (-r))
            eff_w = Gamma0 * (ts ** r) * sec_pr2 + GammaS0
            eps_max_T = max(25.0 * D_cap, 50.0 * eff_w)
            ed_pair = np.array([-eps_max_T, 0.0])

            for ds_val in D_S_list:
                sol_ds = build_two_bath_solver(
                    r=r, T_S=ts, probe_profile=probe_profile, D_S_cutoff=ds_val,
                    Gamma0=Gamma0, GammaS0=GammaS0, T_PG=T_PG_base, D_PG=D_cap
                )
                sw_ds = two_bath_tight_binding_sweep(
                    sol_ds, ed_pair, ts, probe_profile=probe_profile,
                    D_S_cutoff=ds_val, D_PG=D_cap, include_out_of_band_poles=True
                )
                num_by_ds[ds_val].append((float(sw_ds["S_eps_d"][-1]) / LN2) / pref)

        for T_val in T_S_dense:
            ts = float(T_val)
            pref = ratio_S_ref * alpha_r * (ts ** (-r))
            eff_w = Gamma0 * (ts ** r) * sec_pr2 + GammaS0
            eps_max_T = max(25.0 * D_cap, 50.0 * eff_w)
            eq12_curve.append((analytic_tilde_S0_eq12(ts, Gamma0, GammaS0, r) / LN2) / pref)
            for ds_val in D_S_list:
                val_ex = float(eval_analytic_two_bath(
                    0.0, ts, Gamma0, GammaS0, r,
                    probe_profile=probe_profile, D_S_cutoff=ds_val, D_PG=D_cap,
                    eps_max=eps_max_T, include_out_of_band_poles=True
                ))
                ana_by_ds[ds_val].append((val_ex / LN2) / pref)

        probe_profile_results["by_r"][r] = {
            "T_S_scan": T_S_scan_9,
            "T_S_dense": T_S_dense,
            "num_by_ds": {ds: np.array(vals) for ds, vals in num_by_ds.items()},
            "ana_by_ds": {ds: np.array(vals) for ds, vals in ana_by_ds.items()},
            "eq12_curve": np.array(eq12_curve),
        }
        gc.collect()

    print(flush=True)
    return {
        "gamma_s_sweeps": gamma_s_sweeps,
        "finite_T_pg": finite_T_pg_results,
        "probe_profiles": probe_profile_results,
        "log_lines": log_lines,
    }


def plot_phase2_results(
    res_core: dict,
    res_stress: dict,
    filename: str,
) -> None:
    probe_profile = res_core["probe_profile"]
    all_r = sorted(res_core["sweeps"].keys())
    colors = get_color_map(all_r)
    D = res_core["D"]
    D_cap = res_core["D_star"]
    T_S_fixed = res_core["T_S_fixed"]
    T_PG_base = res_core["T_PG_base"]
    ratio_S_ref = res_core["ratio_S_ref"]
    D_S_cut = res_core["D_S_cut"]
    Gamma0 = res_core["Gamma0"]
    GammaS0 = res_core["GammaS0"]
    x_min_T, x_max_T = res_core["T_S_over_D_range"]
    gs_ratio_min, gs_ratio_max = res_core["GS_over_G0_range"]

    if probe_profile == "flat_band":
        probe_formula_tex = r"\Gamma_S(\omega) = \Gamma_{S,0}\,\Theta(D_S - |\omega|)"
        probe_short_label = f"Flat Band $D_S={D_S_cut/D:g}D$"
    else:
        probe_formula_tex = r"\Gamma_S(\omega) = \Gamma_{S,0} / [1 + (\omega/D_S)^2]"
        probe_short_label = f"Lorentzian $D_S={D_S_cut/D:g}D$"

    fig, axes = plt.subplots(2, 4, figsize=(24, 12.8))

    # Panel (a): Probe Derivative vs eps_d / D (auto-scaled to 60 * T_S_fixed / D)
    ax = axes[0, 0]
    min_y_a = 0.0
    for r in all_r:
        sw = res_core["sweeps"][r]
        scale_factor = (T_S_fixed ** r) / ratio_S_ref
        y_vals = sw["dn_dT"] * scale_factor
        min_y_a = min(min_y_a, float(np.min(y_vals)))
        ax.plot(sw["eps_d"] / D, y_vals, "-", color=colors[r], lw=1.8)
    xlim_a = -min(D_cap / D, max(60.0 * (T_S_fixed / D), 0.06 * (D_cap / D)))
    ax.set_xlim(xlim_a, 0.0)
    ax.set_ylim(1.12 * min_y_a if min_y_a < 0 else -90.0, max(4.0, -0.05 * min_y_a))
    ax.set_xlabel(r"Impurity Energy $\epsilon_d / D$")
    ax.set_ylabel(r"Rescaled $\frac{\Gamma_0}{\Gamma_{S,0}} T_S^r \frac{\partial n_d}{\partial T_S}$")
    ax.set_title(f"(a) Step 1b: $\\partial n_d / \\partial T_S$ ($T_S/D={T_S_fixed/D:g}, T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)")
    style_a = [
        Line2D([0], [0], color="k", ls="-", lw=1.8, label=f"Tight-Binding Lyapunov ($N=14\\mathrm{{k}}$, {probe_short_label})"),
    ]
    ax.legend(handles=style_a, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.5, loc="lower left")
    ax.grid(True, alpha=0.25)

    # Panel (b): Two-Bath Integral vs eps_d / D (with low-energy zoom inset for r >= 0)
    ax = axes[0, 1]
    for r in all_r:
        sw = res_core["sweeps"][r]
        norm = sw["prefactor"] * LN2
        ax.plot(sw["eps_d"] / D, sw["tS_exact_curve"] / norm, "--", color="k", lw=1.1, alpha=0.60)
        ax.plot(sw["eps_d"] / D, sw["S_eps_d"] / norm, "-", color=colors[r], lw=1.9)
        ax.axhline(r, color=colors[r], ls=":", alpha=0.65)
    ax.set_xlim(-D_cap / D, 0.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Impurity Energy $\epsilon_d / D$")
    ax.set_ylabel(r"Rescaled $\tilde{S}(\epsilon_d) / [\frac{\Gamma_{S,0}}{\Gamma_0}\alpha(r)T_S^{-r}\ln 2]$")
    ax.set_title(f"(b) Step 1b: Integral $\\tilde{{S}}(\\epsilon_d)$ ($T_S/D={T_S_fixed/D:g}, T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)")
    style_b = [
        Line2D([0], [0], color="k", ls="-", lw=1.9, label=r"Tight-Binding Lyapunov ($-\int \partial_{T_S}n_d\,d\epsilon_d'$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.1, alpha=0.7, label=r"Exact Finite-Cutoff Integral (Eq. 11)"),
        Line2D([0], [0], color="k", ls=":", lw=1.2, label=r"Low-$T_S$ Asymptote: $\tilde{S}(0) \to r$ (Eq. 14)"),
    ]
    ax.legend(handles=style_b, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.0, loc="upper left")
    ax.grid(True, alpha=0.25)

    # Zoom inset in Panel (b) resolving the narrow thermal resonance window [xlim_a, 0] for r >= 0
    ax_ins_b = ax.inset_axes([0.12, 0.50, 0.46, 0.32])
    for r in [rv for rv in all_r if rv >= 0.0]:
        sw = res_core["sweeps"][r]
        norm = sw["prefactor"] * LN2
        ax_ins_b.plot(sw["eps_d"] / D, sw["tS_exact_curve"] / norm, "--", color="k", lw=1.0, alpha=0.65)
        ax_ins_b.plot(sw["eps_d"] / D, sw["S_eps_d"] / norm, "-", color=colors[r], lw=1.6)
        ax_ins_b.axhline(r, color=colors[r], ls=":", alpha=0.55)
    ax_ins_b.set_xlim(xlim_a, 0.0)
    ax_ins_b.set_ylim(-0.05, 1.02)
    ax_ins_b.set_title(r"Zoom near $\epsilon_d = 0$ ($r \geq 0$)", fontsize=7.2)
    ax_ins_b.tick_params(labelsize=6.5)
    ax_ins_b.grid(True, alpha=0.25)

    # Panel (c): Power-Law Scaling vs T_S / D
    ax = axes[0, 2]
    for r in all_r:
        if np.isclose(r, 0.0):
            continue
        tdata = res_core["T_scans"][r]
        ax.loglog(tdata["T_dense"] / D, np.abs(tdata["tS_eq14_dense"]) / (ratio_S_ref * LN2), ":", color=colors[r], lw=1.3, alpha=0.75)
        ax.loglog(tdata["T_dense"] / D, np.abs(tdata["tS_eq12_dense"]) / (ratio_S_ref * LN2), "--", color=colors[r], lw=1.3, alpha=0.75)
        ax.loglog(tdata["T_dense"] / D, np.abs(tdata["tS_exact_dense"]) / (ratio_S_ref * LN2), "-", color=colors[r], lw=1.5, alpha=0.85)
        ax.loglog(tdata["T_grid"] / D, np.abs(tdata["tS_num"]) / (ratio_S_ref * LN2), "o", color=colors[r], ms=5.0)
    ax.set_xlim(x_min_T, x_max_T)
    ax.set_ylim(1e-3, 5e3)
    ax.set_xlabel(r"Probe Temperature $T_S / D$ (log scale)")
    ax.set_ylabel(r"Scaled Magnitude $|\tilde{S}(0)| / (\frac{\Gamma_{S,0}}{\Gamma_0}\ln 2)$")
    ax.set_title(f"(c) Step 1b: Scaling $|\\tilde{{S}}(0)| \\propto T_S^{{-r}}$ ($D_S={D_S_cut/D:g}D, T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)")
    style_c = [
        Line2D([0], [0], color="k", marker="o", ls="None", ms=5, label=r"Tight-Binding Lyapunov ($N=14\mathrm{k}$)"),
        Line2D([0], [0], color="k", ls="-", lw=1.5, label=f"Exact Finite-Cutoff Integral (Eq. 11, $D_S={D_S_cut/D:g}D$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.3, label=r"Unbounded Note Limit (Eq. 12, $D_S\to\infty$)"),
        Line2D([0], [0], color="k", ls=":", lw=1.3, label=r"Low-$T_S$ Power Law $|r|\alpha(r)T_S^{-r}$ (Eq. 14)"),
    ]
    ax.legend(handles=style_c, frameon=True, facecolor="white", framealpha=0.85, fontsize=6.8, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)

    # Panel (d): Reconstructed Entropy vs Exponent r
    ax = axes[0, 3]
    r_vals = np.array(all_r)
    s_rec_num = np.array([res_core["sweeps"][r]["S_recon_num"] for r in r_vals])
    s_rec_ex = np.array([res_core["sweeps"][r]["S_recon_exact"] for r in r_vals])
    r_cont = np.linspace(-0.98, 0.98, 100)
    ax.plot(r_cont, r_cont, "k--", lw=1.8)
    ax.plot(r_vals, s_rec_ex, "s", color="#444444", fillstyle="none", ms=8.5, mew=1.5)
    for r, s_val in zip(r_vals, s_rec_num):
        ax.plot(r, s_val, "o", color=colors[r], ms=8, markeredgecolor="k", markeredgewidth=0.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Power-Law Exponent $r$")
    ax.set_ylabel(r"Reconstructed Entropy $S_{\mathrm{recon}}(0) / \ln 2$")
    ax.set_title(f"(d) Step 1b: Reconstructed $S(0)/\\ln 2$ vs $r$ ($T_S/D={T_S_fixed/D:g}, T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)")
    style_d = [
        Line2D([0], [0], color="k", ls="--", lw=1.8, label=r"Low-$T_S$ Linear Law: $S_{\mathrm{recon}}(0)/\ln 2 = r$ (Eq. 14)"),
        Line2D([0], [0], color="#444444", marker="s", ls="None", fillstyle="none", ms=8, mew=1.5, label=r"Exact Finite-Cutoff Integral (Eq. 11)"),
        Line2D([0], [0], color="k", marker="o", ls="None", ms=7.5, label=f"Tight-Binding Lyapunov ($T_S/D = {T_S_fixed/D:g}$)"),
    ]
    ax.legend(handles=style_d, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.0, loc="upper left")
    ax.grid(True, alpha=0.25)

    ax_ins = ax.inset_axes([0.56, 0.10, 0.39, 0.34])
    r_alpha_grid = np.linspace(-0.95, 0.95, 60)
    alpha_vals = [alpha_func_eq14(float(rv)) for rv in r_alpha_grid]
    ax_ins.plot(r_alpha_grid, alpha_vals, "k-", lw=1.5)
    for r in all_r:
        ax_ins.plot(r, res_core["sweeps"][r]["alpha_r"], "o", color=colors[r], ms=4.5)
    ax_ins.set_title(r"Prefactor $\alpha(r)$ (Eq. 14)", fontsize=7.5)
    ax_ins.set_xlabel(r"$r$", fontsize=7)
    ax_ins.tick_params(labelsize=6.5)
    ax_ins.grid(True, alpha=0.25)

    # Panel (e): Reconstructed Entropy vs T_S / D
    ax = axes[1, 0]
    for r in all_r:
        tdata = res_core["T_scans"][r]
        ax.semilogx(tdata["T_dense"] / D, tdata["S_recon_eq12_dense"], "--", color=colors[r], alpha=0.75, lw=1.4)
        ax.semilogx(tdata["T_dense"] / D, tdata["S_recon_exact_dense"], "-", color=colors[r], alpha=0.85, lw=1.6)
        ax.semilogx(tdata["T_dense"] / D, tdata["S_recon_trunc_dense"], "-.", color=colors[r], alpha=0.45, lw=1.1)
        ax.semilogx(tdata["T_grid"] / D, tdata["S_recon_num"], "o", color=colors[r], ms=5.0)
        ax.axhline(r, color=colors[r], ls=":", alpha=0.45)
    ax.axhline(1.0, color="gray", ls="-.", lw=1.1, alpha=0.6)
    ax.axvline(T_S_fixed / D, color="dimgray", ls=":", lw=1.1, alpha=0.7)
    ax.set_xlim(x_min_T, x_max_T)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Probe Temperature $T_S / D$")
    ax.set_ylabel(r"Reconstructed Entropy $S_{\mathrm{recon}}(0) / \ln 2$")
    ax.set_title(f"(e) Step 1b: $S_{{\\mathrm{{recon}}}}(0)/\\ln 2$ vs $T_S/D$ ($D_S={D_S_cut/D:g}D, T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)")
    style_e = [
        Line2D([0], [0], color="k", marker="o", ls="-", lw=1.6, ms=5, label=f"Tight-Binding & Exact Integral (Eq. 11, $D_S={D_S_cut/D:g}D$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.4, label=r"Unbounded Note Limit (Eq. 12: $D_S\to\infty, \Gamma_S\to 0$)"),
        Line2D([0], [0], color="k", ls="-.", lw=1.1, alpha=0.55, label=f"Band-Truncated Only ($|\\omega|\\leq D^*={D_cap/D:g}D$)"),
        Line2D([0], [0], color="k", ls=":", lw=1.2, label=r"Asymptotes: Low-$T_S$ ($r$, Eq. 14) & High-$T_S$ ($+1$)"),
    ]
    ax.legend(handles=style_e, frameon=True, facecolor="white", framealpha=0.85, fontsize=6.4, loc="lower left")
    ax.grid(True, alpha=0.25)

    # Panel (f): Weak-Coupling Breakdown
    ax = axes[1, 1]
    for r in all_r:
        gdata = res_stress["gamma_s_sweeps"][r]
        ax.semilogx(gdata["ratio_dense"], gdata["S_recon_exact_dense"], "--", color=colors[r], lw=1.4, alpha=0.8)
        ax.semilogx(gdata["ratio_grid"], gdata["S_recon_num"], "o", color=colors[r], ms=5.0)
        ax.axhline(r, color=colors[r], ls=":", alpha=0.45)
    ax.axvline(ratio_S_ref, color="dimgray", ls="-.", lw=1.1, alpha=0.7)
    ax.set_xlim(gs_ratio_min, gs_ratio_max)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Probe Coupling Ratio $\Gamma_{S,0} / \Gamma_0$ (log scale)")
    ax.set_ylabel(r"Reconstructed Entropy $S_{\mathrm{recon}}(0) / \ln 2$")
    ax.set_title(f"(f) Test 1: $\\Gamma_S \\ll \\Gamma(\\omega)$ Breakdown ($T_S/D={T_S_fixed/D:g}, T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)")
    style_f = [
        Line2D([0], [0], color="k", marker="o", ls="None", ms=5, label=r"Tight-Binding Lyapunov ($N=14\mathrm{k}$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.4, label=r"Exact Finite-Cutoff Integral (Eq. 11)"),
        Line2D([0], [0], color="k", ls=":", lw=1.2, label=r"Weak-Probe Limit: $\Gamma_S \to 0$ (Eq. 14)"),
        Line2D([0], [0], color="dimgray", ls="-.", lw=1.1, label=f"Default $\\Gamma_{{S,0}}/\\Gamma_0 = {ratio_S_ref:.0e}$"),
    ]
    ax.legend(handles=style_f, frameon=True, facecolor="white", framealpha=0.85, fontsize=6.8, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)

    # Panel (g): Finite Bath Temp T_PG
    ax = axes[1, 2]
    T_PG_low = res_stress["finite_T_pg"]["T_PG_low"]
    T_PG_mid = res_stress["finite_T_pg"]["T_PG_mid"]
    T_PG_high = res_stress["finite_T_pg"]["T_PG_high"]
    for r, tpg_data in res_stress["finite_T_pg"]["by_r"].items():
        ax.semilogx(tpg_data["T_S_dense"] / D, tpg_data["recon_eq12_curve"], "--", color=colors[r], lw=1.3, alpha=0.70)
        ax.semilogx(tpg_data["T_S_dense"] / D, tpg_data["recon_ana_curve"], "-", color=colors[r], lw=1.6, alpha=0.85)
        ax.semilogx(
            tpg_data["T_S_low_grid"] / D, tpg_data["recon_tpg_0"],
            "o", color=colors[r], ms=5.5, markeredgecolor="k", markeredgewidth=0.4,
        )
        ax.semilogx(
            tpg_data["T_S_mid_grid"] / D, tpg_data["recon_tpg_mid"],
            "s", color=colors[r], ms=6.8, fillstyle="none", mew=1.5,
        )
        ax.semilogx(
            tpg_data["T_S_high_grid"] / D, tpg_data["recon_tpg_high"],
            "^", color=colors[r], ms=6.8, fillstyle="none", mew=1.5,
        )
        ax.axhline(r, color=colors[r], ls=":", alpha=0.45)
    ax.axhline(1.0, color="gray", ls="-.", lw=1.0, alpha=0.6)
    ax.axvline(T_PG_mid / D, color="dimgray", ls="--", lw=1.2, alpha=0.7)
    ax.axvline(T_PG_high / D, color="k", ls="-.", lw=1.2, alpha=0.7)
    ax.set_xlim(x_min_T, x_max_T)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Probe Temperature $T_S / D$")
    ax.set_ylabel(r"Reconstructed Entropy $S_{\mathrm{recon}}(0) / \ln 2$")
    ax.set_title(
        f"(g) Test 2: Varying Bath Temp $T_{{\\mathrm{{PG}}}}/D \\in "
        f"\\{{{T_PG_low/D:g}, {T_PG_mid/D:g}, {T_PG_high/D:g}\\}}$ ($D_S={D_S_cut/D:g}D$)"
    )
    style_g = [
        Line2D([0], [0], color="k", ls="-", lw=1.6, label=f"Exact Finite-Cutoff (Eq. 11, $D_S={D_S_cut/D:g}D$, indep. of $T_{{\\mathrm{{PG}}}}$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.3, label=r"Unbounded Note Limit (Eq. 12, $D_S\to\infty$)"),
        Line2D([0], [0], color="k", marker="o", ls="None", ms=5.5, label=f"Tight-Binding: $T_{{\\mathrm{{PG}}}}/D = {T_PG_low/D:g}$"),
        Line2D([0], [0], color="k", marker="s", ls="None", ms=6.5, fillstyle="none", mew=1.5, label=f"Tight-Binding: $T_{{\\mathrm{{PG}}}}/D = {T_PG_mid/D:g}$"),
        Line2D([0], [0], color="k", marker="^", ls="None", ms=6.5, fillstyle="none", mew=1.5, label=f"Tight-Binding: $T_{{\\mathrm{{PG}}}}/D = {T_PG_high/D:g}$"),
    ]
    ax.legend(handles=style_g, frameon=True, facecolor="white", framealpha=0.85, fontsize=6.4, loc="lower left")
    ax.grid(True, alpha=0.25)

    # Panel (h): Scanning Probe Cutoffs D_S / D (with explicit legend entry for the faded Eq. 12 line)
    ax = axes[1, 3]
    D_S_list = res_stress["probe_profiles"]["D_S_list"]
    ds_styles = [
        (D_S_list[3], "-", "o", 1.8, True, f"Cutoff $D_S = {D_S_list[3]/D:g}D = D^*$ (Eq. 11 & TB)"),
        (D_S_list[2], "--", "s", 1.4, False, f"Cutoff $D_S = {D_S_list[2]/D:g}D$ (Eq. 11 & TB)"),
        (D_S_list[1], "-.", "d", 1.3, False, f"Cutoff $D_S = {D_S_list[1]/D:g}D$ (Eq. 11 & TB)"),
        (D_S_list[0], ":", "^", 1.3, False, f"Cutoff $D_S = {D_S_list[0]/D:g}D$ (Eq. 11 & TB)"),
    ]
    for r, pdata in res_stress["probe_profiles"]["by_r"].items():
        # Faded solid curve: Unbounded Note Eq. (12) limit (D_S -> inf)
        ax.semilogx(pdata["T_S_dense"] / D, pdata["eq12_curve"], "-", color=colors[r], lw=1.0, alpha=0.32)
        for ds_val, ls_v, mk_v, lw_v, filled, _ in ds_styles:
            ax.semilogx(pdata["T_S_dense"] / D, pdata["ana_by_ds"][ds_val], ls=ls_v, color=colors[r], lw=lw_v, alpha=0.85)
            ax.semilogx(
                pdata["T_S_scan"] / D, pdata["num_by_ds"][ds_val],
                marker=mk_v, ls="None", color=colors[r], ms=5.0,
                fillstyle="full" if filled else "none", mew=1.3,
            )
        ax.axhline(r, color=colors[r], ls=":", alpha=0.35)

    ax.axhline(1.0, color="gray", ls="-.", lw=1.0, alpha=0.6)
    for ds_val, ls_v, _, _, _, _ in ds_styles[1:]:
        ax.axvline(ds_val / D, color="dimgray", ls=ls_v, lw=1.1, alpha=0.65)
    ax.set_xlim(x_min_T, x_max_T)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Probe Temperature $T_S / D$")
    ax.set_ylabel(r"Reconstructed Entropy $S_{\mathrm{recon}}(0) / \ln 2$")
    ds_str_list = ", ".join(f"{ds/D:g}" for ds in D_S_list)
    ax.set_title(f"(h) Test 3: Probe Cutoffs $D_S/D \\in \\{{{ds_str_list}\\}}$ ($T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)")
    style_h = [
        Line2D(
            [0], [0], color="k", marker=mk_v, ls=ls_v, ms=5.0, lw=lw_v,
            fillstyle="full" if filled else "none", mew=1.3, label=lbl,
        )
        for _, ls_v, mk_v, lw_v, filled, lbl in ds_styles
    ] + [
        Line2D([0], [0], color="gray", ls="-", lw=1.2, alpha=0.5, label=r"Faded Line: Unbounded Note Limit (Eq. 12, $D_S\to\infty$)"),
        Line2D([0], [0], color="dimgray", ls="--", lw=1.1, label=r"Vertical Lines: Mode Cutoff $T_S = D_S$"),
    ]
    ax.legend(handles=style_h, frameon=True, facecolor="white", framealpha=0.85, fontsize=6.2, loc="lower left")
    ax.grid(True, alpha=0.25)

    fig.subplots_adjust(
        left=0.048,
        right=0.988,
        bottom=0.085,
        top=0.855,
        hspace=0.34,
        wspace=0.26,
    )

    r_handles = [
        Line2D([0], [0], color=colors[r], lw=3.5, label=f"$r = {r:+.1f}$")
        for r in all_r
    ]
    fig.legend(
        handles=r_handles,
        loc="upper center",
        bbox_to_anchor=(0.518, 0.972),
        ncol=len(all_r),
        frameon=True,
        facecolor="#f8f9fa",
        edgecolor="#cccccc",
        fontsize=11.0,
        title=(
            f"Power-Law Exponent $r$ Across All Panels  "
            f"($\\Gamma_{{\\mathrm{{PG}}}}(\\omega) = \\Gamma_0 |\\omega|^r$,   Probe: ${probe_formula_tex}$,   Hardwired Caps: $[T_{{\\min}}=10^{{-4}},\\, D^*=1.0]$)\n"
            f"Default Parameters (Preset: {res_core['preset_name']}, $D = {D:g}, D^* = {D_cap/D:g}D$):  "
            f"$\\Gamma_0 = {Gamma0/D:g}D$,  $\\Gamma_{{S,0}} = {GammaS0/D:g}D$ ($\\Gamma_{{S,0}}/\\Gamma_0 = {ratio_S_ref:.0e}$),  "
            f"$D_S = {D_S_cut/D:g}D$,  $T_{{\\mathrm{{PG}}}} = {T_PG_base/D:g}D$,  $T_S = {T_S_fixed/D:g}D$"
        ),
        title_fontsize=10.6,
    )
    
    fig.savefig(filename, dpi=200, bbox_inches="tight", pad_inches=0.25)

    # -------------------------------------------------------------------------
    # Save each of the 8 panels (and top legend) as individual vectorized PDFs
    # inside /home/amsalda/PseudoGap/results/plots_phase2/
    # -------------------------------------------------------------------------
    plots_dir = os.path.join(RESULTS_DIR, "plots_phase2")
    os.makedirs(plots_dir, exist_ok=True)
    preset_tag = "" if res_core["preset_name"] == "low_T_default" else f"_{res_core['preset_name']}"
    tag = f"_{probe_profile}{preset_tag}"
    panel_names = [
        "panel_a_dn_dT",
        "panel_b_tilde_S_curve",
        "panel_c_power_law_scaling",
        "panel_d_S_recon_vs_r",
        "panel_e_S_recon_vs_TS",
        "panel_f_weak_coupling_test",
        "panel_g_finite_TPG_test",
        "panel_h_probe_cutoffs_test",
    ]
    flat_axes = list(axes.flat)
    for i, (ax_i, pname) in enumerate(zip(flat_axes, panel_names)):
        for j, other_ax in enumerate(flat_axes):
            vis = (i == j)
            other_ax.set_visible(vis)
            for child_ax in getattr(other_ax, "child_axes", []):
                child_ax.set_visible(vis)
        for leg in fig.legends:
            leg.set_visible(False)
        pdf_path = os.path.join(plots_dir, f"{pname}{tag}.pdf")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.08)

    if fig.legends:
        for other_ax in flat_axes:
            other_ax.set_visible(False)
            for child_ax in getattr(other_ax, "child_axes", []):
                child_ax.set_visible(False)
        for leg in fig.legends:
            leg.set_visible(True)
        fig.savefig(
            os.path.join(plots_dir, f"legend_r_banner{tag}.pdf"),
            format="pdf", bbox_inches="tight", pad_inches=0.06,
        )

    plt.close(fig)
    print(f"Saved 8-panel Phase 2 ({probe_profile}) figure to '{filename}' and individual PDFs to '{plots_dir}/'.", flush=True)


if __name__ == "__main__":
    sync_presets_json()

    parser = argparse.ArgumentParser(description="Run Phase 2 Two-Bath Benchmarks.")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset name from PRESET_CONFIGS / phase2_presets.json, or 'all'.",
    )
    parser.add_argument("--name", type=str, default=None, help="Create & run an on-the-fly preset with this name.")
    parser.add_argument("--D", type=float, default=None, help="Override reference energy scale D in [T_min, D*].")
    parser.add_argument("--Gamma0", type=float, default=None, help="Override power-law prefactor Gamma0.")
    parser.add_argument("--GammaS0", type=float, default=None, help="Override probe coupling GammaS0.")
    parser.add_argument("--D_S", type=float, default=None, help="Override default probe cutoff D_S.")
    parser.add_argument("--T_PG", type=float, default=None, help="Override bath temperature T_PG.")
    parser.add_argument("--T_S", type=float, default=None, help="Override fixed probe temperature T_S.")
    args = parser.parse_args()

    if any(v is not None for v in (args.name, args.D, args.Gamma0, args.GammaS0, args.D_S, args.T_PG, args.T_S)):
        base_key = args.preset if args.preset in PRESET_CONFIGS else (
            ACTIVE_PRESET[0] if isinstance(ACTIVE_PRESET, (list, tuple)) else ACTIVE_PRESET
        )
        base = PRESET_CONFIGS[base_key]
        cname = args.name or "custom_cli"
        PRESET_CONFIGS[cname] = make_preset(
            preset_name=cname,
            D=args.D if args.D is not None else base["D"],
            Gamma0=args.Gamma0 if args.Gamma0 is not None else base["Gamma0"],
            GammaS0=args.GammaS0 if args.GammaS0 is not None else base["GammaS0"],
            D_S_cut=args.D_S if args.D_S is not None else base["D_S_cut"],
            T_PG=args.T_PG if args.T_PG is not None else base["T_PG"],
            T_S_fixed=args.T_S if args.T_S is not None else base["T_S_fixed"],
        )
        presets_to_run = [cname]
    elif args.preset is not None:
        presets_to_run = list(PRESET_CONFIGS.keys()) if args.preset == "all" else [args.preset]
    elif isinstance(ACTIVE_PRESET, (list, tuple)):
        presets_to_run = list(ACTIVE_PRESET)
    elif ACTIVE_PRESET == "all":
        presets_to_run = list(PRESET_CONFIGS.keys())
    else:
        presets_to_run = [ACTIVE_PRESET]

    for preset_key in presets_to_run:
        cfg = dict(PRESET_CONFIGS[preset_key])
        cfg["preset_name"] = preset_key
        suffix = cfg.get("file_suffix", "")

        for profile_mode in ("flat_band", "lorentzian"):
            res_core = run_step_1b_core_verification(cfg, probe_profile=profile_mode)
            res_stress = run_step_1b_stress_tests(cfg, probe_profile=profile_mode)

            fig_path = os.path.join(
                RESULTS_DIR, f"phase2_verification_{profile_mode}{suffix}.png"
            )
            plot_phase2_results(res_core, res_stress, filename=fig_path)

            table_path = os.path.join(
                RESULTS_DIR, f"phase2_summary_{profile_mode}{suffix}.txt"
            )
            with open(table_path, "w", encoding="utf-8") as f_out:
                f_out.write("\n".join(res_core["log_lines"] + res_stress["log_lines"]) + "\n")
            print(f"Saved Phase 2 ({profile_mode}, {preset_key}) summary table to '{table_path}'.\n", flush=True)