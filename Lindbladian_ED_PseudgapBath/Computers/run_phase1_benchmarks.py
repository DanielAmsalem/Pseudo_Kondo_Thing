"""
run_phase1_benchmarks.py
------------------------
Executes Phase 1 (Step 1c: QRT & Spectral Function Verification, and Step 1a:
Single-Bath Maxwell Entropy S(eps_d) and S(0)/ln 2 Verification) with N_PG = 10,000
tight-binding modes across all 7 exponents r in {-0.9, -0.5, -0.3, 0.0, +0.3, +0.5, +0.9}.

HARDWIRED IMMUTABLE UV & IR CAPS (Independent of D):
  - UV Bath Cap:                    D*    = 1.0
  - IR Minimum Scale Cap:           T_min = 1.0e-4

SYNCHRONIZED WITH UPDATED `bath.py` AND `spectral_qrt.py`:
  1. Uses `gamma_rule="capped_adaptive"` (where `bath.py` caps base broadenings at
     2 * c_gamma * delta) together with the frequency-local resolution cap in
     `bare_bath_hybridization_dl` (`spectral_qrt.py`), completely eliminating the
     1/eps_q^2 UV Lorentzian tail floor in Gamma_{DL}(w) for r >= 0.5 in Panel (b)
     and in the low-temperature entropy S_{DL}(eps_d) in Panel (f).
  2. Uses `tight_binding_S_eps_d(..., include_out_of_band_poles=True)` from `spectral_qrt.py`
     so the discrete tight-binding entropy retains out-of-band bound-state poles at |w| > D_bath
     and recovers S(0)/ln 2 -> +1 at high T in Panels (g)-(h) without artificial drooping.
  3. Includes a zoomed low-energy inset in Panel (f) so the narrow r > 0 entropy curves
     are clearly resolved alongside the wide r < 0 curves.
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
    dot_spectral_function_dl_freq,
    dot_retarded_gf_qrt_time,
    fourier_transform_qrt_to_spectral,
    tight_binding_S_eps_d,
    continuum_self_energy_infinite_D,
    continuum_self_energy_finite_D,
    continuum_spectral_function,
    analytic_S_eps_d,
    analytic_S0_note_eq7,
)

RESULTS_DIR = (
    "/home/amsalda/PseudoGap/results"
    if os.path.exists("/home/amsalda/PseudoGap")
    else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
)
os.makedirs(RESULTS_DIR, exist_ok=True)

PRESETS_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase1_presets.json")

R_LIST_ALL = [-0.9, -0.5, -0.3, 0.0, 0.3, 0.5, 0.9]
TARGET_N_MODES = 10000
LN2 = np.log(2.0)

# ============================================================================
# HARDWIRED IMMUTABLE UV & IR CAPS (Never change with D)
# ============================================================================
D_STAR: float = 1.0
T_MIN: float = 1.0e-4


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
    D_bath: float = D_STAR,
    T_fixed: float = 1.0e-3,
    T_over_D_range: Optional[Tuple[float, float]] = None,
    D_bath_over_D_list: Optional[List[float]] = None,
    file_suffix: Optional[str] = None,
    description: str = "",
) -> Dict[str, Any]:
    """Build a complete, validated Phase 1 preset dictionary within [T_MIN, D_STAR]."""
    for v, n in [(D, "D"), (Gamma0, "Gamma0"), (D_bath, "D_bath"), (T_fixed, "T_fixed")]:
        assert_within_caps(v, n)

    if T_over_D_range is None:
        T_over_D_range = (T_MIN / D, D_STAR / D)

    if D_bath_over_D_list is None:
        db_abs = sorted(set([
            max(T_MIN, 1.0e-3 * D_STAR),
            max(T_MIN, 1.0e-2 * D_STAR),
            max(T_MIN, 1.0e-1 * D_STAR),
            D_STAR,
        ]))
        D_bath_over_D_list = [val / D for val in db_abs]

    if file_suffix is None:
        file_suffix = "" if preset_name == "low_T_default" else f"_{preset_name}"

    return {
        "preset_name": preset_name,
        "description": description or f"Preset '{preset_name}' (D={D:g}, Gamma0={Gamma0:g}, T_fixed={T_fixed:g})",
        "D": float(D),
        "Gamma0": float(Gamma0),
        "D_bath": float(D_bath),
        "T_fixed": float(T_fixed),
        "T_over_D_range": (float(T_over_D_range[0]), float(T_over_D_range[1])),
        "D_bath_over_D_list": [float(x) for x in D_bath_over_D_list],
        "file_suffix": file_suffix,
    }


# ============================================================================
# SELECT ACTIVE PRESET HERE (No need to edit your .sbatch file)
# Options: "low_T_default", "high_T_plateau", "scaled_D01", or "all"
# ============================================================================
ACTIVE_PRESET: Union[str, List[str]] = "low_T_default"

PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    # 1. Active Default: D_bath = D = D* = 1.0, Gamma0 = 0.1 D, T_fixed = 1e-3 D
    "low_T_default": make_preset(
        preset_name="low_T_default",
        description="Single-bath default with D_bath = D = D* = 1.0, Gamma0 = 0.1D, T_fixed = 1e-3D",
        D=D_STAR,
        Gamma0=D_STAR / 10.0,
        D_bath=D_STAR,
        T_fixed=1.0e-3 * D_STAR,
        T_over_D_range=(1.0e-4, 1.0),
        D_bath_over_D_list=[0.001, 0.01, 0.1, 1.0],
        file_suffix="",
    ),
    # 2. High-T Plateau Preset: D = 0.05 = D*/20, Gamma0 = D = 0.05, D_bath = 20 D = D*
    "high_T_plateau": make_preset(
        preset_name="high_T_plateau",
        description="Single-bath high-T S=ln2 plateau preset (D = D*/20 = 0.05, Gamma0 = D = 0.05, D_bath = 20D = D*)",
        D=0.05,
        Gamma0=0.05,
        D_bath=D_STAR,
        T_fixed=0.01 * 0.05,
        T_over_D_range=(2.0e-3, 20.0),
        D_bath_over_D_list=[0.1, 1.0, 10.0, 20.0],
        file_suffix="_high_T_plateau",
    ),
    # 3. Rescaled D = D*/10 = 0.1 Preset:
    "scaled_D01": make_preset(
        preset_name="scaled_D01",
        description="Rescaled D = D*/10 = 0.1, Gamma0 = 2.5D = 0.25, D_bath = 10D = D*, T_fixed = 0.01D",
        D=0.1,
        Gamma0=0.25,
        D_bath=D_STAR,
        T_fixed=1.0e-3,
        T_over_D_range=(1.0e-3, 10.0),
        D_bath_over_D_list=[0.01, 0.1, 1.0, 10.0],
        file_suffix="_scaled_D01",
    ),
}


def sync_presets_json() -> None:
    """
    Keep in-code `PRESET_CONFIGS` as the source of truth (so editing Python
    always takes effect immediately), while loading any extra custom presets
    added to `phase1_presets.json`.
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


def get_color_map(r_values) -> dict:
    """Assign distinct, high-contrast colors sorted from negative to positive r."""
    unique_r = sorted(set(float(r) for r in r_values))
    cmap = plt.cm.turbo(np.linspace(0.08, 0.92, len(unique_r)))
    return {r: cmap[i] for i, r in enumerate(unique_r)}


def build_single_bath_solver(
    r: float,
    T: float,
    Gamma0: float,
    D_bath: float = D_STAR,
    n_modes: int = TARGET_N_MODES,
    gamma_rule: Literal["adaptive", "capped_adaptive", "uniform"] = "capped_adaptive",
) -> Tuple[Bath, LyapunovSolver]:
    """
    Construct a single-bath Bath and LyapunovSolver:
      - Default `gamma_rule="capped_adaptive"` sets `bath.gammas = min(c_gamma * bin_widths, 2 * c_gamma * delta)`.
        In `bare_bath_hybridization_dl` (`spectral_qrt.py`), this base cap is combined with the
        frequency-local cap `w_loc_cap = c_gamma * max(delta, |w| * ln(Lambda))`, ensuring both
        zero UV tail leakage at low |w| and smooth bin overlap across [-D_bath, +D_bath].
      - For the real-time QRT matrix exponential check (`solver_qrt`), pass `gamma_rule="adaptive"`
        so the frequency-independent matrix A = -i H - Lambda^{(+)} has the same widths as
        `bare_bath_hybridization_dl` at |w| >= 0.05 D_bath.
    """
    assert_within_caps(T, "T")
    assert_within_caps(Gamma0, "Gamma0")
    assert_within_caps(D_bath, "D_bath")

    D_star_pg = min(0.25 * D_bath, max(8.0 * min(T, D_bath), 5.0 * T_MIN))
    bath = Bath(
        name="PG",
        Gamma0=Gamma0,
        r=r,
        T=T,
        D=D_bath,
        discretization="log_linear",
        target_n_modes=n_modes,
        D_star=D_star_pg,
        gamma_rule=gamma_rule,
        c_gamma=1.0,
    )
    return bath, LyapunovSolver([bath])


def run_step_1c_verification(cfg: Dict[str, Any], r_list=R_LIST_ALL) -> dict:
    """
    Execute Step 1c: Bare bath hybridization cutoffs, Kramers-Kronig self-energy,
    dressed frequency-space spectral function A(w), and real-time QRT propagator G^R(t).
    """
    D = float(cfg["D"])
    D_bath = float(cfg["D_bath"])
    Gamma0 = float(cfg["Gamma0"])
    T_fixed = float(cfg["T_fixed"])
    eps_d_test = 0.0

    header = (
        "=" * 116 + "\n"
        f"STEP 1c (SPECTRAL & QRT — Preset: {cfg['preset_name']}): "
        f"D={D:g}, D_bath={D_bath:g} ({D_bath/D:g}D), Gamma0={Gamma0:g} ({Gamma0/D:g}D), T={T_fixed:.0e}\n"
        + "=" * 116
    )
    print(header, flush=True)
    log_lines = [header]

    col_hdr = (
        f"{'r':>6} | {'N_modes':>7} | {'gamma_min':>11} | "
        f"{'RelErr Gamma(w)':>15} | {'MaxDiff QRT-FFT vs Freq':>23} | {'SumRule Int A(w)dw':>18}"
    )
    sep = "-" * 116
    print(col_hdr, flush=True)
    print(sep, flush=True)
    log_lines.extend([col_hdr, sep])

    w_lin = np.linspace(-1.3 * D_bath, 1.3 * D_bath, 1201)
    w_pos_log = np.logspace(np.log10(T_MIN), np.log10(1.3 * D_bath), 300)
    w_full_sorted = np.unique(np.concatenate((-w_pos_log[::-1], [0.0], w_lin, w_pos_log)))

    # QRT time grid and even-length frequency grid (avoiding the w=0 singular spike for r>0)
    t_max = 250.0 / max(Gamma0 * (D_bath ** max(r_list)), 0.02 * D_bath)
    t_grid = np.linspace(0.0, t_max, 4000)
    w_qrt = np.linspace(-0.8 * D_bath, 0.8 * D_bath, 400)

    results_1c = {
        "w_lin": w_lin,
        "w_pos_log": w_pos_log,
        "w_qrt": w_qrt,
        "t_grid": t_grid,
        "by_r": {},
    }

    for r in r_list:
        # Use gamma_rule="capped_adaptive" so bare_bath_hybridization_dl in spectral_qrt.py
        # applies its frequency-local cap w_loc_cap, eliminating UV tail leakage in Panel (b)!
        bath, solver = build_single_bath_solver(
            r=r, T=T_fixed, Gamma0=Gamma0, D_bath=D_bath, n_modes=TARGET_N_MODES, gamma_rule="capped_adaptive"
        )
        sig_dl_lin, gam_dl_lin = bare_bath_hybridization_dl(bath, w_lin)
        _, gam_dl_log = bare_bath_hybridization_dl(bath, w_pos_log)

        gam_cont_lin = np.where(np.abs(w_lin) <= D_bath, bath.continuum_gamma(w_lin), 0.0)
        sig_cont_fin = continuum_self_energy_finite_D(w_lin, Gamma0, r, D_bath)
        sig_cont_inf = continuum_self_energy_infinite_D(w_lin, Gamma0, r)
        gam_cont_log = np.where(w_pos_log <= D_bath, bath.continuum_gamma(w_pos_log), 0.0)

        mask_in = (np.abs(w_lin) > 0.02 * D_bath) & (np.abs(w_lin) < 0.90 * D_bath)
        rel_err_gam = float(
            np.median(np.abs(gam_dl_lin[mask_in] - gam_cont_lin[mask_in]) / np.maximum(gam_cont_lin[mask_in], 1e-12))
        )

        # Spectral function A(w) and sum rule on stretched grid
        _, A_dl_full = dot_spectral_function_dl_freq(solver, eps_d_test, w_full_sorted)
        sum_rule = float(np.trapz(A_dl_full, w_full_sorted))
        _, A_dl_qrt_grid = dot_spectral_function_dl_freq(solver, eps_d_test, w_qrt)
        A_cont_qrt_grid = continuum_spectral_function(eps_d_test, w_qrt, Gamma0, r, D=D_bath)

        # Real-time QRT vs frequency-space cross-check on N=1200 bath
        _, solver_qrt = build_single_bath_solver(
            r=r, T=T_fixed, Gamma0=Gamma0, D_bath=D_bath, n_modes=1200, gamma_rule="adaptive"
        )
        G_ret_t = dot_retarded_gf_qrt_time(solver_qrt, eps_d_test, t_grid)
        _, A_qrt_fft = fourier_transform_qrt_to_spectral(t_grid, G_ret_t, w_qrt)
        _, A_freq_1200 = dot_spectral_function_dl_freq(solver_qrt, eps_d_test, w_qrt)
        mask_qrt = np.abs(w_qrt) >= 0.05 * D_bath
        max_diff_qrt = float(np.max(np.abs(A_qrt_fft[mask_qrt] - A_freq_1200[mask_qrt])))

        results_1c["by_r"][r] = {
            "sig_dl_lin": sig_dl_lin,
            "gam_dl_lin": gam_dl_lin,
            "gam_cont_lin": gam_cont_lin,
            "sig_cont_fin": sig_cont_fin,
            "sig_cont_inf": sig_cont_inf,
            "gam_dl_log": gam_dl_log,
            "gam_cont_log": gam_cont_log,
            "A_dl_qrt_grid": A_dl_qrt_grid,
            "A_cont_qrt_grid": A_cont_qrt_grid,
            "A_qrt_fft": A_qrt_fft,
            "G_ret_t": G_ret_t,
            "sum_rule": sum_rule,
        }

        row_str = (
            f"{r:6.2f} | {bath.n_modes:7d} | {bath.gamma_min:11.3e} | "
            f"{rel_err_gam:15.4e} | {max_diff_qrt:23.4e} | {sum_rule:18.6f}"
        )
        print(row_str, flush=True)
        log_lines.append(row_str)
        gc.collect()

    print(flush=True)
    results_1c["log_lines"] = log_lines
    return results_1c


def run_step_1a_verification(cfg: Dict[str, Any], r_list=R_LIST_ALL) -> dict:
    """
    Execute Step 1a: Single-Bath Tight-Binding & Continuum Maxwell Entropy S(eps_d), S(0)/ln 2.
    Uses `tight_binding_S_eps_d` from `spectral_qrt.py` so that:
      1. At low T, frequency-local broadening eliminates UV tail leakage for r > 0.
      2. At high T (T >= D_bath / 5), the out-of-band tight-binding bound-state poles
         (|w| > D_bath) are retained (`include_out_of_band_poles=True`), recovering
         S(0)/ln 2 -> +1 without drooping!
    """
    D = float(cfg["D"])
    D_bath = float(cfg["D_bath"])
    Gamma0 = float(cfg["Gamma0"])
    T_fixed = float(cfg["T_fixed"])
    t_min_over_D, t_max_over_D = cfg["T_over_D_range"]

    header = (
        "=" * 124 + "\n"
        f"STEP 1a (SINGLE-BATH ENTROPY — Preset: {cfg['preset_name']}): "
        f"D={D:g}, D_bath={D_bath:g} ({D_bath/D:g}D), Gamma0={Gamma0:g} ({Gamma0/D:g}D), T_fixed={T_fixed:.0e} ({T_fixed/D:g}D)\n"
        + "=" * 124
    )
    print(header, flush=True)
    log_lines = ["", header]

    # Solver cross-check across 21 eps_d points
    _, sol_chk = build_single_bath_solver(
        r=0.3, T=T_fixed, Gamma0=Gamma0, D_bath=D_bath, n_modes=600, gamma_rule="capped_adaptive"
    )
    ed_chk_list = np.linspace(-0.3 * D_bath, 0.3 * D_bath, 21)
    max_err_fd = 0.0
    max_err_schur = 0.0
    for ed in ed_chk_list:
        v_lyap = sol_chk.dn_d_dT(float(ed), method="exact_lyapunov")
        v_fd = sol_chk.dn_d_dT(float(ed), method="finite_diff", rel_dT=1e-4)
        v_schur = sol_chk.dn_d_dT(float(ed), method="schur_lddl")
        max_err_fd = max(max_err_fd, abs(v_fd - v_lyap))
        max_err_schur = max(max_err_schur, abs(v_schur - v_lyap))
    chk_str = (
        f"Single-Bath Solver Cross-Check across 21 eps_d points (M={sol_chk.dim}, r=+0.3):\n"
        f"  Max |FD - MatrixLyap| = {max_err_fd:.3e}, Max |SchurLDDL - MatrixLyap| = {max_err_schur:.3e}\n"
    )
    print(chk_str, flush=True)
    log_lines.append(chk_str)

    col_hdr = (
        f"{'r':>6} | {'N_modes':>7} | {'eps_max/D':>9} | "
        f"{'S_TB(0)/ln2':>15} | {'S_CanonFinD/ln2':>15} | {'S_BandTrunc/ln2':>15} | {'S_Eq7(0)/ln2':>15} | {'Target r':>8}"
    )
    sep = "-" * 124
    print(col_hdr, flush=True)
    print(sep, flush=True)
    log_lines.extend([col_hdr, sep])

    results_1a = {
        "preset_name": cfg["preset_name"],
        "D": D,
        "D_bath": D_bath,
        "D_star": D_STAR,
        "T_min": T_MIN,
        "Gamma0": Gamma0,
        "T_fixed": T_fixed,
        "T_over_D_range": (t_min_over_D, t_max_over_D),
        "sweeps": {},
        "T_scans": {},
        "cutoff_scans": {},
    }

    for r in sorted(r_list, reverse=True):
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        eff_width_T = Gamma0 * (T_fixed ** r) * sec_pr2
        eps_max = max(25.0 * D_bath, 50.0 * eff_width_T)

        u = np.linspace(-1.0, 0.0, 161)
        sinh_scale = np.arcsinh(eps_max / (0.05 * T_fixed))
        eps_d_grid = (0.05 * T_fixed) * np.sinh(sinh_scale * u)

        bath, _ = build_single_bath_solver(
            r=r, T=T_fixed, Gamma0=Gamma0, D_bath=D_bath, n_modes=TARGET_N_MODES, gamma_rule="capped_adaptive"
        )
        dn_dT_tb, S_tb_curve = tight_binding_S_eps_d(
            bath, eps_d_grid, T_fixed, eps_max=eps_max, include_out_of_band_poles=True
        )
        S_num_0 = float(S_tb_curve[-1])

        S_canon_curve = analytic_S_eps_d(
            eps_d_grid, T_fixed, Gamma0, r, D=D_bath, eps_max=eps_max, include_out_of_band_poles=True
        )
        S_trunc_0 = float(
            analytic_S_eps_d(0.0, T_fixed, Gamma0, r, D=D_bath, eps_max=eps_max, include_out_of_band_poles=False)
        )
        S_canon_0 = float(S_canon_curve[-1])
        S_eq7_0 = analytic_S0_note_eq7(T_fixed, Gamma0, r)

        results_1a["sweeps"][r] = {
            "eps_d": eps_d_grid,
            "dn_dT": dn_dT_tb,
            "S_eps_d": S_tb_curve,
            "S_canon_curve": S_canon_curve,
            "S_num_0": S_num_0,
            "S_trunc_0": S_trunc_0,
            "S_canon_0": S_canon_0,
            "S_eq7_0": S_eq7_0,
        }

        row_str = (
            f"{r:6.2f} | {bath.n_modes:7d} | {eps_max/D:9.1f} | "
            f"{S_num_0/LN2:15.5f} | {S_canon_0/LN2:15.5f} | "
            f"{S_trunc_0/LN2:15.5f} | {S_eq7_0/LN2:15.5f} | {r:8.4f}"
        )
        print(row_str, flush=True)
        log_lines.append(row_str)
        gc.collect()

    # Temperature Scan across T / D in cfg["T_over_D_range"]
    T_grid = np.logspace(np.log10(t_min_over_D * D), np.log10(t_max_over_D * D), 9)
    T_dense = np.logspace(np.log10(t_min_over_D * D), np.log10(t_max_over_D * D), 60)
    assert_within_caps(T_grid, "T_grid")

    for r in r_list:
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        S_num_T = []
        S_num_trunc_T = []
        for T_val in T_grid:
            tv = float(T_val)
            eff_w = Gamma0 * (tv ** r) * sec_pr2
            eps_max_T = max(25.0 * D_bath, 50.0 * eff_w)

            bath_T, _ = build_single_bath_solver(
                r=r, T=tv, Gamma0=Gamma0, D_bath=D_bath, n_modes=TARGET_N_MODES, gamma_rule="capped_adaptive"
            )
            val_full = float(
                tight_binding_S_eps_d(bath_T, 0.0, tv, eps_max=eps_max_T, include_out_of_band_poles=True)
            )
            val_trunc = float(
                tight_binding_S_eps_d(bath_T, 0.0, tv, eps_max=eps_max_T, include_out_of_band_poles=False)
            )
            S_num_T.append(val_full / LN2)
            S_num_trunc_T.append(val_trunc / LN2)

        S_canon_dense = np.array([
            float(analytic_S_eps_d(
                0.0, float(t), Gamma0, r, D=D_bath,
                eps_max=max(25.0 * D_bath, 50.0 * Gamma0 * (float(t) ** r) * sec_pr2),
                include_out_of_band_poles=True,
            )) / LN2
            for t in T_dense
        ])
        S_trunc_dense = np.array([
            float(analytic_S_eps_d(
                0.0, float(t), Gamma0, r, D=D_bath,
                eps_max=max(25.0 * D_bath, 50.0 * Gamma0 * (float(t) ** r) * sec_pr2),
                include_out_of_band_poles=False,
            )) / LN2
            for t in T_dense
        ])
        S_eq7_dense = np.array([analytic_S0_note_eq7(float(t), Gamma0, r) / LN2 for t in T_dense])

        results_1a["T_scans"][r] = {
            "T_grid": T_grid,
            "S_num": np.array(S_num_T),
            "S_num_trunc": np.array(S_num_trunc_T),
            "T_dense": T_dense,
            "S_canon_dense": S_canon_dense,
            "S_trunc_dense": S_trunc_dense,
            "S_eq7_dense": S_eq7_dense,
        }
        gc.collect()

    # Panel (h): Comparing Bath Cutoffs D_bath / D across T / D
    r_focus = [-0.5, -0.3, 0.3, 0.5]
    D_bath_list = [float(db_mult) * D for db_mult in cfg["D_bath_over_D_list"]]
    for _db in D_bath_list:
        assert_within_caps(_db, "D_bath_list element")

    results_1a["cutoff_scans"] = {
        "D_bath_list": D_bath_list,
        "by_r": {},
    }
    for r in r_focus:
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        num_canon_by_db = {db: [] for db in D_bath_list}
        num_trunc_by_db = {db: [] for db in D_bath_list}
        trunc_by_db = {db: [] for db in D_bath_list}
        canon_by_db = {db: [] for db in D_bath_list}

        for T_val in T_grid:
            tv = float(T_val)
            eff_w = Gamma0 * (tv ** r) * sec_pr2
            for db_val in D_bath_list:
                eps_max_T = max(25.0 * db_val, 50.0 * eff_w)
                bath_db, _ = build_single_bath_solver(
                    r=r, T=tv, Gamma0=Gamma0, D_bath=db_val, n_modes=4000, gamma_rule="capped_adaptive"
                )
                num_canon_by_db[db_val].append(
                    float(tight_binding_S_eps_d(bath_db, 0.0, tv, eps_max=eps_max_T, include_out_of_band_poles=True)) / LN2
                )
                num_trunc_by_db[db_val].append(
                    float(tight_binding_S_eps_d(bath_db, 0.0, tv, eps_max=eps_max_T, include_out_of_band_poles=False)) / LN2
                )

        for T_val in T_dense:
            tv = float(T_val)
            eff_w = Gamma0 * (tv ** r) * sec_pr2
            for db_val in D_bath_list:
                eps_max_T = max(25.0 * db_val, 50.0 * eff_w)
                trunc_by_db[db_val].append(
                    float(analytic_S_eps_d(
                        0.0, tv, Gamma0, r, D=db_val, eps_max=eps_max_T, include_out_of_band_poles=False
                    )) / LN2
                )
                canon_by_db[db_val].append(
                    float(analytic_S_eps_d(
                        0.0, tv, Gamma0, r, D=db_val, eps_max=eps_max_T, include_out_of_band_poles=True
                    )) / LN2
                )

        results_1a["cutoff_scans"]["by_r"][r] = {
            "T_grid": T_grid,
            "T_dense": T_dense,
            "num_canon_by_db": {db: np.array(v) for db, v in num_canon_by_db.items()},
            "num_trunc_by_db": {db: np.array(v) for db, v in num_trunc_by_db.items()},
            "trunc_by_db": {db: np.array(v) for db, v in trunc_by_db.items()},
            "canon_by_db": {db: np.array(v) for db, v in canon_by_db.items()},
        }
        gc.collect()

    print(flush=True)
    results_1a["log_lines"] = log_lines
    return results_1a


def plot_phase1_results(res_1c: dict, res_1a: dict, filename: str) -> None:
    all_r = sorted(res_1a["sweeps"].keys())
    colors = get_color_map(all_r)
    D = res_1a["D"]
    D_bath = res_1a["D_bath"]
    D_cap = res_1a["D_star"]
    Gamma0 = res_1a["Gamma0"]
    T_fixed = res_1a["T_fixed"]
    x_min_T, x_max_T = res_1a["T_over_D_range"]

    fig, axes = plt.subplots(2, 4, figsize=(24, 12.8))

    # Panel (a): Step 1c Bare Hybridization Gamma(w) on Linear Grid [-1.3 D_bath, +1.3 D_bath]
    ax = axes[0, 0]
    w_lin = res_1c["w_lin"]
    for r in all_r:
        cdata = res_1c["by_r"][r]
        ax.plot(w_lin / D_bath, cdata["gam_cont_lin"], "--", color="k", lw=1.0, alpha=0.55)
        ax.plot(w_lin / D_bath, cdata["gam_dl_lin"], "-", color=colors[r], lw=1.7)
    ax.axvline(-1.0, color="gray", ls=":", lw=1.1)
    ax.axvline(1.0, color="gray", ls=":", lw=1.1)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(0.0, 3.0 * Gamma0 * (D_bath ** min(0.0, max(all_r))))
    ax.set_xlabel(r"Frequency $\omega / D_{\mathrm{bath}}$")
    ax.set_ylabel(r"Hybridization $\Gamma(\omega) = -\mathrm{Im}\,\Delta^R(\omega)$")
    ax.set_title(r"(a) Step 1c: Bare Hybridization $\Gamma_{DL}(\omega)$ & UV Cutoffs")
    style_a = [
        Line2D([0], [0], color="k", ls="-", lw=1.7, label=r"Tight-Binding $\Gamma_{DL}(\omega)$ ($N=10\mathrm{k}$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.0, alpha=0.7, label=r"Continuum $\Gamma_0 |\omega|^r \Theta(D_{\mathrm{bath}}-|\omega|)$"),
    ]
    ax.legend(handles=style_a, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.3, loc="upper right")
    ax.grid(True, alpha=0.25)

    # Panel (b): Step 1c Log-Log Hybridization Power-Law Scaling
    ax = axes[0, 1]
    w_pos_log = res_1c["w_pos_log"]
    for r in all_r:
        cdata = res_1c["by_r"][r]
        ax.loglog(w_pos_log / D_bath, cdata["gam_cont_log"], "--", color="k", lw=1.0, alpha=0.55)
        ax.loglog(w_pos_log / D_bath, cdata["gam_dl_log"], "-", color=colors[r], lw=1.7)
    ax.axvline(1.0, color="gray", ls=":", lw=1.2)
    ax.set_xlim(T_MIN / D_bath, 1.3)
    ax.set_ylim(1e-4 * Gamma0, 1e3 * Gamma0)
    ax.set_xlabel(r"Positive Frequency $\omega / D_{\mathrm{bath}}$ (log scale)")
    ax.set_ylabel(r"Hybridization $\Gamma(\omega)$ (log scale)")
    ax.set_title(r"(b) Step 1c: Power-Law Scaling $\Gamma(\omega) = \Gamma_0 |\omega|^r$ (Log-Log)")
    style_b = [
        Line2D([0], [0], color="k", ls="-", lw=1.7, label=r"Tight-Binding $\Gamma_{DL}(\omega)$ ($N=10\mathrm{k}$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.0, alpha=0.7, label=r"Exact Power Law $\Gamma_0 \omega^r$"),
        Line2D([0], [0], color="gray", ls=":", lw=1.2, label=r"UV Band Edge $\omega = D_{\mathrm{bath}}$"),
    ]
    ax.legend(handles=style_b, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.2, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)

    # Panel (c): Step 1c Real Self-Energy Sigma(w) (Finite-D vs Infinite-D vs Tight-Binding)
    ax = axes[0, 2]
    for r in all_r:
        if np.isclose(r, 0.0):
            continue
        cdata = res_1c["by_r"][r]
        ax.plot(w_lin / D_bath, cdata["sig_cont_inf"], ":", color=colors[r], lw=1.1, alpha=0.65)
        ax.plot(w_lin / D_bath, cdata["sig_cont_fin"], "--", color="k", lw=1.0, alpha=0.55)
        ax.plot(w_lin / D_bath, cdata["sig_dl_lin"], "-", color=colors[r], lw=1.7)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-2.5 * Gamma0, 2.5 * Gamma0)
    ax.set_xlabel(r"Frequency $\omega / D_{\mathrm{bath}}$")
    ax.set_ylabel(r"Real Self-Energy $\Sigma(\omega) = \mathrm{Re}\,\Delta^R(\omega)$")
    ax.set_title(r"(c) Step 1c: Kramers-Kronig Self-Energy $\Sigma_{DL}(\omega)$ vs $\Sigma_D(\omega)$")
    style_c = [
        Line2D([0], [0], color="k", ls="-", lw=1.7, label=r"Tight-Binding $\Sigma_{DL}(\omega)$ ($N=10\mathrm{k}$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.0, alpha=0.7, label=r"Exact Finite-Cutoff $\Sigma_D(\omega)$ (with $1/\omega$ tail)"),
        Line2D([0], [0], color="k", ls=":", lw=1.2, label=r"Unbounded Note Eq. (4) $\Sigma_\infty(\omega)$"),
    ]
    ax.legend(handles=style_c, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.0, loc="upper right")
    ax.grid(True, alpha=0.25)

    # Panel (d): Step 1c Dressed Spectral Function A(w) & Real-Time QRT FFT
    ax = axes[0, 3]
    w_qrt = res_1c["w_qrt"]
    for r in [-0.5, 0.0, 0.3, 0.5]:
        cdata = res_1c["by_r"][r]
        ax.plot(w_qrt / D_bath, cdata["A_cont_qrt_grid"], "--", color="k", lw=1.0, alpha=0.55)
        ax.plot(w_qrt / D_bath, cdata["A_dl_qrt_grid"], "-", color=colors[r], lw=1.8)
        ax.plot(w_qrt[::8] / D_bath, cdata["A_qrt_fft"][::8], "o", color=colors[r], ms=4.2, fillstyle="none")
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(0.0, 3.5 / Gamma0)
    ax.set_xlabel(r"Frequency $\omega / D_{\mathrm{bath}}$")
    ax.set_ylabel(r"Dot Spectral Function $A(\epsilon_d=0; \omega)$")
    ax.set_title(r"(d) Step 1c: Spectral Function $A(\omega)$ (Freq. vs Real-Time QRT)")
    style_d = [
        Line2D([0], [0], color="k", ls="-", lw=1.8, label=r"Freq.-Space LDDL Eq. (50a)"),
        Line2D([0], [0], color="k", marker="o", ls="None", ms=4.5, fillstyle="none", label=r"Real-Time QRT FFT ($-i\theta(t)[e^{At}]_{dd}$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.0, alpha=0.7, label=r"Continuum Note Eq. (1)"),
    ]
    ax.legend(handles=style_d, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.0, loc="upper right")
    ax.grid(True, alpha=0.25)

    # Panel (e): Step 1a Maxwell Derivative dN/dT vs eps_d / D
    ax = axes[1, 0]
    for r in all_r:
        sw = res_1a["sweeps"][r]
        y_vals = sw["dn_dT"] * D
        ax.plot(sw["eps_d"] / D, y_vals, "-", color=colors[r], lw=1.8)
    xlim_e = -min(D_bath / D, max(60.0 * (T_fixed / D), 0.06 * (D_bath / D)))
    ax.set_xlim(xlim_e, 0.0)
    ax.set_ylim(-75.0, 3.0)
    ax.set_xlabel(r"Impurity Energy $\epsilon_d / D$")
    ax.set_ylabel(r"Dimensionless Derivative $D\,\frac{\partial n_d}{\partial T}$")
    ax.set_title(f"(e) Step 1a: Derivative $\\partial n_d / \\partial T$ ($T/D={T_fixed/D:g}$)")
    style_e = [
        Line2D([0], [0], color="k", ls="-", lw=1.8, label=f"Tight-Binding ($N=10\\mathrm{{k}}, D_{{\\mathrm{{bath}}}}={D_bath/D:g}D$)"),
    ]
    ax.legend(handles=style_e, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.3, loc="lower left")
    ax.grid(True, alpha=0.25)

    # Panel (f): Step 1a Maxwell Entropy Curve S(eps_d) / ln 2 vs eps_d / D (with low-energy zoom inset for r > 0)
    ax = axes[1, 1]
    for r in all_r:
        sw = res_1a["sweeps"][r]
        ax.plot(sw["eps_d"] / D, sw["S_canon_curve"] / LN2, "--", color="k", lw=1.1, alpha=0.60)
        ax.plot(sw["eps_d"] / D, sw["S_eps_d"] / LN2, "-", color=colors[r], lw=1.9)
        ax.axhline(r, color=colors[r], ls=":", alpha=0.65)
    ax.set_xlim(-D_bath / D, 0.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Impurity Energy $\epsilon_d / D$")
    ax.set_ylabel(r"Normalized Entropy $S(\epsilon_d) / \ln 2$")
    ax.set_title(f"(f) Step 1a: Maxwell Entropy $S(\\epsilon_d)/\\ln 2$ ($T/D={T_fixed/D:g}$)")
    style_f = [
        Line2D([0], [0], color="k", ls="-", lw=1.9, label=r"Tight-Binding ($-\int \partial_T n_d\,d\epsilon_d'$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.1, alpha=0.7, label=r"Canonical Finite-$D$ Integral (Eq. 5)"),
        Line2D([0], [0], color="k", ls=":", lw=1.2, label=r"Low-$T$ Asymptote: $S(0)/\ln 2 \to r$ (Eq. 7)"),
    ]
    ax.legend(handles=style_f, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.0, loc="upper left")
    ax.grid(True, alpha=0.25)

    # Zoom inset in Panel (f) resolving the narrow thermal resonance window [xlim_e, 0] for r >= 0
    ax_ins_f = ax.inset_axes([0.12, 0.50, 0.46, 0.32])
    for r in [rv for rv in all_r if rv >= 0.0]:
        sw = res_1a["sweeps"][r]
        ax_ins_f.plot(sw["eps_d"] / D, sw["S_canon_curve"] / LN2, "--", color="k", lw=1.0, alpha=0.65)
        ax_ins_f.plot(sw["eps_d"] / D, sw["S_eps_d"] / LN2, "-", color=colors[r], lw=1.6)
        ax_ins_f.axhline(r, color=colors[r], ls=":", alpha=0.55)
    ax_ins_f.set_xlim(xlim_e, 0.0)
    ax_ins_f.set_ylim(-0.05, 1.02)
    ax_ins_f.set_title(r"Zoom near $\epsilon_d = 0$ ($r \geq 0$)", fontsize=7.2)
    ax_ins_f.tick_params(labelsize=6.5)
    ax_ins_f.grid(True, alpha=0.25)

    # Panel (g): Step 1a S(0)/ln 2 vs Temperature T / D (Tight-Binding & Canonical climb to +1 without drooping!)
    ax = axes[1, 2]
    for r in all_r:
        tdata = res_1a["T_scans"][r]
        # 1. Canonical Finite-D Continuum (with |w|>D bound state -> reaches +1 without drooping)
        ax.semilogx(tdata["T_dense"] / D, tdata["S_canon_dense"], "-", color=colors[r], lw=1.8, alpha=0.90)
        # 2. Unbounded Note Eq. (7) (D -> inf)
        ax.semilogx(tdata["T_dense"] / D, tdata["S_eq7_dense"], "--", color=colors[r], lw=1.2, alpha=0.70)
        # 3. Band-Truncated Reference (|w| <= D_bath without out-of-band poles)
        ax.semilogx(tdata["T_dense"] / D, tdata["S_trunc_dense"], "-.", color=colors[r], lw=1.1, alpha=0.45)
        # 4. Tight-Binding points (with out-of-band bound states -> reaches +1 without drooping!)
        ax.semilogx(tdata["T_grid"] / D, tdata["S_num"], "o", color=colors[r], ms=5.2)
        ax.axhline(r, color=colors[r], ls=":", alpha=0.40)
    ax.axhline(1.0, color="gray", ls="-.", lw=1.1, alpha=0.6)
    ax.axvline(T_fixed / D, color="dimgray", ls=":", lw=1.1, alpha=0.7)
    ax.set_xlim(x_min_T, x_max_T)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Bath Temperature $T / D$ (log scale)")
    ax.set_ylabel(r"Resonance Entropy $S(0) / \ln 2$")
    ax.set_title(f"(g) Step 1a: $S(0)/\\ln 2$ vs $T/D$ ($D_{{\\mathrm{{bath}}}}={D_bath/D:g}D=D^*$)")
    style_g = [
        Line2D([0], [0], color="k", marker="o", ls="-", lw=1.8, ms=5.0, label=r"Tight-Binding & Canonical Finite-$D$ ($\to +1$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.2, label=r"Unbounded Note Limit (Eq. 7, $D\to\infty \Rightarrow +1$)"),
        Line2D([0], [0], color="k", ls="-.", lw=1.1, alpha=0.55, label=f"Band-Truncated Only ($|\\omega|\\leq D_{{\\mathrm{{bath}}}}={D_bath/D:g}D$)"),
        Line2D([0], [0], color="k", ls=":", lw=1.2, label=r"Asymptotes: Low-$T$ ($r$) & High-$T$ ($+1$)"),
    ]
    ax.legend(handles=style_g, frameon=True, facecolor="white", framealpha=0.85, fontsize=6.4, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)

    # Panel (h): Varying Bath Cutoffs D_bath / D (Canonical Bound-State-Included vs Band-Truncated)
    ax = axes[1, 3]
    D_bath_list = res_1a["cutoff_scans"]["D_bath_list"]
    db_styles = [
        (D_bath_list[3], "-", "o", 1.7, True, f"Cutoff $D_{{\\mathrm{{bath}}}} = {D_bath_list[3]/D:g}D = D^*$"),
        (D_bath_list[2], "--", "s", 1.4, False, f"Cutoff $D_{{\\mathrm{{bath}}}} = {D_bath_list[2]/D:g}D$"),
        (D_bath_list[1], "-.", "d", 1.3, False, f"Cutoff $D_{{\\mathrm{{bath}}}} = {D_bath_list[1]/D:g}D$"),
        (D_bath_list[0], ":", "^", 1.3, False, f"Cutoff $D_{{\\mathrm{{bath}}}} = {D_bath_list[0]/D:g}D$"),
    ]
    for r, cdata in res_1a["cutoff_scans"]["by_r"].items():
        # Faded solid curve & filled circles: Canonical / Tight-Binding with out-of-band poles (-> +1)
        ax.semilogx(
            cdata["T_dense"] / D, cdata["canon_by_db"][D_bath_list[3]],
            "-", color=colors[r], lw=1.2, alpha=0.40
        )
        for db_val, ls_v, mk_v, lw_v, filled, _ in db_styles:
            ax.semilogx(cdata["T_dense"] / D, cdata["trunc_by_db"][db_val], ls=ls_v, color=colors[r], lw=lw_v, alpha=0.85)
            ax.semilogx(
                cdata["T_grid"] / D, cdata["num_trunc_by_db"][db_val],
                marker=mk_v, ls="None", color=colors[r], ms=5.0,
                fillstyle="full" if filled else "none", mew=1.3,
            )
        ax.axhline(r, color=colors[r], ls=":", alpha=0.35)

    ax.axhline(1.0, color="gray", ls="-.", lw=1.0, alpha=0.6)
    for db_val, ls_v, _, _, _, _ in db_styles[1:]:
        ax.axvline(db_val / D, color="dimgray", ls=ls_v, lw=1.1, alpha=0.65)
    ax.set_xlim(x_min_T, x_max_T)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Bath Temperature $T / D$ (log scale)")
    ax.set_ylabel(r"Resonance Entropy $S(0) / \ln 2$")
    db_str_list = ", ".join(f"{db/D:g}" for db in D_bath_list)
    ax.set_title(f"(h) Step 1a: Bath Cutoffs $D_{{\\mathrm{{bath}}}}/D \\in \\{{{db_str_list}\\}}$")
    style_h = [
        Line2D(
            [0], [0], color="k", marker=mk_v, ls=ls_v, ms=5.0, lw=lw_v,
            fillstyle="full" if filled else "none", mew=1.3, label=lbl,
        )
        for _, ls_v, mk_v, lw_v, filled, lbl in db_styles
    ] + [
        Line2D([0], [0], color="gray", ls="-", lw=1.2, alpha=0.5, label=r"Faded Line: With $|\omega|>D_{\mathrm{bath}}$ Poles ($\to +1$)"),
        Line2D([0], [0], color="dimgray", ls="--", lw=1.1, label=r"Vertical Lines: Mode Cutoff $T = D_{\mathrm{bath}}$"),
    ]
    ax.legend(handles=style_h, frameon=True, facecolor="white", framealpha=0.85, fontsize=6.2, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)

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
            f"($\\Gamma(\\omega) = \\Gamma_0 |\\omega|^r$,   Hardwired Caps: $[T_{{\\min}}=10^{{-4}},\\, D^*=1.0]$)\n"
            f"Default Parameters (Preset: {res_1a['preset_name']}, $D = {D:g}, D^* = {D_cap/D:g}D$):  "
            f"$\\Gamma_0 = {Gamma0/D:g}D$,  $D_{{\\mathrm{{bath}}}} = {D_bath/D:g}D$,  $T_{{\\mathrm{{fixed}}}} = {T_fixed/D:g}D$"
        ),
        title_fontsize=10.6,
    )

    fig.savefig(filename, dpi=200, bbox_inches="tight", pad_inches=0.25)

    # -------------------------------------------------------------------------
    # Save each of the 8 panels (and top legend) as individual vectorized PDFs
    # inside /home/amsalda/PseudoGap/results/plots_phase1/
    # -------------------------------------------------------------------------
    plots_dir = os.path.join(RESULTS_DIR, "plots_phase1")
    os.makedirs(plots_dir, exist_ok=True)
    tag = "" if res_1a["preset_name"] == "low_T_default" else f"_{res_1a['preset_name']}"
    panel_names = [
        "panel_a_hybridization_lin",
        "panel_b_hybridization_log",
        "panel_c_self_energy",
        "panel_d_spectral_qrt",
        "panel_e_dn_dT",
        "panel_f_entropy_curve",
        "panel_g_entropy_vs_T",
        "panel_h_bath_cutoffs",
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
    print(f"Saved 8-panel Phase 1 figure to '{filename}' and individual PDFs to '{plots_dir}/'.", flush=True)


if __name__ == "__main__":
    sync_presets_json()

    parser = argparse.ArgumentParser(description="Run Phase 1 Single-Bath Benchmarks.")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset name from PRESET_CONFIGS / phase1_presets.json, or 'all'.",
    )
    parser.add_argument("--name", type=str, default=None, help="Create & run an on-the-fly preset with this name.")
    parser.add_argument("--D", type=float, default=None, help="Override reference energy scale D in [T_min, D*].")
    parser.add_argument("--Gamma0", type=float, default=None, help="Override power-law prefactor Gamma0.")
    parser.add_argument("--D_bath", type=float, default=None, help="Override bath bandwidth cutoff D_bath.")
    parser.add_argument("--T", type=float, default=None, help="Override fixed bath temperature T_fixed.")
    args = parser.parse_args()

    if any(v is not None for v in (args.name, args.D, args.Gamma0, args.D_bath, args.T)):
        base_key = args.preset if args.preset in PRESET_CONFIGS else (
            ACTIVE_PRESET[0] if isinstance(ACTIVE_PRESET, (list, tuple)) else ACTIVE_PRESET
        )
        base = PRESET_CONFIGS[base_key]
        cname = args.name or "custom_cli"
        PRESET_CONFIGS[cname] = make_preset(
            preset_name=cname,
            D=args.D if args.D is not None else base["D"],
            Gamma0=args.Gamma0 if args.Gamma0 is not None else base["Gamma0"],
            D_bath=args.D_bath if args.D_bath is not None else base["D_bath"],
            T_fixed=args.T if args.T is not None else base["T_fixed"],
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

        res_1c = run_step_1c_verification(cfg)
        res_1a = run_step_1a_verification(cfg)

        fig_path = os.path.join(RESULTS_DIR, f"phase1_verification{suffix}.png")
        plot_phase1_results(res_1c, res_1a, filename=fig_path)

        table_path = os.path.join(RESULTS_DIR, f"phase1_summary_table{suffix}.txt")
        with open(table_path, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(res_1c["log_lines"] + res_1a["log_lines"]) + "\n")
        print(f"Saved Phase 1 ({preset_key}) summary table to '{table_path}'.\n", flush=True)