"""
run_phase2_benchmarks.py
------------------------
Executes Phase 2 (Step 1b: Two Baths Connected & Stress-Testing the Weak-Coupling Window)
with N_PG = 10,000 modes in the power-law bath and N_S = 4,000 modes in the probe bath
across all 7 exponents r in {-0.9, -0.5, -0.3, 0.0, +0.3, +0.5, +0.9}.

Runs and compares BOTH physical probe bath models (with D_S = 0.025 D << D):
  1. FLAT SHORT-BAND PROBE:
       Gamma_S(w) = GammaS0 * Theta(D_S - |w|)   (constant GammaS0 inside [-D_S, D_S])
     -> Saved to: /home/amsalda/PseudoGap/results/phase2_verification_flat_band.png
                  /home/amsalda/PseudoGap/results/phase2_summary_flat_band.txt
  2. LORENTZIAN PROBE:
       Gamma_S(w) = GammaS0 / (1 + (w / D_S)^2)  (smooth frequency-dependent cutoff of width D_S)
     -> Saved to: /home/amsalda/PseudoGap/results/phase2_verification_lorentzian.png
                  /home/amsalda/PseudoGap/results/phase2_summary_lorentzian.txt

All physical energy scales and dimensionless ratios are strictly within [10^{-4} D, D]:
  - Power-law bath bandwidth:       D_PG = 1.0
  - Power-law coupling prefactor:   Gamma0 = 0.25 D
  - Power-law bath temperature:     T_PG = 1.0e-4 D  (and 0.005 D, 0.05 D in Panel g)
  - Probe cutoff scale:             D_S = 0.025 D << D
  - Probe peak coupling at w = 0:   GammaS0 = 1.0e-4 D (so GammaS0 / Gamma0 = 4.0e-4)
  - Fixed probe temperature:        T_S_fixed = 0.001 D (satisfies T_PG << T_S << D_S << D)
  - Probe temperature scan range:   T_S in [5.0e-4 D, 0.15 D]
"""

import os
import sys
import gc
import faulthandler
from typing import Literal, Optional

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
    alpha_func_eq14,
    analytic_tilde_S0_eq14,
    analytic_tilde_S0_eq12,
    analytic_two_bath_tilde_S_eps_d,
)

RESULTS_DIR = "/home/amsalda/PseudoGap/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

R_LIST_ALL = [-0.9, -0.5, -0.3, 0.0, 0.3, 0.5, 0.9]
TARGET_N_PG = 10000
TARGET_N_S = 4000
LN2 = np.log(2.0)

# Physical constants strictly obeying [10^{-4} D, D] with D_S << D
D_DEFAULT = 1.0
GAMMA0_DEFAULT = 0.25
GAMMAS0_DEFAULT = 1.0e-4          # 10^{-4} D (ratio GammaS0 / Gamma0 = 4.0e-4)
T_PG_DEFAULT = 1.0e-4             # 10^{-4} D
T_S_FIXED_DEFAULT = 0.001         # 10^{-3} D (satisfies T_PG << T_S << D_S << D)
D_S_CUTOFF_DEFAULT = 0.025        # D_S = 0.025 D << D (short-band / Lorentzian cutoff)

ProbeProfileType = Literal["flat_band", "lorentzian"]


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
    probe_profile: ProbeProfileType = "flat_band",
    D_S_cutoff: float = D_S_CUTOFF_DEFAULT,
    Gamma0: float = GAMMA0_DEFAULT,
    GammaS0: float = GAMMAS0_DEFAULT,
    T_PG: float = T_PG_DEFAULT,
    D_PG: float = D_DEFAULT,
    omega_ir_PG: float = 0.0,
    n_pg: int = TARGET_N_PG,
    n_s: int = TARGET_N_S,
) -> LyapunovSolver:
    """
    Construct the two-bath LyapunovSolver for either:
      - probe_profile == "flat_band":
          Bath "S" is a flat short band \Gamma_S(\omega) = \Gamma_{S,0} \Theta(D_S - |\omega|)
          with tight-binding modes spanning [-D_S_cutoff, +D_S_cutoff].
      - probe_profile == "lorentzian":
          Bath "S" is a Lorentzian \Gamma_S(\omega) = \Gamma_{S,0} / (1 + (\omega / D_S_cutoff)^2)
          with tight-binding modes spanning [-D_PG, +D_PG].
    """
    if probe_profile == "flat_band":
        D_S_band = D_S_cutoff
        probe_fn = None
    elif probe_profile == "lorentzian":
        D_S_band = D_PG
        probe_fn = make_lorentzian_shape(D_S_cutoff)
    else:
        raise ValueError(f"Unknown probe_profile: {probe_profile}")

    D_star_pg = min(0.25 * D_PG, max(8.0 * T_S, 5.0e-3 * D_PG))
    D_star_s = min(0.40 * D_S_band, max(8.0 * T_S, 2.0e-3 * D_S_band))

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
        gamma_rule="adaptive",
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


def eval_analytic_two_bath(
    eps_d: float | np.ndarray,
    T_S: float,
    Gamma0: float,
    GammaS0: float,
    r: float,
    probe_profile: ProbeProfileType,
    D_S_cutoff: float,
    D_PG: float = D_DEFAULT,
    eps_max: Optional[float] = None,
) -> float | np.ndarray:
    """Evaluate the exact two-bath continuum integral (Note Eq. 11) for the chosen probe profile."""
    if probe_profile == "flat_band":
        return analytic_two_bath_tilde_S_eps_d(
            eps_d, T_S, Gamma0, GammaS0, r,
            D_PG=D_PG, D_S=D_S_cutoff, probe_shape_fn=None, eps_max=eps_max
        )
    if probe_profile == "lorentzian":
        return analytic_two_bath_tilde_S_eps_d(
            eps_d, T_S, Gamma0, GammaS0, r,
            D_PG=D_PG, D_S=D_PG, probe_shape_fn=make_lorentzian_shape(D_S_cutoff), eps_max=eps_max
        )
    raise ValueError(f"Unknown probe_profile: {probe_profile}")


def run_step_1b_core_verification(
    probe_profile: ProbeProfileType = "flat_band",
    r_list=R_LIST_ALL,
) -> dict:
    D = D_DEFAULT
    Gamma0 = GAMMA0_DEFAULT
    GammaS0 = GAMMAS0_DEFAULT
    ratio_S_ref = GammaS0 / Gamma0  # = 4.0e-4 (GammaS0 = 1.0e-4 D)
    T_S_fixed = T_S_FIXED_DEFAULT   # = 1.0e-3 D
    T_PG_base = T_PG_DEFAULT        # = 1.0e-4 D
    D_S_cut = D_S_CUTOFF_DEFAULT    # = 0.025 D << D

    prof_desc = (
        f"Flat Short-Band Probe Gamma_S(w) = GammaS0 * Theta(D_S - |w|) with D_S = {D_S_cut}D << D"
        if probe_profile == "flat_band"
        else f"Lorentzian Probe Gamma_S(w) = GammaS0 / (1 + (w/D_S)^2) with D_S = {D_S_cut}D << D"
    )

    header = (
        "=" * 124 + "\n"
        f"STEP 1b (CORE — {probe_profile.upper()}): {prof_desc}\n"
        f"  Parameters: D={D}, Gamma0={Gamma0}, GammaS0={GammaS0:.1e} (GS0/G0={ratio_S_ref:.1e}), "
        f"D_S={D_S_cut}, T_PG={T_PG_base:.1e}, T_S_fixed={T_S_fixed}\n"
        + "=" * 124
    )
    print(header, flush=True)
    log_lines = [header]

    sol_chk = build_two_bath_solver(
        r=0.3, T_S=T_S_fixed, probe_profile=probe_profile, D_S_cutoff=D_S_cut,
        Gamma0=Gamma0, GammaS0=GammaS0, T_PG=T_PG_base, n_pg=300, n_s=150
    )
    ed_list = np.linspace(-0.3,0.3,51)
    for ed in ed_list:
        val_lyap = sol_chk.dn_d_dT(ed, target_bath="S", method="exact_lyapunov")
        val_fd = sol_chk.dn_d_dT(ed, target_bath="S", method="finite_diff", rel_dT=1e-4)
        val_schur = sol_chk.dn_d_dT(ed, target_bath="S", method="schur_lddl")
        chk_str = (
            f"Two-Bath Solver Cross-Check (M={sol_chk.dim}, r=+0.3, target_bath='S', eps_d={ed}):\n"
            f"  |FD - MatrixLyap| = {abs(val_fd - val_lyap):.3e}, "
            f"|SchurLDDL - MatrixLyap| = {abs(val_schur - val_lyap):.3e}\n"
        )
        print(chk_str, flush=True)
        log_lines.append(chk_str)
        gc.collect()

    results_1b = {
        "probe_profile": probe_profile,
        "D": D,
        "Gamma0": Gamma0,
        "GammaS0": GammaS0,
        "ratio_S_ref": ratio_S_ref,
        "T_S_fixed": T_S_fixed,
        "T_PG_base": T_PG_base,
        "D_S_cut": D_S_cut,
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
        eps_max = max(15.0 * D, 50.0 * eff_width_T)

        u = np.linspace(-1.0, 0.0, 161)
        sinh_scale = np.arcsinh(eps_max / (0.05 * T_S_fixed))
        eps_d_grid = (0.05 * T_S_fixed) * np.sinh(sinh_scale * u)

        solver = build_two_bath_solver(
            r=r, T_S=T_S_fixed, probe_profile=probe_profile, D_S_cutoff=D_S_cut,
            Gamma0=Gamma0, GammaS0=GammaS0, T_PG=T_PG_base
        )
        sweep = solver.sweep_eps_d_protocol(eps_d_grid, target_bath="S", method="schur_lddl")
        tS_num_0 = float(sweep["S_eps_d"][-1])

        tS_exact_curve = eval_analytic_two_bath(
            eps_d_grid, T_S_fixed, Gamma0, GammaS0, r,
            probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D, eps_max=eps_max
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

    # Probe Temperature Scan T_S / D in [5e-4, 0.15]
    T_grid = np.logspace(np.log10(5.0e-4), np.log10(0.15), 8)
    T_dense = np.logspace(np.log10(4.0e-4), np.log10(0.20), 50)
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
            eps_max_T = max(20.0 * D, 50.0 * eff_w)
            ed_pair = np.array([-eps_max_T, 0.0])

            sol_T = build_two_bath_solver(
                r=r, T_S=ts, probe_profile=probe_profile, D_S_cutoff=D_S_cut,
                Gamma0=Gamma0, GammaS0=GammaS0, T_PG=T_PG_base
            )
            sw_T = sol_T.sweep_eps_d_protocol(ed_pair, target_bath="S", method="schur_lddl")
            val_num = float(sw_T["S_eps_d"][-1])
            val_ex = float(eval_analytic_two_bath(
                0.0, ts, Gamma0, GammaS0, r,
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D, eps_max=eps_max_T
            ))
            tS_num_T.append(val_num)
            tS_exact_T.append(val_ex)
            S_recon_num_T.append((val_num / LN2) / pref)
            S_recon_exact_T.append((val_ex / LN2) / pref)

        tS_exact_dense = np.array([
            float(eval_analytic_two_bath(
                0.0, float(t), Gamma0, GammaS0, r,
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D,
                eps_max=max(20.0 * D, 50.0 * (Gamma0 * (float(t) ** r) * sec_pr2 + GammaS0))
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
            "tS_eq12_dense": tS_eq12_dense,
            "tS_eq14_dense": tS_eq14_dense,
            "S_recon_exact_dense": (tS_exact_dense / LN2) / pref_dense,
            "S_recon_eq12_dense": (tS_eq12_dense / LN2) / pref_dense,
            "S_recon_eq14_dense": (tS_eq14_dense / LN2) / pref_dense,
        }
        gc.collect()

    print(flush=True)
    results_1b["log_lines"] = log_lines
    return results_1b


def run_step_1b_stress_tests(
    probe_profile: ProbeProfileType = "flat_band",
    r_list=R_LIST_ALL,
) -> dict:
    D = D_DEFAULT
    Gamma0 = GAMMA0_DEFAULT
    GammaS0 = GAMMAS0_DEFAULT
    ratio_S_ref = GammaS0 / Gamma0
    T_S_fixed = T_S_FIXED_DEFAULT
    D_S_cut = D_S_CUTOFF_DEFAULT

    header = (
        "=" * 112 + "\n"
        f"STEP 1b (STRESS TESTS — {probe_profile.upper()}): Weak-Coupling Sweep, Finite T_PG, and Probe Cutoff D_S\n"
        + "=" * 112
    )
    print(header, flush=True)
    log_lines = ["", header]

    # -------------------------------------------------------------------------
    # Stress Test 1 (Panel f): Probe Coupling Ratio \Gamma_S / \Gamma_0 in [4e-4, 1e-1]
    # -------------------------------------------------------------------------
    ratio_grid = np.logspace(np.log10(4.0e-4), -1.0, 7)
    ratio_dense = np.logspace(-4.0, -1.0, 30)
    gamma_s_sweeps = {}

    print(f"Stress Test 1 — S_recon(0)/ln2 vs Probe Coupling Ratio Gamma_S/Gamma_0 in [4e-4, 1e-1] (at T_S/D = {T_S_fixed}):", flush=True)
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
            eps_max = max(15.0 * D, 50.0 * eff_w)
            ed_pair = np.array([-eps_max, 0.0])

            sol = build_two_bath_solver(
                r=r, T_S=T_S_fixed, probe_profile=probe_profile, D_S_cutoff=D_S_cut, Gamma0=Gamma0, GammaS0=gs
            )
            sw = sol.sweep_eps_d_protocol(ed_pair, target_bath="S", method="schur_lddl")
            val_num = float(sw["S_eps_d"][-1])
            recon_num_list.append((val_num / LN2) / pref)

        for rat in ratio_dense:
            gs = float(rat) * Gamma0
            pref = float(rat) * alpha_r * (T_S_fixed ** (-r))
            eff_w = Gamma0 * (T_S_fixed ** r) * sec_pr2 + gs
            eps_max = max(15.0 * D, 50.0 * eff_w)
            val_ex = float(eval_analytic_two_bath(
                0.0, T_S_fixed, Gamma0, gs, r,
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D, eps_max=eps_max
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
    # Stress Test 2 (Panel g): Finite Power-Law Bath Temperature T_{PG} >= 10^{-4} D
    # -------------------------------------------------------------------------
    r_focus = [-0.5, -0.3, 0.3, 0.5]
    T_PG_low = 1.0e-4 * D
    T_PG_mid = 0.005 * D
    T_PG_high = 0.05 * D
    T_S_scan = np.logspace(np.log10(5.0e-4), np.log10(0.15), 8)
    T_S_dense = np.logspace(np.log10(4.0e-4), np.log10(0.20), 50)

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

        for T_val in T_S_scan:
            ts = float(T_val)
            pref = ratio_S_ref * alpha_r * (ts ** (-r))
            eff_w = Gamma0 * (ts ** r) * sec_pr2 + GammaS0
            eps_max_T = max(20.0 * D, 50.0 * eff_w)
            ed_pair = np.array([-eps_max_T, 0.0])

            for t_pg_val, target_list in [
                (T_PG_low, recon_tpg_0),
                (T_PG_mid, recon_tpg_mid),
                (T_PG_high, recon_tpg_high),
            ]:
                sol = build_two_bath_solver(
                    r=r, T_S=ts, probe_profile=probe_profile, D_S_cutoff=D_S_cut,
                    Gamma0=Gamma0, GammaS0=GammaS0, T_PG=t_pg_val
                )
                sw = sol.sweep_eps_d_protocol(ed_pair, target_bath="S", method="schur_lddl")
                target_list.append((float(sw["S_eps_d"][-1]) / LN2) / pref)

        for T_val in T_S_dense:
            ts = float(T_val)
            pref = ratio_S_ref * alpha_r * (ts ** (-r))
            eff_w = Gamma0 * (ts ** r) * sec_pr2 + GammaS0
            eps_max_T = max(20.0 * D, 50.0 * eff_w)
            val_ex = float(eval_analytic_two_bath(
                0.0, ts, Gamma0, GammaS0, r,
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D, eps_max=eps_max_T
            ))
            recon_ana_curve.append((val_ex / LN2) / pref)

        finite_T_pg_results["by_r"][r] = {
            "T_S_scan": T_S_scan,
            "T_S_dense": T_S_dense,
            "recon_tpg_0": np.array(recon_tpg_0),
            "recon_tpg_mid": np.array(recon_tpg_mid),
            "recon_tpg_high": np.array(recon_tpg_high),
            "recon_ana_curve": np.array(recon_ana_curve),
        }
        gc.collect()

    # -------------------------------------------------------------------------
    # Stress Test 3 (Panel h): Direct Comparison of Flat Short-Band (D_S << D),
    # Lorentzian (D_S << D), and Wide-Band (D_S = 0.25 D > T^*).
    # -------------------------------------------------------------------------
    D_S_narrow = D_S_CUTOFF_DEFAULT   # 0.025 D << D
    D_S_wide = 0.25 * D               # 0.25 D > T^*

    probe_profile_results = {
        "D_S_narrow": D_S_narrow,
        "D_S_wide": D_S_wide,
        "by_r": {},
    }
    for r in r_focus:
        alpha_r = alpha_func_eq14(r)
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        recon_narrow_box = []
        recon_narrow_lor = []
        recon_wide = []

        for T_val in T_S_scan:
            ts = float(T_val)
            pref = ratio_S_ref * alpha_r * (ts ** (-r))
            eff_w = Gamma0 * (ts ** r) * sec_pr2 + GammaS0
            eps_max_T = max(20.0 * D, 50.0 * eff_w)
            ed_pair = np.array([-eps_max_T, 0.0])

            # 1. Flat Short-Band Probe D_S = 0.025 D << D
            sol_nbox = build_two_bath_solver(
                r=r, T_S=ts, probe_profile="flat_band", D_S_cutoff=D_S_narrow, Gamma0=Gamma0, GammaS0=GammaS0
            )
            sw_nbox = sol_nbox.sweep_eps_d_protocol(ed_pair, target_bath="S", method="schur_lddl")
            recon_narrow_box.append((float(sw_nbox["S_eps_d"][-1]) / LN2) / pref)

            # 2. Lorentzian Probe D_S = 0.025 D << D
            sol_nlor = build_two_bath_solver(
                r=r, T_S=ts, probe_profile="lorentzian", D_S_cutoff=D_S_narrow, Gamma0=Gamma0, GammaS0=GammaS0
            )
            sw_nlor = sol_nlor.sweep_eps_d_protocol(ed_pair, target_bath="S", method="schur_lddl")
            recon_narrow_lor.append((float(sw_nlor["S_eps_d"][-1]) / LN2) / pref)

            # 3. Wide Probe D_S = 0.25 D > T^* (using current suite's profile type)
            sol_wide = build_two_bath_solver(
                r=r, T_S=ts, probe_profile=probe_profile, D_S_cutoff=D_S_wide, Gamma0=Gamma0, GammaS0=GammaS0
            )
            sw_wide = sol_wide.sweep_eps_d_protocol(ed_pair, target_bath="S", method="schur_lddl")
            recon_wide.append((float(sw_wide["S_eps_d"][-1]) / LN2) / pref)

        probe_profile_results["by_r"][r] = {
            "T_S_scan": T_S_scan,
            "recon_narrow_box": np.array(recon_narrow_box),
            "recon_narrow_lor": np.array(recon_narrow_lor),
            "recon_wide": np.array(recon_wide),
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
    T_S_fixed = res_core["T_S_fixed"]
    ratio_S_ref = res_core["ratio_S_ref"]
    D_S_cut = res_core["D_S_cut"]

    if probe_profile == "flat_band":
        probe_formula_tex = f"\\Gamma_S(\\omega) = \\Gamma_{{S,0}}\\,\\Theta(D_S - |\\omega|)"
        probe_short_label = f"Flat Short-Band $D_S={D_S_cut}D \\ll D$"
    else:
        probe_formula_tex = f"\\Gamma_S(\\omega) = \\Gamma_{{S,0}} / [1 + (\\omega/D_S)^2]"
        probe_short_label = f"Lorentzian $D_S={D_S_cut}D \\ll D$"

    fig, axes = plt.subplots(2, 4, figsize=(24, 12.6))

    # Panel (a)
    ax = axes[0, 0]
    for r in all_r:
        sw = res_core["sweeps"][r]
        scale_factor = (T_S_fixed ** r) / ratio_S_ref
        ax.plot(sw["eps_d"], sw["dn_dT"] * scale_factor, "-", color=colors[r], lw=1.8)
    ax.set_xlim(-0.06, 0.0)
    ax.set_ylim(-90.0, 4.0)
    ax.set_xlabel(r"Impurity Energy $\epsilon_d / D$")
    ax.set_ylabel(r"Rescaled $\frac{\Gamma_0}{\Gamma_{S,0}} T_S^r \frac{\partial n_d}{\partial T_S}$")
    ax.set_title(f"(a) Step 1b: Probe Derivative $\\partial n_d / \\partial T_S$ ($T_S/D={T_S_fixed}$)")
    style_a = [Line2D([0], [0], color="k", ls="-", lw=1.8, label=f"Two-Bath Tight-Binding ({probe_short_label})")]
    ax.legend(handles=style_a, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.8, loc="lower left")
    ax.grid(True, alpha=0.25)

    # Panel (b)
    ax = axes[0, 1]
    for r in all_r:
        sw = res_core["sweeps"][r]
        norm = sw["prefactor"] * LN2
        ax.plot(sw["eps_d"], sw["tS_exact_curve"] / norm, "--", color="k", lw=1.0, alpha=0.55)
        ax.plot(sw["eps_d"], sw["S_eps_d"] / norm, "-", color=colors[r], lw=1.9)
        ax.axhline(r, color=colors[r], ls=":", alpha=0.65)
    ax.set_xlim(-1.0, 0.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Impurity Energy $\epsilon_d / D$")
    ax.set_ylabel(r"Rescaled $\tilde{S}(\epsilon_d) / [\frac{\Gamma_{S,0}}{\Gamma_0}\alpha(r)T_S^{-r}\ln 2]$")
    ax.set_title(f"(b) Step 1b: Two-Bath Integral $\\tilde{{S}}(\\epsilon_d)$ ($T_S/D={T_S_fixed}$)")
    style_b = [
        Line2D([0], [0], color="k", ls="-", lw=1.9, label=r"Tight-Binding Integral $-\int (\partial n_d/\partial T_S)d\epsilon_d'$"),
        Line2D([0], [0], color="k", ls="--", lw=1.1, alpha=0.7, label=r"Two-Bath Continuum Integral (Note Eq. 11)"),
        Line2D([0], [0], color="k", ls=":", lw=1.2, label=r"Note Eq. (14) Target $\tilde{S}(0) \to r$"),
    ]
    ax.legend(handles=style_b, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.25)

    # Panel (c)
    ax = axes[0, 2]
    for r in all_r:
        if np.isclose(r, 0.0):
            continue
        tdata = res_core["T_scans"][r]
        ax.loglog(tdata["T_dense"], np.abs(tdata["tS_eq14_dense"]) / (ratio_S_ref * LN2), ":", color=colors[r], lw=1.3, alpha=0.75)
        ax.loglog(tdata["T_dense"], np.abs(tdata["tS_exact_dense"]) / (ratio_S_ref * LN2), "-", color=colors[r], lw=1.5, alpha=0.85)
        ax.loglog(tdata["T_grid"], np.abs(tdata["tS_num"]) / (ratio_S_ref * LN2), "o", color=colors[r], ms=5.0)
    ax.axvline(D_S_cut, color="k", ls="-.", lw=1.2, alpha=0.75)
    ax.set_xlim(4e-4, 2e-1)
    ax.set_ylim(1e-3, 2e3)
    ax.set_xlabel(r"Probe Temperature $T_S / D$ (log scale)")
    ax.set_ylabel(r"Scaled Magnitude $|\tilde{S}(0)| / (\frac{\Gamma_{S,0}}{\Gamma_0}\ln 2)$")
    ax.set_title(r"(c) Step 1b: Power-Law Scaling $|\tilde{S}(0)| \propto T_S^{-r}$ (Log-Log)")
    style_c = [
        Line2D([0], [0], color="k", marker="o", ls="None", ms=5, label=r"Tight-Binding Lyapunov $|\tilde{S}(0)|$"),
        Line2D([0], [0], color="k", ls="-", lw=1.5, label=f"Two-Bath Continuum ({probe_short_label})"),
        Line2D([0], [0], color="k", ls=":", lw=1.3, label=r"Note Eq. (14) Pure Power Law $|r|\alpha(r)T_S^{-r}$"),
        Line2D([0], [0], color="k", ls="-.", lw=1.2, label=f"Probe Cutoff $T_S = D_S = {D_S_cut}D$"),
    ]
    ax.legend(handles=style_c, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.2, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)

    # Panel (d)
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
    ax.set_title(f"(d) Step 1b: Reconstructed $S(0)/\\ln 2$ vs $r$ ($T_S/D={T_S_fixed}$)")
    style_d = [
        Line2D([0], [0], color="k", ls="--", lw=1.8, label=r"Note Eq. (14) Law: $S_{\mathrm{recon}}(0)/\ln 2 = r$"),
        Line2D([0], [0], color="#444444", marker="s", ls="None", fillstyle="none", ms=8, mew=1.5, label=r"Two-Bath Continuum Integral (Note Eq. 11)"),
        Line2D([0], [0], color="k", marker="o", ls="None", ms=7.5, label=r"Two-Bath Tight-Binding Lyapunov"),
    ]
    ax.legend(handles=style_d, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.5, loc="upper left")
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

    # Panel (e)
    ax = axes[1, 0]
    for r in all_r:
        tdata = res_core["T_scans"][r]
        ax.semilogx(tdata["T_dense"], tdata["S_recon_exact_dense"], "-", color=colors[r], alpha=0.85, lw=1.6)
        ax.semilogx(tdata["T_grid"], tdata["S_recon_num"], "o", color=colors[r], ms=5.0)
        ax.axhline(r, color=colors[r], ls=":", alpha=0.45)
    ax.axvline(D_S_cut, color="k", ls="-.", lw=1.3, alpha=0.8)
    ax.set_xlim(4e-4, 2e-1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Probe Temperature $T_S / D$")
    ax.set_ylabel(r"Reconstructed Entropy $S_{\mathrm{recon}}(0) / \ln 2$")
    ax.set_title(f"(e) Step 1b: $S_{{\\mathrm{{recon}}}}(0)/\\ln 2$ vs $T_S/D$ ({probe_short_label})")
    style_e = [
        Line2D([0], [0], color="k", marker="o", ls="None", ms=5, label=f"Tight-Binding (${probe_formula_tex}$)"),
        Line2D([0], [0], color="k", ls="-", lw=1.6, label=r"Two-Bath Continuum Integral (Note Eq. 11)"),
        Line2D([0], [0], color="k", ls=":", lw=1.2, label=r"Note Eq. (14) Target $S(0)/\ln 2 = r$"),
        Line2D([0], [0], color="k", ls="-.", lw=1.3, label=f"Probe Cutoff $T_S = D_S = {D_S_cut}D$ ($S\\to 0$)"),
    ]
    ax.legend(handles=style_e, frameon=True, facecolor="white", framealpha=0.85, fontsize=6.8, loc="lower left")
    ax.grid(True, alpha=0.25)

    # Panel (f)
    ax = axes[1, 1]
    for r in all_r:
        gdata = res_stress["gamma_s_sweeps"][r]
        ax.semilogx(gdata["ratio_dense"], gdata["S_recon_exact_dense"], "--", color=colors[r], lw=1.4, alpha=0.8)
        ax.semilogx(gdata["ratio_grid"], gdata["S_recon_num"], "o", color=colors[r], ms=5.0)
        ax.axhline(r, color=colors[r], ls=":", alpha=0.45)
    ax.set_xlim(1e-4, 1e-1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Probe Coupling Ratio $\Gamma_{S,0} / \Gamma_0$ (log scale)")
    ax.set_ylabel(r"Reconstructed Entropy $S_{\mathrm{recon}}(0) / \ln 2$")
    ax.set_title(f"(f) Cutoff Test 1: $\\Gamma_S \\ll \\Gamma(\\omega)$ Breakdown ($T_S/D={T_S_fixed}$)")
    style_f = [
        Line2D([0], [0], color="k", marker="o", ls="None", ms=5, label=r"Tight-Binding Lyapunov $S_{\mathrm{recon}}(0)/\ln 2$"),
        Line2D([0], [0], color="k", ls="--", lw=1.4, label=r"Two-Bath Continuum Integral (Note Eq. 11)"),
        Line2D([0], [0], color="k", ls=":", lw=1.2, label=r"Weak-Probe Limit $\Gamma_S \to 0$ (Eq. 14)"),
    ]
    ax.legend(handles=style_f, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.5, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)

    # Panel (g)
    ax = axes[1, 2]
    T_PG_low = res_stress["finite_T_pg"]["T_PG_low"]
    T_PG_mid = res_stress["finite_T_pg"]["T_PG_mid"]
    T_PG_high = res_stress["finite_T_pg"]["T_PG_high"]
    for r, tpg_data in res_stress["finite_T_pg"]["by_r"].items():
        ax.semilogx(tpg_data["T_S_dense"], tpg_data["recon_ana_curve"], "-", color=colors[r], lw=1.7, alpha=0.85)
        ax.semilogx(tpg_data["T_S_scan"], tpg_data["recon_tpg_0"], "o", color=colors[r], ms=5.5)
        ax.semilogx(tpg_data["T_S_scan"], tpg_data["recon_tpg_mid"], "s", color=colors[r], ms=6.5, fillstyle="none", mew=1.3)
        ax.semilogx(tpg_data["T_S_scan"], tpg_data["recon_tpg_high"], "x", color=colors[r], ms=6.5, mew=1.5)
        ax.axhline(r, color=colors[r], ls=":", alpha=0.45)
    ax.axvline(D_S_cut, color="k", ls="-.", lw=1.3, alpha=0.8)
    ax.set_xlim(4e-4, 2e-1)
    ax.set_ylim(-0.65, 0.65)
    ax.set_xlabel(r"Probe Temperature $T_S / D$")
    ax.set_ylabel(r"Reconstructed Entropy $S_{\mathrm{recon}}(0) / \ln 2$")
    ax.set_title(f"(g) Cutoff Test 2: Finite $T_{{\\mathrm{{PG}}}} > 0$ ({probe_short_label})")
    style_g = [
        Line2D([0], [0], color="k", ls="-", lw=1.7, label=r"Two-Bath Continuum (Eq. 11, $\partial_{T_S} f_{\mathrm{PG}} = 0$)"),
        Line2D([0], [0], color="k", marker="o", ls="None", ms=5.5, label=f"Tight-Binding $T_{{\\mathrm{{PG}}}}/D = {T_PG_low:.0e}$"),
        Line2D([0], [0], color="k", marker="s", ls="None", ms=6.5, fillstyle="none", mew=1.3, label=f"Tight-Binding $T_{{\\mathrm{{PG}}}}/D = {T_PG_mid}$"),
        Line2D([0], [0], color="k", marker="x", ls="None", ms=6.5, mew=1.5, label=f"Tight-Binding $T_{{\\mathrm{{PG}}}}/D = {T_PG_high}$"),
        Line2D([0], [0], color="k", ls="-.", lw=1.3, label=f"Probe Cutoff $T_S = D_S = {D_S_cut}D$"),
    ]
    ax.legend(handles=style_g, frameon=True, facecolor="white", framealpha=0.85, fontsize=6.8, loc="lower left")
    ax.grid(True, alpha=0.25)

    # Panel (h): Direct comparison of Flat Short-Band (D_S << D), Lorentzian (D_S << D), and Wide-Band (D_S = 0.25 D)
    ax = axes[1, 3]
    D_S_nar = res_stress["probe_profiles"]["D_S_narrow"]
    D_S_wid = res_stress["probe_profiles"]["D_S_wide"]
    for r, pdata in res_stress["probe_profiles"]["by_r"].items():
        ax.semilogx(pdata["T_S_scan"], pdata["recon_narrow_box"], "^-", color=colors[r], ms=5.2, lw=1.8)
        ax.semilogx(pdata["T_S_scan"], pdata["recon_narrow_lor"], "o:", color=colors[r], ms=4.8, lw=1.4, fillstyle="none")
        ax.semilogx(pdata["T_S_scan"], pdata["recon_wide"], "s--", color=colors[r], ms=4.8, lw=1.2, fillstyle="none", alpha=0.75)
        ax.axhline(r, color=colors[r], ls=":", alpha=0.4)
    ax.axvline(D_S_nar, color="k", ls="-.", lw=1.3)
    ax.axvline(D_S_wid, color="dimgray", ls="--", lw=1.2)
    ax.set_xlim(4e-4, 2e-1)
    ax.set_ylim(-0.75, 0.75)
    ax.set_xlabel(r"Probe Temperature $T_S / D$")
    ax.set_ylabel(r"Reconstructed Entropy $S_{\mathrm{recon}}(0) / \ln 2$")
    ax.set_title(r"(h) Cutoff Test 3: Flat Short-Band vs Lorentzian vs Wide $D_S$")
    style_h = [
        Line2D([0], [0], color="k", marker="^", ls="-", ms=5.2, lw=1.8, label=f"Flat Short-Band $D_S={D_S_nar}D \\ll D$ (Sharp Roll-Off)"),
        Line2D([0], [0], color="k", marker="o", ls=":", ms=4.8, fillstyle="none", label=f"Lorentzian $D_S={D_S_nar}D \\ll D$ (Smooth Roll-Off)"),
        Line2D([0], [0], color="k", marker="s", ls="--", ms=4.8, fillstyle="none", label=f"Wide Probe $D_S={D_S_wid}D > T^*$ (Climbs to $+1$)"),
        Line2D([0], [0], color="k", ls="-.", lw=1.3, label=f"Short-Band Cutoff $T_S = D_S = {D_S_nar}D$"),
    ]
    ax.legend(handles=style_h, frameon=True, facecolor="white", framealpha=0.85, fontsize=6.8, loc="lower left")
    ax.grid(True, alpha=0.25)

    fig.subplots_adjust(
        left=0.048,
        right=0.988,
        bottom=0.085,
        top=0.865,
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
        bbox_to_anchor=(0.518, 0.965),
        ncol=len(all_r),
        frameon=True,
        facecolor="#f8f9fa",
        edgecolor="#cccccc",
        fontsize=11.5,
        title=(
            f"Power-Law Exponent $r$ Across All Panels  "
            f"($\\Gamma_{{\\mathrm{{PG}}}}(\\omega) = \\Gamma_0 |\\omega|^r$,  "
            f"Probe: ${probe_formula_tex}$ with $D_S = {D_S_cut}D \\ll D$,  "
            f"All Scales $\\in [10^{{-4}}D, D]$):"
        ),
        title_fontsize=11.5,
    )

    fig.savefig(filename, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved 8-panel Phase 2 ({probe_profile}) figure to '{filename}'.", flush=True)


if __name__ == "__main__":
    for profile_mode in ("flat_band", "lorentzian"):
        res_core = run_step_1b_core_verification(probe_profile=profile_mode)
        res_stress = run_step_1b_stress_tests(probe_profile=profile_mode)

        fig_path = os.path.join(RESULTS_DIR, f"phase2_verification_{profile_mode}.png")
        plot_phase2_results(res_core, res_stress, filename=fig_path)

        table_path = os.path.join(RESULTS_DIR, f"phase2_summary_{profile_mode}.txt")
        with open(table_path, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(res_core["log_lines"] + res_stress["log_lines"]) + "\n")
        print(f"Saved Phase 2 ({profile_mode}) summary table to '{table_path}'.\n", flush=True)