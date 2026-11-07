"""
run_phase1_benchmarks.py
------------------------
Executes Phase 1 of the verification plan with N_modes = 10,000 tight-binding modes
across all 7 exponents r in {-0.9, -0.5, -0.3, 0.0, +0.3, +0.5, +0.9}[cite: 39, 40].

All figures and numerical summary tables are saved automatically to:
    /home/amsalda/PseudoGap/results/
"""

import os
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
    continuum_self_energy_infinite_D,
    continuum_self_energy_finite_D,
    continuum_spectral_function,
    analytic_S_eps_d,
    analytic_S0_note_eq7,
)

# Dedicated output directory for all current and future results
RESULTS_DIR = "/home/amsalda/PseudoGap/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Full 7-exponent list used across all 8 panels (a)-(h)
R_LIST_ALL = [-0.9, -0.5, -0.3, 0.0, 0.3, 0.5, 0.9]
TARGET_N_MODES = 10000
LN2 = np.log(2.0)


def get_color_map(r_values) -> dict:
    """Assign distinct, high-contrast colors sorted from negative to positive r."""
    unique_r = sorted(set(float(r) for r in r_values))
    cmap = plt.cm.turbo(np.linspace(0.08, 0.92, len(unique_r)))
    return {r: cmap[i] for i, r in enumerate(unique_r)}


def run_step_1c_verification(r_list=R_LIST_ALL) -> dict:
    """Run Step 1c spectral, cutoff, and QRT checks over extended range -1.3 <= w/D <= 1.3."""
    header = (
        "=" * 96 + "\n"
        "STEP 1c: Hybridization Cutoffs (-1.3 <= w/D <= 1.3), Self-Energy & QRT Spectral Function\n"
        + "=" * 96
    )
    print(header)
    log_lines = [header]

    D = 1.0
    Gamma0 = 0.20
    T = 0.002
    eps_d_test = 0.05

    omega_ext = np.linspace(-1.3 * D, 1.3 * D, 1000)
    omega_log = np.logspace(-5.0, np.log10(2.0 * D), 500)

    results_1c = {"T": T, "D": D, "Gamma0": Gamma0, "eps_d_test": eps_d_test, "by_r": {}}
    col_hdr = f"{'r':>6} | {'N_modes':>7} | {'gamma_min':>11} | {'RelErr Gamma(w)':>15} | {'MaxDiff QRT-FFT vs Freq':>23} | {'SumRule Int A(w)dw':>18}"
    sep = "-" * 96
    print(col_hdr)
    print(sep)
    log_lines.extend([col_hdr, sep])

    for r in r_list:
        bath = Bath(
            name="PG",
            Gamma0=Gamma0,
            r=r,
            T=T,
            D=D,
            discretization="log_linear",
            target_n_modes=TARGET_N_MODES,
            D_star=10.0 * T,
            gamma_rule="adaptive",
            c_gamma=1.5,
        )
        solver = LyapunovSolver([bath])

        # Extended linear grid [-1.3 D, +1.3 D]
        re_dl, gam_dl = bare_bath_hybridization_dl(bath, omega_ext)
        gam_cont_untrunc = bath.continuum_gamma(omega_ext)
        sig_cont_infD = continuum_self_energy_infinite_D(omega_ext, Gamma0, r)
        sig_cont_finD = continuum_self_energy_finite_D(
            np.clip(omega_ext, -0.998 * D, 0.998 * D), Gamma0, r, D
        )
        sig_cont_finD[np.abs(omega_ext) >= D] = np.nan

        # 5-decade log grid [10^{-5} D, 2.0 D]
        _, gam_dl_log = bare_bath_hybridization_dl(bath, omega_log)
        gam_cont_log = bath.continuum_gamma(omega_log)
        df_dT_q = bath.dfermi_dT(bath.epsilons)
        denom_log = (omega_log[:, None] - bath.epsilons[None, :]) ** 2 + bath.gammas[None, :] ** 2
        thermal_gam_log = (
            4.0 * T * ((bath.v[None, :] ** 2 * bath.gammas[None, :] / denom_log) @ df_dT_q)
        )

        interior = (np.abs(omega_ext) >= 0.01) & (np.abs(omega_ext) <= 0.8)
        rel_err_gam = np.median(
            np.abs(gam_dl[interior] - gam_cont_untrunc[interior]) / gam_cont_untrunc[interior]
        )

        # Dressed dot spectral function A(\epsilon_d; \omega) on [-0.4 D, 0.4 D]
        omega_dot = np.linspace(-0.4 * D, 0.4 * D, 500)
        _, A_freq = dot_spectral_function_dl_freq(solver, eps_d_test, omega_dot)
        A_cont = continuum_spectral_function(eps_d_test, omega_dot, Gamma0, r, D=D)

        # Real-time QRT evolution + FFT
        bath_qrt = Bath(
            name="PG",
            Gamma0=Gamma0,
            r=r,
            T=T,
            D=D,
            discretization="log_linear",
            target_n_modes=1200,
            D_star=10.0 * T,
            gamma_rule="adaptive",
            c_gamma=1.5,
        )
        solver_qrt = LyapunovSolver([bath_qrt])
        _, A_freq_qrt_grid = dot_spectral_function_dl_freq(solver_qrt, eps_d_test, omega_dot)
        t_grid = np.linspace(0.0, 5000.0, 20001)
        G_ret_t = dot_retarded_gf_qrt_time(solver_qrt, eps_d_test, t_grid)
        _, A_qrt = fourier_transform_qrt_to_spectral(t_grid, G_ret_t, omega_dot)
        max_diff_qrt = np.max(np.abs(A_qrt - A_freq_qrt_grid))

        # Spectral sum rule check across a wide window covering the broadened resonance
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        w_span = max(4.0 * D, 25.0 * Gamma0 * (0.05 ** r) * sec_pr2)
        w_wide = np.linspace(-w_span, w_span, 8001)
        _, A_wide = dot_spectral_function_dl_freq(solver, eps_d_test, w_wide)
        sum_rule = np.trapz(A_wide, w_wide)

        row_str = f"{r:6.2f} | {bath.n_modes:7d} | {bath.gamma_min:11.3e} | {rel_err_gam:15.4e} | {max_diff_qrt:23.4e} | {sum_rule:18.6f}"
        print(row_str)
        log_lines.append(row_str)

        results_1c["by_r"][r] = {
            "omega_ext": omega_ext,
            "gam_dl": gam_dl,
            "gam_cont_untrunc": gam_cont_untrunc,
            "re_dl": re_dl,
            "sig_cont_infD": sig_cont_infD,
            "sig_cont_finD": sig_cont_finD,
            "omega_log": omega_log,
            "gam_dl_log": gam_dl_log,
            "gam_cont_log": gam_cont_log,
            "thermal_gam_log": thermal_gam_log,
            "gamma_min": bath.gamma_min,
            "omega_dot": omega_dot,
            "A_freq": A_freq,
            "A_qrt": A_qrt,
            "A_cont": A_cont,
        }
    print()
    results_1c["log_lines"] = log_lines
    return results_1c


def run_step_1a_verification(r_list=R_LIST_ALL) -> dict:
    """Run Step 1a single-bath Lyapunov Maxwell entropy protocol for all 7 exponents r[cite: 39, 40]."""
    header = (
        "=" * 118 + "\n"
        "STEP 1a: Power-Law Bath Alone — N_modes = 10,000 Tight-Binding Lyapunov dN/dT & Normalized Entropy S / ln(2)\n"
        + "=" * 118
    )
    print(header)
    log_lines = ["", header]

    D = 1.0
    Gamma0 = 0.25
    T_fixed = 0.001

    # Cross-check dense M x M Bartels-Stewart vs. Finite-Diff vs. Schur-Complement LDDL
    b_check = Bath(name="PG", Gamma0=Gamma0, r=0.3, T=T_fixed, D=D, target_n_modes=600)
    s_check = LyapunovSolver([b_check])
    val_lyap = s_check.dn_d_dT(-0.005, method="exact_lyapunov")
    val_fd = s_check.dn_d_dT(-0.005, method="finite_diff", rel_dT=1e-4)
    val_schur = s_check.dn_d_dT(-0.005, method="schur_lddl")
    chk_str = (
        f"Solver Cross-Check (N=600, r=+0.3, eps_d=-0.005): "
        f"|FD - MatrixLyap| = {abs(val_fd - val_lyap):.3e}, "
        f"|SchurLDDL - MatrixLyap| = {abs(val_schur - val_lyap):.3e}\n"
    )
    print(chk_str)
    log_lines.append(chk_str)

    results_1a = {"T_fixed": T_fixed, "D": D, "Gamma0": Gamma0, "sweeps": {}, "T_scans": {}}

    sub_hdr = f"Fixed Temperature Protocol Check at T / D = {T_fixed} with N_modes = {TARGET_N_MODES} (Entropy in units of ln 2):"
    col_hdr = f"{'r':>6} | {'N_modes':>7} | {'eps_max/D':>9} | {'S_num(0)/ln2 [Lyap]':>19} | {'S_ana(0)/ln2 [fin-D]':>20} | {'S_ana(0)/ln2 [Eq.7]':>19} | {'Target r':>10}"
    sep = "-" * 104
    print(sub_hdr)
    print(col_hdr)
    print(sep)
    log_lines.extend([sub_hdr, col_hdr, sep])

    for r in sorted(r_list, reverse=True):
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        eff_width_T = Gamma0 * (T_fixed ** r) * sec_pr2
        eps_max = max(15.0 * D, 50.0 * eff_width_T)

        u = np.linspace(-1.0, 0.0, 201)
        sinh_scale = np.arcsinh(eps_max / (0.05 * T_fixed))
        eps_d_grid = (0.05 * T_fixed) * np.sinh(sinh_scale * u)

        bath = Bath(
            name="PG",
            Gamma0=Gamma0,
            r=r,
            T=T_fixed,
            D=D,
            discretization="log_linear",
            target_n_modes=TARGET_N_MODES,
            D_star=12.0 * T_fixed,
            gamma_rule="capped_adaptive",
            c_gamma=1.2,
            gamma_max=0.2 * T_fixed,
        )
        solver = LyapunovSolver([bath])

        sweep = solver.sweep_eps_d_protocol(eps_d_grid, method="schur_lddl")
        S_num_0 = float(sweep["S_eps_d"][-1])

        S_ana_fin = analytic_S_eps_d(0.0, T_fixed, Gamma0, r, D=D, eps_max=eps_max)
        S_ana_eq7 = analytic_S0_note_eq7(T_fixed, Gamma0, r)

        S_ana_curve = np.array([
            analytic_S_eps_d(float(ed), T_fixed, Gamma0, r, D=D, eps_max=eps_max)
            for ed in eps_d_grid
        ])
        sweep["S_ana_curve"] = S_ana_curve
        sweep["S_num_0"] = S_num_0
        sweep["S_ana_fin"] = S_ana_fin
        sweep["S_ana_eq7"] = S_ana_eq7
        results_1a["sweeps"][r] = sweep

        row_str = (
            f"{r:6.2f} | {bath.n_modes:7d} | {eps_max:9.1f} | "
            f"{S_num_0 / LN2:19.5f} | {S_ana_fin / LN2:20.5f} | "
            f"{S_ana_eq7 / LN2:19.5f} | {r:10.5f}"
        )
        print(row_str)
        log_lines.append(row_str)

    # Temperature scan S(0)/ln 2 vs T / D
    T_grid = np.logspace(-3.2, -0.6, 8)
    T_dense = np.logspace(-3.5, 0.5, 60)
    for r in r_list:
        S0_num_T = []
        S0_fin_T = []
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        for T_val in T_grid:
            eff_w = Gamma0 * (float(T_val) ** r) * sec_pr2
            eps_max_T = max(20.0 * D, 50.0 * eff_w)
            ed_pair = np.array([-eps_max_T, 0.0])

            b_T = Bath(
                name="PG",
                Gamma0=Gamma0,
                r=r,
                T=float(T_val),
                D=D,
                discretization="log_linear",
                target_n_modes=TARGET_N_MODES,
                D_star=min(0.45 * D, 12.0 * float(T_val)),
                gamma_rule="capped_adaptive",
                c_gamma=1.2,
                gamma_max=0.2 * float(T_val),
            )
            sol_T = LyapunovSolver([b_T])
            sw_T = sol_T.sweep_eps_d_protocol(ed_pair, method="schur_lddl")
            S0_num_T.append(float(sw_T["S_eps_d"][-1]))
            S0_fin_T.append(analytic_S_eps_d(0.0, float(T_val), Gamma0, r, D=D, eps_max=eps_max_T))

        S0_eq7_dense = np.array([analytic_S0_note_eq7(float(t), Gamma0, r) for t in T_dense])
        results_1a["T_scans"][r] = {
            "T_grid": T_grid,
            "S0_num": np.array(S0_num_T),
            "S0_fin": np.array(S0_fin_T),
            "T_dense": T_dense,
            "S0_eq7_dense": S0_eq7_dense,
        }

    print()
    results_1a["log_lines"] = log_lines
    return results_1a


def plot_phase1_results(
    res_1c: dict,
    res_1a: dict,
    filename: str = os.path.join(RESULTS_DIR, "phase1_verification.png"),
) -> None:
    """
    Generate the 8-panel (2x4) diagnostic figure with explicit subplot margins
    and bbox_inches='tight' so neither the top legend nor the bottom row (e)-(h)
    can ever be clipped.
    """
    all_r = sorted(set(list(res_1c["by_r"].keys()) + list(res_1a["sweeps"].keys())))
    colors = get_color_map(all_r)

    fig, axes = plt.subplots(2, 4, figsize=(24, 12.5))

    # =========================================================================
    # ROW 1: STEP 1c — HYBRIDIZATION CUTOFFS, SELF-ENERGY & QRT SPECTRAL CHECKS
    # =========================================================================

    # Panel (a): Hybridization \Gamma(\omega) over [-1.3, 1.3], fixed y-axis: 0 -> 0.7
    ax = axes[0, 0]
    for r in all_r:
        data = res_1c["by_r"][r]
        w = data["omega_ext"]
        ax.plot(w, data["gam_cont_untrunc"], "--", color=colors[r], alpha=0.65, lw=1.2)
        ax.plot(w, data["gam_dl"], "-", color=colors[r], lw=1.8)
    ax.axvline(-1.0, color="k", ls=":", alpha=0.7)
    ax.axvline(+1.0, color="k", ls=":", alpha=0.7)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(0.0, 0.7)
    ax.set_xlabel(r"Frequency $\omega / D$")
    ax.set_ylabel(r"Hybridization $\Gamma(\omega)$")
    ax.set_title(r"(a) Step 1c: Hybridization $\Gamma(\omega)$ (UV Cutoff at $|\omega|=D$)")
    style_handles_a = [
        Line2D([0], [0], color="k", ls="-", lw=1.8, label=r"Tight-Binding Lorentzian Sum $\Gamma_{\mathrm{TB}}(\omega)$"),
        Line2D([0], [0], color="k", ls="--", lw=1.2, alpha=0.7, label=r"Analytic Power-Law $\Gamma_0 |\omega|^r$"),
        Line2D([0], [0], color="k", ls=":", lw=1.2, label=r"Tight-Binding Band Edge $|\omega|/D = 1$"),
    ]
    ax.legend(handles=style_handles_a, frameon=True, facecolor="white", framealpha=0.85, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

    # Panel (b): Log-Log plot of \Gamma(\omega), fixed y-axis: 10^-3 -> 10^1
    ax = axes[0, 1]
    for r in all_r:
        data = res_1c["by_r"][r]
        w_log = data["omega_log"]
        ax.loglog(w_log, data["gam_cont_log"], "--", color=colors[r], alpha=0.55, lw=1.2)
        ax.loglog(w_log, data["gam_dl_log"], "-", color=colors[r], lw=1.8)
    if 0.0 in res_1c["by_r"]:
        d0 = res_1c["by_r"][0.0]
        ax.loglog(
            d0["omega_log"],
            np.maximum(d0["thermal_gam_log"], 1e-8),
            "k-.",
            lw=1.6,
        )
        ax.axvline(d0["gamma_min"], color="dimgray", ls=":", lw=1.3, alpha=0.9)
    ax.axvline(1.0, color="k", ls=":", lw=1.3, alpha=0.8)
    ax.set_xlim(1e-5, 2.0)
    ax.set_ylim(1e-3, 1e1)
    ax.set_xlabel(r"Frequency $|\omega| / D$ (log scale)")
    ax.set_ylabel(r"Hybridization $\Gamma(\omega)$ (log scale)")
    ax.set_title(r"(b) Step 1c: IR & UV Cutoffs of $\Gamma(\omega)$ (Log-Log)")
    style_handles_b = [
        Line2D([0], [0], color="k", ls="-", lw=1.8, label=r"Tight-Binding Lorentzian Sum $\Gamma_{\mathrm{TB}}(\omega)$"),
        Line2D([0], [0], color="k", ls="--", lw=1.2, alpha=0.7, label=r"Analytic Power-Law $\Gamma_0 |\omega|^r$"),
        Line2D([0], [0], color="k", ls="-.", lw=1.6, label=r"Fermi-Derivative Window $4T(\partial f/\partial T)\Gamma_{r=0}$"),
        Line2D([0], [0], color="dimgray", ls=":", lw=1.3, label=r"IR Mode-Broadening Cutoff $\gamma_{\min} \propto T$"),
        Line2D([0], [0], color="k", ls=":", lw=1.3, label=r"UV Band-Edge Cutoff $|\omega|/D = 1$"),
    ]
    ax.legend(handles=style_handles_b, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.5, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)

    # Panel (c): Kramers-Kronig Real Self-Energy \Sigma(\omega), fixed y-axis: -0.5 -> 0.5
    ax = axes[0, 2]
    for r in all_r:
        data = res_1c["by_r"][r]
        w = data["omega_ext"]
        ax.plot(w, data["sig_cont_infD"], ":", color=colors[r], alpha=0.75, lw=1.2)
        ax.plot(w, data["sig_cont_finD"], "--", color=colors[r], alpha=0.85, lw=1.2)
        ax.plot(w, data["re_dl"], "-", color=colors[r], lw=1.7)
    ax.axvline(-1.0, color="k", ls=":", alpha=0.5)
    ax.axvline(+1.0, color="k", ls=":", alpha=0.5)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel(r"Frequency $\omega / D$")
    ax.set_ylabel(r"Real Self-Energy $\Sigma(\omega) = \mathrm{Re}\,\Delta^R(\omega)$")
    ax.set_title(r"(c) Step 1c: Real Self-Energy $\Sigma(\omega)$ (Tight-Binding vs Eq. 3)")
    style_handles_c = [
        Line2D([0], [0], color="k", ls="-", lw=1.7, label=r"Tight-Binding Hilbert Sum $\Sigma_{\mathrm{TB}}(\omega)$"),
        Line2D([0], [0], color="k", ls="--", lw=1.2, label=r"Analytic Finite-Bandwidth KK ($|\omega|\leq D$)"),
        Line2D([0], [0], color="k", ls=":", lw=1.2, label=r"Note Eq. (3) ($D\to\infty$): $-\mathrm{sgn}(\omega)\Gamma_0|\omega|^r\tan(\frac{\pi r}{2})$"),
    ]
    ax.legend(handles=style_handles_c, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.5, loc="upper right")
    ax.grid(True, alpha=0.25)

    # Panel (d): Dressed Dot Spectral Function A(\epsilon_d; \omega), fixed y-axis: -0.1 -> 6
    ax = axes[0, 3]
    for r in all_r:
        data = res_1c["by_r"][r]
        w = data["omega_dot"]
        ax.plot(w, data["A_cont"], "--", color=colors[r], lw=1.0, alpha=0.6)
        ax.plot(w, data["A_freq"], "-", color=colors[r], lw=1.7)
        ax.plot(w[::10], data["A_qrt"][::10], "o", color=colors[r], ms=3.2)
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.1, 6.0)
    ax.set_xlabel(r"Frequency $\omega / D$")
    ax.set_ylabel(r"Dot Spectral Function $A(\epsilon_d; \omega)$")
    ax.set_title(r"(d) Step 1c: Dot $A(\epsilon_d=0.05; \omega)$ (Freq vs QRT)")
    style_handles_d = [
        Line2D([0], [0], color="k", ls="-", lw=1.7, label=r"Freq-Domain Tight-Binding (LDDL Eq. 50a)"),
        Line2D([0], [0], color="k", marker="o", ls="None", ms=4.5, label=r"Real-Time QRT Evolution $\mathcal{G}^R(t)$ + FFT"),
        Line2D([0], [0], color="k", ls="--", lw=1.1, alpha=0.7, label=r"Analytic Continuum Formula (Note Eq. 2)"),
    ]
    ax.legend(handles=style_handles_d, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.25)

    # =========================================================================
    # ROW 2: STEP 1a — ENTROPY PROTOCOL (NORMALIZED BY ln 2)
    # =========================================================================

    # Panel (e): Temperature derivative \partial n_d / \partial T vs Impurity Energy \epsilon_d / D
    ax = axes[1, 0]
    for r in all_r:
        sw = res_1a["sweeps"][r]
        ax.plot(sw["eps_d"], sw["dn_dT"], "-", color=colors[r], lw=1.8)
    ax.set_xlim(-0.08, 0.0)
    ax.set_ylim(-75.0, 3.0)
    ax.set_xlabel(r"Impurity Energy $\epsilon_d / D$")
    ax.set_ylabel(r"Occupation Temp. Derivative $\partial n_d / \partial T$")
    ax.set_title(f"(e) Step 1a: $\\partial n_d / \\partial T$ vs $\\epsilon_d / D$ ($T/D = {res_1a['T_fixed']}$)")
    style_handles_e = [
        Line2D([0], [0], color="k", ls="-", lw=1.8, label=r"Tight-Binding Lyapunov $\partial n_d / \partial T$"),
    ]
    ax.legend(handles=style_handles_e, frameon=True, facecolor="white", framealpha=0.85, fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.25)

    # Panel (f): Entropy S(\epsilon_d) / ln 2 vs Impurity Energy \epsilon_d / D (with explicit T/D in title)
    ax = axes[1, 1]
    for r in all_r:
        sw = res_1a["sweeps"][r]
        ax.plot(sw["eps_d"], sw["S_ana_curve"] / LN2, "--", color="k", lw=1.0, alpha=0.55)
        ax.plot(sw["eps_d"], sw["S_eps_d"] / LN2, "-", color=colors[r], lw=1.9)
        ax.axhline(r, color=colors[r], ls=":", alpha=0.65)
    ax.set_xlim(-1.0, 0.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Impurity Energy $\epsilon_d / D$")
    ax.set_ylabel(r"Entropy $S(\epsilon_d) / \ln 2$")
    ax.set_title(f"(f) Step 1a: Entropy $S(\\epsilon_d) / \\ln 2$ vs $\\epsilon_d / D$ ($T/D = {res_1a['T_fixed']}$)")
    style_handles_f = [
        Line2D([0], [0], color="k", ls="-", lw=1.9, label=r"Tight-Binding Integral $-\int (\partial n_d/\partial T)d\epsilon_d'/\ln 2$"),
        Line2D([0], [0], color="k", ls="--", lw=1.1, alpha=0.7, label=r"Analytic Continuum Integral ($|\omega|\leq D$)"),
        Line2D([0], [0], color="k", ls=":", lw=1.2, label=r"Low-$T$ Asymptote $S(0)/\ln 2 = r$"),
    ]
    ax.legend(handles=style_handles_f, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.25)

    # Panel (g): Entropy at \epsilon_d = 0, S(0) / ln 2 vs Temperature T / D
    ax = axes[1, 2]
    for r in all_r:
        tdata = res_1a["T_scans"][r]
        ax.semilogx(tdata["T_dense"], tdata["S0_eq7_dense"] / LN2, "--", color=colors[r], alpha=0.75, lw=1.4)
        ax.semilogx(tdata["T_grid"], tdata["S0_fin"] / LN2, "+", color=colors[r], ms=7.5, mew=1.4)
        ax.semilogx(tdata["T_grid"], tdata["S0_num"] / LN2, "o", color=colors[r], ms=4.8)
        ax.axhline(r, color=colors[r], ls=":", alpha=0.45)
    ax.axhline(1.0, color="gray", ls="-.", lw=1.0, alpha=0.6)
    ax.set_ylim(-1.05, 1.15)
    ax.set_xlabel(r"Temperature $T / D$")
    ax.set_ylabel(r"Entropy at $\epsilon_d = 0$, $S(0) / \ln 2$")
    ax.set_title(r"(g) Step 1a: Entropy at $\epsilon_d = 0$, $S(0) / \ln 2$ vs $T / D$")
    style_handles_g = [
        Line2D([0], [0], color="k", marker="o", ls="None", ms=5, label=r"Tight-Binding Lyapunov $S(0)/\ln 2$"),
        Line2D([0], [0], color="k", marker="+", ls="None", ms=7.5, mew=1.4, label=r"Analytic Finite-Bandwidth ($|\omega|\leq D$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.4, label=r"Note Eq. (7) ($D\to\infty$ Continuum)"),
        Line2D([0], [0], color="k", ls=":", lw=1.2, label=r"Low-$T$ Asymptote $S(0)/\ln 2 = r$"),
    ]
    ax.legend(handles=style_handles_g, frameon=True, facecolor="white", framealpha=0.85, fontsize=7.5, loc="lower right")
    ax.grid(True, alpha=0.25)

    # Panel (h): Entropy at \epsilon_d = 0, S(0) / ln 2 vs Power-Law Exponent r
    ax = axes[1, 3]
    r_vals = np.array(all_r)
    s_num_vals = np.array([res_1a["sweeps"][r]["S_num_0"] / LN2 for r in r_vals])
    s_eq7_vals = np.array([res_1a["sweeps"][r]["S_ana_eq7"] / LN2 for r in r_vals])
    r_cont = np.linspace(-0.98, 0.98, 100)
    ax.plot(r_cont, r_cont, "k--", lw=1.8)
    ax.plot(r_vals, s_eq7_vals, "s", color="#444444", fillstyle="none", ms=8.5, mew=1.5)
    for r, s_val in zip(r_vals, s_num_vals):
        ax.plot(r, s_val, "o", color=colors[r], ms=8, markeredgecolor="k", markeredgewidth=0.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Power-Law Exponent $r$")
    ax.set_ylabel(r"Entropy at $\epsilon_d = 0$, $S(0) / \ln 2$")
    ax.set_title(f"(h) Step 1a: $S(0) / \\ln 2$ vs Exponent $r$ ($T/D = {res_1a['T_fixed']}$)")
    style_handles_h = [
        Line2D([0], [0], color="k", ls="--", lw=1.8, label=r"Exact $T\to 0$ Law: $S(0)/\ln 2 = r$"),
        Line2D([0], [0], color="#444444", marker="s", ls="None", fillstyle="none", ms=8, mew=1.5, label=f"Note Eq. (7) ($D\\to\\infty$, $T/D={res_1a['T_fixed']}$)"),
        Line2D([0], [0], color="k", marker="o", ls="None", ms=7.5, label=f"Tight-Binding Lyapunov ($N={TARGET_N_MODES}$)"),
    ]
    ax.legend(handles=style_handles_h, frameon=True, facecolor="white", framealpha=0.85, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)

    # =========================================================================
    # EXPLICIT SUBPLOT SPACING + GLOBAL TOP LEGEND + BBOX_INCHES='TIGHT'
    # =========================================================================
    fig.subplots_adjust(
        left=0.048,
        right=0.988,
        bottom=0.085,
        top=0.865,
        hspace=0.34,
        wspace=0.25,
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
        title="Power-Law Exponent $r$ Across All Panels  ($\\Gamma(\\omega) = \\Gamma_0 |\\omega|^r$):",
        title_fontsize=12,
    )

    fig.savefig(filename, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved 8-panel Phase 1 verification figure to '{filename}'.")


if __name__ == "__main__":
    res_1c = run_step_1c_verification()
    res_1a = run_step_1a_verification()
    fig_path = os.path.join(RESULTS_DIR, "phase1_verification.png")
    plot_phase1_results(res_1c, res_1a, filename=fig_path)

    # Save the printed summary tables to /home/amsalda/PseudoGap/results/phase1_summary_table.txt
    table_path = os.path.join(RESULTS_DIR, "phase1_summary_table.txt")
    with open(table_path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(res_1c["log_lines"] + res_1a["log_lines"]) + "\n")
    print(f"Saved numerical summary table to '{table_path}'.")