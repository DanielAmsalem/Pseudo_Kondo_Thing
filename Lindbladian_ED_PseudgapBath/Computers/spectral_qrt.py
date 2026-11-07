"""
spectral_qrt.py
---------------
Step 1c, Step 1a, and Step 1b Analytical & Spectral Reference Module.
Uses pure vectorized NumPy Gauss-Legendre quadrature (zero Fortran _quadpack calls).
"""

from typing import Any, Callable, Optional, Tuple
import numpy as np
from scipy import linalg
from bath import Bath

# Precomputed Gauss-Legendre nodes and weights on [0, 1] for fast, non-singular integration
_GL64_S, _GL64_W = np.polynomial.legendre.leggauss(64)
_GL64_S = 0.5 * (_GL64_S + 1.0)
_GL64_W = 0.5 * _GL64_W

_GL128_S, _GL128_W = np.polynomial.legendre.leggauss(128)
_GL128_S = 0.5 * (_GL128_S + 1.0)
_GL128_W = 0.5 * _GL128_W


# ============================================================================
# 1. Discrete LDDL / QRT Spectral Functions (Step 1c)
# ============================================================================

def bare_bath_hybridization_dl(bath: Bath, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the bare discretized lead retarded hybridization function
    \Delta^R_{\alpha;DL}(\omega) = \sum_q |v_q|^2 / (\omega - \epsilon_q + i\gamma_q)
    (LDDL Eqs. 31a, 42b, 43).
    """
    w = np.asarray(omega, dtype=float)
    re_delta = np.zeros_like(w)
    gamma_dl = np.zeros_like(w)

    chunk_size = 1000
    for start in range(0, bath.n_modes, chunk_size):
        end = min(bath.n_modes, start + chunk_size)
        diff = w[:, None] - bath.epsilons[None, start:end]
        gam = bath.gammas[None, start:end]
        v2 = (bath.v[None, start:end]) ** 2
        denom = diff * diff + gam * gam
        re_delta += np.sum(v2 * diff / denom, axis=1)
        gamma_dl += np.sum(v2 * gam / denom, axis=1)

    return re_delta, gamma_dl


def dot_spectral_function_dl_freq(
    solver: Any,
    eps_d: float,
    omega: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the dressed dot retarded Green's function G^R_{dd;DL}(\omega) and
    spectral function A_{d;DL}(\epsilon_d; \omega) = -(1/\pi) Im G^R_{dd;DL}(\omega)
    using LDDL Eq. (50a).
    """
    w = np.asarray(omega, dtype=float)
    re_tot = np.zeros_like(w)
    im_tot = np.zeros_like(w)
    for b in solver.baths:
        re_b, gam_b = bare_bath_hybridization_dl(b, w)
        re_tot += re_b
        im_tot -= gam_b

    G_ret = 1.0 / (w - eps_d - re_tot - 1j * im_tot)
    A_dl = -(1.0 / np.pi) * np.imag(G_ret)
    return G_ret, A_dl


def dot_retarded_gf_qrt_time(
    solver: Any,
    eps_d: float,
    t_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute the real-time retarded dot Green's function via the Fermionic
    Quantum Regression Theorem (LDDL Eqs. 13-17 and Appendix B):
        \mathcal{G}^R_{dd}(t) = -i \theta(t) [exp(A t)]_{dd}
    """
    _, A, _, _ = solver.build_matrices(eps_d)
    eigvals, V = linalg.eig(A)
    V_inv = linalg.inv(V)
    weights = V[0, :] * V_inv[:, 0]
    t_pos = np.asarray(t_grid, dtype=float)
    exp_matrix = np.exp(np.outer(t_pos, eigvals))
    G_ret_t = -1j * (exp_matrix @ weights)
    G_ret_t[t_pos < 0.0] = 0.0
    return G_ret_t


def fourier_transform_qrt_to_spectral(
    t_grid: np.ndarray,
    G_ret_t: np.ndarray,
    omega: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fourier transform \mathcal{G}^R_{dd}(t) to frequency space."""
    t = np.asarray(t_grid, dtype=float)
    w = np.asarray(omega, dtype=float)
    dt = t[1] - t[0]
    phase = np.exp(1j * np.outer(w, t))
    weights = np.full_like(t, dt)
    weights[0] = 0.5 * dt
    weights[-1] = 0.5 * dt
    G_ret_w = phase @ (G_ret_t * weights)
    A_qrt_w = -(1.0 / np.pi) * np.imag(G_ret_w)
    return G_ret_w, A_qrt_w


# ============================================================================
# 2. Phase 1 Single-Bath Continuum Benchmarks (Note Eqs. 1-7)
# ============================================================================

def continuum_self_energy_infinite_D(omega: np.ndarray | float, Gamma0: float, r: float) -> np.ndarray:
    """
    Note Eq. (4): Exact Kramers-Kronig real part of self-energy in the wide-band limit D -> \infty:
        \Sigma(\omega) = -sign(\omega) \Gamma_0 |\omega|^r \tan(\pi r / 2)   for -1 < r < 1.
    """
    w = np.asarray(omega, dtype=float)
    if np.isclose(r, 0.0):
        return np.zeros_like(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(np.abs(w) == 0.0, 0.0, np.power(np.abs(w), r))
    return -np.sign(w) * Gamma0 * mag * np.tan(0.5 * np.pi * r)


def continuum_self_energy_finite_D(
    omega: np.ndarray | float,
    Gamma0: float,
    r: float,
    D: float,
) -> np.ndarray | float:
    """
    Exact Kramers-Kronig real part \Sigma_D(\omega) for a finite bandwidth cutoff [-D, D].
    Evaluated via vectorized 64-point Gauss-Legendre quadrature using u = s^{1/(1-r)}.
    """
    w_arr = np.atleast_1d(np.asarray(omega, dtype=float))
    sig_inf = continuum_self_energy_infinite_D(w_arr, Gamma0, r)
    out = np.copy(sig_inf)

    mask = (np.abs(w_arr) > 0.0) & (np.abs(w_arr) < 0.9995 * D)
    if np.any(mask):
        w_in = w_arr[mask]
        ratio2 = (w_in / D) ** 2
        u2_nodes = _GL64_S ** (2.0 / (1.0 - r))
        integrand_mat = 1.0 / (1.0 - np.outer(ratio2, u2_nodes))
        tail_int = (integrand_mat @ _GL64_W) / (1.0 - r)
        uv_correction = -(2.0 * Gamma0 * w_in / np.pi) * (D ** (r - 1.0)) * tail_int
        out[mask] = sig_inf[mask] - uv_correction

    return out if np.ndim(omega) > 0 else float(out[0])


def continuum_self_energy_ir_finite_D(
    omega: np.ndarray,
    Gamma0: float,
    r: float,
    D: float,
    omega_ir: float,
) -> np.ndarray:
    """
    Exact Kramers-Kronig real part \Sigma(\omega) for an IR-regularized power-law bath
        \Gamma_{PG}(\omega) = \Gamma_0 (\omega^2 + \omega_{IR}^2)^{r/2} \Theta(D - |\omega|)
    evaluated via the exact non-singular subtracted Hilbert identity:
        \Sigma(\omega) = \frac{2\omega}{\pi} \int_0^D d\omega' \frac{\Gamma_{PG}(\omega') - \Gamma_{PG}(\omega)}{\omega^2 - (\omega')^2}
                       + \frac{\Gamma_{PG}(\omega)}{\pi} \ln\left|\frac{D - \omega}{D + \omega}\right|
    """
    w_arr = np.atleast_1d(np.asarray(omega, dtype=float))
    s_nodes = _GL128_S
    w_prime = D * (s_nodes ** 2)
    dw_prime = 2.0 * D * s_nodes * _GL128_W

    gam_wp = Gamma0 * ((w_prime ** 2 + omega_ir ** 2) ** (0.5 * r))
    gam_w = Gamma0 * ((w_arr ** 2 + omega_ir ** 2) ** (0.5 * r))

    diff_num = gam_wp[None, :] - gam_w[:, None]
    diff_den = (w_arr[:, None] ** 2) - (w_prime[None, :] ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(diff_den) > 1e-14, diff_num / diff_den, 0.0)

    integral_part = (2.0 * w_arr / np.pi) * (ratio @ dw_prime)
    w_clip = np.clip(w_arr, -0.9999 * D, 0.9999 * D)
    log_part = (gam_w / np.pi) * np.log(np.abs((D - w_clip) / (D + w_clip)))
    return integral_part + log_part


def continuum_spectral_function(
    eps_d: float,
    omega: np.ndarray | float,
    Gamma0: float,
    r: float,
    D: Optional[float] = None,
) -> np.ndarray:
    """Note Eq. (1): Continuum dot spectral function A(\epsilon_d; \omega)."""
    w = np.asarray(omega, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        gam = Gamma0 * np.where(np.abs(w) == 0.0, 0.0 if r > 0 else np.inf, np.power(np.abs(w), r))
    if D is not None:
        gam = np.where(np.abs(w) <= D, gam, 0.0)
        sig = continuum_self_energy_finite_D(w, Gamma0, r, D)
    else:
        sig = continuum_self_energy_infinite_D(w, Gamma0, r)

    denom = (w - eps_d - sig) ** 2 + gam ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.where(denom > 0.0, (1.0 / np.pi) * gam / denom, 0.0)
    return A


def analytic_S_eps_d(
    eps_d: float,
    T: float,
    Gamma0: float,
    r: float,
    D: Optional[float] = None,
    eps_max: Optional[float] = None,
) -> float:
    """Note Eq. (5): Exact single-bath continuum Maxwell integral S(\epsilon_d)."""
    x_max = 40.0 if D is None else min(40.0, 0.999 * D / T)
    x_nodes = x_max * _GL128_S
    dx_weights = x_max * _GL128_W

    emx = np.exp(-x_nodes)
    thermal_weight = x_nodes * emx / ((1.0 + emx) ** 2)

    def phase_vec(w_vec: np.ndarray) -> np.ndarray:
        gam = Gamma0 * (np.abs(w_vec) ** r)
        sig = (
            continuum_self_energy_finite_D(w_vec, Gamma0, r, D)
            if D is not None
            else continuum_self_energy_infinite_D(w_vec, Gamma0, r)
        )
        upper = np.arctan((eps_d - w_vec + sig) / gam) / np.pi
        if eps_max is None:
            return 0.5 + upper
        lower = np.arctan((-eps_max - w_vec + sig) / gam) / np.pi
        return upper - lower

    diff_phase = phase_vec(-x_nodes * T) - phase_vec(x_nodes * T)
    return float(np.sum(thermal_weight * diff_phase * dx_weights))


def analytic_S0_note_eq7(T: float, Gamma0: float, r: float) -> float:
    """Note Eq. (6) [Eq. 7 in draft]: Single-bath S(0) for D -> \infty, \epsilon_{max} -> \infty."""
    tan_pr2 = np.tan(0.5 * np.pi * r)
    x_max = 45.0
    x_nodes = x_max * _GL128_S
    dx_weights = x_max * _GL128_W

    emx = np.exp(-x_nodes)
    weight = x_nodes * emx / ((1.0 + emx) ** 2)
    arg = tan_pr2 + ((x_nodes * T) ** (1.0 - r)) / Gamma0
    return float((2.0 / np.pi) * np.sum(weight * np.arctan(arg) * dx_weights))


# ============================================================================
# 3. Phase 2 Two-Bath Non-Equilibrium Benchmarks (Note Eqs. 8-14)
# ============================================================================

def alpha_func_eq14(r: float) -> float:
    """
    Note Eq. (14): Dimensionless thermal integral \alpha(r) for r < 2:
        \alpha(r) = \frac{1}{\ln 2} \int_0^\infty dx \frac{x^{1-r}}{(1+e^x)(1+e^{-x})}
    Evaluated via desingularized 128-point Gauss-Legendre quadrature x = x_max * s^{1/(2-r)}.
    """
    ln2 = np.log(2.0)
    if np.isclose(r, 0.0):
        return 1.0

    p = 2.0 - r
    x_max = 50.0
    x_nodes = x_max * (_GL128_S ** (1.0 / p))
    jacobian_w = (x_max ** p / p) * _GL128_W
    emx = np.exp(-x_nodes)
    fermi_factor = emx / ((1.0 + emx) ** 2)
    return float(np.sum(fermi_factor * jacobian_w) / ln2)


def analytic_tilde_S0_eq14(T_S: float, Gamma0: float, GammaS0: float, r: float) -> float:
    """
    Note Eq. (14): Low-temperature asymptotic scaling formula at \epsilon_d = 0:
        \tilde{S}_{Eq.14}(0) = \frac{\Gamma_S}{\Gamma_0} \frac{\alpha(r)}{T_S^r} S(0)
    where S(0) = r \ln 2.
    """
    S0_lowT = r * np.log(2.0)
    return float((GammaS0 / Gamma0) * (alpha_func_eq14(r) / (T_S ** r)) * S0_lowT)


def analytic_tilde_S0_eq12(T_S: float, Gamma0: float, GammaS0: float, r: float) -> float:
    """
    Note Eq. (12) evaluated at \epsilon_d = 0 (weak-probe limit \Gamma_S \ll \Gamma(\omega),
    infinite bandwidth D -> \infty, retaining finite-T_S arctan argument).
    """
    tan_pr2 = np.tan(0.5 * np.pi * r)
    pref = (2.0 / np.pi) * (GammaS0 / (Gamma0 * (T_S ** r)))

    p = 2.0 - r
    x_max = 50.0
    x_nodes = x_max * (_GL128_S ** (1.0 / p))
    jacobian_w = (x_max ** p / p) * _GL128_W

    emx = np.exp(-x_nodes)
    fermi_factor = emx / ((1.0 + emx) ** 2)
    arg = tan_pr2 + ((x_nodes * T_S) ** (1.0 - r)) / Gamma0
    return float(pref * np.sum(fermi_factor * np.arctan(arg) * jacobian_w))


def analytic_two_bath_tilde_S_eps_d(
    eps_d: float | np.ndarray,
    T_S: float,
    Gamma0: float,
    GammaS0: float,
    r: float,
    D_PG: Optional[float] = 1.0,
    D_S: Optional[float] = 1.0,
    omega_ir_PG: float = 0.0,
    probe_shape_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    eps_max: Optional[float] = None,
) -> float | np.ndarray:
    """
    Exact two-bath continuum integral from Note Eqs. (9)-(11) integrated over \epsilon_d' \in [-\epsilon_{max}, \epsilon_d].
    Uses exact subtracted Kramers-Kronig self-energy when omega_ir_PG > 0.
    """
    ed_arr = np.atleast_1d(np.asarray(eps_d, dtype=float))
    D_lim = D_S if D_S is not None else (D_PG if D_PG is not None else None)
    x_max = 45.0 if D_lim is None else min(45.0, 0.999 * D_lim / T_S)

    p = 2.0 - max(r, 0.0)
    x_nodes = x_max * (_GL128_S ** (1.0 / p))
    dx_weights = (x_max / p) * (_GL128_S ** (1.0 / p - 1.0)) * _GL128_W

    emx = np.exp(-x_nodes)
    thermal_weight = x_nodes * emx / ((1.0 + emx) ** 2)

    def eval_branch_matrix(w_vec: np.ndarray) -> np.ndarray:
        if omega_ir_PG > 0.0:
            g_p = Gamma0 * ((w_vec * w_vec + omega_ir_PG * omega_ir_PG) ** (0.5 * r))
            D_use = D_PG if D_PG is not None else 1.0
            s_t = continuum_self_energy_ir_finite_D(w_vec, Gamma0, r, D_use, omega_ir_PG)
            if D_PG is not None:
                g_p = np.where(np.abs(w_vec) <= D_PG, g_p, 0.0)
        else:
            g_p = Gamma0 * (np.abs(w_vec) ** r)
            if D_PG is not None:
                g_p = np.where(np.abs(w_vec) <= D_PG, g_p, 0.0)
                s_t = continuum_self_energy_finite_D(w_vec, Gamma0, r, D_PG)
            else:
                s_t = continuum_self_energy_infinite_D(w_vec, Gamma0, r)

        if probe_shape_fn is not None:
            g_s = GammaS0 * np.asarray(probe_shape_fn(w_vec), dtype=float)
        else:
            g_s = np.full_like(w_vec, GammaS0)
        if D_S is not None:
            g_s = np.where(np.abs(w_vec) <= D_S, g_s, 0.0)

        g_tot = g_p + g_s
        branch = np.where(g_tot > 0.0, g_s / g_tot, 0.0)

        upper = np.arctan((ed_arr[:, None] - w_vec[None, :] + s_t[None, :]) / g_tot[None, :]) / np.pi
        if eps_max is None:
            return branch[None, :] * (0.5 + upper)
        lower = np.arctan((-eps_max - w_vec[None, :] + s_t[None, :]) / g_tot[None, :]) / np.pi
        return branch[None, :] * (upper - lower)

    diff_mat = eval_branch_matrix(-x_nodes * T_S) - eval_branch_matrix(x_nodes * T_S)
    res = diff_mat @ (thermal_weight * dx_weights)
    return res if np.ndim(eps_d) > 0 else float(res[0])