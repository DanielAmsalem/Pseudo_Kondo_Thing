"""
lyapunov_solver.py
------------------
Single-particle continuous Lyapunov solver for quadratic Lindblad-driven
tight-binding impurity models (RL-NESS Appendix C / LDDL Sec. II B & III E).

Solves the M x M matrix Lyapunov equation A C + C A^\dagger + M = 0 via exact
eigendecomposition of A (RL-NESS Appendix C, Eqs. C4-C6), completely avoiding
SciPy's buggy Fortran zgees/ztrsyl wrapper in solve_continuous_lyapunov.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy import linalg
from bath import Bath


def _eval_single_eps_d(args) -> float:
    """Top-level worker function for parallel \epsilon_d sweeps."""
    solver, ed, target_bath, method, T_overrides = args
    return solver.dn_d_dT(
        float(ed),
        target_bath=target_bath,
        method=method,
        T_overrides=T_overrides,
    )


def solve_lyapunov_rl_ness_eq_c6(A: np.ndarray, diag_M: np.ndarray) -> np.ndarray:
    """
    Exact solution of the continuous Lyapunov equation:
        A X + X A^\dagger + diag(diag_M) = 0
    via eigendecomposition A = V diag(\lambda) V^{-1} (RL-NESS Appendix C, Eqs. C4-C6):
        X = \int_0^\infty e^{A t'} diag(diag_M) e^{A^\dagger t'} dt'
          = V [ - (V^{-1} diag(diag_M) (V^{-1})^\dagger)_{jk} / (\lambda_j + \lambda_k^*) ] V^\dagger
    """
    eigvals, V = linalg.eig(A)
    W = linalg.inv(V)
    M_tilde = (W * diag_M[None, :]) @ W.conj().T
    denom = eigvals[:, None] + eigvals[None, :].conj()
    X_tilde = -M_tilde / denom
    return V @ X_tilde @ V.conj().T


class LyapunovSolver:
    def __init__(self, baths: Sequence[Bath]) -> None:
        if len(baths) == 0:
            raise ValueError("At least one Bath must be provided.")
        self.baths: List[Bath] = list(baths)
        self.slices: Dict[str, slice] = {}
        idx = 1
        for b in self.baths:
            self.slices[b.name] = slice(idx, idx + b.n_modes)
            idx += b.n_modes
        self.dim: int = idx

        self.eps_all = np.concatenate([b.epsilons for b in self.baths])
        self.v_all = np.concatenate([b.v for b in self.baths])
        self.gamma_all = np.concatenate([b.gammas for b in self.baths])

    def build_matrices(
        self,
        eps_d: float,
        T_overrides: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        H = np.zeros((self.dim, self.dim), dtype=float)
        H[0, 0] = eps_d
        H[0, 1:] = self.v_all
        H[1:, 0] = self.v_all
        H[1:, 1:] = np.diag(self.eps_all)

        lam1_diag = np.zeros(self.dim, dtype=float)
        lam2_diag = np.zeros(self.dim, dtype=float)
        for b in self.baths:
            t_ovr = None if T_overrides is None else T_overrides.get(b.name, None)
            l1, l2 = b.lindblad_rates(T_override=t_ovr)
            sl = self.slices[b.name]
            lam1_diag[sl] = l1
            lam2_diag[sl] = l2

        lam_plus_diag = lam1_diag + lam2_diag
        A = -1j * H - np.diag(lam_plus_diag)
        return H, A, lam1_diag, lam2_diag

    def solve_ness_correlation_matrix(
        self,
        eps_d: float,
        T_overrides: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        _, A, _, lam2_diag = self.build_matrices(eps_d, T_overrides=T_overrides)
        return solve_lyapunov_rl_ness_eq_c6(A, 2.0 * lam2_diag)

    def solve_ness_observables(
        self,
        eps_d: float,
        T_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float | np.ndarray]:
        C_ness = self.solve_ness_correlation_matrix(eps_d, T_overrides=T_overrides)
        n_d = float(np.real(C_ness[0, 0]))
        currents = {}
        bath_occs = {}
        for b in self.baths:
            sl = self.slices[b.name]
            c_0q = C_ness[0, sl]
            currents[b.name] = float(2.0 * np.sum(b.v * np.imag(c_0q)))
            bath_occs[b.name] = np.real(np.diag(C_ness[sl, sl]))
        return {
            "n_d": n_d,
            "currents": currents,
            "bath_occupations": bath_occs,
            "C_ness": C_ness,
        }

    def solve_exact_dC_dT(
        self,
        eps_d: float,
        target_bath: Optional[str] = None,
        T_overrides: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        _, A, _, _ = self.build_matrices(eps_d, T_overrides=T_overrides)
        dlam2_dT = np.zeros(self.dim, dtype=float)

        for b in self.baths:
            if target_bath is not None and b.name != target_bath:
                continue
            t_ovr = None if T_overrides is None else T_overrides.get(b.name, None)
            sl = self.slices[b.name]
            dlam2_dT[sl] = b.gammas * b.dfermi_dT(b.epsilons, T_override=t_ovr)

        return solve_lyapunov_rl_ness_eq_c6(A, 2.0 * dlam2_dT)

    def _build_schur_quadrature(
        self,
        target_bath: Optional[str] = None,
        T_overrides: Optional[Dict[str, float]] = None,
        n_quad: int = 3000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        baths_active = [
            b for b in self.baths if (target_bath is None or b.name == target_bath)
        ]
        T_ref = max(
            (b.T if T_overrides is None or b.name not in T_overrides else T_overrides[b.name])
            for b in baths_active
        )
        # Integrate strictly inside the physical tight-binding bandwidth [-D_max, D_max]
        # to prevent dividing out-of-band 1/\omega^2 Lorentzian tails (LDDL Appendix A)
        D_max = max(b.D for b in self.baths)
        w_max = 0.9999 * D_max

        u = np.linspace(-1.0, 1.0, n_quad + 1)
        w_floor = min(0.05 * D_max, max(0.02 * T_ref, 5.0e-5 * D_max))
        scale = np.arcsinh(w_max / w_floor)
        w_edges = w_floor * np.sinh(scale * u)
        w_mid = 0.5 * (w_edges[:-1] + w_edges[1:])
        dw = np.diff(w_edges)

        sig_dl = np.zeros_like(w_mid)
        gam_dl = np.zeros_like(w_mid)
        phi_dT = np.zeros_like(w_mid)

        chunk_size = 1000
        for b in self.baths:
            t_ovr = None if T_overrides is None else T_overrides.get(b.name, None)
            df_dT_q = (
                b.dfermi_dT(b.epsilons, T_override=t_ovr)
                if (target_bath is None or b.name == target_bath)
                else np.zeros_like(b.epsilons)
            )
            v2_q = b.v ** 2
            eps_q = b.epsilons
            gam_q = b.gammas

            for start in range(0, b.n_modes, chunk_size):
                end = min(b.n_modes, start + chunk_size)
                diff = w_mid[:, None] - eps_q[None, start:end]
                g_sl = gam_q[None, start:end]
                v2_sl = v2_q[None, start:end]
                denom = diff * diff + g_sl * g_sl

                lorentz = v2_sl * g_sl / denom
                gam_dl += np.sum(lorentz, axis=1)
                sig_dl += np.sum(v2_sl * diff / denom, axis=1)
                if target_bath is None or b.name == target_bath:
                    phi_dT += lorentz @ df_dT_q[start:end]

        return w_mid, dw, sig_dl, gam_dl, phi_dT

    def dn_d_dT(
        self,
        eps_d: float,
        target_bath: Optional[str] = None,
        method: str = "exact_lyapunov",
        rel_dT: float = 1e-4,
        T_overrides: Optional[Dict[str, float]] = None,
    ) -> float:
        if method == "exact_lyapunov":
            dC = self.solve_exact_dC_dT(eps_d, target_bath=target_bath, T_overrides=T_overrides)
            return float(np.real(dC[0, 0]))

        if method == "schur_lddl":
            w_mid, dw, sig_dl, gam_dl, phi_dT = self._build_schur_quadrature(
                target_bath=target_bath, T_overrides=T_overrides
            )
            G2 = 1.0 / ((w_mid - eps_d - sig_dl) ** 2 + gam_dl ** 2)
            return float(np.sum((1.0 / np.pi) * G2 * phi_dT * dw))

        if method == "finite_diff":
            base_T = {
                b.name: (b.T if T_overrides is None or b.name not in T_overrides else T_overrides[b.name])
                for b in self.baths
            }
            baths_to_shift = [b.name for b in self.baths] if target_bath is None else [target_bath]
            ref_T = max(base_T[name] for name in baths_to_shift)
            dT = rel_dT * ref_T
            if dT <= 0.0:
                raise ValueError("Centered finite difference requires target bath temperature T > 0.")

            T_plus = dict(base_T)
            T_minus = dict(base_T)
            for name in baths_to_shift:
                T_plus[name] += dT
                T_minus[name] -= dT

            n_plus = float(np.real(self.solve_ness_correlation_matrix(eps_d, T_overrides=T_plus)[0, 0]))
            n_minus = float(np.real(self.solve_ness_correlation_matrix(eps_d, T_overrides=T_minus)[0, 0]))
            return (n_plus - n_minus) / (2.0 * dT)

        raise ValueError(f"Unknown method '{method}'. Choose 'exact_lyapunov', 'schur_lddl', or 'finite_diff'.")

    def sweep_eps_d_protocol(
        self,
        eps_d_grid: np.ndarray,
        target_bath: Optional[str] = None,
        method: str = "schur_lddl",
        T_overrides: Optional[Dict[str, float]] = None,
        n_workers: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        if method == "schur_lddl":
            w_mid, dw, sig_dl, gam_dl, phi_dT = self._build_schur_quadrature(
                target_bath=target_bath, T_overrides=T_overrides
            )
            diff_ed = w_mid[None, :] - eps_d_grid[:, None] - sig_dl[None, :]
            G2_matrix = 1.0 / (diff_ed ** 2 + gam_dl[None, :] ** 2)
            dn_dT_vals = (1.0 / np.pi) * (G2_matrix @ (phi_dT * dw))

            eps_min = eps_d_grid[0]
            atan_upper = np.arctan((eps_d_grid[:, None] - w_mid[None, :] + sig_dl[None, :]) / gam_dl[None, :])
            atan_lower = np.arctan((eps_min - w_mid[None, :] + sig_dl[None, :]) / gam_dl[None, :])
            S_cum = -((atan_upper - atan_lower) / np.pi) @ ((phi_dT / gam_dl) * dw)

            return {
                "eps_d": eps_d_grid,
                "dn_dT": dn_dT_vals,
                "S_eps_d": S_cum,
            }

        if n_workers is None:
            n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

        if n_workers > 1 and len(eps_d_grid) > 10:
            tasks = [
                (self, float(ed), target_bath, method, T_overrides)
                for ed in eps_d_grid
            ]
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                dn_dT_vals = np.array(list(pool.map(_eval_single_eps_d, tasks)), dtype=float)
        else:
            dn_dT_vals = np.zeros_like(eps_d_grid, dtype=float)
            for i, ed in enumerate(eps_d_grid):
                dn_dT_vals[i] = self.dn_d_dT(
                    float(ed),
                    target_bath=target_bath,
                    method=method,
                    T_overrides=T_overrides,
                )

        d_ed = np.diff(eps_d_grid)
        trapz_increments = 0.5 * (dn_dT_vals[:-1] + dn_dT_vals[1:]) * d_ed
        S_cum = np.concatenate(([0.0], -np.cumsum(trapz_increments)))
        return {
            "eps_d": eps_d_grid,
            "dn_dT": dn_dT_vals,
            "S_eps_d": S_cum,
        }