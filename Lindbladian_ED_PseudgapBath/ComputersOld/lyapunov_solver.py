"""
lyapunov_solver.py
==================
Single-particle continuous Lyapunov solver for quadratic Lindblad-driven
tight-binding impurity models (RL-NESS Appendix C / LDDL Sec. II B & III E).

WHAT THIS MODULE DOES PHYSICALLY
--------------------------------
We want to find the Non-Equilibrium Steady State (NESS) density matrix rho_{NESS}
satisfying the stationary Lindblad master equation (LDDL Eqs. 5, 10, 40; RL-NESS Eqs. 1, 7, 11):
    d/dt rho = L rho = -i [H, rho] + D rho = 0

where the total tight-binding Hamiltonian H in the Star Geometry is:
    H = eps_d d^+ d +
        sum_{a} : (
            sum_{q=1}^{N_a} : eps_{a,q} c_{a,q}^+ c_{a,q}) +
        sum_{a} : (
            sum_{q=1}^{N_a} : v_{a,q} (d^+ c_{a,q} + c_{a,q}^+ d))

and the Lindblad dissipator D acts locally on every bath mode q = (a, n) with
emptying rate lambda_q(1) = gamma_q (1 - f_a(eps_q)) and filling rate
lambda_q(2) = gamma_q f_a(eps_q) (LDDL Eq. 39; RL-NESS Eq. 10):
    D rho = sum_{q=1}^{N_tot} : lambda_q(1) (2 c_q rho c_q^+ - {c_q^+ c_q, rho}) +
            sum_{q=1}^{N_tot} : lambda_q(2) (2 c_q^+ rho c_q - {c_q c_q^+, rho})

Because H is quadratic and the Lindblad jump operators are linear in the fermionic
creation/annihilation operators, Wick's theorem holds and the entire many-body state
is uniquely determined by the M x M single-particle correlation matrix (where M = 1 + N_tot).
In RL-NESS Appendix C (Eqs. C1-C6) and LDDL Sec. II B (Eqs. 21-22), the hole correlation
matrix P_{rs}(t) = <c_r(t) c_s^+(t)> obeys the closed linear matrix differential equation:
    d/dt P = A P + P A^+ + 2 Lambda(1),   where   A = -i H - [Lambda(1) + Lambda(2)]

Equivalently, the normal particle correlation matrix C_{rs}(t) = <c_s^+(t) c_r(t)> = delta_{rs} - P_{rs}(t)
obeys:
    d/dt C = A C + C A^+ + 2 Lambda(2)

In the steady state (d/dt C_{NESS} = 0), finding all equal-time observables (such as the
dot occupation n_d = <d^+ d> = C_{0,0}, bath occupations N_q = <c_q^+ c_q> = C_{q,q}, and
particle currents I_a) reduces to solving the M x M continuous algebraic Lyapunov equation:
    A C_{NESS} + C_{NESS} A^+ + 2 Lambda(2) = 0

THREE SOLVER METHODS IMPLEMENTED & CROSS-CHECKED IN THIS MODULE
---------------------------------------------------------------
1. "exact_lyapunov" (Full M x M Matrix Eigendecomposition Solver, RL-NESS Eq. C6):
   Differentiates the Lyapunov equation analytically with respect to bath temperature T_a:
       A (dC_{NESS} / dT_a) + (dC_{NESS} / dT_a) A^+ + 2 (dLambda(2) / dT_a) = 0
   and solves it in closed form by diagonalizing the M x M matrix A = V diag(lambda) V^{-1}.

2. "finite_diff" (Centered Finite-Difference Matrix Solver):
   Solves A C_{NESS} + C_{NESS} A^+ + 2 Lambda(2) = 0 at temperatures T_a + dT and
   T_a - dT, and evaluates:
       dn_d / dT_a = (n_d(T_a + dT) - n_d(T_a - dT)) / (2 dT)

3. "schur_lddl" (Exact Schur-Complement LDDL Reduction for N_tot >= 10,000 Modes):
   Uses the arrowhead structure of H to analytically eliminate the bath blocks via the
   Schur complement (LDDL Eqs. 24-25, 50a, 58), expressing C_{0,0} and dn_d / dT_a as
   an exact frequency integral over the discrete tight-binding self-energies Sigma_{DL}(w)
   and Gamma_{DL}(w). Because Sigma_{DL}(w) and Gamma_{DL}(w) do not depend on the dot energy
   eps_d, this allows 14,000-mode tight-binding sweeps across hundreds of gate voltages eps_d
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy import linalg
from bath import Bath


def _eval_single_eps_d(args) -> float:
    """
    Top-level worker function for parallel gate-voltage (eps_d) sweeps using
    Python's ProcessPoolExecutor when method="exact_lyapunov" or "finite_diff".
    """
    solver, ed, target_bath, method, T_overrides = args
    return solver.dn_d_dT(
        float(ed),
        target_bath=target_bath,
        method=method,
        T_overrides=T_overrides,
    )


def solve_lyapunov_rl_ness_eq_c6(A: np.ndarray, diag_M: np.ndarray) -> np.ndarray:
    """
    Exact solution of the continuous algebraic Lyapunov matrix equation:
        A X + X A^+ + diag(diag_M) = 0
    via eigendecomposition of A (RL-NESS Appendix C, Eqs. C4-C6).

    Mathematical Procedure
    ----------------------
    1. Any matrix A with eigenvalues having strictly negative real parts (Re(lambda_j) < 0,
       guaranteed by the positive Lindblad damping rates gamma_q > 0) has a unique steady-state
       solution X given by the time integral (RL-NESS Eq. C6):
           X = int_{0}{infty} : exp(A t) diag(diag_M) exp(A^+ t) dt

    2. Diagonalizing A = V diag(lambda_1, ..., lambda_M) V^{-1} (and setting W = V^{-1})
       transforms the source matrix diag(diag_M) into the eigenbasis of A:
           M_tilde = W diag(diag_M) W^+

    3. Carrying out the elementary exponential time integral mode-by-mode:
           int_{0}{infty} : exp((lambda_j + lambda_k^*) t) dt = -1 / (lambda_j + lambda_k^*)
       yields the exact solution in the eigenbasis:
           X_tilde_{j,k} = -M_tilde_{j,k} / (lambda_j + lambda_k^*)

    4. Transforming back to the original tight-binding site basis gives:
           X = V X_tilde V^+

    Why We Use This Instead of `scipy.linalg.solve_continuous_lyapunov`
    -------------------------------------------------------------------
    When two or more baths are connected to the dot and share similar mode energies eps_q
    and damping rates gamma_q, A has near-degenerate complex eigenvalues. SciPy's wrapper
    around the Fortran LAPACK routines `zgees` / `ztrsyl` can segfault when rotating
    near-degenerate complex Schur blocks on multi-core nodes. In contrast, the denominator
    (lambda_j + lambda_k^*) has Re(lambda_j + lambda_k^*) <= -2 gamma_min < 0 and never
    vanishes even when lambda_j == lambda_k

    Parameters
    ----------
    A : np.ndarray (complex), shape (M, M)
        Damped single-particle evolution generator A = -i H - Lambda^{(+)}.
    diag_M : np.ndarray (float), shape (M,)
        Diagonal entries of the source matrix M (either 2 * lambda_q(2) for C_{NESS}
        or 2 * d[lambda_q(2)]/dT for dC_{NESS}/dT).

    Returns
    -------
    X : np.ndarray (complex), shape (M, M)
        Exact solution matrix X satisfying A X + X A^+ + diag(diag_M) = 0.
    """
    eigvals, V = linalg.eig(A)
    W = linalg.inv(V)
    M_tilde = (W * diag_M[None, :]) @ W.conj().T
    denom = eigvals[:, None] + eigvals[None, :].conj()
    X_tilde = -M_tilde / denom
    return V @ X_tilde @ V.conj().T


class LyapunovSolver:
    """
    Assembles and solves the single-particle NESS Lyapunov equation for a resonant
    impurity level at energy eps_d coupled to an arbitrary list of discretized
    tight-binding baths `baths = [bath_1, bath_2, ...]`.

    Single-Particle Basis Ordering
    ------------------------------
    Index 0                      : Impurity fermion mode d
    Indices 1 .. N_1             : Tight-binding modes c_{1, q} of baths[0]
    Indices N_1 + 1 .. N_1 + N_2 : Tight-binding modes c_{2, k} of baths[1], etc.
    Total matrix dimension       : M = 1 + (sum_{a} : N_a)
    """

    def __init__(self, baths: Sequence[Bath]) -> None:
        if len(baths) == 0:
            raise ValueError("At least one Bath object must be provided to LyapunovSolver.")
        self.baths: List[Bath] = list(baths)

        # Record the matrix index slice [start:stop] belonging to each bath's modes
        self.slices: Dict[str, slice] = {}
        idx = 1
        for b in self.baths:
            self.slices[b.name] = slice(idx, idx + b.n_modes)
            idx += b.n_modes
        self.dim: int = idx

        # Concatenate all bath mode energies eps_q, hoppings v_q, and damping rates gamma_q
        self.eps_all = np.concatenate([b.epsilons for b in self.baths])
        self.v_all = np.concatenate([b.v for b in self.baths])
        self.gamma_all = np.concatenate([b.gammas for b in self.baths])

    def build_matrices(
        self,
        eps_d: float,
        T_overrides: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct the M x M single-particle tight-binding Hamiltonian H, the damped
        generator A = -i H - Lambda^{(+)}, and the diagonal Lindblad rate vectors
        lambda(1) and lambda(2) (RL-NESS Appendix C, Eqs. C1-C3; LDDL Eqs. 42a, 48).

        Structure of the Matrices
        -------------------------
        1. Tight-binding Hamiltonian H (Star / Arrowhead Geometry):
               H = [ eps_d    v_{PG}^T             v_S^T            ]
                   [ v_{PG}   diag(eps_{PG, n})    0                ]
                   [ v_S      0                    diag(eps_{S, k}) ]

        2. Diagonal Lindblad rate matrices Lambda^(1) and Lambda^(2):
           Because Lindblad driving acts only on the bath modes and not directly on the
           dot (LDDL Eq. 48), index 0 is 0 and bath mode q = (a, n) has:
               lambda_q(1) = gamma_q (1 - f_a(eps_q; T_a))   [depletion rate]
               lambda_q(2) = gamma_q f_a(eps_q; T_a)         [filling rate]

        3. Total damping matrix Lambda(+) = Lambda(1) + Lambda(2):
           Notice that the Fermi functions f_a(eps_q) cancel out of the sum:
               Lambda(+) = diag(0, gamma_{PG, 1}, ..., gamma_{S, N_S})
           so A = -i H - Lambda(+) is strictly independent of all bath temperatures T_a

        Parameters
        ----------
        eps_d : float
            Impurity on-site energy eps_d.
        T_overrides : Optional[Dict[str, float]]
            Optional dictionary `{bath_name: T_new}` temporarily overriding the temperature
            of specific baths (used during finite-difference temperature shifts).

        Returns
        -------
        H : np.ndarray (float), shape (M, M)
            Single-particle tight-binding Hamiltonian matrix.
        A : np.ndarray (complex), shape (M, M)
            Damped evolution matrix A = -i H - diag(lam_plus_diag).
        lam1_diag : np.ndarray (float), shape (M,)
            Diagonal entries of the Lindblad emptying matrix Lambda(1).
        lam2_diag : np.ndarray (float), shape (M,)
            Diagonal entries of the Lindblad filling matrix Lambda(2).
        """
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
        """
        Solve the M x M continuous Lyapunov equation for the steady-state normal
        correlation matrix C_{rs} = <c_s^+ c_r>_{NESS}:
            A C_{NESS} + C_{NESS} A^+ + 2 Lambda(2) = 0
        """
        _, A, _, lam2_diag = self.build_matrices(eps_d, T_overrides=T_overrides)
        return solve_lyapunov_rl_ness_eq_c6(A, 2.0 * lam2_diag)

    def solve_ness_observables(
        self,
        eps_d: float,
        T_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float | np.ndarray]:
        """
        Compute all equal-time steady-state physical observables from C_{NESS}:
          1. Impurity occupation:
                 n_d = <d^+ d>_{NESS} = Re(C_{0, 0})
          2. Bath mode occupations for each bath a:
                 N_{a, q} = <c_{a, q}^+ c_{a, q}>_{NESS} = Re(C_{q, q})
          3. Steady-state particle current flowing from bath a into the dot
             (RL-NESS Eq. 20; LDDL Eq. 54):
                 I_a = 2 * (sum_{q=1}^{N_a} : v_{a, q} Im(<c_{a, q}^+ d>_{NESS}))
                         = 2 * (sum_{q=1}^{N_a} : v_{a, q} Im(C_{0, q}))
        """
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
        """
        Solve the differentiated M x M Lyapunov equation for the exact temperature
        derivative of the correlation matrix, dC_{NESS} / dT:
            A (dC_{NESS} / dT) + (dC_{NESS} / dT) A^+ + 2 (dLambda(2) / dT) = 0

        How Single-Bath (Step 1a) vs. Two-Bath (Step 1b) Differentiation Works
        ----------------------------------------------------------------------
        Because A = -i H - Lambda^{(+)} does not depend on temperature, differentiating
        A C_{NESS} + C_{NESS} A^+ + 2 Lambda(2) = 0 with respect to T only differentiates
        the Fermi distribution inside Lambda(2)_q = gamma_q f_a(eps_q; T_a):
            d[lambda_q(2)] / dT_a = gamma_q * (df_a(eps_q; T_a) / dT_a)

        - If `target_bath is None` (Step 1a equilibrium protocol):
              Differentiates with respect to a common temperature T shared by ALL connected baths.
        - If `target_bath == "S"` (Step 1b non-equilibrium two-bath protocol):
              Differentiates ONLY with respect to the probe bath temperature T_S while holding
              the power-law bath temperature T_{PG} fixed (so d[lambda_q(2)]/dT_S = 0 for all
              modes q in bath "PG").
        """
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
        """
        Precompute the exact discrete-mode tight-binding self-energies Sigma_{DL}(w),
        Gamma_{DL}(w), and thermal derivative kernel Phi(w) on a 3,000-point sinh-clustered
        frequency quadrature grid inside [-D_max, +D_max].

        Mathematical Derivation of the Schur-Complement LDDL Reduction
        --------------------------------------------------------------
        In LDDL Sec. II B (Eqs. 24-25) and Sec. III D-E (Eqs. 50a, 51a, 58), the (0, 0)
        dot entry of the exact Lyapunov solution C_{NESS} is given by integrating the
        Keldysh Green's function G^K_{dd;DL}(w):
            n_d(eps_d) = C_{0,0}
                       = int_{-infty}{infty} : (1 / pi) |G^R_{dd;DL}(eps_d; w)|^2 *
                         (sum_{a} : (sum_{q=1}^{N_a} : f_a(eps_{a,q}; T_a) |v_{a,q}|^2 gamma_{a,q} / ((w - eps_{a,q})^2 + gamma_{a,q}^2))) dw

        where the dressed dot propagator squared is:
            |G^R_{dd;DL}(eps_d; w)|^2 = 1 / ((w - eps_d - Sigma_{DL}(w))^2 + Gamma_{DL}(w)^2)
        with total discrete tight-binding self-energies:
            Sigma_{DL}(w) = sum_{a} : (sum_{q=1}^{N_a} : |v_{a,q}|^2 (w - eps_{a,q}) / ((w - eps_{a,q})^2 + gamma_{a,q}^2))
            Gamma_{DL}(w) = sum_{a} : (sum_{q=1}^{N_a} : |v_{a,q}|^2 gamma_{a,q} / ((w - eps_{a,q})^2 + gamma_{a,q}^2))

        Differentiating n_d(eps_d) with respect to the temperature T of the active bath(s)
        (`target_bath`) gives:
            dn_d(eps_d) / dT = int_{-D_max}{D_max} : (1 / pi) |G^R_{dd;DL}(eps_d; w)|^2 Phi(w) dw
        where the discrete thermal kernel Phi(w) sums over the active bath modes:
            Phi(w) = sum_{q in active} : (df(eps_q; T) / dT) |v_q|^2 gamma_q / ((w - eps_q)^2 + gamma_q^2)

        Why We Restrict the Quadrature Grid to [-D_max, +D_max]
        -------------------------------------------------------
        All discrete tight-binding modes lie strictly inside [-D_max, +D_max]. Outside
        [-D_max, +D_max], there are no bath modes, only artificial 1/w^2 Lorentzian tails
        from the Lindblad broadenings gamma_q (LDDL Appendix A). Restricting `w_max = 0.9999 * D_max`
        integrates purely over the physical tight-binding band and avoids dividing two 1/w^2
        tails outside the band.
        """
        baths_active = [
            b for b in self.baths if (target_bath is None or b.name == target_bath)
        ]
        T_ref = max(
            (b.T if T_overrides is None or b.name not in T_overrides else T_overrides[b.name])
            for b in baths_active
        )
        D_max = max(b.D for b in self.baths)
        w_max = 0.9999 * D_max

        # Sinh-mapped frequency grid densely clustering quadrature points around w = 0
        # at the thermal scale w_floor ~ 0.02 * T_ref while extending smoothly to +/- w_max.
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
        """
        Compute the temperature derivative of the steady-state dot occupation,
        dn_d(eps_d) / dT, at a single impurity energy eps_d using one of three methods:
          - "exact_lyapunov": Solves the differentiated M x M matrix Lyapunov equation.
          - "schur_lddl":     Evaluates the Schur-complement LDDL frequency integral.
          - "finite_diff":    Evaluates the centered finite difference
                              (n_d(T + dT) - n_d(T - dT)) / (2 dT) with dT = rel_dT * T.
        """
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
        """
        Execute the full gate-voltage sweep protocol across `eps_d_grid`:
          1. Evaluate dn_d(eps_d) / dT at every point in `eps_d_grid`.
          2. Integrate over eps_d' from eps_min = eps_d_grid[0] up to each eps_d to obtain
             the Maxwell / non-equilibrium entropy curve:
                 S(eps_d) = -int_{eps_min}{eps_d} : (dn_d(eps_d') / dT) deps_d'

        Closed-Form Gate Integration in the "schur_lddl" Method
        -------------------------------------------------------
        When `method == "schur_lddl"`, the discrete bath functions Sigma_{DL}(w), Gamma_{DL}(w),
        and Phi(w) are independent of eps_d'. Therefore, the integral over eps_d' of the
        Lorentzian |G^R_{dd;DL}(eps_d'; w)|^2 is carried out analytically in closed form:
            -int_{eps_min}{eps_d} : (1 / pi) |G^R_{dd;DL}(eps_d'; w)|^2 deps_d'
            = -(1 / (pi Gamma_{DL}(w))) * [ arctan((eps_d - w + Sigma_{DL}(w)) / Gamma_{DL}(w))
                                          - arctan((eps_min - w + Sigma_{DL}(w)) / Gamma_{DL}(w)) ]
        Then integrating over w gives the exact cumulative integral S(eps_d) without any
        trapezoidal discretization error in eps_d'!
        """
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

        # Dense matrix solver path (parallelized across CPUs via ProcessPoolExecutor)
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