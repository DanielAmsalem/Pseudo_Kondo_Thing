"""
tv_correspondence.py
--------------------
Exact unitary mapping (t <-> v correspondence) between:
  1. STAR GEOMETRY (diagonal bath basis, parameters {epsilons_n, v_n}):
       H = eps_d*(d^+ d)
            + sum_{n=1}^{N} : eps_n*(c_n^+ c_n)
            + sum_{n=1}^{N} : v_n*(d^+ c_n + c_n^+ d)
     [RL-NESS Eqs. (8)-(9); LDDL Eqs. (29)-(30)]

  2. CHAIN GEOMETRY (1D nearest-neighbor tight-binding)
     parameters {t0, eps_chain_k, t_hop_k}:
       H = eps_d d^+ d + t0 (d^+ f_0 + f_0^+ d)
           + sum_{k=0}^{N-1} : eps_chain_k f_k^+ f_k
           + sum_{k=0}^{N-2} : t_hop_{k+1} (f_k^+ f_{k+1} + f_{k+1}^+ f_k)
     [RL-NESS Eq. (12); LDDL Sec. IV]

Also implements the LDDL / RL-NESS Sec. II C particle-hole level-doubling
transformation (Eqs. 13-14 of RL-NESS) that maps a finite-temperature bath
into two separate 1D tight-binding chains (a filled "particle" chain and an
empty "hole" chain) for MPS / tDMRG / IMEvo simulations.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from scipy import linalg


@dataclass
class ChainGeometry:
    """
    Stores the 1D nearest-neighbor tight-binding chain parameters obtained from
    tridiagonalizing a Star-Geometry bath (RL-NESS Eq. 12).

    Attributes
    ----------
    t0 : float
        Hopping amplitude between the impurity d and the first chain site f_0:
            t_0 = sqrt(sum_{n=1}^{N} : |v_n|^2).
    eps_chain : np.ndarray, shape (N_chain,)
        On-site energies along the 1D chain (k = 0, ..., N_chain - 1).
        For a particle-hole symmetric bath, eps_chain is identically zero.
    t_hop : np.ndarray, shape (N_chain - 1,)
        Nearest-neighbor hopping amplitudes t_{k+1} between site k and site k + 1
        along the 1D chain (k = 0, ..., N_chain - 2).
    U_star_to_chain : Optional[np.ndarray], shape (N_star, N_chain)
        Transformation matrix U_{n, k} relating the Star
        operators c_n to the Chain operators f_k:
            f_k^+ = sum_{n=1}^{N_star} : U_{n, k} c_n^+.
    """
    t0: float
    eps_chain: np.ndarray
    t_hop: np.ndarray
    U_star_to_chain: Optional[np.ndarray] = None

    @property
    def n_sites(self) -> int:
        """Number of sites along the 1D tight-binding chain."""
        return len(self.eps_chain)

    def tridiagonal_matrix(self) -> np.ndarray:
        """Return the dense N_chain x N_chain symmetric tridiagonal bath matrix H_chain."""
        H_ch = np.diag(self.eps_chain.astype(float))
        if len(self.t_hop) > 0:
            idx = np.arange(len(self.t_hop))
            H_ch[idx, idx + 1] = self.t_hop
            H_ch[idx + 1, idx] = self.t_hop
        return H_ch


def star_to_chain(
    epsilons: np.ndarray,
    v: np.ndarray,
    max_chain_len: Optional[int] = None,
    reorthogonalize: bool = True,
    drop_tol: float = 1e-14,
) -> ChainGeometry:
    """
    Map Star-Geometry bath parameters (epsilons_n, v_n) -> 1D Chain-Geometry
    parameters (t0, eps_chain_k, t_hop_k) via Lanczos tridiagonalization
    (RL-NESS Sec. II C, Eq. 12).

    Mathematical Procedure
    ----------------------
    1. The impurity couples only to the normalized linear combination of bath modes:
           |f_0> = (1 / t_0) sum_{n=1}^{N} : v_n |c_n>,   with   t_0 = sqrt(sum_n : |v_n|^2).
    2. Starting from |f_0>, we generate the Krylov subspace of H_bath = diag(epsilons)
       by iteratively applying H_bath and orthogonalizing against previous chain sites:
           eps_k = <f_k | H_bath | f_k>
           |r_{k+1}> = H_bath |f_k> - eps_k |f_k> - t_k |f_{k-1}>
           t_{k+1} = || |r_{k+1}> ||_2
           |f_{k+1}> = |r_{k+1}> / t_{k+1}
    3. Full Gram-Schmidt re-orthogonalization (`reorthogonalize=True`) is applied at
       every step to prevent floating-point loss of orthogonality on large grids.

    Parameters
    ----------
    epsilons : np.ndarray, shape (N,)
        Star-geometry bath mode energies epsilon_n.
    v : np.ndarray, shape (N,)
        Star-geometry impurity-bath hopping amplitudes v_n.
    max_chain_len : Optional[int]
        If provided, truncates the Krylov chain at `max_chain_len` sites (<= N).
    reorthogonalize : bool
        If True (default), performs full two-pass Gram-Schmidt re-orthogonalization
        at each Lanczos step to guarantee machine-precision unitarity.
    drop_tol : float
        Coupling threshold below which decoupled modes (|v_n| <= drop_tol) are pruned
        before tridiagonalization (RL-NESS Sec. II C).

    Returns
    -------
    ChainGeometry
        Dataclass containing (t0, eps_chain, t_hop, U_star_to_chain).
    """
    eps_arr = np.asarray(epsilons, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    if eps_arr.shape != v_arr.shape or eps_arr.ndim != 1:
        raise ValueError("`epsilons` and `v` must be 1D arrays of identical length.")

    # Prune modes with zero coupling to the impurity (e.g. outside the Fermi step)
    active_mask = np.abs(v_arr) > drop_tol
    n_total = len(eps_arr)
    n_active = int(np.sum(active_mask))

    if n_active == 0:
        return ChainGeometry(
            t0=0.0,
            eps_chain=np.zeros(0, dtype=float),
            t_hop=np.zeros(0, dtype=float),
            U_star_to_chain=np.zeros((n_total, 0), dtype=float),
        )

    eps_act = eps_arr[active_mask]
    v_act = v_arr[active_mask]

    n_chain = n_active if max_chain_len is None else min(n_active, int(max_chain_len))
    t0 = float(np.linalg.norm(v_act))

    # Basis vectors U_act[:, k] represent |f_k> in the active Star basis
    U_act = np.zeros((n_active, n_chain), dtype=float)
    eps_chain = np.zeros(n_chain, dtype=float)
    t_hop = np.zeros(max(0, n_chain - 1), dtype=float)

    U_act[:, 0] = v_act / t0

    for k in range(n_chain):
        fk = U_act[:, k]
        h_fk = eps_act * fk
        ak = float(np.dot(fk, h_fk))
        eps_chain[k] = ak

        if k == n_chain - 1:
            break

        rk = h_fk - ak * fk
        if k > 0:
            rk = rk - t_hop[k - 1] * U_act[:, k - 1]

        if reorthogonalize:
            # Two-pass Gram-Schmidt re-orthogonalization against all previous chain states
            for _ in range(2):
                overlaps = U_act[:, : k + 1].T @ rk
                rk = rk - U_act[:, : k + 1] @ overlaps

        bk = float(np.linalg.norm(rk))
        if bk <= drop_tol:
            # Krylov subspace exhausted early (e.g. degenerate Star energies)
            eps_chain = eps_chain[: k + 1]
            t_hop = t_hop[:k]
            U_act = U_act[:, : k + 1]
            break

        t_hop[k] = bk
        U_act[:, k + 1] = rk / bk

    # Embed active unitary matrix back into full N_total Star index space
    U_full = np.zeros((n_total, U_act.shape[1]), dtype=float)
    U_full[active_mask, :] = U_act

    return ChainGeometry(
        t0=t0,
        eps_chain=eps_chain,
        t_hop=t_hop,
        U_star_to_chain=U_full,
    )


def chain_to_star(
    t0: float,
    eps_chain: np.ndarray,
    t_hop: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inverse mapping: 1D Chain-Geometry parameters (t0, eps_chain_k, t_hop_k)
    -> Star-Geometry parameters (epsilons_n, v_n) via tridiagonal eigendecomposition.

    Mathematical Procedure
    ----------------------
    Given the N x N symmetric tridiagonal chain Hamiltonian H_chain with diagonal
    `eps_chain` and off-diagonal `t_hop`, we diagonalize:
        H_chain = W diag(epsilons_1, ..., epsilons_N) W^T
    Because only the first chain site f_0 couples to the impurity with hopping t_0:
        H_coupling = t_0 d^+ f_0 + h.c. = sum_{n=1}^{N} : (t_0 W_{0, n}) d^+ c_n + h.c.
    the Star-Geometry mode energies are the eigenvalues `epsilons_n` and the
    Star-Geometry couplings are `v_n = t_0 * W_{0, n}`.

    Returns
    -------
    epsilons : np.ndarray, shape (N,)
        Recovered Star-geometry bath mode energies epsilon_n (sorted ascending).
    v : np.ndarray, shape (N,)
        Recovered Star-geometry impurity-bath couplings v_n >= 0.
    W_chain_to_star : np.ndarray, shape (N, N)
        Orthogonal eigenvector matrix W_{k, n} such that f_k = sum_n W_{k, n} c_n.
    """
    d_diag = np.asarray(eps_chain, dtype=float)
    e_off = np.asarray(t_hop, dtype=float)
    if len(d_diag) == 0:
        return np.zeros(0), np.zeros(0), np.zeros((0, 0))

    epsilons, W = linalg.eigh_tridiagonal(d_diag, e_off)
    # Fix gauge of eigenvectors so v_n = t_0 W_{0, n} >= 0
    signs = np.where(W[0, :] >= 0.0, 1.0, -1.0)
    W = W * signs[None, :]
    v = float(t0) * W[0, :]
    return epsilons, v, W


def bath_to_lddl_particle_hole_chains(
    epsilons: np.ndarray,
    v: np.ndarray,
    fermi_occ: np.ndarray,
    max_chain_len: Optional[int] = None,
    drop_tol: float = 1e-14,
) -> Dict[str, ChainGeometry]:
    """
    Perform the RL-NESS Sec. II C (Eqs. 13-14, Fig. 2c-d) / LDDL Sec. IV (Eqs. 64-67)
    level-doubling and tridiagonalization into separate Particle and Hole 1D chains.

    Why This Is Used in Matan's RL-NESS Code
    ----------------------------------------
    In the Star Geometry, the Lindblad rates lambda_n(1) = gamma(1 - f(epsilon_n))
    and lambda_n^(2) = gamma f(epsilon_n) depend on the mode index n through the
    Fermi function f(epsilon_n). Directly tridiagonalizing the bath would mix modes
    with different Fermi occupations and make the Lindblad dissipators non-local
    along the 1D chain (LDDL Eq. 63).

    To keep BOTH the Hamiltonian AND the Lindblad driving strictly local on the chain,
    each physical Star mode (eps_n, v_n) is split into:
      1. A "particle" (filled) mode at energy eps_n with coupling:
             v_{p, n} = v_n * sqrt(f(epsilon_n))
         driven purely by creation operators (D_p) toward occupation 1.
      2. A "hole" (empty) mode at energy eps_n with coupling:
             v_{h, n} = v_n * sqrt(1 - f(epsilon_n))
         driven purely by annihilation operators (D_h) toward occupation 0.
    Each set of modes is then tridiagonalized independently into its own 1D Wilson chain

    Returns
    -------
    dict with keys:
      - "particle_chain": ChainGeometry for the filled particle lead (eta = p).
      - "hole_chain":     ChainGeometry for the empty hole lead (eta = h).
    """
    f_arr = np.clip(np.asarray(fermi_occ, dtype=float), 0.0, 1.0)
    v_arr = np.asarray(v, dtype=float)
    eps_arr = np.asarray(epsilons, dtype=float)

    v_particle = v_arr * np.sqrt(f_arr)
    v_hole = v_arr * np.sqrt(1.0 - f_arr)

    chain_p = star_to_chain(eps_arr, v_particle, max_chain_len=max_chain_len, drop_tol=drop_tol)
    chain_h = star_to_chain(eps_arr, v_hole, max_chain_len=max_chain_len, drop_tol=drop_tol)

    return {
        "particle_chain": chain_p,
        "hole_chain": chain_h,
    }


def chain_retarded_hybridization(
    chain: ChainGeometry,
    omega: np.ndarray,
    gamma_uniform: float,
) -> np.ndarray:
    """
    Evaluate the bare retarded bath hybridization Delta(w) directly
    from the 1D Chain-Geometry tridiagonal matrix H_chain with uniform Lindblad
    damping `gamma_uniform`:
        Delta(w) = t_0^2 * [ (w + i gamma_uniform) I - H_chain ]^{-1}_{0, 0}
    Used to verify that the 1D Chain Geometry and the Star Geometry produce the
    exact same hybridization function at every frequency w.
    """
    w_arr = np.asarray(omega, dtype=float)
    if chain.n_sites == 0 or chain.t0 == 0.0:
        return np.zeros_like(w_arr, dtype=complex)

    # Diagonalize H_chain once: H_chain = W diag(E) W^T -> G_{00}(\omega) = sum_n |W_{0,n}|^2 / (\omega - E_n + i\gamma)
    eigvals, W = linalg.eigh_tridiagonal(chain.eps_chain, chain.t_hop)
    weights_0 = (chain.t0 ** 2) * (W[0, :] ** 2)
    denom = w_arr[:, None] - eigvals[None, :] + 1j * gamma_uniform
    return np.sum(weights_0[None, :] / denom, axis=1)


if __name__ == "__main__":
    # Self-test of the t <-> v correspondence module on a power-law bath
    from bath import Bath

    print("Running t <-> v Correspondence Module Self-Test...")
    test_bath = Bath(
        name="PG",
        Gamma0=0.25,
        r=0.3,
        T=0.01,
        D=1.0,
        discretization="log_linear",
        target_n_modes=200,
        gamma_rule="uniform",
        c_gamma=1.0,
    )

    # 1. Forward map: Star (eps_n, v_n) -> Chain (t_0, eps_chain_k, t_hop_k)
    chain = star_to_chain(test_bath.epsilons, test_bath.v)

    # 2. Inverse map: Chain (t_0, eps_chain_k, t_hop_k) -> Star (eps_rec, v_rec)
    eps_rec, v_rec, _ = chain_to_star(chain.t0, chain.eps_chain, chain.t_hop)

    # Sort original Star parameters by ascending energy for direct comparison
    order = np.argsort(test_bath.epsilons)
    eps_orig = test_bath.epsilons[order]
    v_orig = test_bath.v[order]

    err_eps = np.max(np.abs(eps_rec - eps_orig))
    err_v = np.max(np.abs(v_rec - v_orig))
    max_onsite = np.max(np.abs(chain.eps_chain))

    # 3. Verify that the 1D Chain Green's function matches the Star Green's function
    w_test = np.linspace(-0.8, 0.8, 200)
    gamma_u = float(test_bath.gammas[0])
    delta_chain = chain_retarded_hybridization(chain, w_test, gamma_uniform=gamma_u)
    delta_star = np.sum(
        (test_bath.v[None, :] ** 2) / (w_test[:, None] - test_bath.epsilons[None, :] + 1j * gamma_u),
        axis=1,
    )
    err_gf = np.max(np.abs(delta_chain - delta_star))

    # 4. Test LDDL particle-hole chain splitting at finite temperature T = 0.01
    ph_chains = bath_to_lddl_particle_hole_chains(
        test_bath.epsilons, test_bath.v, test_bath.fermi(test_bath.epsilons)
    )
    t0_tot_sq = ph_chains["particle_chain"].t0 ** 2 + ph_chains["hole_chain"].t0 ** 2

    print(f"  Star -> Chain -> Star max |eps_rec - eps_orig| : {err_eps:.3e}")
    print(f"  Star -> Chain -> Star max |v_rec - v_orig|     : {err_v:.3e}")
    print(f"  Max |eps_chain_k| (PH symmetry check -> 0)     : {max_onsite:.3e}")
    print(f"  Max |Delta^R_chain(w) - Delta^R_star(w)|       : {err_gf:.3e}")
    print(f"  LDDL Particle+Hole t0^2 sum check              : |t0_p^2 + t0_h^2 - t0^2| = {abs(t0_tot_sq - chain.t0**2):.3e}")