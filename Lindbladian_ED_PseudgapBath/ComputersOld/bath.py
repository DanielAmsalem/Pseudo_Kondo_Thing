"""
bath.py
=======
Constructs a discretized tight-binding fermionic bath coupled to a quantum impurity
and driven by local Markovian Lindblad reservoirs, following:
  - [LDDL]    Schwarz, Goldstein, Dorda, Arrigoni, Weichselbaum, and von Delft,
              Phys. Rev. B 94, 155142 (2016).
  - [RL-NESS] Lotem, Weichselbaum, von Delft, and Goldstein,
              Phys. Rev. Research 2, 043052 (2020).

WHAT THIS MODULE DOES PHYSICALLY
--------------------------------
In an analytical calculation, a lead is treated as a continuous bath of fermionic
modes characterized by a continuous hybridization function (spectral density):
    Gamma(omega) = pi sum_n : |v_n|^2 * delta(omega - epsilon_n) = Gamma_0 * |omega|^r   for |omega| <= D.

We represent this continuous bath using a FINITE number (N_modes)
of discrete tight-binding modes in the "Star Geometry" (RL-NESS Eqs. 8-9):
    H_bath + H_coupling = sum_{n=1}^{N_modes} : epsilon_n c_n^+ c_n
                        + sum_{n=1}^{N_modes} : v_n (d^+ c_n + c_n^+ d)

`Bath` performs the two-stage discretization procedure from RL-NESS Appendix A:
  Stage 1 (Grid Generation):
      Slices the continuous energy band [-D, +D] into N_modes non-overlapping
      energy intervals (bins) I_n = [a_n, b_n] of width \delta_n = b_n - a_n.
  Stage 2 (Moment Integration & Lindblad Broadening):
      Replaces each energy interval I_n = [a_n, b_n] by a SINGLE representative
      tight-binding mode c_n whose hopping amplitude v_n and on-site energy epsilon_n
      preserve the exact zeroth and first moments of Gamma(omega) across that bin:
          |v_n|^2 = 1\pi * int_{a_n}^{b_n} : Gamma(w) dw
          epsilon_n = int_{a_n}^{b_n} : w * Gamma(w) dw
                    / int_{a_n}^{b_n} : Gamma(w) dw
      Finally, to make the finite set of discrete levels behave as a true open thermal
      reservoir at temperature T and chemical potential mu, each mode c_n is coupled
      to a local Lindblad dissipator with damping/broadening rate gamma_n,(LDDL Eqs. 39-45)
      which broadens each discrete pole into a Lorentzian of width gamma_n.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional, Tuple
import numpy as np
from scipy import optimize

DiscretizationType = Literal["log_linear", "linear", "log"]
GammaRuleType = Literal["adaptive", "capped_adaptive", "uniform"]

# Precomputed 16-point Gauss-Legendre nodes and weights mapped from [-1, 1] -> [0, 1].
# Used to integrate custom or IR-regularized hybridization functions \Gamma(\omega)
# over each energy bin [a_n, b_n] in vectorized NumPy without calling Fortran QUADPACK.
_GL_NODES_01, _GL_WEIGHTS_01 = np.polynomial.legendre.leggauss(16)
_GL_NODES_01 = 0.5 * (_GL_NODES_01 + 1.0)
_GL_WEIGHTS_01 = 0.5 * _GL_WEIGHTS_01


@dataclass
class Bath:
    """
    A single discretized fermionic tight-binding bath coupled to the impurity
    and driven by local Lindblad reservoirs.
    r : float
        Power-law exponent (-1 < r < 1):
          - r > 0 : Pseudogap bath (density of states vanishes at omega = 0).
          - r = 0 : Flat metallic lead (Gamma(w) = Gamma_0 = const).
          - r < 0 : Divergent power-law DOS
    T : float
        Bath temperature T_\alpha (in units where k_B = 1). Enters the Fermi-Dirac
        distribution f_alpha(epsilon_n) that sets the Lindblad filling/emptying rates.
    mu : float
        Bath chemical potential mu_alpha (Fermi energy of this lead, default 0.0).
    D : float
        Ultraviolet (UV) half-bandwidth cutoff D_alpha. Tight-binding modes are generated
        strictly inside the energy interval [-D, +D], so Gamma(omega) = 0 for |omega| > D.
    omega_ir : float
        Optional infrared (IR) regularization cutoff w_ir >= 0 (default 0.0).
        If w_ir > 0, replaces |w|^r with (w^2 + w_ir^2)^{r/2},
    discretization : {"log_linear", "linear", "log"}
        Scheme used to partition the energy band [-D, +D] into discrete intervals:
          - "linear":     Uniform bin width delta across the entire band [-D, +D].
          - "log":        Wilson NRG logarithmic grid with bin edges at +- D * Lambda^{-n}.
          - "log_linear": Hybrid grid (RL-NESS Appendix A, Eq. A1) with uniform spacing
                          delta inside the low-energy window [-D^*, +D^*] and smooth
                          exponentially growing bins ~Lambda^n up to +- D.
    target_n_modes : Optional[int]
        Total number of discrete tight-binding modes N_modes desired across [-D, +D].
        When set (e.g., target_n_modes=10000), the code automatically solves for the
        exact spacing parameters (delta, Lambda, N_log) to produce `target_n_modes` modes.
    D_star : Optional[float]
        Half-width D* of the uniform linear discretization window [-D*, +D*] in the
        "log_linear" scheme (denoted D* in RL-NESS Fig. 2b' and E^* in LDDL Fig. 9).
        Should be chosen several times larger than T
    delta : float
        Uniform energy bin width delta inside the linear window [-D*, +D*] (for
        "log_linear") or across the full band [-D, +D] (for "linear").
    Lambda : float
        Logarithmic discretization parameter Lambda > 1 (Wilson's NRG parameter).
        Controls how rapidly bin widths expand (delta_n ~ Lambda^n) at high
        energies |omega| > D*. Values closer to 1.0 give finer high-energy grids.
    N_log : int
        Number of logarithmic energy shells on each side of omega = 0 when using pure
        "log" discretization (the minimum resolved energy is D Lambda^{-N_log}).
    z_shift : float
        NRG z-shift offset parameter z in [0, 1) (Oliveira & Oliveira, PRB 1994;
        RL-NESS Appendix A, Eq. A1). Shifts the integer interval index n -> n + z
        when placing the logarithmic bin boundaries I_n = [f(n + z), f(n + 1 + z)].
        In coarse logarithmic grids (Lambda ~ 2-6), sliding z in [0, 1) shifts the
        comb of discrete energies across the band so that averaging over several z values
        ("z-averaging") cancels out logarithmic discretization oscillations. For fine
        grids (N_modes ~ 10,000, Lambda ~ 1.002), z_shift = 0.0 is already smooth.
    gamma_rule : {"adaptive", "capped_adaptive", "uniform"}
        Rule for assigning the Lindblad damping/broadening rate gamma_n to mode n:
          - "adaptive":        gamma_n = c_gamma * delta_n (each mode's Lorentzian width
                               scales with its own bin width delta_n; LDDL Eq. 45).
          - "capped_adaptive": gamma_n = min(c_gamma * delta_n, gamma_max). Prevents
                               wide high-energy logarithmic bins near |epsilon_n| ~ D
                               from having huge Lorentzian widths that leak 1/w^2
                               tails back into the low-energy thermal window |w| ~ T
                               (see LDDL Appendix A).
          - "uniform":         gamma_n = c_gamma * delta (constant damping rate across
                               all modes; required when mapping to a local 1D Wilson chain
                               in RL-NESS Sec. II C / LDDL Sec. IV).
    c_gamma : float
        Dimensionless ratio gamma_n / delta_n (~1.0 to 1.5; LDDL Fig. 2 & Eq. 45).
    gamma_max : Optional[float]
        Maximum allowed Lindblad broadening rate when `gamma_rule == "capped_adaptive"`.
    custom_shape : Optional[Callable[[np.ndarray], np.ndarray]]
        Optional dimensionless frequency profile g_alpha(w) overriding |w|^r:
            Gamma_alpha(w) = Gamma_0 * g_alpha(w)   for |w| <= D.
        For example, a Lorentzian probe bath of cutoff width D_S uses:
            custom_shape = lambda w: 1.0 / (1.0 + (w / D_S)**2).

    Attributes Computed Automatically on Initialization
    ---------------------------------------------------
    bins : np.ndarray, shape (N_modes, 2)
        The [a_n, b_n] energy interval boundaries for each discrete mode n.
    bin_widths : np.ndarray, shape (N_modes,)
        Width delta_n = b_n - a_n of each energy bin.
    epsilons : np.ndarray, shape (N_modes,)
        Tight-binding on-site energies epsilon_n (in ascending order from -D to +D).
    v : np.ndarray, shape (N_modes,)
        Tight-binding hopping amplitudes v_n coupling mode c_n to the impurity d.
    gammas : np.ndarray, shape (N_modes,)
        Lindblad damping/broadening rates gamma_n for each mode c_n.
    """
    name: str = "PG"
    Gamma0: float = 0.25 # Hybridization prefactor
    r: float = 0.0
    T: float = 0.002
    mu: float = 0.0
    D: float = 1.0
    omega_ir: float = 0.0
    discretization: DiscretizationType = "log_linear"
    target_n_modes: Optional[int] = None
    D_star: Optional[float] = None
    delta: float = 0.0005
    Lambda: float = 1.01
    N_log: int = 100
    z_shift: float = 0.0
    gamma_rule: GammaRuleType = "capped_adaptive"
    c_gamma: float = 1.2
    gamma_max: Optional[float] = None
    custom_shape: Optional[Callable[[np.ndarray], np.ndarray]] = None

    bins: np.ndarray = field(init=False, repr=False)
    bin_widths: np.ndarray = field(init=False, repr=False)
    epsilons: np.ndarray = field(init=False, repr=False)
    v: np.ndarray = field(init=False, repr=False)
    gammas: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.r <= -1.0 and self.omega_ir <= 0.0 and self.custom_shape is None:
            raise ValueError(f"Exponent r={self.r} <= -1 requires omega_ir > 0 for integrability.")

        # Default linear window half-width D* covers at least 8T around w = 0
        if self.D_star is None:
            self.D_star = min(0.4 * self.D, max(8.0 * self.T, 0.015 * self.D))

        # If a specific total mode count (e.g., 10,000) was requested, solve for (delta, Lambda)
        if self.target_n_modes is not None:
            self._tune_grid_for_target_modes(self.target_n_modes)

        # Build the tight-binding energy bins, mode energies epsilon_n, hoppings v_n, and rates gamma_n
        self.build_grid()

    def _tune_grid_for_target_modes(self, target_n: int) -> None:
        """
        Automatically calibrate (delta, Lambda, N_log) so the symmetric bath has
        `target_n` total tight-binding modes (`target_n // 2` on [-D, 0] and `target_n // 2` on [0, +D]).
        """
        n_half = max(10, target_n // 2)
        if self.discretization == "linear":
            self.delta = self.D / n_half
            return

        if self.discretization == "log":
            self.N_log = n_half - 1
            return

        if self.discretization == "log_linear":
            # Place 40% of the positive modes uniformly inside [0, D^*] with spacing \delta,
            # and 60% of the positive modes in the sinh-logarithmic region [D^*, D].
            n_lin = max(5, int(0.4 * n_half))
            n_log = max(5, n_half - n_lin)
            self.delta = self.D_star / n_lin
            span = self.D - self.D_star
            if span <= 0.0 or n_log * self.delta >= span:
                self.discretization = "linear"
                self.delta = self.D / n_half
                return

            # In RL-NESS Eq. (A1), the n_log-th boundary above D^* sits at:
            #     f(x* + n_log) = (delta / ln Lambda) * sinh(n_log * ln Lambda) + D* = D.
            # We solve this equation for u = ln Lambda > 0 using Brent's root-finding method.
            def residual(u: float) -> float:
                if u <= 1e-12:
                    return self.delta * n_log - span
                arg = min(700.0, n_log * u)
                return (self.delta / u) * np.sinh(arg) - span

            u_max = 600.0 / n_log
            u_sol = optimize.brentq(residual, 1e-10, u_max)
            self.Lambda = float(np.exp(u_sol))

    def continuum_gamma(self, omega: np.ndarray | float) -> np.ndarray:
        """
        Evaluate the target continuum hybridization function Gamma_a(w)
        at frequency `omega`.
        """
        w = np.asarray(omega, dtype=float)
        if self.custom_shape is not None:
            return self.Gamma0 * np.asarray(self.custom_shape(w), dtype=float)
        if self.omega_ir > 0.0:
            return self.Gamma0 * (w ** 2 + self.omega_ir ** 2) ** (0.5 * self.r)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = self.Gamma0 * np.power(np.abs(w), self.r)
        return np.where(np.abs(w) == 0.0, 0.0 if self.r > 0 else np.inf, val)

    def fermi(self, omega: np.ndarray | float, T_override: Optional[float] = None) -> np.ndarray:
        """
        Fermi-Dirac distribution f_a(w) = 1 / (exp((w - mu_a)/T) + 1)
        characterizing the thermal occupation of this bath.
        """
        T_val = self.T if T_override is None else T_override
        w = np.asarray(omega, dtype=float)
        if T_val <= 0.0:
            return np.where(w < self.mu, 1.0, np.where(w > self.mu, 0.0, 0.5))
        x = np.clip((w - self.mu) / T_val, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(x))

    def dfermi_dT(self, omega: np.ndarray | float, T_override: Optional[float] = None) -> np.ndarray:
        """
        Exact temperature derivative of the Fermi-Dirac distribution
        Used when differentiating the Lyapunov equation with respect to bath-temperature T_a.
        """
        T_val = self.T if T_override is None else T_override
        if T_val <= 0.0:
            return np.zeros_like(np.asarray(omega, dtype=float))
        w = np.asarray(omega, dtype=float)
        x = (w - self.mu) / T_val
        x_clip = np.clip(x, -500.0, 500.0)
        f_val = 1.0 / (1.0 + np.exp(x_clip))
        return (x / T_val) * f_val * (1.0 - f_val)

    def _generate_positive_edges(self) -> np.ndarray:
        """
        Stage 1 of Discretization:
        Generate the sorted array of bin boundaries [0 = e_0 < e_1 < ... < e_{N/2} = D]
        partitioning the positive half-band [0, +D] into N/2 intervals I_n = [e_{n-1}, e_n].
        """
        if self.discretization == "linear":
            # Uniformly spaced intervals of width \delta from 0 to D
            n_half = max(1, int(np.round(self.D / self.delta)))
            return np.linspace(0.0, self.D, n_half + 1)

        if self.discretization == "log":
            # Wilson NRG logarithmic grid: boundaries at 0, D \Lambda^{-(N_log - z)}, ..., D \Lambda^{-(1 - z)}, D
            if self.Lambda <= 1.0:
                raise ValueError("Logarithmic discretization requires Lambda > 1.")
            exponents = np.arange(self.N_log, -1, -1, dtype=float) - self.z_shift
            exponents = np.maximum(exponents, 0.0)
            edges = self.D * (self.Lambda ** (-exponents))
            return np.unique(np.concatenate(([0.0], edges, [self.D])))

        if self.discretization == "log_linear":
            # RL-NESS Appendix A, Eq. (A1) smooth hybrid linear-to-logarithmic grid:
            #   f(x) = delta * x                                               for x <= x* = D* / delta
            #   f(x) = (delta / ln Lambda) * sinh[(x - x*) ln Lambda] + D*  for x >  x* = D* / delta
            # Evaluated at x = n + z_shift for integer n = 1, 2, ... up to D.
            if self.Lambda <= 1.0:
                raise ValueError("log_linear discretization requires Lambda > 1.")
            x_star = self.D_star / self.delta
            log_lam = np.log(self.Lambda)

            def f_edge(x: float) -> float:
                if x <= x_star:
                    return self.delta * x
                return (self.delta / log_lam) * np.sinh((x - x_star) * log_lam) + self.D_star

            edges_list = [0.0]
            n = 1
            while True:
                x_val = n + self.z_shift
                val = f_edge(x_val)
                if val >= self.D - 1e-12:
                    edges_list.append(self.D)
                    break
                edges_list.append(val)
                n += 1
                if n > 500000:
                    raise RuntimeError("Grid generation exceeded 500,000 intervals.")
            return np.unique(np.array(edges_list, dtype=float))

        raise ValueError(f"Unknown discretization type: {self.discretization}")

    def build_grid(self) -> None:
        """
        Stage 2 of Discretization:
        For each positive energy bin I_n = [a_n, b_n] in [0, D], compute:
          1. Total coupling weight W_n = int_{a_n}^{b_n} : Gamma(w) dw -> v_n = sqrt{W_n / pi}.
          2. First energy moment  M_n = int_{a_n}^{b_n} : w * Gamma(w) dw -> epsilon_n = M_n / W_n.
        Then mirror the positive bins onto [-D, 0] to guarantee exact particle-hole symmetry
        (epsilon_{-n} = -epsilon_n, v_{-n} = v_n), and assign the Lindblad rates gamma_n.
        """
        pos_edges = self._generate_positive_edges()
        a = pos_edges[:-1]
        b = pos_edges[1:]
        pos_widths = b - a

        if self.custom_shape is None and self.omega_ir <= 0.0:
            # Exact analytical antiderivatives for a pure power law \Gamma(\omega) = \Gamma_0 \omega^r on [a_n, b_n]:
            #   W_n = \Gamma_0 (b_n^{r+1} - a_n^{r+1}) / (r + 1)
            #   M_n = \Gamma_0 (b_n^{r+2} - a_n^{r+2}) / (r + 2)
            w_int = self.Gamma0 * (b ** (self.r + 1.0) - a ** (self.r + 1.0)) / (self.r + 1.0)
            m_int = self.Gamma0 * (b ** (self.r + 2.0) - a ** (self.r + 2.0)) / (self.r + 2.0)
        else:
            # Vectorized 16-point Gauss-Legendre quadrature across all bins [a_n, b_n] simultaneously
            pts = a[:, None] + pos_widths[:, None] * _GL_NODES_01[None, :]
            gam_pts = self.continuum_gamma(pts)
            w_int = pos_widths * (gam_pts @ _GL_WEIGHTS_01)
            m_int = pos_widths * ((pts * gam_pts) @ _GL_WEIGHTS_01)

        pos_v2 = w_int / np.pi
        pos_eps = np.where(w_int > 0.0, m_int / w_int, 0.5 * (a + b))

        # Mirror positive bins onto negative energies [-b_n, -a_n] so modes are ordered from -D to +D
        neg_bins = np.column_stack((-pos_edges[1:][::-1], -pos_edges[:-1][::-1]))
        pos_bins = np.column_stack((pos_edges[:-1], pos_edges[1:]))
        self.bins = np.vstack((neg_bins, pos_bins))
        self.bin_widths = np.concatenate((pos_widths[::-1], pos_widths))
        self.epsilons = np.concatenate((-pos_eps[::-1], pos_eps))
        self.v = np.sqrt(np.concatenate((pos_v2[::-1], pos_v2)))

        # Assign Lindblad damping/broadening rates \gamma_n for each mode
        if self.gamma_rule == "adaptive":
            self.gammas = self.c_gamma * self.bin_widths
        elif self.gamma_rule == "capped_adaptive":
            g_max = self.gamma_max if self.gamma_max is not None else max(0.5 * self.T, 10.0 * self.delta)
            self.gammas = np.minimum(self.c_gamma * self.bin_widths, g_max)
        elif self.gamma_rule == "uniform":
            self.gammas = np.full_like(self.epsilons, self.c_gamma * self.delta)
        else:
            raise ValueError(f"Unknown gamma_rule: {self.gamma_rule}")

    @property
    def n_modes(self) -> int:
        """Total number of discrete tight-binding modes in this bath."""
        return len(self.epsilons)

    @property
    def gamma_min(self) -> float:
        """Smallest Lindblad broadening rate gamma_min (sets the IR resolution scale near w = 0)."""
        return float(np.min(self.gammas))

    def lindblad_rates(self, T_override: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the diagonal Lindblad rate vectors (lambda(1), lambda(2)) for all modes
        in this bath (LDDL Eq. 39 / RL-NESS Eq. 10):
          - lambda_n(1) = gamma_n [1 - f_a(eps_n)]  (emptying / depletion rate)
          - lambda_n(2) = gamma_n f_a(eps_n)        (filling / injection rate)
        Note : lambda_n(1) + lambda_n(2) = gamma_n is independent of T and mu.
        """
        f_q = self.fermi(self.epsilons, T_override=T_override)
        lam1 = self.gammas * (1.0 - f_q)
        lam2 = self.gammas * f_q
        return lam1, lam2

    # =========================================================================
    # Convenience Bridges to the 1D Tight-Binding Chain Geometry (t <-> v)
    # =========================================================================

    def to_chain(self, max_chain_len: Optional[int] = None, reorthogonalize: bool = True):
        """
        Tridiagonalize this bath's Star-Geometry parameters (epsilon_n, v_n) into
        1D Chain-Geometry nearest-neighbor parameters (t_0, epsilon_k, t_k)
        using `tv_correspondence.star_to_chain` (RL-NESS Eq. 12).
        """
        from tv_correspondence import star_to_chain
        return star_to_chain(
            self.epsilons,
            self.v,
            max_chain_len=max_chain_len,
            reorthogonalize=reorthogonalize,
        )

    def to_lddl_particle_hole_chains(
            self,
            T_override: Optional[float] = None,
            max_chain_len: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Map this thermal bath into separate filled ("particle") and empty ("hole")
        1D nearest-neighbor tight-binding chains via RL-NESS Sec. II C (Eqs. 13-14).
        """
        from tv_correspondence import bath_to_lddl_particle_hole_chains
        f_occ = self.fermi(self.epsilons, T_override=T_override)
        return bath_to_lddl_particle_hole_chains(
            self.epsilons,
            self.v,
            f_occ,
            max_chain_len=max_chain_len,
        )
