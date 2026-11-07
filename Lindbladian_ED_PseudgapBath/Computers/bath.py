"""
bath.py
-------
General parameterized Lindblad-Driven Discretized Lead (LDDL / RL-NESS) tight-binding
bath supporting power-law and custom hybridization functions, explicit UV/IR cutoffs,
and linear, logarithmic, or smooth hybrid log-linear discretization grids with
automatic mode-count targeting (e.g., N_modes = 10,000).

All bin integrals use vectorized NumPy Gauss-Legendre quadrature (no Fortran quadpack).
"""

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple
import numpy as np
from scipy import optimize


DiscretizationType = Literal["log_linear", "linear", "log"]
GammaRuleType = Literal["adaptive", "capped_adaptive", "uniform"]

# Precomputed 16-point Gauss-Legendre nodes and weights on [0, 1] for fast, exact bin integration
_GL_NODES_01, _GL_WEIGHTS_01 = np.polynomial.legendre.leggauss(16)
_GL_NODES_01 = 0.5 * (_GL_NODES_01 + 1.0)
_GL_WEIGHTS_01 = 0.5 * _GL_WEIGHTS_01


@dataclass
class Bath:
    """
    Represents a single discretized fermionic tight-binding bath coupled to the
    impurity and driven by local Lindblad reservoirs (LDDL / RL-NESS framework).
    """
    name: str = "PG"
    Gamma0: float = 0.25
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
        if self.D_star is None:
            self.D_star = min(0.4 * self.D, max(8.0 * self.T, 0.015 * self.D))
        if self.target_n_modes is not None:
            self._tune_grid_for_target_modes(self.target_n_modes)
        self.build_grid()

    def _tune_grid_for_target_modes(self, target_n: int) -> None:
        """
        Automatically calibrate (delta, Lambda, N_log) so the symmetric tight-binding
        bath has `target_n` total modes (target_n // 2 positive and target_n // 2 negative).
        """
        n_half = max(10, target_n // 2)
        if self.discretization == "linear":
            self.delta = self.D / n_half
            return

        if self.discretization == "log":
            self.N_log = n_half - 1
            return

        if self.discretization == "log_linear":
            n_lin = max(5, int(0.4 * n_half))
            n_log = max(5, n_half - n_lin)
            self.delta = self.D_star / n_lin
            span = self.D - self.D_star
            if span <= 0.0 or n_log * self.delta >= span:
                self.discretization = "linear"
                self.delta = self.D / n_half
                return

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
        Evaluate ideal continuum hybridization \Gamma_\alpha(\omega) = \Gamma_0 |\omega|^r.
        """
        w = np.asarray(omega, dtype=float)
        if self.custom_shape is not None:
            return self.Gamma0 * np.asarray(self.custom_shape(w), dtype=float)
        if self.omega_ir > 0.0:
            return self.Gamma0 * (w**2 + self.omega_ir**2) ** (0.5 * self.r)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = self.Gamma0 * np.power(np.abs(w), self.r)
        return np.where(np.abs(w) == 0.0, 0.0 if self.r > 0 else np.inf, val)

    def fermi(self, omega: np.ndarray | float, T_override: Optional[float] = None) -> np.ndarray:
        """Fermi-Dirac distribution f_\alpha(\omega) = 1 / (exp((\omega - \mu)/T) + 1)."""
        T_val = self.T if T_override is None else T_override
        w = np.asarray(omega, dtype=float)
        if T_val <= 0.0:
            return np.where(w < self.mu, 1.0, np.where(w > self.mu, 0.0, 0.5))
        x = np.clip((w - self.mu) / T_val, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(x))

    def dfermi_dT(self, omega: np.ndarray | float, T_override: Optional[float] = None) -> np.ndarray:
        """Exact temperature derivative \partial f_\alpha(\omega) / \partial T_\alpha."""
        T_val = self.T if T_override is None else T_override
        if T_val <= 0.0:
            return np.zeros_like(np.asarray(omega, dtype=float))
        w = np.asarray(omega, dtype=float)
        x = (w - self.mu) / T_val
        x_clip = np.clip(x, -500.0, 500.0)
        f_val = 1.0 / (1.0 + np.exp(x_clip))
        return (x / T_val) * f_val * (1.0 - f_val)

    def _generate_positive_edges(self) -> np.ndarray:
        if self.discretization == "linear":
            n_half = max(1, int(np.round(self.D / self.delta)))
            return np.linspace(0.0, self.D, n_half + 1)

        if self.discretization == "log":
            if self.Lambda <= 1.0:
                raise ValueError("Logarithmic discretization requires Lambda > 1.")
            exponents = np.arange(self.N_log, -1, -1, dtype=float) - self.z_shift
            exponents = np.maximum(exponents, 0.0)
            edges = self.D * (self.Lambda ** (-exponents))
            return np.unique(np.concatenate(([0.0], edges, [self.D])))

        if self.discretization == "log_linear":
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
        pos_edges = self._generate_positive_edges()
        a = pos_edges[:-1]
        b = pos_edges[1:]
        pos_widths = b - a

        if self.custom_shape is None and self.omega_ir <= 0.0:
            # Closed-form vectorized power-law moments on [a, b]
            w_int = self.Gamma0 * (b ** (self.r + 1.0) - a ** (self.r + 1.0)) / (self.r + 1.0)
            m_int = self.Gamma0 * (b ** (self.r + 2.0) - a ** (self.r + 2.0)) / (self.r + 2.0)
        else:
            # Vectorized 16-point Gauss-Legendre quadrature across all bins simultaneously
            pts = a[:, None] + pos_widths[:, None] * _GL_NODES_01[None, :]
            gam_pts = self.continuum_gamma(pts)
            w_int = pos_widths * (gam_pts @ _GL_WEIGHTS_01)
            m_int = pos_widths * ((pts * gam_pts) @ _GL_WEIGHTS_01)

        pos_v2 = w_int / np.pi
        pos_eps = np.where(w_int > 0.0, m_int / w_int, 0.5 * (a + b))

        neg_bins = np.column_stack((-pos_edges[1:][::-1], -pos_edges[:-1][::-1]))
        pos_bins = np.column_stack((pos_edges[:-1], pos_edges[1:]))
        self.bins = np.vstack((neg_bins, pos_bins))
        self.bin_widths = np.concatenate((pos_widths[::-1], pos_widths))
        self.epsilons = np.concatenate((-pos_eps[::-1], pos_eps))
        self.v = np.sqrt(np.concatenate((pos_v2[::-1], pos_v2)))

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
        return len(self.epsilons)

    @property
    def gamma_min(self) -> float:
        return float(np.min(self.gammas))

    def lindblad_rates(self, T_override: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        f_q = self.fermi(self.epsilons, T_override=T_override)
        lam1 = self.gammas * (1.0 - f_q)
        lam2 = self.gammas * f_q
        return lam1, lam2