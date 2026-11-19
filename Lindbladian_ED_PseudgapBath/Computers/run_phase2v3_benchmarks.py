"""
run_phase2v3_benchmarks.py  (benchmark2v3)
------------------------------------------
Main changes from benchmark2v2:
  - Removed the artificial out-of-band bound-state pole patch (`include_out_of_band_poles`),
    which previously treated D_S = D* as D_S -> inf while treating D_S < D* as a true
    finite box—causing an unphysical inconsistency between presets and diverging from
    genuine finite-bath dynamics (and IMEvo.jl). Outside [-D_S, D_S], Gamma_S(w) = 0 so
    the probe bath cannot thermally excite exterior states.
  - Consolidated all imports into Section 1 and all constants/flags into Section 2.
  - Added the parameterized `ParamSearch` preset (taking all physical constants and scan
    grids as explicit variables, defaulting to the analytical double-squeeze optimum
    Gamma0 = rho^(1/4) D*) and set `ACTIVE_PRESET = "ParamSearch"` across `RHO_LIST = [1e-4, 1e-3]`.
  - Organized all outputs into `results/phase2v3/{preset_name}/` (with individual panels
    in `phase2plots/`).

Executes Phase 2 (Step 1b: Two Baths Connected & Stress-Testing the Weak-Coupling Window)
with N_PG = 10,000 modes in the power-law bath and N_S = 4,000 modes in the probe bath
for the pseudogap exponents r in {0.3, 0.4, 0.5} (plus -1 < r < 1 sweep in Main Plot B).
"""

from __future__ import annotations

# ============================================================================
# 1. ALL IMPORTS (Standard Library, Third-Party, and Local Project Modules)
# ============================================================================
import os
import sys
import gc
import json
import math
import time
import logging
import warnings
import argparse
import resource
import subprocess
import faulthandler
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

# Suppress benign RuntimeWarnings (e.g. overflow in high-x Fermi exponentials) before scientific imports
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"

import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from bath import Bath
from lyapunov_solver import LyapunovSolver
from spectral_qrt import (
    alpha_func_eq14,
    continuum_self_energy_finite_D,
    analytic_tilde_S0_eq14,
    analytic_tilde_S0_eq12,
)

# ============================================================================
# 2. ALL CONSTANTS, USER CONTROL FLAGS & PATH DEFINITIONS
# ============================================================================
# Primary execution & physics control flags (edit here or override via CLI flags)
RUN_ON_SLURM: bool = False                          # True: SLURM cluster; False: Local PC/Mac (../results)
ACTIVE_PRESET: Union[str, List[str]] = "ParamSearch"  # Preset name, list of names, or "all"
RHO_LIST: List[float] = [1.0e-4, 1.0e-3]            # Dynamic range ratios rho = T_min / D* to sweep
TO_PRINT: List[str] = ["A", "B"]                    # Individual panels to export into `phase2plots/`

# Explicit variable overrides for `ParamSearch` (set any key to a float/list to override auto double-squeeze)
PARAM_SEARCH_OVERRIDES: Dict[str, Any] = {
    "D": None,          # Default: 1.0 * D*
    "D_S_cut": None,    # Default: 1.0 * D*
    "Gamma0": None,     # Default: (rho ** 0.25) * D*  (0.10 D* at rho=1e-4; 0.1778 D* at rho=1e-3)
    "GammaS0": None,    # Default: rho * D* (= T_min)
    "T_PG": None,       # Default: rho * D* (= T_min)
    "T_S_fixed": None,  # Default: geometric midpoint sqrt(T_IR(0.5) * T*(0.5)) = sqrt(rho) * D*
    "TS_panel_B": None, # Default: [T_low_opt, T*(0.5), T_high_opt, D*] chosen inside double squeeze
}

# Default (d, k) parameters for `Ds_to_D_is_k`: D = D* / d, D_S = D / k
DEFAULT_D_DIVISOR: float = 1.0
DEFAULT_K_RATIO: float = 10.0

# Ever-fixed global UV bandwidth cap
D_STAR: float = 1.0

# Target pseudogap exponents:
# - Core & stress-test panels (a)-(h) and Main Plot (A) focus on r in {0.3, 0.4, 0.5}
# - Main Plot (B) sweeps -1 < r < 1 across double-squeeze-selected probe temperatures
R_LIST_ALL: List[float] = [0.3, 0.4, 0.5]
R_LIST_PANEL_B: List[float] = [-0.9, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 0.9]

# Discretization, quadrature & validation constants
TARGET_N_PG: int = 10000
TARGET_N_S: int = 4000
LN2: float = float(np.log(2.0))
PC_RESERVED_CPUS: int = 2
VALID_PANEL_KEYS = {"a", "b", "c", "d", "e", "f", "g", "h", "A", "B"}
ProbeProfileType = Literal["flat_band", "lorentzian"]

_GL64_S, _GL64_W = np.polynomial.legendre.leggauss(64)
_GL64_S = 0.5 * (_GL64_S + 1.0)
_GL64_W = 0.5 * _GL64_W

_GL128_S, _GL128_W = np.polynomial.legendre.leggauss(128)
_GL128_S = 0.5 * (_GL128_S + 1.0)
_GL128_W = 0.5 * _GL128_W

# Directory & environment routing driven by RUN_ON_SLURM
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if RUN_ON_SLURM:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    faulthandler.enable()
    PROJECT_ROOT = (
        "/home/amsalda/PseudoGap"
        if os.path.exists("/home/amsalda/PseudoGap")
        else os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    )
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

BASE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PHASE2V3_ROOT_DIR = os.path.join(BASE_RESULTS_DIR, "phase2v3")
os.makedirs(PHASE2V3_ROOT_DIR, exist_ok=True)
PRESETS_JSON_PATH = os.path.join(SCRIPT_DIR, "phase2_presets.json")


def get_preset_dirs(preset_name: str) -> Tuple[str, str]:
    """Return `(results_dir, phase2_plots_dir)` inside `results/phase2v3/{preset_name}/`."""
    res_dir = os.path.join(PHASE2V3_ROOT_DIR, preset_name)
    plots_dir = os.path.join(res_dir, "phase2plots")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return res_dir, plots_dir


# ============================================================================
# 3. GLOBAL MATPLOTLIB STYLE (KC FIGURE GUIDELINES — VECTOR PDF ONLY)
# ============================================================================
def apply_kc_figure_style() -> None:
    """Configure Matplotlib rcParams following KC's Figure Guidelines."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Myriad Pro", "DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 1.5,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.22,
        "legend.framealpha": 0.94,
        "legend.edgecolor": "#cccccc",
    })


apply_kc_figure_style()


# ============================================================================
# 4. INPUT VALIDATION & SYSTEM HARDWARE / PROGRESS TELEMETRY
# ============================================================================
def sanitize_to_print(raw_to_print: Any) -> List[str]:
    """Validate `TO_PRINT` panel identifiers; fallback to `['A', 'B']` if invalid."""
    default_panels = ["A", "B"]
    if not isinstance(raw_to_print, (list, tuple, set)) or len(raw_to_print) == 0:
        print(f"[Warning] Invalid or empty TO_PRINT={raw_to_print!r}; falling back to {default_panels}.", flush=True)
        return default_panels

    cleaned: List[str] = []
    for item in raw_to_print:
        if not isinstance(item, str) or item.strip() not in VALID_PANEL_KEYS:
            print(
                f"[Warning] Corrupt panel entry {item!r} in TO_PRINT={raw_to_print!r}; "
                f"falling back to {default_panels}.",
                flush=True,
            )
            return default_panels
        key = item.strip()
        if key not in cleaned:
            cleaned.append(key)
    return cleaned


def get_worker_count(run_on_slurm: bool = RUN_ON_SLURM, reserved_cpus: int = PC_RESERVED_CPUS) -> Tuple[int, int]:
    """Determine `(max_cpu, n_workers)`, reserving `reserved_cpus` cores on local PC/Mac."""
    if run_on_slurm:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        max_cpu = int(slurm_cpus) if slurm_cpus and slurm_cpus.isdigit() else (os.cpu_count() or 1)
        return max_cpu, max(1, max_cpu)
    max_cpu = os.cpu_count() or 1
    n_workers = max(1, max_cpu - reserved_cpus)
    return max_cpu, n_workers


def get_system_ram_gb() -> Optional[float]:
    """Query total physical RAM in GB (cross-platform macOS & Linux)."""
    try:
        if sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return int(out) / (1024.0 ** 3)
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024.0 ** 2)
    except Exception:
        pass
    return None


def get_peak_rss_mb() -> float:
    """Return peak resident set size (RSS) in MB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024.0 ** 2) if sys.platform == "darwin" else rss / 1024.0


class RunTelemetry:
    """Lightweight progress, parallel scaling efficiency, and CPU contention monitor."""

    def __init__(
        self,
        stage_label: str,
        total_tasks: int,
        max_cpu: int,
        n_workers: int,
        reserved_cpus: int,
        log_interval_sec: float = 3.0,
    ) -> None:
        self.stage_label = stage_label
        self.total_tasks = total_tasks
        self.max_cpu = max_cpu
        self.n_workers = n_workers
        self.reserved_cpus = reserved_cpus
        self.log_interval_sec = log_interval_sec

        self.t_start = time.perf_counter()
        self.last_log_time = self.t_start
        self.completed_tasks = 0
        self.cumulative_worker_sec = 0.0
        self.task_durations: List[float] = []
        self.contention_events: List[Dict[str, Any]] = []
        self.initial_load_1m = self._read_load_1m()
        self.total_ram_gb = get_system_ram_gb()

    @staticmethod
    def _read_load_1m() -> float:
        try:
            return float(os.getloadavg()[0])
        except Exception:
            return 0.0

    def print_banner(self) -> None:
        ram_str = f"{self.total_ram_gb:.1f} GB" if self.total_ram_gb is not None else "N/A"
        mode_str = "SLURM Cluster" if RUN_ON_SLURM else f"Local PC/Mac (reserved={self.reserved_cpus} CPUs)"
        print(
            f"[Telemetry | {self.stage_label}] Mode: {mode_str} | "
            f"Workers: {self.n_workers}/{self.max_cpu} CPUs | RAM: {ram_str} | "
            f"Initial 1m Load: {self.initial_load_1m:.2f} | Tasks: {self.total_tasks}",
            flush=True,
        )

    def record_task(self, task_duration_sec: float) -> None:
        self.completed_tasks += 1
        self.cumulative_worker_sec += task_duration_sec
        self.task_durations.append(task_duration_sec)

        now = time.perf_counter()
        elapsed = max(1e-9, now - self.t_start)
        is_final = self.completed_tasks == self.total_tasks

        if is_final or (now - self.last_log_time >= self.log_interval_sec):
            self.last_log_time = now
            pct = 100.0 * self.completed_tasks / max(1, self.total_tasks)
            rate = self.completed_tasks / elapsed
            eta = (self.total_tasks - self.completed_tasks) / max(1e-9, rate)
            eff_clamped = min(1.0, self.cumulative_worker_sec / (self.n_workers * elapsed))
            load_1m = self._read_load_1m()

            print(
                f"[Telemetry | {self.stage_label}] {self.completed_tasks:3d}/{self.total_tasks:3d} ({pct:5.1f}%) | "
                f"Elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s | Rate: {rate:5.2f} tasks/s | "
                f"Parallel Eff: {100.0 * eff_clamped:5.1f}% | Load: {load_1m:4.1f}/{self.max_cpu}",
                flush=True,
            )

            if self.completed_tasks >= self.n_workers and (
                eff_clamped < 0.72 or load_1m > 1.15 * self.max_cpu
            ):
                event = {
                    "completed_tasks": self.completed_tasks,
                    "elapsed_sec": round(elapsed, 2),
                    "parallel_efficiency": round(eff_clamped, 4),
                    "load_1m": round(load_1m, 2),
                }
                self.contention_events.append(event)
                print(
                    f"  [Telemetry Warning] CPU contention / throttling detected "
                    f"(Eff={100.0 * eff_clamped:.1f}%, Load={load_1m:.1f}/{self.max_cpu}).",
                    flush=True,
                )

    def summarize(self) -> Dict[str, Any]:
        total_wall = max(1e-9, time.perf_counter() - self.t_start)
        eff = min(1.0, self.cumulative_worker_sec / (self.n_workers * total_wall))
        arr = np.array(self.task_durations, dtype=float) if self.task_durations else np.array([0.0])
        return {
            "stage": self.stage_label,
            "run_on_slurm": RUN_ON_SLURM,
            "hardware": {
                "max_cpu": self.max_cpu,
                "n_workers": self.n_workers,
                "reserved_cpus": self.reserved_cpus,
                "total_ram_gb": round(self.total_ram_gb, 2) if self.total_ram_gb is not None else None,
                "peak_rss_mb": round(get_peak_rss_mb(), 2),
                "initial_load_1m": round(self.initial_load_1m, 2),
                "final_load_1m": round(self._read_load_1m(), 2),
            },
            "timings": {
                "total_tasks": self.total_tasks,
                "total_wall_sec": round(total_wall, 3),
                "cumulative_worker_cpu_sec": round(self.cumulative_worker_sec, 3),
                "parallel_efficiency": round(eff, 4),
                "mean_task_sec": round(float(np.mean(arr)), 4),
                "max_task_sec": round(float(np.max(arr)), 4),
            },
            "contention_events": self.contention_events,
        }


# ============================================================================
# 5. DOUBLE-SQUEEZE DIAGNOSTICS & PRESETS (`ParamSearch` + BASELINE PRESETS)
# ============================================================================
def assert_within_caps(
    val: float | np.ndarray,
    name: str,
    lower_bound: float,
    upper_bound: float = D_STAR,
) -> None:
    """Verify that physical energy scale `val` lies strictly within [lower_bound, upper_bound]."""
    arr = np.asarray(val, dtype=float)
    if np.any(arr < lower_bound * (1.0 - 1e-12)) or np.any(arr > upper_bound * (1.0 + 1e-12)):
        raise ValueError(
            f"Physical parameter '{name}' = {val} violates bounds "
            f"[{lower_bound:.1e}, {upper_bound:.1f}]."
        )


def _all_decades_in_window(t_min: float, d_star: float) -> List[float]:
    """Collect every integer power of 10 (`10^m`) strictly within `[t_min, d_star]`."""
    m_min = int(math.ceil(math.log10(t_min * (1.0 - 1e-12))))
    m_max = int(math.floor(math.log10(d_star * (1.0 + 1e-12))))
    vals = [10.0 ** m for m in range(m_min, m_max + 1)]
    return vals if vals else [t_min, d_star]


def _round_clean_sigfig(val: float, sig: int = 1) -> float:
    """Round a positive float to `sig` significant figures (e.g. 3.16e-3 -> 3e-3)."""
    if val <= 0.0:
        return val
    exp = int(math.floor(math.log10(val)))
    scale = 10.0 ** (exp - sig + 1)
    return round(val / scale) * scale


def compute_double_squeeze_scales(
    r: float,
    Gamma0: float,
    GammaS0: float,
    D: float,
    D_S: float,
    d_star: float = D_STAR,
) -> Dict[str, float]:
    """
    Evaluate both squeeze inequalities for exponent r > 0:
      - Left weak-coupling breakdown: T_IR(r) = (GammaS0 / Gamma0)^(1/r) * D*
      - Self-energy crossover:        T*(r)   = (Gamma0 / D*)^(1/(1-r)) * D*
      - Effective UV probe roll-off:  T_UV    = 0.15 * D_S
      - Optimal low-T_S plateau point:  T_low_opt  = sqrt(max(T_min, T_IR(r)) * T*(r))
      - Optimal high-T_S plateau point: T_high_opt = sqrt(T*(r) * T_UV)
    """
    t_ir = ((GammaS0 / Gamma0) ** (1.0 / r)) * d_star if r > 0.0 else 0.0
    t_star = ((Gamma0 / d_star) ** (1.0 / (1.0 - r))) * d_star
    t_uv = 0.15 * D_S
    t_low_opt = math.sqrt(max(GammaS0, t_ir) * t_star)
    t_high_opt = math.sqrt(t_star * t_uv)
    return {
        "r": float(r),
        "T_IR": float(t_ir),
        "T_star": float(t_star),
        "T_UV_eff": float(t_uv),
        "T_low_opt": float(t_low_opt),
        "T_high_opt": float(t_high_opt),
        "low_window_decades": math.log10(max(1e-30, t_star / max(GammaS0, t_ir))),
        "high_window_decades": math.log10(max(1e-30, t_uv / t_star)),
    }


def make_preset_config(
    preset_name: str,
    D: float,
    D_S_cut: float,
    Gamma0: float,
    rho: float,
    GammaS0: Optional[float] = None,
    T_PG: Optional[float] = None,
    T_S_fixed: Optional[float] = None,
    T_S_range: Optional[Tuple[float, float]] = None,
    TS_panel_B: Optional[List[float]] = None,
    T_PG_list: Optional[List[float]] = None,
    D_S_list: Optional[List[float]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    description: str = "",
) -> Dict[str, Any]:
    """
    Construct a fully parameterized preset configuration where ALL physical energy
    constants and scan grids are defined via `D_STAR` (`D* = 1.0`) and `T_min = rho * D*`.
    """
    rho = float(rho)
    if not (0.0 < rho < 1.0):
        raise ValueError(f"rho = T_min / D* must satisfy 0 < rho < 1; got {rho}.")

    d_cap = float(D_STAR)
    t_min = rho * d_cap
    D_val = float(D)
    D_S_val = float(D_S_cut)
    G0_val = float(Gamma0)

    GS0_val = float(GammaS0) if GammaS0 is not None else t_min
    TPG_val = float(T_PG) if T_PG is not None else t_min
    TS_fix_val = float(T_S_fixed) if T_S_fixed is not None else min(0.1 * d_cap, 10.0 * t_min)
    ts_rng = (float(T_S_range[0]), float(T_S_range[1])) if T_S_range is not None else (t_min, d_cap)

    for v, n in [
        (D_val, "D"), (G0_val, "Gamma0"), (GS0_val, "GammaS0"),
        (D_S_val, "D_S_cut"), (TPG_val, "T_PG"), (TS_fix_val, "T_S_fixed"),
        (ts_rng[0], "T_S_range[0]"), (ts_rng[1], "T_S_range[1]"),
    ]:
        assert_within_caps(v, n, lower_bound=t_min, upper_bound=d_cap)

    gs_ratio_min = min(rho, GS0_val / G0_val)
    gs_ratio_max = max(1.0e-1, 10.0 * gs_ratio_min)

    t_pg_vals = [float(x) for x in T_PG_list] if T_PG_list is not None else [
        t_min,
        (rho ** (2.0 / 3.0)) * d_cap,
        (rho ** (1.0 / 3.0)) * d_cap,
    ]
    d_s_vals = [float(x) for x in D_S_list] if D_S_list is not None else sorted(set([
        t_min,
        (rho ** (2.0 / 3.0)) * d_cap,
        (rho ** (1.0 / 3.0)) * d_cap,
        D_S_val,
        d_cap,
    ]))
    ts_b_vals = [float(x) for x in TS_panel_B] if TS_panel_B is not None else _all_decades_in_window(t_min, d_cap)
    for ts_b in ts_b_vals:
        assert_within_caps(ts_b, "TS_panel_B element", lower_bound=t_min, upper_bound=d_cap)

    squeeze_diag = {
        r: compute_double_squeeze_scales(r, G0_val, GS0_val, D_val, D_S_val, d_star=d_cap)
        for r in R_LIST_ALL
    }

    cfg: Dict[str, Any] = {
        "preset_name": preset_name,
        "description": description or (
            f"Preset '{preset_name}': D={D_val/d_cap:g}D*, D_S={D_S_val/d_cap:g}D*, "
            f"Gamma0={G0_val/d_cap:.4g}D*, GammaS0={GS0_val/d_cap:.1e}D*, T_PG={TPG_val/d_cap:.1e}D*"
        ),
        "rho": rho,
        "D": D_val,
        "D_star": d_cap,
        "T_min": t_min,
        "Gamma0": G0_val,
        "GammaS0": GS0_val,
        "D_S_cut": D_S_val,
        "T_PG": TPG_val,
        "T_S_fixed": TS_fix_val,
        "T_S_range": ts_rng,
        "GS_over_G0_range": (float(gs_ratio_min), float(gs_ratio_max)),
        "T_PG_list": t_pg_vals,
        "D_S_list": d_s_vals,
        "TS_panel_B": ts_b_vals,
        "double_squeeze": squeeze_diag,
        "file_suffix": f"_{preset_name}_rho{rho:.0e}",
    }
    if extra_meta:
        cfg.update(extra_meta)
    return cfg


def make_preset_ParamSearch(
    rho: float,
    D: Optional[float] = None,
    D_S_cut: Optional[float] = None,
    Gamma0: Optional[float] = None,
    GammaS0: Optional[float] = None,
    T_PG: Optional[float] = None,
    T_S_fixed: Optional[float] = None,
    T_S_range: Optional[Tuple[float, float]] = None,
    TS_panel_B: Optional[List[float]] = None,
    T_PG_list: Optional[List[float]] = None,
    D_S_list: Optional[List[float]] = None,
    preset_name: str = "ParamSearch",
) -> Dict[str, Any]:
    """
    Universal variable-parameter preset `ParamSearch` for exploring the double-squeeze
    parameter space:
      1) T_min << T* = (Gamma0 / D*)^(1/(1-r)) D* << T_S << D_S = D*
      2) (GammaS0 / Gamma0)^(1/r) << T_S / D << (Gamma0 / D)^(1/(1-r))

    When parameters are left as `None`, computes the analytical geometric optimum:
      - D = D*, D_S_cut = D*
      - Gamma0 = (rho ** 0.25) * D*  (0.10 D* for rho=1e-4; 0.1778 D* for rho=1e-3)
      - GammaS0 = T_PG = T_min = rho * D*
      - T_S_fixed = (rho ** 0.75) * D*  (sits cleanly on the low-T_S plateau for all r)
      - TS_panel_B: automatically picks [T_low_opt, T*(0.5), T_high_opt, D*] inside [T_min, D*]
        so Plot B captures the low-T_S S=r*ln2 regime (without dipping into T < T_IR for large r),
        the T* crossover, the high-T_S S->ln2 rise, and the D_S UV cutoff.
    """
    d_cap = float(D_STAR)
    t_min = float(rho) * d_cap

    D_val = float(D) if D is not None else 1.0 * d_cap
    DS_val = float(D_S_cut) if D_S_cut is not None else 1.0 * d_cap
    G0_val = float(Gamma0) if Gamma0 is not None else (float(rho) ** 0.25) * d_cap
    GS0_val = float(GammaS0) if GammaS0 is not None else t_min
    TPG_val = float(T_PG) if T_PG is not None else t_min
    TS_fix_val = float(T_S_fixed) if T_S_fixed is not None else (float(rho) ** 0.75) * d_cap

    if TS_panel_B is None:
        sq05 = compute_double_squeeze_scales(0.5, G0_val, GS0_val, D_val, DS_val, d_star=d_cap)
        # Low-T_S point chosen via Plot A's double-squeeze window (above T_IR even for r -> 0.9, well below T*)
        t_ir_09 = ((GS0_val / G0_val) ** (1.0 / 0.9)) * d_cap
        t_low_b = _round_clean_sigfig(max(t_min, 1.5 * t_ir_09, 0.25 * sq05["T_low_opt"]), sig=1)
        t_mid_b = _round_clean_sigfig(sq05["T_star"], sig=1)
        t_high_b = _round_clean_sigfig(min(0.15 * DS_val, sq05["T_high_opt"]), sig=1)
        candidates = [
            min(d_cap, max(t_min, t_low_b)),
            min(d_cap, max(t_min, t_mid_b)),
            min(d_cap, max(t_min, t_high_b)),
            d_cap,
        ]
        TS_panel_B = sorted(set(candidates))

    return make_preset_config(
        preset_name=preset_name,
        D=D_val,
        D_S_cut=DS_val,
        Gamma0=G0_val,
        rho=rho,
        GammaS0=GS0_val,
        T_PG=TPG_val,
        T_S_fixed=TS_fix_val,
        T_S_range=T_S_range,
        TS_panel_B=TS_panel_B,
        T_PG_list=T_PG_list,
        D_S_list=D_S_list,
        description=(
            f"ParamSearch (Double-Squeeze): D={D_val/d_cap:g}D*, D_S={DS_val/d_cap:g}D*, "
            f"Gamma0={G0_val/d_cap:.4g}D*, GammaS0={GS0_val/d_cap:.1e}D*, T_PG={TPG_val/d_cap:.1e}D*"
        ),
    )


def make_preset_Ds_to_D_is_k(
    rho: float,
    d: float = DEFAULT_D_DIVISOR,
    k: float = DEFAULT_K_RATIO,
) -> Dict[str, Any]:
    """Preset `Ds_to_D_is_k`: D = D* / d and D_S = D / k (default d=1, k=10)."""
    d_val = float(d)
    k_val = float(k)
    if d_val <= 0.0 or k_val <= 0.0:
        raise ValueError(f"Divisors d and k must be positive; got d={d_val}, k={k_val}.")
    D_val = D_STAR / d_val
    D_S_val = D_val / k_val
    is_default_dk = np.isclose(d_val, DEFAULT_D_DIVISOR) and np.isclose(k_val, DEFAULT_K_RATIO)
    pname = "Ds_to_D_is_k" if is_default_dk else f"Ds_to_D_is_k_d{d_val:g}_k{k_val:g}"
    return make_preset_config(
        preset_name=pname,
        D=D_val,
        D_S_cut=D_S_val,
        Gamma0=1.0 * D_STAR,
        rho=rho,
        extra_meta={"d": d_val, "k": k_val},
        description=f"D = D*/{d_val:g}, D_S = D/{k_val:g} (all other constants via D*)",
    )


def build_preset_catalog(
    rho: float,
    d: float = DEFAULT_D_DIVISOR,
    k: float = DEFAULT_K_RATIO,
    param_search_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build `ParamSearch` alongside the baseline presets for a given `rho`."""
    overrides = dict(PARAM_SEARCH_OVERRIDES)
    if param_search_overrides:
        overrides.update({key: val for key, val in param_search_overrides.items() if val is not None})

    param_search_cfg = make_preset_ParamSearch(
        rho=rho,
        D=overrides.get("D"),
        D_S_cut=overrides.get("D_S_cut"),
        Gamma0=overrides.get("Gamma0"),
        GammaS0=overrides.get("GammaS0"),
        T_PG=overrides.get("T_PG"),
        T_S_fixed=overrides.get("T_S_fixed"),
        T_S_range=overrides.get("T_S_range"),
        TS_panel_B=overrides.get("TS_panel_B"),
        T_PG_list=overrides.get("T_PG_list"),
        D_S_list=overrides.get("D_S_list"),
    )
    ds_to_d_cfg = make_preset_Ds_to_D_is_k(rho=rho, d=d, k=k)
    return {
        "ParamSearch": param_search_cfg,
        "low_T_default": make_preset_config(
            preset_name="low_T_default",
            D=1.0 * D_STAR,
            D_S_cut=1.0 * D_STAR,
            Gamma0=1.0 * D_STAR,
            rho=rho,
            description="Default setup: D = D*, D_S = D*, Gamma0 = D*, GammaS0 = T_PG = T_min",
        ),
        "low_G0_highT": make_preset_config(
            preset_name="low_G0_highT",
            D=1.0 * D_STAR,
            D_S_cut=1.0 * D_STAR,
            Gamma0=0.1 * D_STAR,
            rho=rho,
            description="Weak-Gamma0 high-T crossover setup: Gamma0 = D*/10 (same GS0/G0 scan range)",
        ),
        ds_to_d_cfg["preset_name"]: ds_to_d_cfg,
    }


def sync_presets_json(catalog: Dict[str, Dict[str, Any]], rho: float) -> None:
    """Record presets (keyed by `{preset_name}_rho{rho:.0e}`) in `phase2_presets.json`."""
    store: Dict[str, Any] = {}
    if os.path.exists(PRESETS_JSON_PATH):
        try:
            with open(PRESETS_JSON_PATH, "r", encoding="utf-8") as f:
                store = json.load(f)
        except Exception:
            store = {}
    for pname, pcfg in catalog.items():
        store[f"{pname}_rho{rho:.0e}"] = pcfg
    with open(PRESETS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)


# ============================================================================
# 6. TWO-BATH SOLVER & FINITE-BAND CONTINUUM INTEGRALS (NO ARTIFICIAL POLES)
# ============================================================================
def _worker_init() -> None:
    """Enforce single-threaded BLAS, suppressed warnings, and Agg backend per worker."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["MPLBACKEND"] = "Agg"


def make_lorentzian_shape(D_S_width: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return Lorentzian frequency profile g_S(w) = 1 / (1 + (w / D_S_width)^2)."""
    return lambda w: 1.0 / (1.0 + (np.asarray(w, dtype=float) / D_S_width) ** 2)


def get_color_map(r_values: Sequence[float]) -> Dict[float, Any]:
    """Assign high-contrast publication colors sorted from smallest to largest r."""
    unique_r = sorted(set(float(r) for r in r_values))
    if len(unique_r) == 2:
        palette = ["#1f77b4", "#d55e00"]
        return {r: palette[i] for i, r in enumerate(unique_r)}
    if len(unique_r) == 3:
        palette = ["#1f77b4", "#009e73", "#d55e00"]
        return {r: palette[i] for i, r in enumerate(unique_r)}
    cmap = plt.cm.turbo(np.linspace(0.12, 0.88, len(unique_r)))
    return {r: cmap[i] for i, r in enumerate(unique_r)}


def build_two_bath_solver(
    r: float,
    T_S: float,
    probe_profile: ProbeProfileType,
    D_S_cutoff: float,
    Gamma0: float,
    GammaS0: float,
    T_PG: float,
    D_PG: float,
    t_min: float,
    d_star: float = D_STAR,
    omega_ir_PG: float = 0.0,
    n_pg: int = TARGET_N_PG,
    n_s: int = TARGET_N_S,
) -> LyapunovSolver:
    """
    Construct the two-bath LyapunovSolver with genuine finite bandwidths `D_PG = D` and
    `D_S_cutoff = D_S` (no exterior modes outside the active bands).
    """
    for val, name in [
        (T_S, "T_S"), (T_PG, "T_PG"), (Gamma0, "Gamma0"),
        (D_S_cutoff, "D_S_cutoff"), (D_PG, "D_PG"),
    ]:
        assert_within_caps(val, name, lower_bound=t_min, upper_bound=d_star)
    # GammaS0 may drop below t_min during the weak-coupling stress sweep (GammaS0 -> 0)
    assert_within_caps(GammaS0, "GammaS0", lower_bound=1e-12, upper_bound=d_star)

    if probe_profile == "flat_band":
        D_S_band = D_S_cutoff
        probe_fn = None
    elif probe_profile == "lorentzian":
        D_S_band = d_star
        probe_fn = make_lorentzian_shape(D_S_cutoff)
    else:
        raise ValueError(f"Unknown probe_profile: {probe_profile}")

    active_scale = min(T_S, D_S_cutoff)
    D_star_pg = min(0.25 * D_PG, max(8.0 * active_scale, 5.0 * t_min))
    D_star_s = min(0.25 * D_S_band, max(8.0 * min(T_S, D_S_band), 5.0 * t_min))

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
) -> Dict[str, np.ndarray]:
    """
    Evaluate the exact finite-band two-bath tight-binding sweep across `eps_d_grid`
    via `solver.sweep_eps_d_protocol` (`schur_lddl`), with zero artificial out-of-band poles.
    """
    return solver.sweep_eps_d_protocol(eps_d_grid, target_bath="S", method="schur_lddl")


def _sigma_pg_continuum_general(w_pos: np.ndarray, Gamma0: float, r: float, D_PG: float) -> np.ndarray:
    """
    Exact finite-bandwidth Kramers-Kronig self-energy Sigma_{PG;D}(w) for w > 0,
    supporting both in-band (w < D_PG) and exterior (w > D_PG) frequencies.
    """
    w_arr = np.asarray(w_pos, dtype=float)
    out = np.zeros_like(w_arr)
    mask_in = w_arr < D_PG
    mask_out = ~mask_in

    if np.any(mask_in):
        out[mask_in] = continuum_self_energy_finite_D(w_arr[mask_in], Gamma0, r, D=D_PG)

    if np.any(mask_out):
        w_ext = w_arr[mask_out]
        s = _GL64_S[None, :]
        ws = _GL64_W[None, :]
        y_sq = s ** (2.0 / (r + 1.0))
        denom = (w_ext[:, None] ** 2) - (D_PG ** 2) * y_sq
        pref = (2.0 * Gamma0 * (D_PG ** (r + 1.0)) * w_ext) / (np.pi * (r + 1.0))
        out[mask_out] = pref * np.sum(ws / denom, axis=1)

    return out


def eval_analytic_two_bath(
    eps_d: float | np.ndarray,
    T_S: float,
    Gamma0: float,
    GammaS0: float,
    r: float,
    probe_profile: ProbeProfileType,
    D_S_cutoff: float,
    D_PG: float,
    d_star: float = D_STAR,
    eps_max: Optional[float] = None,
) -> float | np.ndarray:
    """
    Evaluate the exact finite-band two-bath continuum integral over the active probe
    bandwidth ([-D_S, D_S] for flat_band; [-D*, D*] for lorentzian), folded onto
    [0, min(D_S_band, 45 * T_S)] with zero contribution where Gamma_S(w) = 0.
    """
    scalar_input = np.ndim(eps_d) == 0
    ed_arr = np.atleast_1d(np.asarray(eps_d, dtype=float))
    if eps_max is None:
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        eff_w = Gamma0 * (T_S ** r) * sec_pr2 + GammaS0
        eps_max = max(25.0 * d_star, 100.0 * eff_w)
    eps_min = -abs(float(eps_max))

    D_S_band = D_S_cutoff if probe_profile == "flat_band" else d_star
    probe_fn = None if probe_profile == "flat_band" else make_lorentzian_shape(D_S_cutoff)
    x_full_max = 45.0
    W_max = min(D_S_band, x_full_max * T_S)

    def _eval_band_segment(w_pos: np.ndarray, weights_w: np.ndarray) -> np.ndarray:
        emx = np.exp(-w_pos / T_S)
        df_factor = (w_pos / (T_S ** 2)) * emx / ((1.0 + emx) ** 2)
        sig_pos = _sigma_pg_continuum_general(w_pos, Gamma0, r, D_PG)
        g_p = np.where(w_pos <= D_PG, Gamma0 * (w_pos ** r), 0.0)
        g_s = (
            np.full_like(w_pos, GammaS0)
            if probe_fn is None
            else GammaS0 * np.asarray(probe_fn(w_pos), dtype=float)
        )
        g_tot = g_p + g_s
        branch = g_s / g_tot

        ed_col = ed_arr[:, None]
        ph_neg = (
            np.arctan((ed_col + w_pos[None, :] - sig_pos[None, :]) / g_tot[None, :])
            - np.arctan((eps_min + w_pos[None, :] - sig_pos[None, :]) / g_tot[None, :])
        )
        ph_pos = (
            np.arctan((ed_col - w_pos[None, :] + sig_pos[None, :]) / g_tot[None, :])
            - np.arctan((eps_min - w_pos[None, :] + sig_pos[None, :]) / g_tot[None, :])
        )
        integrand = ((ph_neg - ph_pos) / np.pi) * branch[None, :] * df_factor[None, :]
        return integrand @ weights_w

    total = np.zeros_like(ed_arr)

    # Segment 1: [0, min(D_PG, W_max)] where both PG and S baths are active
    w1_max = min(D_PG, W_max)
    if w1_max > 0.0:
        p = max(0.5, 2.0 - r)
        w1 = w1_max * (_GL128_S ** (1.0 / p))
        dw1 = (w1_max / p) * (_GL128_S ** (1.0 / p - 1.0)) * _GL128_W
        total += _eval_band_segment(w1, dw1)

    # Segment 2: (D_PG, W_max] when probe extends past D_PG (Gamma_PG = 0, Gamma_S > 0)
    if W_max > D_PG * (1.0 + 1e-12):
        eps_edge = 1e-6 * D_PG
        u_min = np.log(eps_edge)
        u_max = np.log((W_max - D_PG) + eps_edge)
        u_nodes = u_min + (u_max - u_min) * _GL128_S
        du_weights = (u_max - u_min) * _GL128_W
        delta_w = np.exp(u_nodes) - eps_edge
        w2 = D_PG + delta_w
        dw2 = (delta_w + eps_edge) * du_weights
        total += _eval_band_segment(w2, dw2)

    return float(total[0]) if scalar_input else total


def _execute_benchmark_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level picklable worker function executed across `ProcessPoolExecutor`."""
    t0 = time.perf_counter()
    kind = task["kind"]
    r = float(task["r"])
    T_S = float(task["T_S"])
    Gamma0 = float(task["Gamma0"])
    GammaS0 = float(task["GammaS0"])
    T_PG = float(task["T_PG"])
    D_S_cutoff = float(task["D_S_cutoff"])
    D_PG = float(task["D_PG"])
    d_star = float(task["d_star"])
    t_min = float(task["t_min"])
    probe_profile: ProbeProfileType = task["probe_profile"]
    eps_d_grid = np.asarray(task["eps_d_grid"], dtype=float)

    solver = build_two_bath_solver(
        r=r,
        T_S=T_S,
        probe_profile=probe_profile,
        D_S_cutoff=D_S_cutoff,
        Gamma0=Gamma0,
        GammaS0=GammaS0,
        T_PG=T_PG,
        D_PG=D_PG,
        t_min=t_min,
        d_star=d_star,
    )
    sweep = two_bath_tight_binding_sweep(solver=solver, eps_d_grid=eps_d_grid)

    out: Dict[str, Any] = {
        "kind": kind,
        "meta": task.get("meta", {}),
        "tS_num_0": float(sweep["S_eps_d"][-1]),
    }
    if kind == "core_sweep":
        out["sweep"] = sweep

    del solver
    gc.collect()
    out["duration_sec"] = time.perf_counter() - t0
    return out


# ============================================================================
# 7. PARALLEL BENCHMARK ORCHESTRATOR (CORE + STRESS TESTS + PANEL B DATA)
# ============================================================================
def run_phase2_parallel_suite(
    cfg: Dict[str, Any],
    probe_profile: ProbeProfileType,
    executor: ProcessPoolExecutor,
    max_cpu: int,
    n_workers: int,
    r_list: List[float] = R_LIST_ALL,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Execute all Step 1b core verification, stress tests, and Main Plot (B) sweeps."""
    D = float(cfg["D"])
    D_cap = float(cfg["D_star"])
    t_min = float(cfg["T_min"])
    Gamma0 = float(cfg["Gamma0"])
    GammaS0 = float(cfg["GammaS0"])
    ratio_S_ref = GammaS0 / Gamma0
    T_S_fixed = float(cfg["T_S_fixed"])
    T_PG_base = float(cfg["T_PG"])
    D_S_cut = float(cfg["D_S_cut"])
    ts_min, ts_max = cfg["T_S_range"]
    gs_ratio_min, gs_ratio_max = cfg["GS_over_G0_range"]
    ts_panel_b = cfg["TS_panel_B"]

    prof_desc = (
        f"Flat Probe Gamma_S(w) = GammaS0 * Theta(D_S - |w|) with D_S = {D_S_cut/D_cap:g}D*"
        if probe_profile == "flat_band"
        else f"Lorentzian Probe Gamma_S(w) = GammaS0 / (1 + (w/D_S)^2) with D_S = {D_S_cut/D_cap:g}D*"
    )

    sq_lines = [
        f"    r={r:+.2f}: T_min={t_min:.1e}, T_IR={(sq['T_IR']):.2e}, "
        f"T_low_opt={sq['T_low_opt']:.2e}, T*={sq['T_star']:.2e}, "
        f"T_high_opt={sq['T_high_opt']:.2e}, T_UV_eff={sq['T_UV_eff']:.2e}, D_S={D_S_cut:g}"
        for r, sq in sorted(cfg["double_squeeze"].items())
    ]

    header = (
        "=" * 124 + "\n"
        f"STEP 1b ({probe_profile.upper()} | Preset: {cfg['preset_name']}): {prof_desc}\n"
        f"  Bounds: [T_min={t_min:.1e}, D*={D_cap:g}] (rho={cfg['rho']:.1e}) | "
        f"D={D/D_cap:g}D*, D_S={D_S_cut/D_cap:g}D*, Gamma0={Gamma0/D_cap:.4g}D*, "
        f"GammaS0={GammaS0/D_cap:.1e}D* (GS0/G0={ratio_S_ref:.2e}), T_PG={T_PG_base/D_cap:.1e}D*, "
        f"T_S_fixed={T_S_fixed/D_cap:.3g}D*\n"
        f"  Double-Squeeze Window Analysis (T_min <= T_IR << T_low_opt << T* << T_high_opt << D_S):\n"
        + "\n".join(sq_lines) + "\n"
        + "=" * 124
    )
    print(header, flush=True)
    log_lines = [header]

    # Quick two-bath solver cross-check on first target exponent
    r_chk = float(r_list[0])
    sol_chk = build_two_bath_solver(
        r=r_chk, T_S=T_S_fixed, probe_profile=probe_profile, D_S_cutoff=D_S_cut,
        Gamma0=Gamma0, GammaS0=GammaS0, T_PG=T_PG_base, D_PG=D, t_min=t_min, d_star=D_cap,
        n_pg=300, n_s=150,
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
        f"Two-Bath Solver Cross-Check across 21 eps_d points in [-0.3D*, 0.3D*] "
        f"(M={sol_chk.dim}, r={r_chk:+.1f}, target_bath='S'):\n"
        f"  Max |FD - MatrixLyap| = {max_err_fd:.3e}, "
        f"Max |SchurLDDL - MatrixLyap| = {max_err_schur:.3e}\n"
    )
    print(chk_str, flush=True)
    log_lines.append(chk_str)
    del sol_chk
    gc.collect()

    # -------------------------------------------------------------------------
    # Build flattened list of all independent tight-binding tasks
    # -------------------------------------------------------------------------
    tasks: List[Dict[str, Any]] = []

    # (A) Core full-grid sweeps at T_S_fixed
    core_eps_grids: Dict[float, Tuple[np.ndarray, float]] = {}
    for r in r_list:
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        eff_width_T = Gamma0 * (T_S_fixed ** r) * sec_pr2 + GammaS0
        eps_max = max(25.0 * D_cap, 50.0 * eff_width_T)
        u = np.linspace(-1.0, 0.0, 161)
        sinh_scale = np.arcsinh(eps_max / (0.05 * T_S_fixed))
        eps_d_grid = (0.05 * T_S_fixed) * np.sinh(sinh_scale * u)
        core_eps_grids[r] = (eps_d_grid, eps_max)

        tasks.append({
            "kind": "core_sweep",
            "meta": {"r": r},
            "r": r,
            "T_S": T_S_fixed,
            "Gamma0": Gamma0,
            "GammaS0": GammaS0,
            "T_PG": T_PG_base,
            "D_S_cutoff": D_S_cut,
            "D_PG": D,
            "d_star": D_cap,
            "t_min": t_min,
            "probe_profile": probe_profile,
            "eps_d_grid": eps_d_grid,
        })

    # (B) Core T_S scan across [T_min, D*] (13 log-spaced points for Panel e & Main Plot A)
    T_grid = np.logspace(np.log10(ts_min), np.log10(ts_max), 13)
    T_dense = np.logspace(np.log10(ts_min), np.log10(ts_max), 75)
    assert_within_caps(T_grid, "T_grid", lower_bound=t_min, upper_bound=D_cap)

    for r in r_list:
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        for idx_t, T_val in enumerate(T_grid):
            ts = float(T_val)
            eff_w = Gamma0 * (ts ** r) * sec_pr2 + GammaS0
            eps_max_T = max(25.0 * D_cap, 50.0 * eff_w)
            tasks.append({
                "kind": "core_T_scan",
                "meta": {"r": r, "idx_t": idx_t},
                "r": r,
                "T_S": ts,
                "Gamma0": Gamma0,
                "GammaS0": GammaS0,
                "T_PG": T_PG_base,
                "D_S_cutoff": D_S_cut,
                "D_PG": D,
                "d_star": D_cap,
                "t_min": t_min,
                "probe_profile": probe_profile,
                "eps_d_grid": np.array([-eps_max_T, 0.0]),
            })

    # (C) Stress Test 1 (Panel f): Probe coupling ratio GammaS0 / Gamma0
    ratio_grid = np.logspace(np.log10(gs_ratio_min), np.log10(gs_ratio_max), 7)
    ratio_dense = np.logspace(np.log10(gs_ratio_min), np.log10(gs_ratio_max), 30)
    for r in r_list:
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        for idx_rat, rat in enumerate(ratio_grid):
            gs = float(rat) * Gamma0
            eff_w = Gamma0 * (T_S_fixed ** r) * sec_pr2 + gs
            eps_max = max(25.0 * D_cap, 50.0 * eff_w)
            tasks.append({
                "kind": "stress_gs",
                "meta": {"r": r, "idx_rat": idx_rat},
                "r": r,
                "T_S": T_S_fixed,
                "Gamma0": Gamma0,
                "GammaS0": gs,
                "T_PG": T_PG_base,
                "D_S_cutoff": D_S_cut,
                "D_PG": D,
                "d_star": D_cap,
                "t_min": t_min,
                "probe_profile": probe_profile,
                "eps_d_grid": np.array([-eps_max, 0.0]),
            })

    # (D) Stress Test 2 (Panel g): Varying power-law bath temperature T_PG
    T_PG_low, T_PG_mid, T_PG_high = [float(x) for x in cfg["T_PG_list"]]
    T_S_scan_15 = np.logspace(np.log10(ts_min), np.log10(ts_max), 15)
    T_S_subgrids = [T_S_scan_15[0::3], T_S_scan_15[1::3], T_S_scan_15[2::3]]
    T_PG_levels = [T_PG_low, T_PG_mid, T_PG_high]

    for r in r_list:
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        for level_idx, (t_pg_val, ts_sub) in enumerate(zip(T_PG_levels, T_S_subgrids)):
            for idx_t, T_val in enumerate(ts_sub):
                ts = float(T_val)
                eff_w = Gamma0 * (ts ** r) * sec_pr2 + GammaS0
                eps_max_T = max(25.0 * D_cap, 50.0 * eff_w)
                tasks.append({
                    "kind": "stress_tpg",
                    "meta": {"r": r, "level_idx": level_idx, "idx_t": idx_t},
                    "r": r,
                    "T_S": ts,
                    "Gamma0": Gamma0,
                    "GammaS0": GammaS0,
                    "T_PG": t_pg_val,
                    "D_S_cutoff": D_S_cut,
                    "D_PG": D,
                    "d_star": D_cap,
                    "t_min": t_min,
                    "probe_profile": probe_profile,
                    "eps_d_grid": np.array([-eps_max_T, 0.0]),
                })

    # (E) Stress Test 3 (Panel h): Varying probe cutoffs D_S (reuses D_S == D_S_cut from Core T_S scan)
    D_S_list = [float(ds_val) for ds_val in cfg["D_S_list"]]
    for _ds in D_S_list:
        assert_within_caps(_ds, "D_S_list element", lower_bound=t_min, upper_bound=D_cap)

    for r in r_list:
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        for ds_val in D_S_list:
            if np.isclose(ds_val, D_S_cut, rtol=1e-12, atol=1e-14):
                continue
            for idx_t, T_val in enumerate(T_grid):
                ts = float(T_val)
                eff_w = Gamma0 * (ts ** r) * sec_pr2 + GammaS0
                eps_max_T = max(25.0 * D_cap, 50.0 * eff_w)
                tasks.append({
                    "kind": "stress_ds",
                    "meta": {"r": r, "ds_val": ds_val, "idx_t": idx_t},
                    "r": r,
                    "T_S": ts,
                    "Gamma0": Gamma0,
                    "GammaS0": GammaS0,
                    "T_PG": T_PG_base,
                    "D_S_cutoff": ds_val,
                    "D_PG": D,
                    "d_star": D_cap,
                    "t_min": t_min,
                    "probe_profile": probe_profile,
                    "eps_d_grid": np.array([-eps_max_T, 0.0]),
                })

    # (F) Main Plot (B): S(0)/ln 2 vs -1 < r < 1 across double-squeeze-selected T_S values
    r_tb_B = sorted(set(float(rv) for rv in R_LIST_PANEL_B))
    for idx_ts_b, ts_val in enumerate(ts_panel_b):
        ts_fl = float(ts_val)
        for idx_r_b, r_val in enumerate(r_tb_B):
            sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r_val)
            eff_w = Gamma0 * (ts_fl ** r_val) * sec_pr2 + GammaS0
            eps_max_T = max(25.0 * D_cap, 100.0 * eff_w)
            tasks.append({
                "kind": "panel_B_tb",
                "meta": {"idx_ts_b": idx_ts_b, "idx_r_b": idx_r_b, "r": r_val},
                "r": r_val,
                "T_S": ts_fl,
                "Gamma0": Gamma0,
                "GammaS0": GammaS0,
                "T_PG": T_PG_base,
                "D_S_cutoff": D_S_cut,
                "D_PG": D,
                "d_star": D_cap,
                "t_min": t_min,
                "probe_profile": probe_profile,
                "eps_d_grid": np.array([-eps_max_T, 0.0]),
            })

    # -------------------------------------------------------------------------
    # Submit all tasks in parallel & monitor via RunTelemetry
    # -------------------------------------------------------------------------
    telemetry = RunTelemetry(
        stage_label=f"{probe_profile} | {cfg['preset_name']} | rho={cfg['rho']:.0e}",
        total_tasks=len(tasks),
        max_cpu=max_cpu,
        n_workers=n_workers,
        reserved_cpus=0 if RUN_ON_SLURM else PC_RESERVED_CPUS,
    )
    telemetry.print_banner()

    raw_sweeps: Dict[float, Dict[str, np.ndarray]] = {}
    raw_core_T: Dict[float, np.ndarray] = {r: np.zeros(len(T_grid)) for r in r_list}
    raw_stress_gs: Dict[float, np.ndarray] = {r: np.zeros(len(ratio_grid)) for r in r_list}
    raw_stress_tpg: Dict[float, List[np.ndarray]] = {
        r: [np.zeros(len(sub)) for sub in T_S_subgrids] for r in r_list
    }
    raw_stress_ds: Dict[float, Dict[float, np.ndarray]] = {
        r: {ds_val: np.zeros(len(T_grid)) for ds_val in D_S_list} for r in r_list
    }
    raw_panel_B: np.ndarray = np.zeros((len(ts_panel_b), len(r_tb_B)))

    futures = [executor.submit(_execute_benchmark_task, t) for t in tasks]
    for fut in as_completed(futures):
        res = fut.result()
        telemetry.record_task(res["duration_sec"])
        kind = res["kind"]
        meta = res["meta"]
        r = float(meta["r"])
        val_0 = float(res["tS_num_0"])

        if kind == "core_sweep":
            raw_sweeps[r] = res["sweep"]
        elif kind == "core_T_scan":
            raw_core_T[r][meta["idx_t"]] = val_0
        elif kind == "stress_gs":
            raw_stress_gs[r][meta["idx_rat"]] = val_0
        elif kind == "stress_tpg":
            raw_stress_tpg[r][meta["level_idx"]][meta["idx_t"]] = val_0
        elif kind == "stress_ds":
            raw_stress_ds[r][float(meta["ds_val"])][meta["idx_t"]] = val_0
        elif kind == "panel_B_tb":
            raw_panel_B[meta["idx_ts_b"], meta["idx_r_b"]] = val_0

    for r in r_list:
        for ds_val in D_S_list:
            if np.isclose(ds_val, D_S_cut, rtol=1e-12, atol=1e-14):
                raw_stress_ds[r][ds_val] = raw_core_T[r].copy()

    # -------------------------------------------------------------------------
    # Assemble Core Results & Evaluate Analytical Continuum Reference Curves
    # -------------------------------------------------------------------------
    results_1b: Dict[str, Any] = {
        "preset_name": cfg["preset_name"],
        "rho": cfg["rho"],
        "probe_profile": probe_profile,
        "D": D,
        "D_star": D_cap,
        "T_min": t_min,
        "Gamma0": Gamma0,
        "GammaS0": GammaS0,
        "ratio_S_ref": ratio_S_ref,
        "T_S_fixed": T_S_fixed,
        "T_PG_base": T_PG_base,
        "D_S_cut": D_S_cut,
        "T_S_range": (ts_min, ts_max),
        "GS_over_G0_range": cfg["GS_over_G0_range"],
        "double_squeeze": cfg["double_squeeze"],
        "sweeps": {},
        "T_scans": {},
        "panel_B": {},
    }
    if "d" in cfg and "k" in cfg:
        results_1b["d"] = cfg["d"]
        results_1b["k"] = cfg["k"]

    col_hdr = (
        f"{'r':>6} | {'alpha(r)':>9} | {'Prefactor':>11} | "
        f"{'~S_Lyap(0)/ln2':>15} | {'~S_Exact2B/ln2':>15} | {'~S_Eq12(0)/ln2':>15} | {'~S_Eq14(0)/ln2':>15} | "
        f"{'S_recon(0)/ln2':>15} | {'Target r':>8}"
    )
    sep = "-" * 124
    print("\n" + col_hdr, flush=True)
    print(sep, flush=True)
    log_lines.extend([col_hdr, sep])

    for r in sorted(r_list, reverse=True):
        alpha_r = alpha_func_eq14(r)
        prefactor = ratio_S_ref * alpha_r * (T_S_fixed ** (-r))
        eps_d_grid, eps_max = core_eps_grids[r]

        sweep = raw_sweeps[r]
        tS_num_0 = float(sweep["S_eps_d"][-1])
        tS_exact_curve = eval_analytic_two_bath(
            eps_d_grid, T_S_fixed, Gamma0, GammaS0, r,
            probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D, d_star=D_cap,
            eps_max=eps_max,
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

    for r in r_list:
        alpha_r = alpha_func_eq14(r)
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        tS_num_T = raw_core_T[r]
        tS_exact_T = np.zeros(len(T_grid))
        S_recon_num_T = np.zeros(len(T_grid))
        S_recon_exact_T = np.zeros(len(T_grid))

        for idx_t, T_val in enumerate(T_grid):
            ts = float(T_val)
            pref = ratio_S_ref * alpha_r * (ts ** (-r))
            eff_w = Gamma0 * (ts ** r) * sec_pr2 + GammaS0
            eps_max_T = max(25.0 * D_cap, 50.0 * eff_w)
            val_ex = float(eval_analytic_two_bath(
                0.0, ts, Gamma0, GammaS0, r,
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D, d_star=D_cap,
                eps_max=eps_max_T,
            ))
            tS_exact_T[idx_t] = val_ex
            S_recon_num_T[idx_t] = (tS_num_T[idx_t] / LN2) / pref
            S_recon_exact_T[idx_t] = (val_ex / LN2) / pref

        tS_exact_dense = np.array([
            float(eval_analytic_two_bath(
                0.0, float(t), Gamma0, GammaS0, r,
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D, d_star=D_cap,
                eps_max=max(25.0 * D_cap, 50.0 * (Gamma0 * (float(t) ** r) * sec_pr2 + GammaS0)),
            ))
            for t in T_dense
        ])
        tS_eq12_dense = np.array([analytic_tilde_S0_eq12(float(t), Gamma0, GammaS0, r) for t in T_dense])
        tS_eq14_dense = np.array([analytic_tilde_S0_eq14(float(t), Gamma0, GammaS0, r) for t in T_dense])
        pref_dense = ratio_S_ref * alpha_r * (T_dense ** (-r))

        results_1b["T_scans"][r] = {
            "T_grid": T_grid,
            "tS_num": tS_num_T,
            "tS_exact": tS_exact_T,
            "S_recon_num": S_recon_num_T,
            "S_recon_exact": S_recon_exact_T,
            "T_dense": T_dense,
            "tS_exact_dense": tS_exact_dense,
            "tS_eq12_dense": tS_eq12_dense,
            "tS_eq14_dense": tS_eq14_dense,
            "S_recon_exact_dense": (tS_exact_dense / LN2) / pref_dense,
            "S_recon_eq12_dense": (tS_eq12_dense / LN2) / pref_dense,
            "S_recon_eq14_dense": (tS_eq14_dense / LN2) / pref_dense,
        }

    # Assemble Main Plot (B) data across -1 < r < 1
    r_dense_B = np.linspace(-0.90, 0.90, 121)
    alpha_tb_B = np.array([alpha_func_eq14(rv) for rv in r_tb_B])
    alpha_dense_B = np.array([alpha_func_eq14(float(rv)) for rv in r_dense_B])
    panel_B_curves: Dict[float, Dict[str, np.ndarray]] = {}

    for idx_ts_b, ts_val in enumerate(ts_panel_b):
        ts_fl = float(ts_val)
        prefs_tb = ratio_S_ref * alpha_tb_B * (ts_fl ** (-np.array(r_tb_B)))
        s_recon_tb_B = (raw_panel_B[idx_ts_b] / LN2) / prefs_tb

        s_recon_ex_B = np.zeros(len(r_dense_B))
        for idx_rd, rv in enumerate(r_dense_B):
            r_fl = float(rv)
            sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r_fl)
            eff_w = Gamma0 * (ts_fl ** r_fl) * sec_pr2 + GammaS0
            eps_max_T = max(25.0 * D_cap, 100.0 * eff_w)
            val_ex = float(eval_analytic_two_bath(
                0.0, ts_fl, Gamma0, GammaS0, r_fl,
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D, d_star=D_cap,
                eps_max=eps_max_T,
            ))
            pref_d = ratio_S_ref * alpha_dense_B[idx_rd] * (ts_fl ** (-r_fl))
            s_recon_ex_B[idx_rd] = (val_ex / LN2) / pref_d

        panel_B_curves[ts_fl] = {
            "T_S": ts_fl,
            "r_tb": np.array(r_tb_B, dtype=float),
            "S_recon_tb": s_recon_tb_B,
            "r_dense": r_dense_B,
            "S_recon_exact": s_recon_ex_B,
        }
    results_1b["panel_B"] = {
        "ts_list": [float(x) for x in ts_panel_b],
        "curves": panel_B_curves,
    }
    results_1b["log_lines"] = log_lines

    # -------------------------------------------------------------------------
    # Assemble Stress-Test Results
    # -------------------------------------------------------------------------
    stress_hdr = (
        "\n" + "=" * 112 + "\n"
        f"STEP 1b (STRESS TESTS — {probe_profile.upper()} | Preset: {cfg['preset_name']}): "
        "Weak-Coupling Sweep, Finite T_PG, and Probe Cutoffs D_S\n"
        + "=" * 112
    )
    print(stress_hdr, flush=True)
    stress_log = [stress_hdr]

    gamma_s_sweeps: Dict[float, Dict[str, np.ndarray]] = {}
    hdr_f = f"{'r':>6} | " + " | ".join([f"GS/G0={rat:.1e}" for rat in ratio_grid[::2]]) + " | Target r"
    print(hdr_f, flush=True)
    print("-" * len(hdr_f), flush=True)
    stress_log.extend([hdr_f, "-" * len(hdr_f)])

    for r in sorted(r_list, reverse=True):
        alpha_r = alpha_func_eq14(r)
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        prefs_grid = ratio_grid * alpha_r * (T_S_fixed ** (-r))
        recon_num_arr = (raw_stress_gs[r] / LN2) / prefs_grid

        recon_ex_dense = []
        for rat in ratio_dense:
            gs = float(rat) * Gamma0
            pref = float(rat) * alpha_r * (T_S_fixed ** (-r))
            eff_w = Gamma0 * (T_S_fixed ** r) * sec_pr2 + gs
            eps_max = max(25.0 * D_cap, 50.0 * eff_w)
            val_ex = float(eval_analytic_two_bath(
                0.0, T_S_fixed, Gamma0, gs, r,
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D, d_star=D_cap,
                eps_max=eps_max,
            ))
            recon_ex_dense.append((val_ex / LN2) / pref)

        gamma_s_sweeps[r] = {
            "ratio_grid": ratio_grid,
            "S_recon_num": recon_num_arr,
            "ratio_dense": ratio_dense,
            "S_recon_exact_dense": np.array(recon_ex_dense),
        }
        row_f = f"{r:6.2f} | " + " | ".join([f"{val:13.5f}" for val in recon_num_arr[::2]]) + f" | {r:8.4f}"
        print(row_f, flush=True)
        stress_log.append(row_f)

    finite_T_pg_results: Dict[str, Any] = {
        "T_PG_low": T_PG_low,
        "T_PG_mid": T_PG_mid,
        "T_PG_high": T_PG_high,
        "D_S_cut": D_S_cut,
        "by_r": {},
    }
    for r in r_list:
        alpha_r = alpha_func_eq14(r)
        recon_by_lvl = []
        for lvl_idx, ts_sub in enumerate(T_S_subgrids):
            prefs = ratio_S_ref * alpha_r * (ts_sub ** (-r))
            recon_by_lvl.append((raw_stress_tpg[r][lvl_idx] / LN2) / prefs)

        finite_T_pg_results["by_r"][r] = {
            "T_S_low_grid": T_S_subgrids[0],
            "T_S_mid_grid": T_S_subgrids[1],
            "T_S_high_grid": T_S_subgrids[2],
            "T_S_dense": T_dense,
            "recon_tpg_0": recon_by_lvl[0],
            "recon_tpg_mid": recon_by_lvl[1],
            "recon_tpg_high": recon_by_lvl[2],
            "recon_ana_curve": results_1b["T_scans"][r]["S_recon_exact_dense"],
            "recon_eq12_curve": results_1b["T_scans"][r]["S_recon_eq12_dense"],
        }

    probe_profile_results: Dict[str, Any] = {
        "D_S_list": D_S_list,
        "by_r": {},
    }
    for r in r_list:
        alpha_r = alpha_func_eq14(r)
        sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r)
        prefs_grid = ratio_S_ref * alpha_r * (T_grid ** (-r))
        prefs_dense = ratio_S_ref * alpha_r * (T_dense ** (-r))

        num_by_ds = {
            ds_val: (raw_stress_ds[r][ds_val] / LN2) / prefs_grid
            for ds_val in D_S_list
        }
        ana_by_ds: Dict[float, np.ndarray] = {}
        for ds_val in D_S_list:
            vals = []
            for t in T_dense:
                ts = float(t)
                eff_w = Gamma0 * (ts ** r) * sec_pr2 + GammaS0
                eps_max_T = max(25.0 * D_cap, 50.0 * eff_w)
                vals.append(float(eval_analytic_two_bath(
                    0.0, ts, Gamma0, GammaS0, r,
                    probe_profile=probe_profile, D_S_cutoff=ds_val, D_PG=D, d_star=D_cap,
                    eps_max=eps_max_T,
                )))
            ana_by_ds[ds_val] = (np.array(vals) / LN2) / prefs_dense

        probe_profile_results["by_r"][r] = {
            "T_S_scan": T_grid,
            "T_S_dense": T_dense,
            "num_by_ds": num_by_ds,
            "ana_by_ds": ana_by_ds,
            "eq12_curve": results_1b["T_scans"][r]["S_recon_eq12_dense"],
        }

    res_stress = {
        "gamma_s_sweeps": gamma_s_sweeps,
        "finite_T_pg": finite_T_pg_results,
        "probe_profiles": probe_profile_results,
        "log_lines": stress_log,
    }
    telemetry_summary = telemetry.summarize()
    print(flush=True)
    return results_1b, res_stress, telemetry_summary


# ============================================================================
# 8. MODULAR KC-STYLED PANEL RENDERERS (PDF ONLY)
# ============================================================================
def _kc_font_sizes(standalone: bool) -> Dict[str, float]:
    """Return KC-proportioned font sizes (tick < axis label +25% < bold panel tag)."""
    if standalone:
        return {"tick": 14.0, "label": 17.5, "panel_tag": 24.0, "title": 16.0, "legend": 11.5}
    return {"tick": 10.5, "label": 13.2, "panel_tag": 17.0, "title": 11.2, "legend": 8.0}


def _apply_kc_axes_polish(ax: plt.Axes, fs: Dict[str, float]) -> None:
    """Enforce 0.5 pt spines/ticks and standardized tick label sizes on `ax`."""
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(which="both", width=0.5, labelsize=fs["tick"])
    ax.grid(True, which="major", linewidth=0.4, alpha=0.22)


def draw_panel_a(ax: plt.Axes, res_core: dict, colors: dict, standalone: bool = False) -> None:
    fs = _kc_font_sizes(standalone)
    all_r = sorted(res_core["sweeps"].keys())
    D, D_cap = res_core["D"], res_core["D_star"]
    T_S_fixed, T_PG_base = res_core["T_S_fixed"], res_core["T_PG_base"]
    ratio_S_ref, D_S_cut = res_core["ratio_S_ref"], res_core["D_S_cut"]
    probe_short = f"Flat $D_S={D_S_cut/D_cap:g}D^*$" if res_core["probe_profile"] == "flat_band" else f"Lorentzian $D_S={D_S_cut/D_cap:g}D^*$"

    min_y = 0.0
    for r in all_r:
        sw = res_core["sweeps"][r]
        y_vals = sw["dn_dT"] * ((T_S_fixed ** r) / ratio_S_ref)
        min_y = min(min_y, float(np.min(y_vals)))
        ax.plot(sw["eps_d"] / D, y_vals, "-", color=colors[r], lw=1.8)
    xlim_a = -min(D_cap / D, max(60.0 * (T_S_fixed / D), 0.06 * (D_cap / D)))
    ax.set_xlim(xlim_a, 0.0)
    ax.set_ylim(1.12 * min_y if min_y < 0 else -90.0, max(4.0, -0.05 * min_y))
    ax.set_xlabel(r"Impurity Energy $\epsilon_d / D$", fontsize=fs["label"])
    ax.set_ylabel(r"Rescaled $\frac{\Gamma_0}{\Gamma_{S,0}} T_S^r \frac{\partial n_d}{\partial T_S}$", fontsize=fs["label"])
    ax.set_title(f"(a) Step 1b: $\\partial n_d / \\partial T_S$ ($T_S/D={T_S_fixed/D:.3g}, T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)", fontsize=fs["title"])
    handles = [Line2D([0], [0], color="k", ls="-", lw=1.8, label=f"Tight-Binding Lyapunov ($N=14\\mathrm{{k}}$, {probe_short})")]
    if standalone:
        handles += [Line2D([0], [0], color=colors[r], lw=2.2, label=f"$r = {r:+.1f}$") for r in all_r]
    ax.legend(handles=handles, frameon=True, facecolor="white", fontsize=fs["legend"], loc="lower left")
    _apply_kc_axes_polish(ax, fs)


def draw_panel_b(ax: plt.Axes, res_core: dict, colors: dict, standalone: bool = False) -> None:
    fs = _kc_font_sizes(standalone)
    all_r = sorted(res_core["sweeps"].keys())
    D, D_cap = res_core["D"], res_core["D_star"]
    T_S_fixed, T_PG_base = res_core["T_S_fixed"], res_core["T_PG_base"]
    s_ylim = (-0.15, 1.08) if min(all_r) >= 0.0 else (-1.05, 1.08)

    for r in all_r:
        sw = res_core["sweeps"][r]
        norm = sw["prefactor"] * LN2
        ax.plot(sw["eps_d"] / D, sw["tS_exact_curve"] / norm, "--", color="k", lw=1.1, alpha=0.65)
        ax.plot(sw["eps_d"] / D, sw["S_eps_d"] / norm, "-", color=colors[r], lw=1.8)
        ax.axhline(r, color=colors[r], ls=":", lw=1.0, alpha=0.65)
    ax.set_xlim(-D_cap / D, 0.0)
    ax.set_ylim(*s_ylim)
    ax.set_xlabel(r"Impurity Energy $\epsilon_d / D$", fontsize=fs["label"])
    ax.set_ylabel(r"Rescaled $\tilde{S}(\epsilon_d) / [\frac{\Gamma_{S,0}}{\Gamma_0}\alpha(r)T_S^{-r}\ln 2]$", fontsize=fs["label"])
    ax.set_title(f"(b) Step 1b: Integral $\\tilde{{S}}(\\epsilon_d)$ ($T_S/D={T_S_fixed/D:.3g}, T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)", fontsize=fs["title"])
    handles = [
        Line2D([0], [0], color="k", ls="-", lw=1.8, label=r"Tight-Binding Lyapunov ($-\int \partial_{T_S}n_d\,d\epsilon_d'$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.1, label=r"Exact Finite-Band Integral (Eq. 11)"),
        Line2D([0], [0], color="k", ls=":", lw=1.0, label=r"Low-$T_S$ Asymptote: $\tilde{S}(0) \to r$ (Eq. 14)"),
    ]
    if standalone:
        handles += [Line2D([0], [0], color=colors[r], lw=2.2, label=f"$r = {r:+.1f}$") for r in all_r]
    ax.legend(handles=handles, frameon=True, facecolor="white", fontsize=fs["legend"] * 0.90, loc="upper left")
    _apply_kc_axes_polish(ax, fs)

    xlim_a = -min(D_cap / D, max(60.0 * (T_S_fixed / D), 0.06 * (D_cap / D)))
    ax_ins = ax.inset_axes([0.14, 0.46, 0.45, 0.30])
    for r in all_r:
        sw = res_core["sweeps"][r]
        norm = sw["prefactor"] * LN2
        ax_ins.plot(sw["eps_d"] / D, sw["tS_exact_curve"] / norm, "--", color="k", lw=1.0, alpha=0.65)
        ax_ins.plot(sw["eps_d"] / D, sw["S_eps_d"] / norm, "-", color=colors[r], lw=1.5)
        ax_ins.axhline(r, color=colors[r], ls=":", lw=0.9, alpha=0.55)
    ax_ins.set_xlim(xlim_a, 0.0)
    ax_ins.set_ylim(-0.05, 0.65)
    ax_ins.set_title(r"Zoom near $\epsilon_d = 0$", fontsize=fs["tick"] * 0.85)
    for sp in ax_ins.spines.values():
        sp.set_linewidth(0.5)
    ax_ins.tick_params(labelsize=fs["tick"] * 0.8, width=0.5)
    ax_ins.grid(True, linewidth=0.4, alpha=0.22)


def draw_panel_c(ax: plt.Axes, res_core: dict, colors: dict, standalone: bool = False) -> None:
    fs = _kc_font_sizes(standalone)
    all_r = sorted(res_core["sweeps"].keys())
    D, D_cap, ratio_S_ref, D_S_cut, T_PG_base = res_core["D"], res_core["D_star"], res_core["ratio_S_ref"], res_core["D_S_cut"], res_core["T_PG_base"]
    ts_min, ts_max = res_core["T_S_range"]

    for r in all_r:
        if np.isclose(r, 0.0):
            continue
        tdata = res_core["T_scans"][r]
        ax.loglog(tdata["T_dense"] / D, np.abs(tdata["tS_eq14_dense"]) / (ratio_S_ref * LN2), ":", color=colors[r], lw=1.2)
        ax.loglog(tdata["T_dense"] / D, np.abs(tdata["tS_eq12_dense"]) / (ratio_S_ref * LN2), "--", color=colors[r], lw=1.2)
        ax.loglog(tdata["T_dense"] / D, np.abs(tdata["tS_exact_dense"]) / (ratio_S_ref * LN2), "-", color=colors[r], lw=1.6)
        ax.loglog(tdata["T_grid"] / D, np.abs(tdata["tS_num"]) / (ratio_S_ref * LN2), "o", color=colors[r], ms=5.5)
    ax.set_xlim(ts_min / D, ts_max / D)
    ax.set_ylim(1e-2, 1e2)
    ax.set_xlabel(r"Probe Temperature $T_S / D$ (log scale)", fontsize=fs["label"])
    ax.set_ylabel(r"Scaled Magnitude $|\tilde{S}(0)| / (\frac{\Gamma_{S,0}}{\Gamma_0}\ln 2)$", fontsize=fs["label"])
    ax.set_title(f"(c) Step 1b: Scaling $|\\tilde{{S}}(0)| \\propto T_S^{{-r}}$ ($D_S={D_S_cut/D_cap:g}D^*, T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)", fontsize=fs["title"])
    handles = [
        Line2D([0], [0], color="k", marker="o", ls="None", ms=5, label=r"Tight-Binding Lyapunov ($N=14\mathrm{k}$)"),
        Line2D([0], [0], color="k", ls="-", lw=1.6, label=f"Exact Finite-Band Integral (Eq. 11, $D_S={D_S_cut/D_cap:g}D^*$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.2, label=r"Unbounded Note Limit (Eq. 12, $D_S\to\infty$)"),
        Line2D([0], [0], color="k", ls=":", lw=1.2, label=r"Low-$T_S$ Power Law $|r|\alpha(r)T_S^{-r}$ (Eq. 14)"),
    ]
    if standalone:
        handles += [Line2D([0], [0], color=colors[r], lw=2.2, label=f"$r = {r:+.1f}$") for r in all_r]
    ax.legend(handles=handles, frameon=True, facecolor="white", fontsize=fs["legend"] * 0.88, loc="upper right")
    _apply_kc_axes_polish(ax, fs)


def draw_panel_d(ax: plt.Axes, res_core: dict, colors: dict, standalone: bool = False) -> None:
    fs = _kc_font_sizes(standalone)
    all_r = sorted(res_core["sweeps"].keys())
    D, T_S_fixed, T_PG_base = res_core["D"], res_core["T_S_fixed"], res_core["T_PG_base"]
    r_vals = np.array(all_r)
    s_rec_num = np.array([res_core["sweeps"][r]["S_recon_num"] for r in r_vals])
    s_rec_ex = np.array([res_core["sweeps"][r]["S_recon_exact"] for r in r_vals])

    r_cont = np.linspace(0.0, 0.98, 100)
    ax.plot(r_cont, r_cont, "k--", lw=1.6)
    ax.plot(r_vals, s_rec_ex, "s", color="#444444", fillstyle="none", ms=8.0, mew=1.4)
    for r, s_val in zip(r_vals, s_rec_num):
        ax.plot(r, s_val, "o", color=colors[r], ms=7.5, markeredgecolor="k", markeredgewidth=0.5)
    ax.set_xlim(0.0, 0.85)
    ax.set_ylim(0.0, 0.85)
    ax.set_xlabel(r"Power-Law Exponent $r$", fontsize=fs["label"])
    ax.set_ylabel(r"Reconstructed Entropy $S_{\mathrm{recon}}(0) / \ln 2$", fontsize=fs["label"])
    ax.set_title(f"(d) Step 1b: Reconstructed $S(0)/\\ln 2$ vs $r$ ($T_S/D={T_S_fixed/D:.3g}, T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)", fontsize=fs["title"])
    handles = [
        Line2D([0], [0], color="k", ls="--", lw=1.6, label=r"Low-$T_S$ Linear Law: $S_{\mathrm{recon}}(0)/\ln 2 = r$ (Eq. 14)"),
        Line2D([0], [0], color="#444444", marker="s", ls="None", fillstyle="none", ms=7.5, mew=1.4, label=r"Exact Finite-Band Integral (Eq. 11)"),
        Line2D([0], [0], color="k", marker="o", ls="None", ms=7.0, label=f"Tight-Binding Lyapunov ($T_S/D = {T_S_fixed/D:.3g}$)"),
    ]
    ax.legend(handles=handles, frameon=True, facecolor="white", fontsize=fs["legend"] * 0.90, loc="upper left")
    _apply_kc_axes_polish(ax, fs)

    ax_ins = ax.inset_axes([0.55, 0.12, 0.40, 0.34])
    r_alpha_grid = np.linspace(0.0, 0.95, 60)
    ax_ins.plot(r_alpha_grid, [alpha_func_eq14(float(rv)) for rv in r_alpha_grid], "k-", lw=1.4)
    for r in all_r:
        ax_ins.plot(r, res_core["sweeps"][r]["alpha_r"], "o", color=colors[r], ms=4.8)
    ax_ins.set_title(r"Prefactor $\alpha(r)$ (Eq. 14)", fontsize=fs["tick"] * 0.85)
    ax_ins.set_xlabel(r"$r$", fontsize=fs["tick"] * 0.85)
    for sp in ax_ins.spines.values():
        sp.set_linewidth(0.5)
    ax_ins.tick_params(labelsize=fs["tick"] * 0.8, width=0.5)
    ax_ins.grid(True, linewidth=0.4, alpha=0.22)


def draw_panel_e(ax: plt.Axes, res_core: dict, colors: dict, standalone: bool = False) -> None:
    fs = _kc_font_sizes(standalone)
    all_r = sorted(res_core["sweeps"].keys())
    D, D_cap, T_S_fixed, D_S_cut = res_core["D"], res_core["D_star"], res_core["T_S_fixed"], res_core["D_S_cut"]
    ts_min, ts_max = res_core["T_S_range"]
    s_ylim = (-0.15, 1.08) if min(all_r) >= 0.0 else (-1.05, 1.08)

    for r in all_r:
        tdata = res_core["T_scans"][r]
        ax.semilogx(tdata["T_dense"] / D, tdata["S_recon_eq12_dense"], "--", color=colors[r], lw=1.3, alpha=0.75)
        ax.semilogx(tdata["T_dense"] / D, tdata["S_recon_exact_dense"], "-", color=colors[r], lw=1.8)
        ax.semilogx(tdata["T_grid"] / D, tdata["S_recon_num"], "o", color=colors[r], ms=5.5, markeredgecolor="k", markeredgewidth=0.4)
        ax.axhline(r, color=colors[r], ls=":", lw=1.0, alpha=0.55)
    ax.axhline(1.0, color="dimgray", ls="-.", lw=1.0, alpha=0.6)
    ax.axvline(T_S_fixed / D, color="dimgray", ls=":", lw=1.0, alpha=0.7)
    ax.set_xlim(ts_min / D, ts_max / D)
    ax.set_ylim(*s_ylim)
    ax.set_xlabel(r"Probe Temperature $T_S / D$", fontsize=fs["label"])
    ax.set_ylabel(r"Reconstructed Entropy $S_{\mathrm{recon}}(0) / \ln 2$", fontsize=fs["label"])
    ax.set_title(f"(e) Step 1b: $S_{{\\mathrm{{recon}}}}(0)/\\ln 2$ vs $T_S/D$ ($D={D/D_cap:g}D^*, D_S={D_S_cut/D_cap:g}D^*$)", fontsize=fs["title"])
    handles = [
        Line2D([0], [0], color="k", marker="o", ls="-", lw=1.7, ms=5, label=f"Tight-Binding & Finite-Band (Eq. 11, $D_S={D_S_cut/D_cap:g}D^*$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.3, label=r"Unbounded Note Limit (Eq. 12: $D_S\to\infty, \Gamma_S\to 0$)"),
        Line2D([0], [0], color="k", ls=":", lw=1.0, label=r"Asymptotes: Low-$T_S$ ($r$, Eq. 14) & High-$T_S$ ($+1$)"),
    ]
    if standalone:
        handles += [Line2D([0], [0], color=colors[r], lw=2.2, label=f"$r = {r:+.1f}$") for r in all_r]
    ax.legend(handles=handles, frameon=True, facecolor="white", fontsize=fs["legend"] * 0.85, loc="upper left")
    _apply_kc_axes_polish(ax, fs)


def draw_panel_f(ax: plt.Axes, res_core: dict, res_stress: dict, colors: dict, standalone: bool = False) -> None:
    fs = _kc_font_sizes(standalone)
    all_r = sorted(res_core["sweeps"].keys())
    D, T_S_fixed, T_PG_base, ratio_S_ref = res_core["D"], res_core["T_S_fixed"], res_core["T_PG_base"], res_core["ratio_S_ref"]
    gs_ratio_min, gs_ratio_max = res_core["GS_over_G0_range"]
    s_ylim = (-0.15, 1.08) if min(all_r) >= 0.0 else (-1.05, 1.08)

    for r in all_r:
        gdata = res_stress["gamma_s_sweeps"][r]
        ax.semilogx(gdata["ratio_dense"], gdata["S_recon_exact_dense"], "--", color=colors[r], lw=1.5)
        ax.semilogx(gdata["ratio_grid"], gdata["S_recon_num"], "o", color=colors[r], ms=5.5, markeredgecolor="k", markeredgewidth=0.4)
        ax.axhline(r, color=colors[r], ls=":", lw=1.0, alpha=0.55)
    ax.axvline(ratio_S_ref, color="dimgray", ls="-.", lw=1.0, alpha=0.7)
    ax.set_xlim(gs_ratio_min, gs_ratio_max)
    ax.set_ylim(*s_ylim)
    ax.set_xlabel(r"Probe Coupling Ratio $\Gamma_{S,0} / \Gamma_0$ (log scale)", fontsize=fs["label"])
    ax.set_ylabel(r"Reconstructed Entropy $S_{\mathrm{recon}}(0) / \ln 2$", fontsize=fs["label"])
    ax.set_title(f"(f) Test 1: $\\Gamma_S \\ll \\Gamma(\\omega)$ Breakdown ($T_S/D={T_S_fixed/D:.3g}, T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)", fontsize=fs["title"])
    handles = [
        Line2D([0], [0], color="k", marker="o", ls="None", ms=5, label=r"Tight-Binding Lyapunov ($N=14\mathrm{k}$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.5, label=r"Exact Finite-Band Integral (Eq. 11)"),
        Line2D([0], [0], color="k", ls=":", lw=1.0, label=r"Weak-Probe Limit: $\Gamma_S \to 0$ (Eq. 14)"),
        Line2D([0], [0], color="dimgray", ls="-.", lw=1.0, label=f"Default $\\Gamma_{{S,0}}/\\Gamma_0 = {ratio_S_ref:.2e}$"),
    ]
    if standalone:
        handles += [Line2D([0], [0], color=colors[r], lw=2.2, label=f"$r = {r:+.1f}$") for r in all_r]
    ax.legend(handles=handles, frameon=True, facecolor="white", fontsize=fs["legend"] * 0.88, loc="upper left")
    _apply_kc_axes_polish(ax, fs)


def draw_panel_g(ax: plt.Axes, res_core: dict, res_stress: dict, colors: dict, standalone: bool = False) -> None:
    """Diagnostic Panel (g): Varies bath temperature T_PG/D across {rho D*/D, rho^(2/3) D*/D, rho^(1/3) D*/D}."""
    fs = _kc_font_sizes(standalone)
    all_r = sorted(res_core["sweeps"].keys())
    D, D_cap, D_S_cut = res_core["D"], res_core["D_star"], res_core["D_S_cut"]
    ts_min, ts_max = res_core["T_S_range"]
    s_ylim = (-0.15, 1.08) if min(all_r) >= 0.0 else (-1.05, 1.08)

    T_PG_low = res_stress["finite_T_pg"]["T_PG_low"]
    T_PG_mid = res_stress["finite_T_pg"]["T_PG_mid"]
    T_PG_high = res_stress["finite_T_pg"]["T_PG_high"]

    for r, tpg_data in res_stress["finite_T_pg"]["by_r"].items():
        ax.semilogx(tpg_data["T_S_dense"] / D, tpg_data["recon_eq12_curve"], "--", color=colors[r], lw=1.2, alpha=0.65)
        ax.semilogx(tpg_data["T_S_dense"] / D, tpg_data["recon_ana_curve"], "-", color=colors[r], lw=1.8)
        ax.semilogx(
            tpg_data["T_S_low_grid"] / D, tpg_data["recon_tpg_0"],
            "o", color=colors[r], ms=5.8, markeredgecolor="k", markeredgewidth=0.4,
        )
        ax.semilogx(
            tpg_data["T_S_mid_grid"] / D, tpg_data["recon_tpg_mid"],
            "s", color=colors[r], ms=6.5, fillstyle="none", mew=1.4,
        )
        ax.semilogx(
            tpg_data["T_S_high_grid"] / D, tpg_data["recon_tpg_high"],
            "^", color=colors[r], ms=6.8, fillstyle="none", mew=1.4,
        )
        ax.axhline(r, color=colors[r], ls=":", lw=1.0, alpha=0.55)
    ax.axhline(1.0, color="dimgray", ls="-.", lw=1.0, alpha=0.6)
    ax.axvline(T_PG_mid / D, color="dimgray", ls="--", lw=1.0, alpha=0.65)
    ax.axvline(T_PG_high / D, color="k", ls="-.", lw=1.0, alpha=0.65)
    ax.set_xlim(ts_min / D, ts_max / D)
    ax.set_ylim(*s_ylim)
    ax.set_xlabel(r"Probe Temperature $T_S / D$", fontsize=fs["label"])
    ax.set_ylabel(r"Reconstructed Entropy $S_{\mathrm{recon}}(0) / \ln 2$", fontsize=fs["label"])
    ax.set_title(
        f"(g) Test 2: Varying Bath Temp $T_{{\\mathrm{{PG}}}}/D \\in "
        f"\\{{{T_PG_low/D:.3g}, {T_PG_mid/D:.3g}, {T_PG_high/D:.3g}\\}}$ ($D_S={D_S_cut/D_cap:g}D^*$)",
        fontsize=fs["title"],
    )
    handles = [
        Line2D([0], [0], color="k", ls="-", lw=1.8, label=f"Exact Finite-Band (Eq. 11, $D_S={D_S_cut/D_cap:g}D^*$, indep. of $T_{{\\mathrm{{PG}}}}$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.2, label=r"Unbounded Note Limit (Eq. 12, $D_S\to\infty$)"),
        Line2D([0], [0], color="k", marker="o", ls="None", ms=5.5, label=f"Tight-Binding: $T_{{\\mathrm{{PG}}}}/D = {T_PG_low/D:.3g}$"),
        Line2D([0], [0], color="k", marker="s", ls="None", ms=6.2, fillstyle="none", mew=1.4, label=f"Tight-Binding: $T_{{\\mathrm{{PG}}}}/D = {T_PG_mid/D:.3g}$"),
        Line2D([0], [0], color="k", marker="^", ls="None", ms=6.5, fillstyle="none", mew=1.4, label=f"Tight-Binding: $T_{{\\mathrm{{PG}}}}/D = {T_PG_high/D:.3g}$"),
    ]
    if standalone:
        handles += [Line2D([0], [0], color=colors[r], lw=2.2, label=f"$r = {r:+.1f}$") for r in all_r]
    ax.legend(handles=handles, frameon=True, facecolor="white", fontsize=fs["legend"] * 0.85, loc="upper left")
    _apply_kc_axes_polish(ax, fs)


def draw_panel_h(ax: plt.Axes, res_core: dict, res_stress: dict, colors: dict, standalone: bool = False) -> None:
    fs = _kc_font_sizes(standalone)
    all_r = sorted(res_core["sweeps"].keys())
    D, D_cap, T_PG_base = res_core["D"], res_core["D_star"], res_core["T_PG_base"]
    ts_min, ts_max = res_core["T_S_range"]
    s_ylim = (-0.15, 1.08) if min(all_r) >= 0.0 else (-1.05, 1.08)

    D_S_list = res_stress["probe_profiles"]["D_S_list"]
    base_styles = [
        ("-",  "o", 1.8, True),
        ("--", "s", 1.5, False),
        ("-.", "d", 1.4, False),
        (":",  "^", 1.4, False),
        ("-",  "v", 1.3, False),
    ]
    ds_desc = sorted(D_S_list, reverse=True)
    ds_styles = [
        (ds_val, *base_styles[idx % len(base_styles)], f"Cutoff $D_S = {ds_val/D_cap:.3g}D^*$")
        for idx, ds_val in enumerate(ds_desc)
    ]
    for r, pdata in res_stress["probe_profiles"]["by_r"].items():
        ax.semilogx(pdata["T_S_dense"] / D, pdata["eq12_curve"], "-", color=colors[r], lw=0.9, alpha=0.30)
        for ds_val, ls_v, mk_v, lw_v, filled, _ in ds_styles:
            ax.semilogx(pdata["T_S_dense"] / D, pdata["ana_by_ds"][ds_val], ls=ls_v, color=colors[r], lw=lw_v)
            ax.semilogx(
                pdata["T_S_scan"] / D, pdata["num_by_ds"][ds_val],
                marker=mk_v, ls="None", color=colors[r], ms=5.2,
                fillstyle="full" if filled else "none", mew=1.3,
            )
        ax.axhline(r, color=colors[r], ls=":", lw=0.9, alpha=0.40)

    ax.axhline(1.0, color="dimgray", ls="-.", lw=1.0, alpha=0.6)
    for ds_val, ls_v, _, _, _, _ in ds_styles[1:]:
        ax.axvline(ds_val / D, color="dimgray", ls=ls_v, lw=1.0, alpha=0.60)
    ax.set_xlim(ts_min / D, ts_max / D)
    ax.set_ylim(*s_ylim)
    ax.set_xlabel(r"Probe Temperature $T_S / D$", fontsize=fs["label"])
    ax.set_ylabel(r"Reconstructed Entropy $S_{\mathrm{recon}}(0) / \ln 2$", fontsize=fs["label"])
    ds_str_list = ", ".join(f"{ds/D_cap:.3g}" for ds in ds_desc)
    ax.set_title(f"(h) Test 3: Probe Cutoffs $D_S/D^* \\in \\{{{ds_str_list}\\}}$ ($T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)", fontsize=fs["title"])
    handles = [
        Line2D([0], [0], color="k", marker=mk_v, ls=ls_v, ms=5.0, lw=lw_v,
               fillstyle="full" if filled else "none", mew=1.3, label=lbl)
        for _, ls_v, mk_v, lw_v, filled, lbl in ds_styles
    ] + [
        Line2D([0], [0], color="dimgray", ls="--", lw=1.0, label=r"Vertical Lines: Mode Cutoff $T_S = D_S$"),
    ]
    if standalone:
        handles += [Line2D([0], [0], color=colors[r], lw=2.2, label=f"$r = {r:+.1f}$") for r in all_r]
    ax.legend(handles=handles, frameon=True, facecolor="white", fontsize=fs["legend"] * 0.85, loc="upper left")
    _apply_kc_axes_polish(ax, fs)


def draw_panel_A(ax: plt.Axes, res_core: dict, colors: dict, standalone: bool = False) -> None:
    """
    Main Plot (A): Entropy S(0)/ln 2 vs. Probe Temperature T_S/D (13 log-spaced points)
    at fixed T_PG = T_min (matching Plot B).
    """
    fs = _kc_font_sizes(standalone)
    all_r = sorted(res_core["sweeps"].keys())
    D = res_core["D"]
    ts_min, ts_max = res_core["T_S_range"]
    s_ylim = (-0.15, 1.08) if min(all_r) >= 0.0 else (-1.05, 1.08)

    for r in all_r:
        tdata = res_core["T_scans"][r]
        ax.semilogx(tdata["T_dense"] / D, tdata["S_recon_exact_dense"], "-", color=colors[r], lw=1.8)
        ax.semilogx(
            tdata["T_grid"] / D, tdata["S_recon_num"], "o", color=colors[r],
            ms=6.0, markeredgecolor="k", markeredgewidth=0.5, zorder=4,
        )
        ax.axhline(r, color=colors[r], ls=":", lw=1.0, alpha=0.55)

    ax.axhline(1.0, color="dimgray", ls="-.", lw=1.0, alpha=0.6)
    ax.set_xlim(ts_min / D, ts_max / D)
    ax.set_ylim(*s_ylim)
    ax.set_xlabel(r"Probe Temperature $T_S / D$", fontsize=fs["label"])
    ax.set_ylabel(r"Entropy $S(0) / \ln 2$", fontsize=fs["label"])
    ax.set_title("Entropy vs. Probe Temperature", fontsize=fs["title"], pad=8)

    ax.text(
        0.03, 0.95, "A", transform=ax.transAxes,
        fontsize=fs["panel_tag"], fontweight="bold", va="top", ha="left",
    )

    handles = [
        Line2D([0], [0], color=colors[r], marker="o", ls="-", lw=1.8, ms=6.0,
               markeredgecolor="k", markeredgewidth=0.5, label=f"$r = {r:+.1f}$")
        for r in all_r
    ] + [
        Line2D([0], [0], color="k", marker="o", ls="None", ms=5.8,
               markeredgecolor="k", markeredgewidth=0.5, label="Tight-Binding"),
        Line2D([0], [0], color="k", ls="-", lw=1.8, label="Continuum"),
    ]
    ax.legend(handles=handles, frameon=True, facecolor="white", fontsize=fs["legend"], loc="lower left")
    _apply_kc_axes_polish(ax, fs)


def draw_panel_B(ax: plt.Axes, res_core: dict, standalone: bool = False) -> None:
    """
    Main Plot (B): Entropy S(0) / ln 2 as a function of -1 < r < 1
    across double-squeeze-selected probe temperatures T_S / D.
    """
    fs = _kc_font_sizes(standalone)
    D = res_core["D"]
    panel_B_data = res_core["panel_B"]
    ts_list = panel_B_data["ts_list"]

    ts_styles = [
        ("#0072b2", "-",  "o", "full"),
        ("#009e73", "--", "s", "none"),
        ("#e69f00", "-.", "^", "none"),
        ("#cc79a7", ":",  "d", "none"),
    ]

    r_diag = np.linspace(-0.98, 0.98, 120)
    ax.plot(r_diag, r_diag, "k--", lw=1.2, alpha=0.75, label=r"Low-$T_S$ Limit: $S(0)/\ln 2 = r$")
    ax.axhline(0.0, color="gray", ls=":", lw=0.7, alpha=0.45)
    ax.axvline(0.0, color="gray", ls=":", lw=0.7, alpha=0.45)

    for idx, ts_val in enumerate(ts_list):
        col, ls_v, mk_v, fill_v = ts_styles[idx % len(ts_styles)]
        cdata = panel_B_data["curves"][ts_val]
        ts_over_d = ts_val / D
        exp_pow = int(round(np.log10(ts_over_d)))
        if np.isclose(ts_over_d, 1.0):
            ts_tex = r"1"
        elif np.isclose(ts_over_d, 10.0 ** exp_pow):
            ts_tex = rf"10^{{{exp_pow}}}"
        else:
            ep = int(math.floor(math.log10(ts_over_d)))
            mant = ts_over_d / (10.0 ** ep)
            ts_tex = rf"{mant:g}\times 10^{{{ep}}}"

        ax.plot(cdata["r_dense"], cdata["S_recon_exact"], ls=ls_v, color=col, lw=1.8)
        ax.plot(
            cdata["r_tb"], cdata["S_recon_tb"], marker=mk_v, ls="None", color=col,
            ms=6.2, fillstyle=fill_v, mew=1.4 if fill_v == "none" else 0.5,
            markeredgecolor="k" if fill_v == "full" else col,
        )
        ax.plot(
            [], [], color=col, ls=ls_v, lw=1.8, marker=mk_v, ms=6.0,
            fillstyle=fill_v, mew=1.4 if fill_v == "none" else 0.5,
            markeredgecolor="k" if fill_v == "full" else col,
            label=rf"$T_S / D = {ts_tex}$",
        )

    ax.text(
        0.03, 0.95, "B", transform=ax.transAxes,
        fontsize=fs["panel_tag"], fontweight="bold", va="top", ha="left",
    )
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.05, 1.08)
    ax.set_xlabel(r"Bath Power-Law Exponent $r \in (-1, 1)$", fontsize=fs["label"])
    ax.set_ylabel(r"Entropy $S(0) / \ln 2$", fontsize=fs["label"])
    ax.set_title("Entropy vs. Bath Exponent", fontsize=fs["title"], pad=8)
    ax.legend(frameon=True, facecolor="white", fontsize=fs["legend"], loc="lower right")
    _apply_kc_axes_polish(ax, fs)


# ============================================================================
# 9. PDF EXPORT PIPELINE (MAIN AB, 3-ROW VERIFICATION, & SELECTIVE TO_PRINT)
# ============================================================================
def export_all_phase2_pdfs(
    res_core: dict,
    res_stress: dict,
    probe_profile: ProbeProfileType,
    suffix: str,
    to_print_keys: List[str],
    results_dir: str,
    plots_dir: str,
) -> None:
    """
    Generate exclusively vector PDF outputs inside `results/phase2v3/{preset_name}/`:
      1. `phase2_main_AB_{profile}{suffix}.pdf`: Side-by-side Main Plots (A) and (B) with a
         common upper banner reporting all shared fixed parameters in terms of D*.
      2. `phase2_verification_{profile}{suffix}.pdf`: Full 3-row diagnostic multi-plot
         (Row 1: a-d, Row 2: e-h, Row 3: A & B).
      3. `phase2plots/panel_<key>_{profile}{suffix}.pdf`: Individual PDFs for `to_print_keys`.
    """
    all_r = sorted(res_core["sweeps"].keys())
    colors = get_color_map(all_r)
    D = res_core["D"]
    D_cap = res_core["D_star"]
    rho = res_core["rho"]
    Gamma0 = res_core["Gamma0"]
    GammaS0 = res_core["GammaS0"]
    ratio_S_ref = res_core["ratio_S_ref"]
    D_S_cut = res_core["D_S_cut"]
    T_PG_base = res_core["T_PG_base"]
    T_S_fixed = res_core["T_S_fixed"]
    preset_name = res_core["preset_name"]

    probe_formula_tex = (
        r"\Gamma_S(\omega) = \Gamma_{S,0}\,\Theta(D_S - |\omega|)"
        if probe_profile == "flat_band"
        else r"\Gamma_S(\omega) = \Gamma_{S,0} / [1 + (\omega/D_S)^2]"
    )

    d_tex = r"D = D^*" if np.isclose(D, D_cap) else rf"D = {D/D_cap:g}D^*"
    ds_tex = r"D_S = D^*" if np.isclose(D_S_cut, D_cap) else rf"D_S = {D_S_cut/D_cap:g}D^*"
    preset_disp = (
        f"{preset_name} (d={res_core['d']:g}, k={res_core['k']:g})"
        if ("d" in res_core and "k" in res_core)
        else preset_name
    )

    # Keep {preset_disp} outside math delimiters so underscores render as literal text
    shared_params_banner = (
        f"Preset: {preset_disp}  |  "
        f"$\\Gamma_{{\\mathrm{{PG}}}}(\\omega) = \\Gamma_0 |\\omega|^r$,   "
        f"${probe_formula_tex}$,   "
        f"$\\rho = T_{{\\min}}/D^* = {rho:.0e}$\n"
        f"Fixed Constants ($D^* = {D_cap:g}$):  ${d_tex}$,   "
        f"${ds_tex}$,   "
        f"$\\Gamma_0 = {Gamma0/D_cap:.4g}D^*$,   "
        f"$\\Gamma_{{S,0}} = {GammaS0/D_cap:.1e}D^*$,   "
        f"$T_{{\\mathrm{{PG}}}} = {T_PG_base/D_cap:.1e}D^*$"
    )

    # -------------------------------------------------------------------------
    # 1. Side-by-Side Main Reference PDF: Plot (A) and Plot (B)
    # -------------------------------------------------------------------------
    fig_ab = plt.figure(figsize=(14.6, 6.2))
    gs_ab = GridSpec(
        1, 2, figure=fig_ab,
        left=0.068, right=0.985, bottom=0.125, top=0.790,
        wspace=0.24,
    )
    ax_A = fig_ab.add_subplot(gs_ab[0, 0])
    ax_B = fig_ab.add_subplot(gs_ab[0, 1])

    draw_panel_A(ax_A, res_core, colors, standalone=True)
    draw_panel_B(ax_B, res_core, standalone=True)

    fig_ab.text(
        0.526, 0.962, shared_params_banner,
        ha="center", va="top", fontsize=12.2,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8f9fa", edgecolor="#cccccc", linewidth=0.6),
    )

    main_ab_path = os.path.join(results_dir, f"phase2_main_AB_{probe_profile}{suffix}.pdf")
    fig_ab.savefig(main_ab_path, format="pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig_ab)
    print(f"Saved side-by-side Main (A, B) PDF to '{main_ab_path}'.", flush=True)

    # -------------------------------------------------------------------------
    # 2. Full 3-Row Verification Multi-Plot PDF (Rows 1-2: a-h; Row 3: A & B)
    # -------------------------------------------------------------------------
    fig_ver = plt.figure(figsize=(24, 18.8))
    gs = GridSpec(
        3, 4, figure=fig_ver,
        left=0.048, right=0.988, bottom=0.055, top=0.900,
        hspace=0.34, wspace=0.26,
    )

    ax_a = fig_ver.add_subplot(gs[0, 0])
    ax_b = fig_ver.add_subplot(gs[0, 1])
    ax_c = fig_ver.add_subplot(gs[0, 2])
    ax_d = fig_ver.add_subplot(gs[0, 3])
    ax_e = fig_ver.add_subplot(gs[1, 0])
    ax_f = fig_ver.add_subplot(gs[1, 1])
    ax_g = fig_ver.add_subplot(gs[1, 2])
    ax_h = fig_ver.add_subplot(gs[1, 3])
    ax_row3_A = fig_ver.add_subplot(gs[2, 0:2])
    ax_row3_B = fig_ver.add_subplot(gs[2, 2:4])

    draw_panel_a(ax_a, res_core, colors, standalone=False)
    draw_panel_b(ax_b, res_core, colors, standalone=False)
    draw_panel_c(ax_c, res_core, colors, standalone=False)
    draw_panel_d(ax_d, res_core, colors, standalone=False)
    draw_panel_e(ax_e, res_core, colors, standalone=False)
    draw_panel_f(ax_f, res_core, res_stress, colors, standalone=False)
    draw_panel_g(ax_g, res_core, res_stress, colors, standalone=False)
    draw_panel_h(ax_h, res_core, res_stress, colors, standalone=False)
    draw_panel_A(ax_row3_A, res_core, colors, standalone=False)
    draw_panel_B(ax_row3_B, res_core, standalone=False)

    r_handles = [Line2D([0], [0], color=colors[r], lw=3.0, label=f"$r = {r:+.1f}$") for r in all_r]
    fig_ver.legend(
        handles=r_handles,
        loc="upper center",
        bbox_to_anchor=(0.518, 0.980),
        ncol=len(all_r),
        frameon=True,
        facecolor="#f8f9fa",
        edgecolor="#cccccc",
        fontsize=11.5,
        title=(
            f"Phase 2 Verification Suite (benchmark2v3)  (Probe: ${probe_formula_tex}$,  $\\rho = T_{{\\min}}/D^* = {rho:.0e}$,  $D^* = {D_cap:g}$)\n"
            f"Preset ({preset_disp}):  ${d_tex}$,  ${ds_tex}$,  "
            f"$\\Gamma_0 = {Gamma0/D_cap:.4g}D^*$,  $\\Gamma_{{S,0}} = {GammaS0/D_cap:.1e}D^*$ "
            f"($\\Gamma_{{S,0}}/\\Gamma_0 = {ratio_S_ref:.2e}$),  "
            f"$T_{{\\mathrm{{PG}}}} = {T_PG_base/D_cap:.1e}D^*$,  $T_S = {T_S_fixed/D_cap:.3g}D^*$"
        ),
        title_fontsize=11.0,
    )

    ver_pdf_path = os.path.join(results_dir, f"phase2_verification_{probe_profile}{suffix}.pdf")
    fig_ver.savefig(ver_pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.20)
    plt.close(fig_ver)
    print(f"Saved 3-row Verification PDF to '{ver_pdf_path}'.", flush=True)

    # -------------------------------------------------------------------------
    # 3. Selective Standalone Panel PDFs in `phase2plots/` (`TO_PRINT`)
    # -------------------------------------------------------------------------
    panel_dispatch = {
        "a": lambda ax: draw_panel_a(ax, res_core, colors, standalone=True),
        "b": lambda ax: draw_panel_b(ax, res_core, colors, standalone=True),
        "c": lambda ax: draw_panel_c(ax, res_core, colors, standalone=True),
        "d": lambda ax: draw_panel_d(ax, res_core, colors, standalone=True),
        "e": lambda ax: draw_panel_e(ax, res_core, colors, standalone=True),
        "f": lambda ax: draw_panel_f(ax, res_core, res_stress, colors, standalone=True),
        "g": lambda ax: draw_panel_g(ax, res_core, res_stress, colors, standalone=True),
        "h": lambda ax: draw_panel_h(ax, res_core, res_stress, colors, standalone=True),
        "A": lambda ax: draw_panel_A(ax, res_core, colors, standalone=True),
        "B": lambda ax: draw_panel_B(ax, res_core, standalone=True),
    }

    for pkey in to_print_keys:
        fig_single = plt.figure(figsize=(7.8, 5.9))
        gs_s = GridSpec(1, 1, figure=fig_single, left=0.14, right=0.97, bottom=0.14, top=0.81)
        ax_single = fig_single.add_subplot(gs_s[0, 0])
        panel_dispatch[pkey](ax_single)
        if pkey in ("A", "B"):
            fig_single.text(
                0.555, 0.965,
                f"{preset_disp} ($D^*={D_cap:g}$): ${d_tex},\\; {ds_tex},\\; "
                f"\\Gamma_0 = {Gamma0/D_cap:.4g}D^*,\\; \\Gamma_{{S,0}} = {GammaS0/D_cap:.1e}D^*,\\; "
                f"T_{{\\mathrm{{PG}}}} = {T_PG_base/D_cap:.1e}D^*$",
                ha="center", va="top", fontsize=10.2,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8f9fa", edgecolor="#cccccc", linewidth=0.5),
            )
        single_pdf_path = os.path.join(plots_dir, f"panel_{pkey}_{probe_profile}{suffix}.pdf")
        fig_single.savefig(single_pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.10)
        plt.close(fig_single)

    print(
        f"Saved requested TO_PRINT={to_print_keys} individual PDFs to '{plots_dir}/'.",
        flush=True,
    )


# ============================================================================
# 10. MAIN ENTRY POINT (SPAWN-SAFE FOR macOS & LINUX MULTIPROCESSING)
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2 Two-Bath Benchmarks (benchmark2v3).")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset name ('ParamSearch', 'low_T_default', 'low_G0_highT', 'Ds_to_D_is_k', or 'all').",
    )
    parser.add_argument("--G0", type=float, default=None, help="Override Gamma0 in `ParamSearch`.")
    parser.add_argument("--GS0", type=float, default=None, help="Override GammaS0 in `ParamSearch`.")
    parser.add_argument("--D", type=float, default=None, help="Override PG bandwidth D in `ParamSearch`.")
    parser.add_argument("--DS", type=float, default=None, help="Override probe bandwidth D_S in `ParamSearch`.")
    parser.add_argument("--TPG", type=float, default=None, help="Override bath temperature T_PG in `ParamSearch`.")
    parser.add_argument("--TS-fixed", type=float, default=None, help="Override fixed T_S in `ParamSearch`.")
    parser.add_argument("--ts-B", nargs="+", type=float, default=None, help="Override Plot B T_S list in `ParamSearch`.")
    parser.add_argument("--d", type=float, default=DEFAULT_D_DIVISOR, help="Divisor d for `Ds_to_D_is_k` (D = D*/d).")
    parser.add_argument("--k", type=float, default=DEFAULT_K_RATIO, help="Ratio k for `Ds_to_D_is_k` (D_S = D/k).")
    parser.add_argument(
        "--rho",
        nargs="+",
        type=float,
        default=None,
        help="One or more dynamic range ratios rho = T_min / D* to loop over (default: RHO_LIST).",
    )
    parser.add_argument(
        "--to-print",
        nargs="+",
        default=None,
        help="Panel letters to export individually to `phase2plots/` (default: A B).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override worker count (default: max_cpu - 2 on PC/Mac).",
    )
    args = parser.parse_args()

    active_to_print = sanitize_to_print(args.to_print if args.to_print is not None else TO_PRINT)
    rho_values_to_run = [float(val) for val in (args.rho if args.rho is not None else RHO_LIST)]

    cli_param_overrides = {
        "D": args.D,
        "D_S_cut": args.DS,
        "Gamma0": args.G0,
        "GammaS0": args.GS0,
        "T_PG": args.TPG,
        "T_S_fixed": args.TS_fixed,
        "TS_panel_B": args.ts_B,
    }

    max_cpu, default_workers = get_worker_count(run_on_slurm=RUN_ON_SLURM, reserved_cpus=PC_RESERVED_CPUS)
    n_workers = max(1, int(args.workers)) if args.workers is not None else default_workers

    print(
        f"Initializing Phase 2 Suite (benchmark2v3) | RUN_ON_SLURM={RUN_ON_SLURM} | PROJECT_ROOT='{PROJECT_ROOT}'\n"
        f"Base Output Directory: '{PHASE2V3_ROOT_DIR}/'\n"
        f"Rho Values to Loop Over: {rho_values_to_run}\n"
        f"TO_PRINT Panels: {active_to_print}\n"
        f"CPU Pool: {n_workers} active workers (out of {max_cpu} logical CPUs)\n",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as executor:
        for rho_val in rho_values_to_run:
            catalog = build_preset_catalog(
                rho=rho_val,
                d=args.d,
                k=args.k,
                param_search_overrides=cli_param_overrides,
            )
            sync_presets_json(catalog, rho=rho_val)

            if args.preset is not None:
                if args.preset == "all":
                    presets_to_run = list(catalog.keys())
                elif args.preset == "Ds_to_D_is_k":
                    ds_cfg = make_preset_Ds_to_D_is_k(rho=rho_val, d=args.d, k=args.k)
                    catalog[ds_cfg["preset_name"]] = ds_cfg
                    presets_to_run = [ds_cfg["preset_name"]]
                else:
                    presets_to_run = [args.preset]
            elif isinstance(ACTIVE_PRESET, (list, tuple)):
                presets_to_run = list(ACTIVE_PRESET)
            elif ACTIVE_PRESET == "all":
                presets_to_run = list(catalog.keys())
            else:
                presets_to_run = [ACTIVE_PRESET]

            for pname in presets_to_run:
                if pname not in catalog:
                    raise ValueError(f"Unknown preset '{pname}'. Available presets: {list(catalog.keys())}")

            print(
                f"========================================================================================\n"
                f"LOOP ITERATION: rho = T_min / D* = {rho_val:.1e}  |  Presets: {presets_to_run}\n"
                f"========================================================================================",
                flush=True,
            )

            for preset_key in presets_to_run:
                cfg = catalog[preset_key]
                results_dir, plots_dir = get_preset_dirs(preset_key)
                suffix = cfg["file_suffix"]

                print(
                    f">>> Running Preset '{preset_key}' (rho={rho_val:.1e}) -> Output Folder: '{results_dir}/'\n"
                    f"    {cfg['description']}",
                    flush=True,
                )

                for profile_mode in ("flat_band", "lorentzian"):
                    res_core, res_stress, telem_summary = run_phase2_parallel_suite(
                        cfg=cfg,
                        probe_profile=profile_mode,
                        executor=executor,
                        max_cpu=max_cpu,
                        n_workers=n_workers,
                        r_list=R_LIST_ALL,
                    )

                    export_all_phase2_pdfs(
                        res_core=res_core,
                        res_stress=res_stress,
                        probe_profile=profile_mode,
                        suffix=suffix,
                        to_print_keys=active_to_print,
                        results_dir=results_dir,
                        plots_dir=plots_dir,
                    )

                    table_path = os.path.join(results_dir, f"phase2_summary_{profile_mode}{suffix}.txt")
                    with open(table_path, "w", encoding="utf-8") as f_out:
                        f_out.write("\n".join(res_core["log_lines"] + res_stress["log_lines"]) + "\n")
                    print(f"Saved Phase 2 ({profile_mode}) summary table to '{table_path}'.", flush=True)

                    telem_path = os.path.join(results_dir, f"phase2_telemetry_{profile_mode}{suffix}.json")
                    with open(telem_path, "w", encoding="utf-8") as f_tel:
                        json.dump(telem_summary, f_tel, indent=2)
                    print(f"Saved Phase 2 ({profile_mode}) telemetry report to '{telem_path}'.\n", flush=True)
