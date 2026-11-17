"""
run_phase2_benchmarks.py  (benchmark2v2)
----------------------------------------
Main changes from last version: selective panel printing (TO_PRINT), dedicated
/results/phase2v2 output folder, single-T_PG Main Plot (A) with 11 T_S points,
side-by-side main_AB PDF, vector PDF-only export, and KC figure aesthetics.

Executes Phase 2 (Step 1b: Two Baths Connected & Stress-Testing the Weak-Coupling Window)
with N_PG = 10,000 modes in the power-law bath and N_S = 4,000 modes in the probe bath
for the pseudogap exponents r in {0.3, 0.5} (plus -1 < r < 1 sweep in Main Plot B).

PRIMARY EXECUTION & PHYSICS FLAGS:
  1. RUN_ON_SLURM (bool):
       - True  : Configures SLURM/cluster single-threaded BLAS environment variables,
                 enables faulthandler, and writes outputs to `/home/amsalda/PseudoGap/results/phase2v2`.
       - False : Local PC / macOS mode; skips SLURM environment setup and resolves paths
                 one folder up from `/Computers` to `../results/phase2v2` inside
                 `/Lindbladian_ED_pseudgapBath/`.
  2. RHO (float):
       - Dynamic range ratio rho = T_min / D* (e.g. 1e-3 for 3 decades, 1e-4 for 4 decades),
         with the UV bandwidth cap D* = 1.0 ever-fixed and T_min = RHO * D*.
  3. TO_PRINT (list[str]):
       - Panel keys from {"a".."h", "A", "B"} to export individually as vector PDFs inside
         `results/phase2v2/phase2plots/` (falls back to `["A", "B"]` if empty or invalid).

CANONICAL ENERGY SCALE ASSIGNMENT (Maximizes the Two-Sided Weak-Coupling Window):
  - D = D* = 1.0                 (Ever-fixed UV bandwidth cap)
  - T_min = RHO * D*             (IR energy scale floor)
  - Gamma0 = D* = 1.0            (Pushes self-energy crossover T* = (Gamma0/D*)^(1/(1-r)) D* -> D*)
  - GammaS0 = T_min = RHO * D*   (Weak-probe coupling at the IR floor; ratio GammaS0/Gamma0 = RHO)
  - T_PG = T_min = RHO * D*      (Baseline power-law bath temperature)
  - D_S = D* = 1.0               (Baseline wide-band probe cutoff)
  - T_S_fixed = min(0.1 D*, 10 T_min)

MULTIPROCESSING & TELEMETRY:
  - Uses `concurrent.futures.ProcessPoolExecutor` with `max_cpu - PC_RESERVED_CPUS` workers
    on local PC/Mac (e.g., 20 - 2 = 18 workers) across all flattened tight-binding tasks.
  - Logs live progress, throughput, parallel scaling efficiency, and CPU contention warnings,
    and saves a structured JSON telemetry report alongside the PDFs and summary tables.

SYNCHRONIZED WITH `bath.py`, `spectral_qrt.py`, AND `run_phase1_benchmarks.py`:
  1. Uses `gamma_rule="capped_adaptive"` for the power-law bath `bath_pg`, eliminating
     1/eps_q^2 UV Lorentzian tail leakage into the low-energy pseudogap for r > 0.
  2. Retains the out-of-band bound-state pole contribution (|w| > D_PG) when the probe
     is at the wide-band cap (D_S = D*), matching `analytic_two_bath_tilde_S_eps_d` in
     `spectral_qrt.py` so wide-band curves reach S(0)/ln 2 -> +1 at high T_S.
"""

from __future__ import annotations

import os
import sys
import gc
import json
import time
import logging
import warnings
import argparse
import resource
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

# Suppress benign RuntimeWarnings (e.g. overflow in high-x Fermi exponentials) in main & workers
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"

# ============================================================================
# 1. PRIMARY USER CONTROL FLAGS (Edit these or override via CLI flags)
# ============================================================================
RUN_ON_SLURM: bool = False       # True: SLURM cluster env & /PseudoGap/results; False: PC ../results
RHO: float = 1.0e-3              # Dynamic range ratio rho = T_min / D* (1e-3 = 3 orders of magnitude)
TO_PRINT: List[str] = ["A", "B"] # Individual panels to export into `results/phase2v2/phase2plots/`

# Target pseudogap exponents:
# - Core & stress-test panels (a)-(h) and Main Plot (A) focus on r in {0.3, 0.5}
# - Main Plot (B) sweeps -1 < r < 1 at T_S/D in {1e-3, 1e-2, 1e-1, 1}
R_LIST_ALL: List[float] = [0.3, 0.4, 0.5]
R_LIST_PANEL_B: List[float] = [-0.9, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 0.9]
TS_OVER_D_PANEL_B: List[float] = [1.0e-3, 1.0e-2, 1.0e-1, 1.0]

# Ever-fixed UV bandwidth cap and derived IR floor
D_STAR: float = 1.0
T_MIN: float = RHO * D_STAR

# Local PC/Mac CPU reservation (dedicates max_cpu - PC_RESERVED_CPUS cores to workers)
PC_RESERVED_CPUS: int = 2

VALID_PANEL_KEYS = {"a", "b", "c", "d", "e", "f", "g", "h", "A", "B"}

# ============================================================================
# 2. ENVIRONMENT & DIRECTORY ROUTING (Driven by RUN_ON_SLURM)
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if RUN_ON_SLURM:
    # Cluster preamble: prevent BLAS thread oversubscription and enable crash tracebacks
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["MPLBACKEND"] = "Agg"

    import faulthandler
    faulthandler.enable()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    PROJECT_ROOT = (
        "/home/amsalda/PseudoGap"
        if os.path.exists("/home/amsalda/PseudoGap")
        else os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    )
else:
    # Local PC / macOS mode: step one folder up from /Computers to /Lindbladian_ED_pseudgapBath/
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase2v2")
PHASE2_PLOTS_DIR = os.path.join(RESULTS_DIR, "phase2plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PHASE2_PLOTS_DIR, exist_ok=True)
PRESETS_JSON_PATH = os.path.join(SCRIPT_DIR, "phase2_presets.json")

# ============================================================================
# 3. GLOBAL MATPLOTLIB STYLE (KC FIGURE GUIDELINES — VECTOR PDF ONLY)
# ============================================================================
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


def apply_kc_figure_style() -> None:
    """
    Configure Matplotlib rcParams following KC's Figure Guidelines:
      - Vector PDF output with editable TrueType text (`pdf.fonttype = 42`)
      - Standardized sans-serif typography (`Myriad Pro` with clean fallback)
      - Thin, unobtrusive 0.5 pt axes frames and tick marks
      - Goldilocks 1.5 pt default curve thickness
    """
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

from bath import Bath
from lyapunov_solver import LyapunovSolver
from spectral_qrt import (
    bare_bath_hybridization_dl,
    alpha_func_eq14,
    analytic_tilde_S0_eq14,
    analytic_tilde_S0_eq12,
    analytic_two_bath_tilde_S_eps_d,
)

# ============================================================================
# 4. DISCRETIZATION, QUADRATURE & INPUT VALIDATION
# ============================================================================
TARGET_N_PG: int = 10000
TARGET_N_S: int = 4000
LN2: float = float(np.log(2.0))

# Precomputed 64-point Gauss-Legendre nodes/weights on [0, 1] for out-of-band pole quadrature
_GL64_S, _GL64_W = np.polynomial.legendre.leggauss(64)
_GL64_S = 0.5 * (_GL64_S + 1.0)
_GL64_W = 0.5 * _GL64_W

ProbeProfileType = Literal["flat_band", "lorentzian"]


def sanitize_to_print(raw_to_print: Any) -> List[str]:
    """
    Validate `TO_PRINT` list of panel identifiers (from {'a'..'h', 'A', 'B'}).
    If `raw_to_print` is None, not a list/tuple/set of strings, empty, or contains
    corrupt/unknown tokens, logs a warning and returns the default `['A', 'B']`.
    """
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


# ============================================================================
# 5. SYSTEM HARDWARE & PROGRESS TELEMETRY (For PC / Mac & Cluster Monitoring)
# ============================================================================
def get_worker_count(run_on_slurm: bool = RUN_ON_SLURM, reserved_cpus: int = PC_RESERVED_CPUS) -> Tuple[int, int]:
    """
    Determine (max_cpu, n_workers).
    On PC/Mac (`run_on_slurm=False`), allocates `max(1, max_cpu - reserved_cpus)` workers
    (e.g., 20 - 2 = 18 workers) so macOS and agentic AI processes keep 2 cores free.
    """
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
                        kb = int(line.split()[1])
                        return kb / (1024.0 ** 2)
    except Exception:
        pass
    return None


def get_peak_rss_mb() -> float:
    """Return peak resident set size (RSS) in MB (handles macOS bytes vs Linux KiB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024.0 ** 2)
    return rss / 1024.0


class RunTelemetry:
    """
    Lightweight progress, parallel scaling efficiency, and CPU contention monitor.
    Designed for local Mac/PC runs (and future agentic AI parameter scans) to detect
    whether background CPU load or memory pressure slows down worker throughput.
    """

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
            # Parallel efficiency compares cumulative pure worker compute time to wall-clock CPU budget
            eff = self.cumulative_worker_sec / (self.n_workers * elapsed)
            eff_clamped = min(1.0, eff)
            load_1m = self._read_load_1m()

            print(
                f"[Telemetry | {self.stage_label}] {self.completed_tasks:3d}/{self.total_tasks:3d} ({pct:5.1f}%) | "
                f"Elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s | Rate: {rate:5.2f} tasks/s | "
                f"Parallel Eff: {100.0 * eff_clamped:5.1f}% | Load: {load_1m:4.1f}/{self.max_cpu}",
                flush=True,
            )

            # Flag CPU contention once workers have warmed up past the first wave
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
# 6. RATIO-DRIVEN ENERGY SCALE CONFIGURATION (D* = 1.0, Gamma0 = D*, GammaS0 = T_min)
# ============================================================================
def assert_within_caps(
    val: float | np.ndarray,
    name: str,
    t_min: float = T_MIN,
    d_star: float = D_STAR,
) -> None:
    """Verify that physical energy scale `val` lies strictly within [t_min, d_star]."""
    arr = np.asarray(val, dtype=float)
    if np.any(arr < t_min * (1.0 - 1e-12)) or np.any(arr > d_star * (1.0 + 1e-12)):
        raise ValueError(
            f"Physical parameter '{name}' = {val} violates bounds "
            f"[T_min={t_min:.1e}, D*={d_star:.1f}]."
        )


def make_config_from_rho(
    rho: float = RHO,
    preset_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Construct the canonical parameter set inside [T_min, D*] with:
      - D = D* = 1.0            (ever-fixed UV cutoff)
      - T_min = rho * D*        (IR scale floor)
      - Gamma0 = D* = 1.0       (maximizes the self-energy crossover scale T* -> D*)
      - GammaS0 = T_min         (weak-probe coupling at the IR floor; GammaS0/Gamma0 = rho)
      - T_PG = T_min            (baseline power-law bath temperature)
      - D_S_cut = D* = 1.0      (baseline wide-band probe cutoff)
      - T_S_fixed = min(0.1 * D*, 10 * T_min)
    All knob scan lists (T_PG/D, D_S/D) are log-spaced inside [rho, 1.0].
    """
    rho = float(rho)
    if not (0.0 < rho < 1.0):
        raise ValueError(f"RHO = T_min / D* must satisfy 0 < RHO < 1; got {rho}.")

    d_cap = D_STAR
    t_min = rho * d_cap
    D = d_cap
    Gamma0 = d_cap
    GammaS0 = t_min
    D_S_cut = d_cap
    T_PG = t_min
    T_S_fixed = min(0.1 * d_cap, 10.0 * t_min)

    for v, n in [
        (D, "D"), (Gamma0, "Gamma0"), (GammaS0, "GammaS0"),
        (D_S_cut, "D_S_cut"), (T_PG, "T_PG"), (T_S_fixed, "T_S_fixed"),
    ]:
        assert_within_caps(v, n, t_min=t_min, d_star=d_cap)

    gs_ratio_min = GammaS0 / Gamma0  # Equals rho
    gs_ratio_max = max(1.0e-1, 10.0 * gs_ratio_min)

    # Evenly log-spaced knob values inside [rho, 1.0]
    t_pg_over_d_list = [rho, rho ** (2.0 / 3.0), rho ** (1.0 / 3.0)]
    d_s_over_d_list = [rho, rho ** (2.0 / 3.0), rho ** (1.0 / 3.0), 1.0]

    # Filter Panel (B) T_S/D list to remain within [rho, 1.0]
    ts_panel_b = [val for val in TS_OVER_D_PANEL_B if val >= rho * (1.0 - 1e-12)]
    if not ts_panel_b:
        ts_panel_b = [rho, 1.0]

    auto_name = preset_name or f"rho{rho:.0e}"
    return {
        "preset_name": auto_name,
        "description": (
            f"D=D*={d_cap:g}, rho=T_min/D*={rho:.1e}, "
            f"Gamma0=D*={Gamma0:g}, GammaS0=T_min={GammaS0:.1e}, T_S_fixed={T_S_fixed:g}"
        ),
        "rho": rho,
        "D": float(D),
        "D_star": float(d_cap),
        "T_min": float(t_min),
        "Gamma0": float(Gamma0),
        "GammaS0": float(GammaS0),
        "D_S_cut": float(D_S_cut),
        "T_PG": float(T_PG),
        "T_S_fixed": float(T_S_fixed),
        "T_S_over_D_range": (rho, 1.0),
        "GS_over_G0_range": (float(gs_ratio_min), float(gs_ratio_max)),
        "T_PG_over_D_list": [float(x) for x in t_pg_over_d_list],
        "D_S_over_D_list": [float(x) for x in d_s_over_d_list],
        "TS_over_D_panel_B": [float(x) for x in ts_panel_b],
        "file_suffix": f"_{auto_name}",
    }


def sync_presets_json(active_cfg: Dict[str, Any]) -> None:
    """Record the active ratio-driven configuration in `phase2_presets.json`."""
    store: Dict[str, Any] = {}
    if os.path.exists(PRESETS_JSON_PATH):
        try:
            with open(PRESETS_JSON_PATH, "r", encoding="utf-8") as f:
                store = json.load(f)
        except Exception:
            store = {}
    store[active_cfg["preset_name"]] = active_cfg
    with open(PRESETS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)


# ============================================================================
# 7. TWO-BATH SOLVER & MULTIPROCESSING WORKER KERNELS
# ============================================================================
def _worker_init() -> None:
    """
    Initializer executed once per spawned worker process.
    Suppresses benign RuntimeWarnings, enforces single-threaded BLAS (preventing 18x20
    thread thrashing on macOS Accelerate), and sets headless Matplotlib rendering.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["MPLBACKEND"] = "Agg"


def make_lorentzian_shape(D_S_width: float):
    """Return Lorentzian frequency profile g_S(w) = 1 / (1 + (w / D_S_width)^2)."""
    return lambda w: 1.0 / (1.0 + (np.asarray(w, dtype=float) / D_S_width) ** 2)


def get_color_map(r_values: Sequence[float]) -> Dict[float, Any]:
    """Assign high-contrast publication colors sorted from smallest to largest r."""
    unique_r = sorted(set(float(r) for r in r_values))
    if len(unique_r) == 2:
        palette = ["#1f77b4", "#d55e00"]  # Royal blue (r=0.3) & vermilion (r=0.5)
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
    D_PG: float = D_STAR,
    t_min: float = T_MIN,
    omega_ir_PG: float = 0.0,
    n_pg: int = TARGET_N_PG,
    n_s: int = TARGET_N_S,
) -> LyapunovSolver:
    """
    Construct the two-bath LyapunovSolver with bounds t_min <= x <= D_PG.
    Uses `gamma_rule="capped_adaptive"` on `bath_pg` so high-energy logarithmic bins
    near |eps_q| ~ D_PG do not leak 1/w^2 Lorentzian tails into the low-energy pseudogap.
    """
    for val, name in [
        (T_S, "T_S"), (T_PG, "T_PG"), (Gamma0, "Gamma0"),
        (GammaS0, "GammaS0"), (D_S_cutoff, "D_S_cutoff"), (D_PG, "D_PG"),
    ]:
        assert_within_caps(val, name, t_min=t_min, d_star=D_PG)

    if probe_profile == "flat_band":
        D_S_band = D_S_cutoff
        probe_fn = None
    elif probe_profile == "lorentzian":
        D_S_band = D_PG
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
    T_S: float,
    probe_profile: ProbeProfileType,
    D_S_cutoff: float,
    D_PG: float = D_STAR,
    include_out_of_band_poles: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Evaluate the two-bath tight-binding sweep across `eps_d_grid`:
      1. Inside the bath band (|w| <= D_PG): `solver.sweep_eps_d_protocol` (`schur_lddl`).
      2. Outside the power-law bath band (|w| > D_PG): adds the out-of-band bound-state pole
         contribution when the probe is not hard-truncated below D_PG, recovering S(0)->ln 2 at high T_S.
    """
    sweep = solver.sweep_eps_d_protocol(eps_d_grid, target_bath="S", method="schur_lddl")

    has_subband_hard_cutoff = (
        probe_profile == "flat_band" and D_S_cutoff < 0.9999 * D_PG
    )
    x_full_max = 45.0
    if (
        include_out_of_band_poles
        and not has_subband_hard_cutoff
        and (1.0001 * D_PG / T_S < x_full_max)
    ):
        bath_pg = solver.baths[0]
        bath_s = solver.baths[1]
        Gamma0 = bath_pg.Gamma0
        r = bath_pg.r
        GammaS0 = bath_s.Gamma0

        x_start = 1.0001 * D_PG / T_S
        span = x_full_max - x_start
        x2 = x_start + span * _GL64_S
        dx2 = span * _GL64_W
        w_pos = x2 * T_S
        emx2 = np.exp(-x2)
        tw2 = x2 * emx2 / ((1.0 + emx2) ** 2)

        sig_pos, _ = bare_bath_hybridization_dl(bath_pg, w_pos)
        g_p = Gamma0 * (w_pos ** r)
        if probe_profile == "lorentzian":
            g_s = GammaS0 / (1.0 + (w_pos / D_S_cutoff) ** 2)
        else:
            g_s = np.full_like(w_pos, GammaS0)
        branch = g_s / (g_p + g_s)

        eps_min = float(eps_d_grid[0])
        ed_col = eps_d_grid[:, None]

        ph_neg = 0.5 * (
            np.sign(ed_col + w_pos[None, :] - sig_pos[None, :])
            - np.sign(eps_min + w_pos[None, :] - sig_pos[None, :])
        )
        ph_pos = 0.5 * (
            np.sign(ed_col - w_pos[None, :] + sig_pos[None, :])
            - np.sign(eps_min - w_pos[None, :] + sig_pos[None, :])
        )

        S_pole = ((ph_neg - ph_pos) * branch[None, :]) @ (tw2 * dx2)
        sweep["S_eps_d"] = sweep["S_eps_d"] + S_pole

    return sweep


def eval_analytic_two_bath(
    eps_d: float | np.ndarray,
    T_S: float,
    Gamma0: float,
    GammaS0: float,
    r: float,
    probe_profile: ProbeProfileType,
    D_S_cutoff: float,
    D_PG: float = D_STAR,
    eps_max: Optional[float] = None,
    include_out_of_band_poles: bool = True,
) -> float | np.ndarray:
    """Evaluate the two-bath continuum integral (Note Eq. 11) for the chosen probe profile."""
    if probe_profile == "flat_band":
        return analytic_two_bath_tilde_S_eps_d(
            eps_d, T_S, Gamma0, GammaS0, r,
            D_PG=D_PG, D_S=D_S_cutoff, probe_shape_fn=None, eps_max=eps_max,
            include_out_of_band_poles=include_out_of_band_poles,
        )
    if probe_profile == "lorentzian":
        return analytic_two_bath_tilde_S_eps_d(
            eps_d, T_S, Gamma0, GammaS0, r,
            D_PG=D_PG, D_S=D_PG, probe_shape_fn=make_lorentzian_shape(D_S_cutoff), eps_max=eps_max,
            include_out_of_band_poles=include_out_of_band_poles,
        )
    raise ValueError(f"Unknown probe_profile: {probe_profile}")


def _execute_benchmark_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Top-level picklable worker function executed in parallel across `ProcessPoolExecutor`.
    Runs a single independent tight-binding sweep or resonance evaluation and returns
    the result along with its pure wall-clock compute duration for telemetry.
    """
    t0 = time.perf_counter()
    kind = task["kind"]
    r = float(task["r"])
    T_S = float(task["T_S"])
    Gamma0 = float(task["Gamma0"])
    GammaS0 = float(task["GammaS0"])
    T_PG = float(task["T_PG"])
    D_S_cutoff = float(task["D_S_cutoff"])
    D_PG = float(task["D_PG"])
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
    )
    sweep = two_bath_tight_binding_sweep(
        solver=solver,
        eps_d_grid=eps_d_grid,
        T_S=T_S,
        probe_profile=probe_profile,
        D_S_cutoff=D_S_cutoff,
        D_PG=D_PG,
        include_out_of_band_poles=True,
    )

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
# 8. PARALLEL BENCHMARK ORCHESTRATOR (CORE + STRESS TESTS + PANEL B DATA)
# ============================================================================
def run_phase2_parallel_suite(
    cfg: Dict[str, Any],
    probe_profile: ProbeProfileType,
    executor: ProcessPoolExecutor,
    max_cpu: int,
    n_workers: int,
    r_list: List[float] = R_LIST_ALL,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Execute all Step 1b core verification, stress tests, and Main Plot (B) sweeps
    by flattening every independent tight-binding evaluation into `executor` and
    tracking progress/contention via `RunTelemetry`.
    """
    D = float(cfg["D"])
    D_cap = float(cfg["D_star"])
    t_min = float(cfg["T_min"])
    Gamma0 = float(cfg["Gamma0"])
    GammaS0 = float(cfg["GammaS0"])
    ratio_S_ref = GammaS0 / Gamma0
    T_S_fixed = float(cfg["T_S_fixed"])
    T_PG_base = float(cfg["T_PG"])
    D_S_cut = float(cfg["D_S_cut"])
    ts_min_over_D, ts_max_over_D = cfg["T_S_over_D_range"]
    gs_ratio_min, gs_ratio_max = cfg["GS_over_G0_range"]
    ts_b_over_d = cfg["TS_over_D_panel_B"]

    prof_desc = (
        f"Flat Probe Gamma_S(w) = GammaS0 * Theta(D_S - |w|) with D_S = {D_S_cut/D:g}D"
        if probe_profile == "flat_band"
        else f"Lorentzian Probe Gamma_S(w) = GammaS0 / (1 + (w/D_S)^2) with D_S = {D_S_cut/D:g}D"
    )

    header = (
        "=" * 124 + "\n"
        f"STEP 1b ({probe_profile.upper()} | Preset: {cfg['preset_name']}): {prof_desc}\n"
        f"  Bounds: [T_min={t_min:.1e}, D*={D_cap:g}] (rho={cfg['rho']:.1e}) | "
        f"Gamma0=D*={Gamma0:g}, GammaS0=T_min={GammaS0:.1e} (GS0/G0={ratio_S_ref:.1e}), "
        f"D_S={D_S_cut/D:g}D, T_PG={T_PG_base:.1e}, T_S_fixed={T_S_fixed:g} ({T_S_fixed/D:g}D)\n"
        + "=" * 124
    )
    print(header, flush=True)
    log_lines = [header]

    # Quick two-bath solver cross-check on first target exponent
    r_chk = float(r_list[0])
    sol_chk = build_two_bath_solver(
        r=r_chk, T_S=T_S_fixed, probe_profile=probe_profile, D_S_cutoff=D_S_cut,
        Gamma0=Gamma0, GammaS0=GammaS0, T_PG=T_PG_base, D_PG=D_cap, t_min=t_min,
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
            "D_PG": D_cap,
            "t_min": t_min,
            "probe_profile": probe_profile,
            "eps_d_grid": eps_d_grid,
        })

    # (B) Core T_S scan across [rho * D*, D*] (11 log-spaced points for Panel e & Main Plot A)
    T_grid = np.logspace(np.log10(ts_min_over_D * D), np.log10(ts_max_over_D * D), 11)
    T_dense = np.logspace(np.log10(ts_min_over_D * D), np.log10(ts_max_over_D * D), 65)
    assert_within_caps(T_grid, "T_grid", t_min=t_min, d_star=D_cap)

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
                "D_PG": D_cap,
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
                "D_PG": D_cap,
                "t_min": t_min,
                "probe_profile": probe_profile,
                "eps_d_grid": np.array([-eps_max, 0.0]),
            })

    # (D) Stress Test 2 (Panel g): Varying power-law bath temperature T_PG
    tpg_over_d = cfg["T_PG_over_D_list"]
    T_PG_low = float(tpg_over_d[0]) * D
    T_PG_mid = float(tpg_over_d[1]) * D
    T_PG_high = float(tpg_over_d[2]) * D
    T_S_scan_15 = np.logspace(np.log10(ts_min_over_D * D), np.log10(ts_max_over_D * D), 15)
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
                    "D_PG": D_cap,
                    "t_min": t_min,
                    "probe_profile": probe_profile,
                    "eps_d_grid": np.array([-eps_max_T, 0.0]),
                })

    # (E) Stress Test 3 (Panel h): Varying probe cutoffs D_S / D (reuses D_S == D_S_cut from Core T_S scan)
    D_S_list = [float(ds_mult) * D for ds_mult in cfg["D_S_over_D_list"]]
    for _ds in D_S_list:
        assert_within_caps(_ds, "D_S_list element", t_min=t_min, d_star=D_cap)

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
                    "D_PG": D_cap,
                    "t_min": t_min,
                    "probe_profile": probe_profile,
                    "eps_d_grid": np.array([-eps_max_T, 0.0]),
                })

    # (F) Main Plot (B): S(0)/ln 2 vs -1 < r < 1 at T_S/D in {1e-3, 1e-2, 1e-1, 1}
    r_tb_B = sorted(set(float(rv) for rv in R_LIST_PANEL_B))
    for idx_ts_b, ts_mult in enumerate(ts_b_over_d):
        ts_val = float(ts_mult) * D
        for idx_r_b, r_val in enumerate(r_tb_B):
            sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r_val)
            eff_w = Gamma0 * (ts_val ** r_val) * sec_pr2 + GammaS0
            eps_max_T = max(25.0 * D_cap, 100.0 * eff_w)
            tasks.append({
                "kind": "panel_B_tb",
                "meta": {"idx_ts_b": idx_ts_b, "idx_r_b": idx_r_b, "r": r_val},
                "r": r_val,
                "T_S": ts_val,
                "Gamma0": Gamma0,
                "GammaS0": GammaS0,
                "T_PG": T_PG_base,
                "D_S_cutoff": D_S_cut,
                "D_PG": D_cap,
                "t_min": t_min,
                "probe_profile": probe_profile,
                "eps_d_grid": np.array([-eps_max_T, 0.0]),
            })

    # -------------------------------------------------------------------------
    # Submit all tasks in parallel & monitor via RunTelemetry
    # -------------------------------------------------------------------------
    telemetry = RunTelemetry(
        stage_label=f"{probe_profile} | {cfg['preset_name']}",
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
    raw_panel_B: np.ndarray = np.zeros((len(ts_b_over_d), len(r_tb_B)))

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
        "T_S_over_D_range": (ts_min_over_D, ts_max_over_D),
        "GS_over_G0_range": cfg["GS_over_G0_range"],
        "sweeps": {},
        "T_scans": {},
        "panel_B": {},
    }

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
            probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D_cap,
            eps_max=eps_max, include_out_of_band_poles=True,
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
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D_cap,
                eps_max=eps_max_T, include_out_of_band_poles=True,
            ))
            tS_exact_T[idx_t] = val_ex
            S_recon_num_T[idx_t] = (tS_num_T[idx_t] / LN2) / pref
            S_recon_exact_T[idx_t] = (val_ex / LN2) / pref

        tS_exact_dense = np.array([
            float(eval_analytic_two_bath(
                0.0, float(t), Gamma0, GammaS0, r,
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D_cap,
                eps_max=max(25.0 * D_cap, 50.0 * (Gamma0 * (float(t) ** r) * sec_pr2 + GammaS0)),
                include_out_of_band_poles=True,
            ))
            for t in T_dense
        ])
        tS_trunc_dense = np.array([
            float(eval_analytic_two_bath(
                0.0, float(t), Gamma0, GammaS0, r,
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D_cap,
                eps_max=max(25.0 * D_cap, 50.0 * (Gamma0 * (float(t) ** r) * sec_pr2 + GammaS0)),
                include_out_of_band_poles=False,
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
            "tS_trunc_dense": tS_trunc_dense,
            "tS_eq12_dense": tS_eq12_dense,
            "tS_eq14_dense": tS_eq14_dense,
            "S_recon_exact_dense": (tS_exact_dense / LN2) / pref_dense,
            "S_recon_trunc_dense": (tS_trunc_dense / LN2) / pref_dense,
            "S_recon_eq12_dense": (tS_eq12_dense / LN2) / pref_dense,
            "S_recon_eq14_dense": (tS_eq14_dense / LN2) / pref_dense,
        }

    # Assemble Main Plot (B) data across -1 < r < 1 (with smooth uniform cutoff)
    r_dense_B = np.linspace(-0.90, 0.90, 121)
    alpha_tb_B = np.array([alpha_func_eq14(rv) for rv in r_tb_B])
    alpha_dense_B = np.array([alpha_func_eq14(float(rv)) for rv in r_dense_B])
    panel_B_curves: Dict[float, Dict[str, np.ndarray]] = {}

    for idx_ts_b, ts_mult in enumerate(ts_b_over_d):
        ts_val = float(ts_mult) * D
        prefs_tb = ratio_S_ref * alpha_tb_B * (ts_val ** (-np.array(r_tb_B)))
        s_recon_tb_B = (raw_panel_B[idx_ts_b] / LN2) / prefs_tb

        s_recon_ex_B = np.zeros(len(r_dense_B))
        for idx_rd, rv in enumerate(r_dense_B):
            r_fl = float(rv)
            sec_pr2 = 1.0 / np.cos(0.5 * np.pi * r_fl)
            eff_w = Gamma0 * (ts_val ** r_fl) * sec_pr2 + GammaS0
            eps_max_T = max(25.0 * D_cap, 100.0 * eff_w)
            val_ex = float(eval_analytic_two_bath(
                0.0, ts_val, Gamma0, GammaS0, r_fl,
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D_cap,
                eps_max=eps_max_T, include_out_of_band_poles=True,
            ))

            pref_d = ratio_S_ref * alpha_dense_B[idx_rd] * (ts_val ** (-r_fl))
            s_recon_ex_B[idx_rd] = (val_ex / LN2) / pref_d

        panel_B_curves[float(ts_mult)] = {
            "T_S": ts_val,
            "r_tb": np.array(r_tb_B, dtype=float),
            "S_recon_tb": s_recon_tb_B,
            "r_dense": r_dense_B,
            "S_recon_exact": s_recon_ex_B,
        }
    results_1b["panel_B"] = {
        "ts_over_d_list": [float(x) for x in ts_b_over_d],
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
                probe_profile=probe_profile, D_S_cutoff=D_S_cut, D_PG=D_cap,
                eps_max=eps_max, include_out_of_band_poles=True,
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
                    probe_profile=probe_profile, D_S_cutoff=ds_val, D_PG=D_cap,
                    eps_max=eps_max_T, include_out_of_band_poles=True,
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
# 9. MODULAR KC-STYLED PANEL RENDERERS (PDF ONLY)
# ============================================================================
def _kc_font_sizes(standalone: bool) -> Dict[str, float]:
    """
    Return KC-proportioned font sizes:
      - Axis labels ~25% larger than tick numbers
      - Panel letters (A, B, a..h) substantially larger than axis labels
    """
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
    probe_short = f"Flat $D_S={D_S_cut/D:g}D$" if res_core["probe_profile"] == "flat_band" else f"Lorentzian $D_S={D_S_cut/D:g}D$"

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
    ax.set_title(f"(a) Step 1b: $\\partial n_d / \\partial T_S$ ($T_S/D={T_S_fixed/D:g}, T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)", fontsize=fs["title"])
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
    ax.set_title(f"(b) Step 1b: Integral $\\tilde{{S}}(\\epsilon_d)$ ($T_S/D={T_S_fixed/D:g}, T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)", fontsize=fs["title"])
    handles = [
        Line2D([0], [0], color="k", ls="-", lw=1.8, label=r"Tight-Binding Lyapunov ($-\int \partial_{T_S}n_d\,d\epsilon_d'$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.1, label=r"Exact Finite-Cutoff Integral (Eq. 11)"),
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
    D, ratio_S_ref, D_S_cut, T_PG_base = res_core["D"], res_core["ratio_S_ref"], res_core["D_S_cut"], res_core["T_PG_base"]
    x_min_T, x_max_T = res_core["T_S_over_D_range"]

    for r in all_r:
        if np.isclose(r, 0.0):
            continue
        tdata = res_core["T_scans"][r]
        ax.loglog(tdata["T_dense"] / D, np.abs(tdata["tS_eq14_dense"]) / (ratio_S_ref * LN2), ":", color=colors[r], lw=1.2)
        ax.loglog(tdata["T_dense"] / D, np.abs(tdata["tS_eq12_dense"]) / (ratio_S_ref * LN2), "--", color=colors[r], lw=1.2)
        ax.loglog(tdata["T_dense"] / D, np.abs(tdata["tS_exact_dense"]) / (ratio_S_ref * LN2), "-", color=colors[r], lw=1.6)
        ax.loglog(tdata["T_grid"] / D, np.abs(tdata["tS_num"]) / (ratio_S_ref * LN2), "o", color=colors[r], ms=5.5)
    ax.set_xlim(x_min_T, x_max_T)
    ax.set_ylim(1e-2, 1e2)
    ax.set_xlabel(r"Probe Temperature $T_S / D$ (log scale)", fontsize=fs["label"])
    ax.set_ylabel(r"Scaled Magnitude $|\tilde{S}(0)| / (\frac{\Gamma_{S,0}}{\Gamma_0}\ln 2)$", fontsize=fs["label"])
    ax.set_title(f"(c) Step 1b: Scaling $|\\tilde{{S}}(0)| \\propto T_S^{{-r}}$ ($D_S={D_S_cut/D:g}D, T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)", fontsize=fs["title"])
    handles = [
        Line2D([0], [0], color="k", marker="o", ls="None", ms=5, label=r"Tight-Binding Lyapunov ($N=14\mathrm{k}$)"),
        Line2D([0], [0], color="k", ls="-", lw=1.6, label=f"Exact Finite-Cutoff Integral (Eq. 11, $D_S={D_S_cut/D:g}D$)"),
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
    ax.set_title(f"(d) Step 1b: Reconstructed $S(0)/\\ln 2$ vs $r$ ($T_S/D={T_S_fixed/D:g}, T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)", fontsize=fs["title"])
    handles = [
        Line2D([0], [0], color="k", ls="--", lw=1.6, label=r"Low-$T_S$ Linear Law: $S_{\mathrm{recon}}(0)/\ln 2 = r$ (Eq. 14)"),
        Line2D([0], [0], color="#444444", marker="s", ls="None", fillstyle="none", ms=7.5, mew=1.4, label=r"Exact Finite-Cutoff Integral (Eq. 11)"),
        Line2D([0], [0], color="k", marker="o", ls="None", ms=7.0, label=f"Tight-Binding Lyapunov ($T_S/D = {T_S_fixed/D:g}$)"),
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
    D, D_cap, T_S_fixed, T_PG_base, D_S_cut = res_core["D"], res_core["D_star"], res_core["T_S_fixed"], res_core["T_PG_base"], res_core["D_S_cut"]
    x_min_T, x_max_T = res_core["T_S_over_D_range"]
    s_ylim = (-0.15, 1.08) if min(all_r) >= 0.0 else (-1.05, 1.08)

    for r in all_r:
        tdata = res_core["T_scans"][r]
        ax.semilogx(tdata["T_dense"] / D, tdata["S_recon_eq12_dense"], "--", color=colors[r], lw=1.3, alpha=0.75)
        ax.semilogx(tdata["T_dense"] / D, tdata["S_recon_exact_dense"], "-", color=colors[r], lw=1.8)
        ax.semilogx(tdata["T_dense"] / D, tdata["S_recon_trunc_dense"], "-.", color=colors[r], lw=1.1, alpha=0.50)
        ax.semilogx(tdata["T_grid"] / D, tdata["S_recon_num"], "o", color=colors[r], ms=5.5, markeredgecolor="k", markeredgewidth=0.4)
        ax.axhline(r, color=colors[r], ls=":", lw=1.0, alpha=0.55)
    ax.axhline(1.0, color="dimgray", ls="-.", lw=1.0, alpha=0.6)
    ax.axvline(T_S_fixed / D, color="dimgray", ls=":", lw=1.0, alpha=0.7)
    ax.set_xlim(x_min_T, x_max_T)
    ax.set_ylim(*s_ylim)
    ax.set_xlabel(r"Probe Temperature $T_S / D$", fontsize=fs["label"])
    ax.set_ylabel(r"Reconstructed Entropy $S_{\mathrm{recon}}(0) / \ln 2$", fontsize=fs["label"])
    ax.set_title(f"(e) Step 1b: $S_{{\\mathrm{{recon}}}}(0)/\\ln 2$ vs $T_S/D$ ($D_S={D_S_cut/D:g}D, T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)", fontsize=fs["title"])
    handles = [
        Line2D([0], [0], color="k", marker="o", ls="-", lw=1.7, ms=5, label=f"Tight-Binding & Exact Integral (Eq. 11, $D_S={D_S_cut/D:g}D$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.3, label=r"Unbounded Note Limit (Eq. 12: $D_S\to\infty, \Gamma_S\to 0$)"),
        Line2D([0], [0], color="k", ls="-.", lw=1.1, alpha=0.55, label=f"Band-Truncated Only ($|\\omega|\\leq D^*={D_cap/D:g}D$)"),
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
    ax.set_title(f"(f) Test 1: $\\Gamma_S \\ll \\Gamma(\\omega)$ Breakdown ($T_S/D={T_S_fixed/D:g}, T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)", fontsize=fs["title"])
    handles = [
        Line2D([0], [0], color="k", marker="o", ls="None", ms=5, label=r"Tight-Binding Lyapunov ($N=14\mathrm{k}$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.5, label=r"Exact Finite-Cutoff Integral (Eq. 11)"),
        Line2D([0], [0], color="k", ls=":", lw=1.0, label=r"Weak-Probe Limit: $\Gamma_S \to 0$ (Eq. 14)"),
        Line2D([0], [0], color="dimgray", ls="-.", lw=1.0, label=f"Default $\\Gamma_{{S,0}}/\\Gamma_0 = {ratio_S_ref:.1e}$"),
    ]
    if standalone:
        handles += [Line2D([0], [0], color=colors[r], lw=2.2, label=f"$r = {r:+.1f}$") for r in all_r]
    ax.legend(handles=handles, frameon=True, facecolor="white", fontsize=fs["legend"] * 0.88, loc="upper left")
    _apply_kc_axes_polish(ax, fs)


def draw_panel_g(ax: plt.Axes, res_core: dict, res_stress: dict, colors: dict, standalone: bool = False) -> None:
    """Diagnostic Panel (g): Varies bath temperature T_PG/D across {rho, rho^(2/3), rho^(1/3)}."""
    fs = _kc_font_sizes(standalone)
    all_r = sorted(res_core["sweeps"].keys())
    D, D_S_cut = res_core["D"], res_core["D_S_cut"]
    x_min_T, x_max_T = res_core["T_S_over_D_range"]
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
    ax.set_xlim(x_min_T, x_max_T)
    ax.set_ylim(*s_ylim)
    ax.set_xlabel(r"Probe Temperature $T_S / D$", fontsize=fs["label"])
    ax.set_ylabel(r"Reconstructed Entropy $S_{\mathrm{recon}}(0) / \ln 2$", fontsize=fs["label"])
    ax.set_title(
        f"(g) Test 2: Varying Bath Temp $T_{{\\mathrm{{PG}}}}/D \\in "
        f"\\{{{T_PG_low/D:g}, {T_PG_mid/D:g}, {T_PG_high/D:g}\\}}$ ($D_S={D_S_cut/D:g}D$)",
        fontsize=fs["title"],
    )
    handles = [
        Line2D([0], [0], color="k", ls="-", lw=1.8, label=f"Exact Finite-Cutoff (Eq. 11, $D_S={D_S_cut/D:g}D$, indep. of $T_{{\\mathrm{{PG}}}}$)"),
        Line2D([0], [0], color="k", ls="--", lw=1.2, label=r"Unbounded Note Limit (Eq. 12, $D_S\to\infty$)"),
        Line2D([0], [0], color="k", marker="o", ls="None", ms=5.5, label=f"Tight-Binding: $T_{{\\mathrm{{PG}}}}/D = {T_PG_low/D:g}$"),
        Line2D([0], [0], color="k", marker="s", ls="None", ms=6.2, fillstyle="none", mew=1.4, label=f"Tight-Binding: $T_{{\\mathrm{{PG}}}}/D = {T_PG_mid/D:g}$"),
        Line2D([0], [0], color="k", marker="^", ls="None", ms=6.5, fillstyle="none", mew=1.4, label=f"Tight-Binding: $T_{{\\mathrm{{PG}}}}/D = {T_PG_high/D:g}$"),
    ]
    if standalone:
        handles += [Line2D([0], [0], color=colors[r], lw=2.2, label=f"$r = {r:+.1f}$") for r in all_r]
    ax.legend(handles=handles, frameon=True, facecolor="white", fontsize=fs["legend"] * 0.85, loc="upper left")
    _apply_kc_axes_polish(ax, fs)


def draw_panel_h(ax: plt.Axes, res_core: dict, res_stress: dict, colors: dict, standalone: bool = False) -> None:
    fs = _kc_font_sizes(standalone)
    all_r = sorted(res_core["sweeps"].keys())
    D, T_PG_base = res_core["D"], res_core["T_PG_base"]
    x_min_T, x_max_T = res_core["T_S_over_D_range"]
    s_ylim = (-0.15, 1.08) if min(all_r) >= 0.0 else (-1.05, 1.08)

    D_S_list = res_stress["probe_profiles"]["D_S_list"]
    ds_styles = [
        (D_S_list[3], "-", "o", 1.8, True, f"Cutoff $D_S = {D_S_list[3]/D:g}D = D^*$ (Eq. 11 & TB)"),
        (D_S_list[2], "--", "s", 1.5, False, f"Cutoff $D_S = {D_S_list[2]/D:g}D$ (Eq. 11 & TB)"),
        (D_S_list[1], "-.", "d", 1.4, False, f"Cutoff $D_S = {D_S_list[1]/D:g}D$ (Eq. 11 & TB)"),
        (D_S_list[0], ":", "^", 1.4, False, f"Cutoff $D_S = {D_S_list[0]/D:g}D$ (Eq. 11 & TB)"),
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
    ax.set_xlim(x_min_T, x_max_T)
    ax.set_ylim(*s_ylim)
    ax.set_xlabel(r"Probe Temperature $T_S / D$", fontsize=fs["label"])
    ax.set_ylabel(r"Reconstructed Entropy $S_{\mathrm{recon}}(0) / \ln 2$", fontsize=fs["label"])
    ds_str_list = ", ".join(f"{ds/D:g}" for ds in D_S_list)
    ax.set_title(f"(h) Test 3: Probe Cutoffs $D_S/D \\in \\{{{ds_str_list}\\}}$ ($T_{{\\mathrm{{PG}}}}/D={T_PG_base/D:g}$)", fontsize=fs["title"])
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
    Main Plot (A): Entropy S(0)/ln 2 vs. Probe Temperature T_S/D (11 log-spaced points)
    at fixed T_PG = T_min (matching Plot B), keeping the figure clean and uncluttered.
    """
    fs = _kc_font_sizes(standalone)
    all_r = sorted(res_core["sweeps"].keys())
    D = res_core["D"]
    x_min_T, x_max_T = res_core["T_S_over_D_range"]
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
    ax.set_xlim(x_min_T, x_max_T)
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
    across T_S / D in {1e-3, 1e-2, 1e-1, 1} at the same fixed parameters as Plot (A).
    """
    fs = _kc_font_sizes(standalone)
    panel_B_data = res_core["panel_B"]
    ts_mults = panel_B_data["ts_over_d_list"]

    ts_styles = [
        ("#0072b2", "-",  "o", "full"),   # T_S/D = 1e-3
        ("#009e73", "--", "s", "none"),   # T_S/D = 1e-2
        ("#e69f00", "-.", "^", "none"),   # T_S/D = 1e-1
        ("#cc79a7", ":",  "d", "none"),   # T_S/D = 1
    ]

    r_diag = np.linspace(-0.98, 0.98, 120)
    ax.plot(r_diag, r_diag, "k--", lw=1.2, alpha=0.75, label=r"Low-$T_S$ Limit: $S(0)/\ln 2 = r$")
    ax.axhline(0.0, color="gray", ls=":", lw=0.7, alpha=0.45)
    ax.axvline(0.0, color="gray", ls=":", lw=0.7, alpha=0.45)

    for idx, ts_mult in enumerate(ts_mults):
        col, ls_v, mk_v, fill_v = ts_styles[idx % len(ts_styles)]
        cdata = panel_B_data["curves"][ts_mult]
        exp_pow = int(round(np.log10(ts_mult)))
        ts_tex = r"1" if np.isclose(ts_mult, 1.0) else (
            rf"10^{{{exp_pow}}}" if np.isclose(ts_mult, 10.0 ** exp_pow) else f"{ts_mult:g}"
        )
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
# 10. PDF EXPORT PIPELINE (MAIN AB, 3-ROW VERIFICATION, & SELECTIVE TO_PRINT)
# ============================================================================
def export_all_phase2_pdfs(
    res_core: dict,
    res_stress: dict,
    probe_profile: ProbeProfileType,
    suffix: str,
    to_print_keys: List[str],
) -> None:
    """
    Generate exclusively vector PDF outputs inside `results/phase2v2/`:
      1. `phase2_main_AB_{profile}{suffix}.pdf`: Side-by-side Main Plots (A) and (B) with a
         common upper banner reporting all shared fixed parameters.
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

    probe_formula_tex = (
        r"\Gamma_S(\omega) = \Gamma_{S,0}\,\Theta(D_S - |\omega|)"
        if probe_profile == "flat_band"
        else r"\Gamma_S(\omega) = \Gamma_{S,0} / [1 + (\omega/D_S)^2]"
    )

    shared_params_banner = (
        f"$\\Gamma_{{\\mathrm{{PG}}}}(\\omega) = \\Gamma_0 |\\omega|^r$,   "
        f"${probe_formula_tex}$,   "
        f"$\\rho = T_{{\\min}}/D^* = {rho:.0e}$\n"
        f"Fixed Constants:  $D = D^* = {D_cap:g}$,   "
        f"$\\Gamma_0 = {Gamma0/D:g}D$,   "
        f"$\\Gamma_{{S,0}} = {GammaS0/D:g}D$,   "
        f"$T_{{\\mathrm{{PG}}}} = {T_PG_base/D:g}D$,   "
        f"$D_S = {D_S_cut/D:g}D$"
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

    # Common upper legend / parameter box reporting all shared fixed constants
    fig_ab.text(
        0.526, 0.962, shared_params_banner,
        ha="center", va="top", fontsize=12.5,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8f9fa", edgecolor="#cccccc", linewidth=0.6),
    )

    main_ab_path = os.path.join(RESULTS_DIR, f"phase2_main_AB_{probe_profile}{suffix}.pdf")
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
            f"Phase 2 Verification Suite  ($r \\in \\{{0.3, 0.5\\}}$ for a–h & A;  $r \\in (-1, 1)$ for B | "
            f"Probe: ${probe_formula_tex}$,  $\\rho = T_{{\\min}}/D^* = {rho:.0e}$,  $D^* = {D_cap:g}$)\n"
            f"Config ({res_core['preset_name']}):  "
            f"$\\Gamma_0 = D^* = {Gamma0/D:g}D$,  $\\Gamma_{{S,0}} = T_{{\\min}} = {GammaS0/D:g}D$ "
            f"($\\Gamma_{{S,0}}/\\Gamma_0 = {ratio_S_ref:.1e}$),  "
            f"$D_S = {D_S_cut/D:g}D$,  $T_{{\\mathrm{{PG}}}} = {T_PG_base/D:g}D$,  $T_S = {T_S_fixed/D:g}D$"
        ),
        title_fontsize=11.0,
    )

    ver_pdf_path = os.path.join(RESULTS_DIR, f"phase2_verification_{probe_profile}{suffix}.pdf")
    fig_ver.savefig(ver_pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.20)
    plt.close(fig_ver)
    print(f"Saved 3-row Verification PDF to '{ver_pdf_path}'.", flush=True)

    # -------------------------------------------------------------------------
    # 3. Selective Standalone Panel PDFs in `results/phase2v2/phase2plots/` (`TO_PRINT`)
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
                f"$D = D^* = {D_cap:g},\\; \\Gamma_0 = {Gamma0/D:g}D,\\; "
                f"\\Gamma_{{S,0}} = {GammaS0/D:g}D,\\; T_{{\\mathrm{{PG}}}} = {T_PG_base/D:g}D,\\; "
                f"D_S = {D_S_cut/D:g}D$",
                ha="center", va="top", fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8f9fa", edgecolor="#cccccc", linewidth=0.5),
            )
        single_pdf_path = os.path.join(PHASE2_PLOTS_DIR, f"panel_{pkey}_{probe_profile}{suffix}.pdf")
        fig_single.savefig(single_pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.10)
        plt.close(fig_single)

    print(
        f"Saved requested TO_PRINT={to_print_keys} individual PDFs to '{PHASE2_PLOTS_DIR}/'.",
        flush=True,
    )


# ============================================================================
# 11. MAIN ENTRY POINT (SPAWN-SAFE FOR macOS & LINUX MULTIPROCESSING)
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2 Two-Bath Benchmarks (benchmark2v2).")
    parser.add_argument(
        "--rho",
        type=float,
        default=RHO,
        help="Dynamic range ratio rho = T_min / D* (with D* = 1.0, Gamma0 = D*, GammaS0 = T_min).",
    )
    parser.add_argument(
        "--to-print",
        nargs="+",
        default=None,
        help="Panel letters to export individually to `results/phase2v2/phase2plots/` (default: A B).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override worker count (default: max_cpu - 2 on PC/Mac).",
    )
    args = parser.parse_args()

    active_to_print = sanitize_to_print(args.to_print if args.to_print is not None else TO_PRINT)
    cfg = make_config_from_rho(rho=args.rho)
    sync_presets_json(cfg)

    max_cpu, default_workers = get_worker_count(run_on_slurm=RUN_ON_SLURM, reserved_cpus=PC_RESERVED_CPUS)
    n_workers = max(1, int(args.workers)) if args.workers is not None else default_workers
    suffix = cfg["file_suffix"]

    print(
        f"Initializing Phase 2 Suite (benchmark2v2) | RUN_ON_SLURM={RUN_ON_SLURM} | PROJECT_ROOT='{PROJECT_ROOT}'\n"
        f"Active Config: {cfg['description']}\n"
        f"TO_PRINT Panels: {active_to_print} -> '{PHASE2_PLOTS_DIR}/'\n"
        f"CPU Pool: {n_workers} active workers (out of {max_cpu} logical CPUs)\n",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as executor:
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
            )

            table_path = os.path.join(RESULTS_DIR, f"phase2_summary_{profile_mode}{suffix}.txt")
            with open(table_path, "w", encoding="utf-8") as f_out:
                f_out.write("\n".join(res_core["log_lines"] + res_stress["log_lines"]) + "\n")
            print(f"Saved Phase 2 ({profile_mode}) summary table to '{table_path}'.", flush=True)

            telem_path = os.path.join(RESULTS_DIR, f"phase2_telemetry_{profile_mode}{suffix}.json")
            with open(telem_path, "w", encoding="utf-8") as f_tel:
                json.dump(telem_summary, f_tel, indent=2)
            print(f"Saved Phase 2 ({profile_mode}) telemetry report to '{telem_path}'.\n", flush=True)
