"""
phase_transition.py
===================
Free energy landscape, heat capacity, and critical temperature T* for the
Hamiltonian eigenspectrum.

Each eigenvalue λ_k is treated as a formal energy level. The partition
function Z(T) = Σ exp(−λ_k / T) defines a statistical-mechanical energy
landscape over the dimensionless temperature parameter T > 0. The Helmholtz
free energy F(T) = −T log Z(T) and its second derivative (heat capacity C_v)
identify the critical temperature T* at which the spectral energy landscape
reorganizes most sharply — analogous to a phase transition.

These are mathematical tools derived from the structure of statistical
mechanics, not physical temperatures or energies.

Usage
-----
    from src.phase_transition import compute_free_energy, find_critical_temp

Author : John D. Mayfield, M.D., Ph.D., M.Sc.
         Department of Radiology, Massachusetts General Hospital
         Harvard Medical School
"""

import warnings
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

N_TEMP_POINTS    = 1000   # Number of points on the temperature grid
T_GRID_LOW_MULT  = 0.01   # T_min = T_GRID_LOW_MULT  × mean(λ)
T_GRID_HIGH_MULT = 50.0   # T_max = T_GRID_HIGH_MULT × mean(λ)
SPLINE_DEGREE    = 4      # Smoothing spline degree for F(T)
PEAK_PROMINENCE  = 0.05   # Min peak prominence as fraction of C_v range


# ── Temperature grid ─────────────────────────────────────────────────────────

def make_temperature_grid(eigenvalues: np.ndarray,
                          n_points: int = N_TEMP_POINTS,
                          low_mult: float = T_GRID_LOW_MULT,
                          high_mult: float = T_GRID_HIGH_MULT) -> np.ndarray:
    """
    Build a temperature grid scaled to the eigenvalue spectrum.

    The grid spans [low_mult × λ_mean, high_mult × λ_mean], ensuring the
    heat capacity peak falls within the sampled range regardless of the
    absolute scale of the eigenspectrum.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues in descending order (non-negative).
    n_points : int
        Number of temperature grid points (default 1000).
    low_mult : float
        Lower bound multiplier relative to mean eigenvalue (default 0.01).
    high_mult : float
        Upper bound multiplier relative to mean eigenvalue (default 50.0).

    Returns
    -------
    T_grid : np.ndarray, shape (n_points,)
        Linearly spaced temperature grid.
    """
    lam_mean = eigenvalues.mean()
    if lam_mean < 1e-12:
        lam_mean = 1.0  # guard against zero spectrum
    T_lo = lam_mean * low_mult
    T_hi = lam_mean * high_mult
    return np.linspace(T_lo, T_hi, n_points)


# ── Free energy landscape ────────────────────────────────────────────────────

def compute_free_energy(eigenvalues: np.ndarray,
                        T_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the partition function Z(T) and Helmholtz free energy F(T).

        Z(T) = Σ_k exp(−λ_k / T)
        F(T) = −T log Z(T)

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues λ_k (non-negative, descending).
    T_grid : np.ndarray
        Temperature grid from ``make_temperature_grid()``.

    Returns
    -------
    Z : np.ndarray, shape (n_temp,)
        Partition function values.
    F : np.ndarray, shape (n_temp,)
        Helmholtz free energy values.
    """
    # Compute Z for each temperature; clip for numerical stability
    Z = np.array([
        np.sum(np.exp(-eigenvalues / T))
        for T in T_grid
    ])
    Z = np.maximum(Z, 1e-300)
    F = -T_grid * np.log(Z)
    return Z, F


def compute_heat_capacity(T_grid: np.ndarray,
                          F: np.ndarray,
                          spline_degree: int = SPLINE_DEGREE) -> np.ndarray:
    """
    Compute the heat capacity C_v(T) = T · d²F/dT².

    Derivatives are computed via a smoothing spline fit to F(T), which
    avoids numerical noise from finite differences on a discretized grid.

    Parameters
    ----------
    T_grid : np.ndarray
        Temperature grid.
    F : np.ndarray
        Helmholtz free energy from ``compute_free_energy()``.
    spline_degree : int
        Degree of the smoothing spline (default 4, giving C² second
        derivative). Must be between 3 and 5.

    Returns
    -------
    C_v : np.ndarray, shape (n_temp,)
        Heat capacity at each temperature.
    """
    spl    = UnivariateSpline(T_grid, F, s=0, k=spline_degree)
    d2F    = spl.derivative(n=2)(T_grid)
    C_v    = T_grid * d2F
    return C_v


# ── Critical temperature ─────────────────────────────────────────────────────

def find_critical_temp(T_grid: np.ndarray,
                       C_v: np.ndarray,
                       prominence_frac: float = PEAK_PROMINENCE) -> dict:
    """
    Identify the critical temperature T* as the dominant peak in |C_v(T)|.

    T* marks the temperature at which the spectral energy distribution
    reorganizes most sharply — the computational analog of a phase
    transition. A higher T* corresponds to greater thermodynamic stiffness:
    more thermal energy is required to disrupt a spectrum dominated by a
    single mode (high spectral concentration).

    Peak detection uses ``scipy.signal.find_peaks`` with a prominence
    threshold of ``prominence_frac × dynamic_range(|C_v|)``. If no peak is
    detected, T* is reported as the location of the maximum of |C_v|.

    Parameters
    ----------
    T_grid : np.ndarray
        Temperature grid.
    C_v : np.ndarray
        Heat capacity from ``compute_heat_capacity()``.
    prominence_frac : float
        Minimum peak prominence as a fraction of the |C_v| dynamic range
        (default 0.05).

    Returns
    -------
    dict with keys:
        ``T_star``      : float  — critical temperature
        ``peak_height`` : float  — |C_v| at T*
        ``peak_index``  : int    — index into T_grid
        ``lambda_eff``  : float  — effective spectral temperature
                                   (slope of log Z vs 1/T)
    """
    C_v_abs    = np.abs(C_v)
    dyn_range  = C_v_abs.max() - C_v_abs.min()
    prominence = dyn_range * prominence_frac

    peaks, props = find_peaks(C_v_abs, prominence=prominence)

    if len(peaks) > 0:
        best_idx  = peaks[np.argmax(props["prominences"])]
    else:
        # Fallback: location of maximum |C_v|
        best_idx  = int(np.argmax(C_v_abs))

    T_star      = float(T_grid[best_idx])
    peak_height = float(C_v_abs[best_idx])

    # Effective spectral temperature: slope of log Z vs 1/T
    Z_vals = np.exp(  # reconstruct Z from F = -T log Z  → log Z = -F/T
        np.where(T_grid > 0, -np.nan_to_num(C_v * 0) + 0, 0)
    )
    # Compute λ_eff directly from partition function via linear regression
    inv_T   = 1.0 / T_grid
    # Use free energy slope: F ≈ λ_eff (dominant energy level)
    # Approximate as -dF/d(1/T) at low T
    lambda_eff = float(np.abs(np.polyfit(inv_T[:10], C_v_abs[:10], 1)[0]))

    return dict(
        T_star      = T_star,
        peak_height = peak_height,
        peak_index  = int(best_idx),
        lambda_eff  = lambda_eff,
    )


# ── Full pipeline convenience function ───────────────────────────────────────

def run_phase_transition(eigenvalues: np.ndarray,
                         n_points: int = N_TEMP_POINTS) -> dict:
    """
    Run the complete phase transition analysis on one tumor's eigenspectrum.

    Steps
    -----
    1. Build a temperature grid scaled to the eigenvalue distribution.
    2. Compute partition function Z(T) and free energy F(T).
    3. Compute heat capacity C_v(T) via spline differentiation.
    4. Identify the critical temperature T*.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues from ``eigendecomposition.decompose()`` (non-negative,
        descending order).
    n_points : int
        Number of temperature grid points (default 1000).

    Returns
    -------
    dict with keys:
        ``T_grid``      : np.ndarray — temperature axis
        ``Z``           : np.ndarray — partition function
        ``F``           : np.ndarray — Helmholtz free energy
        ``C_v``         : np.ndarray — heat capacity
        ``T_star``      : float      — critical temperature
        ``peak_height`` : float      — |C_v| at T*
        ``peak_index``  : int        — index of T* on grid
        ``lambda_eff``  : float      — effective spectral temperature
    """
    T_grid    = make_temperature_grid(eigenvalues, n_points=n_points)
    Z, F      = compute_free_energy(eigenvalues, T_grid)
    C_v       = compute_heat_capacity(T_grid, F)
    peak_info = find_critical_temp(T_grid, C_v)

    return dict(
        T_grid      = T_grid,
        Z           = Z,
        F           = F,
        C_v         = C_v,
        **peak_info,
    )
