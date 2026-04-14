"""
eigendecomposition.py
=====================
Hamiltonian construction and eigendecomposition for multimodal omics data.

The covariance matrix of the standardized feature matrix is treated as a
Hamiltonian H = X^T X / n. Its eigenvectors represent the collective modes
of coordinated molecular variation across modalities; its eigenvalues
represent their energetic scale. Survival-discriminating modes are identified
by ranking eigenscores against a binary survival label.

Usage
-----
    from src.eigendecomposition import preprocess_modality, build_hamiltonian, \
        rank_modes, compute_eigenindex, compute_eta

Author : John D. Mayfield, M.D., Ph.D., M.Sc.
         Department of Radiology, Massachusetts General Hospital
         Harvard Medical School
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

MAX_MISSING_FRAC = 0.40   # Drop features with > 40% missing values
N_FEATURES       = 500    # Maximum features per modality, selected by variance
N_MODES          = 30     # Number of eigenmodes to retain
TOP_K_MODES      = 3      # Number of top survival modes used for eigenindex
KM_QUANTILE      = 0.50   # Quantile threshold for binary survival label


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_modality(df: pd.DataFrame,
                        max_missing: float = MAX_MISSING_FRAC,
                        n_features: int = N_FEATURES,
                        modality_prefix: str | None = None) -> pd.DataFrame:
    """
    Prepare a single omics modality for eigendecomposition.

    Steps
    -----
    1. Drop features with missing fraction above ``max_missing``.
    2. Impute remaining missing values with the per-feature median.
    3. Retain the top ``n_features`` by variance.
    4. Optionally prefix column names with ``modality_prefix::gene``.

    Parameters
    ----------
    df : pd.DataFrame
        Samples × features matrix (rows = samples, columns = features).
    max_missing : float
        Maximum allowed missing fraction per feature (default 0.40).
    n_features : int
        Maximum number of features to retain (default 500).
    modality_prefix : str or None
        If provided, columns are renamed to ``prefix::original_name``.

    Returns
    -------
    pd.DataFrame
        Cleaned, imputed, variance-filtered feature matrix.
    """
    df = df.copy()

    # Drop high-missing features
    df = df.loc[:, df.isnull().mean() < max_missing]

    # Median imputation
    df = df.fillna(df.median())

    # Variance-based feature selection
    if df.shape[1] > n_features:
        top_cols = df.var().nlargest(n_features).index
        df = df[top_cols]

    # Optional modality prefix
    if modality_prefix is not None:
        df.columns = [f"{modality_prefix}::{c}" for c in df.columns]

    return df


def align_modalities(modality_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Align multiple modality DataFrames to their shared sample set and
    concatenate into a single multiomics matrix.

    Only samples present in all modalities are retained (complete-data
    alignment). Features from each modality are prefixed with the modality
    name before concatenation.

    Parameters
    ----------
    modality_dict : dict
        Mapping of modality name → preprocessed DataFrame
        (samples × features, index = sample IDs).

    Returns
    -------
    pd.DataFrame
        Aligned multiomics matrix (samples × all features).

    Raises
    ------
    ValueError
        If no modalities are provided or no shared samples exist.
    """
    if not modality_dict:
        raise ValueError("modality_dict is empty.")

    frames = list(modality_dict.values())
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.join(frame, how="inner")

    if merged.shape[0] == 0:
        raise ValueError(
            "No shared samples across modalities after alignment. "
            "Check that sample IDs are consistent."
        )

    return merged


# ── Hamiltonian construction ──────────────────────────────────────────────────

def build_hamiltonian(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """
    Standardize the feature matrix and construct the sample covariance
    Hamiltonian H = X^T X / n.

    The covariance matrix is treated as a Hamiltonian because its
    eigenspectrum plays the same structural role as the Hamiltonian operator
    in quantum mechanics: eigenvectors are the collective modes of molecular
    co-variation and eigenvalues are their energetic scale.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Raw (unscaled) feature matrix.

    Returns
    -------
    H : np.ndarray, shape (n_features, n_features)
        Symmetric sample covariance matrix (the Hamiltonian).
    scaler : StandardScaler
        Fitted scaler (mean and scale) for later projection of new data.
    """
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X)
    n      = X_std.shape[0]
    H      = (X_std.T @ X_std) / n
    return H, scaler


# ── Eigendecomposition ────────────────────────────────────────────────────────

def decompose(H: np.ndarray,
              n_modes: int = N_MODES) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose the Hamiltonian H into eigenvalues and eigenvectors.

    Uses ``numpy.linalg.eigh`` (LAPACK dsyevd), which exploits symmetry for
    exact decomposition. Eigenvalues are returned in descending order.
    Negative eigenvalues arising from numerical noise are clipped to zero.

    Parameters
    ----------
    H : np.ndarray, shape (n_features, n_features)
        Symmetric Hamiltonian matrix.
    n_modes : int
        Number of leading eigenmodes to retain (default 30).

    Returns
    -------
    eigenvalues : np.ndarray, shape (n_modes,)
        Eigenvalues λ_k in descending order (non-negative).
    eigenvectors : np.ndarray, shape (n_features, n_modes)
        Corresponding eigenvectors v_k (columns), sorted descending.
    """
    evals, evecs = np.linalg.eigh(H)

    # Descending order
    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    # Clip numerical negatives
    evals = np.maximum(evals, 0.0)

    n_keep = min(n_modes, len(evals))
    return evals[:n_keep], evecs[:, :n_keep]


def compute_eigenscores(X_std: np.ndarray,
                        eigenvectors: np.ndarray) -> np.ndarray:
    """
    Project standardized samples onto the eigenvector basis.

    The eigenscore for sample i and mode k is:
        s_{ik} = x_i^T v_k

    Parameters
    ----------
    X_std : np.ndarray, shape (n_samples, n_features)
        Standardized feature matrix (output of StandardScaler.transform).
    eigenvectors : np.ndarray, shape (n_features, n_modes)
        Eigenvector matrix from ``decompose()``.

    Returns
    -------
    scores : np.ndarray, shape (n_samples, n_modes)
        Per-sample eigenscores for each mode.
    """
    return X_std @ eigenvectors


def spectral_concentration(eigenvalues: np.ndarray) -> float:
    """
    Compute spectral concentration c = λ_1 / Σ λ_k.

    Spectral concentration measures the fraction of total variance carried
    by the dominant eigenmode. High c indicates that molecular co-variation
    is organized into one dominant collective pattern (e.g., VHL-driven
    clear cell RCC). Low c indicates distributed multi-driver architecture
    (e.g., GBM).

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues in descending order (non-negative).

    Returns
    -------
    float
        Spectral concentration in [0, 1].
    """
    total = eigenvalues.sum()
    if total < 1e-12:
        return 0.0
    return float(eigenvalues[0] / total)


# ── Survival-ranked mode selection ───────────────────────────────────────────

def _binary_survival_label(os_time: np.ndarray,
                            os_event: np.ndarray,
                            quantile: float = KM_QUANTILE) -> np.ndarray:
    """
    Construct a binary survival label for AUC-based mode ranking.

    A sample is labelled 1 (short survival event) if:
        - it has a recorded event (os_event == 1), AND
        - its survival time is at or below the ``quantile``-th quantile
          of survival times among all event samples.

    Parameters
    ----------
    os_time : np.ndarray
        Overall survival time (days).
    os_event : np.ndarray
        Event indicator (1 = event, 0 = censored).
    quantile : float
        Quantile threshold applied to event times (default 0.5).

    Returns
    -------
    y_bin : np.ndarray, shape (n_samples,)
        Binary label array (int, 0 or 1).
    """
    event_mask = os_event.astype(bool)
    if event_mask.sum() == 0:
        return np.zeros(len(os_time), dtype=int)

    threshold = np.quantile(os_time[event_mask], quantile)
    y_bin = ((os_time <= threshold) & event_mask).astype(int)
    return y_bin


def rank_modes(scores: np.ndarray,
               os_time: np.ndarray,
               os_event: np.ndarray,
               top_k: int = TOP_K_MODES) -> tuple[list[int], list[float]]:
    """
    Rank eigenmodes by their ability to discriminate survival (AUC).

    Each mode's eigenscore vector is evaluated against a binary survival
    label. AUC is direction-corrected: max(AUC, 1 - AUC) so that modes
    predicting better or worse survival both score above 0.5.

    Parameters
    ----------
    scores : np.ndarray, shape (n_samples, n_modes)
        Eigenscores from ``compute_eigenscores()``.
    os_time : np.ndarray
        Overall survival time.
    os_event : np.ndarray
        Event indicator.
    top_k : int
        Number of top modes to return (default 3).

    Returns
    -------
    top_indices : list[int]
        Indices of the top-k modes sorted by descending AUC.
    top_aucs : list[float]
        Corresponding AUC values.
    """
    y_bin = _binary_survival_label(os_time, os_event)
    n_events = int(y_bin.sum())

    mode_aucs = []
    for k in range(scores.shape[1]):
        if n_events < 5 or (y_bin == 0).sum() < 5:
            mode_aucs.append((k, 0.5))
            continue
        try:
            auc = roc_auc_score(y_bin, scores[:, k])
            mode_aucs.append((k, max(auc, 1.0 - auc)))
        except Exception:
            mode_aucs.append((k, 0.5))

    mode_aucs.sort(key=lambda x: x[1], reverse=True)
    top_indices = [m[0] for m in mode_aucs[:top_k]]
    top_aucs    = [m[1] for m in mode_aucs[:top_k]]
    return top_indices, top_aucs


def compute_eigenindex(scores: np.ndarray,
                       top_indices: list[int]) -> np.ndarray:
    """
    Compute the composite eigenindex as the mean of the top survival modes.

    Parameters
    ----------
    scores : np.ndarray, shape (n_samples, n_modes)
        Eigenscores from ``compute_eigenscores()``.
    top_indices : list[int]
        Mode indices from ``rank_modes()``.

    Returns
    -------
    eigenindex : np.ndarray, shape (n_samples,)
        Composite eigenindex (mean across selected modes).
    """
    return scores[:, top_indices].mean(axis=1)


def compute_eta(eigenindex: np.ndarray, c: float) -> np.ndarray:
    """
    Compute the universal malignancy coordinate η.

        η_i = (s_i − median(s)) / c

    Subtracting the cohort median centers η at zero. Dividing by spectral
    concentration c normalizes for differences in spectral architecture
    across tumor types, placing patients on a common scale regardless of
    cancer type.

    Parameters
    ----------
    eigenindex : np.ndarray
        Composite eigenindex from ``compute_eigenindex()``.
    c : float
        Spectral concentration from ``spectral_concentration()``.

    Returns
    -------
    eta : np.ndarray
        Universal malignancy coordinate η.
    """
    if c < 1e-9:
        c = 1e-9  # guard against division by zero
    return (eigenindex - np.nanmedian(eigenindex)) / c


# ── Full pipeline convenience function ───────────────────────────────────────

def run_eigendecomposition(modality_dict: dict[str, pd.DataFrame],
                           os_time: np.ndarray,
                           os_event: np.ndarray,
                           n_modes: int = N_MODES,
                           top_k: int = TOP_K_MODES) -> dict:
    """
    Run the complete Hamiltonian eigendecomposition pipeline on one cohort.

    Steps
    -----
    1. Align modalities to shared samples.
    2. Build Hamiltonian H = X^T X / n.
    3. Decompose H into eigenvalues and eigenvectors.
    4. Compute per-sample eigenscores.
    5. Rank modes by survival AUC; compute composite eigenindex.
    6. Compute spectral concentration c and universal malignancy coordinate η.

    Parameters
    ----------
    modality_dict : dict
        Mapping of modality name → preprocessed DataFrame.
        Each DataFrame should have already been passed through
        ``preprocess_modality()`` with the appropriate prefix.
    os_time : np.ndarray
        Overall survival time aligned to the sample index of the
        aligned multiomics matrix.
    os_event : np.ndarray
        Event indicator aligned to the same index.
    n_modes : int
        Number of eigenmodes to retain (default 30).
    top_k : int
        Number of top survival modes for the eigenindex (default 3).

    Returns
    -------
    dict with keys:
        ``eigenvalues``    : np.ndarray, shape (n_modes,)
        ``eigenvectors``   : np.ndarray, shape (n_features, n_modes)
        ``scaler``         : fitted StandardScaler
        ``feature_names``  : list[str]
        ``scores``         : np.ndarray, shape (n_samples, n_modes)
        ``eigenindex``     : np.ndarray, shape (n_samples,)
        ``eta``            : np.ndarray, shape (n_samples,)
        ``spectral_conc``  : float
        ``top_mode_indices``: list[int]
        ``top_mode_aucs``  : list[float]
        ``n_samples``      : int
        ``n_features``     : int
    """
    # Align
    X_df          = align_modalities(modality_dict)
    feature_names = list(X_df.columns)
    X             = X_df.values.astype(float)

    # Align survival to sample axis
    n_samples     = X.shape[0]
    os_time       = os_time[:n_samples]
    os_event      = os_event[:n_samples]

    # Hamiltonian
    H, scaler     = build_hamiltonian(X)
    X_std         = scaler.transform(X)

    # Eigendecomposition
    evals, evecs  = decompose(H, n_modes=n_modes)

    # Eigenscores
    scores        = compute_eigenscores(X_std, evecs)

    # Survival-ranked modes
    top_idx, top_aucs = rank_modes(scores, os_time, os_event, top_k=top_k)

    # Composite eigenindex and eta
    eigenindex    = compute_eigenindex(scores, top_idx)
    c             = spectral_concentration(evals)
    eta           = compute_eta(eigenindex, c)

    return dict(
        eigenvalues     = evals,
        eigenvectors    = evecs,
        scaler          = scaler,
        feature_names   = feature_names,
        scores          = scores,
        eigenindex      = eigenindex,
        eta             = eta,
        spectral_conc   = c,
        top_mode_indices= top_idx,
        top_mode_aucs   = top_aucs,
        n_samples       = n_samples,
        n_features      = X.shape[1],
    )
