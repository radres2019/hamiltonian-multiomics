"""
hamiltonian-multiomics
======================
Hamiltonian eigendecomposition framework for pan-cancer multiomics survival
analysis.

Modules
-------
eigendecomposition
    Preprocessing, Hamiltonian construction, eigendecomposition, survival-
    ranked mode selection, eigenindex, and universal malignancy coordinate η.

phase_transition
    Free energy landscape, heat capacity, and critical temperature T* for
    the Hamiltonian eigenspectrum.
"""

from .eigendecomposition import (
    preprocess_modality,
    align_modalities,
    build_hamiltonian,
    decompose,
    compute_eigenscores,
    spectral_concentration,
    rank_modes,
    compute_eigenindex,
    compute_eta,
    run_eigendecomposition,
)

from .phase_transition import (
    make_temperature_grid,
    compute_free_energy,
    compute_heat_capacity,
    find_critical_temp,
    run_phase_transition,
)

__all__ = [
    "preprocess_modality",
    "align_modalities",
    "build_hamiltonian",
    "decompose",
    "compute_eigenscores",
    "spectral_concentration",
    "rank_modes",
    "compute_eigenindex",
    "compute_eta",
    "run_eigendecomposition",
    "make_temperature_grid",
    "compute_free_energy",
    "compute_heat_capacity",
    "find_critical_temp",
    "run_phase_transition",
]
