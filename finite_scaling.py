"""finite_scaling.py
=====================
Finite-size numerical measurements for the impurity-driven QWZ project.

This module is the numerical-measurement layer between ``QWZmodel.py`` and
future exponent-fitting utilities.  It deliberately generates raw and
summary observables but does *not* optimize critical exponents, smooth curves,
or perform data-collapse fits.

Implemented backends
--------------------
1. Disorder ensembles
   * fixed-count (canonical) impurities on a finite two-dimensional sample;
   * Bernoulli (grand-canonical) impurities;
   * Bernoulli masks for transfer-matrix slices.

2. Bott-index finite-size measurements on an ``Lx x Ly`` torus
   * dense Hermitian diagonalization;
   * occupied-space Bott index using polar-unitarized projected position
     operators;
   * realization-level records and disorder summaries;
   * checkpointable parameter scans.

3. Quasi-one-dimensional transfer-matrix measurements
   * transverse QWZ slice Hamiltonian built from the public blocks in
     :class:`QWZmodel.QWZModel`;
   * stabilized QR accumulation of Lyapunov exponents;
   * quasi-1D localization length ``xi_M`` and normalized length
     ``Lambda_M = xi_M / M``;
   * realization-level records, summaries, and checkpointable scans.

Layer separation
----------------
``QWZmodel.py``
    Defines the Hamiltonian and its matrix blocks.

``finite_scaling.py``
    Generates finite-size observables: Bott outcomes and transfer-matrix
    localization data.

``exponents.py`` (future)
    Fits ``nu``, ``z``, and multifractal exponents from saved data.

Conventions
-----------
The QWZ real-space convention is inherited directly from ``QWZmodel.py``.
No hopping matrix is redefined here.  The transfer direction is ``x`` and the
transverse direction is periodic ``y``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Iterator, Literal, Mapping, Optional, Sequence
import csv
import json
import math
import os
import shutil

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.linalg as la

from QWZmodel import QWZModel, Site


ArrayC = npt.NDArray[np.complex128]
ArrayF = npt.NDArray[np.float64]
ArrayB = npt.NDArray[np.bool_]
EnsembleName = Literal["fixed_count", "bernoulli"]
RoundingRule = Literal["round", "floor", "ceil"]


# =============================================================================
# Generic validation and random-number helpers
# =============================================================================


def _validate_probability(value: float, name: str = "probability") -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]; received {value!r}")
    return value


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; received {value!r}")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive; received {value}")
    return value


def _validate_nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; received {value!r}")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative; received {value}")
    return value


def _coerce_rng(
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.random.Generator:
    """Return one RNG while preventing ambiguous ``rng`` + ``seed`` inputs."""
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both")
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be an instance of numpy.random.Generator")
        return rng
    return np.random.default_rng(seed)


def _round_count(expected: float, rule: RoundingRule) -> int:
    if rule == "round":
        # Python/NumPy bankers rounding is undesirable for a physical count.
        # For nonnegative expected counts, floor(x + 1/2) is conventional.
        return int(np.floor(float(expected) + 0.5))
    if rule == "floor":
        return int(np.floor(expected))
    if rule == "ceil":
        return int(np.ceil(expected))
    raise ValueError(f"unknown rounding rule: {rule!r}")


def _canonical_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer; received {seed!r}")
    return int(seed)


# =============================================================================
# Disorder ensembles
# =============================================================================


def all_lattice_sites(model: QWZModel) -> tuple[Site, ...]:
    """Return all sites in deterministic ``y``-major, then ``x`` order."""
    return tuple((x, y) for y in range(model.Ly) for x in range(model.Lx))


def impurity_count_from_density(
    n_sites: int,
    density: float,
    rounding: RoundingRule = "round",
) -> int:
    """Convert a requested density to a finite-system impurity count."""
    n_sites = _validate_positive_integer(n_sites, "n_sites")
    density = _validate_probability(density, "density")
    return min(n_sites, max(0, _round_count(density * n_sites, rounding)))


def sample_fixed_count_impurities(
    model: QWZModel,
    *,
    density: Optional[float] = None,
    n_impurities: Optional[int] = None,
    rounding: RoundingRule = "round",
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> tuple[Site, ...]:
    """Sample a uniform fixed-count impurity configuration.

    Exactly one of ``density`` or ``n_impurities`` must be supplied.  When a
    density is supplied, the actual finite-size density is

    ``n_impurities / model.n_sites``.
    """
    if (density is None) == (n_impurities is None):
        raise ValueError("Supply exactly one of density or n_impurities")

    if density is not None:
        count = impurity_count_from_density(model.n_sites, density, rounding)
    else:
        count = _validate_nonnegative_integer(n_impurities, "n_impurities")
        if count > model.n_sites:
            raise ValueError(
                f"n_impurities={count} exceeds the number of sites {model.n_sites}"
            )

    generator = _coerce_rng(rng=rng, seed=seed)
    if count == 0:
        return ()

    chosen = generator.choice(model.n_sites, size=count, replace=False)
    sites = [(int(index % model.Lx), int(index // model.Lx)) for index in chosen]
    return tuple(sorted(sites, key=lambda site: (site[1], site[0])))


def sample_bernoulli_impurities(
    model: QWZModel,
    *,
    density: float,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> tuple[Site, ...]:
    """Sample independent Bernoulli impurities on every finite-system site."""
    density = _validate_probability(density, "density")
    generator = _coerce_rng(rng=rng, seed=seed)
    mask = generator.random((model.Ly, model.Lx)) < density
    ys, xs = np.nonzero(mask)
    return tuple((int(x), int(y)) for y, x in zip(ys, xs))


def sample_impurities(
    model: QWZModel,
    *,
    ensemble: EnsembleName,
    density: float,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    rounding: RoundingRule = "round",
) -> tuple[Site, ...]:
    """Unified finite-system impurity sampler."""
    if ensemble == "fixed_count":
        return sample_fixed_count_impurities(
            model,
            density=density,
            rounding=rounding,
            rng=rng,
            seed=seed,
        )
    if ensemble == "bernoulli":
        return sample_bernoulli_impurities(
            model,
            density=density,
            rng=rng,
            seed=seed,
        )
    raise ValueError(f"unknown impurity ensemble: {ensemble!r}")


def random_slice_impurity_mask(
    width: int,
    *,
    density: float,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> ArrayB:
    """Independent Bernoulli impurity mask for one transfer-matrix slice."""
    width = _validate_positive_integer(width, "width")
    density = _validate_probability(density, "density")
    generator = _coerce_rng(rng=rng, seed=seed)
    return np.asarray(generator.random(width) < density, dtype=bool)


# =============================================================================
# Bott-index backend
# =============================================================================


def diagonalize_hermitian(matrix) -> tuple[ArrayF, ArrayC]:
    """Dense Hermitian eigendecomposition with ascending eigenvalues."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"matrix must be square; received shape {array.shape}")
    eigenvalues, eigenvectors = la.eigh(array, check_finite=True, overwrite_a=False)
    return np.asarray(eigenvalues, dtype=float), np.asarray(eigenvectors, dtype=complex)


def occupied_subspace(
    eigenvalues: npt.ArrayLike,
    eigenvectors: npt.ArrayLike,
    *,
    fermi_energy: float = 0.0,
    zero_tolerance: float = 1.0e-12,
    reject_fermi_degeneracy: bool = True,
) -> ArrayC:
    """Return columns spanning the states below the Fermi energy.

    An exact finite-system eigenvalue at the Fermi energy makes the projector
    convention ambiguous.  By default this raises ``ValueError`` rather than
    silently choosing an occupation.
    """
    evals = np.asarray(eigenvalues, dtype=float)
    evecs = np.asarray(eigenvectors, dtype=np.complex128)
    if evals.ndim != 1 or evecs.shape != (evals.size, evals.size):
        raise ValueError("eigenvalues/eigenvectors have incompatible shapes")

    fermi_energy = float(fermi_energy)
    zero_tolerance = abs(float(zero_tolerance))
    distance = np.abs(evals - fermi_energy)
    if reject_fermi_degeneracy and np.any(distance <= zero_tolerance):
        closest = float(np.min(distance))
        raise ValueError(
            "an eigenvalue lies at the Fermi energy within tolerance; "
            f"minimum |E-E_F|={closest:.3e}"
        )

    occupied = evals < (fermi_energy - zero_tolerance)
    if not np.any(occupied):
        raise ValueError("occupied subspace is empty")
    if np.all(occupied):
        raise ValueError("all states are occupied")
    return np.asarray(evecs[:, occupied], dtype=np.complex128)


def _polar_unitary(matrix: ArrayC, singular_tolerance: float = 1.0e-12) -> ArrayC:
    """Nearest unitary from an SVD polar decomposition."""
    u, singular_values, vh = la.svd(matrix, full_matrices=False, check_finite=True)
    if singular_values.size == 0 or float(np.min(singular_values)) <= singular_tolerance:
        raise np.linalg.LinAlgError(
            "projected position operator is numerically singular; "
            f"minimum singular value={float(np.min(singular_values)):.3e}"
        )
    return np.asarray(u @ vh, dtype=np.complex128)


def bott_index_from_occupied_states(
    occupied_vectors: npt.ArrayLike,
    positions: npt.ArrayLike,
    *,
    Lx: int,
    Ly: int,
    polar_unitarize: bool = True,
    singular_tolerance: float = 1.0e-12,
) -> dict[str, float | int]:
    """Compute the Bott index in the occupied subspace.

    The projected position operators are formed directly in the occupied
    basis.  Polar unitarization suppresses finite-size nonunitarity before the
    unitary commutator is diagonalized.

    Returns both the floating result and its nearest integer.  The sign follows
    the ``V U V^dagger U^dagger`` convention used here; absolute Bott values are
    convention independent for the current project.
    """
    vectors = np.asarray(occupied_vectors, dtype=np.complex128)
    coordinates = np.asarray(positions, dtype=float)
    Lx = _validate_positive_integer(Lx, "Lx")
    Ly = _validate_positive_integer(Ly, "Ly")

    if vectors.ndim != 2:
        raise ValueError("occupied_vectors must be a two-dimensional array")
    if coordinates.shape != (vectors.shape[0], 2):
        raise ValueError(
            "positions must have shape (Hilbert-space dimension, 2); "
            f"received {coordinates.shape}"
        )

    phase_x = np.exp(2.0j * np.pi * coordinates[:, 0] / Lx)
    phase_y = np.exp(2.0j * np.pi * coordinates[:, 1] / Ly)

    projected_x = vectors.conj().T @ (phase_x[:, None] * vectors)
    projected_y = vectors.conj().T @ (phase_y[:, None] * vectors)

    if polar_unitarize:
        projected_x = _polar_unitary(projected_x, singular_tolerance)
        projected_y = _polar_unitary(projected_y, singular_tolerance)

    commutator = (
        projected_y
        @ projected_x
        @ projected_y.conj().T
        @ projected_x.conj().T
    )
    phases = np.angle(la.eigvals(commutator, check_finite=True))
    bott_float = float(np.sum(phases) / (2.0 * np.pi))
    bott_integer = int(np.rint(bott_float))
    integer_residual = float(abs(bott_float - bott_integer))

    return {
        "bott_float": bott_float,
        "bott": bott_integer,
        "abs_bott": abs(bott_integer),
        "integer_residual": integer_residual,
        "n_occupied": int(vectors.shape[1]),
    }


def bott_index_from_eigensystem(
    eigenvalues: npt.ArrayLike,
    eigenvectors: npt.ArrayLike,
    positions: npt.ArrayLike,
    *,
    Lx: int,
    Ly: int,
    fermi_energy: float = 0.0,
    zero_tolerance: float = 1.0e-12,
    reject_fermi_degeneracy: bool = True,
    polar_unitarize: bool = True,
) -> dict[str, float | int]:
    """Bott index from a sorted Hermitian eigensystem."""
    occupied = occupied_subspace(
        eigenvalues,
        eigenvectors,
        fermi_energy=fermi_energy,
        zero_tolerance=zero_tolerance,
        reject_fermi_degeneracy=reject_fermi_degeneracy,
    )
    return bott_index_from_occupied_states(
        occupied,
        positions,
        Lx=Lx,
        Ly=Ly,
        polar_unitarize=polar_unitarize,
    )


def nearest_fermi_gap(
    eigenvalues: npt.ArrayLike,
    *,
    fermi_energy: float = 0.0,
) -> dict[str, float]:
    """Finite-system spectral distances immediately below and above ``E_F``."""
    evals = np.sort(np.asarray(eigenvalues, dtype=float))
    below = evals[evals < fermi_energy]
    above = evals[evals >= fermi_energy]
    if below.size == 0 or above.size == 0:
        return {
            "highest_occupied": np.nan,
            "lowest_unoccupied": np.nan,
            "spectral_gap": np.nan,
            "min_abs_energy": float(np.min(np.abs(evals - fermi_energy))),
        }
    highest = float(below[-1])
    lowest = float(above[0])
    return {
        "highest_occupied": highest,
        "lowest_unoccupied": lowest,
        "spectral_gap": lowest - highest,
        "min_abs_energy": float(np.min(np.abs(evals - fermi_energy))),
    }


def bott_realization(
    model: QWZModel,
    *,
    h: float,
    impurities: Sequence[Site],
    fermi_energy: float = 0.0,
    zero_tolerance: float = 1.0e-12,
    reject_fermi_degeneracy: bool = True,
) -> dict[str, float | int]:
    """Compute one finite-system Bott realization."""
    if not (model.pbc_x and model.pbc_y):
        raise ValueError("Bott calculations require periodic boundaries in both axes")

    sites = model.canonicalize_impurities(impurities)
    start = perf_counter()
    hamiltonian = model.hamiltonian(impurities=sites, h=h, sparse=False)
    eigenvalues, eigenvectors = diagonalize_hermitian(hamiltonian)
    bott = bott_index_from_eigensystem(
        eigenvalues,
        eigenvectors,
        model.positions(),
        Lx=model.Lx,
        Ly=model.Ly,
        fermi_energy=fermi_energy,
        zero_tolerance=zero_tolerance,
        reject_fermi_degeneracy=reject_fermi_degeneracy,
    )
    gap = nearest_fermi_gap(eigenvalues, fermi_energy=fermi_energy)
    elapsed = perf_counter() - start

    return {
        **bott,
        **gap,
        "Lx": model.Lx,
        "Ly": model.Ly,
        "L": model.Lx if model.Lx == model.Ly else np.nan,
        "h": float(h),
        "n_impurities": len(sites),
        "p_eff": len(sites) / model.n_sites,
        "fermi_energy": float(fermi_energy),
        "runtime_sec": float(elapsed),
    }


def make_realization_seed(base_seed: int, *indices: int) -> int:
    """Deterministically combine a base seed and integer scan indices."""
    entropy = [_canonical_seed(base_seed)] + [int(index) for index in indices]
    sequence = np.random.SeedSequence(entropy)
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def bott_point(
    model: QWZModel,
    *,
    h: float,
    density: float,
    n_realizations: int,
    ensemble: EnsembleName = "fixed_count",
    seed0: int = 0,
    rounding: RoundingRule = "round",
    fermi_energy: float = 0.0,
    zero_tolerance: float = 1.0e-12,
    reject_fermi_degeneracy: bool = True,
) -> pd.DataFrame:
    """Run several Bott realizations at one ``(model size, h)`` point."""
    n_realizations = _validate_positive_integer(n_realizations, "n_realizations")
    density = _validate_probability(density, "density")
    rows: list[dict[str, object]] = []

    for realization in range(n_realizations):
        seed = make_realization_seed(seed0, model.Lx, model.Ly, realization)
        impurities = sample_impurities(
            model,
            ensemble=ensemble,
            density=density,
            seed=seed,
            rounding=rounding,
        )
        result = bott_realization(
            model,
            h=h,
            impurities=impurities,
            fermi_energy=fermi_energy,
            zero_tolerance=zero_tolerance,
            reject_fermi_degeneracy=reject_fermi_degeneracy,
        )
        rows.append(
            {
                **result,
                "realization": realization,
                "seed": seed,
                "ensemble": ensemble,
                "p_target": density,
                "rounding": rounding if ensemble == "fixed_count" else "not_applicable",
            }
        )

    return pd.DataFrame(rows)


def summarize_bott_raw(
    raw: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("Lx", "Ly", "h", "ensemble", "p_target"),
) -> pd.DataFrame:
    """Summarize realization-level Bott records without fitting exponents."""
    required = {"bott", "abs_bott", "bott_float", "integer_residual", "p_eff"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"raw Bott table is missing columns: {sorted(missing)}")

    groups = [column for column in group_columns if column in raw.columns]
    if not groups:
        raise ValueError("no requested group columns are present")

    summary = (
        raw.groupby(groups, dropna=False, as_index=False)
        .agg(
            n=("bott", "count"),
            mean_B=("bott", "mean"),
            std_B=("bott", "std"),
            mean_absB=("abs_bott", "mean"),
            P_absB1=("abs_bott", lambda values: float(np.mean(np.asarray(values) == 1))),
            P_B0=("bott", lambda values: float(np.mean(np.asarray(values) == 0))),
            mean_B_float=("bott_float", "mean"),
            max_integer_residual=("integer_residual", "max"),
            mean_p_eff=("p_eff", "mean"),
            std_p_eff=("p_eff", "std"),
            mean_gap=("spectral_gap", "mean"),
            median_min_abs_energy=("min_abs_energy", "median"),
            mean_runtime_sec=("runtime_sec", "mean"),
        )
        .sort_values(groups)
        .reset_index(drop=True)
    )
    probability = summary["P_absB1"].to_numpy(dtype=float)
    count = summary["n"].to_numpy(dtype=float)
    summary["P_absB1_sem_binomial"] = np.sqrt(probability * (1.0 - probability) / count)
    if "Lx" in summary.columns and "Ly" in summary.columns:
        summary["L"] = np.where(summary["Lx"] == summary["Ly"], summary["Lx"], np.nan)
    return summary


# =============================================================================
# Transfer-matrix backend
# =============================================================================


@dataclass(frozen=True)
class TransferWorkspace:
    """Precomputed, width-dependent matrices for a QWZ transfer calculation."""

    model: QWZModel
    energy: float
    width: int
    slice_dim: int
    transfer_dim: int
    interslice: ArrayC
    interslice_adjoint: ArrayC
    identity_slice: ArrayC
    zero_slice: ArrayC
    lower_transfer: ArrayC
    constant_B: ArrayC


def transfer_model_from_template(template: QWZModel, width: int) -> QWZModel:
    """Construct a one-slice model carrying the template's physical parameters."""
    width = _validate_positive_integer(width, "width")
    return QWZModel(
        Lx=1,
        Ly=width,
        t=template.t,
        u=template.u,
        A=template.A,
        pbc=(False, True),
    )


def prepare_transfer_workspace(
    model: QWZModel,
    *,
    energy: float = 0.0,
) -> TransferWorkspace:
    """Precompute matrices that do not depend on the slice disorder."""
    if not model.pbc_y:
        raise ValueError("the transfer-matrix transverse direction y must be periodic")
    width = model.Ly
    slice_dim = model.norb * width
    transfer_dim = 2 * slice_dim
    identity = np.eye(slice_dim, dtype=np.complex128)
    zero = np.zeros((slice_dim, slice_dim), dtype=np.complex128)
    interslice = np.kron(np.eye(width, dtype=np.complex128), model.hopping_x)
    if abs(la.det(model.hopping_x)) <= 1.0e-14:
        raise np.linalg.LinAlgError("QWZ x-hopping block is singular; transfer recursion is undefined")
    constant_B = -la.solve(interslice, interslice.conj().T, assume_a="gen")
    lower = np.hstack([identity, zero])
    return TransferWorkspace(
        model=model,
        energy=float(energy),
        width=width,
        slice_dim=slice_dim,
        transfer_dim=transfer_dim,
        interslice=np.asarray(interslice),
        interslice_adjoint=np.asarray(interslice.conj().T),
        identity_slice=identity,
        zero_slice=zero,
        lower_transfer=lower,
        constant_B=np.asarray(constant_B),
    )


def build_slice_hamiltonian(
    model: QWZModel,
    impurity_mask: npt.ArrayLike,
    *,
    h: float,
) -> ArrayC:
    """Build one periodic transverse QWZ slice from model public blocks."""
    if not model.pbc_y:
        raise ValueError("slice Hamiltonian requires periodic y boundary conditions")
    mask = np.asarray(impurity_mask, dtype=bool)
    if mask.shape != (model.Ly,):
        raise ValueError(
            f"impurity_mask must have shape ({model.Ly},); received {mask.shape}"
        )

    width = model.Ly
    matrix = np.zeros((model.norb * width, model.norb * width), dtype=np.complex128)
    for y in range(width):
        current = slice(model.norb * y, model.norb * (y + 1))
        matrix[current, current] += model.local_onsite_block(h=h, is_impurity=bool(mask[y]))

        next_y = (y + 1) % width
        neighbor = slice(model.norb * next_y, model.norb * (next_y + 1))
        matrix[current, neighbor] += model.hopping_y
        matrix[neighbor, current] += model.hopping_y.conj().T

    return matrix


def transfer_matrix_for_slice(
    workspace: TransferWorkspace,
    impurity_mask: npt.ArrayLike,
    *,
    h: float,
) -> ArrayC:
    """Return the first-order spatial transfer matrix for one random slice."""
    h_slice = build_slice_hamiltonian(workspace.model, impurity_mask, h=h)
    upper_left = la.solve(
        workspace.interslice,
        workspace.energy * workspace.identity_slice - h_slice,
        assume_a="gen",
    )
    upper = np.hstack([upper_left, workspace.constant_B])
    return np.vstack([upper, workspace.lower_transfer])


def transfer_recurrence_residual(
    workspace: TransferWorkspace,
    impurity_mask: npt.ArrayLike,
    *,
    h: float,
    psi_previous: npt.ArrayLike,
    psi_current: npt.ArrayLike,
) -> float:
    """Numerically verify the slice Schrödinger recurrence."""
    previous = np.asarray(psi_previous, dtype=np.complex128)
    current = np.asarray(psi_current, dtype=np.complex128)
    if previous.shape != (workspace.slice_dim,) or current.shape != (workspace.slice_dim,):
        raise ValueError("psi_previous and psi_current must be slice vectors")

    transfer = transfer_matrix_for_slice(workspace, impurity_mask, h=h)
    propagated = transfer @ np.concatenate([current, previous])
    next_state = propagated[: workspace.slice_dim]
    slice_h = build_slice_hamiltonian(workspace.model, impurity_mask, h=h)
    residual = (
        workspace.interslice @ next_state
        + slice_h @ current
        + workspace.interslice_adjoint @ previous
        - workspace.energy * current
    )
    return float(la.norm(residual) / max(1.0, la.norm(current)))


def lyapunov_exponents_qr(
    model: QWZModel,
    *,
    h: float,
    density: float,
    n_slices: int,
    energy: float = 0.0,
    seed: int = 0,
    qr_interval: int = 1,
    positive_tolerance: float = 1.0e-10,
    verbose: bool = False,
    return_gammas: bool = True,
) -> dict[str, object]:
    """Compute transfer-matrix Lyapunov exponents by stabilized QR iteration.

    Disorder is Bernoulli independently on every transverse site and every
    slice.  ``model.Ly`` is the strip width; ``model.Lx`` is not used.
    """
    density = _validate_probability(density, "density")
    n_slices = _validate_positive_integer(n_slices, "n_slices")
    qr_interval = _validate_positive_integer(qr_interval, "qr_interval")
    seed = _canonical_seed(seed)
    positive_tolerance = abs(float(positive_tolerance))

    workspace = prepare_transfer_workspace(model, energy=energy)
    generator = np.random.default_rng(seed)
    q_matrix = np.eye(workspace.transfer_dim, dtype=np.complex128)
    log_diagonal = np.zeros(workspace.transfer_dim, dtype=float)
    completed_length = 0
    qr_count = 0
    start = perf_counter()

    progress_step = max(1, n_slices // 5)
    for beginning in range(0, n_slices, qr_interval):
        block_length = min(qr_interval, n_slices - beginning)
        for _ in range(block_length):
            mask = random_slice_impurity_mask(
                workspace.width,
                density=density,
                rng=generator,
            )
            q_matrix = transfer_matrix_for_slice(workspace, mask, h=h) @ q_matrix

        q_matrix, r_matrix = la.qr(
            q_matrix,
            mode="economic",
            overwrite_a=True,
            check_finite=False,
        )
        diagonal = np.maximum(np.abs(np.diag(r_matrix)), 1.0e-300)
        log_diagonal += np.log(diagonal)
        completed_length += block_length
        qr_count += 1

        if verbose and (completed_length % progress_step < block_length or completed_length == n_slices):
            print(f"  slices {completed_length}/{n_slices}")

    gammas = np.sort(np.real(log_diagonal / completed_length))
    positive = gammas[gammas > positive_tolerance]
    if positive.size == 0:
        gamma_min = np.nan
        xi_m = np.inf
        lambda_m = np.inf
    else:
        gamma_min = float(positive[0])
        xi_m = float(1.0 / gamma_min)
        lambda_m = float(xi_m / workspace.width)

    pairing_residual = float(np.max(np.abs(gammas + gammas[::-1])))
    elapsed = perf_counter() - start
    result: dict[str, object] = {
        "M": workspace.width,
        "h": float(h),
        "density": density,
        "p_target": density,
        "energy": float(energy),
        "n_slices": n_slices,
        "qr_interval": qr_interval,
        "n_qr": qr_count,
        "seed": seed,
        "gamma_min_pos": gamma_min,
        "xi_M": xi_m,
        "Lambda": lambda_m,
        "lyapunov_pairing_residual": pairing_residual,
        "runtime_sec": float(elapsed),
        "t": model.t,
        "u": model.u,
        "A": model.A,
    }
    if return_gammas:
        result["gammas"] = gammas
    return result


def summarize_transfer_raw(
    raw: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("M", "h", "density", "energy"),
) -> pd.DataFrame:
    """Summarize realization-level transfer records without fitting ``nu``."""
    required = {"gamma_min_pos", "xi_M", "Lambda", "runtime_sec"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"raw transfer table is missing columns: {sorted(missing)}")
    groups = [column for column in group_columns if column in raw.columns]
    if not groups:
        raise ValueError("no requested group columns are present")

    summary = (
        raw.groupby(groups, dropna=False, as_index=False)
        .agg(
            n=("Lambda", "count"),
            mean_gamma=("gamma_min_pos", "mean"),
            std_gamma=("gamma_min_pos", "std"),
            mean_xi=("xi_M", "mean"),
            std_xi=("xi_M", "std"),
            mean_Lambda=("Lambda", "mean"),
            std_Lambda=("Lambda", "std"),
            median_Lambda=("Lambda", "median"),
            min_Lambda=("Lambda", "min"),
            max_Lambda=("Lambda", "max"),
            max_pairing_residual=("lyapunov_pairing_residual", "max"),
            mean_runtime_sec=("runtime_sec", "mean"),
        )
        .sort_values(groups)
        .reset_index(drop=True)
    )
    summary["sem_Lambda"] = summary["std_Lambda"] / np.sqrt(summary["n"])
    return summary


# =============================================================================
# Checkpointed CSV I/O and scan drivers
# =============================================================================


def _float_key(value: float) -> str:
    return format(float(value), ".17g")


def _append_csv_row(path: os.PathLike[str] | str, row: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _safe_existing_table(
    path: os.PathLike[str] | str,
    *,
    required_columns: Iterable[str],
) -> pd.DataFrame:
    """Load a checkpoint or move an incompatible/corrupted file aside."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        table = pd.read_csv(path)
        missing = set(required_columns).difference(table.columns)
        if missing:
            raise ValueError(f"missing required columns {sorted(missing)}")
        return table
    except Exception as error:
        suffix = ".corrupted_or_incompatible.bak.csv"
        backup = path.with_name(path.stem + suffix)
        counter = 1
        while backup.exists():
            backup = path.with_name(path.stem + f".{counter}" + suffix)
            counter += 1
        shutil.move(str(path), str(backup))
        print(f"Checkpoint could not be used: {error!r}")
        print(f"Moved old file to: {backup}")
        return pd.DataFrame()


def write_run_metadata(
    path: os.PathLike[str] | str,
    metadata: Mapping[str, object],
) -> None:
    """Write JSON metadata next to raw numerical data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(metadata), handle, indent=2, sort_keys=True, default=str)


def run_bott_scan(
    *,
    sizes: Sequence[int],
    h_values: Sequence[float],
    template: QWZModel,
    density: float,
    n_realizations: int,
    ensemble: EnsembleName = "fixed_count",
    seed0: int = 0,
    rounding: RoundingRule = "round",
    fermi_energy: float = 0.0,
    raw_csv: os.PathLike[str] | str = "bott_raw.csv",
    summary_csv: Optional[os.PathLike[str] | str] = None,
    boundary_label: str = "unspecified",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Checkpointable square-torus Bott scan.

    One CSV row is saved per ``(L, h, realization)``.  This driver is intended
    for production data generation; exponent fitting belongs elsewhere.
    """
    density = _validate_probability(density, "density")
    n_realizations = _validate_positive_integer(n_realizations, "n_realizations")
    raw_csv = Path(raw_csv)
    if summary_csv is None:
        summary_csv = raw_csv.with_name(raw_csv.stem.replace("_raw", "") + "_summary.csv")
    summary_csv = Path(summary_csv)

    old = _safe_existing_table(
        raw_csv,
        required_columns=("L", "h", "realization", "seed"),
    )
    done = {
        (int(row.L), _float_key(row.h), int(row.realization), int(row.seed))
        for row in old.itertuples(index=False)
    }

    for size_index, size in enumerate(sizes):
        size = _validate_positive_integer(size, "size")
        model = QWZModel(
            Lx=size,
            Ly=size,
            t=template.t,
            u=template.u,
            A=template.A,
            pbc=True,
        )
        for h_index, h in enumerate(h_values):
            h = float(h)
            for realization in range(n_realizations):
                seed = make_realization_seed(
                    seed0,
                    11,
                    size_index,
                    size,
                    h_index,
                    realization,
                )
                key = (size, _float_key(h), realization, seed)
                if key in done:
                    if verbose:
                        print(f"[skip Bott] L={size}, h={h:.8g}, r={realization}")
                    continue

                if verbose:
                    print(f"[Bott] L={size}, h={h:.8g}, realization={realization}, seed={seed}")
                impurities = sample_impurities(
                    model,
                    ensemble=ensemble,
                    density=density,
                    seed=seed,
                    rounding=rounding,
                )
                result = bott_realization(
                    model,
                    h=h,
                    impurities=impurities,
                    fermi_energy=fermi_energy,
                )
                row = {
                    **result,
                    "L": size,
                    "realization": realization,
                    "seed": seed,
                    "ensemble": ensemble,
                    "p_target": density,
                    "rounding": rounding if ensemble == "fixed_count" else "not_applicable",
                    "boundary": boundary_label,
                    "t": template.t,
                    "u": template.u,
                    "A": template.A,
                }
                _append_csv_row(raw_csv, row)
                done.add(key)

    raw = pd.read_csv(raw_csv)
    summary = summarize_bott_raw(
        raw,
        group_columns=(
            "Lx",
            "Ly",
            "h",
            "ensemble",
            "p_target",
            "boundary",
            "t",
            "u",
            "A",
        ),
    )
    summary.to_csv(summary_csv, index=False)
    return raw, summary


def run_transfer_scan(
    *,
    widths: Sequence[int],
    h_values: Sequence[float],
    template: QWZModel,
    density: float,
    n_slices: int,
    n_seeds: int,
    seed0: int = 0,
    energy: float = 0.0,
    qr_interval: int = 1,
    raw_csv: os.PathLike[str] | str = "transfer_raw.csv",
    summary_csv: Optional[os.PathLike[str] | str] = None,
    boundary_label: str = "unspecified",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Checkpointable quasi-1D transfer-matrix scan."""
    density = _validate_probability(density, "density")
    n_slices = _validate_positive_integer(n_slices, "n_slices")
    n_seeds = _validate_positive_integer(n_seeds, "n_seeds")
    raw_csv = Path(raw_csv)
    if summary_csv is None:
        summary_csv = raw_csv.with_name(raw_csv.stem.replace("_raw", "") + "_summary.csv")
    summary_csv = Path(summary_csv)

    old = _safe_existing_table(
        raw_csv,
        required_columns=("M", "h", "seed"),
    )
    done = {
        (int(row.M), _float_key(row.h), int(row.seed))
        for row in old.itertuples(index=False)
    }

    for width_index, width in enumerate(widths):
        width = _validate_positive_integer(width, "width")
        model = transfer_model_from_template(template, width)
        for h_index, h in enumerate(h_values):
            h = float(h)
            for seed_index in range(n_seeds):
                seed = make_realization_seed(
                    seed0,
                    29,
                    width_index,
                    width,
                    h_index,
                    seed_index,
                )
                key = (width, _float_key(h), seed)
                if key in done:
                    if verbose:
                        print(f"[skip TM] M={width}, h={h:.8g}, seed={seed}")
                    continue

                if verbose:
                    print(
                        f"[TM] M={width}, h={h:.8g}, seed {seed_index + 1}/{n_seeds}, "
                        f"n_slices={n_slices}, seed={seed}"
                    )
                result = lyapunov_exponents_qr(
                    model,
                    h=h,
                    density=density,
                    n_slices=n_slices,
                    energy=energy,
                    seed=seed,
                    qr_interval=qr_interval,
                    verbose=False,
                    return_gammas=False,
                )
                row = {
                    **result,
                    "seed_index": seed_index,
                    "boundary": boundary_label,
                }
                _append_csv_row(raw_csv, row)
                done.add(key)

    raw = pd.read_csv(raw_csv)
    summary = summarize_transfer_raw(
        raw,
        group_columns=("M", "h", "density", "energy", "boundary", "t", "u", "A"),
    )
    summary.to_csv(summary_csv, index=False)
    return raw, summary


__all__ = [
    "TransferWorkspace",
    "all_lattice_sites",
    "impurity_count_from_density",
    "sample_fixed_count_impurities",
    "sample_bernoulli_impurities",
    "sample_impurities",
    "random_slice_impurity_mask",
    "diagonalize_hermitian",
    "occupied_subspace",
    "bott_index_from_occupied_states",
    "bott_index_from_eigensystem",
    "nearest_fermi_gap",
    "bott_realization",
    "bott_point",
    "summarize_bott_raw",
    "make_realization_seed",
    "transfer_model_from_template",
    "prepare_transfer_workspace",
    "build_slice_hamiltonian",
    "transfer_matrix_for_slice",
    "transfer_recurrence_residual",
    "lyapunov_exponents_qr",
    "summarize_transfer_raw",
    "write_run_metadata",
    "run_bott_scan",
    "run_transfer_scan",
]
