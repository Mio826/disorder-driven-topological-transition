"""exponents.py
================
Scaling analysis and critical-exponent extraction for the impurity-driven
QWZ project.

This module is intentionally *analysis only*.  It does not construct a
Hamiltonian and it does not generate disorder realizations.  Its inputs are
``pandas.DataFrame`` objects produced by numerical backends such as
``finite_scaling.py`` or equivalent external calculations.

Public analyses
---------------
1. Correlation/localization exponent ``nu``
   * :func:`fit_nu_from_bott` uses a binomial likelihood for Bott outcomes.
   * :func:`fit_nu_from_transfer` uses weighted nonlinear least squares for
     normalized quasi-1D localization lengths.
   * :func:`diagnose_bott_width_scaling` provides a transparent width-based
     diagnostic, but is not intended as the primary estimator.

2. Dynamical exponent ``z``
   * :func:`fit_z_from_dos_powerlaw` fits
     ``rho(E) = A |E|^(d/z - 1)`` in a user-selected energy window.

3. Multifractal exponents
   * :func:`site_probabilities` and
     :func:`generalized_participation_ratios` construct wavefunction moments.
   * :func:`fit_multifractal_spectrum` extracts average and typical
     ``tau(q)`` and ``D_q`` from realization-level ``P_q(L)`` data.

Design
------
The numerical core is functional.  Immutable configuration dataclasses record
all fitting choices, while structured result dataclasses retain parameters,
uncertainties, predictions, residuals, bootstrap samples, and metadata.

The first production version deliberately keeps the fitting models explicit
and inspectable.  More elaborate logarithmic corrections and automated model
selection can be added without changing the top-level API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence
import math

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.special import expit, xlogy, xlog1py
from scipy.stats import norm


ArrayF = npt.NDArray[np.float64]
ArrayC = npt.NDArray[np.complex128]


__all__ = [
    "BottNuConfig",
    "TransferNuConfig",
    "DOSZConfig",
    "MultifractalConfig",
    "ScalingFitResult",
    "MultifractalResult",
    "validate_bott_data",
    "validate_transfer_data",
    "validate_dos_data",
    "validate_multifractal_data",
    "diagnose_bott_width_scaling",
    "fit_nu_from_bott",
    "fit_nu_from_transfer",
    "fit_z_from_dos_powerlaw",
    "site_probabilities",
    "box_probabilities",
    "generalized_participation_ratios",
    "fit_multifractal_spectrum",
    "legendre_spectrum",
    "plot_bott_nu_result",
    "plot_transfer_nu_result",
    "plot_dos_z_result",
    "plot_multifractal_result",
]


# =============================================================================
# Configuration dataclasses
# =============================================================================


@dataclass(frozen=True)
class BottNuConfig:
    """Configuration for binomial Bott-probability scaling.

    The scaling model is

    ``P(L,h) = logistic(sum_{k=0}^K c_k x^k)``,
    ``x = (h-h_c) L^(1/nu)``.

    Raw realization data are preferred.  Summary data are accepted when they
    contain a probability column and a sample-count column.
    """

    size_col: str = "L"
    h_col: str = "h"
    outcome_col: str = "abs_bott"
    probability_col: str = "P_absB1"
    count_col: str = "n"
    min_size: Optional[float] = None
    h_window: Optional[tuple[float, float]] = None
    polynomial_order: int = 3
    nu_bounds: tuple[float, float] = (0.2, 6.0)
    hc_bounds: Optional[tuple[float, float]] = None
    coefficient_bound: float = 50.0
    n_bootstrap: int = 0
    confidence_level: float = 0.95
    seed: int = 0
    maxiter: int = 5000


@dataclass(frozen=True)
class TransferNuConfig:
    """Configuration for transfer-matrix one-parameter scaling.

    The leading model is

    ``Lambda(M,h) = sum_{k=0}^K c_k x^k``,
    ``x = (h-h_c) M^(1/nu)``.

    An optional irrelevant correction can be included as

    ``M^y sum_{k=0}^{K_i} d_k x^k``, with ``y<0``.
    """

    width_col: str = "M"
    h_col: str = "h"
    raw_value_col: str = "Lambda"
    summary_value_col: str = "mean_Lambda"
    summary_error_col: str = "sem_Lambda"
    min_width: Optional[float] = None
    h_window: Optional[tuple[float, float]] = None
    polynomial_order: int = 3
    include_irrelevant: bool = False
    irrelevant_order: int = 0
    irrelevant_exponent: Optional[float] = None
    nu_bounds: tuple[float, float] = (0.2, 8.0)
    hc_bounds: Optional[tuple[float, float]] = None
    coefficient_bound: float = 100.0
    error_floor_fraction: float = 1.0e-4
    n_bootstrap: int = 0
    confidence_level: float = 0.95
    seed: int = 0
    max_nfev: int = 20000


@dataclass(frozen=True)
class DOSZConfig:
    """Configuration for a DOS power-law estimate of ``z``.

    The fitted form is

    ``rho(E) = A |E|^alpha``, where ``alpha = d/z - 1``.

    This is a controlled estimator only within an energy window where finite
    level spacing, ultraviolet curvature, offsets, and logarithmic corrections
    are negligible.
    """

    energy_col: str = "E"
    dos_col: str = "rho"
    error_col: Optional[str] = None
    dimension: float = 2.0
    energy_window: Optional[tuple[float, float]] = None
    min_points: int = 4
    n_bootstrap: int = 0
    confidence_level: float = 0.95
    seed: int = 0


@dataclass(frozen=True)
class MultifractalConfig:
    """Configuration for ``P_q(L)`` multifractal scaling."""

    size_col: str = "L"
    q_col: str = "q"
    pq_col: str = "Pq"
    realization_col: Optional[str] = "realization"
    min_size: Optional[float] = None
    q_values: Optional[tuple[float, ...]] = None
    compute_typical: bool = True
    compute_average: bool = True
    n_bootstrap: int = 0
    confidence_level: float = 0.95
    seed: int = 0


# =============================================================================
# Result dataclasses
# =============================================================================


@dataclass
class ScalingFitResult:
    """Structured result for a scalar-exponent fit."""

    method: str
    success: bool
    parameters: dict[str, float]
    standard_errors: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]
    objective_value: float
    degrees_of_freedom: int
    predictions: pd.DataFrame
    residuals: pd.DataFrame
    bootstrap_samples: Optional[pd.DataFrame]
    metadata: dict[str, Any]
    message: str

    def summary_frame(self) -> pd.DataFrame:
        """Return parameters and uncertainty information as a table."""
        rows: list[dict[str, Any]] = []
        for name, value in self.parameters.items():
            ci = self.confidence_intervals.get(name, (np.nan, np.nan))
            rows.append(
                {
                    "parameter": name,
                    "estimate": value,
                    "standard_error": self.standard_errors.get(name, np.nan),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                }
            )
        return pd.DataFrame(rows)


@dataclass
class MultifractalResult:
    """Structured result for a multifractal spectrum."""

    spectrum: pd.DataFrame
    scale_table: pd.DataFrame
    bootstrap_samples: Optional[pd.DataFrame]
    metadata: dict[str, Any]
    success: bool
    message: str


# =============================================================================
# Generic validation helpers
# =============================================================================


def _require_dataframe(data: pd.DataFrame, name: str = "data") -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas.DataFrame")
    if data.empty:
        raise ValueError(f"{name} is empty")
    return data.copy()


def _require_columns(data: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns).difference(data.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _finite_rows(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    array = data[list(columns)].apply(pd.to_numeric, errors="coerce")
    mask = np.all(np.isfinite(array.to_numpy(dtype=float)), axis=1)
    return data.loc[mask].copy()


def _validate_interval(interval: Optional[tuple[float, float]], name: str) -> None:
    if interval is None:
        return
    if len(interval) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    low, high = map(float, interval)
    if not np.isfinite(low) or not np.isfinite(high) or not low < high:
        raise ValueError(f"{name} must satisfy finite low < high")


def _validate_confidence(level: float) -> float:
    level = float(level)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    return level


def _normal_interval(value: float, error: float, level: float) -> tuple[float, float]:
    if not np.isfinite(error):
        return (np.nan, np.nan)
    zscore = float(norm.ppf(0.5 + 0.5 * level))
    return (float(value - zscore * error), float(value + zscore * error))


def _percentile_interval(values: ArrayF, level: float) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (np.nan, np.nan)
    tail = 50.0 * (1.0 - level)
    return tuple(map(float, np.percentile(finite, [tail, 100.0 - tail])))


def _numerical_hessian(function, point: ArrayF, rel_step: float = 2.0e-4) -> ArrayF:
    """Central finite-difference Hessian for modest parameter counts."""
    point = np.asarray(point, dtype=float)
    n = point.size
    steps = rel_step * np.maximum(1.0, np.abs(point))
    hessian = np.zeros((n, n), dtype=float)
    f0 = float(function(point))

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = steps[i]
        hessian[i, i] = (
            float(function(point + ei)) - 2.0 * f0 + float(function(point - ei))
        ) / steps[i] ** 2

        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = steps[j]
            value = (
                float(function(point + ei + ej))
                - float(function(point + ei - ej))
                - float(function(point - ei + ej))
                + float(function(point - ei - ej))
            ) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = value
            hessian[j, i] = value

    return hessian


def _covariance_from_hessian(function, point: ArrayF) -> Optional[ArrayF]:
    try:
        hessian = _numerical_hessian(function, point)
        covariance = np.linalg.pinv(hessian, rcond=1.0e-10)
        if not np.all(np.isfinite(covariance)):
            return None
        return covariance
    except Exception:
        return None


def _linear_fit(
    x: ArrayF,
    y: ArrayF,
    sigma: Optional[ArrayF] = None,
) -> tuple[ArrayF, ArrayF, ArrayF, float, int]:
    """Fit ``y = intercept + slope*x`` and return covariance/residuals."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones_like(x), x])

    if sigma is None:
        weights = np.ones_like(y)
    else:
        sigma = np.asarray(sigma, dtype=float)
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError("all supplied linear-fit errors must be finite and positive")
        weights = 1.0 / sigma**2

    sqrt_w = np.sqrt(weights)
    weighted_design = design * sqrt_w[:, None]
    weighted_y = y * sqrt_w
    beta, *_ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
    prediction = design @ beta
    residual = y - prediction
    dof = max(0, len(y) - len(beta))
    chi2 = float(np.sum(weights * residual**2))

    information = design.T @ (weights[:, None] * design)
    covariance = np.linalg.pinv(information, rcond=1.0e-12)
    if sigma is None and dof > 0:
        covariance = covariance * (chi2 / dof)

    return beta, covariance, residual, chi2, dof


# =============================================================================
# Data contracts
# =============================================================================


def _prepare_bott_counts(data: pd.DataFrame, config: BottNuConfig) -> pd.DataFrame:
    data = _require_dataframe(data, "Bott data")
    _validate_interval(config.h_window, "h_window")

    if config.size_col not in data.columns:
        if config.size_col == "L" and {"Lx", "Ly"}.issubset(data.columns):
            equal = data["Lx"].to_numpy() == data["Ly"].to_numpy()
            if not np.all(equal):
                raise ValueError("Bott scaling requires square samples when size_col='L'")
            data["L"] = data["Lx"].to_numpy()
        else:
            raise ValueError(f"Bott data lacks size column {config.size_col!r}")
    _require_columns(data, [config.size_col, config.h_col], "Bott data")

    if config.outcome_col in data.columns:
        work = _finite_rows(data, [config.size_col, config.h_col, config.outcome_col])
        outcome = work[config.outcome_col].to_numpy(dtype=float)
        success = np.isclose(np.abs(outcome), 1.0).astype(int)
        work = work.assign(_success=success)
        counts = (
            work.groupby([config.size_col, config.h_col], as_index=False)
            .agg(k=("_success", "sum"), n=("_success", "count"))
        )
    else:
        _require_columns(
            data,
            [config.size_col, config.h_col, config.probability_col, config.count_col],
            "Bott summary",
        )
        counts = _finite_rows(
            data,
            [config.size_col, config.h_col, config.probability_col, config.count_col],
        )[[config.size_col, config.h_col, config.probability_col, config.count_col]].copy()
        probability = counts[config.probability_col].to_numpy(dtype=float)
        n = counts[config.count_col].to_numpy(dtype=float)
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError("Bott probabilities must lie in [0,1]")
        if np.any(n <= 0.0) or np.any(np.abs(n - np.rint(n)) > 1.0e-8):
            raise ValueError("Bott sample counts must be positive integers")
        counts["n"] = np.rint(n).astype(int)
        counts["k"] = np.rint(probability * counts["n"].to_numpy()).astype(int)
        counts = counts[[config.size_col, config.h_col, "k", "n"]]

    counts = counts.rename(columns={config.size_col: "L", config.h_col: "h"})
    counts["L"] = counts["L"].astype(float)
    counts["h"] = counts["h"].astype(float)
    counts["k"] = counts["k"].astype(int)
    counts["n"] = counts["n"].astype(int)

    if np.any(counts["L"] <= 0.0):
        raise ValueError("all Bott sizes must be positive")
    if np.any(counts["k"] < 0) or np.any(counts["k"] > counts["n"]):
        raise ValueError("Bott successes must satisfy 0 <= k <= n")

    if config.min_size is not None:
        counts = counts[counts["L"] >= float(config.min_size)]
    if config.h_window is not None:
        low, high = map(float, config.h_window)
        counts = counts[(counts["h"] >= low) & (counts["h"] <= high)]

    counts = counts.sort_values(["L", "h"]).reset_index(drop=True)
    if counts.empty:
        raise ValueError("no Bott data remain after filtering")
    if counts["L"].nunique() < 2:
        raise ValueError("Bott scaling requires at least two system sizes")
    if len(counts) <= config.polynomial_order + 3:
        raise ValueError("too few Bott points for the requested polynomial order")

    counts["p_obs"] = counts["k"] / counts["n"]
    return counts


def validate_bott_data(data: pd.DataFrame, config: Optional[BottNuConfig] = None) -> pd.DataFrame:
    """Validate Bott data and return aggregated binomial counts."""
    return _prepare_bott_counts(data, config or BottNuConfig())


def _prepare_transfer_summary(data: pd.DataFrame, config: TransferNuConfig) -> pd.DataFrame:
    data = _require_dataframe(data, "transfer data")
    _validate_interval(config.h_window, "h_window")
    _require_columns(data, [config.width_col, config.h_col], "transfer data")

    if config.raw_value_col in data.columns:
        work = _finite_rows(data, [config.width_col, config.h_col, config.raw_value_col])
        summary = (
            work.groupby([config.width_col, config.h_col], as_index=False)
            .agg(
                value=(config.raw_value_col, "mean"),
                std=(config.raw_value_col, "std"),
                n=(config.raw_value_col, "count"),
            )
        )
        summary["error"] = summary["std"] / np.sqrt(summary["n"])
    else:
        _require_columns(
            data,
            [config.width_col, config.h_col, config.summary_value_col],
            "transfer summary",
        )
        columns = [config.width_col, config.h_col, config.summary_value_col]
        has_error = config.summary_error_col in data.columns
        if has_error:
            columns.append(config.summary_error_col)
        summary = _finite_rows(data, columns)[columns].copy()
        summary = summary.rename(columns={config.summary_value_col: "value"})
        if has_error:
            summary = summary.rename(columns={config.summary_error_col: "error"})
        else:
            summary["error"] = np.nan
        summary["n"] = np.nan

    summary = summary.rename(columns={config.width_col: "M", config.h_col: "h"})
    summary["M"] = summary["M"].astype(float)
    summary["h"] = summary["h"].astype(float)
    summary["value"] = summary["value"].astype(float)
    summary["error"] = pd.to_numeric(summary["error"], errors="coerce")

    if np.any(summary["M"] <= 0.0):
        raise ValueError("all transfer widths must be positive")

    if config.min_width is not None:
        summary = summary[summary["M"] >= float(config.min_width)]
    if config.h_window is not None:
        low, high = map(float, config.h_window)
        summary = summary[(summary["h"] >= low) & (summary["h"] <= high)]

    summary = summary.sort_values(["M", "h"]).reset_index(drop=True)
    if summary.empty:
        raise ValueError("no transfer data remain after filtering")
    if summary["M"].nunique() < 2:
        raise ValueError("transfer scaling requires at least two widths")

    positive_errors = summary.loc[
        np.isfinite(summary["error"]) & (summary["error"] > 0.0), "error"
    ]
    scale = max(float(np.nanmedian(np.abs(summary["value"]))), 1.0)
    floor = float(config.error_floor_fraction) * scale
    if not positive_errors.empty:
        floor = max(floor, 1.0e-6 * float(np.median(positive_errors)))
    invalid = ~np.isfinite(summary["error"]) | (summary["error"] <= 0.0)
    summary.loc[invalid, "error"] = floor
    summary["error"] = np.maximum(summary["error"].to_numpy(dtype=float), floor)

    return summary


def validate_transfer_data(
    data: pd.DataFrame,
    config: Optional[TransferNuConfig] = None,
) -> pd.DataFrame:
    """Validate transfer data and return a standardized summary table."""
    return _prepare_transfer_summary(data, config or TransferNuConfig())


def _prepare_dos(data: pd.DataFrame, config: DOSZConfig) -> pd.DataFrame:
    data = _require_dataframe(data, "DOS data")
    _validate_interval(config.energy_window, "energy_window")
    required = [config.energy_col, config.dos_col]
    if config.error_col is not None:
        required.append(config.error_col)
    _require_columns(data, required, "DOS data")
    work = _finite_rows(data, required)[required].copy()
    work = work.rename(columns={config.energy_col: "E", config.dos_col: "rho"})
    if config.error_col is not None:
        work = work.rename(columns={config.error_col: "error"})
    else:
        work["error"] = np.nan

    work["abs_E"] = np.abs(work["E"].to_numpy(dtype=float))
    work = work[(work["abs_E"] > 0.0) & (work["rho"] > 0.0)]
    if config.energy_window is not None:
        low, high = map(float, config.energy_window)
        work = work[(work["abs_E"] >= low) & (work["abs_E"] <= high)]
    work = work.sort_values("abs_E").reset_index(drop=True)

    if len(work) < int(config.min_points):
        raise ValueError(f"DOS fit requires at least {config.min_points} valid points")
    if float(config.dimension) <= 0.0:
        raise ValueError("dimension must be positive")
    return work


def validate_dos_data(data: pd.DataFrame, config: Optional[DOSZConfig] = None) -> pd.DataFrame:
    """Validate DOS data and return a standardized positive-energy table."""
    return _prepare_dos(data, config or DOSZConfig())


def _prepare_multifractal(data: pd.DataFrame, config: MultifractalConfig) -> pd.DataFrame:
    data = _require_dataframe(data, "multifractal data")
    _require_columns(data, [config.size_col, config.q_col, config.pq_col], "multifractal data")
    work = _finite_rows(data, [config.size_col, config.q_col, config.pq_col])[
        [column for column in [config.size_col, config.q_col, config.pq_col, config.realization_col]
         if column is not None and column in data.columns]
    ].copy()
    work = work.rename(
        columns={config.size_col: "L", config.q_col: "q", config.pq_col: "Pq"}
    )
    if config.realization_col is not None and config.realization_col in work.columns:
        work = work.rename(columns={config.realization_col: "realization"})
    elif "realization" not in work.columns:
        work["realization"] = np.arange(len(work), dtype=int)

    work["L"] = work["L"].astype(float)
    work["q"] = work["q"].astype(float)
    work["Pq"] = work["Pq"].astype(float)
    if np.any(work["L"] <= 0.0):
        raise ValueError("all multifractal sizes must be positive")
    if np.any(work["Pq"] <= 0.0):
        raise ValueError("all generalized participation ratios must be positive")

    if config.min_size is not None:
        work = work[work["L"] >= float(config.min_size)]
    if config.q_values is not None:
        requested = np.asarray(config.q_values, dtype=float)
        mask = np.zeros(len(work), dtype=bool)
        for qvalue in requested:
            mask |= np.isclose(work["q"].to_numpy(dtype=float), qvalue, rtol=0.0, atol=1e-12)
        work = work[mask]

    work = work.sort_values(["q", "L", "realization"]).reset_index(drop=True)
    if work.empty:
        raise ValueError("no multifractal data remain after filtering")
    for qvalue, group in work.groupby("q"):
        if group["L"].nunique() < 2:
            raise ValueError(f"q={qvalue} has fewer than two system sizes")
    return work


def validate_multifractal_data(
    data: pd.DataFrame,
    config: Optional[MultifractalConfig] = None,
) -> pd.DataFrame:
    """Validate realization-level ``P_q(L)`` data."""
    return _prepare_multifractal(data, config or MultifractalConfig())


# =============================================================================
# Bott width diagnostic
# =============================================================================


def _interpolate_crossing(h: ArrayF, p: ArrayF, target: float) -> float:
    h = np.asarray(h, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(h)
    h = h[order]
    p = p[order]
    orientation = 1.0 if np.corrcoef(h, p)[0, 1] >= 0.0 else -1.0
    if orientation < 0.0:
        h = h[::-1]
        p = p[::-1]
    # A cumulative envelope makes interpolation stable against small sampling noise.
    p = np.maximum.accumulate(p)
    unique_p, indices = np.unique(p, return_index=True)
    unique_h = h[indices]
    if target < unique_p.min() or target > unique_p.max():
        return np.nan
    return float(np.interp(target, unique_p, unique_h))


def diagnose_bott_width_scaling(
    data: pd.DataFrame,
    config: Optional[BottNuConfig] = None,
    lower_probability: float = 0.25,
    upper_probability: float = 0.75,
) -> ScalingFitResult:
    """Estimate ``nu`` from the Bott transition width.

    This transparent diagnostic uses ``width ~ L^(-1/nu)``.  It is useful for
    initial inspection but discards binomial information and should not replace
    :func:`fit_nu_from_bott` in a final analysis.
    """
    config = config or BottNuConfig()
    counts = _prepare_bott_counts(data, config)
    if not 0.0 < lower_probability < upper_probability < 1.0:
        raise ValueError("probability levels must satisfy 0 < lower < upper < 1")

    rows: list[dict[str, float]] = []
    for L, group in counts.groupby("L"):
        h = group["h"].to_numpy(dtype=float)
        p = group["p_obs"].to_numpy(dtype=float)
        h_low = _interpolate_crossing(h, p, lower_probability)
        h_mid = _interpolate_crossing(h, p, 0.5)
        h_high = _interpolate_crossing(h, p, upper_probability)
        width = abs(h_high - h_low)
        rows.append({"L": L, "h_low": h_low, "h50": h_mid, "h_high": h_high, "width": width})

    table = pd.DataFrame(rows)
    fit = table[np.isfinite(table["width"]) & (table["width"] > 0.0)].copy()
    if len(fit) < 2:
        raise ValueError("at least two valid Bott widths are required")
    beta, covariance, residual, chi2, dof = _linear_fit(
        np.log(fit["L"].to_numpy(dtype=float)),
        np.log(fit["width"].to_numpy(dtype=float)),
    )
    intercept, slope = map(float, beta)
    if slope >= 0.0:
        raise RuntimeError("Bott width does not decrease with system size")
    nu = -1.0 / slope
    slope_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    nu_error = slope_error / slope**2
    hc = float(np.nanmean(table["h50"]))

    predictions = fit.copy()
    predictions["width_fit"] = np.exp(intercept) * predictions["L"] ** slope
    residuals = predictions[["L", "width"]].copy()
    residuals["residual"] = predictions["width"] - predictions["width_fit"]
    level = _validate_confidence(config.confidence_level)
    return ScalingFitResult(
        method="bott_width_diagnostic",
        success=True,
        parameters={"nu": nu, "hc_mean_h50": hc, "slope": slope, "intercept": intercept},
        standard_errors={"nu": nu_error, "hc_mean_h50": np.nan, "slope": slope_error, "intercept": float(np.sqrt(max(covariance[0, 0], 0.0)))},
        confidence_intervals={"nu": _normal_interval(nu, nu_error, level)},
        objective_value=chi2,
        degrees_of_freedom=dof,
        predictions=predictions,
        residuals=residuals,
        bootstrap_samples=None,
        metadata={"config": asdict(config), "width_table": table, "warning": "diagnostic estimator"},
        message="Bott width diagnostic completed",
    )


# =============================================================================
# Bott binomial fit for nu
# =============================================================================


def _bott_parameter_names(order: int) -> list[str]:
    return ["hc", "log_nu"] + [f"c{k}" for k in range(order + 1)]


def _bott_prediction(theta: ArrayF, L: ArrayF, h: ArrayF, order: int) -> tuple[ArrayF, ArrayF]:
    hc = theta[0]
    nu = np.exp(theta[1])
    coefficients = theta[2:]
    x = (h - hc) * L ** (1.0 / nu)
    eta = np.polynomial.polynomial.polyval(x, coefficients)
    return expit(eta), x


def _bott_initial(counts: pd.DataFrame, config: BottNuConfig) -> ArrayF:
    h50_values = []
    for _, group in counts.groupby("L"):
        h50_values.append(
            _interpolate_crossing(
                group["h"].to_numpy(dtype=float),
                group["p_obs"].to_numpy(dtype=float),
                0.5,
            )
        )
    finite_h50 = np.asarray(h50_values, dtype=float)
    finite_h50 = finite_h50[np.isfinite(finite_h50)]
    hc0 = float(np.mean(finite_h50)) if finite_h50.size else float(np.median(counts["h"]))
    nu0 = float(np.clip(1.0, *config.nu_bounds))
    correlation = np.corrcoef(counts["h"], counts["p_obs"])[0, 1]
    orientation = 1.0 if not np.isfinite(correlation) or correlation >= 0.0 else -1.0
    coefficients = np.zeros(config.polynomial_order + 1, dtype=float)
    coefficients[0] = 0.0
    if config.polynomial_order >= 1:
        coefficients[1] = orientation
    return np.concatenate([[hc0, np.log(nu0)], coefficients])


def _fit_bott_counts_once(
    counts: pd.DataFrame,
    config: BottNuConfig,
    initial: Optional[ArrayF] = None,
) -> tuple[Any, Any]:
    L = counts["L"].to_numpy(dtype=float)
    h = counts["h"].to_numpy(dtype=float)
    k = counts["k"].to_numpy(dtype=float)
    n = counts["n"].to_numpy(dtype=float)
    order = int(config.polynomial_order)
    if order < 1:
        raise ValueError("Bott polynomial_order must be at least one")

    def nll(theta: ArrayF) -> float:
        probability, _ = _bott_prediction(theta, L, h, order)
        probability = np.clip(probability, 1.0e-12, 1.0 - 1.0e-12)
        return float(-np.sum(xlogy(k, probability) + xlog1py(n - k, -probability)))

    theta0 = _bott_initial(counts, config) if initial is None else np.asarray(initial, dtype=float)
    hc_bounds = config.hc_bounds or (float(counts["h"].min()), float(counts["h"].max()))
    if not hc_bounds[0] < hc_bounds[1]:
        raise ValueError("invalid hc bounds")
    if not 0.0 < config.nu_bounds[0] < config.nu_bounds[1]:
        raise ValueError("nu_bounds must satisfy 0 < low < high")
    coefficient_bound = float(config.coefficient_bound)
    bounds = [hc_bounds, tuple(np.log(config.nu_bounds))] + [(-coefficient_bound, coefficient_bound)] * (order + 1)
    result = minimize(
        nll,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(config.maxiter), "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    return result, nll


def fit_nu_from_bott(
    data: pd.DataFrame,
    config: Optional[BottNuConfig] = None,
) -> ScalingFitResult:
    """Fit ``h_c`` and ``nu`` from Bott outcomes using binomial likelihood."""
    config = config or BottNuConfig()
    counts = _prepare_bott_counts(data, config)
    result, nll = _fit_bott_counts_once(counts, config)
    theta = np.asarray(result.x, dtype=float)
    order = int(config.polynomial_order)
    probability, x = _bott_prediction(
        theta,
        counts["L"].to_numpy(dtype=float),
        counts["h"].to_numpy(dtype=float),
        order,
    )

    names = _bott_parameter_names(order)
    transformed_values = [theta[0], np.exp(theta[1]), *theta[2:]]
    transformed_names = ["hc", "nu", *names[2:]]
    parameters = {name: float(value) for name, value in zip(transformed_names, transformed_values)}

    covariance_theta = _covariance_from_hessian(nll, theta) if result.success else None
    standard_errors = {name: np.nan for name in transformed_names}
    if covariance_theta is not None:
        diagonal = np.sqrt(np.maximum(np.diag(covariance_theta), 0.0))
        standard_errors["hc"] = float(diagonal[0])
        standard_errors["nu"] = float(np.exp(theta[1]) * diagonal[1])
        for index, name in enumerate(transformed_names[2:], start=2):
            standard_errors[name] = float(diagonal[index])

    predictions = counts.copy()
    predictions["x_scaling"] = x
    predictions["p_fit"] = probability
    residuals = predictions[["L", "h", "k", "n", "p_obs", "p_fit"]].copy()
    residuals["residual"] = residuals["p_obs"] - residuals["p_fit"]
    residuals["pearson_residual"] = (
        residuals["k"].to_numpy(dtype=float) - residuals["n"].to_numpy(dtype=float) * probability
    ) / np.sqrt(np.maximum(residuals["n"].to_numpy(dtype=float) * probability * (1.0 - probability), 1.0e-12))

    bootstrap_frame: Optional[pd.DataFrame] = None
    if int(config.n_bootstrap) > 0 and result.success:
        rng = np.random.default_rng(config.seed)
        rows = []
        warm = theta.copy()
        for replicate in range(int(config.n_bootstrap)):
            boot = counts.copy()
            boot["k"] = rng.binomial(boot["n"].to_numpy(dtype=int), probability)
            boot["p_obs"] = boot["k"] / boot["n"]
            try:
                boot_result, _ = _fit_bott_counts_once(boot, config, initial=warm)
                if boot_result.success:
                    warm = np.asarray(boot_result.x, dtype=float)
                    rows.append(
                        {
                            "replicate": replicate,
                            "hc": warm[0],
                            "nu": np.exp(warm[1]),
                            **{f"c{k}": warm[2 + k] for k in range(order + 1)},
                        }
                    )
            except Exception:
                continue
        bootstrap_frame = pd.DataFrame(rows) if rows else None

    level = _validate_confidence(config.confidence_level)
    confidence_intervals: dict[str, tuple[float, float]] = {}
    for name, value in parameters.items():
        if bootstrap_frame is not None and name in bootstrap_frame.columns:
            confidence_intervals[name] = _percentile_interval(
                bootstrap_frame[name].to_numpy(dtype=float), level
            )
            standard_errors[name] = float(bootstrap_frame[name].std(ddof=1))
        else:
            confidence_intervals[name] = _normal_interval(value, standard_errors.get(name, np.nan), level)

    return ScalingFitResult(
        method="bott_binomial_scaling",
        success=bool(result.success),
        parameters=parameters,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        objective_value=float(result.fun),
        degrees_of_freedom=max(0, len(counts) - len(theta)),
        predictions=predictions,
        residuals=residuals,
        bootstrap_samples=bootstrap_frame,
        metadata={"config": asdict(config), "n_groups": len(counts), "n_trials": int(counts["n"].sum()), "raw_optimizer": result},
        message=str(result.message),
    )


# =============================================================================
# Transfer-matrix fit for nu
# =============================================================================


def _transfer_unpack(theta: ArrayF, config: TransferNuConfig) -> tuple[float, float, ArrayF, Optional[float], Optional[ArrayF]]:
    order = int(config.polynomial_order)
    hc = float(theta[0])
    nu = float(np.exp(theta[1]))
    leading = np.asarray(theta[2: 2 + order + 1], dtype=float)
    cursor = 2 + order + 1
    if not config.include_irrelevant:
        return hc, nu, leading, None, None
    if config.irrelevant_exponent is None:
        y = -float(np.exp(theta[cursor]))
        cursor += 1
    else:
        y = float(config.irrelevant_exponent)
    correction = np.asarray(theta[cursor: cursor + int(config.irrelevant_order) + 1], dtype=float)
    return hc, nu, leading, y, correction


def _transfer_prediction(theta: ArrayF, M: ArrayF, h: ArrayF, config: TransferNuConfig) -> tuple[ArrayF, ArrayF]:
    hc, nu, leading, y, correction = _transfer_unpack(theta, config)
    x = (h - hc) * M ** (1.0 / nu)
    value = np.polynomial.polynomial.polyval(x, leading)
    if y is not None and correction is not None:
        value = value + M**y * np.polynomial.polynomial.polyval(x, correction)
    return value, x


def _transfer_initial(summary: pd.DataFrame, config: TransferNuConfig) -> ArrayF:
    grouped = summary.groupby("h")["value"].agg(["mean", "var"]).reset_index()
    hc0 = float(grouped.loc[grouped["var"].fillna(np.inf).idxmin(), "h"])
    nu0 = float(np.clip(1.0, *config.nu_bounds))
    leading = np.zeros(int(config.polynomial_order) + 1, dtype=float)
    leading[0] = float(summary["value"].mean())
    if config.polynomial_order >= 1:
        x0 = (summary["h"].to_numpy(dtype=float) - hc0) * summary["M"].to_numpy(dtype=float) ** (1.0 / nu0)
        if np.std(x0) > 0.0:
            leading[1] = float(np.cov(x0, summary["value"].to_numpy(dtype=float), ddof=1)[0, 1] / np.var(x0, ddof=1))
    parts = [np.array([hc0, np.log(nu0)]), leading]
    if config.include_irrelevant:
        if config.irrelevant_exponent is None:
            parts.append(np.array([0.0]))  # y=-exp(0)=-1
        parts.append(np.zeros(int(config.irrelevant_order) + 1, dtype=float))
    return np.concatenate(parts)


def _fit_transfer_once(
    summary: pd.DataFrame,
    config: TransferNuConfig,
    initial: Optional[ArrayF] = None,
):
    M = summary["M"].to_numpy(dtype=float)
    h = summary["h"].to_numpy(dtype=float)
    value = summary["value"].to_numpy(dtype=float)
    error = summary["error"].to_numpy(dtype=float)
    theta0 = _transfer_initial(summary, config) if initial is None else np.asarray(initial, dtype=float)

    def residual(theta: ArrayF) -> ArrayF:
        prediction, _ = _transfer_prediction(theta, M, h, config)
        return (value - prediction) / error

    hc_bounds = config.hc_bounds or (float(summary["h"].min()), float(summary["h"].max()))
    coefficient_bound = float(config.coefficient_bound)
    lower = [hc_bounds[0], np.log(config.nu_bounds[0])] + [-coefficient_bound] * (int(config.polynomial_order) + 1)
    upper = [hc_bounds[1], np.log(config.nu_bounds[1])] + [coefficient_bound] * (int(config.polynomial_order) + 1)
    if config.include_irrelevant:
        if config.irrelevant_exponent is None:
            lower.append(np.log(1.0e-3))
            upper.append(np.log(20.0))
        lower += [-coefficient_bound] * (int(config.irrelevant_order) + 1)
        upper += [coefficient_bound] * (int(config.irrelevant_order) + 1)
    result = least_squares(
        residual,
        theta0,
        bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        max_nfev=int(config.max_nfev),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )
    return result, residual


def fit_nu_from_transfer(
    data: pd.DataFrame,
    config: Optional[TransferNuConfig] = None,
) -> ScalingFitResult:
    """Fit ``h_c`` and ``nu`` from normalized localization-length data."""
    config = config or TransferNuConfig()
    summary = _prepare_transfer_summary(data, config)
    result, residual_function = _fit_transfer_once(summary, config)
    theta = np.asarray(result.x, dtype=float)
    prediction, x = _transfer_prediction(
        theta,
        summary["M"].to_numpy(dtype=float),
        summary["h"].to_numpy(dtype=float),
        config,
    )
    hc, nu, leading, y, correction = _transfer_unpack(theta, config)
    parameters: dict[str, float] = {"hc": hc, "nu": nu}
    for index, value in enumerate(leading):
        parameters[f"c{index}"] = float(value)
    if y is not None:
        parameters["y"] = float(y)
    if correction is not None:
        for index, value in enumerate(correction):
            parameters[f"d{index}"] = float(value)

    residual_vector = residual_function(theta)
    dof = max(0, len(summary) - len(theta))
    covariance_theta: Optional[ArrayF] = None
    if result.success:
        information = result.jac.T @ result.jac
        covariance_theta = np.linalg.pinv(information, rcond=1.0e-12)
        if dof > 0:
            covariance_theta *= float(np.sum(residual_vector**2) / dof)

    standard_errors = {name: np.nan for name in parameters}
    if covariance_theta is not None:
        diagonal = np.sqrt(np.maximum(np.diag(covariance_theta), 0.0))
        standard_errors["hc"] = float(diagonal[0])
        standard_errors["nu"] = float(nu * diagonal[1])
        cursor = 2
        for index in range(len(leading)):
            standard_errors[f"c{index}"] = float(diagonal[cursor])
            cursor += 1
        if config.include_irrelevant:
            if config.irrelevant_exponent is None:
                standard_errors["y"] = float(abs(y) * diagonal[cursor])
                cursor += 1
            else:
                standard_errors["y"] = 0.0
            for index in range(0 if correction is None else len(correction)):
                standard_errors[f"d{index}"] = float(diagonal[cursor])
                cursor += 1

    predictions = summary.copy()
    predictions["x_scaling"] = x
    predictions["value_fit"] = prediction
    residuals = predictions[["M", "h", "value", "error", "value_fit"]].copy()
    residuals["residual"] = residuals["value"] - residuals["value_fit"]
    residuals["standardized_residual"] = residuals["residual"] / residuals["error"]

    bootstrap_frame: Optional[pd.DataFrame] = None
    if int(config.n_bootstrap) > 0 and result.success:
        rng = np.random.default_rng(config.seed)
        rows = []
        warm = theta.copy()
        for replicate in range(int(config.n_bootstrap)):
            boot = summary.copy()
            boot["value"] = prediction + boot["error"].to_numpy(dtype=float) * rng.normal(size=len(boot))
            try:
                boot_result, _ = _fit_transfer_once(boot, config, initial=warm)
                if boot_result.success:
                    warm = np.asarray(boot_result.x, dtype=float)
                    bhc, bnu, bleading, by, bcorrection = _transfer_unpack(warm, config)
                    row = {"replicate": replicate, "hc": bhc, "nu": bnu}
                    row.update({f"c{k}": value for k, value in enumerate(bleading)})
                    if by is not None:
                        row["y"] = by
                    if bcorrection is not None:
                        row.update({f"d{k}": value for k, value in enumerate(bcorrection)})
                    rows.append(row)
            except Exception:
                continue
        bootstrap_frame = pd.DataFrame(rows) if rows else None

    level = _validate_confidence(config.confidence_level)
    confidence_intervals: dict[str, tuple[float, float]] = {}
    for name, value in parameters.items():
        if bootstrap_frame is not None and name in bootstrap_frame.columns:
            confidence_intervals[name] = _percentile_interval(bootstrap_frame[name].to_numpy(dtype=float), level)
            standard_errors[name] = float(bootstrap_frame[name].std(ddof=1))
        else:
            confidence_intervals[name] = _normal_interval(value, standard_errors.get(name, np.nan), level)

    return ScalingFitResult(
        method="transfer_one_parameter_scaling",
        success=bool(result.success),
        parameters=parameters,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        objective_value=float(np.sum(residual_vector**2)),
        degrees_of_freedom=dof,
        predictions=predictions,
        residuals=residuals,
        bootstrap_samples=bootstrap_frame,
        metadata={"config": asdict(config), "n_points": len(summary), "raw_optimizer": result},
        message=str(result.message),
    )


# =============================================================================
# DOS power-law fit for z
# =============================================================================


def _fit_dos_once(work: pd.DataFrame, config: DOSZConfig) -> tuple[ArrayF, ArrayF, ArrayF, float, int]:
    x = np.log(work["abs_E"].to_numpy(dtype=float))
    y = np.log(work["rho"].to_numpy(dtype=float))
    sigma_log: Optional[ArrayF] = None
    if config.error_col is not None:
        error = work["error"].to_numpy(dtype=float)
        if np.all(np.isfinite(error) & (error > 0.0)):
            sigma_log = error / work["rho"].to_numpy(dtype=float)
    return _linear_fit(x, y, sigma_log)


def fit_z_from_dos_powerlaw(
    data: pd.DataFrame,
    config: Optional[DOSZConfig] = None,
) -> ScalingFitResult:
    """Estimate ``z`` from ``rho(E) = A |E|^(d/z-1)``."""
    config = config or DOSZConfig()
    work = _prepare_dos(data, config)
    beta, covariance, residual, chi2, dof = _fit_dos_once(work, config)
    log_amplitude, alpha = map(float, beta)
    denominator = alpha + 1.0
    if denominator <= 0.0:
        raise RuntimeError("fitted DOS exponent gives nonpositive alpha+1 and hence unphysical z")
    dimension = float(config.dimension)
    z = dimension / denominator
    amplitude = float(np.exp(log_amplitude))
    errors_beta = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    alpha_error = float(errors_beta[1])
    z_error = abs(dimension / denominator**2) * alpha_error
    amplitude_error = amplitude * float(errors_beta[0])

    predictions = work.copy()
    predictions["rho_fit"] = amplitude * predictions["abs_E"] ** alpha
    residuals = predictions[["E", "abs_E", "rho", "rho_fit"]].copy()
    residuals["log_residual"] = np.log(residuals["rho"]) - np.log(residuals["rho_fit"])

    bootstrap_frame: Optional[pd.DataFrame] = None
    if int(config.n_bootstrap) > 0:
        rng = np.random.default_rng(config.seed)
        rows = []
        for replicate in range(int(config.n_bootstrap)):
            indices = rng.integers(0, len(work), size=len(work))
            boot = work.iloc[indices].copy().sort_values("abs_E")
            if boot["abs_E"].nunique() < 2:
                continue
            try:
                b_beta, *_ = _fit_dos_once(boot, config)
                b_log_amp, b_alpha = map(float, b_beta)
                if b_alpha + 1.0 > 0.0:
                    rows.append(
                        {
                            "replicate": replicate,
                            "amplitude": np.exp(b_log_amp),
                            "alpha": b_alpha,
                            "z": dimension / (b_alpha + 1.0),
                        }
                    )
            except Exception:
                continue
        bootstrap_frame = pd.DataFrame(rows) if rows else None

    parameters = {"z": z, "alpha": alpha, "amplitude": amplitude}
    standard_errors = {"z": z_error, "alpha": alpha_error, "amplitude": amplitude_error}
    level = _validate_confidence(config.confidence_level)
    confidence_intervals = {}
    for name, value in parameters.items():
        if bootstrap_frame is not None and name in bootstrap_frame.columns:
            confidence_intervals[name] = _percentile_interval(bootstrap_frame[name].to_numpy(dtype=float), level)
            standard_errors[name] = float(bootstrap_frame[name].std(ddof=1))
        else:
            confidence_intervals[name] = _normal_interval(value, standard_errors[name], level)

    return ScalingFitResult(
        method="dos_powerlaw_z",
        success=True,
        parameters=parameters,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        objective_value=chi2,
        degrees_of_freedom=dof,
        predictions=predictions,
        residuals=residuals,
        bootstrap_samples=bootstrap_frame,
        metadata={"config": asdict(config), "n_points": len(work), "warning": "power-law window estimator"},
        message="DOS power-law fit completed",
    )


# =============================================================================
# Wavefunction moments and multifractal analysis
# =============================================================================


def site_probabilities(state: npt.ArrayLike, norb: int = 2, normalize: bool = True) -> ArrayF:
    """Convert a site-major orbital state vector into site probabilities."""
    array = np.asarray(state)
    if array.ndim != 1:
        raise ValueError("state must be a one-dimensional vector")
    if isinstance(norb, (bool, np.bool_)) or int(norb) <= 0 or int(norb) != norb:
        raise ValueError("norb must be a positive integer")
    norb = int(norb)
    if array.size % norb != 0:
        raise ValueError("state length is not divisible by norb")
    probability = np.sum(np.abs(array.reshape(-1, norb)) ** 2, axis=1).astype(float)
    total = float(np.sum(probability))
    if normalize:
        if not total > 0.0:
            raise ValueError("state has zero norm")
        probability /= total
    return probability


def box_probabilities(
    probabilities: npt.ArrayLike,
    shape: tuple[int, int],
    box_size: int = 1,
    normalize: bool = True,
) -> ArrayF:
    """Coarse-grain site probabilities into nonoverlapping square boxes.

    ``shape`` is ``(Ly, Lx)``.  Both dimensions must be divisible by
    ``box_size``; silent cropping is deliberately forbidden.
    """
    probability = np.asarray(probabilities, dtype=float)
    if probability.ndim != 1:
        raise ValueError("probabilities must be a one-dimensional site array")
    Ly, Lx = map(int, shape)
    if Ly <= 0 or Lx <= 0 or probability.size != Ly * Lx:
        raise ValueError("shape must match the number of site probabilities")
    if isinstance(box_size, (bool, np.bool_)) or int(box_size) != box_size or int(box_size) <= 0:
        raise ValueError("box_size must be a positive integer")
    box_size = int(box_size)
    if Ly % box_size != 0 or Lx % box_size != 0:
        raise ValueError("Ly and Lx must both be divisible by box_size")
    if np.any(~np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    total = float(np.sum(probability))
    if normalize:
        if not total > 0.0:
            raise ValueError("probabilities have zero total weight")
        probability = probability / total
    grid = probability.reshape(Ly, Lx)
    boxes = grid.reshape(Ly // box_size, box_size, Lx // box_size, box_size).sum(axis=(1, 3))
    return boxes.ravel()


def generalized_participation_ratios(
    probabilities: npt.ArrayLike,
    q_values: Sequence[float],
    *,
    shape: Optional[tuple[int, int]] = None,
    box_size: int = 1,
    normalize: bool = True,
) -> pd.DataFrame:
    """Compute ``P_q = sum_b mu_b^q`` for a probability distribution."""
    probability = np.asarray(probabilities, dtype=float)
    if shape is None:
        if box_size != 1:
            raise ValueError("shape is required when box_size != 1")
        if normalize:
            total = float(np.sum(probability))
            if not total > 0.0:
                raise ValueError("probabilities have zero total weight")
            probability = probability / total
        mu = probability
    else:
        mu = box_probabilities(probability, shape, box_size=box_size, normalize=normalize)

    if np.any(mu < 0.0) or np.any(~np.isfinite(mu)):
        raise ValueError("box probabilities must be finite and nonnegative")
    rows = []
    tiny = np.finfo(float).tiny
    for qvalue in q_values:
        qvalue = float(qvalue)
        if not np.isfinite(qvalue):
            raise ValueError("q values must be finite")
        if qvalue < 0.0 and np.any(mu <= 0.0):
            raise ValueError("negative q is undefined in the presence of empty boxes")
        if np.isclose(qvalue, 1.0):
            Pq = float(np.sum(mu))
        elif np.isclose(qvalue, 0.0):
            Pq = float(np.sum(mu > 0.0))
        else:
            Pq = float(np.sum(np.maximum(mu, tiny) ** qvalue))
        rows.append({"q": qvalue, "Pq": Pq, "n_boxes": len(mu), "box_size": int(box_size)})
    return pd.DataFrame(rows)


def _spectrum_from_prepared(work: pd.DataFrame, config: MultifractalConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    scale_rows: list[dict[str, float]] = []
    spectrum_rows: list[dict[str, float]] = []

    for qvalue, qgroup in work.groupby("q"):
        grouped = qgroup.groupby("L")
        scale = grouped["Pq"].agg(["count", "mean", "std"]).reset_index()
        scale["mean_log_Pq"] = grouped["Pq"].apply(lambda values: float(np.mean(np.log(values)))).to_numpy()
        scale["std_log_Pq"] = grouped["Pq"].apply(lambda values: float(np.std(np.log(values), ddof=1)) if len(values) > 1 else np.nan).to_numpy()
        scale["sem_log_Pq"] = scale["std_log_Pq"] / np.sqrt(scale["count"])
        scale["q"] = float(qvalue)
        scale_rows.extend(scale.to_dict("records"))

        row: dict[str, float] = {"q": float(qvalue), "n_sizes": float(scale["L"].nunique())}
        x = np.log(scale["L"].to_numpy(dtype=float))

        if config.compute_typical:
            y_typ = scale["mean_log_Pq"].to_numpy(dtype=float)
            beta_typ, covariance_typ, _, chi2_typ, dof_typ = _linear_fit(x, y_typ)
            tau_typ = -float(beta_typ[1])
            tau_typ_error = float(np.sqrt(max(covariance_typ[1, 1], 0.0)))
            row.update(
                {
                    "tau_typical": tau_typ,
                    "tau_typical_error": tau_typ_error,
                    "Dq_typical": np.nan if np.isclose(qvalue, 1.0) else tau_typ / (qvalue - 1.0),
                    "Dq_typical_error": np.nan if np.isclose(qvalue, 1.0) else tau_typ_error / abs(qvalue - 1.0),
                    "chi2_typical": chi2_typ,
                    "dof_typical": float(dof_typ),
                }
            )

        if config.compute_average:
            y_avg = np.log(scale["mean"].to_numpy(dtype=float))
            beta_avg, covariance_avg, _, chi2_avg, dof_avg = _linear_fit(x, y_avg)
            tau_avg = -float(beta_avg[1])
            tau_avg_error = float(np.sqrt(max(covariance_avg[1, 1], 0.0)))
            row.update(
                {
                    "tau_average": tau_avg,
                    "tau_average_error": tau_avg_error,
                    "Dq_average": np.nan if np.isclose(qvalue, 1.0) else tau_avg / (qvalue - 1.0),
                    "Dq_average_error": np.nan if np.isclose(qvalue, 1.0) else tau_avg_error / abs(qvalue - 1.0),
                    "chi2_average": chi2_avg,
                    "dof_average": float(dof_avg),
                }
            )
        spectrum_rows.append(row)

    spectrum = pd.DataFrame(spectrum_rows).sort_values("q").reset_index(drop=True)
    scale_table = pd.DataFrame(scale_rows).sort_values(["q", "L"]).reset_index(drop=True)
    return spectrum, scale_table


def legendre_spectrum(q_values: npt.ArrayLike, tau_values: npt.ArrayLike) -> pd.DataFrame:
    """Construct the numerical Legendre spectrum ``alpha(q), f(alpha)``."""
    q = np.asarray(q_values, dtype=float)
    tau = np.asarray(tau_values, dtype=float)
    mask = np.isfinite(q) & np.isfinite(tau)
    q = q[mask]
    tau = tau[mask]
    if q.size < 3:
        raise ValueError("at least three finite q values are needed for a Legendre spectrum")
    order = np.argsort(q)
    q = q[order]
    tau = tau[order]
    if np.any(np.diff(q) <= 0.0):
        raise ValueError("q values must be unique")
    alpha = np.gradient(tau, q, edge_order=2)
    f_alpha = q * alpha - tau
    return pd.DataFrame({"q": q, "tau": tau, "alpha": alpha, "f_alpha": f_alpha})


def fit_multifractal_spectrum(
    data: pd.DataFrame,
    config: Optional[MultifractalConfig] = None,
) -> MultifractalResult:
    """Fit average and typical multifractal exponents from ``P_q(L)`` data."""
    config = config or MultifractalConfig()
    if not config.compute_typical and not config.compute_average:
        raise ValueError("at least one of compute_typical/compute_average must be True")
    work = _prepare_multifractal(data, config)
    spectrum, scale_table = _spectrum_from_prepared(work, config)

    for prefix in ("typical", "average"):
        tau_col = f"tau_{prefix}"
        if tau_col in spectrum.columns and spectrum[tau_col].notna().sum() >= 3:
            legendre = legendre_spectrum(spectrum["q"], spectrum[tau_col])
            spectrum[f"alpha_{prefix}"] = legendre["alpha"].to_numpy()
            spectrum[f"f_alpha_{prefix}"] = legendre["f_alpha"].to_numpy()

    bootstrap_frame: Optional[pd.DataFrame] = None
    if int(config.n_bootstrap) > 0:
        rng = np.random.default_rng(config.seed)
        rows = []
        grouped = list(work.groupby(["q", "L"], sort=True))
        for replicate in range(int(config.n_bootstrap)):
            boot_groups = []
            for _, group in grouped:
                indices = rng.integers(0, len(group), size=len(group))
                boot_groups.append(group.iloc[indices].copy())
            boot = pd.concat(boot_groups, ignore_index=True)
            try:
                boot_spectrum, _ = _spectrum_from_prepared(boot, config)
                for _, row in boot_spectrum.iterrows():
                    output = {"replicate": replicate, "q": row["q"]}
                    for column in ["tau_typical", "Dq_typical", "tau_average", "Dq_average"]:
                        if column in row.index:
                            output[column] = row[column]
                    rows.append(output)
            except Exception:
                continue
        bootstrap_frame = pd.DataFrame(rows) if rows else None

    level = _validate_confidence(config.confidence_level)
    if bootstrap_frame is not None:
        for column in ["tau_typical", "Dq_typical", "tau_average", "Dq_average"]:
            if column not in spectrum.columns or column not in bootstrap_frame.columns:
                continue
            low_values = []
            high_values = []
            errors = []
            for qvalue in spectrum["q"]:
                values = bootstrap_frame.loc[np.isclose(bootstrap_frame["q"], qvalue), column].to_numpy(dtype=float)
                low, high = _percentile_interval(values, level)
                low_values.append(low)
                high_values.append(high)
                errors.append(float(np.nanstd(values, ddof=1)))
            spectrum[f"{column}_ci_low"] = low_values
            spectrum[f"{column}_ci_high"] = high_values
            spectrum[f"{column}_bootstrap_error"] = errors

    return MultifractalResult(
        spectrum=spectrum,
        scale_table=scale_table,
        bootstrap_samples=bootstrap_frame,
        metadata={"config": asdict(config), "n_rows": len(work), "n_realizations": int(work["realization"].nunique())},
        success=True,
        message="Multifractal spectrum fit completed",
    )


# =============================================================================
# Plotting helpers
# =============================================================================


def plot_bott_nu_result(
    result: ScalingFitResult,
    *,
    figsize=(12.0, 4.5),
):
    """
    Plot observed Bott probabilities, fitted finite-size curves,
    and the scaling collapse.

    Each size uses the same color for its data points and fitted curve.
    The universal fitted collapse is shown as a black solid line.
    """
    import matplotlib.pyplot as plt

    data = result.predictions.copy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        dpi=150,
    )

    # --------------------------------------------------------
    # Left: probability versus h
    # --------------------------------------------------------

    for L, group in data.groupby("L"):
        group = group.sort_values("h")
        point_artist = axes[0].plot(
            group["h"],
            group["p_obs"],
            "o",
            label=f"L={L:g}",)[0]
        color = point_artist.get_color()

        axes[0].plot(
            group["h"],
            group["p_fit"],
            "-",
            color=color,
            linewidth=1.5,)

        # Same size color in the collapse panel
        order = np.argsort(
            group["x_scaling"].to_numpy())

        axes[1].plot(
            group["x_scaling"].to_numpy()[order],
            group["p_obs"].to_numpy()[order],
            "o",
            color=color,
            label=f"L={L:g}",)

    # --------------------------------------------------------
    # Right: universal fitted scaling function
    # --------------------------------------------------------

    sorted_all = (
        data
        .sort_values("x_scaling")
        .drop_duplicates(
            subset="x_scaling"))

    axes[1].plot(
        sorted_all["x_scaling"],
        sorted_all["p_fit"],
        "-",
        color="black",
        linewidth=1.5,
        label="fit",)

    axes[0].set(
        xlabel="h",
        ylabel=r"$P_{|B|=1}$",
        title="Bott probability fit",)

    axes[1].set(
        xlabel=r"$(h-h_c)L^{1/\nu}$",
        ylabel=r"$P_{|B|=1}$",
        title="Bott scaling collapse",)

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)

    fig.tight_layout()

    return fig, axes
    return fig, axes


def plot_transfer_nu_result(result: ScalingFitResult, *, figsize=(12.0, 4.5)):
    """Plot normalized localization data and fitted collapse."""
    import matplotlib.pyplot as plt

    data = result.predictions
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=150)
    for M, group in data.groupby("M"):
        group = group.sort_values("h")
        axes[0].errorbar(group["h"], group["value"], yerr=group["error"], marker="o", linestyle="none", label=f"M={M:g}")
        axes[0].plot(group["h"], group["value_fit"], "-")
        order = np.argsort(group["x_scaling"])
        axes[1].errorbar(group["x_scaling"].to_numpy()[order], group["value"].to_numpy()[order], yerr=group["error"].to_numpy()[order], marker="o", linestyle="none", label=f"M={M:g}")
        axes[1].plot(group["x_scaling"].to_numpy()[order], group["value_fit"].to_numpy()[order], "-")
    axes[0].set(xlabel="h", ylabel=r"$\Lambda_M$", title="Transfer-matrix scaling fit")
    axes[1].set(xlabel=r"$(h-h_c)M^{1/\nu}$", ylabel=r"$\Lambda_M$", title="Localization-length collapse")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig, axes


def plot_dos_z_result(result: ScalingFitResult, *, figsize=(6.2, 4.5)):
    """Plot a log-log DOS power-law fit."""
    import matplotlib.pyplot as plt

    data = result.predictions.sort_values("abs_E")
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.loglog(data["abs_E"], data["rho"], "o", label="data")
    ax.loglog(data["abs_E"], data["rho_fit"], "-", label="fit")
    ax.set_xlabel(r"$|E|$")
    ax.set_ylabel(r"$\rho(E)$")
    ax.set_title(rf"DOS power law: $z={result.parameters['z']:.4g}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_multifractal_result(result: MultifractalResult, *, figsize=(12.0, 4.5)):
    """Plot ``tau(q)`` and ``D_q`` for available average/typical spectra."""
    import matplotlib.pyplot as plt

    spectrum = result.spectrum
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=150)
    for prefix, label in (("typical", "typical"), ("average", "average")):
        tau_col = f"tau_{prefix}"
        dq_col = f"Dq_{prefix}"
        if tau_col in spectrum.columns:
            axes[0].plot(spectrum["q"], spectrum[tau_col], "o-", label=label)
        if dq_col in spectrum.columns:
            axes[1].plot(spectrum["q"], spectrum[dq_col], "o-", label=label)
    axes[0].set(xlabel="q", ylabel=r"$\tau(q)$", title="Mass exponents")
    axes[1].set(xlabel="q", ylabel=r"$D_q$", title="Generalized fractal dimensions")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig, axes
