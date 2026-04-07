"""
Post-hoc calibration utilities for probabilistic precipitation models.

Provides two strategies:
  A) Variance scaling — inflate/deflate Gamma scale to achieve target coverage
  B) Quantile recalibration — map nominal quantiles to empirical-adjusted quantiles
"""

import numpy as np
from scipy.stats import gamma as sp_gamma, kstest


def find_optimal_scale_factor(
    y: np.ndarray,
    concentration: np.ndarray,
    rate: np.ndarray,
    target_coverage: float = 0.90,
    search_range: tuple[float, float] = (0.5, 5.0),
    step: float = 0.05,
) -> float:
    """Search for a scale factor that achieves the target 90% PI coverage.

    Multiplies the Gamma scale (1/rate) by the factor, then checks coverage.
    Returns the factor minimizing |actual_coverage - target_coverage|.
    """
    scale = 1.0 / rate
    lo_q = (1 - target_coverage) / 2
    hi_q = 1 - lo_q

    best_factor = 1.0
    best_gap = float("inf")

    factors = np.arange(search_range[0], search_range[1] + step, step)
    for sf in factors:
        adj_scale = scale * sf
        lo = sp_gamma.ppf(lo_q, a=concentration, scale=adj_scale)
        hi = sp_gamma.ppf(hi_q, a=concentration, scale=adj_scale)
        cov = float(np.mean((y >= lo) & (y <= hi)))
        gap = abs(cov - target_coverage)
        if gap < best_gap:
            best_gap = gap
            best_factor = float(sf)

    return best_factor


def compute_pit(
    y: np.ndarray,
    concentration: np.ndarray,
    rate: np.ndarray,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """Compute PIT values: F(y | concentration, adjusted_scale)."""
    scale = (1.0 / rate) * scale_factor
    return sp_gamma.cdf(y, a=concentration, scale=scale)


def pit_ks_statistic(pit: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov test of PIT against Uniform[0,1]."""
    stat, pval = kstest(pit, "uniform")
    return float(stat), float(pval)


def build_quantile_recalibration_map(
    pit_val: np.ndarray,
    nominal_levels: list[float] | None = None,
) -> dict[float, float]:
    """Build a mapping from nominal quantile levels to recalibrated levels.

    For each nominal level q, the adjusted level is the empirical fraction of
    validation PIT values <= q. If model says '5th percentile' but 12% of
    observations fall below it, the adjusted quantile becomes 0.12.
    """
    if nominal_levels is None:
        nominal_levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    recal_map = {}
    for q in nominal_levels:
        recal_map[q] = float(np.mean(pit_val <= q))
    return recal_map


def apply_quantile_recalibration(
    concentration: np.ndarray,
    rate: np.ndarray,
    recal_map: dict[float, float],
    scale_factor: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute recalibrated quantiles using the recalibration map."""
    scale = (1.0 / rate) * scale_factor
    result = {}
    for nominal_q, adjusted_q in recal_map.items():
        result[f"quant_{nominal_q}"] = sp_gamma.ppf(adjusted_q, a=concentration, scale=scale)
    return result
