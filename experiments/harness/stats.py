"""Small-sample statistics for sweep aggregation (Student-t CI)."""
from __future__ import annotations

import math
import statistics
from typing import Optional, Sequence, Tuple

from scipy import stats as scipy_stats


def mean_ci(
    values: Sequence[float], confidence: float = 0.95
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (mean, sample_std, ci_half_width) for a small sample.

    Uses the Student-t distribution (appropriate for the few repetitions a
    GPU sweep affords). With n < 2 no spread can be estimated, so std and
    the CI half-width are None.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return (None, None, None)
    mean = sum(vals) / n
    if n == 1:
        return (mean, None, None)
    std = statistics.stdev(vals)
    t_crit = float(scipy_stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    ci_half = t_crit * std / math.sqrt(n)
    return (mean, std, ci_half)
