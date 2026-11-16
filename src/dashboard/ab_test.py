"""Two-proportion A/B test for conversion events (independent users)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erfc, sqrt
from statistics import NormalDist


@dataclass(frozen=True)
class ABTestResult:
    control_cvr_pct: float
    treatment_cvr_pct: float
    lift_pct: float | None
    p_value: float
    z_statistic: float
    ci_low_pp: float
    ci_high_pp: float
    significant_at_005: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _wilson_interval(successes: int, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval for one binomial rate."""
    p = successes / n
    z2 = z * z
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    half_width = z * sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def analyze_ab_test(
    control_users: int,
    control_conversions: int,
    treatment_users: int,
    treatment_conversions: int,
    alpha: float = 0.05,
) -> ABTestResult:
    """Pooled two-sided z-test; Newcombe-Wilson CI for absolute CVR difference.

    CI is expressed in percentage points for (treatment - control), not lift %.
    Assumes randomized independent user-level Bernoulli conversion outcomes.
    """
    for label, value in (
        ("control_users", control_users),
        ("control_conversions", control_conversions),
        ("treatment_users", treatment_users),
        ("treatment_conversions", treatment_conversions),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} must be an integer")
    if control_users <= 0 or treatment_users <= 0:
        raise ValueError("Both groups require at least one user")
    if not (0 <= control_conversions <= control_users):
        raise ValueError("Control conversions must be between 0 and control users")
    if not (0 <= treatment_conversions <= treatment_users):
        raise ValueError("Treatment conversions must be between 0 and treatment users")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    p_control = control_conversions / control_users
    p_treatment = treatment_conversions / treatment_users
    pooled = (control_conversions + treatment_conversions) / (
        control_users + treatment_users
    )
    variance = pooled * (1 - pooled) * (1 / control_users + 1 / treatment_users)
    z_statistic = (p_treatment - p_control) / sqrt(variance) if variance else 0.0
    p_value = erfc(abs(z_statistic) / sqrt(2)) if variance else 1.0
    z_crit = NormalDist().inv_cdf(1 - alpha / 2)
    control_low, control_high = _wilson_interval(
        control_conversions, control_users, z_crit
    )
    treatment_low, treatment_high = _wilson_interval(
        treatment_conversions, treatment_users, z_crit
    )

    # Newcombe (1998) hybrid score CI; resistant to small counts and boundary rates.
    difference = p_treatment - p_control
    lower = difference - sqrt(
        (p_treatment - treatment_low) ** 2 + (control_high - p_control) ** 2
    )
    upper = difference + sqrt(
        (treatment_high - p_treatment) ** 2 + (p_control - control_low) ** 2
    )
    lift_pct = ((p_treatment / p_control) - 1) * 100 if p_control > 0 else None
    return ABTestResult(
        control_cvr_pct=p_control * 100,
        treatment_cvr_pct=p_treatment * 100,
        lift_pct=lift_pct,
        p_value=p_value,
        z_statistic=z_statistic,
        ci_low_pp=max(-100.0, lower * 100),
        ci_high_pp=min(100.0, upper * 100),
        significant_at_005=p_value < alpha,
    )
