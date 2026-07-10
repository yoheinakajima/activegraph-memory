"""Held-out calibration helpers for operator-specific candidate gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .profiles import MemoryRuntimeProfile


@dataclass(frozen=True)
class OperatorCalibrationResult:
    operator: str
    threshold: float
    samples: int
    accepted: int
    precision: float
    coverage: float
    target_precision: float


def calibrate_operator_thresholds(
    records: Iterable[dict[str, Any]],
    *,
    target_precision: float = 0.9,
    minimum_accepted: int = 5,
    thresholds: Iterable[float] | None = None,
) -> dict[str, OperatorCalibrationResult]:
    """Choose the broadest held-out threshold meeting a precision target.

    Each record must carry ``operator``, ``confidence``, and ``correct``.
    ``proof_complete`` and ``conflict_free`` default to true. Records that fail
    either structural gate remain in the sample count but cannot be accepted.
    """

    if not 0.0 <= target_precision <= 1.0:
        raise ValueError("target_precision must be between 0 and 1")
    grid = sorted(
        set(
            round(float(value), 4)
            for value in (thresholds or [value / 100 for value in range(50, 100, 2)])
            if 0.0 <= float(value) <= 1.0
        )
    )
    if not grid:
        raise ValueError("thresholds must contain at least one value between 0 and 1")
    by_operator: dict[str, list[dict[str, Any]]] = {}
    for raw in records:
        operator = str(raw.get("operator") or "unknown")
        by_operator.setdefault(operator, []).append(raw)

    results = {}
    for operator, samples in sorted(by_operator.items()):
        candidates = []
        fallback = None
        for threshold in grid:
            accepted = [
                sample
                for sample in samples
                if bool(sample.get("proof_complete", True))
                and bool(sample.get("conflict_free", True))
                and float(sample.get("confidence") or 0.0) >= threshold
            ]
            precision = (
                sum(bool(sample.get("correct")) for sample in accepted) / len(accepted)
                if accepted
                else 0.0
            )
            result = OperatorCalibrationResult(
                operator=operator,
                threshold=threshold,
                samples=len(samples),
                accepted=len(accepted),
                precision=round(precision, 4),
                coverage=round(len(accepted) / len(samples), 4) if samples else 0.0,
                target_precision=target_precision,
            )
            if fallback is None or (result.precision, result.accepted) > (fallback.precision, fallback.accepted):
                fallback = result
            if len(accepted) >= minimum_accepted and precision >= target_precision:
                candidates.append(result)
        results[operator] = candidates[0] if candidates else fallback or OperatorCalibrationResult(
            operator=operator,
            threshold=1.0,
            samples=len(samples),
            accepted=0,
            precision=0.0,
            coverage=0.0,
            target_precision=target_precision,
        )
    return results


def apply_operator_calibration(
    profile: MemoryRuntimeProfile,
    calibration: dict[str, OperatorCalibrationResult],
) -> MemoryRuntimeProfile:
    """Return a profile with held-out thresholds merged over its defaults."""

    return profile.model_copy(
        update={
            "operator_min_confidence": {
                **profile.operator_min_confidence,
                **{operator: result.threshold for operator, result in calibration.items()},
            }
        }
    )
