from __future__ import annotations

from dataclasses import dataclass

from .validation import ValidationAudit


@dataclass(slots=True)
class QualityReport:
    merged_pairs: int
    rejected_pairs: int
    conflicts: int
    precision_proxy: float
    recall_proxy: float


def build_quality_report(audit: ValidationAudit) -> QualityReport:
    merged = len(audit.merged_pairs)
    rejected = len(audit.rejected_pairs)
    conflicts = len(audit.conflict_log)
    denominator = max(1, merged + rejected + conflicts)

    # Lightweight proxy metrics for iterative gating.
    precision = merged / max(1, merged + conflicts)
    recall = merged / denominator

    return QualityReport(
        merged_pairs=merged,
        rejected_pairs=rejected,
        conflicts=conflicts,
        precision_proxy=round(precision, 4),
        recall_proxy=round(recall, 4),
    )

