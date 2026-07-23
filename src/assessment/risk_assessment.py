from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskResult:
    severity_class: int
    severity_label: str
    risk_level: str   # "Low" | "Medium" | "High"
    message: str


SEVERITY_LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}


def assess_risk(severity_class: int) -> RiskResult:
    """
    Map DR severity class (0..4) -> risk level.

    Why rule-based:
    - Transparent and easy to validate.
    - Avoids giving treatment recommendations (non-medical guidance only).
    """
    if severity_class not in SEVERITY_LABELS:
        raise ValueError(f"Invalid severity_class: {severity_class}. Expected 0..4")

    label = SEVERITY_LABELS[severity_class]

    # Simple mapping (can be refined later)
    if severity_class in (0, 1):
        risk = "Low"
        msg = "Low risk based on predicted severity. Regular screening is still recommended."
    elif severity_class == 2:
        risk = "Medium"
        msg = "Medium risk based on predicted severity. Consider scheduling an ophthalmology review."
    else:  # 3,4
        risk = "High"
        msg = "High risk based on predicted severity. Prompt ophthalmology consultation is recommended."

    return RiskResult(
        severity_class=severity_class,
        severity_label=label,
        risk_level=risk,
        message=msg,
    )