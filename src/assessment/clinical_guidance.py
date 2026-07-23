from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


DISCLAIMER = (
    "Disclaimer: This tool is for educational and screening support purposes only. "
    "It does not provide a medical diagnosis. Please consult a qualified ophthalmologist "
    "for a comprehensive eye examination and clinical decision-making."
)


@dataclass(frozen=True)
class GuidanceResult:
    severity_class: int
    guidance: str
    disclaimer: str = DISCLAIMER


def generate_guidance(severity_class: int) -> GuidanceResult:
    """
    Generate general, non-treatment guidance text based on severity class.
    """
    if severity_class == 0:
        guidance = (
            "Prediction indicates no signs of diabetic retinopathy in this image. "
            "Continue routine diabetic eye screening as advised by your healthcare provider."
        )
    elif severity_class == 1:
        guidance = (
            "Prediction indicates mild diabetic retinopathy. "
            "Consider an eye examination to confirm findings and monitor progression."
        )
    elif severity_class == 2:
        guidance = (
            "Prediction indicates moderate diabetic retinopathy. "
            "Please consult an ophthalmologist for a comprehensive evaluation."
        )
    elif severity_class == 3:
        guidance = (
            "Prediction indicates severe diabetic retinopathy. "
            "Prompt ophthalmology review is recommended."
        )
    elif severity_class == 4:
        guidance = (
            "Prediction indicates proliferative diabetic retinopathy. "
            "Urgent ophthalmology consultation is recommended."
        )
    else:
        raise ValueError("severity_class must be an integer in the range 0..4")

    return GuidanceResult(severity_class=severity_class, guidance=guidance)