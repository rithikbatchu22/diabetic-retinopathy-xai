from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


@dataclass
class ReportData:
    patient_id: Optional[str]
    prediction_label: str
    confidence: float
    risk_level: str
    guidance: str
    disclaimer: str
    created_at: datetime


def _np_rgb_to_pil(img_rgb_uint8: np.ndarray) -> Image.Image:
    if img_rgb_uint8.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB image")
    return Image.fromarray(img_rgb_uint8)


def generate_pdf_report(
    retinal_image_rgb: np.ndarray,
    gradcam_overlay_rgb: np.ndarray,
    report: ReportData,
) -> bytes:
    """
    Generate a PDF report and return PDF bytes (for Streamlit download).

    Why bytes:
    - Streamlit can offer a download button from bytes
    - also easy to save to disk
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, height - 2 * cm, "Diabetic Retinopathy AI Screening Report")

    # Meta
    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, height - 2.7 * cm, f"Date/Time: {report.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if report.patient_id:
        c.drawString(2 * cm, height - 3.2 * cm, f"Patient ID: {report.patient_id}")

    # Prediction block
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, height - 4.1 * cm, "Prediction Summary")

    c.setFont("Helvetica", 11)
    c.drawString(2 * cm, height - 4.8 * cm, f"Predicted Severity: {report.prediction_label}")
    c.drawString(2 * cm, height - 5.4 * cm, f"Confidence: {report.confidence * 100:.2f}%")
    c.drawString(2 * cm, height - 6.0 * cm, f"Risk Level: {report.risk_level}")

    # Images
    # Convert numpy -> PIL -> temporary in-memory PNG
    img1 = _np_rgb_to_pil(retinal_image_rgb)
    img2 = _np_rgb_to_pil(gradcam_overlay_rgb)

    img1_buf = BytesIO()
    img2_buf = BytesIO()
    img1.save(img1_buf, format="PNG")
    img2.save(img2_buf, format="PNG")
    img1_buf.seek(0)
    img2_buf.seek(0)

    # Layout
    y_img_top = height - 7.0 * cm
    img_w = 8.0 * cm
    img_h = 8.0 * cm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(2 * cm, y_img_top, "Input Image")
    c.drawString(11 * cm, y_img_top, "Grad-CAM Overlay")

    c.drawImage(ImageReader(img1_buf), 2 * cm, y_img_top - img_h - 0.5 * cm, width=img_w, height=img_h)
    c.drawImage(ImageReader(img2_buf), 11 * cm, y_img_top - img_h - 0.5 * cm, width=img_w, height=img_h)

    # Guidance
    y_text = y_img_top - img_h - 2.0 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y_text, "General Guidance")
    c.setFont("Helvetica", 10)
    text_obj = c.beginText(2 * cm, y_text - 0.7 * cm)
    for line in wrap_text(report.guidance, max_len=95):
        text_obj.textLine(line)
    c.drawText(text_obj)

    # Disclaimer (bottom)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2 * cm, 2.2 * cm, "Disclaimer")
    c.setFont("Helvetica", 8)
    text_obj = c.beginText(2 * cm, 1.7 * cm)
    for line in wrap_text(report.disclaimer, max_len=110):
        text_obj.textLine(line)
    c.drawText(text_obj)

    c.showPage()
    c.save()

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def wrap_text(text: str, max_len: int = 90):
    words = text.split()
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > max_len:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += len(w) + 1
    if cur:
        lines.append(" ".join(cur))
    return lines


# reportlab needs this import for drawImage with BytesIO
from reportlab.lib.utils import ImageReader