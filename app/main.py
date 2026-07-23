import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH (fix for Streamlit Cloud)
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
import streamlit as st
from pathlib import Path
from datetime import datetime
import numpy as np

from src.data.preprocessing import PreprocessConfig, preprocess_image, read_image_rgb
from src.quality.image_quality import check_image_quality
from src.assessment.risk_assessment import assess_risk
from src.assessment.clinical_guidance import generate_guidance
from src.explainability.xai_pipeline import load_model_from_checkpoint, predict_with_gradcam
from src.report.pdf_generator import ReportData, generate_pdf_report

st.set_page_config(page_title="DR Severity + XAI", layout="wide")

st.title("Explainable Diabetic Retinopathy Severity Classification")
st.caption("Assistive screening tool — not a medical diagnosis.")

MODEL_PATH = Path("models/best_model.pth")


@st.cache_resource
def load_model():
    model, device = load_model_from_checkpoint(MODEL_PATH)
    return model, device


uploaded = st.file_uploader("Upload a retinal fundus image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    # Save uploaded image temporarily
    tmp_dir = Path("reports/generated")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    tmp_path.write_bytes(uploaded.getbuffer())

    # Read original image
    orig_rgb = read_image_rgb(tmp_path)

    st.subheader("Uploaded Image")
    st.image(orig_rgb, use_container_width=True)

    # Preprocess (for quality check and model)
    pre_cfg = PreprocessConfig(image_size=384, crop_black=True, enhance_contrast=False)
    img_pre_rgb = preprocess_image(tmp_path, pre_cfg)

    st.subheader("Image Quality Check")
    q = check_image_quality(img_pre_rgb)
    if not q.quality_ok:
        st.error(f"Image quality check failed: {q.message}")
        st.stop()
    else:
        st.success("Image quality OK")

    # Load model and run XAI prediction
    model, device = load_model()
    result = predict_with_gradcam(model, device, tmp_path)

    # Risk + guidance
    risk = assess_risk(result.pred_class)
    guidance = generate_guidance(result.pred_class)

    # Display results
    st.subheader("Prediction")
    col1, col2, col3 = st.columns(3)
    col1.metric("Severity", result.pred_label)
    col2.metric("Confidence", f"{result.confidence*100:.2f}%")
    col3.metric("Risk Level", risk.risk_level)

    st.subheader("Explainability (Grad-CAM)")
    c1, c2 = st.columns(2)
    c1.image(result.overlay_rgb, caption="Grad-CAM Overlay", use_container_width=True)
    c2.image(result.heatmap_rgb, caption="Grad-CAM Heatmap", use_container_width=True)

    st.subheader("General Guidance")
    st.write(guidance.guidance)
    st.info(guidance.disclaimer)

    # PDF report
    report_data = ReportData(
        patient_id=None,
        prediction_label=result.pred_label,
        confidence=result.confidence,
        risk_level=risk.risk_level,
        guidance=guidance.guidance,
        disclaimer=guidance.disclaimer,
        created_at=datetime.now(),
    )

    pdf_bytes = generate_pdf_report(
        retinal_image_rgb=img_pre_rgb,
        gradcam_overlay_rgb=result.overlay_rgb,
        report=report_data,
    )

    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="dr_report.pdf",
        mime="application/pdf",
    )
else:
    st.info("Upload an image to start.")