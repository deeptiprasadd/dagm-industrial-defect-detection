import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import os

from src.model import DefectCNN
from src.transforms import image_transform
from src.gradcam import GradCAM

st.set_page_config(
    page_title="Industrial Defect Detection",
    page_icon="🔍",
    layout="centered"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Project Overview",
        "🔍 Inspect Industrial Image",
        "⚠ Limitations & Ethics"
    ]
)

# --------------------------------------------------
# Safe Model Loader
# --------------------------------------------------
@st.cache_resource
def load_model_safe():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists("defect_model.pth"):
        return None, None, device

    model = DefectCNN().to(device)
    model.load_state_dict(
        torch.load("defect_model.pth", map_location=device)
    )
    model.eval()

    cam = GradCAM(model, model.model.layer4[-1])
    return model, cam, device

# --------------------------------------------------
# PAGE 1 — OVERVIEW
# --------------------------------------------------
if page == "🏠 Project Overview":
    st.title("AI-Based Industrial Defect Detection")

    st.markdown("""
    This application demonstrates how **Artificial Intelligence**
    can be used to detect surface defects in industrial materials.

    The system is designed as a **decision-support tool** and
    includes explainability to increase trust in predictions.
    """)

    st.info(
        "Note: This public demo runs in **demo mode** unless a trained model is provided."
    )

# --------------------------------------------------
# PAGE 2 — INSPECT IMAGE
# --------------------------------------------------
elif page == "🔍 Inspect Industrial Image":
    st.title("Inspect Industrial Image")

    uploaded = st.file_uploader(
        "Upload an industrial surface image",
        type=["bmp", "png", "jpg", "jpeg"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image")

        model, cam, device = load_model_safe()

        x = image_transform(image).unsqueeze(0).to(device)

        # ---------------- DEMO MODE ----------------
        if model is None:
            st.warning("Running in DEMO MODE (no trained model available).")

            gray = np.array(image.convert("L"))
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(
                np.array(image), 0.6, heatmap, 0.4, 0
            )

            st.subheader("🧠 Explainability (Illustrative)")
            st.image(
                overlay,
                caption="Illustrative heatmap (demo mode)"
            )

            st.info(
                "In a full deployment, this heatmap would highlight regions influencing the AI decision."
            )

        # ---------------- REAL MODEL ----------------
        else:
            with torch.no_grad():
                prob = torch.sigmoid(model(x)).item()

            label = "DEFECT" if prob > 0.5 else "NO DEFECT"

            st.subheader("🔎 Prediction")
            st.write(f"**Result:** {label}")
            st.write(f"**Confidence:** {prob:.2f}")

            heatmap = cam.generate(x)
            heatmap = cv2.resize(heatmap, image.size)
            heatmap = cv2.applyColorMap(
                np.uint8(255 * heatmap), cv2.COLORMAP_JET
            )

            overlay = cv2.addWeighted(
                np.array(image), 0.6, heatmap, 0.4, 0
            )

            st.subheader("🧠 Explainability (Grad-CAM)")
            st.image(overlay)

# --------------------------------------------------
# PAGE 3 — ETHICS
# --------------------------------------------------
elif page == "⚠ Limitations & Ethics":
    st.title("Limitations & Ethical Considerations")

    st.markdown("""
    ### Limitations
    - Works only on industrial texture images
    - Not suitable for natural or medical images
    - Sensitive to lighting and resolution

    ### Ethical Use
    - Should not replace human inspectors
    - Used as a decision-support system
    - Explainability improves transparency

    ### Responsible AI
    Human oversight is required for critical decisions.
    """)

