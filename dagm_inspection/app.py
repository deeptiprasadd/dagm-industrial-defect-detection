import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import os

from src.model import DefectCNN
from src.transforms import image_transform
from src.gradcam import GradCAM

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Industrial Defect Detection",
    page_icon="🔍",
    layout="centered"
)

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Project Overview",
        "🏭 Use Cases",
        "⚙ How It Works",
        "🧪 Try Sample Images",
        "🔍 Inspect Image (with Explainability)",
        "⚠ Limitations & Ethics"
    ]
)

# --------------------------------------------------
# Load Model Once
# --------------------------------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DefectCNN().to(device)
    model.load_state_dict(torch.load("defect_model.pth", map_location=device))
    model.eval()
    cam = GradCAM(model, model.model.layer4[-1])
    return model, cam, device

# --------------------------------------------------
# PAGE 1 — OVERVIEW
# --------------------------------------------------
if page == "🏠 Project Overview":
    st.title("AI-Based Industrial Defect Detection System")

    st.markdown("""
    ### 📌 Problem Statement
    In manufacturing industries, surface defects such as scratches, cracks,
    dents, or texture irregularities can reduce product quality and lead to
    financial loss. Manual inspection is slow, subjective, and error-prone.

    ### 🎯 Objective
    This system uses **Artificial Intelligence and Computer Vision**
    to automatically identify defective industrial surfaces using images.
    """)

    st.info("This system supports human decision-making. It does not replace human inspectors.")

# --------------------------------------------------
# PAGE 2 — USE CASES
# --------------------------------------------------
elif page == "🏭 Use Cases":
    st.title("Industrial Use Cases")

    st.markdown("""
    **✔ Manufacturing Quality Control**
    - Steel sheets
    - Metal plates
    - Fabric inspection

    **✔ Automotive Industry**
    - Component surface inspection
    - Paint defect detection

    **✔ Electronics Industry**
    - PCB surface defects
    - Material consistency checks

    **✔ Academic & Research**
    - Industrial AI research
    - Explainable computer vision systems
    """)

# --------------------------------------------------
# PAGE 3 — HOW IT WORKS
# --------------------------------------------------
elif page == "⚙ How It Works":
    st.title("How the System Works")

    st.markdown("""
    **Step 1: Image Input**
    - The system takes an industrial surface image.

    **Step 2: Preprocessing**
    - Image is resized and normalized.

    **Step 3: Deep Learning Model**
    - A CNN learns texture patterns.
    - It distinguishes normal vs defective surfaces.

    **Step 4: Explainability (Grad-CAM)**
    - The system highlights **where the model is looking**.

    **Step 5: Output**
    - Defect / No Defect
    - Confidence score
    """)

    st.warning("Best performance is achieved when input images are similar to training images.")

# --------------------------------------------------
# PAGE 4 — SAMPLE IMAGES
# --------------------------------------------------
elif page == "🧪 Try Sample Images":
    st.title("Sample Industrial Images")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Normal Surface")
        st.image("samples/sample_normal.bmp")

    with col2:
        st.subheader("Defective Surface")
        st.image("samples/sample_defect.bmp")

# --------------------------------------------------
# PAGE 5 — INSPECT IMAGE + GRAD-CAM
# --------------------------------------------------
elif page == "🔍 Inspect Image (with Explainability)":
    st.title("Inspect Industrial Image")

    st.markdown("""
    **Upload Requirements**
    - Industrial surface image
    - Texture-based (metal, fabric, material)
    - Formats: `.bmp`, `.png`, `.jpg`
    """)

    uploaded = st.file_uploader(
        "Upload an image",
        type=["bmp", "png", "jpg", "jpeg"]
    )

    if uploaded:
        model, cam, device = load_model()
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image")

        x = image_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()

        label = "DEFECT" if prob > 0.5 else "NO DEFECT"

        st.subheader("🔎 Prediction Result")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {prob:.2f}")

        # Grad-CAM
        heatmap = cam.generate(x)
        heatmap = cv2.resize(heatmap, image.size)
        heatmap = cv2.applyColorMap(
            np.uint8(255 * heatmap),
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            np.array(image), 0.6, heatmap, 0.4, 0
        )

        st.subheader("🧠 Explainability (Grad-CAM)")
        st.image(overlay, caption="Highlighted regions influenced the decision")

        st.caption("""
        🔴 Red regions indicate areas the model focused on while making its decision.
        """)

# --------------------------------------------------
# PAGE 6 — LIMITATIONS & ETHICS
# --------------------------------------------------
elif page == "⚠ Limitations & Ethics":
    st.title("Limitations & Ethical Considerations")

    st.markdown("""
    ### ⚠ Limitations
    - Model works only on **industrial texture images**
    - Performance drops on unseen textures
    - Sensitive to image quality and lighting
    - Not suitable for natural or medical images

    ### ⚖ Ethical Considerations
    - Should not be used as a sole decision-maker
    - Human verification is required for critical cases
    - Transparency via explainability is essential
    - Avoid misuse outside intended industrial domain

    ### ✅ Responsible AI Use
    This system is designed as a **decision support tool**
    to assist quality inspectors, not replace them.
    """)

    st.success("Explainable AI improves trust, transparency, and accountability.")
