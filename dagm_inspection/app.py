import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import torch

# Optional: model imports kept for future real-model use
from src.model import DefectCNN
from src.transforms import image_transform
from src.gradcam import GradCAM

# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
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
# Safe Model Loader (future-ready)
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
# CLASSICAL DEFECT DETECTION (ROBUST & VISIBLE)
# --------------------------------------------------
def classical_defect_detection(image_rgb):
    """
    image_rgb: RGB numpy image
    returns: annotated RGB image, defect_score
    """

    h, w, _ = image_rgb.shape

    # 1. Grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # 2. Contrast enhancement (critical)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Edge detection
    edges = cv2.Canny(enhanced, 60, 140)

    # 4. Morphological closing (connect thin scratches)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 5. Fill regions
    filled = cv2.dilate(closed, kernel, iterations=2)

    # 6. Find contours
    contours, _ = cv2.findContours(
        filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    overlay = image_rgb.copy()
    defect_score = 0.0

    if contours:
        # Largest meaningful contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 500:  # noise suppression threshold
            defect_score = area / (h * w)

            # Bounding box (RED)
            x, y, bw, bh = cv2.boundingRect(largest)
            cv2.rectangle(
                overlay,
                (x, y),
                (x + bw, y + bh),
                (255, 0, 0),  # RED (RGB)
                4
            )

            # Enclosing circle (BLUE)
            (cx, cy), radius = cv2.minEnclosingCircle(largest)
            cv2.circle(
                overlay,
                (int(cx), int(cy)),
                int(radius),
                (0, 0, 255),  # BLUE (RGB)
                4
            )

            # Label
            cv2.putText(
                overlay,
                "DEFECT",
                (x, max(y - 12, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                3
            )

    return overlay, defect_score

# --------------------------------------------------
# PAGE 1 — OVERVIEW
# --------------------------------------------------
if page == "🏠 Project Overview":
    st.title("AI-Based Industrial Defect Detection")

    st.markdown("""
    This system detects **surface defects in industrial materials** using a
    **hybrid inspection approach**:

    - **Deep Learning (when a trained model is available)**
    - **Classical Computer Vision (robust demo mode)**

    The goal is **accurate, explainable, and visually clear defect localization**.
    """)

    st.info(
        "This public demo uses classical inspection to ensure real detection "
        "even without a trained AI model."
    )

# --------------------------------------------------
# PAGE 2 — INSPECT IMAGE
# --------------------------------------------------
elif page == "🔍 Inspect Industrial Image":
    st.title("Inspect Industrial Image")

    st.markdown("""
    **Supported Images**
    - Industrial surface textures
    - Metal, fabric, material surfaces
    - Formats: BMP, PNG, JPG
    """)

    uploaded = st.file_uploader(
        "Upload an industrial surface image",
        type=["bmp", "png", "jpg", "jpeg"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        model, cam, device = load_model_safe()

        # ---------------- DEMO / CLASSICAL MODE ----------------
        if model is None:
            st.warning("⚙ Running in Classical Inspection Mode")

            overlay, defect_score = classical_defect_detection(image_np)

            st.subheader("🔎 Inspection Result")

            if defect_score > 0.003:
                st.error(f"⚠ Defect Detected (Severity Score: {defect_score:.4f})")
            else:
                st.success(f"✅ No Significant Defect (Score: {defect_score:.4f})")

            st.subheader("🧠 Defect Localization")
            st.image(
                overlay,
                caption="🔴 Red box: defect boundary | 🔵 Blue circle: defect region",
                use_column_width=True
            )

            st.info("""
            **How this works (Simple Explanation):**
            - Texture irregularities create strong edges
            - Morphological operations group defect regions
            - The largest abnormal region is marked explicitly
            """)

        # ---------------- AI MODE (FUTURE READY) ----------------
        else:
            x = image_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                prob = torch.sigmoid(model(x)).item()

            label = "DEFECT" if prob > 0.5 else "NO DEFECT"

            st.subheader("🔎 AI Prediction")
            st.write(f"**Result:** {label}")
            st.write(f"**Confidence:** {prob:.2f}")

# --------------------------------------------------
# PAGE 3 — ETHICS
# --------------------------------------------------
elif page == "⚠ Limitations & Ethics":
    st.title("Limitations & Ethical Considerations")

    st.markdown("""
    ### Limitations
    - Classical mode detects **texture irregularities**, not semantic meaning
    - AI performance depends on training data quality
    - Lighting and resolution affect detection

    ### Ethical Use
    - Designed as a **decision-support tool**
    - Human verification is required
    - Explainability improves trust and accountability

    ### Responsible AI
    Hybrid inspection ensures robustness and transparency.
    """)
