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
    FINAL industrial-grade defect detection.
    Balanced: not over-sensitive, not over-strict.
    """

    h, w, _ = image_rgb.shape
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Contrast enhancement (critical for subtle scratches)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Gradient magnitude (texture change)
    gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. Threshold using percentile (NOT mean/std – too fragile)
    thresh_val = np.percentile(grad, 97.5)
    _, binary = cv2.threshold(grad, thresh_val, 255, cv2.THRESH_BINARY)

    # 4. Morphology to connect thin scratches
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # 5. Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    overlay = image_rgb.copy()
    defect_score = 0.0

    if not contours:
        return overlay, 0.0

    # 6. Evaluate contours properly
    best_cnt = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150:  # absolute minimum evidence
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = max(bw / (bh + 1e-6), bh / (bw + 1e-6))

        # scratches are elongated → high aspect ratio
        if aspect_ratio < 2.0:
            continue

        score = area * aspect_ratio
        if score > best_score:
            best_score = score
            best_cnt = cnt

    if best_cnt is None:
        return overlay, 0.0

    # 7. Draw results (GUARANTEED visible)
    area = cv2.contourArea(best_cnt)
    defect_score = area / (h * w)

    x, y, bw, bh = cv2.boundingRect(best_cnt)
    cv2.rectangle(
        overlay,
        (x, y),
        (x + bw, y + bh),
        (255, 0, 0),  # RED
        4
    )

    (cx, cy), radius = cv2.minEnclosingCircle(best_cnt)
    cv2.circle(
        overlay,
        (int(cx), int(cy)),
        int(radius),
        (0, 0, 255),  # BLUE
        4
    )

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
