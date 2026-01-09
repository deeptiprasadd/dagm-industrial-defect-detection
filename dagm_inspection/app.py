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
# Classical CV Defect Detection (DEMO MODE)
# --------------------------------------------------
def classical_defect_detection(image_np):
    """
    image_np: RGB numpy image
    returns: RGB image with box + circle, defect_score
    """

    # --- 1. Convert to grayscale ---
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # --- 2. Enhance contrast (CRITICAL) ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # --- 3. Edge + region detection ---
    edges = cv2.Canny(enhanced, 60, 140)

    # --- 4. Morphological closing (connect scratch) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    # --- 5. Fill regions ---
    filled = cv2.dilate(closed, kernel, iterations=2)

    # --- 6. Find contours ---
    contours, _ = cv2.findContours(
        filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Copy image for drawing
    overlay = image_np.copy()
    h, w, _ = image_np.shape
    defect_score = 0.0

    if contours:
        # --- 7. Select largest meaningful contour ---
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 500:  # STRICT threshold (prevents noise)
            defect_score = area / (h * w)

            # --- Bounding box ---
            x, y, bw, bh = cv2.boundingRect(largest)
            cv2.rectangle(
                overlay,
                (x, y),
                (x + bw, y + bh),
                (255, 0, 0),  # RED (RGB)
                4             # THICK
            )

            # --- Enclosing circle ---
            (cx, cy), radius = cv2.minEnclosingCircle(largest)
            cv2.circle(
                overlay,
                (int(cx), int(cy)),
                int(radius),
                (0, 0, 255),  # BLUE (RGB)
                4
            )

            # --- Label ---
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
    This system detects surface defects in industrial materials using:

    - **Deep Learning (when trained model is available)**
    - **Classical Computer Vision (fallback inspection mode)**

    The design ensures the system remains functional, explainable,
    and reliable even without a trained AI model.
    """)

# --------------------------------------------------
# PAGE 2 — INSPECT IMAGE
# --------------------------------------------------
elif page == "🔍 Inspect Industrial Image":
    st.title("Inspect Industrial Image")

    st.markdown("""
    **Supported Images**
    - Industrial surface textures
    - Metal, fabric, material surfaces
    - BMP, PNG, JPG formats
    """)

    uploaded = st.file_uploader(
        "Upload an industrial surface image",
        type=["bmp", "png", "jpg", "jpeg"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)

        st.image(image, caption="Uploaded Image")

        model, cam, device = load_model_safe()

        # ---------------- DEMO MODE ----------------
        if model is None:
            st.warning("⚙ Running in Classical Inspection Mode (Demo)")

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
            - Sharp texture changes are detected
            - Edges and irregular patterns are highlighted
            - Higher abnormal pixel density indicates defects
            """)

        # ---------------- REAL MODEL ----------------
        else:
            x = image_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                prob = torch.sigmoid(model(x)).item()

            label = "DEFECT" if prob > 0.5 else "NO DEFECT"

            st.subheader("🔎 AI Prediction")
            st.write(f"**Result:** {label}")
            st.write(f"**Confidence:** {prob:.2f}")

            heatmap = cam.generate(x)
            heatmap = cv2.resize(heatmap, image.size)
            heatmap = cv2.applyColorMap(
                np.uint8(255 * heatmap), cv2.COLORMAP_JET
            )

            overlay = cv2.addWeighted(
                image_np, 0.6, heatmap, 0.4, 0
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
    - Classical mode detects texture irregularities, not semantic defects
    - AI mode depends on training data quality
    - Lighting and resolution affect results

    ### Ethical Considerations
    - System is a **decision-support tool**
    - Human verification is required
    - Explainability improves trust

    ### Responsible AI
    Hybrid design ensures transparency and robustness.
    """)

