import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
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
# Sidebar
# --------------------------------------------------
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Project Overview",
        "🔍 Inspect Industrial Image",
        "📘 System Design Justification"
    ]
)

# --------------------------------------------------
# Load CNN (OPTIONAL, EXPLANATION ONLY)
# --------------------------------------------------
@st.cache_resource
def load_model_safe():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists("defect_model.pth"):
        return None, None, device

    model = DefectCNN().to(device)
    model.load_state_dict(torch.load("defect_model.pth", map_location=device))
    model.eval()

    cam = GradCAM(model, model.model.layer4[-1])
    return model, cam, device

# --------------------------------------------------
# CLASSICAL DEFECT DETECTION (AUTHORITATIVE)
# --------------------------------------------------
def classical_defect_detection(image_rgb):
    h, w, _ = image_rgb.shape
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Remove background texture using top-hat filtering
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_bg)
    anomaly = cv2.absdiff(gray, background)

    # 2. Normalize anomaly map
    anomaly = cv2.normalize(anomaly, None, 0, 255, cv2.NORM_MINMAX)

    # 3. Threshold only strongest anomalies
    thresh = np.percentile(anomaly, 99.5)
    _, binary = cv2.threshold(anomaly, thresh, 255, cv2.THRESH_BINARY)

    # 4. Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    overlay = image_rgb.copy()

    if not contours:
        return overlay, False

    # 5. Select SINGLE most anomalous region
    strongest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(strongest)

    # Minimum sanity check
    if area < 0.002 * (h * w):
        return overlay, False

    x, y, bw, bh = cv2.boundingRect(strongest)

    cv2.rectangle(
        overlay,
        (x, y),
        (x + bw, y + bh),
        (255, 0, 0),
        4
    )

    cv2.putText(
        overlay,
        "DEFECT (approx.)",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    return overlay, True

# --------------------------------------------------
# PAGE 1 — OVERVIEW
# --------------------------------------------------
if page == "🏠 Project Overview":
    st.title("Industrial Defect Detection System")

    st.markdown("""
## Industrial Surface Defect Inspection System

### Project Overview

This application is an **automated industrial surface inspection system**
designed to assist in **quality control and defect identification**
for texture-based manufacturing materials.

The system analyzes a **single industrial surface image at a time**
and determines whether **defect-like anomalies** are present.
When a defect is detected, the system highlights **one representative
anomalous region** to guide further human inspection.

---

### Purpose of the System

In real-world manufacturing, surface defects such as **scratches,
pits, cracks, or material inconsistencies** can affect product quality.
Manual inspection is time-consuming and subjective.

This project demonstrates how **computer vision and AI concepts**
can be applied to support inspectors by:
- Reducing manual workload
- Providing consistent inspection assistance
- Highlighting areas of potential concern

---

### How the System Works (High Level)

1. The uploaded image is analyzed for **texture irregularities**
2. Background material patterns are suppressed
3. The most prominent anomaly (if any) is identified
4. A visual marker is drawn to indicate the **approximate defect location**

The system focuses on **defect presence detection**, not
pixel-perfect segmentation.

---

### ⚠ Important Usage Notes

- Upload **only ONE image at a time**
- Images should represent **industrial surface textures**
- The highlighted region is an **approximate localization**
- Results should be **verified by a human inspector**

This system is intended as a **decision-support tool**,  
not a fully autonomous inspection solution.

---

### Intended Users

- Quality control engineers
- Manufacturing inspectors
- Industrial AI researchers
- Students learning applied computer vision

---

### Design Philosophy

This project prioritizes:
- **Stability over overconfidence**
- **Transparency over false precision**
- **Explainability over black-box decisions**

By clearly defining its scope and limitations, the system
demonstrates responsible and realistic AI design.
""")


# --------------------------------------------------
# PAGE 2 — INSPECT IMAGE
# --------------------------------------------------
elif page == "🔍 Inspect Industrial Image":
    st.title("Inspect Industrial Image")
    
    st.warning(
    "Upload **only ONE industrial surface image at a time**.\n\n"
    "This system is designed for **single-image inspection**. "
    "Uploading multiple images or stitched images may lead to incorrect results."
    )
 

    uploaded = st.file_uploader(
        "Upload an industrial surface image",
        type=["bmp", "png", "jpg", "jpeg"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        # ----- Classical Detection -----
        overlay, defect_found = classical_defect_detection(image_np)

        st.subheader("🔎 Detection Result")

        if defect_found:
            st.error("❌ DEFECT DETECTED")
            st.image(
                overlay,
                caption="🔴 Red boxes indicate detected defect regions",
                use_column_width=True
            )
        else:
            st.success("✅ NO DEFECT")

        # ----- CNN Explainability (Optional) -----
        model, cam, device = load_model_safe()
        if model is not None:
            x = image_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.sigmoid(model(x)).item()

            cam_map = cam.generate(x)
            cam_map = cv2.resize(cam_map, image.size)
            heatmap = cv2.applyColorMap(
                np.uint8(255 * cam_map), cv2.COLORMAP_JET
            )
            overlay_cam = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

            st.subheader("🧠 Explainability (Grad-CAM)")
            st.write(f"CNN confidence (informational only): {prob:.2f}")
            st.image(
                overlay_cam,
                caption="Grad-CAM highlights regions influencing the CNN",
                use_column_width=True
            )
            st.info(
    "ℹ This inspection result is based on texture analysis.\n\n"
    "For critical quality decisions, human verification is recommended."
            )


# --------------------------------------------------
# PAGE 3 — JUSTIFICATION
# --------------------------------------------------
elif page == "📘 System Design Justification":
    st.title("System Design Justification")

    st.markdown("""
### 🔍 Industrial Surface Inspection – How to Use

This tool performs **automated inspection of industrial surface textures**
to determine whether **defect-like anomalies** are present.

#### ✅ What You Should Upload
- A **single** grayscale or RGB image
- Industrial surfaces such as:
  - Metal sheets
  - Fabric rolls
  - Machined materials
- Image formats: **BMP, PNG, JPG**

#### ❌ What You Should NOT Upload
- Multiple images in one frame
- Natural scenes (people, animals, objects)
- Medical or satellite images
- Images with text overlays or annotations

#### 🧠 What the System Does
- Analyzes texture irregularities
- Suppresses background material noise
- Detects the **most prominent anomaly region**
- Highlights **one representative defect location**

#### ⚠ Important Note on Results
This system answers the question:
> **“Is there a defect-like anomaly present in this surface?”**

It does **not** guarantee pixel-perfect defect boundaries.
The highlighted region is an **approximate localization** intended
to guide human inspection.

####  Intended Use
This tool is designed as a **decision-support system**
for quality inspection — not as a replacement for human experts.
""")

