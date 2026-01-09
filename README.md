# Industrial Defect Detection using AI

## Overview
This project implements an AI-based industrial surface defect detection system
using deep learning and explainable AI techniques.

The system analyzes industrial texture images and predicts whether a surface
contains defects. Explainability is provided using Grad-CAM visualization.

## Features
- Deep learning-based defect detection
- User-friendly Streamlit web interface
- Explainable AI (Grad-CAM)
- Sample images for testing
- Ethical and limitation-aware design

## Use Cases
- Manufacturing quality control
- Surface inspection in automotive industry
- Industrial AI research and education

## Input Image Requirements
- Industrial surface images (texture-based)
- Formats: BMP, PNG, JPG
- Grayscale or low-color variation preferred

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
