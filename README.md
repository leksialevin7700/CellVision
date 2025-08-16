# 🧬 CellVision AI

**CellVision AI** is an advanced AI-powered dashboard designed for researchers and medical professionals to analyze, classify, and visualize cell images. The platform leverages deep learning and computer vision to empower efficient, accurate, and interactive cancer cell image analysis.

---

## 🚀 Features

- **Single Image Prediction:** Upload a cell image for instant benign/malignant classification with confidence scores.
- **Batch Analysis:** Upload a CSV of images for large-scale predictions, view result distributions, and export batch results.
- **Metrics & Evaluation:** Upload ground-truth labels to generate metrics, confusion matrix, and performance reports.
- **Example Images:** Browse random samples of benign and malignant cell images for reference.
- **OpenCV Tools:** Apply and visualize image processing and augmentation, including:
  - Grayscale conversion
  - Gaussian Blur
  - Canny Edge Detection
  - Adaptive Threshold
  - Contour Detection
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Image Rotation
  - Horizontal & Vertical Flip
  - Add Gaussian Noise
- **Cell Counting:** Automated cell counting and size distribution analysis for microscopy images.
- **Time-lapse Analysis:** Track and analyze cell count changes across a sequence of images.
- **Session Prediction History:** Access and export all predictions and results for reproducibility and audit.
- **Model Info:** View details on the model architecture (ResNet-18), datasets, and latest updates.
- **Researcher Chatbot:** Ask medical research questions and get expert answers via integration with PubMed and Gemini APIs.

---

## 📊 Project Visualization  

You can explore a demo/visual overview of **CellVision AI** here:  

[![Watch the Demo](https://img.shields.io/badge/▶️%20Watch%20Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://drive.google.com/file/d/1zESqNwTT3qc-HpwFKauR1oarrPsvsQkO/view?usp=sharing)

---

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit, Custom CSS  
- **Deep Learning & Computer Vision:** PyTorch, Torchvision, ResNet-18, OpenCV, PIL  
- **Data Processing & Analysis:** Pandas, NumPy, Scikit-learn  
- **Visualization:** Matplotlib, Seaborn  
- **Utilities & APIs:** Requests, PubMed API, Gemini API, XML, OS, Random, Datetime, IO  
- **Session Management:** Streamlit Session State  

---

## 📥 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/leksialevin7700/cellvision-ai.git
   cd cellvision-ai
