![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

# 🧠 AI Adaptive Image Compression with OCR

An AI-based adaptive image compression system that intelligently preserves
important visual regions (such as text and fine details) while aggressively
compressing less important areas to significantly reduce file size.

---

## 🚀 Overview

Traditional image compression applies uniform compression across the entire image,
often degrading critical information like text and details.

This project introduces an **adaptive compression pipeline powered by deep learning
and OCR**, allowing the system to:
- Detect important regions automatically
- Preserve readability and clarity
- Reduce overall image size efficiently

---

## ✨ Key Features

- 🔍 Tile-based image analysis
- 🧠 Deep Learning classifier to identify important regions
- 🔤 OCR-aware importance detection using Tesseract
- 🎯 Adaptive compression strategy
- 🖼 Visual heatmap of important regions
- 🌐 Interactive Streamlit web interface
- 📉 Up to 60–80% file size reduction with minimal quality loss

---

## 🧩 How It Works

1. The image is divided into small tiles
2. Each tile is analyzed using:
   - A trained neural network
   - OCR (to detect text presence)
3. Tiles are classified as:
   - Important
   - Not Important
4. Different compression levels are applied:
   - High quality for important tiles
   - Aggressive compression for others
5. The image is reconstructed into an optimized output

---

## 📂 Project Structure

app.py          # Streamlit web application
train.py        # Model training script
classify.py     # Dataset preprocessing & OCR labeling

Other folders (dataset, processed, outputs, models) are excluded from the repository.

---

## ⚙️ Installation

```bash
git clone https://github.com/USERNAME/ai-adaptive-image-compression.git
cd ai-adaptive-image-compression
pip install -r requirements.txt
```

---

## 🔧 External Dependencies

This project requires **Tesseract OCR** to be installed separately.

- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Linux: sudo apt install tesseract-ocr
- macOS: brew install tesseract

Make sure Tesseract is added to your system PATH.

Full technical documentation is available in `PROJECT_DOCUMENTATION.md`.


---

## ▶️ Running the Application

```bash
streamlit run app.py
```

---

## 🧪 Training the Model

```bash
python train.py
```

Dataset and trained models are excluded due to size.

---

## 🛠 Technologies Used

Python, TensorFlow, OpenCV, Tesseract OCR, Streamlit, NumPy, Scikit-learn

---

## 👤 Author

Ahmed Al Faleet
Software Engineering Student – Libya 🇱🇾
