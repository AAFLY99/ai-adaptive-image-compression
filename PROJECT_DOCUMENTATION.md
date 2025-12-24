# 📘 Project Documentation
## AI-Based Adaptive Image Compression with OCR

---

## 1. Introduction

This project implements an **AI-driven adaptive image compression system**
that preserves visually important regions (especially text) while reducing
overall image size.

The system combines:
- Optical Character Recognition (OCR)
- Deep Learning (CNN-based classifier)
- Adaptive resolution scaling
- Interactive visualization via Streamlit

---

## 2. System Overview

Traditional image compression techniques apply uniform compression across
the entire image, which often degrades important information.

This system analyzes image content and applies **content-aware compression**
based on predicted importance.

---

## 3. Pipeline Workflow

1. Input image is uploaded
2. Image is split into fixed-size tiles
3. OCR detects text regions and generates spatial masks
4. CNN classifies each tile as important or not
5. Adaptive resolution scaling is applied
6. Tiles are reconstructed into final compressed image
7. Compression statistics are displayed

---

## 4. OCR Module

- Uses **Tesseract OCR**
- Detects text bounding boxes
- Converts them into binary spatial masks
- Masks are used as an additional input channel

This improves detection of semantically important regions.

---

## 5. Deep Learning Model

### Architecture
- Backbone: MobileNetV2 (pretrained on ImageNet)
- Input: RGB + OCR mask (4 channels)
- Output: Binary importance score (sigmoid)

### Training Strategy
- Binary cross-entropy loss
- Class weighting to handle imbalance
- Early stopping and learning rate scheduling

---

## 6. Adaptive Compression Strategy

- Important tiles:
  - High resolution
  - High-quality interpolation
- Non-important tiles:
  - Aggressive downscaling
  - Lower interpolation quality

This approach significantly reduces file size while preserving clarity.

---

## 7. User Interface

The Streamlit web interface allows:
- Image upload
- Real-time parameter tuning
- Side-by-side comparison
- Compression statistics
- Importance heatmap visualization
- Download of compressed image

---

## 8. Results

- Up to 60–80% reduction in image size
- Text readability preserved
- Clear visualization of important regions

---

## 9. Limitations

- OCR performance depends on image quality
- Binary classification only
- Images only (no video)

---

## 10. Future Improvements

- Face detection integration
- Video compression support
- Transformer-based models
- Real-time deployment

---

## 11. Conclusion

This project demonstrates how combining OCR with deep learning enables
intelligent, content-aware image compression suitable for real-world
applications.

---

## Author
Ahmed Al Faleet  
Software Engineering Student – Libya 🇱🇾
