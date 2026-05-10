# Computer Vision System: Image Classification, Object Detection & OCR

## Overview

A comprehensive computer vision pipeline combining image classification (CNN + Transfer Learning), object detection (YOLO), and text extraction (OCR) into a single integrated system.

## Technologies Used

- **Python** — core language
- **TensorFlow / Keras** — CNN and Transfer Learning models
- **MobileNetV2** — pre-trained model for transfer learning
- **YOLOv8 (Ultralytics)** — real-time object detection
- **Tesseract OCR + pytesseract** — text extraction from images
- **OpenCV** — image preprocessing
- **Matplotlib** — visualisation
- **Google Colab (GPU)** — training environment

## Setup

### Install dependencies

```bash
pip install tensorflow ultralytics pytesseract opencv-python
apt-get install tesseract-ocr
```

### Run in Google Colab

1. Open the notebook in Google Colab
2. Set runtime to **GPU**
3. Run all cells in order

## Project Structure

The notebook is divided into 5 stages:

### Stage 1 — Image Classification (CIFAR-10)
- Custom CNN with Conv2D, MaxPooling, Dropout layers
- Data augmentation (rotation, zoom, horizontal flip)
- Trained for 20 epochs

### Stage 2 — Transfer Learning (MobileNetV2)
- Images resized from 32×32 to 96×96
- MobileNetV2 pre-trained on ImageNet as base
- Fine-tuned for 10-class CIFAR-10 classification
- Compared against custom CNN

### Stage 3 — Object Detection (YOLOv8)
- YOLOv8n model for real-time detection
- Tested on street/urban scene images
- Identifies and labels multiple objects per image

### Stage 4 — OCR Text Extraction
- OpenCV preprocessing (grayscale, thresholding)
- Tesseract OCR on original and preprocessed images
- Comparison of OCR quality with and without preprocessing

### Stage 5 — Integrated System
- Single function that accepts any image and automatically:
  - Detects objects with YOLO
  - Extracts text if present
  - Returns structured analysis results

## Models Compared

| Model | Dataset | Epochs | Notes |
|---|---|---|---|
| Custom CNN | CIFAR-10 | 20 | Built from scratch |
| MobileNetV2 | CIFAR-10 | 10 | Transfer learning from ImageNet |
| YOLOv8n | Pre-trained | — | Object detection |

## Key Concepts Demonstrated

- **Convolutional Neural Networks** — image feature extraction
- **Data Augmentation** — improving model generalisation
- **Transfer Learning** — reusing pre-trained weights
- **Object Detection** — bounding boxes and class labels
- **OCR Pipeline** — image preprocessing + text recognition
- **System Integration** — combining multiple AI models into one pipeline
