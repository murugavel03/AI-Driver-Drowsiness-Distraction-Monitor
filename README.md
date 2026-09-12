# 🎭 Real-Time Face Mask Detection System

A **Computer Vision** project that detects whether people are wearing face masks in real-time using a webcam, video file, or static image. Built with **Python**, **OpenCV**, and **Transfer Learning (MobileNetV2)**.

---

## 📌 Project Overview

| Attribute      | Details                                      |
|---------------|----------------------------------------------|
| **Domain**     | Computer Vision / Deep Learning              |
| **Language**   | Python 3.8+                                  |
| **Framework**  | TensorFlow / Keras + OpenCV                  |
| **Model**      | MobileNetV2 (Transfer Learning)              |
| **Task**       | Binary Classification: Mask / No Mask        |
| **Dataset**    | ~1,376 images (with_mask + without_mask)     |
| **Accuracy**   | ~98% on validation set                       |

---

## 📁 Project Structure

```
face-mask-detector/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── download_dataset.py                # Step 1: Download dataset & face detector
├── train_mask_detector.py             # Step 2: Train the MobileNetV2 model
├── evaluate.py                        # Step 3: Evaluate model performance
├── detect_mask_video.py               # Step 4a: Real-time webcam/video detection
├── detect_from_image.py               # Step 4b: Static image detection
│
├── dataset/
│   ├── with_mask/                     # Training images (masked faces)
│   └── without_mask/                  # Training images (unmasked faces)
│
├── face_detector/
│   ├── deploy.prototxt                # OpenCV DNN face detector config
│   └── res10_300x300_ssd_iter_140000.caffemodel  # Face detector weights
│
├── models/
│   └── mask_detector.model            # Saved Keras model (generated after training)
│
└── plots/
    ├── mask_training_plot.png         # Training accuracy/loss curves
    └── confusion_matrix.png           # Evaluation confusion matrix
```

---

## ⚙️ Environment Setup

### Prerequisites
- Python **3.8 or higher**
- pip (Python package manager)
- A webcam (for real-time detection)
- Internet connection (for downloading dataset and pretrained weights)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/face-mask-detector.git
cd face-mask-detector
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ If you encounter issues with TensorFlow on older hardware, use:
> `pip install tensorflow-cpu`

---

## 🚀 Running the Project

### Step 4 — Download Dataset and Face Detector

```bash
python download_dataset.py
```

This will:
- Download the **face mask image dataset** (~45 MB)
- Download the **OpenCV DNN face detector** (ResNet10 SSD Caffe model)
- Organize images into `dataset/with_mask/` and `dataset/without_mask/`

### Step 5 — Train the Model

```bash
python train_mask_detector.py
```

**Optional arguments:**
```bash
python train_mask_detector.py --epochs 20 --dataset dataset --model models/mask_detector.model
```

This will:
- Load and preprocess images
- Apply data augmentation
- Fine-tune **MobileNetV2** on your dataset
- Save trained model to `models/mask_detector.model`
- Save training plot to `plots/mask_training_plot.png`

> Expected training time: ~5–15 minutes on CPU, ~2–3 minutes on GPU.

### Step 6 — Evaluate the Model

```bash
python evaluate.py
```

Outputs:
- Accuracy, Precision, Recall, F1-Score
- Per-class Classification Report
- Confusion matrix saved to `plots/confusion_matrix.png`

### Step 7a — Real-Time Detection (Webcam)

```bash
python detect_mask_video.py
```

Press **`q`** to quit.

**Use a video file instead:**
```bash
python detect_mask_video.py --input path/to/video.mp4
```

**Save output video:**
```bash
python detect_mask_video.py --output output.avi
```

### Step 7b — Static Image Detection

```bash
python detect_from_image.py --image path/to/image.jpg
```

**Save annotated result:**
```bash
python detect_from_image.py --image path/to/image.jpg --output result.jpg
```

---

## 🧠 How It Works

```
Input Frame
    │
    ▼
OpenCV DNN Face Detector (ResNet10 SSD)
    │  Detects face bounding boxes
    ▼
Extract Face ROIs
    │  Crop + resize to 224×224
    ▼
MobileNetV2 Classifier (Fine-Tuned)
    │  Predicts: [Mask, No Mask]
    ▼
Annotated Output
    │  Bounding box + label + confidence
    ▼
Display / Save
```

### Model Architecture

1. **Face Detection**: OpenCV DNN with pre-trained ResNet10 SSD (Caffe) — fast, accurate, CPU-friendly
2. **Classification Head**: MobileNetV2 backbone (ImageNet weights, frozen) + custom head:
   - `AveragePooling2D(7×7)`
   - `Dense(128, relu)`
   - `Dropout(0.5)`
   - `Dense(2, softmax)`

---

## 📊 Performance

| Metric     | Value   |
|-----------|---------|
| Accuracy   | ~98%    |
| Precision  | ~98%    |
| Recall     | ~98%    |
| F1-Score   | ~98%    |

*(Results on 20% held-out test split)*

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named 'cv2'` | Run `pip install opencv-python` |
| `No module named 'imutils'` | Run `pip install imutils` |
| Webcam not detected | Try `--input 0` or `--input 1` |
| Low accuracy | Increase `--epochs` to 30–40 |
| Out of memory | Reduce batch size in `train_mask_detector.py` (line `BATCH_SIZE = 16`) |
| Face not detected | Lower confidence: `--confidence 0.3` |

---

## 📚 References

- [MobileNetV2 — Howard et al., 2018](https://arxiv.org/abs/1801.04381)
- [OpenCV DNN Face Detector](https://github.com/opencv/opencv/tree/master/samples/dnn)
- [Face Mask Dataset — Prajna Bhandary](https://github.com/prajnasb/observations)
- [TensorFlow / Keras Documentation](https://www.tensorflow.org/api_docs)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

*Developed as part of the Computer Vision course project submission.*
