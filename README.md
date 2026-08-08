# Real-Time Surgical Tool Detection Using YOLO

An AI-powered computer vision system designed for smart operating rooms (OR) to track and identify surgical instruments in real-time under constrained compute resources. This project implements and benchmarks three variants from the YOLO family (**YOLOv8n**, **YOLOv8m**, and **YOLOv9c**) to evaluate their suitability for healthcare automation and procedural safety.

---

## 📌 Features
- **Multi-Model Support:** Native integration and benchmarking of YOLOv8n (Nano), YOLOv8m (Medium), and YOLOv9c (Custom with Programmable Gradient Information).
- **Real-Time Detection:** Sub-50ms inference speed optimized for live surgical feeds and video tracking.
- **Dual-Input Interface:** Web application UI that supports both manual drag-and-drop image uploads and live camera feeds.
- **Robust Multi-Scale Fusion:** Uses enhanced Feature Pyramid Networks (FPN) and Anchor-Free center-point prediction to handle cluttered surgical trays and instrument occlusions.

---

## 📊 Dataset & Preprocessing

- **Source:** [Kaggle Labeled Surgical Tools Dataset](https://kaggle.com).
- **Classes Managed (Minimum >200 samples per class):**
  1. `Scalpel`
  2. `Straight Dissection Clamp`
  3. `Straight Mayo Scissor`
  4. `Curved Mayo Scissor`
- **Data Split:** 70% Training (1,834 images), 20% Validation (786 images), 10% Testing.
- **Pipeline Processing:** Annotation conversion (COCO → YOLO format), uniform 640×640 image resizing, and extensive data augmentation (Mosaic 9x, Random Flipping, and HSV adjustments).

---

## 📈 Performance Comparison Matrix

The models were trained for **100 epochs** with a batch size of 16 using an NVIDIA GPU. YOLOv9c emerged as the winning architecture due to its superior capacity to eliminate off-diagonal class confusion.

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Inference Speed / Capacity |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **YOLOv8n** | 96.57% | 96.85% | 99.01% | 87.32% | ~45ms (Fastest/Lightweight ~3.2M params) |
| **YOLOv8m** | 97.21% | 97.19% | 99.31% | 88.91% | Balanced Trade-off (~25.4M params) |
| **YOLOv9c** | **97.69%** | **97.86%** | **99.35%** | **89.24%** | Best Generalization (~25.5M params) |

---

## 🏗️ System & Frontend Architecture

The system uses a clean, decoupled modular frontend deployment pattern built on **Streamlit** and powered by a **PyTorch/Ultralytics** backend processing runtime.

```text
SURGI-YOLO Frontend Layer (Streamlit)
 ├── Navigation Module (Home, Upload/Camera, About Pages)
 ├── Model Selection (Dynamic weight swapping)
 └── Results Visualization (Side-by-side original vs. high-visibility bounding boxes)

Backend Integration Layer
 ├── PyTorch Deep Learning Runtime (GPU Acceleration with CPU Fallback)
 └── Image Preprocessing & Inference Validation Pipeline
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com
cd Surgical-Tool-Detection-YOLO
```

### 2. Environment Setup
Set up a virtual environment and install the required dependencies (PyTorch, Ultralytics, Streamlit):
```bash
pip install -r requirements.txt
```

### 3. Data Configuration (`data.yaml`)
Ensure your configuration file points directly to your path layout structure:
```yaml
path: Surgical-Dataset/surgical_yolo_dataset
train: images/train
val: images/val

names:
  0: Scalpel
  1: Straight Dissection Clamp
  2: Straight Mayo Scissor
  3: Curved Mayo Scissor
```

### 4. Running the Web App Interface
Launch the dashboard platform locally:
```bash
streamlit run app.py
```

---

