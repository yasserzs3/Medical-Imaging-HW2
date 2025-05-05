# Medical Imaging Object Detection Project

A comprehensive pipeline for medical image object detection, supporting multiple state-of-the-art models including YOLOv8, Faster R-CNN, and SSD. The project provides a complete ETL pipeline for processing medical images and training object detection models.

## Features

- Multi-model support:
  - YOLOv8 (Ultralytics implementation)
  - Faster R-CNN (PyTorch implementation)
  - SSD (TensorFlow implementation)
- Complete ETL pipeline for medical image processing
- Data augmentation and normalization capabilities
- Comprehensive training pipeline with metrics tracking
- TensorBoard integration for training visualization
- Support for COCO format annotations

## Project Structure

```
├── etl.py                 # Main ETL pipeline script
├── train.py              # Multi-model training script
├── infer.py              # Model inference script
├── models/               # Model implementations
│   ├── yolov8/          # YOLOv8 implementation
│   ├── faster_rcnn/     # Faster R-CNN implementation
│   └── ssd/             # SSD implementation
├── data/                 # Data directories
│   ├── raw/             # Raw input data (COCO format)
│   ├── processed/       # Processed datasets
│   └── runs/            # Training logs and results
└── etl/                 # ETL modules
    ├── extract.py       # Data extraction
    ├── transform.py     # Data transformation
    └── load.py          # Data loading
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medical-imaging-detection.git
cd medical-imaging-detection
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation (ETL)

Prepare your medical imaging data for model training:

```bash
python etl.py --data-dir data/raw --output-dir data/processed --model yolo --apply-augmentation
```

#### ETL Options:
- `--data-dir`: Input data directory (default: `data/raw`)
- `--output-dir`: Output directory for processed data (default: `data/processed`)
- `--model`: Target model format (`yolo`, `ssd`) - can be specified multiple times
- `--apply-augmentation`: Enable data augmentation
- `--apply-normalization`: Enable image normalization
- `--quiet`: Suppress verbose output

### 2. Model Training

Train object detection models:

```bash
python train.py --model yolo --epochs 50 --batch-size 16
```

#### Training Options:
- `--model`: Model architecture (`yolo`, `faster_rcnn`, `ssd`)
- `--runs-dir`: Training logs directory (default: `data/runs`)
- `--models-dir`: Model weights directory (default: `models`)
- `--yolo-size`: YOLOv8 model size (`n`, `s`, `m`, `l`, `x`)
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--device`: Training device (`0`, `0,1,2,3`, or `cpu`)

### 3. Model Inference

Run inference on new images:

```bash
python infer.py --model yolo --weights models/yolov8/weights/model_best.pth --input data/test_images
```

## Dependencies

- PyTorch >= 1.9.0
- Torchvision >= 0.10.0
- Ultralytics (YOLOv8) >= 8.0.0
- TensorFlow >= 2.8.0
- TensorFlow Object Detection API >= 0.1.1
- Albumentations >= 1.0.0
- OpenCV >= 4.5.0
- NumPy >= 1.20.0
- Pillow >= 8.0.0
- PyYAML >= 6.0
- pycocotools >= 2.0.0
- TensorBoard >= 2.8.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Options:
- `--data-dir`: Directory with YOLOv8 format data (default: `data/processed/yolo`)
- `--runs-dir`: Directory for training runs and logs (default: `data/runs/yolov8`)
- `--weights-dir`: Directory for saved model weights (default: `models/yolov8/weights`)

## Requirements

This project requires the following dependencies:
- PyTorch
- Torchvision
- Ultralytics (YOLOv8)
- Albumentations
- OpenCV
- NumPy
- Pillow

Install all dependencies with:
```bash
pip install -r requirements.txt
``` 