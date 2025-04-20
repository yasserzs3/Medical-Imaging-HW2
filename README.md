# Medical Imaging Object Detection Project

This project provides a complete pipeline for processing medical imaging data and training object detection models on it.

## Project Structure

- `etl.py` - Extract, Transform, Load pipeline for processing raw medical images into model-ready datasets
- `train.py` - Multi-model training script that supports training YOLOv8, Mask R-CNN, and SSD models
- `models/` - Model-specific implementations and trained model weights
  - `models/yolov8/` - YOLOv8 implementation
    - `models/yolov8/yolo_train.py` - YOLOv8 training script
    - `models/yolov8/weights/` - Saved YOLOv8 model weights
  - `models/r-cnn/` - Mask R-CNN implementation (placeholder for future development)
  - `models/ssd/` - SSD implementation (placeholder for future development)
- `data/` - Data directories
  - `data/raw/` - Raw input data in COCO format
  - `data/processed/` - Processed datasets ready for model training
  - `data/runs/` - Training run logs and intermediate results
- `etl/` - ETL modules
  - `etl/extract.py` - Data extraction functionality
  - `etl/transform.py` - Data transformation and augmentation
  - `etl/load.py` - Data loading for training

## Usage

### Data Preparation (ETL)

Prepare your data for model training:

```bash
python etl.py --data-dir data/raw --output-dir data/processed --model yolo --apply-augmentation
```

Options:
- `--model`: Model format to generate data for (`yolo`, `maskrcnn`, `ssd`)
- `--apply-augmentation`: Apply data augmentation techniques
- `--apply-normalization`: Apply normalization to images

### Training Models

Train object detection models:

```bash
python train.py --model yolo --epochs 50 --batch-size 16
```

Options:
- `--model`: Model to train (`yolo`, `maskrcnn`, `ssd`)
- `--runs-dir`: Directory for training runs and logs (default: `data/runs`)
- `--models-dir`: Directory for model weights (default: `models`)
- `--yolo-size`: YOLOv8 model size (`n`, `s`, `m`, `l`, `x`)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--device`: Device to use (e.g., `0` or `0,1,2,3` or `cpu`)

### Training YOLOv8 Only

For a more focused YOLOv8 training:

```bash
python models/yolov8/yolo_train.py --model-size n --epochs 50 --batch-size 16
```

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