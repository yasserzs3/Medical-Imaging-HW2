import os
import argparse
from pathlib import Path
import time
import yaml
import torch
import shutil
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

def train_yolov8(
    data_dir='data/processed/yolo',
    runs_dir='data/runs/yolov8',
    weights_dir='models/yolov8/weights',
    model_size='n',  # nano
    epochs=50,
    batch_size=16,
    imgsz=640,
    device='0',  # Default to first GPU
    pretrained=True,
    resume=False,
    project=None,
    name=None,
    verbose=True
):
    """
    Train YOLOv8 model on the prepared dataset.
    
    Args:
        data_dir (str): Directory with data in YOLOv8 format
        runs_dir (str): Directory to save training runs and logs
        weights_dir (str): Directory to save best model weights
        model_size (str): YOLOv8 model size (n, s, m, l, x)
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        imgsz (int): Image size for training
        device (str): Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)
        pretrained (bool): Whether to use pretrained weights
        resume (bool): Resume training from last checkpoint
        project (str): Project name
        name (str): Experiment name
        verbose (bool): Whether to print verbose output
        
    Returns:
        dict: Training metrics
    """
    start_time = time.time()
    
    # Create output directories if they don't exist
    Path(runs_dir).mkdir(exist_ok=True, parents=True)
    Path(weights_dir).mkdir(exist_ok=True, parents=True)
    
    # Check if data.yaml exists
    data_yaml = Path(data_dir) / 'dataset.yaml'
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found at {data_yaml}")
    
    # Load dataset configuration
    with open(data_yaml, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Set up model and training configuration
    model_name = f"yolov8{model_size}.pt"  # e.g., yolov8n.pt
    
    # Set up YOLOv8 model with weights directory
    if pretrained:
        # Set the directory for downloading weights
        os.environ['YOLO_CONFIG_DIR'] = str(Path(weights_dir).resolve())
        
        # Download pre-trained weights to the weights directory
        pretrained_path = Path(weights_dir) / model_name
        if not pretrained_path.exists():
            if verbose:
                print(f"Downloading pre-trained weights to {pretrained_path}")
            # Download weights and save to weights directory
            downloaded_path = attempt_download_asset(model_name)
            if downloaded_path != str(pretrained_path):
                if verbose:
                    print(f"Moving weights from {downloaded_path} to {pretrained_path}")
                shutil.move(downloaded_path, pretrained_path)
                if verbose:
                    print(f"Moved weights to {pretrained_path}")
        
        # Load model from the weights directory
        model = YOLO(pretrained_path)
        if verbose:
            print(f"Loaded pre-trained model from {pretrained_path}")
    else:
        # Initialize a new model
        model = YOLO(model_name)
    
    # Define training arguments
    exp_name = name or f'yolov8_{model_size}'
    
    train_args = {
        'data': str(data_yaml),
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'save': True,
        'cache': True,
        'patience': 20,  # Early stopping patience
        'project': project or runs_dir,
        'name': exp_name,
        'exist_ok': True,
        'device': device  # Explicitly set device
    }
    
    if resume:
        # Find latest weights file in the weights directory
        best_weights = Path(weights_dir) / f'{exp_name}_best.pt'
        if best_weights.exists():
            train_args['resume'] = str(best_weights)
            if verbose:
                print(f"Resuming training from {best_weights}")
    
    if verbose:
        print("\n=== STARTING YOLOV8 TRAINING ===\n")
        print(f"Dataset: {data_yaml}")
        print(f"Model: YOLOv8-{model_size}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Image Size: {imgsz}")
        print(f"Device: {device}")
        print(f"Pretrained: {pretrained}")
        print(f"Runs Directory: {runs_dir}")
        print(f"Weights Directory: {weights_dir}")
    
    # Train the model
    results = model.train(**train_args)
    
    # Get training metrics
    metrics = {
        'training_time': time.time() - start_time,
        'final_mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
        'final_mAP50_95': results.results_dict.get('metrics/mAP50-95(B)', 0),
        'best_epoch': results.results_dict.get('metrics/best_epoch', 0),
        'final_precision': results.results_dict.get('metrics/precision(B)', 0),
        'final_recall': results.results_dict.get('metrics/recall(B)', 0)
    }
    
    # Copy the best weights to the weights directory
    run_dir = Path(runs_dir) / exp_name
    best_weights_path = run_dir / 'weights' / 'best.pt'
    
    if best_weights_path.exists():
        target_weights_path = Path(weights_dir) / f'{exp_name}_best.pt'
        shutil.copy2(best_weights_path, target_weights_path)
        if verbose:
            print(f"\nBest weights saved to: {target_weights_path}")
    
    if verbose:
        print("\n=== TRAINING COMPLETE ===\n")
        print(f"Training time: {metrics['training_time']:.2f} seconds")
        print(f"Best epoch: {metrics['best_epoch']}")
        print(f"mAP50: {metrics['final_mAP50']:.4f}")
        print(f"mAP50-95: {metrics['final_mAP50_95']:.4f}")
        print(f"Precision: {metrics['final_precision']:.4f}")
        print(f"Recall: {metrics['final_recall']:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model')
    parser.add_argument('--data-dir', type=str, default='data/processed/yolo', help='Directory with YOLOv8 format data')
    parser.add_argument('--runs-dir', type=str, default='data/runs/yolov8', help='Directory to save training runs and logs')
    parser.add_argument('--weights-dir', type=str, default='models/yolov8/weights', help='Directory to save best model weights')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--device', type=str, default='0', help='Device to use (e.g., 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--no-pretrained', action='store_true', help='Do not use pretrained weights')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--project', type=str, help='Project name')
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    train_yolov8(
        data_dir=args.data_dir,
        runs_dir=args.runs_dir,
        weights_dir=args.weights_dir,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        pretrained=not args.no_pretrained,
        resume=args.resume,
        project=args.project,
        name=args.name,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main() 