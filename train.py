import os
import argparse
from pathlib import Path
import time
import yaml
import torch

# Import model-specific training functions
from models.yolov8.yolo_train import train_yolov8

def train_maskrcnn(
    data_dir='data/processed/maskrcnn',
    runs_dir='data/runs/maskrcnn',
    weights_dir='models/maskrcnn/weights',
    epochs=20,
    batch_size=4,
    device='0',  # Default to first GPU
    verbose=True
):
    """
    Train Mask R-CNN model on the prepared dataset.
    
    Note: This is a placeholder for future implementation.
    """
    if verbose:
        print("\n=== MASK R-CNN TRAINING NOT YET IMPLEMENTED ===\n")
    
    # Placeholder to simulate training
    time.sleep(2)
    
    return {
        'status': 'not implemented',
        'message': 'Mask R-CNN training not yet implemented'
    }


def train_ssd(
    data_dir='data/processed/ssd',
    runs_dir='data/runs/ssd',
    weights_dir='models/ssd/weights',
    epochs=25,
    batch_size=8,
    device='0',  # Default to first GPU
    verbose=True
):
    """
    Train SSD model on the prepared dataset.
    
    Note: This is a placeholder for future implementation.
    """
    if verbose:
        print("\n=== SSD TRAINING NOT YET IMPLEMENTED ===\n")
    
    # Placeholder to simulate training
    time.sleep(2)
    
    return {
        'status': 'not implemented',
        'message': 'SSD training not yet implemented'
    }


def train_models(
    data_dir='data/processed',
    runs_dir='data/runs',
    models_dir='models',
    models=None,
    yolo_size='n',
    epochs=None,
    batch_size=None,
    device='0',  # Default to first GPU
    verbose=True
):
    """
    Train multiple object detection models.
    
    Args:
        data_dir (str): Base data directory with processed data
        runs_dir (str): Base directory for training runs and logs
        models_dir (str): Base directory for model weights
        models (list): List of models to train (yolo, maskrcnn, ssd)
        yolo_size (str): YOLOv8 model size (n, s, m, l, x)
        epochs (dict): Dictionary of epochs for each model type
        batch_size (dict): Dictionary of batch sizes for each model type
        device (str): Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)
        verbose (bool): Whether to print verbose output
        
    Returns:
        dict: Dictionary of training metrics for each model
    """
    start_time = time.time()
    
    # Default to empty list if None
    if models is None:
        models = []
    
    # Default epochs for each model type
    default_epochs = {
        'yolo': 50,
        'maskrcnn': 20,
        'ssd': 25
    }
    
    # Default batch sizes for each model type
    default_batch_size = {
        'yolo': 16,
        'maskrcnn': 4,
        'ssd': 8
    }
    
    # Use provided epochs or defaults
    epochs = epochs or default_epochs
    batch_size = batch_size or default_batch_size
    
    # Dictionary to store metrics for each model
    metrics = {}
    
    if verbose:
        print("\n=== STARTING MULTI-MODEL TRAINING ===\n")
        print(f"Models to train: {', '.join(models) if models else 'None'}")
        print(f"Using device: {device}")
    
    # Train each model
    for model in models:
        if model == 'yolo':
            if verbose:
                print(f"\n=== TRAINING YOLOV8 MODEL ===\n")
            
            yolo_metrics = train_yolov8(
                data_dir=f"{data_dir}/yolo",
                runs_dir=f"{runs_dir}/yolov8",
                weights_dir=f"{models_dir}/yolov8/weights",
                model_size=yolo_size,
                epochs=epochs.get('yolo', default_epochs['yolo']),
                batch_size=batch_size.get('yolo', default_batch_size['yolo']),
                device=device,
                verbose=verbose
            )
            
            metrics['yolo'] = yolo_metrics
            
        elif model == 'maskrcnn':
            if verbose:
                print(f"\n=== TRAINING MASK R-CNN MODEL ===\n")
            
            maskrcnn_metrics = train_maskrcnn(
                data_dir=f"{data_dir}/maskrcnn",
                runs_dir=f"{runs_dir}/maskrcnn",
                weights_dir=f"{models_dir}/maskrcnn/weights",
                epochs=epochs.get('maskrcnn', default_epochs['maskrcnn']),
                batch_size=batch_size.get('maskrcnn', default_batch_size['maskrcnn']),
                device=device,
                verbose=verbose
            )
            
            metrics['maskrcnn'] = maskrcnn_metrics
            
        elif model == 'ssd':
            if verbose:
                print(f"\n=== TRAINING SSD MODEL ===\n")
            
            ssd_metrics = train_ssd(
                data_dir=f"{data_dir}/ssd",
                runs_dir=f"{runs_dir}/ssd",
                weights_dir=f"{models_dir}/ssd/weights",
                epochs=epochs.get('ssd', default_epochs['ssd']),
                batch_size=batch_size.get('ssd', default_batch_size['ssd']),
                device=device,
                verbose=verbose
            )
            
            metrics['ssd'] = ssd_metrics
    
    total_time = time.time() - start_time
    if verbose:
        print(f"\n=== MULTI-MODEL TRAINING COMPLETED IN {total_time:.2f} SECONDS ===\n")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train object detection models')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Base data directory')
    parser.add_argument('--runs-dir', type=str, default='data/runs', help='Base directory for training runs and logs')
    parser.add_argument('--models-dir', type=str, default='models', help='Base directory for model weights')
    parser.add_argument('--model', type=str, choices=['yolo', 'maskrcnn', 'ssd'], action='append', 
                        help='Model to train (can be specified multiple times)')
    parser.add_argument('--yolo-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--device', type=str, default='0', help='Device to use (e.g., 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Default to yolo if no model selected
    models = args.model if args.model else ['yolo']
    
    # Set up epochs dictionary
    epochs = {}
    if args.epochs:
        for model in models:
            epochs[model] = args.epochs
    
    # Set up batch size dictionary
    batch_size = {}
    if args.batch_size:
        for model in models:
            batch_size[model] = args.batch_size
    
    train_models(
        data_dir=args.data_dir,
        runs_dir=args.runs_dir,
        models_dir=args.models_dir,
        models=models,
        yolo_size=args.yolo_size,
        epochs=epochs or None,
        batch_size=batch_size or None,
        device=args.device,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main() 