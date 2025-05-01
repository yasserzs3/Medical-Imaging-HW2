import os
import argparse
from pathlib import Path
import time
import yaml
import torch
import sys
from models.faster_rcnn.train import main as train_faster_rcnn

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

# Import model-specific training functions
from models.yolov8.yolo_train import train_yolov8
from models.faster_rcnn.train import (
    get_model,
    train_one_epoch,
    evaluate,
    get_transform,
    COCODataset
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def train_faster_rcnn_model(
    data_dir='data/processed/faster_rcnn',
    runs_dir='data/runs/faster_rcnn',
    weights_dir='models/faster_rcnn/weights',
    epochs=20,
    batch_size=4,
    device='0',  # Default to first GPU
    verbose=True,
    lr=0.0005,
    momentum=0.9,
    weight_decay=0.0005
):
    """
    Train Faster R-CNN model on the prepared dataset.
    
    Args:
        data_dir (str): Directory containing the dataset
        runs_dir (str): Directory for training runs and logs
        weights_dir (str): Directory to save model weights
        epochs (int): Number of epochs to train for
        batch_size (int): Batch size for training
        device (str): Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)
        verbose (bool): Whether to print verbose output
        lr (float): Learning rate
        momentum (float): Momentum for SGD
        weight_decay (float): Weight decay for SGD
    
    Returns:
        dict: Training metrics
    """
    start_time = time.time()
    
    # Create directories
    runs_dir = Path(runs_dir)
    weights_dir = Path(weights_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up device
    if device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{device}')
    
    if verbose:
        print("\n=== STARTING FASTER R-CNN TRAINING ===\n")
        print(f"Using device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=runs_dir / 'logs')
    
    # Create datasets
    train_dataset = COCODataset(
        root=os.path.join(data_dir, 'train'),
        annotation=os.path.join(data_dir, 'train', '_annotations.coco.json'),
        transforms=get_transform()
    )
    
    valid_dataset = COCODataset(
        root=os.path.join(data_dir, 'valid'),
        annotation=os.path.join(data_dir, 'valid', '_annotations.coco.json'),
        transforms=get_transform()
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for Windows compatibility
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 for Windows compatibility
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Get the model
    model = get_model(num_classes=train_dataset.coco.getCatIds() + 1)  # +1 for background
    model.to(device)
    
    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    
    # Print training configuration
    if verbose:
        print("\nTraining Configuration:")
        print(f"  Batch Size: {batch_size}")
        print(f"  Number of Epochs: {epochs}")
        print(f"  Learning Rate: {lr}")
        print(f"  Device: {device}")
        print(f"  Number of Classes: {len(train_dataset.coco.getCatIds()) + 1}")
        print(f"  Training Images: {len(train_dataset)}")
        print(f"  Validation Images: {len(valid_dataset)}")
        print(f"  Iterations per Epoch: {len(train_loader)}\n")
    
    # Train the model
    best_val_loss = float('inf')
    metrics_history = []
    
    for epoch in range(epochs):
        # Train for one epoch
        train_loss = train_one_epoch(model, optimizer, train_loader, device, None)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, valid_loader, device)
        
        # Print epoch summary
        if verbose:
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Training Loss: {train_loss:.4f}")
            print(f"  Validation Metrics:")
            print(f"    Loss: {val_metrics['loss']:.4f}")
            print(f"    Precision: {val_metrics['precision']:.4f}")
            print(f"    Recall: {val_metrics['recall']:.4f}")
            print(f"    F1-Score: {val_metrics['f1']:.4f}\n")
        
        # Log validation metrics to TensorBoard
        writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        writer.add_scalar('val/precision', val_metrics['precision'], epoch)
        writer.add_scalar('val/recall', val_metrics['recall'], epoch)
        writer.add_scalar('val/f1', val_metrics['f1'], epoch)
        
        # Update the learning rate
        lr_scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_metrics': val_metrics
        }
        torch.save(checkpoint, weights_dir / f'checkpoint_{epoch}.pth')
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), weights_dir / 'model_best.pth')
        
        # Save metrics history
        metrics_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_metrics': val_metrics
        })
    
    # Save final model
    torch.save(model.state_dict(), weights_dir / 'model_final.pth')
    
    # Save metrics history
    with open(runs_dir / 'metrics_history.yaml', 'w') as f:
        yaml.safe_dump(metrics_history, f)
    
    writer.close()
    
    total_time = time.time() - start_time
    if verbose:
        print(f"\n=== FASTER R-CNN TRAINING COMPLETED IN {total_time:.2f} SECONDS ===\n")
    
    return {
        'status': 'success',
        'metrics_history': metrics_history,
        'best_val_loss': best_val_loss,
        'total_time': total_time
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
        models (list): List of models to train (yolo, faster_rcnn, ssd)
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
        'faster_rcnn': 20,
        'ssd': 25
    }
    
    # Default batch sizes for each model type
    default_batch_size = {
        'yolo': 16,
        'faster_rcnn': 4,
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
            
        elif model == 'faster_rcnn':
            if verbose:
                print(f"\n=== TRAINING FASTER R-CNN MODEL ===\n")
            
            faster_rcnn_metrics = train_faster_rcnn_model(
                data_dir=f"{data_dir}/faster_rcnn",
                runs_dir=f"{runs_dir}/faster_rcnn",
                weights_dir=f"{models_dir}/faster_rcnn/weights",
                epochs=epochs.get('faster_rcnn', default_epochs['faster_rcnn']),
                batch_size=batch_size.get('faster_rcnn', default_batch_size['faster_rcnn']),
                device=device,
                verbose=verbose
            )
            
            metrics['faster_rcnn'] = faster_rcnn_metrics
            
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
    parser.add_argument('--model', type=str, choices=['yolo', 'faster_rcnn', 'ssd'], action='append', 
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