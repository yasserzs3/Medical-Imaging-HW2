import os
import argparse
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torch.utils.data import DataLoader
import sys
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))
from etl.load import CocoDetectionMaskRCNNDataset

def collate_fn(batch):
    """
    Custom collate function for the dataloader.
    
    Args:
        batch (list): Batch of data
    
    Returns:
        tuple: (images, targets)
    """
    return tuple(zip(*batch))

def get_transform(train):
    """
    Get transforms for training or validation.
    
    Args:
        train (bool): Whether to use training transforms
    
    Returns:
        torchvision.transforms.Compose: Transforms
    """
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())
    if train:
        transforms.append(torchvision.transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.04, hue=0
        ))
    return torchvision.transforms.Compose(transforms)

def get_model(num_classes, pretrained=True):
    """
    Get a Mask R-CNN model.
    
    Args:
        num_classes (int): Number of classes (including background)
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: Mask R-CNN model
    """
    # Load a pre-trained model
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)
    
    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    
    # Replace the mask predictor with a new one for our number of classes
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.
    
    Args:
        box1 (torch.Tensor): First box [x1, y1, x2, y2]
        box2 (torch.Tensor): Second box [x1, y1, x2, y2]
    
    Returns:
        float: IoU value
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def calculate_mask_iou(mask1, mask2):
    """
    Calculate IoU between two masks.
    
    Args:
        mask1 (torch.Tensor): First mask
        mask2 (torch.Tensor): Second mask
    
    Returns:
        float: IoU value
    """
    intersection = (mask1 & mask2).sum().float()
    union = (mask1 | mask2).sum().float()
    return intersection / union if union > 0 else 0

def evaluate_metrics(model, data_loader, device):
    """
    Evaluate model metrics.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (torch.utils.data.DataLoader): Data loader
        device (torch.device): Device to evaluate on
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    total_box_iou = 0
    total_mask_iou = 0
    total_class_acc = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Get predictions
            outputs = model(images)
            
            # Calculate metrics for each image
            for output, target in zip(outputs, targets):
                # Box IoU
                pred_boxes = output['boxes']
                target_boxes = target['boxes']
                box_ious = []
                for pred_box in pred_boxes:
                    for target_box in target_boxes:
                        box_ious.append(calculate_iou(pred_box, target_box))
                if box_ious:
                    total_box_iou += max(box_ious)
                
                # Mask IoU
                if 'masks' in output and 'masks' in target:
                    pred_masks = output['masks'] > 0.5
                    target_masks = target['masks']
                    mask_ious = []
                    for pred_mask in pred_masks:
                        for target_mask in target_masks:
                            mask_ious.append(calculate_mask_iou(pred_mask, target_mask))
                    if mask_ious:
                        total_mask_iou += max(mask_ious)
                
                # Classification accuracy
                pred_labels = output['labels']
                target_labels = target['labels']
                correct = (pred_labels == target_labels).sum().item()
                total_class_acc += correct / len(target_labels)
                
                total_samples += 1
    
    # Calculate averages
    metrics = {
        'box_iou': total_box_iou / total_samples if total_samples > 0 else 0,
        'mask_iou': total_mask_iou / total_samples if total_samples > 0 else 0,
        'class_acc': total_class_acc / total_samples if total_samples > 0 else 0
    }
    
    return metrics

def train_one_epoch(model, optimizer, data_loader, device, epoch, writer, print_freq=10):
    """
    Train for one epoch.
    
    Args:
        model (torch.nn.Module): Model to train
        optimizer (torch.optim.Optimizer): Optimizer
        data_loader (torch.utils.data.DataLoader): Data loader
        device (torch.device): Device to train on
        epoch (int): Current epoch
        writer (SummaryWriter): TensorBoard writer
        print_freq (int): How often to print progress
    """
    model.train()
    header = f'Epoch: [{epoch}]'
    
    # Initialize metrics
    total_loss = 0
    total_box_loss = 0
    total_mask_loss = 0
    total_class_loss = 0
    total_objectness_loss = 0
    total_rpn_box_loss = 0
    
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Reduce losses over all GPUs for logging purposes
        loss_value = losses.item()
        
        if not torch.isfinite(torch.tensor(loss_value)):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss_value
        total_box_loss += loss_dict['loss_box_reg'].item()
        total_mask_loss += loss_dict['loss_mask'].item()
        total_class_loss += loss_dict['loss_classifier'].item()
        total_objectness_loss += loss_dict['loss_objectness'].item()
        total_rpn_box_loss += loss_dict['loss_rpn_box_reg'].item()
        
        # Log metrics
        if i % print_freq == 0:
            # Calculate averages
            avg_loss = total_loss / (i + 1)
            avg_box_loss = total_box_loss / (i + 1)
            avg_mask_loss = total_mask_loss / (i + 1)
            avg_class_loss = total_class_loss / (i + 1)
            avg_objectness_loss = total_objectness_loss / (i + 1)
            avg_rpn_box_loss = total_rpn_box_loss / (i + 1)
            
            # Print metrics
            print(f"{header} [{i}/{len(data_loader)}]")
            print(f"  Total Loss: {avg_loss:.4f}")
            print(f"  Box Loss: {avg_box_loss:.4f}")
            print(f"  Mask Loss: {avg_mask_loss:.4f}")
            print(f"  Class Loss: {avg_class_loss:.4f}")
            print(f"  Objectness Loss: {avg_objectness_loss:.4f}")
            print(f"  RPN Box Loss: {avg_rpn_box_loss:.4f}")
            
            # Log to TensorBoard
            writer.add_scalar('train/total_loss', avg_loss, epoch * len(data_loader) + i)
            writer.add_scalar('train/box_loss', avg_box_loss, epoch * len(data_loader) + i)
            writer.add_scalar('train/mask_loss', avg_mask_loss, epoch * len(data_loader) + i)
            writer.add_scalar('train/class_loss', avg_class_loss, epoch * len(data_loader) + i)
            writer.add_scalar('train/objectness_loss', avg_objectness_loss, epoch * len(data_loader) + i)
            writer.add_scalar('train/rpn_box_loss', avg_rpn_box_loss, epoch * len(data_loader) + i)
    
    # Return average metrics for the epoch
    return {
        'total_loss': total_loss / len(data_loader),
        'box_loss': total_box_loss / len(data_loader),
        'mask_loss': total_mask_loss / len(data_loader),
        'class_loss': total_class_loss / len(data_loader),
        'objectness_loss': total_objectness_loss / len(data_loader),
        'rpn_box_loss': total_rpn_box_loss / len(data_loader)
    }

def main():
    parser = argparse.ArgumentParser(description='Train Mask R-CNN model')
    parser.add_argument('--data-dir', type=str, default='data/processed/maskrcnn',
                        help='Directory containing the dataset')
    parser.add_argument('--output-dir', type=str, default='models/r-cnn/outputs',
                        help='Directory to save model outputs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay for SGD')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='How often to print progress')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=output_dir / 'logs')
    
    # Create datasets
    train_dataset = CocoDetectionMaskRCNNDataset(
        root=os.path.join(args.data_dir, 'train'),
        annFile=os.path.join(args.data_dir, 'train', '_annotations.coco.json'),
        transform=get_transform(train=True)
    )
    
    valid_dataset = CocoDetectionMaskRCNNDataset(
        root=os.path.join(args.data_dir, 'valid'),
        annFile=os.path.join(args.data_dir, 'valid', '_annotations.coco.json'),
        transform=get_transform(train=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Get the model
    device = torch.device(args.device)
    model = get_model(num_classes=train_dataset.num_classes + 1)  # +1 for background
    model.to(device)
    
    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    
    # Print training configuration
    print("\nTraining Configuration:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Number of Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Device: {args.device}")
    print(f"  Number of Classes: {train_dataset.num_classes + 1}")
    print(f"  Training Images: {len(train_dataset)}")
    print(f"  Validation Images: {len(valid_dataset)}")
    print(f"  Iterations per Epoch: {len(train_loader)}\n")
    
    # Train the model
    for epoch in range(args.epochs):
        # Train for one epoch
        metrics = train_one_epoch(model, optimizer, train_loader, device, epoch, writer, args.print_freq)
        
        # Evaluate on validation set
        val_metrics = evaluate_metrics(model, valid_loader, device)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Training Metrics:")
        print(f"    Total Loss: {metrics['total_loss']:.4f}")
        print(f"    Box Loss: {metrics['box_loss']:.4f}")
        print(f"    Mask Loss: {metrics['mask_loss']:.4f}")
        print(f"    Class Loss: {metrics['class_loss']:.4f}")
        print(f"    Objectness Loss: {metrics['objectness_loss']:.4f}")
        print(f"    RPN Box Loss: {metrics['rpn_box_loss']:.4f}")
        print(f"  Validation Metrics:")
        print(f"    Box IoU: {val_metrics['box_iou']:.4f}")
        print(f"    Mask IoU: {val_metrics['mask_iou']:.4f}")
        print(f"    Classification Accuracy: {val_metrics['class_acc']:.4f}\n")
        
        # Log validation metrics to TensorBoard
        writer.add_scalar('val/box_iou', val_metrics['box_iou'], epoch)
        writer.add_scalar('val/mask_iou', val_metrics['mask_iou'], epoch)
        writer.add_scalar('val/class_acc', val_metrics['class_acc'], epoch)
        
        # Update the learning rate
        lr_scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'val_metrics': val_metrics
        }
        torch.save(checkpoint, output_dir / f'checkpoint_{epoch}.pth')
        
        # Save the final model
        if epoch == args.epochs - 1:
            torch.save(model.state_dict(), output_dir / 'model_final.pth')
    
    writer.close()

if __name__ == '__main__':
    main() 