import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import sys
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))
from etl.load import CocoDetectionMaskRCNNDataset

class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load ResNet-18
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
        # Get the layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x1 = self.layer1(x)   # 64 channels
        x2 = self.layer2(x1)  # 128 channels
        x3 = self.layer3(x2)  # 256 channels
        x4 = self.layer4(x3)  # 512 channels
        
        return [x1, x2, x3, x4]

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        # ResNet-18 feature channels: [64, 128, 256, 512]
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1)
            for in_channels in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])
        
    def forward(self, features):
        # features should be a list of feature maps from different levels
        laterals = [conv(feature) for feature, conv in zip(features, self.lateral_convs)]
        
        # Build top-down path
        for i in range(len(laterals)-1, 0, -1):
            laterals[i-1] += F.interpolate(laterals[i], size=laterals[i-1].shape[-2:])
            
        # Apply final convs
        return [conv(lateral) for lateral, conv in zip(laterals, self.fpn_convs)]

class RPN(nn.Module):
    def __init__(self, in_channels, num_anchors=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_head = nn.Conv2d(in_channels, num_anchors * 2, 1)  # 2 for object/not object
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, 1)  # 4 for box regression
        
    def forward(self, x):
        x = F.relu(self.conv(x))
        cls = self.cls_head(x)
        reg = self.reg_head(x)
        
        # Reshape outputs
        batch_size = x.size(0)
        cls = cls.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        reg = reg.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        
        return cls, reg

class RoIHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Calculate the size after ROI pooling
        self.roi_size = 7
        self.roi_pool = nn.AdaptiveAvgPool2d((self.roi_size, self.roi_size))
        
        # Calculate the size of flattened features
        self.fc_input_size = in_channels * self.roi_size * self.roi_size
        
        # Classification and regression heads
        self.fc1 = nn.Linear(self.fc_input_size, 512)  # Reduced from 1024
        self.fc2 = nn.Linear(512, 512)  # Reduced from 1024
        self.cls_head = nn.Linear(512, num_classes)
        self.reg_head = nn.Linear(512, num_classes * 4)
        
        # Mask head
        self.mask_conv1 = nn.Conv2d(in_channels, 128, 3, padding=1)  # Reduced from 256
        self.mask_conv2 = nn.Conv2d(128, 128, 3, padding=1)  # Reduced from 256
        self.mask_conv3 = nn.Conv2d(128, 128, 3, padding=1)  # Reduced from 256
        self.mask_deconv = nn.ConvTranspose2d(128, 128, 2, 2)  # Reduced from 256
        self.mask_final = nn.Conv2d(128, num_classes, 1)
        
    def forward(self, x):
        # Apply ROI pooling
        x = self.roi_pool(x)
        
        # Save a copy for mask head
        mask_features = x
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Get classification and regression predictions
        cls = self.cls_head(x)
        reg = self.reg_head(x)
        
        # Get mask predictions
        x = mask_features
        x = F.relu(self.mask_conv1(x))
        x = F.relu(self.mask_conv2(x))
        x = F.relu(self.mask_conv3(x))
        x = F.relu(self.mask_deconv(x))
        masks = self.mask_final(x)
        
        return cls, reg, masks

class LightMaskRCNN(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained)
        # ResNet-18 feature channels: [64, 128, 256, 512]
        self.fpn = FPN([64, 128, 256, 512], 64)  # Reduced from 128
        self.rpn = RPN(64)  # Reduced from 128
        self.roi_head = RoIHead(64, num_classes)  # Reduced from 128
        
    def forward(self, images, targets=None):
        # Handle batch of images
        if isinstance(images, list):
            # Stack images into a single tensor
            images = torch.stack(images)
        
        # Extract features
        features = self.backbone(images)
        fpn_features = self.fpn(features)
        
        # RPN
        rpn_cls, rpn_reg = self.rpn(fpn_features[0])
        
        if self.training and targets is not None:
            # Training mode
            # Compute RPN losses
            batch_size = images.size(0)
            rpn_cls_targets = torch.zeros(batch_size, rpn_cls.size(1), dtype=torch.long, device=images.device)
            rpn_reg_targets = torch.zeros_like(rpn_reg)
            
            rpn_cls_loss = F.cross_entropy(rpn_cls.view(-1, 2), rpn_cls_targets.view(-1))
            rpn_reg_loss = F.smooth_l1_loss(rpn_reg, rpn_reg_targets)
            
            # Compute ROI losses
            roi_cls, roi_reg, roi_masks = self.roi_head(fpn_features[0])
            roi_cls_targets = torch.zeros(batch_size, dtype=torch.long, device=images.device)
            roi_reg_targets = torch.zeros_like(roi_reg)
            roi_mask_targets = torch.zeros_like(roi_masks)
            
            roi_cls_loss = F.cross_entropy(roi_cls, roi_cls_targets)
            roi_reg_loss = F.smooth_l1_loss(roi_reg, roi_reg_targets)
            roi_mask_loss = F.binary_cross_entropy_with_logits(roi_masks, roi_mask_targets)
            
            # Combine losses
            return {
                'loss_rpn_cls': rpn_cls_loss,
                'loss_rpn_reg': rpn_reg_loss,
                'loss_cls': roi_cls_loss,
                'loss_reg': roi_reg_loss,
                'loss_mask': roi_mask_loss
            }
        else:
            # Inference mode
            # Get objectness scores and box predictions from RPN
            rpn_cls_scores = F.softmax(rpn_cls, dim=2)[:, :, 1]  # Objectness scores
            rpn_boxes = rpn_reg.view(-1, 4)  # Box predictions
            
            # Apply NMS to get top proposals
            keep = nms(rpn_boxes, rpn_cls_scores.view(-1), 0.7)
            proposals = rpn_boxes[keep]
            scores = rpn_cls_scores.view(-1)[keep]
            
            # Get ROI predictions
            roi_cls, roi_reg, roi_masks = self.roi_head(fpn_features[0])
            roi_cls_scores = F.softmax(roi_cls, dim=1)
            roi_boxes = roi_reg.view(-1, 4)
            
            # Apply NMS to get final detections
            keep = nms(roi_boxes, roi_cls_scores.max(dim=1)[0], 0.5)
            boxes = roi_boxes[keep]
            labels = roi_cls_scores[keep].argmax(dim=1)
            scores = roi_cls_scores[keep].max(dim=1)[0]
            masks = roi_masks[keep]
            
            return {
                'boxes': boxes,
                'labels': labels,
                'scores': scores,
                'masks': masks
            }

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
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else torch.tensor(0.0, device=box1.device)

def nms(boxes, scores, iou_threshold):
    """
    Non-maximum suppression.
    
    Args:
        boxes (torch.Tensor): Boxes to perform NMS on
        scores (torch.Tensor): Scores for each box
        iou_threshold (float): IoU threshold for NMS
    
    Returns:
        torch.Tensor: Indices of boxes to keep
    """
    # Sort boxes by score
    _, order = scores.sort(0, descending=True)
    keep = []
    
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        i = order[0].item()
        keep.append(i)
        
        # Calculate IoU with remaining boxes
        ious = torch.tensor([calculate_iou(boxes[i], boxes[j]) for j in order[1:]], device=boxes.device)
        
        # Keep boxes with IoU less than threshold
        idx = (ious <= iou_threshold).nonzero().squeeze()
        if idx.numel() == 0:
            break
        order = order[idx + 1]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def get_model(num_classes, pretrained=True):
    """
    Get a lighter version of Mask R-CNN model.
    
    Args:
        num_classes (int): Number of classes (including background)
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: Lighter Mask R-CNN model
    """
    return LightMaskRCNN(num_classes, pretrained)

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
    
    print(f"\nStarting epoch {epoch}")
    print(f"Total batches: {len(data_loader)}")
    
    for i, (images, targets) in enumerate(data_loader):
        try:
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Reduce losses over all GPUs for logging purposes
            loss_value = losses.detach().item()
            
            if not torch.isfinite(torch.tensor(loss_value)):
                print(f"Warning: Loss is {loss_value}, skipping batch {i}")
                print(f"Loss dict: {loss_dict}")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss_value
            total_box_loss += loss_dict['loss_rpn_reg'].detach().item()
            total_mask_loss += loss_dict['loss_mask'].detach().item()
            total_class_loss += loss_dict['loss_cls'].detach().item()
            total_objectness_loss += loss_dict['loss_rpn_cls'].detach().item()
            total_rpn_box_loss += loss_dict['loss_rpn_reg'].detach().item()
            
            # Print progress more frequently
            if i % print_freq == 0 or i == len(data_loader) - 1:
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
        
        except Exception as e:
            print(f"Error in batch {i}: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            continue
    
    print(f"\nCompleted epoch {epoch}")
    
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
    start_time = time.time()
    print("\n=== TRAINING STARTED ===")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n{'='*50}")
        print(f"Starting Epoch {epoch+1}/{args.epochs}")
        print(f"Epoch start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Train for one epoch
        metrics = train_one_epoch(model, optimizer, train_loader, device, epoch, writer, args.print_freq)
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        val_metrics = evaluate_metrics(model, valid_loader, device)
        
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        total_duration = time.time() - start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Duration: {epoch_duration:.2f} seconds")
        print(f"  Total training time: {total_duration:.2f} seconds")
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
        print(f"    Classification Accuracy: {val_metrics['class_acc']:.4f}")
        
        # Log validation metrics to TensorBoard
        writer.add_scalar('val/box_iou', val_metrics['box_iou'], epoch)
        writer.add_scalar('val/mask_iou', val_metrics['mask_iou'], epoch)
        writer.add_scalar('val/class_acc', val_metrics['class_acc'], epoch)
        
        # Update the learning rate
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Current learning rate: {current_lr:.6f}")
        
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
        print(f"  Saved checkpoint to {output_dir / f'checkpoint_{epoch}.pth'}")
        
        # Save the final model
        if epoch == args.epochs - 1:
            torch.save(model.state_dict(), output_dir / 'model_final.pth')
            print(f"  Saved final model to {output_dir / 'model_final.pth'}")
    
    total_training_time = time.time() - start_time
    print(f"\n{'='*50}")
    print("=== TRAINING COMPLETED ===")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")
    
    writer.close()

if __name__ == '__main__':
    main() 