import os
import argparse
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from tqdm import tqdm
import json
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
import csv
from datetime import datetime
import math

def get_transform():
    return A.Compose([
        A.Resize(640, 640),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # Get category information
        self.categories = {cat['id']: cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())}
        self.num_classes = len(self.categories)
        
        # Validate dataset
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate the dataset and print statistics"""
        print(f"\nDataset Statistics:")
        print(f"Number of images: {len(self.ids)}")
        print(f"Number of categories: {self.num_classes}")
        print(f"Categories: {self.categories}")
        
        # Count annotations per category
        ann_count = defaultdict(int)
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                ann_count[ann['category_id']] += 1
        
        print("\nAnnotations per category:")
        for cat_id, count in ann_count.items():
            print(f"{self.categories[cat_id]}: {count}")
        print()
    
    def _validate_boxes(self, boxes, img_width, img_height):
        """Validate and fix bounding boxes"""
        valid_boxes = []
        for box in boxes:
            x1, y1, w, h = box
            # Ensure box is within image boundaries
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            w = min(w, img_width - x1)
            h = min(h, img_height - y1)
            
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
                
            # Convert to [x1, y1, x2, y2] format
            valid_boxes.append([x1, y1, x1 + w, y1 + h])
        
        return valid_boxes
    
    def __getitem__(self, idx):
        try:
            # Load image
            img_id = self.ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.root, img_info['file_name'])
            
            # Load and validate image
            img = Image.open(img_path).convert("RGB")
            img_width, img_height = img.size
            img = np.array(img)
            
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"Invalid image format: {img_path}")
            
            # Load annotations
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            boxes = []
            labels = []
            valid_anns = [] # Keep track of valid annotations

            for ann in anns:
                bbox = ann['bbox']  # [x, y, width, height]
                category_id = ann['category_id']
                
                # Validate category
                if category_id not in self.categories:
                    continue
                
                # Preliminary validation (e.g., basic format check if needed)
                # ...

                valid_anns.append({'bbox': bbox, 'category_id': category_id})


            # Validate boxes and keep corresponding labels
            validated_boxes = []
            validated_labels = []
            for ann_data in valid_anns:
                box = ann_data['bbox']
                label = ann_data['category_id']

                x1, y1, w, h = box
                # Ensure box is within image boundaries
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                w = min(w, img_width - x1)
                h = min(h, img_height - y1)

                # Skip invalid boxes AND their labels
                if w <= 0 or h <= 0:
                    continue

                # Convert to [x1, y1, x2, y2] format and store
                validated_boxes.append([x1, y1, x1 + w, y1 + h])
                validated_labels.append(label)


            if len(validated_boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.int64)
            else:
                boxes = torch.as_tensor(validated_boxes, dtype=torch.float32)
                labels = torch.as_tensor(validated_labels, dtype=torch.int64) # Use validated labels

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = torch.tensor([img_id])
            target["area"] = torch.tensor([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes])
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
            
            if self.transforms is not None:
                transformed = self.transforms(image=img, bboxes=boxes.numpy(), labels=labels.numpy())
                img = transformed['image']
                target["boxes"] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                target["labels"] = torch.as_tensor(transformed['labels'], dtype=torch.int64)
            
            return img, target
            
        except Exception as e:
            print(f"Error loading image {img_id}: {str(e)}")
            # Return a valid empty sample
            empty_img = torch.zeros((3, 800, 800), dtype=torch.float32)
            empty_target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": torch.tensor([img_id]),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.int64)
            }
            return empty_img, empty_target
    
    def __len__(self):
        return len(self.ids)

def get_model(num_classes):
    # Load a pre-trained model
    backbone = torchvision.models.resnet34(weights='DEFAULT')
    # Remove the last fully connected layer
    layers = list(backbone.children())[:-2]
    backbone = torch.nn.Sequential(*layers)
    backbone.out_channels = 512  # ResNet34's last layer has 512 channels
    
    # Define anchor generator with more scales and aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128, 256, 512),),  # Adjusted scales for smaller objects
        aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),)
    )
    
    # Define ROI Pooler with smaller output size
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=14,  # Increased from 7 to 14 for better feature resolution
        sampling_ratio=2
    )
    
    # Create Faster R-CNN model with modified parameters
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        rpn_pre_nms_top_n_train=3000,
        rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1500,
        rpn_post_nms_top_n_test=300,  # Reduced to limit noisy detections
        rpn_nms_thresh=0.6,
        box_score_thresh=0.3,  # Increased to reduce false positives
        box_nms_thresh=0.4,
        box_detections_per_img=100  # Limited to reduce noisy detections
    )
    
    return model

def calculate_metrics(predictions, targets, iou_threshold=0.5, confidence_threshold=0.3):
    """
    Calculate precision, recall, F1-score, and mAP for a single image
    """
    if len(predictions['boxes']) == 0:
        return 0, 0, 0, 0
    
    # Convert predictions to numpy for easier calculation
    pred_boxes = predictions['boxes'].cpu().numpy()
    pred_scores = predictions['scores'].cpu().numpy()
    pred_labels = predictions['labels'].cpu().numpy()
    
    # Filter predictions by confidence threshold
    confidence_mask = pred_scores >= confidence_threshold
    pred_boxes = pred_boxes[confidence_mask]
    pred_scores = pred_scores[confidence_mask]
    pred_labels = pred_labels[confidence_mask]
    
    if len(pred_boxes) == 0:
        return 0, 0, 0, 0
    
    # Convert targets to numpy
    target_boxes = targets['boxes'].cpu().numpy()
    target_labels = targets['labels'].cpu().numpy()
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(target_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, target_box in enumerate(target_boxes):
            iou_matrix[i, j] = calculate_iou(pred_box, target_box)
    
    # Match predictions to ground truth
    matched_preds = set()
    matched_targets = set()
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Sort predictions by confidence
    sorted_indices = np.argsort(-pred_scores)
    
    # Calculate mAP
    aps = []
    for class_id in np.unique(target_labels):
        # Get predictions and ground truth for this class
        class_pred_mask = pred_labels == class_id
        class_target_mask = target_labels == class_id
        
        if not np.any(class_target_mask):
            continue
            
        class_pred_boxes = pred_boxes[class_pred_mask]
        class_pred_scores = pred_scores[class_pred_mask]
        class_target_boxes = target_boxes[class_target_mask]
        
        # Calculate IoU for this class
        class_iou_matrix = np.zeros((len(class_pred_boxes), len(class_target_boxes)))
        for i, pred_box in enumerate(class_pred_boxes):
            for j, target_box in enumerate(class_target_boxes):
                class_iou_matrix[i, j] = calculate_iou(pred_box, target_box)
        
        # Calculate precision-recall curve
        precisions = []
        recalls = []
        thresholds = np.arange(0.5, 1.0, 0.05)
        
        for threshold in thresholds:
            tp = 0
            fp = 0
            fn = 0
            
            # Match predictions to ground truth
            matched_targets = set()
            for i, pred_box in enumerate(class_pred_boxes):
                if class_pred_scores[i] < threshold:
                    continue
                    
                best_iou = 0
                best_target_idx = -1
                
                for j, target_box in enumerate(class_target_boxes):
                    if j in matched_targets:
                        continue
                        
                    iou = class_iou_matrix[i, j]
                    if iou > best_iou:
                        best_iou = iou
                        best_target_idx = j
                
                if best_iou >= iou_threshold:
                    tp += 1
                    matched_targets.add(best_target_idx)
                else:
                    fp += 1
            
            fn = len(class_target_boxes) - len(matched_targets)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP for this class
        ap = 0
        for i in range(len(thresholds) - 1):
            ap += (recalls[i + 1] - recalls[i]) * precisions[i]
        aps.append(ap)
    
    # Calculate mAP
    mAP = np.mean(aps) if aps else 0
    
    # Calculate overall precision, recall, and F1
    for pred_idx in sorted_indices:
        if pred_idx in matched_preds:
            continue
            
        # Find best matching target
        best_iou = 0
        best_target_idx = -1
        
        for target_idx in range(len(target_boxes)):
            if target_idx in matched_targets:
                continue
                
            if pred_labels[pred_idx] == target_labels[target_idx]:
                iou = iou_matrix[pred_idx, target_idx]
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = target_idx
        
        # If IoU > threshold, count as true positive
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_preds.add(pred_idx)
            matched_targets.add(best_target_idx)
        else:
            false_positives += 1
    
    false_negatives = len(target_boxes) - len(matched_targets)
    
    # Calculate final metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score, mAP

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes
    """
    # Convert to [x1, y1, x2, y2] format if needed
    if len(box1) == 4:
        x1_1, y1_1, x2_1, y2_1 = box1
    else:
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
    if len(box2) == 4:
        x1_2, y1_2, x2_2, y2_2 = box2
    else:
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0

def evaluate(model, data_loader, device):
    # Save original mode and switch to train mode to compute losses
    was_training = model.training
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Metrics tracking
    metrics = defaultdict(list)
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Skip if no boxes
            if all(len(t['boxes']) == 0 for t in targets):
                continue
            
            # Calculate loss
            loss_dict = model(images, targets)
            combined_loss = None
            for v in loss_dict.values():
                if torch.is_tensor(v):
                    term = v.sum() if v.dim() > 0 else v
                elif isinstance(v, (int, float)):
                    term = torch.tensor(v, dtype=torch.float32, device=device)
                else:
                    continue
                combined_loss = term if combined_loss is None else combined_loss + term
            # Skip if no valid loss terms
            if combined_loss is None:
                continue
            losses = combined_loss.item()
            
            # Skip batch if loss is NaN
            if math.isnan(losses):
                continue
                
            total_loss += losses
            num_batches += 1
            
            # Get predictions by switching to eval mode
            model.eval()
            predictions = model(images)
            # Switch back to train mode for next iteration
            model.train()
            
            # Calculate metrics for each image
            for pred, target in zip(predictions, targets):
                precision, recall, f1, mAP = calculate_metrics(pred, target)
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                metrics['f1'].append(f1)
                metrics['mAP'].append(mAP)
    
    # Calculate average metrics
    avg_metrics = {
        'loss': total_loss / num_batches if num_batches > 0 else float('inf'),
        'precision': np.mean(metrics['precision']) if metrics['precision'] else 0,
        'recall': np.mean(metrics['recall']) if metrics['recall'] else 0,
        'f1': np.mean(metrics['f1']) if metrics['f1'] else 0,
        'mAP': np.mean(metrics['mAP']) if metrics['mAP'] else 0
    }
    
    # Restore original training state
    if not was_training:
        model.eval()
    return avg_metrics

def train_one_epoch(model, optimizer, data_loader, device, scaler):
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Metrics tracking
    metrics = defaultdict(list)
    
    for images, targets in tqdm(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Skip if no boxes
        if all(len(t['boxes']) == 0 for t in targets):
            continue
        
        # Forward pass with targets for loss calculation
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Skip batch if loss is NaN
        if torch.isnan(losses):
            print("Skipping batch with NaN loss")
            continue
            
        optimizer.zero_grad()
        losses.backward()
        
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
        optimizer.step()
        
        total_loss += losses.detach().item()
        num_batches += 1
        
        # Calculate metrics for each image
        with torch.no_grad():
            # Switch to eval mode temporarily for predictions
            model.eval()
            predictions = model(images)
            model.train()  # Switch back to train mode
            
            for pred, target in zip(predictions, targets):
                precision, recall, f1, mAP = calculate_metrics(pred, target)
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                metrics['f1'].append(f1)
                metrics['mAP'].append(mAP)
    
    # Calculate average metrics
    avg_metrics = {
        'loss': total_loss / num_batches if num_batches > 0 else float('inf'),
        'precision': np.mean(metrics['precision']) if metrics['precision'] else 0,
        'recall': np.mean(metrics['recall']) if metrics['recall'] else 0,
        'f1': np.mean(metrics['f1']) if metrics['f1'] else 0,
        'mAP': np.mean(metrics['mAP']) if metrics['mAP'] else 0
    }
    
    return avg_metrics

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define constant data directories
    train_data_dir = os.path.join('data', 'raw', 'train')
    val_data_dir = os.path.join('data', 'raw', 'valid')
    train_ann_file = os.path.join(train_data_dir, '_annotations.coco.json')
    val_ann_file = os.path.join(val_data_dir, '_annotations.coco.json')
    
    # Create datasets with improved augmentations
    train_transform = A.Compose([
        A.Resize(800, 800),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3),
        A.Rotate(limit=10, p=0.2),  # Replaced RandomRotate90 with controlled rotation
        A.RandomScale(scale_limit=0.1, p=0.2),  # Reduced scale limit
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    val_transform = A.Compose([
        A.Resize(800, 800),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    # Create datasets
    print("\nLoading training dataset...")
    train_dataset = COCODataset(
        root=train_data_dir,
        annotation=train_ann_file,
        transforms=train_transform
    )
    
    print("\nLoading validation dataset...")
    val_dataset = COCODataset(
        root=val_data_dir,
        annotation=val_ann_file,
        transforms=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Create model
    model = get_model(num_classes=train_dataset.num_classes + 1)  # +1 for background
    model.to(device)
    
    # Create optimizer with higher learning rate and momentum
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)  # Increased learning rate
    
    # Create learning rate scheduler with warmup
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.002,  # Increased max learning rate
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,  # Reduced warmup period
        div_factor=25.0,  # Adjusted for higher learning rate
        final_div_factor=100.0
    )
    
    # Create output directory
    output_dir = os.path.join('data', 'runs', 'faster_rcnn')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create CSV file for logging metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'training_metrics_{timestamp}.csv')
    
    # Write CSV header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch',
            'train_loss',
            'val_loss',
            'train_precision',
            'train_recall',
            'train_f1',
            'train_mAP',
            'val_precision',
            'val_recall',
            'val_f1',
            'val_mAP',
            'learning_rate'
        ])
    
    # Training loop
    best_val_loss = float('inf')
    metrics_history = defaultdict(list)
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = train_one_epoch(model, optimizer, train_loader, device, None)
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Train Precision: {train_metrics['precision']:.4f}")
        print(f"Train Recall: {train_metrics['recall']:.4f}")
        print(f"Train F1-Score: {train_metrics['f1']:.4f}")
        print(f"Train mAP: {train_metrics['mAP']:.4f}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1-Score: {val_metrics['f1']:.4f}")
        print(f"Val mAP: {val_metrics['mAP']:.4f}")
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Update learning rate
        lr_scheduler.step()
        
        # Save metrics history
        for metric in ['loss', 'precision', 'recall', 'f1', 'mAP']:
            metrics_history[f'train_{metric}'].append(train_metrics[metric])
            metrics_history[f'val_{metric}'].append(val_metrics[metric])
        
        # Log metrics to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{train_metrics['loss']:.4f}",
                f"{val_metrics['loss']:.4f}",
                f"{train_metrics['precision']:.4f}",
                f"{train_metrics['recall']:.4f}",
                f"{train_metrics['f1']:.4f}",
                f"{train_metrics['mAP']:.4f}",
                f"{val_metrics['precision']:.4f}",
                f"{val_metrics['recall']:.4f}",
                f"{val_metrics['f1']:.4f}",
                f"{val_metrics['mAP']:.4f}",
                f"{current_lr:.6f}"
            ])
        
        # Save checkpoint
        if val_metrics['mAP'] > max(metrics_history['val_mAP'][:-1], default=0):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'metrics_history': metrics_history,
            }, os.path.join(output_dir, 'best_model.pth'))
        
        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'metrics_history': metrics_history,
        }, os.path.join(output_dir, 'latest_model.pth'))
        
        # Plot metrics
        plt.figure(figsize=(15, 10))
        
        # Plot losses
        plt.subplot(2, 2, 1)
        plt.plot(metrics_history['train_loss'], label='Train Loss')
        plt.plot(metrics_history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot precision and recall
        plt.subplot(2, 2, 2)
        plt.plot(metrics_history['train_precision'], label='Train Precision')
        plt.plot(metrics_history['train_recall'], label='Train Recall')
        plt.plot(metrics_history['val_precision'], label='Val Precision')
        plt.plot(metrics_history['val_recall'], label='Val Recall')
        plt.title('Precision and Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        
        # Plot F1-score
        plt.subplot(2, 2, 3)
        plt.plot(metrics_history['train_f1'], label='Train F1')
        plt.plot(metrics_history['val_f1'], label='Val F1')
        plt.title('F1-Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        
        # Plot mAP
        plt.subplot(2, 2, 4)
        plt.plot(metrics_history['train_mAP'], label='Train mAP')
        plt.plot(metrics_history['val_mAP'], label='Val mAP')
        plt.title('mAP')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'metrics_epoch_{epoch+1}.png'))
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)  # 1 class + background
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    
    args = parser.parse_args()
    
    main(args) 