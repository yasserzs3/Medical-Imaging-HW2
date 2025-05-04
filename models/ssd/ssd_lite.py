import torch
import torch.nn as nn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from typing import Optional, Dict, Any
from torchvision.ops import boxes as box_ops

class SSDLiteWrapper:
    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        Initialize SSDLite model wrapper
        
        Args:
            num_classes (int): Number of classes (including background)
            pretrained (bool): Whether to use pretrained weights
        """
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = None
        
    def get_model(self) -> nn.Module:
        """
        Get the SSDLite model with specified number of classes
        
        Returns:
            nn.Module: Configured SSDLite model
        """
        # Load pretrained model
        self.model = ssdlite320_mobilenet_v3_large(pretrained=self.pretrained)
        
        # Replace classification head to match custom number of classes
        original_head = self.model.head.classification_head
        # Extract in_channels from each prediction block (depthwise conv) in module_list
        module_list = original_head.module_list
        in_channels_list = []
        for block in module_list:
            # block is Sequential(Conv2dNormActivation, Conv2d)
            # block[0] is Conv2dNormActivation, block[0][0] is the Conv2d layer
            depthwise = block[0]
            conv_layer = depthwise[0]
            in_channels_list.append(conv_layer.in_channels)
        # Determine number of anchors per location for each feature map
        num_anchors_list = self.model.anchor_generator.num_anchors_per_location()
        # Assign new classification head
        self.model.head.classification_head = SSDLiteClassificationHead(
            in_channels=in_channels_list,
            num_anchors=num_anchors_list,
            num_classes=self.num_classes,
            norm_layer=nn.BatchNorm2d
        )
        
        # Store the original postprocess_detections method
        original_postprocess = self.model.postprocess_detections
        
        # Override the postprocess_detections method to handle NMS on CPU
        def custom_postprocess_detections(head_outputs, anchors, image_sizes):
            device = head_outputs["bbox_regression"].device
            
            # Get classification and regression outputs
            classification = head_outputs["cls_logits"]
            regression = head_outputs["bbox_regression"]
            
            # Move everything to CPU
            regression_cpu = regression.cpu()
            classification_cpu = classification.cpu()
            anchors_cpu = [anchor.cpu() for anchor in anchors]
            
            # Process each image in the batch
            detections = []
            batch_size = classification.shape[0]
            
            for i in range(batch_size):
                # Process each feature map level
                boxes_list = []
                scores_list = []
                
                for j, (reg, anc) in enumerate(zip(regression_cpu[i], anchors_cpu)):
                    # Handle regression output
                    if reg.dim() == 1:
                        # If 1D, reshape to match anchor format
                        reg = reg.view(-1, 4)
                    else:
                        # If multi-dimensional, reshape appropriately
                        reg = reg.reshape(-1, 4)
                    
                    # Decode boxes for this feature map level
                    boxes = self.model.box_coder.decode_single(reg, anc)
                    boxes_list.append(boxes)
                    
                    # Handle classification output
                    scores = torch.sigmoid(classification_cpu[i][j])
                    if scores.dim() == 1:
                        # If 1D, reshape to match number of classes
                        scores = scores.view(-1, self.num_classes)
                    else:
                        # If multi-dimensional, reshape appropriately
                        scores = scores.reshape(-1, self.num_classes)
                    scores_list.append(scores)
                
                # Concatenate boxes and scores from all feature map levels
                boxes = torch.cat(boxes_list, dim=0)
                scores = torch.cat(scores_list, dim=0)
                
                # Remove low scoring boxes
                inds = torch.where(scores > self.model.score_thresh)[0]
                boxes, scores = boxes[inds], scores[inds]
                
                # Remove empty boxes
                keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
                boxes, scores = boxes[keep], scores[keep]
                
                # Non-maximum suppression on CPU
                keep = box_ops.batched_nms(
                    boxes, 
                    scores.max(1)[0], 
                    torch.zeros_like(scores.max(1)[0], dtype=torch.int64), 
                    self.model.nms_thresh
                )
                keep = keep[:self.model.detections_per_img]
                
                # Prepare final detections
                boxes, scores = boxes[keep], scores[keep]
                labels = torch.argmax(scores, dim=1)
                
                # Move results back to original device
                result = {
                    "boxes": boxes.to(device),
                    "scores": scores.max(1)[0].to(device),
                    "labels": labels.to(device) + 1  # Background is 0, so add 1 to all labels
                }
                detections.append(result)
            
            return detections
            
        # Override the forward method to handle both training and inference
        original_forward = self.model.forward
        def custom_forward(images, targets=None):
            if targets is not None:
                # Compute losses using original forward in training mode
                was_training = self.model.training
                self.model.train()
                loss_dict = original_forward(images, targets)
                # Restore original training mode
                if not was_training:
                    self.model.eval()
                return loss_dict
            else:
                # During inference, use our custom postprocessing
                self.model.postprocess_detections = custom_postprocess_detections
                result = original_forward(images, targets)
                # Restore original postprocessing
                self.model.postprocess_detections = original_postprocess
                return result
        
        # Replace the forward method
        self.model.forward = custom_forward
        
        # Set model to training mode
        self.model.train()
        
        return self.model
    
    def save_weights(self, path: str) -> None:
        """
        Save model weights
        
        Args:
            path (str): Path to save weights
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call get_model() first.")
        torch.save(self.model.state_dict(), path)
    
    def load_weights(self, path: str) -> None:
        """
        Load model weights
        
        Args:
            path (str): Path to weights file
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call get_model() first.")
        self.model.load_state_dict(torch.load(path))
    
    def get_transform(self, train: bool = True) -> Any:
        """
        Get data transforms for training/inference
        
        Args:
            train (bool): Whether to get training transforms
            
        Returns:
            Any: Transforms for the model
        """
        return self.model.transform 