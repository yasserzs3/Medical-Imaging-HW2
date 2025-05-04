import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from pathlib import Path
import random
import json
import shutil
import yaml
import torch
import torchvision.transforms as transforms

class DataTransformer:
    def __init__(self, output_dir=None, apply_normalization=False):
        """
        Initialize the data transformer.
        
        Args:
            output_dir (str, optional): Directory to save processed data. If None, no default directory is created.
            apply_normalization (bool): Whether to apply normalization
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(exist_ok=True, parents=True)
        self.apply_normalization = apply_normalization
        
        # Create transformation pipelines
        self.base_transform = self.create_base_transform()
        self.augmentation_transform = self.create_augmentation_transform()
    
    def create_base_transform(self):
        """
        Create the base transformation pipeline.
        
        Returns:
            A.Compose: Base transformation pipeline
        """
        transforms = [
            A.Resize(640, 640),  # Resize to target size
        ]
        
        # Add normalization only if requested
        if self.apply_normalization:
            transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            
        return A.Compose(transforms, bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
    
    def create_augmentation_transform(self):
        """
        Create the augmentation transformation pipeline.
        
        Returns:
            A.Compose: Augmentation transformation pipeline
        """
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(height=608, width=608, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.RandomRotate90(p=0.2),
            A.Rotate(limit=5, p=0.3),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0.04, val_shift_limit=0, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.GaussNoise(p=0.3),
            A.Resize(640, 640),  # Final resize to target size
        ]
        
        # Add normalization only if requested
        if self.apply_normalization:
            transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            
        return A.Compose(transforms, bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
    
    def transform_image(self, image, bboxes, category_ids, augment=False):
        """
        Transform an image and its bounding boxes.
        
        Args:
            image (np.ndarray): Image to transform
            bboxes (list): List of bounding boxes in COCO format [x, y, width, height]
            category_ids (list): List of category IDs
            augment (bool): Whether to apply augmentation
        
        Returns:
            tuple: (transformed_image, transformed_bboxes, transformed_category_ids)
        """
        if augment:
            transform = self.augmentation_transform
        else:
            transform = self.base_transform
        
        # Apply transformation
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        
        return transformed['image'], transformed['bboxes'], transformed['category_ids']
    
    def transform_dataset(self, annotations, image_paths, split, apply_augmentation=True, augmentation_factor=2):
        """
        Transform a dataset split.
        
        Args:
            annotations (dict): Annotations dictionary for the split
            image_paths (list): List of image paths
            split (str): Dataset split name
            apply_augmentation (bool): Whether to apply augmentation
            augmentation_factor (int): Number of augmentations per image
        
        Returns:
            dict: Transformed dataset
        """
        # Create output directories
        split_dir = self.output_dir / split
        split_dir.mkdir(exist_ok=True)
        
        # Prepare new annotation
        new_anno = {
            "info": annotations.get("info", {}),
            "licenses": annotations.get("licenses", []),
            "categories": annotations.get("categories", []),  # Preserve categories
            "images": [],
            "annotations": []
        }
        
        # Create a map from file_name to image_id
        file_to_id = {img['file_name']: img['id'] for img in annotations['images']}
        
        # Create a map from image_id to its annotations
        id_to_annos = {}
        for anno in annotations['annotations']:
            image_id = anno['image_id']
            if image_id not in id_to_annos:
                id_to_annos[image_id] = []
            id_to_annos[image_id].append(anno)
        
        # Process each image
        new_image_id = 0
        new_anno_id = 0
        
        for image_info in annotations['images']:
            image_id = image_info['id']
            file_name = image_info['file_name']
            
            # Find matching image path
            image_path = None
            for path in image_paths:
                if Path(path).name == file_name:
                    image_path = path
                    break
            
            if not image_path:
                print(f"Warning: Image {file_name} not found in image paths")
                continue
            
            # Load image
            try:
                image = np.array(Image.open(image_path))
                if image.ndim == 2:  # Convert grayscale to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:  # Remove alpha channel
                    image = image[:, :, :3]
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
            
            # Get annotations for this image
            img_annos = id_to_annos.get(image_id, [])
            if not img_annos:
                print(f"Warning: No annotations found for image {file_name}")
                continue
            
            # Extract bounding boxes and category IDs
            bboxes = [anno['bbox'] for anno in img_annos]
            category_ids = [anno['category_id'] for anno in img_annos]
            
            # Process the base image
            base_image_filename = f"{new_image_id:06d}.jpg"
            transformed_image, transformed_bboxes, transformed_category_ids = self.transform_image(
                image, bboxes, category_ids, augment=False
            )
            
            # Save image - handle normalized vs non-normalized differently
            if self.apply_normalization:
                # Denormalize before saving if normalization was applied
                transformed_pil = Image.fromarray((transformed_image * 255).astype(np.uint8))
            else:
                # If no normalization, just save as-is
                transformed_pil = Image.fromarray(transformed_image.astype(np.uint8))
                
            transformed_pil.save(split_dir / base_image_filename)
            
            # Add to annotations
            new_image_info = dict(image_info)
            new_image_info['id'] = new_image_id
            new_image_info['file_name'] = base_image_filename
            new_image_info['width'] = 640
            new_image_info['height'] = 640
            new_anno['images'].append(new_image_info)
            
            # Add bounding box annotations
            for bbox, cat_id in zip(transformed_bboxes, transformed_category_ids):
                new_anno['annotations'].append({
                    'id': new_anno_id,
                    'image_id': new_image_id,
                    'category_id': cat_id,
                    'bbox': [float(x) for x in bbox],
                    'area': float(bbox[2] * bbox[3]),
                    'iscrowd': 0,
                    'segmentation': []
                })
                new_anno_id += 1
            
            new_image_id += 1
            
            # Apply augmentations if needed
            if apply_augmentation and augmentation_factor > 0:
                for aug_idx in range(augmentation_factor):
                    aug_image_filename = f"{new_image_id:06d}.jpg"
                    aug_image, aug_bboxes, aug_category_ids = self.transform_image(
                        image, bboxes, category_ids, augment=True
                    )
                    
                    # Save augmented image - handle normalized vs non-normalized differently
                    if self.apply_normalization:
                        # Denormalize before saving if normalization was applied
                        aug_pil = Image.fromarray((aug_image * 255).astype(np.uint8))
                    else:
                        # If no normalization, just save as-is
                        aug_pil = Image.fromarray(aug_image.astype(np.uint8))
                        
                    aug_pil.save(split_dir / aug_image_filename)
                    
                    # Add to annotations
                    aug_image_info = dict(image_info)
                    aug_image_info['id'] = new_image_id
                    aug_image_info['file_name'] = aug_image_filename
                    aug_image_info['width'] = 640
                    aug_image_info['height'] = 640
                    new_anno['images'].append(aug_image_info)
                    
                    # Add bounding box annotations
                    for bbox, cat_id in zip(aug_bboxes, aug_category_ids):
                        new_anno['annotations'].append({
                            'id': new_anno_id,
                            'image_id': new_image_id,
                            'category_id': cat_id,
                            'bbox': [float(x) for x in bbox],
                            'area': float(bbox[2] * bbox[3]),
                            'iscrowd': 0,
                            'segmentation': []
                        })
                        new_anno_id += 1
                    
                    new_image_id += 1
        
        print(f"Processed {split} split: {len(new_anno['images'])} images, {len(new_anno['annotations'])} annotations")
        
        return new_anno
    
    def transform_all(self, annotations, image_paths, apply_augmentation=True):
        """
        Transform all dataset splits.
        
        Args:
            annotations (dict): Dictionary of annotations for each split
            image_paths (dict): Dictionary of image paths for each split
            apply_augmentation (bool): Whether to apply augmentation
        
        Returns:
            dict: Dictionary of transformed annotations
        """
        transformed = {}
        
        for split in annotations.keys():
            print(f"\nTransforming {split} split...")
            
            # Apply different augmentation settings for different splits
            if split == 'train':
                aug_factor = 2 if apply_augmentation else 0
            else:
                aug_factor = 0  # No augmentation for valid and test
            
            transformed[split] = self.transform_dataset(
                annotations[split], 
                image_paths[split], 
                split, 
                apply_augmentation=(split == 'train' and apply_augmentation),
                augmentation_factor=aug_factor
            )
        
        return transformed
    
    def convert_to_yolov8_format(self, annotations, output_dir):
        """
        [DEPRECATED] Convert COCO annotations to YOLOv8 format.
        This method requires intermediate processed files. 
        Consider using convert_to_yolov8_format_direct instead.
        
        Args:
            annotations (dict): Dictionary of annotations for each split
            output_dir (str): Output directory for YOLOv8 format data
        """
        print("WARNING: Using deprecated convert_to_yolov8_format method. Consider using convert_to_yolov8_format_direct instead.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create YOLOv8 directory structure
        images_dir = output_dir / 'images'
        labels_dir = output_dir / 'labels'
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            # YOLOv8 uses 'val' instead of 'valid'
            source_split = 'valid' if split == 'val' else split
            
            # Skip if split doesn't exist in annotations
            if source_split not in annotations:
                print(f"Warning: {source_split} split not found in annotations")
                continue
                
            # Create directories
            split_images_dir = images_dir / split
            split_labels_dir = labels_dir / split
            
            split_images_dir.mkdir(exist_ok=True, parents=True)
            split_labels_dir.mkdir(exist_ok=True, parents=True)
            
            anno = annotations[source_split]
            
            # Map from image_id to file_name
            id_to_file = {img['id']: img['file_name'] for img in anno['images']}
            
            # Map from image_id to its dimensions
            id_to_dims = {img['id']: (img['width'], img['height']) for img in anno['images']}
            
            # Group annotations by image_id
            annotations_by_image = {}
            for a in anno['annotations']:
                image_id = a['image_id']
                if image_id not in annotations_by_image:
                    annotations_by_image[image_id] = []
                annotations_by_image[image_id].append(a)
            
            # Process each image
            for img in anno['images']:
                image_id = img['id']
                file_name = img['file_name']
                width, height = id_to_dims[image_id]
                
                # Copy image to YOLOv8 format - use original filename to maintain traceability
                src_path = self.output_dir / source_split / file_name
                dst_path = split_images_dir / file_name
                
                if src_path.exists():
                    shutil.copy(src_path, dst_path)
                
                # Create YOLO label file
                label_file = split_labels_dir / (Path(file_name).stem + '.txt')
                
                with open(label_file, 'w') as f:
                    if image_id in annotations_by_image:
                        for a in annotations_by_image[image_id]:
                            # COCO bbox: [x, y, width, height]
                            # YOLO bbox: [x_center/width, y_center/height, width/width, height/height]
                            bbox = a['bbox']
                            x, y, w, h = bbox
                            
                            # Convert to YOLO format (normalized)
                            x_center = (x + w / 2) / width
                            y_center = (y + h / 2) / height
                            w_norm = w / width
                            h_norm = h / height
                            
                            # Class ID (0-indexed in YOLO)
                            class_id = a['category_id'] - 1  # Adjust if necessary for your dataset
                            
                            # Write to file: class_id, x_center, y_center, width, height
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            
            print(f"Converted {source_split} split to YOLOv8 format in {split} directory")
        
        # Create dataset.yaml file
        categories = [cat['name'] for cat in next(iter(annotations.values()))['categories']]
        
        with open(output_dir / 'dataset.yaml', 'w') as f:
            f.write(f"# YOLOv8 dataset configuration\n")
            f.write(f"path: {output_dir.absolute()}\n")
            f.write(f"train: images/train\n")
            f.write(f"val: images/val\n")
            f.write(f"test: images/test\n\n")
            f.write(f"# Number of classes\n")
            f.write(f"nc: {len(categories)}\n\n")
            f.write(f"# Class names\n")
            f.write(f"names:\n")
            for i, name in enumerate(categories):
                f.write(f"  {i}: {name}\n")
        
        print(f"Created YOLOv8 dataset.yaml configuration file")

    def get_dataset_stats(self, annotations):
        """
        Get dataset statistics.
        
        Args:
            annotations (dict): Annotations dictionary
        
        Returns:
            dict: Dataset statistics
        """
        stats = {}
        
        for split, anno in annotations.items():
            if not anno:
                continue
                
            # Count categories
            categories = {cat['id']: cat['name'] for cat in anno.get('categories', [])}
            cat_counts = {cat_name: 0 for cat_id, cat_name in categories.items()}
            
            # Count annotations per category
            for a in anno.get('annotations', []):
                cat_id = a['category_id'] 
                cat_name = categories.get(cat_id, f"Unknown-{cat_id}")
                cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
            
            # Collect bounding box stats
            if len(anno.get('annotations', [])) > 0:
                boxes = np.array([a['bbox'] for a in anno['annotations']])
                box_widths = boxes[:, 2]
                box_heights = boxes[:, 3]
                
                stats[split] = {
                    'num_images': len(anno.get('images', [])),
                    'num_annotations': len(anno.get('annotations', [])),
                    'categories': categories,
                    'category_counts': cat_counts,
                    'avg_box_width': float(np.mean(box_widths)),
                    'avg_box_height': float(np.mean(box_heights)),
                    'min_box_width': float(np.min(box_widths)),
                    'min_box_height': float(np.min(box_heights)),
                    'max_box_width': float(np.max(box_widths)),
                    'max_box_height': float(np.max(box_heights)),
                }
            else:
                stats[split] = {
                    'num_images': len(anno.get('images', [])),
                    'num_annotations': 0,
                    'categories': categories,
                    'category_counts': cat_counts,
                    'avg_box_width': 0,
                    'avg_box_height': 0,
                    'min_box_width': 0,
                    'min_box_height': 0,
                    'max_box_width': 0,
                    'max_box_height': 0,
                }
        
        return stats

    def convert_to_yolov8_format_direct(self, annotations, image_paths, output_dir, apply_augmentation=False):
        """
        Convert raw annotations directly to YOLOv8 format without saving intermediate files.
        
        Args:
            annotations (dict): Dictionary of annotations for each split
            image_paths (dict): Dictionary of image paths for each split
            output_dir (str): Output directory for YOLOv8 format data
            apply_augmentation (bool): Whether to apply augmentation to training data
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create YOLOv8 directory structure
        images_dir = output_dir / 'images'
        labels_dir = output_dir / 'labels'
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            # YOLOv8 uses 'val' instead of 'valid'
            source_split = 'valid' if split == 'val' else split
            
            # Skip if split doesn't exist in annotations
            if source_split not in annotations:
                print(f"Warning: {source_split} split not found in annotations")
                continue
                
            # Create directories
            split_images_dir = images_dir / split
            split_labels_dir = labels_dir / split
            
            split_images_dir.mkdir(exist_ok=True, parents=True)
            split_labels_dir.mkdir(exist_ok=True, parents=True)
            
            anno = annotations[source_split]
            
            # Create a map from file_name to image_id
            file_to_id = {img['file_name']: img['id'] for img in anno['images']}
            
            # Create a map from image_id to its annotations
            id_to_annos = {}
            for anno_item in anno['annotations']:
                image_id = anno_item['image_id']
                if image_id not in id_to_annos:
                    id_to_annos[image_id] = []
                id_to_annos[image_id].append(anno_item)
            
            # Process each image
            processed_count = 0
            for image_info in anno['images']:
                image_id = image_info['id']
                file_name = image_info['file_name']
                
                # Find matching image path
                image_path = None
                for path in image_paths[source_split]:
                    if Path(path).name == file_name:
                        image_path = path
                        break
                
                if not image_path:
                    print(f"Warning: Image {file_name} not found in image paths")
                    continue
                
                # Load image
                try:
                    image = np.array(Image.open(image_path))
                    if image.ndim == 2:  # Convert grayscale to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    elif image.shape[2] == 4:  # Remove alpha channel
                        image = image[:, :, :3]
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    continue
                
                # Get annotations for this image
                img_annos = id_to_annos.get(image_id, [])
                if not img_annos:
                    print(f"Warning: No annotations found for image {file_name}")
                    continue
                
                # Extract bounding boxes and category IDs
                bboxes = [anno_item['bbox'] for anno_item in img_annos]
                category_ids = [anno_item['category_id'] for anno_item in img_annos]
                
                # Process the base image
                out_img_filename = f"{processed_count:06d}.jpg"
                
                # Apply base transform
                transformed_image, transformed_bboxes, transformed_category_ids = self.transform_image(
                    image, bboxes, category_ids, augment=False
                )
                
                # Save the image
                if self.apply_normalization:
                    # Denormalize before saving if normalization was applied
                    transformed_pil = Image.fromarray((transformed_image * 255).astype(np.uint8))
                else:
                    # If no normalization, just save as-is
                    transformed_pil = Image.fromarray(transformed_image.astype(np.uint8))
                
                # Save the transformed image
                img_output_path = split_images_dir / out_img_filename
                transformed_pil.save(img_output_path)
                
                # Create YOLO label file
                width, height = 640, 640  # Size after resizing
                label_file = split_labels_dir / (Path(out_img_filename).stem + '.txt')
                
                with open(label_file, 'w') as f:
                    for bbox, cat_id in zip(transformed_bboxes, transformed_category_ids):
                        # COCO bbox: [x, y, width, height]
                        # YOLO bbox: [x_center/width, y_center/height, width/width, height/height]
                        x, y, w, h = bbox
                        
                        # Convert to YOLO format (normalized)
                        x_center = (x + w / 2) / width
                        y_center = (y + h / 2) / height
                        w_norm = w / width
                        h_norm = h / height
                        
                        # Class ID (0-indexed in YOLO)
                        class_id = cat_id - 1  # Adjust for YOLO format
                        
                        # Write to file: class_id, x_center, y_center, width, height
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                
                processed_count += 1
                
                # Apply augmentations for training data
                if source_split == 'train' and apply_augmentation:
                    for aug_idx in range(2):  # Create 2 augmented versions per image
                        aug_img_filename = f"{processed_count:06d}.jpg"
                        
                        # Apply augmentation transform
                        aug_image, aug_bboxes, aug_category_ids = self.transform_image(
                            image, bboxes, category_ids, augment=True
                        )
                        
                        # Save augmented image
                        if self.apply_normalization:
                            # Denormalize before saving if normalization was applied
                            aug_pil = Image.fromarray((aug_image * 255).astype(np.uint8))
                        else:
                            # If no normalization, just save as-is
                            aug_pil = Image.fromarray(aug_image.astype(np.uint8))
                        
                        # Save the augmented image
                        aug_output_path = split_images_dir / aug_img_filename
                        aug_pil.save(aug_output_path)
                        
                        # Create YOLO label file for augmented image
                        aug_label_file = split_labels_dir / (Path(aug_img_filename).stem + '.txt')
                        
                        with open(aug_label_file, 'w') as f:
                            for bbox, cat_id in zip(aug_bboxes, aug_category_ids):
                                # Convert to YOLO format
                                x, y, w, h = bbox
                                x_center = (x + w / 2) / width
                                y_center = (y + h / 2) / height
                                w_norm = w / width
                                h_norm = h / height
                                
                                # Class ID (0-indexed in YOLO)
                                class_id = cat_id - 1
                                
                                # Write to file
                                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                        
                        processed_count += 1
            
            print(f"Processed {source_split} split: {processed_count} images")
        
        # Create dataset.yaml file
        categories = [cat['name'] for cat in next(iter(annotations.values()))['categories']]
        
        with open(output_dir / 'dataset.yaml', 'w') as f:
            f.write(f"# YOLOv8 dataset configuration\n")
            f.write(f"path: {output_dir.absolute()}\n")
            f.write(f"train: images/train\n")
            f.write(f"val: images/val\n")
            f.write(f"test: images/test\n\n")
            f.write(f"# Number of classes\n")
            f.write(f"nc: {len(categories)}\n\n")
            f.write(f"# Class names\n")
            f.write(f"names:\n")
            for i, name in enumerate(categories):
                f.write(f"  {i}: {name}\n")
        
        print(f"Created YOLOv8 dataset.yaml configuration file at {output_dir / 'dataset.yaml'}")

    def convert_to_ssd_format(self, annotations, image_paths, output_dir, apply_augmentation=False):
        """
        Convert dataset to SSD format.
        
        Args:
            annotations (dict): Dictionary of annotations for each split
            image_paths (dict): Dictionary of image paths for each split
            output_dir (str): Output directory
            apply_augmentation (bool): Whether to apply augmentation
        
        Returns:
            dict: Dataset statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create directories for each output split: train, valid, test
        for split in ['train', 'valid', 'test']:
            (output_dir / split).mkdir(exist_ok=True, parents=True)

        # Determine categories from annotations
        # annotations is a dict of split_name -> split_annotations
        first_split = next((s for s in ['train', 'valid', 'test'] if s in annotations), None)
        if not first_split:
            raise ValueError("No annotation splits found in annotations")
        categories = annotations[first_split].get('categories', [])
        if not categories:
            raise ValueError(f"No categories found in annotations[{first_split}]")

        # Create dataset.yaml
        dataset_yaml = {
            'path': str(output_dir),
            'train': 'train',
            'valid': 'valid',
            'test': 'test',
            'names': {cat['id']: cat['name'] for cat in categories}
        }
        with open(output_dir / 'dataset.yaml', 'w') as f:
            yaml.safe_dump(dataset_yaml, f)

        stats = {}
        # Process each split: train, valid, test
        for split in ['train', 'valid', 'test']:
            if split not in annotations:
                print(f"Warning: {split} split not found in annotations")
                continue
            split_ann = annotations[split]
            split_paths = image_paths.get(split, [])
            # Transform dataset for this split
            transformed = self.transform_dataset(
                split_ann,
                split_paths,
                split,
                apply_augmentation=(split == 'train' and apply_augmentation)
            )
            # Save COCO format annotations
            ann_path = output_dir / split / '_annotations.coco.json'
            with open(ann_path, 'w') as f:
                json.dump(transformed, f)
            # Compute statistics for this split
            stats[split] = self.get_dataset_stats({split: transformed})[split]

        return stats


