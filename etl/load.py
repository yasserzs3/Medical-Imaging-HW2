import os
import json
import shutil
import torch
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader

class CocoDetectionDataset(Dataset):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        """
        Custom COCO detection dataset.
        
        Args:
            root (str): Root directory where images are stored
            annFile (str): Path to the annotation file
            transform (callable, optional): Transform to be applied on the image
            target_transform (callable, optional): Transform to be applied on the target
        """
        self.root = Path(root)
        
        with open(annFile, 'r') as f:
            self.coco = json.load(f)
        
        self.ids = [img['id'] for img in self.coco['images']]
        self.transform = transform
        self.target_transform = target_transform
        
        # Create a mapping from image ID to file name
        self.id_to_file = {img['id']: img['file_name'] for img in self.coco['images']}
        
        # Create a mapping from image ID to its annotations
        self.id_to_annos = {}
        for anno in self.coco['annotations']:
            img_id = anno['image_id']
            if img_id not in self.id_to_annos:
                self.id_to_annos[img_id] = []
            self.id_to_annos[img_id].append(anno)
        
        # Create a mapping from category ID to name
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.coco['categories']}
        
        # Get the number of categories
        self.num_classes = len(self.coco['categories'])
        
    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Args:
            index (int): Index
            
        Returns:
            tuple: (image, target) where target is the object bounding boxes and labels
        """
        id = self.ids[index]
        
        # Load image
        file_name = self.id_to_file[id]
        img_path = self.root / file_name
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations
        annotations = self.id_to_annos.get(id, [])
        
        # Prepare target
        boxes = []
        labels = []
        
        for anno in annotations:
            bbox = anno['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            # Convert from [x, y, w, h] (COCO format) to [x1, y1, x2, y2] (Pascal VOC format)
            x1, y1, x2, y2 = x, y, x + w, y + h
            boxes.append([x1, y1, x2, y2])
            labels.append(anno['category_id'])
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([id], dtype=torch.int64)
        }
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target
    
    def __len__(self):
        """
        Return the length of the dataset.
        
        Returns:
            int: Length of the dataset
        """
        return len(self.ids)

class DataLoader:
    def __init__(self, coco_data_dir="data_processed", yolo_data_dir="data_yolo"):
        """
        Initialize the data loader.
        
        Args:
            coco_data_dir (str): Directory with COCO format data
            yolo_data_dir (str): Directory with YOLO format data
        """
        self.coco_data_dir = Path(coco_data_dir)
        self.yolo_data_dir = Path(yolo_data_dir)
        
    def load_coco_dataset(self, split, batch_size=8, num_workers=4):
        """
        Load a COCO format dataset for a specific split.
        
        Args:
            split (str): Dataset split ('train', 'valid', or 'test')
            batch_size (int): Batch size
            num_workers (int): Number of worker threads
        
        Returns:
            torch.utils.data.DataLoader: DataLoader for the split
        """
        split_dir = self.coco_data_dir / split
        ann_file = split_dir / "_annotations.coco.json"
        
        if not split_dir.exists() or not ann_file.exists():
            raise FileNotFoundError(f"Cannot find {split} dataset at {split_dir}")
        
        # Define transformations
        if split == 'train':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.04, hue=0
                ),
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        
        # Create dataset
        dataset = CocoDetectionDataset(
            root=split_dir,
            annFile=ann_file,
            transform=transform
        )
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
        
        return dataloader, dataset
    
    def _collate_fn(self, batch):
        """
        Custom collate function for the dataloader.
        
        Args:
            batch (list): Batch of data
        
        Returns:
            tuple: (images, targets)
        """
        images = []
        targets = []
        
        for img, target in batch:
            images.append(img)
            targets.append(target)
        
        return images, targets
    
    def prepare_pytorch_datasets(self, batch_size=8, num_workers=4):
        """
        Prepare PyTorch datasets for all splits.
        
        Args:
            batch_size (int): Batch size
            num_workers (int): Number of worker threads
        
        Returns:
            dict: Dictionary of dataloaders for each split
        """
        dataloaders = {}
        datasets = {}
        
        for split in ['train', 'valid', 'test']:
            try:
                dataloader, dataset = self.load_coco_dataset(split, batch_size, num_workers)
                dataloaders[split] = dataloader
                datasets[split] = dataset
                print(f"Loaded {split} dataset with {len(dataset)} samples")
            except FileNotFoundError as e:
                print(f"Warning: {e}")
        
        return dataloaders, datasets
    
    def prepare_yolo_config(self):
        """
        Prepare YOLO configuration.
        
        Returns:
            str: Path to the YOLO config file
        """
        config_file = self.yolo_data_dir / "dataset.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"YOLO config file not found at {config_file}")
        
        return str(config_file)
    
    def prepare_yolov8_format(self, annotations, output_dir):
        """
        Prepare data for YOLOv8 format.
        
        Args:
            annotations (dict): Dictionary of annotations for each split
            output_dir (str): Output directory for YOLOv8 data
            
        Returns:
            str: Path to the output directory
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
                
                # Copy image to YOLOv8 format
                src_path = self.coco_data_dir / source_split / file_name
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
            
            print(f"Processed {source_split} split to YOLOv8 format in {split} directory")
        
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
        
        return str(output_dir)
    
    def prepare_maskrcnn_format(self, annotations, output_dir="data_maskrcnn"):
        """
        Prepare data for Mask R-CNN format.
        
        Args:
            annotations (dict): Dictionary of annotations for each split
            output_dir (str): Output directory for Mask R-CNN data
        
        Returns:
            str: Path to the output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a README file as a placeholder
        with open(output_dir / "README.md", 'w') as f:
            f.write("# Mask R-CNN Dataset\n\n")
            f.write("Placeholder for Mask R-CNN format dataset.\n")
            f.write("This will be implemented in a future update.\n")
        
        return str(output_dir)
    
    def prepare_ssd_format(self, annotations, output_dir="data_ssd"):
        """
        Prepare data for SSD format.
        
        Args:
            annotations (dict): Dictionary of annotations for each split
            output_dir (str): Output directory for SSD data
        
        Returns:
            str: Path to the output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a README file as a placeholder
        with open(output_dir / "README.md", 'w') as f:
            f.write("# SSD Dataset\n\n")
            f.write("Placeholder for SSD format dataset.\n")
            f.write("This will be implemented in a future update.\n")
        
        return str(output_dir)

if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Prepare datasets for PyTorch-based models
    dataloaders, datasets = loader.prepare_pytorch_datasets()
    
    # Prepare YOLO configuration
    yolo_config = loader.prepare_yolo_config()
    print(f"YOLO config file: {yolo_config}")
    
    # Prepare Mask R-CNN format data
    maskrcnn_dir = loader.prepare_maskrcnn_format()
    print(f"Mask R-CNN data directory: {maskrcnn_dir}")
    
    # Prepare SSD format data
    ssd_dir = loader.prepare_ssd_format()
    print(f"SSD data directory: {ssd_dir}")
