import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

class DataExtractor:
    def __init__(self, data_dir):
        """
        Initialize the data extractor.
        
        Args:
            data_dir (str): Path to the dataset directory
        """
        self.data_dir = Path(data_dir)
        self.splits = ['train', 'valid', 'test']
        
    def extract_annotations(self):
        """
        Extract annotations from COCO JSON files for all splits.
        
        Returns:
            dict: Dictionary containing annotations for each split
        """
        annotations = {}
        
        for split in self.splits:
            split_dir = self.data_dir / split
            anno_file = split_dir / "_annotations.coco.json"
            
            if not anno_file.exists():
                print(f"Warning: Annotation file for {split} not found at {anno_file}")
                continue
                
            with open(anno_file, 'r') as f:
                anno_data = json.load(f)
                
            annotations[split] = anno_data
            print(f"Loaded {split} annotations: {len(anno_data['images'])} images, {len(anno_data['annotations'])} annotations")
            
        return annotations
    
    def extract_images(self):
        """
        Extract image paths for all splits.
        
        Returns:
            dict: Dictionary containing image paths for each split
        """
        image_paths = {}
        
        for split in self.splits:
            split_dir = self.data_dir / split
            
            if not split_dir.exists():
                print(f"Warning: Split directory {split} not found at {split_dir}")
                continue
                
            images = [str(f) for f in split_dir.glob("*.jpg")]
            image_paths[split] = images
            print(f"Found {len(images)} images for {split} split")
            
        return image_paths
    
    def load_image(self, image_path):
        """
        Load an image from disk.
        
        Args:
            image_path (str): Path to the image
        
        Returns:
            PIL.Image: Loaded image
        """
        try:
            return Image.open(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def extract_all(self):
        """
        Extract all data.
        
        Returns:
            tuple: (annotations, image_paths)
        """
        print("Extracting dataset...")
        annotations = self.extract_annotations()
        image_paths = self.extract_images()
        
        return annotations, image_paths
    
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
            categories = {cat['id']: cat['name'] for cat in anno['categories']}
            cat_counts = {cat_name: 0 for cat_id, cat_name in categories.items()}
            
            # Count annotations per category
            for a in anno['annotations']:
                cat_id = a['category_id'] 
                cat_name = categories.get(cat_id, f"Unknown-{cat_id}")
                cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
            
            # Collect bounding box stats
            boxes = np.array([a['bbox'] for a in anno['annotations']])
            box_widths = boxes[:, 2]
            box_heights = boxes[:, 3]
            
            stats[split] = {
                'num_images': len(anno['images']),
                'num_annotations': len(anno['annotations']),
                'categories': categories,
                'category_counts': cat_counts,
                'avg_box_width': float(np.mean(box_widths)),
                'avg_box_height': float(np.mean(box_heights)),
                'min_box_width': float(np.min(box_widths)),
                'min_box_height': float(np.min(box_heights)),
                'max_box_width': float(np.max(box_widths)),
                'max_box_height': float(np.max(box_heights)),
            }
        
        return stats

if __name__ == "__main__":
    # Example usage
    extractor = DataExtractor("data")
    annotations, image_paths = extractor.extract_all()
    
    # Print dataset statistics
    stats = extractor.get_dataset_stats(annotations)
    print("\nDataset Statistics:")
    for split, split_stats in stats.items():
        print(f"\n{split.upper()} Split:")
        print(f"  Images: {split_stats['num_images']}")
        print(f"  Annotations: {split_stats['num_annotations']}")
        print(f"  Categories: {list(split_stats['categories'].values())}")
        print(f"  Annotations per category: {split_stats['category_counts']}")
        print(f"  Average box dimensions: {split_stats['avg_box_width']:.1f}x{split_stats['avg_box_height']:.1f}")
