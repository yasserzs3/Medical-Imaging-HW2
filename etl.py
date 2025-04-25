import os
import argparse
from pathlib import Path
import time
from etl.extract import DataExtractor
from etl.transform import DataTransformer
from etl.load import DataLoader

def run_etl_pipeline(
    data_dir='data/raw',
    output_dir='data/processed',
    models=None,
    apply_augmentation=False,
    apply_normalization=False,
    verbose=True
):
    """
    Run the ETL pipeline optimized for YOLOv8 format.
    
    Args:
        data_dir (str): Input data directory
        output_dir (str): Base output directory for processed data
        models (list): List of models to generate data for (yolo, maskrcnn, ssd)
        apply_augmentation (bool): Whether to apply data augmentation
        apply_normalization (bool): Whether to apply normalization to images
        verbose (bool): Whether to print verbose output
    
    Returns:
        tuple: (output directories, stats)
    """
    start_time = time.time()
    
    # Default to empty list if None
    if models is None:
        models = []
        
    # Create directories if they don't exist
    Path(data_dir).mkdir(exist_ok=True, parents=True)
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Define model-specific paths
    model_dirs = {
        'yolo': Path(output_dir) / 'yolo',
        'maskrcnn': Path(output_dir) / 'maskrcnn',
        'ssd': Path(output_dir) / 'ssd'
    }
    
    # Create directories for requested models
    for model in models:
        model_dirs[model].mkdir(exist_ok=True, parents=True)
    
    if verbose:
        print("\n=== STARTING ETL PIPELINE ===\n")
        print(f"Input directory: {data_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Models to generate: {', '.join(models) if models else 'None'}")
        print(f"Apply augmentation: {apply_augmentation}")
        print(f"Apply normalization: {apply_normalization}")
    
    # 1. EXTRACT - Use DataExtractor to extract data
    if verbose:
        print("\n=== EXTRACTION PHASE ===\n")
        
    extractor = DataExtractor(data_dir)
    annotations, image_paths = extractor.extract_all()
    
    stats = {}
    stats['extraction'] = extractor.get_dataset_stats(annotations)
    
    if verbose:
        print("\nExtraction Statistics:")
        for split, split_stats in stats['extraction'].items():
            if not split_stats:
                continue
            print(f"\n{split.upper()} Split:")
            print(f"  Images: {split_stats['num_images']}")
            print(f"  Annotations: {split_stats['num_annotations']}")
            print(f"  Categories: {list(split_stats['categories'].values())}")
            print(f"  Annotations per category: {split_stats['category_counts']}")
            print(f"  Average box dimensions: {split_stats['avg_box_width']:.1f}x{split_stats['avg_box_height']:.1f}")
    
    # Process for each requested model format
    for model in models:
        if model == 'yolo':
            if verbose:
                print(f"\n=== PROCESSING FOR YOLO FORMAT ===\n")
            
            # Create YOLO directories
            yolo_dir = model_dirs['yolo']
            
            # Process data directly to YOLO format
            transformer = DataTransformer(apply_normalization=apply_normalization)
            transformer.convert_to_yolov8_format_direct(
                annotations, 
                image_paths, 
                str(yolo_dir), 
                apply_augmentation=apply_augmentation
            )
            
            if verbose:
                print(f"YOLOv8 dataset created at: {yolo_dir}")
                print(f"Dataset structure:")
                print(f"  - {yolo_dir}/images/train/")
                print(f"  - {yolo_dir}/images/val/")
                print(f"  - {yolo_dir}/images/test/")
                print(f"  - {yolo_dir}/labels/train/")
                print(f"  - {yolo_dir}/labels/val/")
                print(f"  - {yolo_dir}/labels/test/")
                print(f"  - {yolo_dir}/dataset.yaml")
            
        elif model == 'maskrcnn':
            if verbose:
                print(f"\n=== PROCESSING FOR MASK R-CNN FORMAT ===\n")
            
            # Placeholder for Mask R-CNN processing
            maskrcnn_dir = model_dirs['maskrcnn']
            maskrcnn_dir.mkdir(exist_ok=True, parents=True)
            
            # Create a README file as a placeholder
            with open(maskrcnn_dir / "README.md", 'w') as f:
                f.write("# Mask R-CNN Dataset\n\n")
                f.write("Placeholder for Mask R-CNN format dataset.\n")
                f.write("This will be implemented in a future update.\n")
            
            if verbose:
                print(f"Created placeholder for Mask R-CNN at: {maskrcnn_dir}")
                
        elif model == 'ssd':
            if verbose:
                print(f"\n=== PROCESSING FOR SSD FORMAT ===\n")
            
            # Process data to SSD format
            transformer = DataTransformer(output_dir=str(model_dirs['ssd']), apply_normalization=apply_normalization)
            transformer.convert_to_ssd_format(
                annotations,
                image_paths,
                str(model_dirs['ssd']),
                apply_augmentation=apply_augmentation
            )
            
            if verbose:
                print(f"SSD dataset created at: {model_dirs['ssd']}")
                print(f"Dataset structure:")
                print(f"  - {model_dirs['ssd']}/train/")
                print(f"  - {model_dirs['ssd']}/val/")
                print(f"  - {model_dirs['ssd']}/test/")
                print(f"  - {model_dirs['ssd']}/dataset.yaml")
    
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\n=== ETL PIPELINE COMPLETED IN {elapsed_time:.2f} SECONDS ===\n")
    
    # Prepare output directories dictionary
    output_dirs = {}
    
    # Add model-specific paths
    for model in models:
        output_dirs[model] = str(model_dirs[model])
    
    return output_dirs, stats

def main():
    parser = argparse.ArgumentParser(description='ETL pipeline for preparing medical imaging data for object detection models')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Input data directory')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory for processed data')
    parser.add_argument('--model', type=str, choices=['yolo', 'maskrcnn', 'ssd'], action='append', 
                        help='Model format to generate (can be specified multiple times)')
    parser.add_argument('--apply-augmentation', action='store_true', help='Apply data augmentation transformations')
    parser.add_argument('--apply-normalization', action='store_true', help='Apply normalization to images (makes images look weird when visualized)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Default to yolo if no model selected
    models = args.model if args.model else ['yolo']
    
    run_etl_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models=models,
        apply_augmentation=args.apply_augmentation,
        apply_normalization=args.apply_normalization,
        verbose=not args.quiet
    )

if __name__ == "__main__":
    main() 