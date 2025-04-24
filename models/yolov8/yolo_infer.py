import argparse
import time
from pathlib import Path
from ultralytics import YOLO


def predict_yolov8(
    weights: str,
    source: str,
    device: str = '0',
    conf: float = 0.25,
    iou: float = 0.45,
    save_txt: bool = False,
    save_conf: bool = False,
    project: str = 'data/runs/predict',
    name: str = 'yolov8_infer',
    exist_ok: bool = True,
    verbose: bool = True
):
    """
    Run YOLOv8 inference on images, videos, or streams.

    Args:
        weights (str): Path to the model weights (.pt).
        source (str): Input source (file, directory, webcam index).
        device (str): Device to run inference (e.g., 'cpu' or '0').
        conf (float): Confidence threshold.
        iou (float): IoU threshold for NMS.
        save_txt (bool): Save results as .txt files.
        save_conf (bool): Save confidence scores in labels.
        project (str): Directory to save results.
        name (str): Subdirectory name under project.
        exist_ok (bool): Overwrite existing results directory.
        verbose (bool): Print verbose messages.

    Returns:
        list: Inference result objects.
    """
    if verbose:
        print(f"Loading YOLOv8 model from: {weights}")
    model = YOLO(weights)

    if verbose:
        print(f"Running inference on source: {source}")
    start_time = time.time()
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        device=device,
        save=True,
        save_txt=save_txt,
        save_conf=save_conf,
        project=project,
        name=name,
        exist_ok=exist_ok,
        verbose=verbose
    )
    elapsed = time.time() - start_time
    if verbose:
        print(f"Inference completed in {elapsed:.2f} seconds")
        print(f"Results saved to: {Path(project) / name}")
    return results


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Inference Script')
    parser.add_argument('--weights-dir', type=str, default=str(Path(__file__).parent / 'weights'),
                        help='Directory containing YOLOv8 weights')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n, s, m, l, x)')
    parser.add_argument('--source', type=str, required=True,
                        help='Image/dir/video path or webcam index (e.g., 0)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to run inference on (e.g., "cpu" or "0")')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save detection results as TXT files')
    parser.add_argument('--save-conf', action='store_true',
                        help='Include confidence scores in labels')
    parser.add_argument('--project', type=str, default='data/runs/predict',
                        help='Directory to save predictions')
    parser.add_argument('--name', type=str, default=None,
                        help='Name of the inference run (default: yolov8_<size>_<timestamp>)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    args = parser.parse_args()

    verbose = not args.quiet
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_name = args.name or f'yolov8_{args.model_size}_{timestamp}'
    weights = str(Path(args.weights_dir) / f'yolov8{args.model_size}.pt')

    predict_yolov8(
        weights=weights,
        source=args.source,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.project,
        name=run_name,
        exist_ok=True,
        verbose=verbose
    )


if __name__ == '__main__':
    main() 