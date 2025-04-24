import argparse
import time
from pathlib import Path
from models.yolov8.yolo_infer import predict_yolov8


def predict_maskrcnn(
    weights: str,
    source: str,
    device: str,
    conf: float,
    iou: float,
    save_txt: bool,
    save_conf: bool,
    project: str,
    name: str,
    exist_ok: bool,
    verbose: bool
):
    """
    Placeholder for Mask R-CNN inference.
    """
    if verbose:
        print("\n=== MASK R-CNN INFERENCE NOT YET IMPLEMENTED ===\n")
    return {}


def predict_ssd(
    weights: str,
    source: str,
    device: str,
    conf: float,
    iou: float,
    save_txt: bool,
    save_conf: bool,
    project: str,
    name: str,
    exist_ok: bool,
    verbose: bool
):
    """
    Placeholder for SSD inference.
    """
    if verbose:
        print("\n=== SSD INFERENCE NOT YET IMPLEMENTED ===\n")
    return {}


def main():
    parser = argparse.ArgumentParser(description="Multi-model inference for object detection")
    parser.add_argument(
        '--source', type=str, required=True,
        help="Input source (image/video/directory or webcam index)"
    )
    parser.add_argument(
        '--runs-dir', type=str, default='data/runs',
        help="Base directory to save inference runs"
    )
    parser.add_argument(
        '--model', type=str, choices=['yolo', 'maskrcnn', 'ssd'],
        action='append', help="Model(s) for inference (can be specified multiple times)"
    )
    parser.add_argument(
        '--yolo-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
        help="YOLOv8 model size (n, s, m, l, x)"
    )
    parser.add_argument(
        '--device', type=str, default='0',
        help="Device to run inference on (e.g., 'cpu' or '0')"
    )
    parser.add_argument(
        '--conf', type=float, default=0.25,
        help="Confidence threshold for detection"
    )
    parser.add_argument(
        '--iou', type=float, default=0.45,
        help="IoU threshold for NMS"
    )
    parser.add_argument(
        '--save-txt', action='store_true',
        help="Save detection results as txt files"
    )
    parser.add_argument(
        '--save-conf', action='store_true',
        help="Include confidence scores in label files"
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help="Suppress verbose output"
    )
    args = parser.parse_args()

    selected_models = args.model if args.model else ['yolo']
    verbose = not args.quiet
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    for model in selected_models:
        run_name = f"{model}_{timestamp}"
        project = str(Path(args.runs_dir) / model)

        if model == 'yolo':
            weights = str(Path('models/yolov8/weights') / f'yolov8{args.yolo_size}.pt')
            predict_yolov8(
                weights=weights,
                source=args.source,
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                save_txt=args.save_txt,
                save_conf=args.save_conf,
                project=project,
                name=run_name,
                exist_ok=True,
                verbose=verbose
            )
        elif model == 'maskrcnn':
            weights = str(Path('models/maskrcnn/weights') / '')
            predict_maskrcnn(
                weights,
                args.source,
                args.device,
                args.conf,
                args.iou,
                args.save_txt,
                args.save_conf,
                project,
                run_name,
                True,
                verbose
            )
        elif model == 'ssd':
            weights = str(Path('models/ssd/weights') / '')
            predict_ssd(
                weights,
                args.source,
                args.device,
                args.conf,
                args.iou,
                args.save_txt,
                args.save_conf,
                project,
                run_name,
                True,
                verbose
            )

if __name__ == '__main__':
    main() 