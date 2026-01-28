from ultralytics import YOLO
import argparse
import os
try:
    from torch.serialization import add_safe_globals
    import torch.nn as nn
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn import modules as um
    add_safe_globals([
        DetectionModel,
        nn.Sequential,
        getattr(um, "Conv", None),
        getattr(um, "RepConv", None),
        getattr(um, "C2f", None),
        getattr(um, "Bottleneck", None),
        getattr(um, "SPPF", None),
    ])
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 inference on single image or directory")
    parser.add_argument("--weights", required=True, help="Path to trained YOLOv8 weights (.pt)")
    parser.add_argument("--source", required=True, help="Image file or directory for inference")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", default="0", help="Device id, e.g., '0' or 'cpu'")
    parser.add_argument("--project", default="runs/detect", help="Output project directory")
    parser.add_argument("--name", default="predict", help="Run name")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Source path not found: {args.source}")

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
    )


if __name__ == "__main__":
    main()
