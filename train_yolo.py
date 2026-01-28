from ultralytics import YOLO
import argparse
import os
import torch
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
    parser = argparse.ArgumentParser(description="Train YOLOv8 on NEU-DET dataset")
    parser.add_argument("--weights", default="yolov8s.pt", help="Pretrained weights to start from")
    parser.add_argument("--data", default="neu_det.yaml", help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="0", help="Device id, e.g., '0' or 'cpu'")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--project", default="runs/detect", help="Training project directory")
    parser.add_argument("--name", default="train", help="Run name")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    try:
        model = YOLO(args.weights)
    except Exception:
        base = os.path.basename(args.weights)
        name, ext = os.path.splitext(base)
        fallback = name if ext == ".pt" else "yolov8n"
        yaml = f"{fallback}.yaml"
        model = YOLO(yaml)
    dev = args.device
    try:
        if dev != "cpu" and not torch.cuda.is_available():
            dev = "cpu"
    except Exception:
        if dev != "cpu":
            dev = "cpu"
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=dev,
        workers=args.workers,
        project=args.project,
        name=args.name,
        amp=False,
    )


if __name__ == "__main__":
    main()
