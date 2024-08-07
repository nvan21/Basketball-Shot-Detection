from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8l.pt")
    results = model.train(
        data="data.yaml",
        optimizer="Adam",
        epochs=50,
        name="runs/detect/yolov8l_results",
        device=0,
        save=True,
    )
