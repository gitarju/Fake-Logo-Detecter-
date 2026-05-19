from ultralytics import YOLO
import os

def main():
    # Path to the data.yaml file
    data_yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolo_dataset', 'data.yaml'))

    print(f"[INFO] Initializing YOLOv8 nano model...")
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    print(f"[INFO] Starting training on {data_yaml_path}...")
    # Train the model
    # We will train for 10 epochs for this prototype to show the capability
    # In a real-world scenario, you would train for 50-100 epochs.
    results = model.train(
        data=data_yaml_path,
        epochs=10,
        imgsz=640,
        project='models',
        name='logo_detector'
    )

    print(f"[INFO] Training completed. Model weights saved to models/logo_detector/weights/best.pt")

if __name__ == '__main__':
    main()
