from ultralytics import YOLO


def main():
    # 1. Load the pre-trained model you downloaded
    # (Using pre-trained weights is highly recommended for faster convergence)
    model = YOLO("yolo26n.pt")

    # 2. Train the model on your custom dataset
    print("Starting training...")
    model.train(
        data="./input/datasets/vn_license_plate_dataset/data.yaml",  # Path to your YAML file
        epochs=50,  # Number of training loops (start with 50-100)
        imgsz=416,  # Image size (640 is standard)
        batch=8,  # Batch size (adjust based on your GPU RAM)
        device="CPU",  # device=0 for GPU, device='cpu' for CPU
        project="anpr_track",  # Folder name where results will be saved
        name="plate_detector",  # Sub-folder name for this specific run
    )

    print("Training complete! Model saved to anpr_track/plate_detector/weights/best.pt")


if __name__ == "__main__":
    # Ensure this block is used to avoid issues if using multiprocessing (like multiple GPUs)
    main()
