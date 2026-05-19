import cv2
import os
from ultralytics import YOLO

class LogoDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initializes the YOLO model for logo detection.
        """
        if not os.path.exists(model_path) and model_path != 'yolov8n.pt':
            print(f"[WARNING] Model {model_path} not found. Falling back to yolov8n.pt")
            model_path = 'yolov8n.pt'
            
        self.model = YOLO(model_path)

    def detect(self, image_path):
        """
        Detect objects in the given image.
        Returns the original image, and a list of bounding boxes (x1, y1, x2, y2).
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        results = self.model(image)
        
        bboxes = []
        # Extract bounding boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # get bounding box coordinates in (left, top, right, bottom) format
                x1, y1, x2, y2 = box.xyxy[0]
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))
                
        # If YOLO fails to detect any objects, we can fall back to returning the whole image 
        # as a single bounding box so OpenCV can still try feature matching on the whole image.
        if not bboxes:
            h, w, _ = image.shape
            bboxes.append((0, 0, w, h))

        return image, bboxes
