# Precision Product Verification System

## Overview
This project is a computer vision pipeline designed to verify the authenticity of physical products by analyzing their brand logos. It compares a product photo against a verified digital reference logo to detect counterfeits based on structural geometry, color accuracy, and micro-defects.

## Data Flow and Architecture

The system operates in a sequential, four-step pipeline to ensure high accuracy without requiring deep learning model retraining for every new logo.

### 1. Localization (YOLOv8)
When a product image is uploaded, a custom-trained YOLOv8 object detection model scans the image to locate potential brand logos. It extracts a bounding box around the logo, which is padded by 15 percent to ensure the outer edges are not cut off.

### 2. Lighting Normalization (CLAHE)
The cropped product photo often suffers from harsh lighting, glare, or poor contrast. The system applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to normalize the lighting, ensuring shadows and reflections do not interfere with feature matching.

### 3. Geometric and Structural Verification (ResNet18 and RANSAC)
The normalized crop is passed through two structural checks:
* Deep Features: A ResNet18 model extracts high-level semantic embeddings to verify the general shape and layout of the logo using cosine similarity.
* Geometric Alignment: SIFT keypoints are extracted and matched using a RANSAC algorithm. This calculates a Homography Matrix, verifying that the physical 2D shape of the product logo mathematically matches the reference logo, ignoring all background noise.

### 4. Micro-Defect and Color Authentication
* Color Segmentation (GrabCut): OpenCV's GrabCut algorithm mathematically segments the logo from the background. A Bhattacharyya distance check is then performed on the HSV color histograms to ensure the brand colors are exact.
* Micro-Defect Scan (Warped SSIM): Using the Homography Matrix from step 3, the product photo is warped and flattened to perfectly overlay the reference logo. A pixel-by-pixel Structural Similarity Index (SSIM) scan is performed using normalized cross-correlation to catch microscopic counterfeit defects, such as incorrect font weights or border thicknesses.

If the product passes all checks, it is classified as Genuine. If the colors are wrong or micro-defects are found, the score is heavily penalized and it is classified as Fake.

## Instructions to Run

### Prerequisites
Ensure you have Python 3.9 or higher installed. Install the required dependencies:

```bash
pip install streamlit opencv-python ultralytics torch torchvision numpy
```

### Running the Web Interface
The primary way to interact with the system is through the Streamlit web dashboard.

1. Ensure you have genuine reference logos placed inside the `reference_logos/` directory (or use the web interface to upload a reference).
2. Ensure your custom YOLOv8 model weights are located at `models/best.pt` (the system will default to `yolov8n.pt` if missing).
3. Run the Streamlit application:

```bash
streamlit run app.py
```
If you encounter a launcher error on Windows, run it using the Python module flag:
```bash
python -m streamlit run app.py
```

### Running the Command Line Interface
You can also process images directly via the command line using `main.py`:

```bash
python main.py --image path/to/product_photo.jpg --output result.jpg
```

## Team Members
This project was developed by:
* Arjun. A
* Athul T K
* Sruthi E P
* Vishnu K

Find more projects and updates on [Arjun's GitHub](https://github.com/gitarju).
