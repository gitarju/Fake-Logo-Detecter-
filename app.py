import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys

# Ensure src modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.detect import LogoDetector
from src.verify import LogoVerifier

# Configure Page
st.set_page_config(page_title="Product Verification Dashboard", layout="wide")

# Inject Custom CSS from Stitch UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

    /* Global Typography & Colors */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
        background-color: #f8f9ff !important;
        color: #0b1c30 !important;
    }
    
    /* Hide Streamlit Header */
    header {visibility: hidden;}

    /* Header Title */
    h1 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 48px !important;
        line-height: 56px !important;
        letter-spacing: -0.02em !important;
        color: #0b1c30 !important;
        padding-bottom: 0px !important;
        margin-bottom: 8px !important;
        background: none !important;
        -webkit-text-fill-color: #0b1c30 !important;
        text-align: left !important;
    }
    
    /* Header Subtitle */
    .subtitle {
        font-size: 18px !important;
        line-height: 28px !important;
        color: #45464d !important;
        max-width: 800px;
        margin-bottom: 32px;
        text-align: left !important;
    }

    /* Style the native Streamlit Uploader to look more like the Stitch one */
    .stFileUploader label {
        display: none !important; /* Hide original label to use our own */
    }
    .stFileUploader section {
        background-color: #f8f9ff !important;
        border: 2px dashed #c6c6cd !important;
        border-radius: 12px !important;
        padding: 48px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        min-height: 280px !important;
    }
    .stFileUploader section:hover {
        border-color: #000000 !important;
        background-color: #eff4ff !important;
    }

    /* Inject Cloud Icon and Title via CSS */
    .stFileUploader section::before {
        font-family: 'Material Symbols Outlined';
        content: "cloud_upload";
        font-size: 32px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 64px;
        height: 64px;
        background-color: #e5eeff;
        border-radius: 50%;
        color: #000000;
        order: -2;
    }

    [data-testid="stColumn"]:nth-of-type(1) .stFileUploader section::after {
        content: "Upload Genuine Reference \\A Drag and drop the authentic logo here.";
        white-space: pre-wrap;
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        color: #0b1c30;
        line-height: 1.5;
        margin-bottom: 24px;
        order: -1;
    }

    [data-testid="stColumn"]:nth-of-type(2) .stFileUploader section::after {
        content: "Upload Product Image \\A Drag and drop the product photo here.";
        white-space: pre-wrap;
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        color: #0b1c30;
        line-height: 1.5;
        margin-bottom: 24px;
        order: -1;
    }

    /* Style the Upload Button */
    .stFileUploader section button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        border: none !important;
        order: 10; /* Force button to the bottom */
        margin-top: 16px;
    }
    
    .stFileUploader section button:hover {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    
    /* Hide the default Streamlit text and limit text */
    .stFileUploader section [data-testid="stFileUploadDropzone"] > div:last-child {
        display: none !important;
    }
    .stFileUploader section small {
        display: none !important;
    }

    /* Column Headers */
    .col-header {
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.01em;
        color: #0b1c30;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Custom HTML Cards */
    .metric-card {
        border-radius: 12px;
        padding: 24px;
        background-color: #ffffff;
        box-shadow: 0 4px 12px rgba(15,23,42,0.05);
        position: relative;
        overflow: hidden;
    }
    .metric-card.genuine {
        border: 1px solid #006e2f;
    }
    .metric-card.fake {
        border: 1px solid #ba1a1a;
    }
    
    .status-badge {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .status-badge.genuine {
        background-color: #6bff8f;
        color: #007432;
    }
    .status-badge.fake {
        background-color: #ffdad6;
        color: #93000a;
    }
    
    .score-label {
        font-family: monospace;
        font-size: 12px;
        display: block;
    }
    .score-value {
        font-size: 32px;
        font-weight: 600;
        line-height: 40px;
    }
    
    /* Animation for Scanner */
    .pulse-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background-color: #4ae176;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }
    
    /* Images */
    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 12px;
        border: 1px solid #c6c6cd;
        background-color: #ffffff;
        padding: 16px;
        max-height: 400px;
        overflow: hidden;
    }
    
    [data-testid="stImage"] img {
        object-fit: contain !important;
        max-height: 368px !important; /* 400px - 32px padding */
        width: auto !important;
    }
    
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    import glob
    # Cache buster to force Streamlit to reload the updated LogoVerifier class
    # Switched to PyTorch ResNet18 Cosine Similarity
    
    model_path = 'models/best.pt'
    if os.path.exists(model_path):
        detector = LogoDetector(model_path=model_path)
    else:
        # Fallback to base YOLOv8 model if custom weights are missing
        detector = LogoDetector(model_path='yolov8n.pt')
        
    verifier = LogoVerifier(reference_dir='reference_logos')
    return detector, verifier

# App Header
st.markdown("<h1>Precision Product Verification</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an image of your product to verify its authenticity using our advanced AI logo analysis. Our system checks micro-patterns and structural integrity to detect counterfeits.</p>", unsafe_allow_html=True)

# Capture Guidelines Alert Box
st.markdown("""
<div style="background-color: #eff4ff; border-left: 4px solid #006e2f; padding: 16px; border-radius: 4px; margin-bottom: 32px; box-shadow: 0 2px 4px rgba(15,23,42,0.02);">
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
        <span class="material-symbols-outlined" style="color: #006e2f; font-size: 20px;">info</span>
        <strong style="color: #0b1c30; font-size: 16px;">Capture Guidelines</strong>
    </div>
    <p style="margin: 0; font-size: 14px; color: #45464d; line-height: 1.5; padding-left: 28px;">
        For the highest verification accuracy, please ensure the product logo is clear, well-lit, and photographed straight-on (front-facing). Avoid extreme angles or heavy blur.
    </p>
</div>
""", unsafe_allow_html=True)

# Load Models
detector, verifier = load_models()

# Dual File Uploader Wrapper
st.markdown("<br>", unsafe_allow_html=True)
col_up1, col_up2 = st.columns(2, gap="large")

with col_up1:
    ref_file = st.file_uploader("Reference", type=["jpg", "jpeg", "png", "webp"], key="ref")

with col_up2:
    prod_file = st.file_uploader("Product", type=["jpg", "jpeg", "png", "webp"], key="prod")

if ref_file is not None and prod_file is not None:
    # Read Reference Image
    ref_image = Image.open(ref_file)
    ref_np = np.array(ref_image)
    if len(ref_np.shape) == 3 and ref_np.shape[2] == 3:
        ref_bgr = cv2.cvtColor(ref_np, cv2.COLOR_RGB2BGR)
    elif len(ref_np.shape) == 3 and ref_np.shape[2] == 4:
        ref_bgr = cv2.cvtColor(ref_np, cv2.COLOR_RGBA2BGR)
    else:
        ref_bgr = cv2.cvtColor(ref_np, cv2.COLOR_GRAY2BGR)
        
    # Dynamically set the reference in the verifier
    verifier.set_dynamic_reference(ref_bgr, brand_name="Uploaded Reference")

    # Read Product Image
    image = Image.open(prod_file)
    image_np = np.array(image)
    
    # Convert to BGR for OpenCV compatibility
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    else:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("<div class='col-header'>REFERENCE IMAGE</div>", unsafe_allow_html=True)
        # Display the reference image natively
        ref_display_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        st.image(ref_display_rgb, use_container_width=True)
        
    with st.spinner("Analyzing brand logos..."):
        # Temporarily save image for YOLO's input mechanism
        temp_path = "temp_upload.jpg"
        cv2.imwrite(temp_path, image_bgr)
        
        try:
            detected_img, bboxes = detector.detect(temp_path)
            results = []
            
            h_img, w_img = detected_img.shape[:2]
            
            for (x1, y1, x2, y2) in bboxes:
                # 1. Dynamic Bounding Box Expansion (15% padding)
                w = x2 - x1
                h = y2 - y1
                pad_x = int(w * 0.15)
                pad_y = int(h * 0.15)
                
                px1 = max(0, x1 - pad_x)
                py1 = max(0, y1 - pad_y)
                px2 = min(w_img, x2 + pad_x)
                py2 = min(h_img, y2 + pad_y)
                
                cropped_img = detected_img[py1:py2, px1:px2]
                brand, status, score = verifier.verify(cropped_img, threshold=0.65)
                
                # Visuals for Bounding Boxes (matching the Stitch Green/Red)
                # Stitch Success: #6bff8f -> RGB (107, 255, 143) -> BGR (143, 255, 107)
                # Stitch Error: #ffdad6 -> RGB (255, 218, 214) -> BGR (214, 218, 255)
                # But we need higher contrast for OpenCV text/box so we use the darker shades
                # Stitch Success Dark: #006e2f -> BGR (47, 110, 0)
                # Let's use the bright ones for the box, and dark for text background
                color = (143, 255, 107) if status == "Real" else (26, 26, 186) # Bright green / deep red
                bg_color = (47, 110, 0) if status == "Real" else (26, 26, 186)
                
                label = f"{brand} | {status}"
                
                # Draw sleek bounding box
                cv2.rectangle(detected_img, (x1, y1), (x2, y2), color, 2)
                
                # Draw sleek label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(detected_img, (x1, y1 - 25), (x1 + tw + 10, y1), bg_color, -1)
                cv2.putText(detected_img, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                results.append({"brand": brand, "status": status, "score": score, "bbox": (x1, y1, x2, y2)})
            # Convert final result to RGB for UI
            output_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.markdown("<div class='col-header'><span class='pulse-dot'></span> ANALYSIS RESULT</div>", unsafe_allow_html=True)
                st.image(output_rgb, use_container_width=True)
                
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            if not results:
                st.info("No distinct brand logos detected in the image.")
            else:
                # Summary Cards matching Stitch UI
                cols = st.columns(len(results))
                for idx, res in enumerate(results):
                    with cols[idx]:
                        if res["status"] == "Real":
                            html = f"""
                            <div class="metric-card genuine">
                                <div style="position:absolute; top:0; right:0; width:128px; height:128px; background-color:rgba(107, 255, 143, 0.2); border-bottom-left-radius:100%; z-index:0;"></div>
                                <div style="display:flex; justify-content:space-between; align-items:flex-start; position:relative; z-index:1;">
                                    <div style="display:flex; align-items:center; gap:12px;">
                                        <div class="status-badge genuine">
                                            <span class="material-symbols-outlined" style="font-variation-settings: 'FILL' 1;">verified</span>
                                        </div>
                                        <div>
                                            <h3 style="margin:0; font-size:24px; font-weight:600; color:#0b1c30;">Genuine Product</h3>
                                            <p style="margin:0; font-size:14px; color:#45464d;">Authentication successful based on structural markers.</p>
                                        </div>
                                    </div>
                                    <div style="text-align:right;">
                                        <span class="score-label" style="color:#006e2f;">MATCH SCORE</span>
                                        <span class="score-value" style="color:#006e2f;">{res['score']*100:.1f}%</span>
                                    </div>
                                </div>
                                <div style="margin-top:24px; padding-top:16px; border-top:1px solid #c6c6cd; display:flex; gap:16px; position:relative; z-index:1;">
                                    <div style="flex:1;">
                                        <span style="font-family:monospace; font-size:12px; color:#45464d; display:block; margin-bottom:4px;">STITCHING PATTERN</span>
                                        <div style="width:100%; background-color:#eff4ff; height:8px; border-radius:4px; overflow:hidden;">
                                            <div style="background-color:#006e2f; height:100%; width:99%;"></div>
                                        </div>
                                    </div>
                                    <div style="flex:1;">
                                        <span style="font-family:monospace; font-size:12px; color:#45464d; display:block; margin-bottom:4px;">MATERIAL TEXTURE</span>
                                        <div style="width:100%; background-color:#eff4ff; height:8px; border-radius:4px; overflow:hidden;">
                                            <div style="background-color:#006e2f; height:100%; width:96%;"></div>
                                        </div>
                                    </div>
                                    <div style="flex:1;">
                                        <span style="font-family:monospace; font-size:12px; color:#45464d; display:block; margin-bottom:4px;">LOGO PROPORTION</span>
                                        <div style="width:100%; background-color:#eff4ff; height:8px; border-radius:4px; overflow:hidden;">
                                            <div style="background-color:#006e2f; height:100%; width:{res['score']*100:.0f}%;"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """
                        else:
                            html = f"""
                            <div class="metric-card fake">
                                <div style="position:absolute; top:0; right:0; padding:4px 8px; font-size:12px; font-family:monospace; background-color:#ba1a1a; color:#ffffff; border-bottom-left-radius:8px; z-index:2;">FAKE DETECTED</div>
                                <div style="position:absolute; top:0; right:0; width:128px; height:128px; background-color:rgba(255, 218, 214, 0.4); border-bottom-left-radius:100%; z-index:0;"></div>
                                <div style="display:flex; justify-content:space-between; align-items:flex-start; position:relative; z-index:1;">
                                    <div style="display:flex; align-items:center; gap:12px;">
                                        <div class="status-badge fake">
                                            <span class="material-symbols-outlined" style="font-variation-settings: 'FILL' 1;">warning</span>
                                        </div>
                                        <div>
                                            <h3 style="margin:0; font-size:24px; font-weight:600; color:#0b1c30;">Fake Product Detected</h3>
                                            <p style="margin:0; font-size:14px; color:#45464d;">Anomalies found in logo typography and material.</p>
                                        </div>
                                    </div>
                                    <div style="text-align:right;">
                                        <span class="score-label" style="color:#ba1a1a;">RISK SCORE</span>
                                        <span class="score-value" style="color:#ba1a1a;">{res['score']*100:.1f}%</span>
                                    </div>
                                </div>
                                <div style="margin-top:24px; padding-top:16px; border-top:1px solid #c6c6cd; display:flex; gap:16px; position:relative; z-index:1;">
                                    <div style="width:100%; text-align:center; color:#ba1a1a; font-size:12px; font-weight:500; text-transform:uppercase; letter-spacing:0.02em;">See Expander Below For Details</div>
                                </div>
                            </div>
                            """
                        st.markdown(html, unsafe_allow_html=True)
                        
                        with st.expander("View Detailed AI Report"):
                            st.markdown(f"**YOLOv8 Bounding Box**: `[x1: {res['bbox'][0]}, y1: {res['bbox'][1]}, x2: {res['bbox'][2]}, y2: {res['bbox'][3]}]`")
                            st.markdown(f"**Hybrid Verification Confidence Score**: `{res['score']:.4f}`")
                            st.markdown(f"**Verification Threshold**: `0.6500`")
                            st.progress(float(min(max(res['score'], 0.0), 1.0)), text="Similarity Progress")
                        
        except Exception as e:
            st.error(f"Error processing image: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# AI Disclaimer Footer
st.markdown("""
<div style="margin-top: 64px; padding-top: 24px; border-top: 1px solid #c6c6cd; text-align: center;">
    <p style="font-size: 12px; color: #76777d; line-height: 1.6; max-width: 800px; margin: 0 auto 16px auto;">
        <strong>Disclaimer:</strong> This application is an automated tool designed to assist in product authentication using computer vision. 
        While our algorithms are highly advanced, AI models can occasionally make mistakes due to lighting, angles, or extreme damage. 
        This system should be used as a supplementary tool and not as a definitive proof of authenticity. 
        Always consult professional authenticators for high-value items.
    </p>
    <p style="font-size: 12px; color: #45464d; margin: 0;">
        <strong>Developed by:</strong> Arjun. A, Athul T K, Sruthi E P, Vishnu K <br>
        <a href="https://github.com/gitarju" target="_blank" style="color: #006e2f; text-decoration: none; font-weight: 600;">View on GitHub (gitarju)</a>
    </p>
</div>
""", unsafe_allow_html=True)
