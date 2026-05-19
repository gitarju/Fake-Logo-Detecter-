import cv2
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

class LogoVerifier:
    def __init__(self, reference_dir='reference_logos'):
        """
        Initializes the LogoVerifier with a pre-trained ResNet18 model for feature extraction.
        """
        self.reference_dir = reference_dir
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained ResNet18 and remove the classification head
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.reference_data = self._load_references()

    def _get_embedding(self, image_bgr):
        """
        Passes a BGR image through ResNet18 to get a 512D feature embedding.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(input_tensor)
        
        # Flatten the embedding: shape will be (512,)
        return embedding.squeeze()

    def _load_references(self):
        """
        Loads all reference logos and precomputes their embeddings.
        """
        reference_data = {}
        if not os.path.exists(self.reference_dir):
            return reference_data

        image_paths = glob.glob(os.path.join(self.reference_dir, '*.[jp][pn]*')) # match jpg, png, jpeg
        
        for path in image_paths:
            brand_name = os.path.splitext(os.path.basename(path))[0]
            img = cv2.imread(path)
            if img is not None:
                emb = self._get_embedding(img)
                reference_data[brand_name] = {'image': img, 'embedding': emb}
        
        return reference_data

    def set_dynamic_reference(self, image, brand_name="Uploaded Reference"):
        """
        Sets a single dynamic reference image, computing its embedding immediately.
        Expects a BGR image (numpy array).
        """
        emb = self._get_embedding(image)
        self.reference_data = {brand_name: {'image': image, 'embedding': emb}}

    def _apply_clahe(self, img):
        """
        Applies Contrast Limited Adaptive Histogram Equalization to normalize lighting.
        """
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except:
            return img

    def _apply_grabcut(self, img):
        """
        Uses OpenCV's GrabCut to automatically segment the foreground (logo) 
        from the background (e.g. car grille).
        """
        try:
            h, w = img.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            # Define a rectangle slightly smaller than the image (assuming logo is centered)
            rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            
            mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
            return mask2 * 255
        except:
            # Fallback to circular mask if GrabCut fails
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (w//2, h//2), int(min(h, w) / 2 * 0.7), 255, -1)
            return mask

    def _apply_contour_mask(self, img):
        """
        Uses Hough Circle Transform and contour finding to isolate the logo.
        HoughCircles is highly resistant to background stripes/noise.
        """
        try:
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # 1. Try Hough Circles first (Perfect for BMW)
            # param2 is the accumulator threshold. Lower = more false circles. 30 is a good strict baseline.
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h/2,
                param1=50, param2=30, minRadius=int(min(h,w)*0.2), maxRadius=int(min(h,w)*0.6)
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                # Pick the most prominent circle
                i = circles[0, 0]
                # Draw the circle mask, shrinking radius by 5% to avoid background bleed
                cv2.circle(mask, (i[0], i[1]), int(i[2] * 0.95), 255, -1)
                return mask
                
            # 2. Fallback to Contours for non-circular but solid logos
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > (h * w) * 0.15:
                    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                    circle_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.circle(circle_mask, (int(x), int(y)), int(radius * 0.95), 255, -1)
                    return cv2.bitwise_and(mask, circle_mask)
            
            # 3. Fallback to GrabCut
            return self._apply_grabcut(img)
            
        except Exception:
            return self._apply_grabcut(img)

    def _compute_color_similarity(self, img1, img2):
        """
        Computes color histogram similarity using Bhattacharyya distance.
        Uses GrabCut to segment the logo and ignore background colors.
        """
        try:
            # Convert to HSV for robust color representation
            hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
            
            # Use Contour masking (falls back to GrabCut) to perfectly separate background
            mask1 = self._apply_contour_mask(img1)
            mask2 = self._apply_contour_mask(img2)
            
            # Calculate histograms using the mask
            hist1 = cv2.calcHist([hsv1], [0, 1], mask1, [50, 60], [0, 180, 0, 256])
            hist2 = cv2.calcHist([hsv2], [0, 1], mask2, [50, 60], [0, 180, 0, 256])
            
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            # Compare using Bhattacharyya distance (0 = identical, 1 = completely different)
            distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            return max(0.0, 1.0 - distance)
        except Exception as e:
            return 0.0

    def _compute_structural_similarity(self, img1, img2):
        """
        Computes a structural similarity score using SIFT and RANSAC geometric verification.
        Returns the score and the Homography matrix M for later warping.
        """
        try:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)
            
            if des1 is None or len(des1) == 0 or des2 is None or len(des2) == 0:
                return 0.0, None
                
            index_params = dict(algorithm=1, trees=5) 
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(des1, des2, k=2)
            
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            M_out = None
            # 3. Geometric Consistency Verification (RANSAC)
            if len(good_matches) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography and count geometrically valid inliers
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                M_out = M
                
                if mask is not None:
                    inliers = np.sum(mask)
                    # 10 inliers is an excellent geometric match
                    score = min(1.0, inliers / 10.0)
                else:
                    score = min(1.0, len(good_matches) / 20.0)
            else:
                score = min(1.0, len(good_matches) / 20.0)
                
            return float(score), M_out
        except Exception as e:
            print(f"SIFT/RANSAC error: {e}")
            return 0.0, None

    def _compute_warped_ssim(self, img_crop, img_ref, M):
        """
        Warps the cropped photo onto the reference image using the RANSAC homography matrix.
        Then computes the Structural Similarity Index (SSIM) to find micro-defects.
        """
        try:
            if M is None:
                return 0.0
                
            h, w = img_ref.shape[:2]
            warped_crop = cv2.warpPerspective(img_crop, M, (w, h))
            
            gray_crop = cv2.cvtColor(warped_crop, cv2.COLOR_BGR2GRAY)
            gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
            
            # Pixel-by-pixel normalized cross correlation (acts as SSIM when perfectly aligned)
            score = cv2.matchTemplate(gray_crop, gray_ref, cv2.TM_CCOEFF_NORMED)[0][0]
            return max(0.0, float(score))
        except Exception:
            return 0.0

    def verify(self, cropped_image, threshold=0.65):
        """
        Compares a cropped image to all reference logos using an advanced ensemble metric.
        """
        if not self.reference_data:
            return "Unknown", "Fake (No References)", 0.0

        # Get embedding for cropped image (Raw)
        cropped_emb = self._get_embedding(cropped_image)
        
        # 2. CLAHE Image Normalization (Improves color/structural feature extraction)
        clahe_crop = self._apply_clahe(cropped_image)

        best_match_brand = "Unknown"
        best_match_score = -1.0

        for brand, data in self.reference_data.items():
            ref_emb = data['embedding']
            ref_img = data['image']
            
            # Compute Deep Feature Cosine Similarity
            resnet_sim = F.cosine_similarity(cropped_emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()
            
            # Compute Pixel-Level Structural Similarity (RANSAC)
            ssim_score, M = self._compute_structural_similarity(clahe_crop, ref_img)
            
            # Compute Micro-Defect SSIM penalty if RANSAC homography was found
            micro_ssim_score = self._compute_warped_ssim(clahe_crop, ref_img, M)
            
            # 4. Compute Color Similarity
            color_score = self._compute_color_similarity(clahe_crop, ref_img)
            
            # Ensemble Score
            # Base structural identity is the best of deep features or exact geometric matches
            base_structural = max(resnet_sim, ssim_score)
            
            # Critical Fix: Color as a Gatekeeper Multiplier
            color_multiplier = 0.6 + (0.4 * color_score)
            
            # New Fix: Micro-Defect Multiplier
            # If RANSAC found a geometric match but the pixel-level SSIM is poor, it's a high-quality fake.
            if M is not None:
                defect_multiplier = 0.7 + (0.3 * micro_ssim_score)
            else:
                defect_multiplier = 1.0
                
            hybrid_score = base_structural * color_multiplier * defect_multiplier
            
            if hybrid_score > best_match_score:
                best_match_score = hybrid_score
                best_match_brand = brand

        if best_match_score >= threshold:
            status = "Real"
        else:
            status = "Fake"
            
        return best_match_brand, status, best_match_score
