import cv2
import numpy as np
import os

class MatchVisualizer:
    def __init__(self):
        # Hardcoded file paths
        self.image0_path = r"C:\Users\zarif\OneDrive\Documents\Photogrammetry\Converse2Normal\20241126_221855.jpg"
        self.image1_path = r"C:\Users\zarif\OneDrive\Documents\Photogrammetry\Converse2Normal\20241126_221853.jpg"
        self.feat0_path = r"C:\Users\zarif\OneDrive\Documents\Photogrammetry\MeshroomCache\SuperPointFeatureExtraction\cf3c74960a15930e8bbb4bc4fd9f50134dfe945c\1816727059.dspsift.feat"
        self.feat1_path = r"C:\Users\zarif\OneDrive\Documents\Photogrammetry\MeshroomCache\SuperPointFeatureExtraction\cf3c74960a15930e8bbb4bc4fd9f50134dfe945c\492971557.dspsift.feat"
        self.matches_path = r"C:\Users\zarif\OneDrive\Documents\Photogrammetry\MeshroomCache\SuperGlueFeatureMatching\c1e2462ae54031bc3224b2a0bb8629ed7e041ee4\0.matches.txt"
        
        # Transformation states
        self.rotation0 = 0  # 0, 90, 180, 270 degrees
        self.rotation1 = 0
        self.flip0 = 0  # 0=none, 1=horizontal, -1=vertical
        self.flip1 = 0
        
        # Viewport control
        self.scale = 0.5  # Start zoomed out
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.last_x, self.last_y = 0, 0
        
        # Load all data
        self.load_data()
        
    def load_data(self):
        # Load images
        self.orig_image0 = cv2.imread(self.image0_path)
        self.orig_image1 = cv2.imread(self.image1_path)
        
        if self.orig_image0 is None or self.orig_image1 is None:
            raise ValueError("Could not load one or both images")
        
        # Load keypoints
        self.orig_keypoints0 = self.load_keypoints(self.feat0_path)
        self.orig_keypoints1 = self.load_keypoints(self.feat1_path)
        
        # Load matches
        self.matches = self.load_matches(self.matches_path)
        
    def load_keypoints(self, feat_path):
        keypoints = []
        with open(feat_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    keypoints.append(cv2.KeyPoint(x, y, 5))  # size=5
        return keypoints
    
    def load_matches(self, matches_path):
        matches = []
        with open(matches_path, 'r') as f:
            lines = f.readlines()
            if len(lines) < 4:
                raise ValueError("Invalid matches file format")
            
            # Skip header lines (first 3 lines)
            for line in lines[3:]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    idx0, idx1 = int(parts[0]), int(parts[1])
                    matches.append(cv2.DMatch(idx0, idx1, 0))  # distance=0
        return matches
    
    def transform_image(self, img, rotation, flip):
        # Apply flip first
        if flip == 1:
            img = cv2.flip(img, 1)
        elif flip == -1:
            img = cv2.flip(img, 0)
            
        # Then apply rotation
        if rotation == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif rotation == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        return img
    
    def get_transformed_images(self):
        img0 = self.transform_image(self.orig_image0.copy(), self.rotation0, self.flip0)
        img1 = self.transform_image(self.orig_image1.copy(), self.rotation1, self.flip1)
        return img0, img1
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_x, self.last_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.last_x
            dy = y - self.last_y
            self.offset_x += dx
            self.offset_y += dy
            self.last_x, self.last_y = x, y
        elif event == cv2.EVENT_MOUSEWHEEL:
            zoom_factor = 1.1
            if flags < 0:  # Scroll down
                self.scale /= zoom_factor
            else:  # Scroll up
                self.scale *= zoom_factor
    
    def visualize_matches(self):
        cv2.namedWindow("Feature Matches", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Feature Matches", self.mouse_callback)
        
        while True:
            # Get transformed images
            img0, img1 = self.get_transformed_images()
            
            # Scale images
            h0, w0 = img0.shape[:2]
            h1, w1 = img1.shape[:2]
            scaled_w0, scaled_h0 = int(w0 * self.scale), int(h0 * self.scale)
            scaled_w1, scaled_h1 = int(w1 * self.scale), int(h1 * self.scale)
            
            scaled_img0 = cv2.resize(img0, (scaled_w0, scaled_h0))
            scaled_img1 = cv2.resize(img1, (scaled_w1, scaled_h1))
            
            # Create canvas
            canvas_h = max(scaled_h0, scaled_h1)
            canvas_w = scaled_w0 + scaled_w1
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            
            # Place images on canvas
            canvas[:scaled_h0, :scaled_w0] = scaled_img0
            canvas[:scaled_h1, scaled_w0:scaled_w0+scaled_w1] = scaled_img1
            
            # Apply viewport offset
            view_h, view_w = 800, 1200  # Viewport size
            viewport = np.zeros((view_h, view_w, 3), dtype=np.uint8)
            
            # Calculate visible region
            x1 = max(0, -self.offset_x)
            y1 = max(0, -self.offset_y)
            x2 = min(canvas_w, view_w - self.offset_x)
            y2 = min(canvas_h, view_h - self.offset_y)
            
            # Calculate viewport region
            vx1 = max(0, self.offset_x)
            vy1 = max(0, self.offset_y)
            vx2 = min(view_w, self.offset_x + canvas_w)
            vy2 = min(view_h, self.offset_y + canvas_h)
            
            # Copy visible portion to viewport
            if x1 < x2 and y1 < y2 and vx1 < vx2 and vy1 < vy2:
                viewport[vy1:vy2, vx1:vx2] = canvas[y1:y2, x1:x2]
            
            # Scale keypoints for display
            scaled_kps0 = [cv2.KeyPoint(
                x=kp.pt[0] * self.scale,
                y=kp.pt[1] * self.scale,
                size=kp.size * self.scale
            ) for kp in self.orig_keypoints0]
            
            scaled_kps1 = [cv2.KeyPoint(
                x=kp.pt[0] * self.scale + scaled_w0,
                y=kp.pt[1] * self.scale,
                size=kp.size * self.scale
            ) for kp in self.orig_keypoints1]
            
            # Draw matches
            match_vis = cv2.drawMatches(
                scaled_img0, scaled_kps0,
                scaled_img1, scaled_kps1,
                self.matches, viewport,
                matchColor=(0, 255, 0),
                singlePointColor=(255, 0, 0),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            # Show controls
            controls = [
                "Controls:",
                "1/2 - Rotate left/right image (90°)",
                "q/w - Flip left image (horiz/vert)",
                "e/r - Flip right image (horiz/vert)",
                "Mouse: Drag to pan, Wheel to zoom",
                "0 - Reset view",
                "ESC - Exit"
            ]
            
            y = 30
            for line in controls:
                cv2.putText(match_vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(match_vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y += 25
            
            # Show zoom level
            cv2.putText(match_vis, f"Zoom: {self.scale:.2f}x", (10, y+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(match_vis, f"Zoom: {self.scale:.2f}x", (10, y+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Feature Matches", match_vis)
            
            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('1'):
                self.rotation0 = (self.rotation0 + 90) % 360
            elif key == ord('2'):
                self.rotation1 = (self.rotation1 + 90) % 360
            elif key == ord('q'):
                self.flip0 = 1 if self.flip0 != 1 else 0
            elif key == ord('w'):
                self.flip0 = -1 if self.flip0 != -1 else 0
            elif key == ord('e'):
                self.flip1 = 1 if self.flip1 != 1 else 0
            elif key == ord('r'):
                self.flip1 = -1 if self.flip1 != -1 else 0
            elif key == ord('0'):
                self.scale = 0.5
                self.offset_x = 0
                self.offset_y = 0
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        visualizer = MatchVisualizer()
        visualizer.visualize_matches()
    except Exception as e:
        print(f"Error: {str(e)}")