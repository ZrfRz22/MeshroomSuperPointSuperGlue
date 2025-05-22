import cv2
import numpy as np
import os

class ImageViewer:
    def __init__(self):
        # Hardcoded file paths
        self.image_path = r"C:\Users\zarif\OneDrive\Documents\Photogrammetry\Converse2Normal\20241126_221853.jpg"
        self.feat_path = r"C:\Users\zarif\OneDrive\Documents\Photogrammetry\MeshroomCache\SuperPointFeatureExtraction\cf3c74960a15930e8bbb4bc4fd9f50134dfe945c\492971557.dspsift.feat"
        
        # Viewport parameters
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.last_x, self.last_y = 0, 0
        self.rotation_angle = 0  # 0, 90, 180, 270 degrees
        
        # Load image and keypoints
        self.load_image_and_keypoints()
        
    def load_image_and_keypoints(self):
        # Read the image
        if not os.path.exists(self.image_path):
            print(f"Error: Image file not found at {self.image_path}")
            return False
        
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            print(f"Error: Could not read image at {self.image_path}")
            return False
        
        # Read keypoints from feat file
        if not os.path.exists(self.feat_path):
            print(f"Error: Feature file not found at {self.feat_path}")
            return False
        
        kpts_data = []
        with open(self.feat_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    kpts_data.append([float(parts[0]), float(parts[1])])
        
        self.original_kpts = np.array(kpts_data, dtype=np.float32)
        self.reset_view()
        return True
    
    def reset_view(self):
        self.scale = min(1.0, 800/self.original_image.shape[1], 600/self.original_image.shape[0])
        self.offset_x = 0
        self.offset_y = 0
        self.current_kpts = self.original_kpts.copy()
        self.rotation_angle = 0
    
    def get_transformed_keypoints(self):
        return [cv2.KeyPoint(
            x=(pt[0] * self.scale + self.offset_x),
            y=(pt[1] * self.scale + self.offset_y),
            size=5*self.scale) for pt in self.current_kpts]
    
    def get_rotated_image(self):
        if self.rotation_angle == 0:
            return self.original_image.copy()
        elif self.rotation_angle == 90:
            return cv2.rotate(self.original_image, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle == 180:
            return cv2.rotate(self.original_image, cv2.ROTATE_180)
        elif self.rotation_angle == 270:
            return cv2.rotate(self.original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return self.original_image.copy()
    
    def get_visible_image(self):
        rotated_img = self.get_rotated_image()
        h, w = rotated_img.shape[:2]
        scaled_w, scaled_h = int(w * self.scale), int(h * self.scale)
        
        # Scale the image
        scaled_img = cv2.resize(rotated_img, (scaled_w, scaled_h))
        
        # Create a black canvas
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Calculate visible area
        x1 = max(0, -self.offset_x)
        y1 = max(0, -self.offset_y)
        x2 = min(scaled_w, 800 - self.offset_x)
        y2 = min(scaled_h, 600 - self.offset_y)
        
        # Calculate canvas area
        cx1 = max(0, self.offset_x)
        cy1 = max(0, self.offset_y)
        cx2 = min(800, self.offset_x + scaled_w)
        cy2 = min(600, self.offset_y + scaled_h)
        
        # Copy visible portion to canvas
        if x1 < x2 and y1 < y2 and cx1 < cx2 and cy1 < cy2:
            canvas[cy1:cy2, cx1:cx2] = scaled_img[y1:y2, x1:x2]
        
        return canvas
    
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
    
    def run(self):
        cv2.namedWindow("Image with Keypoints")
        cv2.setMouseCallback("Image with Keypoints", self.mouse_callback)
        
        print("Keyboard controls:")
        print("+/- - Zoom in/out")
        print("0 - Reset view")
        print("r - Rotate image (keypoints stay fixed)")
        print("f - Flip horizontally")
        print("v - Flip vertically")
        print("Mouse drag - Pan image")
        print("ESC or q - Quit")
        
        while True:
            # Get the visible portion of the image
            visible_img = self.get_visible_image()
            
            # Draw keypoints (they remain in original position)
            keypoints = self.get_transformed_keypoints()
            img_with_keypoints = cv2.drawKeypoints(
                visible_img, 
                keypoints, 
                None, 
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            
            # Show scale and rotation info
            cv2.putText(img_with_keypoints, f"Zoom: {self.scale:.2f}x, Rotation: {self.rotation_angle}°", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Image with Keypoints", img_with_keypoints)
            
            key = cv2.waitKey(10) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break
            elif key == ord('+') or key == ord('='):
                self.scale *= 1.1
            elif key == ord('-') or key == ord('_'):
                self.scale /= 1.1
            elif key == ord('0'):
                self.reset_view()
            elif key == ord('r'):  # Rotate image only (keypoints stay fixed)
                self.rotation_angle = (self.rotation_angle + 90) % 360
            elif key == ord('f'):  # Flip horizontal
                self.original_image = cv2.flip(self.original_image, 1)
                w = self.original_image.shape[1]
                self.original_kpts[:, 0] = w - self.original_kpts[:, 0]
            elif key == ord('v'):  # Flip vertical
                self.original_image = cv2.flip(self.original_image, 0)
                h = self.original_image.shape[0]
                self.original_kpts[:, 1] = h - self.original_kpts[:, 1]
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    viewer = ImageViewer()
    if viewer.load_image_and_keypoints():
        viewer.run()