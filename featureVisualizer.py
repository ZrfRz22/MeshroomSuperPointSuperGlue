# Standard Library 
import os
import sys
import json
import struct
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image

# Imported Libraries
import cv2
import numpy as np

# ==================== Logger ====================
class Logger:
    # Logger class to log errors and progress
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def log(self, message: str, level: str = "INFO") -> None:
        # Print log in real-time
        print(f"[{level}] {message}", file=sys.stdout)
        sys.stdout.flush()

# ==================== Feature Loader ====================
class FeaturesLoader:
    # Loads feature keypoints and descriptors
    def __init__(self, features_dir: str):
        self.features_dir = features_dir
        self.logger = Logger()
    
    def load(self, view_id: str) -> Tuple[np.ndarray, np.ndarray, int]:
        # Load features for an image
        feat_path = os.path.join(self.features_dir, f"{view_id}.dspsift.feat")
        desc_path = os.path.join(self.features_dir, f"{view_id}.dspsift.desc")
        
        # Load keypoints
        kpts = []
        with open(feat_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    kpts.append([x, y])

        # Load descriptors
        with open(desc_path, 'rb') as f:
            num_features = struct.unpack('<I', f.read(4))[0]
            desc_dim = struct.unpack('<I', f.read(4))[0]
            desc = np.fromfile(f, dtype=np.uint8, count=num_features * desc_dim)
            desc = desc.reshape(num_features, desc_dim)
        
        return np.array(kpts), np.array(desc), desc_dim

# ==================== Match Loader ====================
class MatchesLoader:
    # Load feature matches from the matches file
    def __init__(self):
        self.logger = Logger()
    
    def load(self, match_dir: str) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
        # Load all matches for image pairs
        all_matches = defaultdict(list)
        
        # Loop through every matches.txt file
        for match_file in os.listdir(match_dir):
            if not match_file.endswith('.matches.txt'):
                continue
                
            file_path = os.path.join(match_dir, match_file)
            self._parse_match_file(file_path, all_matches)
        
        return all_matches
    
    def _parse_match_file(self, file_path: str, all_matches: Dict) -> None:
        # Parses a matches.txt file
        with open(file_path, 'r') as f:
            current_pair = None
            current_count = 0
            matches_read = 0
            state = 'looking_for_pair'
            
            for line in f:
                line = line.strip()
                if not line:
                    continue  
                
                if state == 'looking_for_pair': # Gets image pair IDs
                    parts = line.split()
                    if len(parts) == 2:
                        current_pair = (parts[0], parts[1])
                        state = 'looking_for_count'
                elif state == 'looking_for_count': # Matches number
                    if line == '1':
                        state = 'looking_for_type' # Describer type
                elif state == 'looking_for_type':
                    parts = line.split()
                    if len(parts) == 2 and parts[0] == 'dspsift':
                        current_count = int(parts[1])
                        matches_read = 0
                        state = 'reading_matches' # Gets Matches
                elif state == 'reading_matches':
                    parts = line.split()
                    if len(parts) == 2:
                        idx0, idx1 = map(int, parts)
                        all_matches[current_pair].append((idx0, idx1))
                        matches_read += 1
                        if matches_read >= current_count:
                            state = 'looking_for_pair' # Repeat process for every image pair

# ==================== Image Pair Viewer ====================
class ImagePairViewer:
    # Handles display of image pairs 
    def __init__(self):
        # Parameters for Text display
        self.max_display_size = 800
        self.font_scale = 2.0  
        self.font_thickness = 4  
        self.font = cv2.FONT_HERSHEY_DUPLEX  
        self.text_color = (0, 0, 0)  
        self.text_bg_color = (255, 255, 255) 
        self.text_position = (20, 50)  
        self.text_padding = 10  
        self.show_all_keypoints = True # Option to display all keypoints or not
        
    def show_pair(self, img1: np.ndarray, img2: np.ndarray, title1: str = "Image 1", title2: str = "Image 2") -> None:
        # Display two images side by side in separate windows
        img1_disp = self._resize_for_display(img1)
        img2_disp = self._resize_for_display(img2)
        
        cv2.namedWindow(title1, cv2.WINDOW_NORMAL)
        cv2.namedWindow(title2, cv2.WINDOW_NORMAL)
        cv2.imshow(title1, img1_disp)
        cv2.imshow(title2, img2_disp)
    
    def add_text_overlay(self, img: np.ndarray, text: str) -> np.ndarray:
        # Add text overlay to an image with background for readability
        img_with_text = img.copy()
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness
        )
        
        # Calculate background rectangle coordinates
        x, y = self.text_position
        bg_top_left = (x - self.text_padding, y - text_height - self.text_padding)
        bg_bottom_right = (x + text_width + self.text_padding, y + self.text_padding)
        
        # Draw background rectangle
        cv2.rectangle(
            img_with_text, bg_top_left, bg_bottom_right, 
            self.text_bg_color, -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            img_with_text, text, (x, y), 
            self.font, self.font_scale, self.text_color, self.font_thickness,
            cv2.LINE_AA
        )
        
        return img_with_text
        
    def _resize_for_display(self, img: np.ndarray) -> np.ndarray:
        # Resize image scale for display
        h, w = img.shape[:2]
        scale = min(1.0, self.max_display_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h))

# ==================== Match Visualizer ====================
class MatchVisualizer:
    # Visualizes matched keypoints between image pairs
    def __init__(self):
        self.keypoint_radius = 3
        self.highlight_radius = 10
        self.highlight_thickness = 2
        self.keypoint_color = (0, 255, 0)  # Green for general keypoints
        self.highlight_color = (0, 0, 255)  # Red for matched keypoints
        
    def draw_keypoints(self, img: np.ndarray, kpts: np.ndarray, 
                        highlight_idx: Optional[int] = None, 
                        show_all: bool = True) -> np.ndarray:  
        # Draws all keypoints 
        img_with_kpts = img.copy()
        
        # Only draw all keypoints if show_all is True
        if show_all:
            for i, kpt in enumerate(kpts):
                x, y = int(kpt[0]), int(kpt[1])
                if i == highlight_idx:
                    self._draw_highlighted_keypoint(img_with_kpts, x, y)
                else:
                    cv2.circle(img_with_kpts, (x, y), self.keypoint_radius, 
                            self.keypoint_color, -1)
        elif highlight_idx is not None and 0 <= highlight_idx < len(kpts):
            # Only draw the highlighted keypoint
            kpt = kpts[highlight_idx]
            x, y = int(kpt[0]), int(kpt[1])
            self._draw_highlighted_keypoint(img_with_kpts, x, y)
            
        return img_with_kpts
    
    def _draw_highlighted_keypoint(self, img: np.ndarray, x: int, y: int) -> None:
        # Draw a highlighted keypoint
        cv2.circle(img, (x, y), self.highlight_radius, self.highlight_color, -1)
        cv2.circle(img, (x, y), self.highlight_radius + 2, self.highlight_color, self.highlight_thickness)

# ==================== Match Viewer ====================
class MatchViewer:
    # Main class to visualize matched keypoints between image pairs
    def __init__(self, input_sfm: str, input_features: str, input_matches: str):
        self.logger = Logger()
        self.input_sfm = input_sfm
        self.input_features = input_features
        self.input_matches = input_matches
        
        # Load data
        self.sfm_data = self._load_sfm_data()
        self.all_matches = MatchesLoader().load(input_matches)
        self.image_pairs = list(self.all_matches.keys())
        
        # Initialize components
        self.feature_loader = FeaturesLoader(input_features)
        self.viewer = ImagePairViewer()
        self.visualizer = MatchVisualizer()
        
        # State management
        self.current_pair_index = 0
        self.current_match_index = 0
        
        # Caching
        self.image_cache = {}
        self.feature_cache = {}
        self.original_dims = {}

    def _load_sfm_data(self) -> Dict[str, Any]:
        # Load SfM JSON file
        with open(self.input_sfm) as f:
            return json.load(f)

    def _get_image_path(self, view_id: str) -> Optional[str]:
        # Get image file path from SfM JSON using view ID
        for view in self.sfm_data['views']:
            if view['viewId'] == view_id:
                return view['path']
        return None

    def _load_image(self, view_id: str) -> Optional[np.ndarray]:
        # Load an image
        if view_id in self.image_cache:
            return self.image_cache[view_id]
        
        image_path = self._get_image_path(view_id)
        if not image_path:
            return None
        
        # Gets the image's raw pixel orientation
        try:
            pil_img = Image.open(image_path).copy()
            img = np.array(pil_img)

            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            self.original_dims[view_id] = img.shape[:2]
            self.image_cache[view_id] = img
            return img

        except Exception as e:
            self.logger.log(f"Error loading image {image_path}: {str(e)}", "ERROR")
            return None

    def _load_features(self, view_id: str) -> Optional[np.ndarray]:
        # Load keypoints for a view
        if view_id in self.feature_cache:
            return self.feature_cache[view_id]
        
        try:
            kpts, _, _ = self.feature_loader.load(view_id)
            self.feature_cache[view_id] = kpts
            return kpts
        except Exception as e:
            self.logger.log(f"Error loading features for {view_id}: {str(e)}", "ERROR")
            return None
    
    def show_matches(self) -> None:
        # Display matches
        if not self.image_pairs:
            self.logger.log("No image pairs with matches found", "WARNING")
            return
        
        while True:
            view_id0, view_id1 = self.image_pairs[self.current_pair_index]
            matches = self.all_matches[(view_id0, view_id1)]
            
            # Load data 
            img0 = self._load_image(view_id0)
            img1 = self._load_image(view_id1)
            kpts0 = self._load_features(view_id0)
            kpts1 = self._load_features(view_id1)
            
            # Check if any data failed to load 
            if (img0 is None or img1 is None or 
                kpts0 is None or kpts1 is None or 
                not isinstance(kpts0, np.ndarray) or 
                not isinstance(kpts1, np.ndarray) or 
                not matches):
                self._handle_invalid_pair(view_id0, view_id1, matches)
                continue
            
            # Prepare visualizations
            idx0, idx1 = matches[self.current_match_index % len(matches)]
            img0_disp = self._prepare_image_display(img0, kpts0, idx0, view_id0)
            img1_disp = self._prepare_image_display(img1, kpts1, idx1, view_id1)
            
            # Show images
            self.viewer.show_pair(img0_disp, img1_disp)
            
            # Handle user input
            if not self._handle_user_input(len(matches)):
                break

        cv2.destroyAllWindows()

    def _prepare_image_display(self, img: np.ndarray, kpts: np.ndarray,
                             highlight_idx: int, view_id: str) -> np.ndarray:
        # Prepare an image for display with keypoints and text
        img_disp = self.visualizer.draw_keypoints(
            img, kpts, highlight_idx, 
            self.viewer.show_all_keypoints 
        )
        
        kpt_info = ""
        if 0 <= highlight_idx < len(kpts):
            x, y = kpts[highlight_idx]
            kpt_info = f" ({x:.0f},{y:.0f})"
        
        text = f"Image: {view_id} | Match: {self.current_match_index + 1}/{len(self.all_matches[self.image_pairs[self.current_pair_index]])}{kpt_info}"
        
        return self.viewer.add_text_overlay(img_disp, text)

    def _handle_invalid_pair(self, view_id0: str, view_id1: str, matches: List) -> None:
        # Handle cases where pair data couldn't be loaded
        if not matches:
            self.logger.log(f"No matches for pair {view_id0}-{view_id1}", "WARNING")
        else:
            self.logger.log(f"Failed to load data for pair {view_id0}-{view_id1}", "ERROR")
        
        # Move to next pair
        self.current_pair_index = (self.current_pair_index + 1) % len(self.image_pairs)
        self.current_match_index = 0

    def _handle_user_input(self, num_matches: int) -> bool:
        # Process user keyboard input
        print("\nNavigation:")
        print(f"  Current pair: {self.current_pair_index + 1}/{len(self.image_pairs)}")
        print(f"  Current match: {self.current_match_index + 1}/{num_matches}")
        print("Controls:")
        print("  Up/Down: Previous/Next match")
        print("  Left/Right: Previous/Next pair")
        print("  'a': Toggle all keypoints visibility") 
        print("  ESC: Quit")
        
        key = cv2.waitKeyEx(0)
        
        if key == 27:  # ESC
            return False
        elif key == 2490368:  # Up arrow
            self.current_match_index = (self.current_match_index - 1) % num_matches
        elif key == 2621440:  # Down arrow
            self.current_match_index = (self.current_match_index + 1) % num_matches
        elif key == 2424832:  # Left arrow
            self.current_pair_index = (self.current_pair_index - 1) % len(self.image_pairs)
            self.current_match_index = 0
        elif key == 2555904:  # Right arrow
            self.current_pair_index = (self.current_pair_index + 1) % len(self.image_pairs)
            self.current_match_index = 0
        elif key == ord('a'):  # Toggle all keypoints
            self.viewer.show_all_keypoints = not self.viewer.show_all_keypoints
            self.logger.log(f"Showing all keypoints: {self.viewer.show_all_keypoints}")
        
        return True

# ==================== Main ====================
def main():
    # Retrieves inputs from frontend
    parser = argparse.ArgumentParser(description='Meshroom Match Viewer')
    parser.add_argument('--inputSfM', required=True, help='Path to SfM data file')
    parser.add_argument('--inputFeatures', required=True, help='Path to features directory')
    parser.add_argument('--inputMatches', required=True, help='Path to matches directory')
    
    args = parser.parse_args()
    
    if not all(os.path.exists(path) for path in [args.inputSfM, args.inputFeatures, args.inputMatches]):
        print("Error: One or more input paths do not exist")
        exit(1)

    viewer = MatchViewer(args.inputSfM, args.inputFeatures, args.inputMatches)
    viewer.show_matches()

if __name__ == "__main__":
    main()