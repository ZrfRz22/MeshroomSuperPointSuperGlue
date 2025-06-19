import os
import cv2
import numpy as np
import argparse
import struct
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class Logger:
    def log(self, message: str, level: str = "INFO"):
        print(f"[{level}] {message}")

class DSPSiftFeatureLoader:
    def __init__(self, features_dir: str):
        self.features_dir = features_dir
        self.logger = Logger()
    
    def load(self, view_id: str) -> Tuple[np.ndarray, np.ndarray, int]:
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
            desc = np.fromfile(f, dtype=np.uint8, count=num_features*desc_dim)
            desc = desc.reshape(num_features, desc_dim)
        
        return np.array(kpts), np.array(desc), desc_dim

class MatchesLoader:
    def __init__(self):
        self.logger = Logger()
    
    def load(self, match_dir: str) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
        all_matches = defaultdict(list)
        
        for match_file in os.listdir(match_dir):
            if not match_file.endswith('.matches.txt'):
                continue
                
            file_path = os.path.join(match_dir, match_file)
            
            with open(file_path, 'r') as f:
                current_pair = None
                current_count = 0
                matches_read = 0
                state = 'looking_for_pair'
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if state == 'looking_for_pair':
                        parts = line.split()
                        if len(parts) == 2:
                            current_pair = (parts[0], parts[1])
                            state = 'looking_for_count'
                            
                    elif state == 'looking_for_count':
                        if line == '1':
                            state = 'looking_for_type'
                            
                    elif state == 'looking_for_type':
                        parts = line.split()
                        if len(parts) == 2 and parts[0] == 'dspsift':
                            current_count = int(parts[1])
                            matches_read = 0
                            state = 'reading_matches'
                            
                    elif state == 'reading_matches':
                        parts = line.split()
                        if len(parts) == 2:
                            idx0, idx1 = map(int, parts)
                            all_matches[current_pair].append((idx0, idx1))
                            matches_read += 1
                            if matches_read >= current_count:
                                state = 'looking_for_pair'
        
        return all_matches

class MatchViewer:
    def __init__(self, input_sfm: str, input_features: str, input_matches: str):
        self.logger = Logger()
        self.input_sfm = input_sfm
        self.input_features = input_features
        self.input_matches = input_matches
        
        with open(input_sfm) as f:
            self.sfm_data = json.load(f)
        
        self.feature_loader = DSPSiftFeatureLoader(input_features)
        self.matches_loader = MatchesLoader()
        self.all_matches = self.matches_loader.load(input_matches)
        
        self.current_pair_index = 0
        self.current_match_index = 0
        self.image_pairs = list(self.all_matches.keys())
        
        self.image_cache = {}
        self.feature_cache = {}
        self.original_dims = {}
        self.max_display_size = 800  # Maximum display dimension

        self.rotate_ccw = True  # Default to counter-clockwise rotation
        self.flip_vertical = False


    def _get_image_path(self, view_id: str) -> Optional[str]:
        for view in self.sfm_data['views']:
            if view['viewId'] == view_id:
                return view['path']
        return None
    
    def _load_image(self, view_id: str) -> Optional[np.ndarray]:
        if view_id in self.image_cache:
            return self.image_cache[view_id]
        
        image_path = self._get_image_path(view_id)
        if not image_path:
            return None
        
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Store original dimensions before rotation
        h, w = img.shape[:2]
        self.original_dims[view_id] = (h, w)
        
        if self.rotate_ccw:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # Apply vertical flip if enabled
        if self.flip_vertical:
            img = cv2.flip(img, 0)  # 0 means vertical flip

        self.image_cache[view_id] = img
        return img
    
    def _load_features(self, view_id: str) -> Optional[np.ndarray]:
        if view_id in self.feature_cache:
            return self.feature_cache[view_id]
        
        try:
            kpts, _, _ = self.feature_loader.load(view_id)
            self.feature_cache[view_id] = kpts
            return kpts
        except:
            return None
    
    def _draw_keypoint(self, img: np.ndarray, kpt: np.ndarray) -> np.ndarray:
        x, y = int(kpt[0]), int(kpt[1])
        cv2.circle(img, (x, y), 10, (0, 0, 255), -1)  # red filled
        cv2.circle(img, (x, y), 12, (0, 0, 255), 2)   # red border
        return img
    
    def _draw_all_keypoints(self, img: np.ndarray, kpts: np.ndarray, exclude_idx: Optional[int] = None) -> np.ndarray:
        for i, kpt in enumerate(kpts):
            if exclude_idx is not None and i == exclude_idx:
                continue
            x, y = int(kpt[0]), int(kpt[1])
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)  # small green circles
        return img

    def _resize_for_display(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        max_display_height = 600  # Just like the simple viewer
        scale = min(1.0, max_display_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h))
    
    def show_matches(self):
        if not self.image_pairs:
            print("No image pairs with matches found")
            return
        
        # Create windows
        cv2.namedWindow("Image 1", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Image 2", cv2.WINDOW_NORMAL)
        
        while True:
            view_id0, view_id1 = self.image_pairs[self.current_pair_index]
            matches = self.all_matches[(view_id0, view_id1)]
            
            img0 = self._load_image(view_id0)
            img1 = self._load_image(view_id1)
            kpts0 = self._load_features(view_id0)
            kpts1 = self._load_features(view_id1)
            
            if img0 is None or img1 is None or kpts0 is None or kpts1 is None:
                print(f"Failed to load data for pair {view_id0}-{view_id1}")
                continue
            
            if not matches:
                print(f"No matches for pair {view_id0}-{view_id1}")
                continue
            
            idx0, idx1 = matches[self.current_match_index % len(matches)]
            
            img0_disp = img0.copy()
            img1_disp = img1.copy()

            # Draw all keypoints except the selected one
            img0_disp = self._draw_all_keypoints(img0_disp, kpts0, exclude_idx=idx0)
            img1_disp = self._draw_all_keypoints(img1_disp, kpts1, exclude_idx=idx1)

            # Highlight selected match keypoints
            if 0 <= idx0 < len(kpts0):
                self._draw_keypoint(img0_disp, kpts0[idx0])
            if 0 <= idx1 < len(kpts1):
                self._draw_keypoint(img1_disp, kpts1[idx1])
            
            # Resize images
            img0_disp = self._resize_for_display(img0_disp)
            img1_disp = self._resize_for_display(img1_disp)

            # Prepare overlay text
            text_info_0 = f"Image: {view_id0} | Match {self.current_match_index + 1}/{len(matches)}"
            text_info_1 = f"Image: {view_id1} | Match {self.current_match_index + 1}/{len(matches)}"

            # Keypoint coordinates (adjust for rotation if necessary)
            if 0 <= idx0 < len(kpts0):
                x0, y0 = kpts0[idx0]
                text_info_0 += f" | Kpt: ({x0:.1f}, {y0:.1f})"
            if 0 <= idx1 < len(kpts1):
                x1, y1 = kpts1[idx1]
                text_info_1 += f" | Kpt: ({x1:.1f}, {y1:.1f})"

            # Draw overlay text
            cv2.putText(img0_disp, text_info_0, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img1_disp, text_info_1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show images
            cv2.imshow("Image 1", img0_disp)
            cv2.imshow("Image 2", img1_disp)
            
            print(f"\nPair {self.current_pair_index + 1}/{len(self.image_pairs)}: {view_id0} - {view_id1}")
            print(f"Match {self.current_match_index + 1}/{len(matches)}")
            print("Controls:")
            print("  Up/Down: Previous/Next match")
            print("  Left/Right: Previous/Next pair")
            print("  ESC: Quit")
            
            key = cv2.waitKeyEx(0)  # Use waitKeyEx for better key code detection
            
            # Handle key presses
            if key == 27:  # ESC
                break
            elif key == 2490368:  # Up arrow
                self.current_match_index = (self.current_match_index - 1) % len(matches)
            elif key == 2621440:  # Down arrow
                self.current_match_index = (self.current_match_index + 1) % len(matches)
            elif key == 2424832:  # Left arrow
                self.current_pair_index = (self.current_pair_index - 1) % len(self.image_pairs)
                self.current_match_index = 0
            elif key == 2555904:  # Right arrow
                self.current_pair_index = (self.current_pair_index + 1) % len(self.image_pairs)
                self.current_match_index = 0
            elif key == ord('n'):  # Next pair
                self.current_pair_index = (self.current_pair_index + 1) % len(self.image_pairs)
                self.current_match_index = 0
            elif key == ord('p'):  # Previous pair
                self.current_pair_index = (self.current_pair_index - 1) % len(self.image_pairs)
                self.current_match_index = 0
            elif key == ord('r'):
                self.rotate_ccw = not self.rotate_ccw
                self.image_cache.clear()  # Clear cache to reload with new rotation
            elif key == ord('f'):  # Flip images vertically
                self.flip_vertical = not self.flip_vertical
                self.image_cache.clear()

        cv2.destroyAllWindows()

if __name__ == "__main__":
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