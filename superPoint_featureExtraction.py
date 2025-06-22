# Standard Library
import argparse
import json
import os
import struct
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple
from PIL import Image

# Third-Party Libraries 
import cv2
import numpy as np
import torch

#  Local Modules
from superpoint import SuperPoint

# ==================== Configuration Classes ====================
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple
import json
import sys
import time

# ==================== Configuration Classes ====================

# Configuration for the SuperPoint feature extractor
@dataclass
class SuperPointConfig:
    weights_path: str            # Path to the model weights file
    descriptor_dim: int = 256    # Dimensionality of feature descriptors
    nms_radius: int = 4          # Non-maximum suppression radius
    keypoint_threshold: float = 0.005  # Threshold for keypoint detection
    max_keypoints: int = -1      # Maximum number of keypoints to detect (-1 for no limit)
    remove_borders: int = 4      # Border pixels to ignore around image edges

# Configuration for the feature extraction pipeline
@dataclass
class FeatureExtractionConfig:
    input_sfm: str               # Path to input SfM (Structure from Motion) data file
    output_dir: str              # Directory to save extracted features
    describer_type: str = "dspsift"  # Type of feature descriptor to use

# ==================== Logger (Singleton) ====================
class Logger:
    _instance = None # Singleton instance

    def __new__(cls):
        # Ensure only one instance of Logger exists
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.start_time = time.time()  # Initialize start time
        return cls._instance

    # Log a message with timestamp and elapsed time
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed = timedelta(seconds=time.time() - self.start_time)
        log_entry = f"[{level}][{timestamp}][{elapsed}] {message}"

        if level == "ERROR":
            print(log_entry, file=sys.stderr)
        else:
            print(log_entry, file=sys.stdout)

        sys.stdout.flush()

    # Log progress information with percentage
    def progress(self, current: int, total: int, message: str = ""):
        progress = (current / total) * 100
        self.log(f"PROGRESS: {current}/{total} ({progress:.1f}%) {message}")

# ==================== SFM Parser ====================
class SFMParser:
    # Parser for SfM data files
    def __init__(self, logger: Logger):
        self.logger = logger

    # Parse an SfM file to extract view information
    def parse(self, sfm_path: str) -> List[Tuple[str, str]]:
        self.logger.log(f"Parsing SFM file: {sfm_path}", "DEBUG")
        try:
            with open(sfm_path, 'r') as f:
                data = json.load(f)  # Load JSON data from file

            # Extract view ID and image path pairs from the SfM file
            views = [(str(v["viewId"]), v["path"]) for v in data["views"]]
            self.logger.log(f"Found {len(views)} images to process", "INFO")
            return views
        except Exception as e:
            self.logger.log(f"Failed to parse SFM file: {str(e)}", "ERROR")
            raise


# ==================== Abstract Feature Extractor ====================
class FeatureExtractor(ABC):
    # Extract features from an image
    @abstractmethod
    def extract(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

# ==================== Concrete Feature Extractor ====================
class DSPSiftSuperPointExtractor(FeatureExtractor):
    # SuperPoint feature extractor with DSP-SIFT output format
    def __init__(self, config: SuperPointConfig, logger: Logger):
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.model = self._load_model()

    # Load and initialize the SuperPoint model
    def _load_model(self):
        try:
            self.logger.log(f"Loading SuperPoint model with config: {self.config}", "DEBUG")
            model = SuperPoint(self.config.__dict__, self.config.weights_path).to(self.device)
            self.logger.log("SuperPoint model initialized", "INFO")
            return model.eval()
        except Exception as e:
            self.logger.log(f"Failed to load model: {str(e)}", "ERROR")
            raise

    # Extract features from an image file
    def extract(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.logger.log(f"Extracting features from {image_path}", "DEBUG")

        try:
            # Load image using PIL (Python Imaging Library)
            pil_img = Image.open(image_path)
            
            # Convert PIL image to NumPy array (raw pixel data)
            image = np.array(pil_img)
            
            # Convert to grayscale if the image is in RGB (3 channels)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Raise error if image has more than 3 dimensions (e.g. 4D RGBA or others)
            elif len(image.shape) > 3:
                raise ValueError(f"Unexpected image shape: {image.shape}")

            # Disable gradient computation (inference mode)
            with torch.no_grad():
                # Normalize image to range [0, 1], convert to tensor, add batch & channel dimensions
                img_tensor = torch.from_numpy(image / 255.).float()[None, None].to(self.device)

                # Pass the image through the SuperPoint model
                output = self.model({'image': img_tensor})

                # Extract keypoints: convert from (y, x) to (x, y) if needed
                keypoints = output['keypoints'][0].cpu().numpy()  # shape: (N, 2)

                # Extract descriptors and transpose to (N, descriptor_dim)
                descriptors = output['descriptors'][0].t().cpu().numpy()

                # Extract confidence scores for each keypoint
                scores = output['scores'][0].cpu().numpy()

                # Normalize descriptors to unit length (L2 normalization)
                descriptors = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)

            # Log successful feature extraction
            self.logger.log(f"Extracted {len(keypoints)} features from {image_path}", "INFO")
            
            # Return keypoints, descriptors, and scores
            return keypoints, descriptors.astype(np.float32), scores

        except Exception as e:
            # Log any exception during feature extraction
            self.logger.log(f"Feature extraction failed for {image_path}: {str(e)}", "ERROR")
            raise


# ==================== Feature Extractor Factory ====================
class FeatureExtractorFactory:
    # Create a feature extractor instance
    @staticmethod
    def create_extractor(extractor_type: str, config: SuperPointConfig, logger: Logger) -> FeatureExtractor:
        if extractor_type == "dspsift":
            return DSPSiftSuperPointExtractor(config, logger)
        raise ValueError(f"Unsupported extractor type: {extractor_type}")


# ==================== Abstract Feature Saver ====================
class FeatureSaver(ABC):
    # Abstract base class for feature savers
    def __init__(self, output_dir: str, logger: Logger):
        self.output_dir = output_dir
        self.logger = logger
        os.makedirs(output_dir, exist_ok=True)
        self.logger.log(f"Initialized feature saver with output directory: {output_dir}", "DEBUG")

    # Save features to disk
    @abstractmethod
    def save(self, image_id: str, keypoints: np.ndarray,
             descriptors: np.ndarray, scores: np.ndarray) -> None:
        pass

# ==================== Concrete Feature Saver ====================
class DSPSiftFeatureSaver(FeatureSaver):
    # Saver for DSP-SIFT format features with additional .npz saving
    def save(self, image_id: str, keypoints: np.ndarray,
         descriptors: np.ndarray, scores: np.ndarray) -> None:
        # Save features in .feat format (for DSP-SIFT) and a .npz archive
        save_start = time.time()
        try:
            # Define output file paths
            feat_path = os.path.join(self.output_dir, f"{image_id}.dspsift.feat")
            desc_path = os.path.join(self.output_dir, f"{image_id}.dspsift.desc")
            npz_path = os.path.join(self.output_dir, f"{image_id}.features.npz")

            # Save coordinates to .feat file
            with open(feat_path, 'w') as f:
                for x, y in keypoints:
                    f.write(f"{x} {y} 1.0 0.0\n")

            # Save descriptors to .desc file
            with open(desc_path, 'wb') as f:
                f.write(struct.pack('<I', keypoints.shape[0]))
                f.write(struct.pack('<I', descriptors.shape[1]))
                f.write(descriptors.tobytes())

            # Save data to .npz file
            np.savez_compressed(npz_path, 
                            keypoints=keypoints,
                            descriptors=descriptors, 
                            scores=scores)

            save_time = time.time() - save_start
            self.logger.log(f"Saved {len(keypoints)} features for {image_id} in {save_time:.3f}s", "DEBUG")
            
        except Exception as e:
            self.logger.log(f"Failed to save features for {image_id}: {str(e)}", "ERROR")
            raise

# ==================== Feature Saver Factory ====================
class FeatureSaverFactory:
    # Create a feature saver instance
    @staticmethod
    def create_saver(saver_type: str, output_dir: str, logger: Logger) -> FeatureSaver:
        if saver_type == "dspsift":
            logger.log("Creating DSP-SIFT feature saver", "INFO")
            return DSPSiftFeatureSaver(output_dir, logger)
        raise ValueError(f"Unsupported saver type: {saver_type}")

# ==================== SuperPoint Extraction Pipeline ====================
class SuperPointExtractionPipeline:
    # Pipeline for SuperPoint feature extraction
    def __init__(self, config: FeatureExtractionConfig, superpoint_config: SuperPointConfig):
        self.logger = Logger()
        self.config = config
        self.views = SFMParser(self.logger).parse(config.input_sfm)
        self.extractor = FeatureExtractorFactory.create_extractor(
            config.describer_type, superpoint_config, self.logger)
        self.saver = FeatureSaverFactory.create_saver(
            config.describer_type, config.output_dir, self.logger)
        
        os.makedirs(config.output_dir, exist_ok=True)
        self.logger.log("Feature extraction pipeline initialized", "INFO")
    
    def run(self):
        # Run the feature extraction pipeline
        total_images = len(self.views)
        total_features = 0
        start_time = time.time()
        
        for idx, (image_id, image_path) in enumerate(self.views, 1):
            try:
                self.logger.progress(idx, total_images, f"Processing {image_id}")
                
                # Extract and save features
                keypoints, descriptors, scores = self.extractor.extract(image_path)
                self.saver.save(image_id, keypoints, descriptors, scores)
                total_features += len(keypoints)
                
            except Exception as e:
                self.logger.log(f"Skipping image {image_id}: {str(e)}", "ERROR")
                continue
        
        # Final statistics
        total_time = time.time() - start_time
        self.logger.log("\n=== Extraction Complete ===", "INFO")
        self.logger.log(f"Total images processed: {idx}/{total_images}", "INFO")
        self.logger.log(f"Total features extracted: {total_features}", "INFO")
        self.logger.log(f"Average features per image: {total_features/idx:.1f}", "INFO")
        self.logger.log(f"Total processing time: {total_time:.2f} seconds", "INFO")
        self.logger.log(f"Processing rate: {idx/total_time:.2f} images/second", "INFO")

# ==================== Main ====================
def main():
    # Main entry point for SuperPoint feature extraction
    parser = argparse.ArgumentParser(description='SuperPoint feature extraction')
    parser.add_argument('--input', required=True, help='Input SfM file')
    parser.add_argument('--weights', required=True, help='Model weights path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--maxKeypoints', type=int, default=-1, help='Max keypoints per image')
    parser.add_argument('--nmsRadius', type=int, default=4, help='Non-maximum suppression radius')
    parser.add_argument('--describerTypes', default='dspsift', help='Feature type')
    
    args = parser.parse_args()
    
    # Create configurations
    superpoint_config = SuperPointConfig(
        weights_path=args.weights,
        max_keypoints=args.maxKeypoints,
        nms_radius=args.nmsRadius
    )
    
    extraction_config = FeatureExtractionConfig(
        input_sfm=args.input,
        output_dir=args.output,
        describer_type=args.describerTypes
    )
    
    # Run pipeline
    pipeline = SuperPointExtractionPipeline(extraction_config, superpoint_config)
    pipeline.run()

if __name__ == "__main__":
    main()

