# ==================== SuperGlue Implementation with RANSAC ====================
import os
import numpy as np
import torch
import json
import struct
import sys
import time
import argparse
import cv2
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from superglue import SuperGlue

# ==================== Configuration Classes ====================
@dataclass
class SuperGlueConfig:
    """Configuration for the SuperGlue feature matcher"""
    weights_path: str            # Path to the model weights file
    weights_type: str            # Type of weights ('indoor' or 'outdoor')
    match_threshold: float = 0.7 # Matching threshold
    sinkhorn_iterations: int = 20 # Number of Sinkhorn iterations
    force_cpu: bool = False      # Force CPU usage even if GPU is available

@dataclass
class FeatureMatchingConfig:
    """Configuration for the feature matching pipeline"""
    input_sfm: str               # Path to input SfM data file
    pairs_file: str              # File containing image pairs to match
    features_dir: str            # Directory containing input features
    output_dir: str              # Directory to save matches
    describer_type: str = "dspsift"  # Type of feature descriptor
    ransac_threshold: float = 3.0   # RANSAC inlier threshold (pixels)
    ransac_max_trials: int = 1000   # Max RANSAC iterations

# ==================== Logger (Singleton) ====================
class Logger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.start_time = time.time()
        return cls._instance
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed = timedelta(seconds=time.time() - self.start_time)
        log_entry = f"[{level}][{timestamp}][{elapsed}] {message}"
        
        if level == "ERROR":
            print(log_entry, file=sys.stderr)
        else:
            print(log_entry, file=sys.stdout)
        
        sys.stdout.flush()

    def progress(self, current: int, total: int, message: str = ""):
        progress = (current / total) * 100
        self.log(f"PROGRESS: {current}/{total} ({progress:.1f}%) {message}")

# ==================== Feature Loader (Abstract) ====================
class FeatureLoader(ABC):
    """Abstract base class for feature loaders"""
    @abstractmethod
    def load(self, features_dir: str, view_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load features for a given view ID"""
        pass

# ==================== DPSIFT Loader ====================
class DSPSiftLoader(FeatureLoader):
    """Loader for SuperPoint .npz format features"""
    def load(self, features_dir: str, view_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logger = Logger()
        
        # Define .npz path
        npz_path = os.path.join(features_dir, f"{view_id}.features.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"SuperPoint .npz file not found: {npz_path}")

        # Load SuperPoint features
        with np.load(npz_path) as data:
            keypoints = data['keypoints']      # (N, 2)
            descriptors = data['descriptors']  # (N, D)
            scores = data['scores']            # (N,)

        logger.log(f"Loaded {len(keypoints)} SuperPoint features for {view_id}", "INFO")
        return keypoints, descriptors, scores

# ==================== Feature Loader Factory ====================
class FeatureLoaderFactory:
    """Factory for creating feature loaders"""
    @staticmethod
    def create_loader(describer_type: str) -> FeatureLoader:
        """Create a feature loader instance"""
        if describer_type == "dspsift":
            Logger().log("Creating DSP-SIFT feature loader", "INFO")
            return DSPSiftLoader()
        raise ValueError(f"Unsupported describer type: {describer_type}")

# ==================== Match Saver ====================
class MatchSaver:
    """Saver for match results"""
    def __init__(self, describer_type: str):
        self.describer_type = describer_type
    
    def save(self, output_path: str, view_id0: str, view_id1: str, matches: np.ndarray):
        """Save matches to disk in DSP-SIFT format"""
        # Filter out invalid matches (-1 indicates no match)
        valid_matches = [(i, m) for i, m in enumerate(matches) if m != -1]
        
        with open(output_path, 'a') as f:
            # Write match header
            f.write(f"{view_id0} {view_id1}\n")
            f.write("1\n")  # Version number
            f.write(f"{self.describer_type} {len(valid_matches)}\n")
            # Write individual matches
            for i, j in valid_matches:
                f.write(f"{i} {j}\n")
        
        Logger().log(f"Saved {len(valid_matches)} matches between {view_id0} and {view_id1}", "INFO")

# ==================== RANSAC Filter ====================
class RANSACFilter:
    """Class for filtering matches using RANSAC"""
    def __init__(self, threshold: float = 3.0, max_trials: int = 1000):
        self.threshold = threshold  # Inlier threshold (pixels)
        self.max_trials = max_trials  # Max RANSAC iterations
        self.min_samples = 8 # Fundamental Matrix requires at least 8 points
        self.logger = Logger()
    
    def filter_matches(self, kpts0: np.ndarray, kpts1: np.ndarray, matches: np.ndarray) -> np.ndarray:
        """Filter matches using the Fundamental Matrix and RANSAC to find inliers."""
        valid_indices = np.where(matches != -1)[0]
        if len(valid_indices) < self.min_samples:
            self.logger.log(f"Not enough matches ({len(valid_indices)}) for RANSAC", "WARNING")
            return matches
        
        pts0 = kpts0[valid_indices]
        pts1 = kpts1[matches[valid_indices]]
        
        # Find Fundamental Matrix using RANSAC
        try:
            # Use cv2.findFundamentalMat instead of findHomography
            F, mask = cv2.findFundamentalMat(
                pts0, pts1,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=self.threshold, # This is the key threshold in pixels
                confidence=0.999, # Desired confidence level
                maxIters=self.max_trials
            )

            if mask is None: # RANSAC failed to find a consensus
                self.logger.log("RANSAC with Fundamental Matrix failed to find a model.", "WARNING")
                return np.full_like(matches, -1) # Discard all matches for this pair

            inliers = mask.ravel().astype(bool)
            
            # Create new matches array with outliers removed
            filtered_matches = matches.copy()
            filtered_matches[valid_indices[~inliers]] = -1
            
            self.logger.log(
                f"RANSAC: Kept {inliers.sum()}/{len(valid_indices)} inliers "
                f"(threshold={self.threshold}px)", 
                "DEBUG"
            )
            return filtered_matches
            
        except Exception as e:
            self.logger.log(f"RANSAC failed: {str(e)}", "WARNING")
            return matches

# ==================== SuperGlue Matcher ====================
class SuperGlueMatcher:
    """Wrapper for SuperGlue matching model"""
    def __init__(self, config: SuperGlueConfig):
        self.logger = Logger()
        self.device = 'cuda' if torch.cuda.is_available() and not config.force_cpu else 'cpu'
        self.model = self._load_model(config)
    
    def _load_model(self, config: SuperGlueConfig):
        """Load and initialize the SuperGlue model"""
        try:
            model = SuperGlue({
                'weights': config.weights_type,
                'match_threshold': config.match_threshold,
                'sinkhorn_iterations': config.sinkhorn_iterations
            }, config.weights_path).eval().to(self.device)
            self.logger.log("SuperGlue model initialized", "INFO")
            return model
        except Exception as e:
            self.logger.log(f"Failed to load model: {str(e)}", "ERROR")
            raise
    
    def match(self, data: Dict[str, torch.Tensor]) -> np.ndarray:
        """Run SuperGlue matching on input data"""
        with torch.no_grad():
            return self.model(data)['matches0'][0].cpu().numpy()

# ==================== Feature Matching Pipeline ====================
class FeatureMatchingPipeline:
    """Pipeline for SuperGlue feature matching with RANSAC"""
    def __init__(self, config: FeatureMatchingConfig, superglue_config: SuperGlueConfig):
        self.logger = Logger()
        self.config = config
        self.superglue = SuperGlueMatcher(superglue_config)
        self.loader = FeatureLoaderFactory.create_loader(config.describer_type)
        self.saver = MatchSaver(config.describer_type)
        self.ransac = RANSACFilter(
            threshold=config.ransac_threshold,
            max_trials=config.ransac_max_trials
        )
        
        # Load image shapes from SfM data
        with open(config.input_sfm) as f:
            self.shapes = {view['viewId']: (1, 1, int(view['height']), int(view['width'])) 
                          for view in json.load(f)['views']}
        
        os.makedirs(config.output_dir, exist_ok=True)
        self.logger.log(
            f"Initialized pipeline with RANSAC (threshold={config.ransac_threshold}px, "
            f"max_trials={config.ransac_max_trials})", 
            "INFO"
        )
    
    def run(self):
        """Run the feature matching pipeline"""
        pairs = self._load_pairs()
        output_path = os.path.join(self.config.output_dir, "0.matches.txt")
        
        # Clear existing matches file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        
        total_matches = 0
        total_pairs = len(pairs)
        start_time = time.time()
        
        for idx, (id0, id1) in enumerate(pairs, 1):
            num_matches = self._process_pair(idx, total_pairs, id0, id1, output_path)
            total_matches += num_matches
            self.logger.progress(idx, total_pairs, f"Matches: {total_matches}")
        
        # Final statistics
        total_time = time.time() - start_time
        self.logger.log("\n=== Matching Complete ===", "INFO")
        self.logger.log(f"Total pairs processed: {total_pairs}", "INFO")
        self.logger.log(f"Total matches found: {total_matches}", "INFO")
        self.logger.log(f"Average matches per pair: {total_matches/total_pairs:.1f}", "INFO")
        self.logger.log(f"Total processing time: {total_time:.2f} seconds", "INFO")
        self.logger.log(f"Processing rate: {total_pairs/total_time:.2f} pairs/second", "INFO")

    def _load_pairs(self) -> List[Tuple[str, str]]:
        """Load image pairs from pairs file"""
        pairs = set()
        with open(self.config.pairs_file) as f:
            for line in f:
                ids = line.strip().split()
                # Create all possible ordered pairs from the line
                pairs.update(tuple(sorted((ids[i], ids[j]))) for i in range(len(ids)) for j in range(i+1, len(ids)))
        self.logger.log(f"Loaded {len(pairs)} image pairs", "INFO")
        return list(pairs)
    
    def _process_pair(self, pair_idx: int, total_pairs: int, id0: str, id1: str, output_path: str) -> int:
        """Process a single image pair and return number of matches"""
        try:
            # Load features for both images
            kpts0, desc0, scores0 = self.loader.load(self.config.features_dir, id0)
            kpts1, desc1, scores1 = self.loader.load(self.config.features_dir, id1)
            
            # Prepare input data for SuperGlue
            data = {
                'keypoints0': torch.from_numpy(kpts0).float().unsqueeze(0).to(self.superglue.device),
                'keypoints1': torch.from_numpy(kpts1).float().unsqueeze(0).to(self.superglue.device),
                'descriptors0': (torch.from_numpy(desc0.T).float()).unsqueeze(0).to(self.superglue.device),
                'descriptors1': (torch.from_numpy(desc1.T).float()).unsqueeze(0).to(self.superglue.device),
                'scores0': torch.from_numpy(scores0).float().unsqueeze(0).to(self.superglue.device),
                'scores1': torch.from_numpy(scores1).float().unsqueeze(0).to(self.superglue.device),
                'image0': torch.empty(self.shapes[id0]).to(self.superglue.device),
                'image1': torch.empty(self.shapes[id1]).to(self.superglue.device)
            }
            
            # Run SuperGlue matching
            raw_matches = self.superglue.match(data)
            raw_match_count = np.sum(raw_matches != -1)
            
            # Apply RANSAC filtering
            filtered_matches = self.ransac.filter_matches(kpts0, kpts1, raw_matches)
            filtered_match_count = np.sum(filtered_matches != -1)
            
            self.logger.log(
                f"Pair {pair_idx}/{total_pairs}: {id0}-{id1} | "
                f"Matches: {filtered_match_count} (after RANSAC) / {raw_match_count} (before)", 
                "INFO"
            )
            
            # Save filtered results
            self.saver.save(output_path, id0, id1, filtered_matches)
            
            return filtered_match_count
            
        except Exception as e:
            self.logger.log(f"Failed to process pair {id0}-{id1}: {str(e)}", "ERROR")
            return 0

# ==================== Main ====================
def main():
    """Main entry point for SuperGlue feature matching"""
    parser = argparse.ArgumentParser(description='SuperGlue feature matching with RANSAC')
    parser.add_argument('--input', required=True, help='Input SfM file')
    parser.add_argument('--pairs', required=True, help='Pairs file')
    parser.add_argument('--featuresFolder', required=True, help='Features directory')
    parser.add_argument('--weights', required=True, help='Model weights path')
    parser.add_argument('--weightsType', choices=['indoor', 'outdoor'], required=True)
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--describerTypes', default='dspsift', help='Feature type')
    parser.add_argument('--matchThreshold', type=float, default=0.7)
    parser.add_argument('--sinkhornIterations', type=int, default=20)
    parser.add_argument('--forceCpu', action='store_true')
    parser.add_argument('--ransacThreshold', type=float, default=1.5, 
                       help='RANSAC inlier threshold (pixels)')
    parser.add_argument('--ransacMaxTrials', type=int, default=1000, 
                       help='Max RANSAC iterations')
    
    args = parser.parse_args()
    
    # Create configurations
    superglue_config = SuperGlueConfig(
        weights_path=args.weights,
        weights_type=args.weightsType,
        match_threshold=args.matchThreshold,
        sinkhorn_iterations=args.sinkhornIterations,
        force_cpu=args.forceCpu
    )
    
    matching_config = FeatureMatchingConfig(
        input_sfm=args.input,
        pairs_file=args.pairs,
        features_dir=args.featuresFolder,
        output_dir=args.output,
        describer_type=args.describerTypes,
        ransac_threshold=args.ransacThreshold,
        ransac_max_trials=args.ransacMaxTrials
    )
    
    # Run pipeline
    FeatureMatchingPipeline(matching_config, superglue_config).run()

if __name__ == "__main__":
    main()