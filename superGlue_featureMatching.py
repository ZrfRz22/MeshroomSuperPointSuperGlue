# ==================== SuperGlue Implementation ====================
import os
import numpy as np
import torch
import json
import struct
import sys
import time
import argparse
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from superglue import SuperGlue

# ==================== SuperGlue Implementation ====================
"""
SuperGlue Feature Matcher Implementation

This module implements a pipeline for feature matching using the SuperGlue model.
It includes configuration, logging, feature loading, matching, and saving components.
"""

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
    """Loader for DSP-SIFT format features"""
    def load(self, features_dir: str, view_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load features from DSP-SIFT format files"""
        logger = Logger()
        
        # Define file paths
        feat_path = os.path.join(features_dir, f"{view_id}.dspsift.feat")
        desc_path = os.path.join(features_dir, f"{view_id}.dspsift.desc")
        conf_path = os.path.join(features_dir, f"{view_id}.confidence.txt")

        # Load keypoints (x, y)
        with open(feat_path, 'r') as f:
            kpts = np.array([list(map(float, line.strip().split()[:2])) for line in f], 
                          dtype=np.float32)

        # Load descriptors (binary format)
        with open(desc_path, 'rb') as f:
            num_desc = struct.unpack('<I', f.read(4))[0]  # Number of descriptors
            desc_dim = struct.unpack('<I', f.read(4))[0]  # Descriptor dimension
            desc = np.fromfile(f, dtype=np.uint8, count=num_desc*desc_dim).reshape(-1, desc_dim)

        # Load scores (text format)
        with open(conf_path, 'r') as f:
            scores = np.array([float(line.strip()) for line in f])

        logger.log(f"Loaded {len(kpts)} features for {view_id}", "INFO")
        return kpts, desc, scores

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
    """Pipeline for SuperGlue feature matching"""
    def __init__(self, config: FeatureMatchingConfig, superglue_config: SuperGlueConfig):
        self.logger = Logger()
        self.config = config
        self.superglue = SuperGlueMatcher(superglue_config)
        self.loader = FeatureLoaderFactory.create_loader(config.describer_type)
        self.saver = MatchSaver(config.describer_type)
        
        # Load image shapes from SfM data
        with open(config.input_sfm) as f:
            self.shapes = {view['viewId']: (1, 1, int(view['height']), int(view['width'])) 
                          for view in json.load(f)['views']}
        
        os.makedirs(config.output_dir, exist_ok=True)
        self.logger.log("Feature matching pipeline initialized", "INFO")
    
    def run(self):
        """Run the feature matching pipeline"""
        pairs = self._load_pairs()
        output_path = os.path.join(self.config.output_dir, "0.matches.txt")
        
        # Clear existing matches file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        
        total_matches = 0
        start_time = time.time()
        
        for idx, (id0, id1) in enumerate(pairs, 1):
            num_matches = self._process_pair(idx, len(pairs), id0, id1, output_path)
            total_matches += num_matches
        
        # Final statistics
        total_time = time.time() - start_time
        self.logger.log("\n=== Matching Complete ===", "INFO")
        self.logger.log(f"Total pairs processed: {len(pairs)}", "INFO")
        self.logger.log(f"Total matches found: {total_matches}", "INFO")
        self.logger.log(f"Average matches per pair: {total_matches/len(pairs):.1f}", "INFO")
        self.logger.log(f"Total processing time: {total_time:.2f} seconds", "INFO")
        self.logger.log(f"Processing rate: {len(pairs)/total_time:.2f} pairs/second", "INFO")
        self.logger.log("Matching completed successfully", "INFO")

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
        logger = Logger()
        logger.log(f"Processing pair {pair_idx}/{total_pairs}: {id0} vs {id1}", "INFO")
        
        try:
            # Load features for both images
            kpts0, desc0, scores0 = self.loader.load(self.config.features_dir, id0)
            kpts1, desc1, scores1 = self.loader.load(self.config.features_dir, id1)
            
            # Prepare input data dictionary for SuperGlue
            data = {
                'keypoints0': torch.from_numpy(kpts0).float().unsqueeze(0).to(self.superglue.device),
                'keypoints1': torch.from_numpy(kpts1).float().unsqueeze(0).to(self.superglue.device),
                'descriptors0': (torch.from_numpy(desc0.T).float() / 255.0).unsqueeze(0).to(self.superglue.device),
                'descriptors1': (torch.from_numpy(desc1.T).float() / 255.0).unsqueeze(0).to(self.superglue.device),
                'scores0': torch.from_numpy(scores0).float().unsqueeze(0).to(self.superglue.device),
                'scores1': torch.from_numpy(scores1).float().unsqueeze(0).to(self.superglue.device),
                'image0': torch.empty(self.shapes[id0]).to(self.superglue.device),
                'image1': torch.empty(self.shapes[id1]).to(self.superglue.device)
            }
            
            # Run SuperGlue matching
            matches = self.superglue.match(data)
            num_matches = np.sum(matches != -1)
            logger.log(f"Found {num_matches} matches", "INFO")
            
            # Save results
            self.saver.save(output_path, id0, id1, matches)
            
            return num_matches
            
        except Exception as e:
            logger.log(f"Failed to process pair {id0}-{id1}: {str(e)}", "ERROR")
            return 0

# ==================== Main ====================
def main():
    """Main entry point for SuperGlue feature matching"""
    parser = argparse.ArgumentParser(description='SuperGlue feature matching')
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
        describer_type=args.describerTypes
    )
    
    # Run pipeline
    FeatureMatchingPipeline(matching_config, superglue_config).run()

if __name__ == "__main__":
    main()