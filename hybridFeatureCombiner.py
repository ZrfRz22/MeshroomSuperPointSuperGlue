# Standard Library 
import os
import sys
import json
import time
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Imported Libraries 
import numpy as np


# ==================== Configuration ====================
@dataclass
class HybridCombinerConfig:
    # Configuration for the hybrid feature combination pipeline
    input_sfm: str                  # Path to input SfM data file
    input_features_dir: str         # Directory containing original features (e.g., SIFT)
    input_matches_dir: str          # Directory containing original matches
    superpoint_features_dir: str    # Directory containing SuperPoint features
    superglue_matches_dir: str      # Directory containing SuperGlue matches
    output_features_dir: str        # Output directory for combined features
    output_matches_dir: str         # Output directory for combined matches
    describer_types: str = "dspsift" # Type of features being combined
    distance_threshold: float = 2.0 # Distance threshold for considering features as duplicates

# ==================== Logger ====================
class Logger:
    _instance = None  # Singleton instance

    # Simple logging class with timing capabilities
    def __init__(self):
        self.start_time = time.time()
    
    def log(self, message: str, level: str = "INFO"):
        # Log a message with timestamp and elapsed time
        elapsed = time.time() - self.start_time
        log_entry = f"[{level}][{elapsed:.2f}s] {message}"
        
        print(log_entry, file=sys.stdout)
        sys.stdout.flush()
    
    def progress(self, current: int, total: int, message: str = ""):
        # Log progress information
        progress = (current / total) * 100
        self.log(f"PROGRESS: {current}/{total} ({progress:.1f}%) {message}")

# ==================== Abstract Feature Loader ====================
class FeatureLoader(ABC):
    # Must be implemented by all feature loader classes
    @abstractmethod
    def load(self, view_id: str) -> np.ndarray:  # Changed to return only keypoints
        # Load features for a given image/view ID
        pass

# ==================== Concrete Feature Loader ====================
class DSPSiftFeatureLoader(FeatureLoader):
    # Loads DSP-SIFT features from files

    def __init__(self, features_dir: str):
        # Set the directory containing feature files
        self.features_dir = features_dir
        self.logger = Logger()

    def load(self, view_id: str) -> np.ndarray:  # Changed to return only keypoints
        # Loads keypoints for one image
        self.logger.log(f"Loading features for view {view_id}", "DEBUG")

        # Set file paths for keypoints
        feat_path = os.path.join(self.features_dir, f"{view_id}.dspsift.feat")

        # Load keypoints from the .feat file
        load_start = time.time()
        kpts = []
        try:
            with open(feat_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        x, y = float(parts[0]), float(parts[1])
                        scale = float(parts[2]) if len(parts) > 2 else 1.0
                        orientation = float(parts[3]) if len(parts) > 3 else 0.0
                        kpts.append([x, y, scale, orientation])
        except Exception as e:
            # Log and raise error if keypoints fail to load
            self.logger.log(f"Failed to load keypoints for {view_id}: {str(e)}", "ERROR")
            raise

        # Log how long it took to load
        load_time = time.time() - load_start
        self.logger.log(f"Loaded {len(kpts)} features for {view_id} in {load_time:.3f}s", "DEBUG")

        # Return keypoints
        return np.array(kpts)

# ==================== Feature Loader Factory ====================
class FeatureLoaderFactory:
    # Returns the correct loader based on the given type

    @staticmethod
    def create_loader(loader_type: str, features_dir: str) -> FeatureLoader:
        # Create a DSP-SIFT loader if type is "dspsift"
        if loader_type == "dspsift":
            return DSPSiftFeatureLoader(features_dir)
        # Raise error if loader type is unknown
        raise ValueError(f"Unknown loader type: {loader_type}")

# ==================== Abstract Feature Combiner ====================
class FeatureCombiner(ABC):
    # Base class for feature combiners
    @abstractmethod
    def combine(self, features1: np.ndarray, features2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Method to combine two sets of features (keypoints only)
        pass

# ==================== Concrete Feature Combiner ====================
class DSPSiftFeatureCombiner(FeatureCombiner):
    # Combines DSP-SIFT features and removes duplicates
    def __init__(self, distance_threshold: float = 2.0):
        self.distance_threshold = distance_threshold
        self.logger = Logger()
    
    def combine(self, features1: np.ndarray, features2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Combine features and remove duplicates based on distance
        self.logger.log("Starting feature combination", "DEBUG")
        orig_kpts = features1
        super_kpts = features2
        
        # Log number of features
        self.logger.log(f"Original features: {len(orig_kpts)}", "DEBUG")
        self.logger.log(f"SuperPoint features: {len(super_kpts)}", "DEBUG")
        
        # Handle cases with no features
        if len(orig_kpts) == 0:
            self.logger.log("No original features, returning SuperPoint features", "DEBUG")
            return super_kpts, np.arange(len(super_kpts))
        if len(super_kpts) == 0:
            self.logger.log("No SuperPoint features, returning original features", "DEBUG")
            return orig_kpts, np.arange(len(orig_kpts))
        
        # Start combining
        combine_start = time.time()
        combined_kpts = []
        index_mapping = []
        duplicates_found = 0
        
        # Add original features first
        combined_kpts.extend(orig_kpts)
        orig_indices = list(range(len(orig_kpts)))
        
        # Check each SuperPoint feature
        for i, super_kpt in enumerate(super_kpts):
            duplicate = False
            super_xy = super_kpt[:2]
            
            for j, orig_kpt in enumerate(orig_kpts):
                orig_xy = orig_kpt[:2]
                if np.linalg.norm(super_xy - orig_xy) < self.distance_threshold:
                    duplicate = True
                    duplicates_found += 1
                    # Replace if SuperPoint has larger scale
                    if super_kpt[2] > orig_kpt[2]:
                        combined_kpts[j] = super_kpt
                    break
            
            # If not duplicate, add it
            if not duplicate:
                combined_kpts.append(super_kpt)
                orig_indices.append(len(orig_kpts) + i)
        
        # Log combination result
        combine_time = time.time() - combine_start
        self.logger.log(
            f"Combined {len(orig_kpts)} + {len(super_kpts)} => {len(combined_kpts)} features "
            f"(duplicates: {duplicates_found}) in {combine_time:.3f}s", 
            "DEBUG"
        )
        
        return np.array(combined_kpts), np.array(orig_indices)

# ==================== Feature Combiner Factory ====================
class FeatureCombinerFactory:
    # Factory to create a feature combiner
    @staticmethod
    def create_combiner(combiner_type: str, distance_threshold: float = 2.0) -> FeatureCombiner:
        # Return DSP-SIFT combiner if requested
        if combiner_type == "dspsift":
            return DSPSiftFeatureCombiner(distance_threshold)
        # Raise error for unknown type
        raise ValueError(f"Unknown combiner type: {combiner_type}")

# ==================== Abstract Feature Saver ====================
class FeatureSaver(ABC):
    # Abstract base class for feature savers
    @abstractmethod
    def save(self, view_id: str, kpts: np.ndarray):
        # Save features to disk (keypoints only)
        pass

# ==================== Concrete Feature Saver ====================
class DSPSiftFeatureSaver(FeatureSaver):
    # Saver for DSP-SIFT format features
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.logger = Logger()
        os.makedirs(output_dir, exist_ok=True)
        self.logger.log(f"Initialized DSP-SIFT saver with output directory: {output_dir}", "DEBUG")
    
    def save(self, view_id: str, kpts: np.ndarray):
        # Save features in DSP-SIFT format (keypoints only)
        if len(kpts) == 0:
            self.logger.log(f"No features to save for {view_id}", "WARNING")
            return
            
        save_start = time.time()
        try:
            # Prepare file path for keypoints
            feat_path = os.path.join(self.output_dir, f"{view_id}.dspsift.feat")
            
            # Save keypoints
            with open(feat_path, 'w') as f:
                for kpt in kpts:
                    f.write(f"{kpt[0]} {kpt[1]} {kpt[2]} {kpt[3]}\n")
            
            save_time = time.time() - save_start
            self.logger.log(
                f"Saved {len(kpts)} features for {view_id} in {save_time:.3f}s", 
                "DEBUG"
            )
            
        except Exception as e:
            self.logger.log(f"Failed to save features for {view_id}: {str(e)}", "ERROR")
            raise

# ==================== Feature Saver Factory ====================
class FeatureSaverFactory:
    # Factory for creating feature savers
    @staticmethod
    def create_saver(saver_type: str, output_dir: str) -> FeatureSaver:
        # Create a feature saver instance
        if saver_type == "dspsift":
            return DSPSiftFeatureSaver(output_dir)
        raise ValueError(f"Unknown saver type: {saver_type}")

# ==================== Matches Loader ====================
class MatchesLoader:
    # Loader for match files
    def __init__(self):
        self.logger = Logger()
    
    def load(self, match_dir: str) -> Tuple[Dict[Tuple[str, str], List[Tuple[int, int]]], Dict[Tuple[str, str], str]]:
        # Load matches from directory
        self.logger.log(f"Loading matches from directory: {match_dir}", "DEBUG")
        all_matches = defaultdict(list)
        pair_to_file = {}
        
        if not match_dir or not os.path.exists(match_dir):
            self.logger.log(f"Match directory {match_dir} does not exist or is empty", "WARNING")
            return all_matches, pair_to_file
        
        # Find all match files in directory
        match_files = [f for f in os.listdir(match_dir) if f.endswith('.matches.txt')]
        if not match_files:
            self.logger.log(f"No match files found in {match_dir}", "WARNING")
            return all_matches, pair_to_file
        
        total_matches = 0
        load_start = time.time()
        
        # Process each match file
        for match_file in match_files:
            file_path = os.path.join(match_dir, match_file)
            file_matches = 0
            
            try:
                with open(file_path, 'r') as f:
                    current_pair = None
                    current_count = 0
                    matches_read = 0
                    state = 'looking_for_pair'
                    
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Parse image pair
                        if state == 'looking_for_pair':
                            parts = line.split()
                            if len(parts) == 2:
                                current_pair = (parts[0], parts[1])
                                pair_to_file[current_pair] = match_file
                                state = 'looking_for_count'
                        
                        # Parse match count
                        elif state == 'looking_for_count':
                            if line == '1':
                                state = 'looking_for_type'
                        
                        # Parse descriptor type and count
                        elif state == 'looking_for_type':
                            parts = line.split()
                            if len(parts) == 2 and parts[0] == 'dspsift':
                                current_count = int(parts[1])
                                matches_read = 0
                                state = 'reading_matches'
                        
                        # Read match pairs
                        elif state == 'reading_matches':
                            parts = line.split()
                            if len(parts) == 2:
                                idx0, idx1 = map(int, parts)
                                all_matches[current_pair].append((idx0, idx1))
                                matches_read += 1
                                file_matches += 1
                                if matches_read >= current_count:
                                    state = 'looking_for_pair'
                
                self.logger.log(f"Loaded {file_matches} matches from {match_file}", "DEBUG")
                total_matches += file_matches
                
            except Exception as e:
                self.logger.log(f"Error loading match file {match_file}: {str(e)}", "ERROR")
                continue
        
        # Log total matches loaded
        load_time = time.time() - load_start
        self.logger.log(
            f"Loaded {total_matches} matches from {len(match_files)} files in {load_time:.3f}s", 
            "DEBUG"
        )
        
        return all_matches, pair_to_file

# ==================== Matches Combiner ====================
class MatchesCombiner:
    # Combiner for matches from different sources
    def __init__(self):
        self.logger = Logger()
    
    def combine(self, orig_matches: Dict[Tuple[str, str], List[Tuple[int, int]]], 
                super_matches: Dict[Tuple[str, str], List[Tuple[int, int]]],
                feature_mappings: Dict[str, Tuple[np.ndarray, np.ndarray]],
                pair_to_file: Dict[Tuple[str, str], str]) -> Dict[str, Dict[Tuple[str, str], Set[Tuple[int, int]]]]:
        # Combine matches from original and SuperGlue sources
        self.logger.log("Starting match combination", "DEBUG")
        combine_start = time.time()
        
        # Log number of matches and pairs
        orig_pairs = len(orig_matches)
        super_pairs = len(super_matches)
        self.logger.log(f"Original matches: {sum(len(m) for m in orig_matches.values())} across {orig_pairs} pairs", "DEBUG")
        self.logger.log(f"SuperGlue matches: {sum(len(m) for m in super_matches.values())} across {super_pairs} pairs", "DEBUG")
        
        # Map match files to pairs
        file_to_pairs = defaultdict(list)
        for pair in orig_matches.keys():
            if pair in pair_to_file:
                file_to_pairs[pair_to_file[pair]].append(pair)
        
        combined_results = defaultdict(dict)
        total_combined_matches = 0
        total_pairs = 0
        
        # Process each match file and its pairs
        for match_file, pairs in file_to_pairs.items():
            for pair in pairs:
                view_id0, view_id1 = pair
                total_pairs += 1
                
                # Get original and SuperGlue matches
                orig_pair_matches = orig_matches.get(pair, [])
                super_pair_matches = super_matches.get(pair, [])
                
                # Get feature index mappings
                orig_mapping0, super_mapping0 = feature_mappings.get(view_id0, (np.array([]), np.array([])))
                orig_mapping1, super_mapping1 = feature_mappings.get(view_id1, (np.array([]), np.array([])))
                
                combined_matches = set()
                
                # Add original matches
                for idx0, idx1 in orig_pair_matches:
                    if idx0 < len(orig_mapping0) and idx1 < len(orig_mapping1):
                        combined_matches.add((orig_mapping0[idx0], orig_mapping1[idx1]))
                
                # Add SuperGlue matches
                for idx0, idx1 in super_pair_matches:
                    if idx0 < len(super_mapping0) and idx1 < len(super_mapping1):
                        combined_idx0 = super_mapping0[idx0]
                        combined_idx1 = super_mapping1[idx1]
                        combined_matches.add((combined_idx0, combined_idx1))
                
                # Store combined matches
                if combined_matches:
                    combined_results[match_file][pair] = combined_matches
                    total_combined_matches += len(combined_matches)
        
        # Log total combined stats
        combine_time = time.time() - combine_start
        self.logger.log(
            f"Combined {total_pairs} pairs into {total_combined_matches} matches in {combine_time:.3f}s", 
            "DEBUG"
        )
        
        return combined_results

# ==================== Matches Saver ====================
class MatchesSaver:
    # Saver for combined matches
    def __init__(self):
        self.logger = Logger()
    
    def save(self, combined_results: Dict[str, Dict[Tuple[str, str], Set[Tuple[int, int]]]], output_dir: str):
        # Save combined matches to disk
        self.logger.log(f"Saving combined matches to {output_dir}", "DEBUG")
        save_start = time.time()
        total_files = 0
        total_pairs = 0
        total_matches = 0
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Write each match file
            for match_file, pairs_matches in combined_results.items():
                output_path = os.path.join(output_dir, match_file)
                total_files += 1
                
                with open(output_path, 'w') as f:
                    for pair, matches in pairs_matches.items():
                        view_id0, view_id1 = pair
                        total_pairs += 1
                        total_matches += len(matches)
                        
                        f.write(f"{view_id0} {view_id1}\n")
                        f.write("1\n")
                        f.write(f"dspsift {len(matches)}\n")
                        for idx0, idx1 in matches:
                            f.write(f"{idx0} {idx1}\n")
                
                # Log after saving each file
                self.logger.log(f"Saved {len(pairs_matches)} pairs to {match_file}", "DEBUG")
            
            # Log final saving stats
            save_time = time.time() - save_start
            self.logger.log(
                f"Saved {total_matches} matches across {total_pairs} pairs in {total_files} files in {save_time:.3f}s", 
                "DEBUG"
            )
        
        except Exception as e:
            # Log error if saving fails
            self.logger.log(f"Failed to save matches: {str(e)}", "ERROR")
            raise

# ==================== Hybrid Feature Combiner Pipeline ====================
class HybridFeatureCombiner:
    # Main pipeline for combining features and matches
    def __init__(self, config: HybridCombinerConfig):
        self.logger = Logger()
        self.config = config
        
        # Initialize components
        self.sift_loader = FeatureLoaderFactory.create_loader(config.describer_types, config.input_features_dir)
        self.superpoint_loader = FeatureLoaderFactory.create_loader(config.describer_types, config.superpoint_features_dir)
        self.feature_combiner = FeatureCombinerFactory.create_combiner(config.describer_types, config.distance_threshold)
        self.feature_saver = FeatureSaverFactory.create_saver(config.describer_types, config.output_features_dir)
        
        self.matches_loader = MatchesLoader()
        self.matches_combiner = MatchesCombiner()
        self.matches_saver = MatchesSaver()
    
    def run(self):
        # Run the whole feature and match combination process
        start_time = time.time()
        self.logger.log("Starting hybrid feature combination")
        
        # Load SfM views
        with open(self.config.input_sfm, 'r') as f:
            sfm_data = json.load(f)
        views = {view['viewId']: view for view in sfm_data['views']}
        self.logger.log(f"Loaded {len(views)} views from SfM data")
        
        # Combine features for each view
        feature_mappings = {}
        total_original_features = 0
        total_super_features = 0
        total_combined_features = 0
        
        for i, view_id in enumerate(views):
            self.logger.progress(i+1, len(views), f"Combining features for view {view_id}")
            
            # Load features
            load_start = time.time()
            orig_kpts = self.sift_loader.load(view_id)
            super_kpts = self.superpoint_loader.load(view_id)
            load_time = time.time() - load_start
            
            self.logger.log(f"Loaded {len(orig_kpts)} original and {len(super_kpts)} SuperPoint features for {view_id} in {load_time:.2f}s")
            total_original_features += len(orig_kpts)
            total_super_features += len(super_kpts)
            
            # Combine features
            combine_start = time.time()
            combined_kpts, orig_mapping = self.feature_combiner.combine(orig_kpts, super_kpts)
            combine_time = time.time() - combine_start
            
            self.logger.log(f"Combined to {len(combined_kpts)} features for {view_id} in {combine_time:.2f}s")
            total_combined_features += len(combined_kpts)
            
            # Save combined features
            save_start = time.time()
            self.feature_saver.save(view_id, combined_kpts)
            save_time = time.time() - save_start
            self.logger.log(f"Saved combined features for {view_id} in {save_time:.2f}s")
            
            # Map SuperPoint features to combined indices
            super_mapping = []
            for i in range(len(super_kpts)):
                if i < len(orig_mapping) - len(orig_kpts):
                    super_mapping.append(orig_mapping[len(orig_kpts) + i])
                else:
                    super_xy = super_kpts[i][:2]
                    for j, orig_xy in enumerate(orig_kpts[:, :2]):
                        if np.linalg.norm(super_xy - orig_xy) < self.config.distance_threshold:
                            super_mapping.append(orig_mapping[j])
                            break
            
            feature_mappings[view_id] = (orig_mapping, np.array(super_mapping))
        
        # Log feature stats
        self.logger.log("\n=== Feature Combination Statistics ===")
        self.logger.log(f"Total original features: {total_original_features}")
        self.logger.log(f"Total SuperPoint features: {total_super_features}")
        self.logger.log(f"Total combined features: {total_combined_features}")
        self.logger.log(f"Feature reduction: {(total_original_features + total_super_features - total_combined_features) / (total_original_features + total_super_features) * 100:.1f}%")
        self.logger.log("Feature combination complete")
        
        # Load and combine matches
        self.logger.log("\nStarting match combination")
        match_load_start = time.time()
        orig_matches, pair_to_file = self.matches_loader.load(self.config.input_matches_dir)
        super_matches, _ = self.matches_loader.load(self.config.superglue_matches_dir)
        match_load_time = time.time() - match_load_start
        
        self.logger.log(f"Loaded {len(orig_matches)} original and {len(super_matches)} SuperGlue match sets in {match_load_time:.2f}s")
        
        # Combine matches
        combine_start = time.time()
        combined_results = self.matches_combiner.combine(
            orig_matches, super_matches, feature_mappings, pair_to_file
        )
        combine_time = time.time() - combine_start
        
        # Match statistics
        total_orig_matches = sum(len(m) for m in orig_matches.values())
        total_super_matches = sum(len(m) for m in super_matches.values())
        total_combined_matches = sum(len(m) for pairs in combined_results.values() for m in pairs.values())
        
        self.logger.log("\n=== Match Combination Statistics ===")
        self.logger.log(f"Total original matches: {total_orig_matches}")
        self.logger.log(f"Total SuperGlue matches: {total_super_matches}")
        self.logger.log(f"Total combined matches: {total_combined_matches}")
        self.logger.log(f"Match combination completed in {combine_time:.2f}s")
        
        # Save matches
        save_start = time.time()
        self.matches_saver.save(combined_results, self.config.output_matches_dir)
        save_time = time.time() - save_start
        self.logger.log(f"Saved combined matches in {save_time:.2f}s")
        
        # Final log
        total_time = time.time() - start_time
        self.logger.log("\n=== Hybrid Combination Complete ===")
        self.logger.log(f"Total processing time: {total_time:.2f} seconds")
        self.logger.log(f"Combined features saved to: {self.config.output_features_dir}")
        self.logger.log(f"Combined matches saved to: {self.config.output_matches_dir}")

# ==================== Main ====================
def main():
    """Main entry point for the hybrid feature combiner"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Combine features and matches from SIFT and SuperPoint/SuperGlue')
    parser.add_argument('--inputSfM', required=True, help='Input SfMData file')
    parser.add_argument('--inputFeatures', required=True, help='Original feature directory')
    parser.add_argument('--inputMatches', required=True, help='Folder containing match files')
    parser.add_argument('--superpointFeatures', required=True, help='SuperPoint feature directory')
    parser.add_argument('--superglueMatches', required=True, help='Folder containing SuperGlue match files')
    parser.add_argument('--outputFeatures', required=True, help='Output directory for combined features')
    parser.add_argument('--outputMatches', required=True, help='Output directory for combined matches')
    parser.add_argument('--describerTypes', default='dspsift', help='Feature type')
    
    # Get arguments from parser
    args = parser.parse_args()
    
    # Create configuration object with provided arguments
    config = HybridCombinerConfig(
        input_sfm=args.inputSfM,
        input_features_dir=args.inputFeatures,
        input_matches_dir=args.inputMatches,
        superpoint_features_dir=args.superpointFeatures,
        superglue_matches_dir=args.superglueMatches,
        output_features_dir=args.outputFeatures,
        output_matches_dir=args.outputMatches,
        describer_types=args.describerTypes
    )
    
    # Create the hybrid combiner with the config
    combiner = HybridFeatureCombiner(config)
    
    # Run the hybrid combination pipeline
    combiner.run()

# Run main if this script is executed
if __name__ == "__main__":
    main()