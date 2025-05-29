import os
import json
import numpy as np
from collections import defaultdict
import argparse
import struct

def log_message(message):
    print(f"[HybridFeatureCombiner] {message}")

def load_features(feature_dir, view_id, is_superpoint=False):
    """Load features from either SIFT or SuperPoint format"""
    if not feature_dir:
        return np.array([]), np.array([]), 0
        
    if is_superpoint:
        feat_path = os.path.join(feature_dir, f"{view_id}.dspsift.feat")
        desc_path = os.path.join(feature_dir, f"{view_id}.dspsift.desc")
    else:
        feat_path = os.path.join(feature_dir, f"{view_id}.dspsift.feat")
        desc_path = os.path.join(feature_dir, f"{view_id}.dspsift.desc")
    
    # Load keypoints (x, y, scale, orientation)
    kpts = []
    if os.path.exists(feat_path):
        with open(feat_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    scale = float(parts[2]) if len(parts) > 2 else 1.0
                    orientation = float(parts[3]) if len(parts) > 3 else 0.0
                    kpts.append([x, y, scale, orientation])
    
    # Load descriptors
    desc = []
    desc_dim = 0
    if os.path.exists(desc_path):
        with open(desc_path, 'rb') as f:
            num_features = struct.unpack('<I', f.read(4))[0]
            desc_dim = struct.unpack('<I', f.read(4))[0]
            desc = np.fromfile(f, dtype=np.uint8, count=num_features*desc_dim)
            desc = desc.reshape(num_features, desc_dim)
    
    return np.array(kpts), np.array(desc), desc_dim

def combine_features(orig_kpts, orig_desc, orig_dim, super_kpts, super_desc, super_dim, distance_threshold=2.0):
    """Combine features from both sources, maintaining original descriptor dimensions."""
    if len(orig_kpts) == 0:
        return super_kpts, super_desc, super_dim, np.arange(len(super_kpts))
    if len(super_kpts) == 0:
        return orig_kpts, orig_desc, orig_dim, np.arange(len(orig_kpts))
    
    combined_kpts = []
    combined_desc = []
    index_mapping = []
    
    # Add all original features first
    combined_kpts.extend(orig_kpts)
    combined_desc.extend(orig_desc)
    orig_indices = list(range(len(orig_kpts)))
    
    # For each SuperPoint feature, check for duplicates
    for i, (super_kpt, super_d) in enumerate(zip(super_kpts, super_desc)):
        duplicate = False
        super_xy = super_kpt[:2]
        
        for j, orig_kpt in enumerate(orig_kpts):
            orig_xy = orig_kpt[:2]
            if np.linalg.norm(super_xy - orig_xy) < distance_threshold:
                duplicate = True
                if super_kpt[2] > orig_kpt[2]:  # Keep feature with larger scale
                    combined_kpts[j] = super_kpt
                    combined_desc[j] = super_d  # Keep full SuperPoint descriptor
                break
        
        if not duplicate:
            combined_kpts.append(super_kpt)
            combined_desc.append(super_d)  # Keep full SuperPoint descriptor
            orig_indices.append(len(orig_kpts) + i)
    
    # Determine output descriptor dimension (max of both)
    output_dim = max(orig_dim, super_dim)
    
    # Pad descriptors if needed
    if orig_dim != super_dim:
        padded_desc = []
        for i, desc in enumerate(combined_desc):
            if i < len(orig_kpts):  # Original descriptor
                if orig_dim < output_dim:
                    # Pad original descriptor with zeros
                    padded = np.zeros(output_dim, dtype=np.uint8)
                    padded[:orig_dim] = desc
                    padded_desc.append(padded)
                else:
                    padded_desc.append(desc)
            else:  # SuperPoint descriptor
                if super_dim < output_dim:
                    # Pad SuperPoint descriptor with zeros (shouldn't happen as SuperPoint is larger)
                    padded = np.zeros(output_dim, dtype=np.uint8)
                    padded[:super_dim] = desc
                    padded_desc.append(padded)
                else:
                    padded_desc.append(desc)
        combined_desc = np.array(padded_desc)
    else:
        combined_desc = np.array(combined_desc)
    
    return np.array(combined_kpts), combined_desc, output_dim, np.array(orig_indices)

def save_combined_features(output_dir, view_id, kpts, desc, desc_dim):
    """Save combined features in a unified format with mixed descriptor dimensions"""
    if len(kpts) == 0:
        return
        
    # Save keypoints
    feat_path = os.path.join(output_dir, f"{view_id}.dspsift.feat")
    with open(feat_path, 'w') as f:
        for kpt in kpts:
            f.write(f"{kpt[0]} {kpt[1]} {kpt[2]} {kpt[3]}\n")
    
    # Save descriptors
    desc_path = os.path.join(output_dir, f"{view_id}.dspsift.desc")
    with open(desc_path, 'wb') as f:
        f.write(struct.pack('<I', len(kpts)))
        f.write(struct.pack('<I', desc_dim))  # Write actual descriptor dimension
        f.write(desc.astype(np.uint8).tobytes())

from collections import defaultdict
import os

def load_matches_from_file(match_file_path):
    """Load matches from a file in the specific format shown"""
    matches = defaultdict(list)
    if not match_file_path or not os.path.exists(match_file_path):
        log_message(f"Match file not found: {match_file_path}")
        return matches
        
    log_message(f"Loading matches from: {match_file_path}")
    
    with open(match_file_path, 'r') as f:
        current_pair = None
        current_count = 0
        matches_read = 0
        line_number = 0
        state = 'looking_for_pair'  # Tracks parsing state
        
        for line in f:
            line_number += 1
            line = line.strip()
            if not line:
                continue
                
            # State machine to parse the file
            if state == 'looking_for_pair':
                # Expecting "view1 view2" format
                parts = line.split()
                if len(parts) == 2:
                    current_pair = (parts[0], parts[1])
                    state = 'looking_for_count'
                    log_message(f"Found pair: {current_pair}")
                else:
                    log_message(f"Warning: Unexpected line {line_number} (expected pair): {line}")
                    
            elif state == 'looking_for_count':
                # Expecting "1" (which we'll ignore)
                if line == '1':
                    state = 'looking_for_type'
                else:
                    log_message(f"Warning: Expected '1' on line {line_number}, got: {line}")
                    state = 'looking_for_pair'  # Reset
                    
            elif state == 'looking_for_type':
                # Expecting "dspsift N" format
                parts = line.split()
                if len(parts) == 2 and parts[0] == 'dspsift':
                    try:
                        current_count = int(parts[1])
                        matches_read = 0
                        state = 'reading_matches'
                        log_message(f"Expecting {current_count} matches")
                    except ValueError:
                        log_message(f"Warning: Invalid count on line {line_number}: {line}")
                        state = 'looking_for_pair'  # Reset
                else:
                    log_message(f"Warning: Expected 'dspsift N' on line {line_number}, got: {line}")
                    state = 'looking_for_pair'  # Reset
                    
            elif state == 'reading_matches':
                # Reading actual matches "idx1 idx2"
                parts = line.split()
                if len(parts) == 2:
                    try:
                        idx0, idx1 = map(int, parts)
                        matches[current_pair].append((idx0, idx1))
                        matches_read += 1
                        if matches_read >= current_count:
                            state = 'looking_for_pair'
                    except ValueError:
                        log_message(f"Warning: Invalid match data on line {line_number}: {line}")
                else:
                    log_message(f"Warning: Expected match pair on line {line_number}, got: {line}")
                
    log_message(f"Loaded {sum(len(v) for v in matches.values())} matches from {len(matches)} pairs")
    return matches

def combine_all_matches(orig_matches, super_matches, feature_mappings):
    """Combine all matches from both sources using combined feature indices"""
    combined_matches = defaultdict(list)
    
    # Process all pairs from both match sets
    all_pairs = set(orig_matches.keys()).union(set(super_matches.keys()))
    
    for pair in all_pairs:
        view_id0, view_id1 = pair
        orig_pair_matches = orig_matches.get(pair, [])
        super_pair_matches = super_matches.get(pair, [])
        
        # Get feature mappings for both views
        orig_mapping0, super_mapping0 = feature_mappings.get(view_id0, (np.array([]), np.array([])))
        orig_mapping1, super_mapping1 = feature_mappings.get(view_id1, (np.array([]), np.array([])))
        
        # Combine matches for this pair
        combined_pair_matches = set()
        
        # Add original matches (mapped to combined indices)
        for idx0, idx1 in orig_pair_matches:
            if idx0 < len(orig_mapping0) and idx1 < len(orig_mapping1):
                combined_pair_matches.add((orig_mapping0[idx0], orig_mapping1[idx1]))
        
        # Add SuperGlue matches (mapped to combined indices)
        for idx0, idx1 in super_pair_matches:
            if idx0 < len(super_mapping0) and idx1 < len(super_mapping1):
                combined_idx0 = super_mapping0[idx0] if idx0 < len(super_mapping0) else -1
                combined_idx1 = super_mapping1[idx1] if idx1 < len(super_mapping1) else -1
                if combined_idx0 != -1 and combined_idx1 != -1:
                    combined_pair_matches.add((combined_idx0, combined_idx1))
        
        if combined_pair_matches:
            combined_matches[pair] = list(combined_pair_matches)
    
    return combined_matches

def save_matches_to_file(output_file_path, combined_matches):
    """Save all combined matches to a 0.matches.txt file"""
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    with open(output_file_path, 'w') as f:
        for pair, matches in combined_matches.items():
            view_id0, view_id1 = pair
            f.write(f"{view_id0} {view_id1}\n")
            f.write("1\n")  # Number of matching algorithms
            f.write(f"dspsift {len(matches)}\n")
            for idx0, idx1 in matches:
                f.write(f"{idx0} {idx1}\n")

def main(args):
    log_message("Starting hybrid feature combination")
    
    # Create output directories
    os.makedirs(args.outputFeatures, exist_ok=True)
    os.makedirs(args.outputMatches, exist_ok=True)
    
    # Load SfM data
    with open(args.inputSfM, 'r') as f:
        sfm_data = json.load(f)
    views = {view['viewId']: view for view in sfm_data['views']}
    log_message(f"Loaded {len(views)} views from SfM data")
    
    # Process each view to combine features
    feature_mappings = {}  # {view_id: (orig_mapping, super_mapping)}
    
    for view_id in views:
        # Load original features from first directory only
        orig_kpts, orig_desc, orig_dim = load_features(args.inputFeatures, view_id, is_superpoint=False)
        
        # Load SuperPoint features from first directory only
        super_kpts, super_desc, super_dim = load_features(args.superpointFeatures, view_id, is_superpoint=True)
        
        # Combine features while maintaining original descriptor dimensions
        combined_kpts, combined_desc, combined_dim, orig_mapping = combine_features(
            orig_kpts, orig_desc, orig_dim,
            super_kpts, super_desc, super_dim
        )
        
        # Save combined features
        save_combined_features(args.outputFeatures, view_id, combined_kpts, combined_desc, combined_dim)
        
        # Create mapping for SuperPoint features
        super_mapping = []
        for i in range(len(super_kpts)):
            if i < len(orig_mapping) - len(orig_kpts):
                super_mapping.append(orig_mapping[len(orig_kpts) + i])
            else:
                super_xy = super_kpts[i][:2]
                for j, orig_xy in enumerate(orig_kpts[:, :2]):
                    if np.linalg.norm(super_xy - orig_xy) < 2.0:
                        super_mapping.append(orig_mapping[j])
                        break
        
        feature_mappings[view_id] = (orig_mapping, np.array(super_mapping))
    
    log_message("Feature combination complete")
    
    # Load original matches from 0.matches.txt
    orig_match_file = os.path.join(args.inputMatches, "0.matches.txt")
    orig_matches = load_matches_from_file(orig_match_file)
    log_message(f"Loaded {sum(len(v) for v in orig_matches.values())} original matches from {len(orig_matches)} pairs")
    
    # Load SuperGlue matches from 0.matches.txt
    super_match_file = os.path.join(args.superglueMatches, "0.matches.txt")
    super_matches = load_matches_from_file(super_match_file)
    log_message(f"Loaded {sum(len(v) for v in super_matches.values())} SuperGlue matches from {len(super_matches)} pairs")
    
    # Combine all matches
    combined_matches = combine_all_matches(orig_matches, super_matches, feature_mappings)
    log_message(f"Combined to {sum(len(v) for v in combined_matches.values())} matches across {len(combined_matches)} pairs")
    
    # Save combined matches to 0.matches.txt
    output_match_file = os.path.join(args.outputMatches, "0.matches.txt")
    save_matches_to_file(output_match_file, combined_matches)
    log_message("Match combination complete")
    log_message(f"Combined features saved to: {args.outputFeatures}")
    log_message(f"Combined matches saved to: {output_match_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine features and matches from SIFT and SuperPoint/SuperGlue')
    parser.add_argument('--inputSfM', required=True, help='Input SfMData file')
    parser.add_argument('--inputFeatures', required=True, help='Original feature directory (first in list will be used)')
    parser.add_argument('--inputMatches', required=True, help='Folder containing 0.matches.txt file')
    parser.add_argument('--superpointFeatures', required=True, help='SuperPoint feature directory (first in list will be used)')
    parser.add_argument('--superglueMatches', required=True, help='Folder containing 0.matches.txt file')
    parser.add_argument('--describerTypes', nargs='+', default=['dspsift'])
    parser.add_argument('--outputFeatures', required=True, help='Output directory for combined features')
    parser.add_argument('--outputMatches', required=True, help='Output folder for 0.matches.txt file')
    
    args = parser.parse_args()
    main(args)