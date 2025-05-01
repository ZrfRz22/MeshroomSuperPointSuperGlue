import os
import numpy as np
import torch
import json
import argparse
from superglue import SuperGlue
from itertools import combinations
import struct
import sys
import time

def log_message(message, flush=True):
    print(f"[DEBUG][{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stdout)
    if flush:
        sys.stdout.flush()

def dpsift_load_features(features_dir, view_id):
    """Load features in DpSift format with proper path handling"""
    feat_path = os.path.join(features_dir, f"{view_id}.feat")
    desc_path = os.path.join(features_dir, f"{view_id}.desc")
    
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    if not os.path.exists(desc_path):
        raise FileNotFoundError(f"Descriptor file not found: {desc_path}")
    
    log_message(f"Loading features for view {view_id} from:\n- {feat_path}\n- {desc_path}")
    
    # Load keypoints
    with open(feat_path, 'rb') as f:
        num_kpts = struct.unpack('<I', f.read(4))[0]
        kpts_data = np.fromfile(f, dtype=np.float32, count=num_kpts*4)
        kpts = kpts_data.reshape(-1, 4)[:, :2]
    
    # Load descriptors
    with open(desc_path, 'rb') as f:
        num_desc = struct.unpack('<I', f.read(4))[0]
        desc_dim = struct.unpack('<I', f.read(4))[0]
        desc = np.fromfile(f, dtype=np.uint8, count=num_desc*desc_dim).reshape(-1, desc_dim)
    
    log_message(f"Loaded {len(kpts)} keypoints and {desc.shape} descriptors for view {view_id}")
    return kpts, desc

def dpsift_save_matches(output_path, view_id0, view_id1, matches):
    """Save matches in DpSift format with sequential numbering and dspsift label"""
    valid_matches = [(i, m) for i, m in enumerate(matches) if m != -1]
    log_message(f"Saving {len(valid_matches)} matches between {view_id0} and {view_id1} to {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"{view_id0} {view_id1}\n")
        f.write("1\n")
        f.write(f"dspsift {len(valid_matches)}\n")  # Changed from superglue to dspsift
        for i, j in valid_matches:
            f.write(f"{i} {j}\n")
    log_message("Matches saved successfully")

def main():    
    parser = argparse.ArgumentParser(description='SuperGlue feature matching')
    parser.add_argument('--input', required=True, help='Input SfMData file')
    parser.add_argument('--pairs', required=True, help='File containing image pairs')
    parser.add_argument('--features', required=True, nargs='+', help='Feature directories')
    parser.add_argument('--output', required=True, help='Output directory for matches')
    parser.add_argument('--weights', required=True, help='Path to SuperGlue weights')
    parser.add_argument('--weightsType', choices=['indoor', 'outdoor'], required=True)
    parser.add_argument('--matchThreshold', type=float, default=0.7)
    parser.add_argument('--sinkhornIterations', type=int, default=20)
    parser.add_argument('--describerType', choices=['dspsift'], default='dspsift')
    parser.add_argument('--forceCpu', action='store_true')
    args = parser.parse_args()

    log_message(f"Starting SuperGlue matching with args: {vars(args)}")

    # Clean and validate feature paths
    features_dirs = [d for d in args.features if d.strip()]
    if not features_dirs:
        log_message("ERROR: No valid feature directories provided")
        return
    features_dir = features_dirs[0]  # Use first valid path
    
    if not os.path.exists(features_dir):
        log_message(f"ERROR: Feature directory does not exist: {features_dir}")
        return

    # Setup
    device = 'cuda' if torch.cuda.is_available() and not args.forceCpu else 'cpu'
    log_message(f"Using {args.describerType} features on {device.upper()} from directory: {features_dir}")

    # Model config
    config = {
        'weights': args.weightsType,
        'match_threshold': args.matchThreshold,
        'sinkhorn_iterations': args.sinkhornIterations
    }
    log_message(f"Model config: {config}")
    
    model = SuperGlue(config, args.weights).eval().to(device)
    log_message("Model initialized and set to eval mode")

    # Read pairs
    with open(args.pairs) as f:
        pairs = [tuple(line.strip().split()[:2]) for line in f if line.strip()]
    log_message(f"Loaded {len(pairs)} image pairs from {args.pairs}")

    os.makedirs(args.output, exist_ok=True)
    log_message(f"Created output directory: {args.output}")

    # Get image shapes
    with open(args.input) as f:
        sfm_data = json.load(f)
        shapes = {view['viewId']: (1, 1, int(view['height']), int(view['width'])) 
                 for view in sfm_data['views']}
    log_message(f"Loaded image shapes for {len(shapes)} views")

    # Process pairs
    for idx, (id0, id1) in enumerate(pairs):
        try:
            log_message(f"\nProcessing pair {idx+1}/{len(pairs)}: {id0} - {id1}")
            
            # Load features with proper path handling
            kpts0, desc0 = dpsift_load_features(features_dir, id0)
            kpts1, desc1 = dpsift_load_features(features_dir, id1)
            
            # Convert descriptors to float32 in [0,1] range
            desc0 = torch.from_numpy(desc0.T).float() / 255.0
            desc1 = torch.from_numpy(desc1.T).float() / 255.0

            # Prepare input tensor
            data = {
                'keypoints0': torch.from_numpy(kpts0).float().unsqueeze(0).to(device),
                'keypoints1': torch.from_numpy(kpts1).float().unsqueeze(0).to(device),
                'descriptors0': desc0.unsqueeze(0).to(device),
                'descriptors1': desc1.unsqueeze(0).to(device),
                'scores0': torch.ones(1, len(kpts0)).to(device),
                'scores1': torch.ones(1, len(kpts1)).to(device),
                'image0': torch.empty(shapes[id0]).to(device),
                'image1': torch.empty(shapes[id1]).to(device)
            }

            # Match features
            with torch.no_grad():
                pred = model(data)
            matches = pred['matches0'][0].cpu().numpy()
            num_matches = np.sum(matches != -1)
            log_message(f"Found {num_matches} matches between {id0} and {id1}")

            # Save matches with sequential numbering
            output_path = os.path.join(args.output, f"{idx}.matches.txt")  # Changed to sequential numbering
            dpsift_save_matches(output_path, id0, id1, matches)

        except Exception as e:
            log_message(f"ERROR processing pair {id0}-{id1}: {str(e)}", flush=True)
            continue

    log_message("\nMatching process completed")

if __name__ == "__main__":
    main()