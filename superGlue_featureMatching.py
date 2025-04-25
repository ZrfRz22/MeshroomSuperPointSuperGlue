import os
import numpy as np
import torch
import json
import argparse
from superglue import SuperGlue
from itertools import combinations

def load_meshroom_features(features_dirs, view_id):
    """Load features from Meshroom's binary format, searching multiple directories"""
    for features_dir in features_dirs:
        feat_path = os.path.join(features_dir, f"{view_id}.feat")
        desc_path = os.path.join(features_dir, f"{view_id}.desc")
        
        if os.path.exists(feat_path) and os.path.exists(desc_path):
            # Keypoints: [x, y] - already in correct format from SuperPoint
            kpts = np.fromfile(feat_path, dtype=np.float32).reshape(-1, 2)
            
            # Descriptors: [N, 256] - already in correct format from modified SuperPoint
            desc = np.fromfile(desc_path, dtype=np.float32).reshape(-1, 256)
            
            return kpts, desc
    
    raise FileNotFoundError(f"Could not find feature files for view {view_id} in any of the provided directories")

def save_meshroom_matches(matches, output_path):
    """Save matches in AliceVision binary format"""
    valid_matches = matches[matches != -1]
    with open(output_path, 'wb') as f:
        np.array(len(valid_matches), dtype=np.uint32).tofile(f)
        for i, j in enumerate(matches):
            if j != -1:
                np.array([i, j], dtype=np.uint32).tofile(f)


def read_image_pairs(pairs_path):
    """Read pairwise combinations from lines with multiple IDs"""
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            ids = line.strip().split()
            if len(ids) >= 2:
                # Make all pairwise combinations
                pairs.extend(list(combinations(ids, 2)))
    return pairs


def get_image_shapes(sfm_data, view_ids):
    """Get image dimensions from SFM data"""
    shapes = {}
    for view in sfm_data['views']:
        if view['viewId'] in view_ids:
            shapes[view['viewId']] = (1, 1, 1080, 1920)  # (batch, channel, height, width)
    return shapes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Path to input.sfm")
    parser.add_argument('--pairs', required=True, help="Path to pairs file")
    parser.add_argument('--features', required=True, nargs='+', help="One or more feature directories")
    parser.add_argument('--output', required=True, help="Output directory")
    parser.add_argument('--weights', required=True, help="SuperGlue weights path")
    parser.add_argument('--weightsType', required=True, choices=['indoor', 'outdoor'], help="Type of weights used (indoor/outdoor)")
    parser.add_argument('--matchThreshold', type=float, default=0.5)
    parser.add_argument('--forceCpu', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and not args.forceCpu else 'cpu'

    # Load SFM data
    with open(args.input) as f:
        sfm_data = json.load(f)

    # Initialize SuperGlue
    config = {
        'weights': args.weightsType,
        'match_threshold': args.matchThreshold,
        'sinkhorn_iterations': 100
    }
    superglue = SuperGlue(config, args.weights).eval().to(device)

    # Read image pairs
    pairs = read_image_pairs(args.pairs)
    os.makedirs(args.output, exist_ok=True)

    # Get image shapes
    view_ids = set()
    for pair in pairs:
        view_ids.update(pair)
    image_shapes = get_image_shapes(sfm_data, view_ids)

    for view_id0, view_id1 in pairs:
        try:
            kpts0, desc0 = load_meshroom_features(args.features, view_id0)
            kpts1, desc1 = load_meshroom_features(args.features, view_id1)

            data = {
                'keypoints0': torch.from_numpy(kpts0).float().unsqueeze(0).to(device),
                'keypoints1': torch.from_numpy(kpts1).float().unsqueeze(0).to(device),
                'descriptors0': torch.from_numpy(desc0.T).float().unsqueeze(0).to(device),
                'descriptors1': torch.from_numpy(desc1.T).float().unsqueeze(0).to(device),
                'scores0': torch.ones(1, len(kpts0)).to(device),
                'scores1': torch.ones(1, len(kpts1)).to(device),
                'image0': torch.empty(image_shapes[view_id0]).to(device),
                'image1': torch.empty(image_shapes[view_id1]).to(device)
            }

            with torch.no_grad():
                pred = superglue(data)

            matches = pred['matches0'][0].cpu().numpy()
            output_path = os.path.join(args.output, f"{view_id0}_{view_id1}.matches")
            save_meshroom_matches(matches, output_path)

            print(f"Generated matches for {view_id0}-{view_id1}: {len(matches[matches != -1])} matches")

        except Exception as e:
            print(f"Failed to match {view_id0}-{view_id1}: {str(e)}")

if __name__ == "__main__":
    main()
