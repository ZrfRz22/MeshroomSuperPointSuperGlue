import os
import numpy as np
import torch
import json
import argparse
import superglue as SuperGlue

def load_meshroom_features(features_dir, view_id):
    """Load features from Meshroom's binary format"""
    feat_path = os.path.join(features_dir, f"{view_id}.feat")
    desc_path = os.path.join(features_dir, f"{view_id}.desc")
    
    # Keypoints: [x, y, scale, orientation, response, octave]
    kpts = np.fromfile(feat_path, dtype=np.float32).reshape(-1, 6)[:, :2]  # Keep only x,y
    desc = np.fromfile(desc_path, dtype=np.float32).reshape(-1, 256)  # Descriptors
    return kpts, desc

def save_meshroom_matches(matches, output_path):
    """Save matches in AliceVision binary format"""
    valid_matches = matches[matches != -1]
    with open(output_path, 'wb') as f:
        np.array(len(valid_matches), dtype=np.uint32).tofile(f)
        for i, j in enumerate(matches):
            if j != -1:
                np.array([i, j], dtype=np.uint32).tofile(f)

def read_image_pairs(pairs_path):
    """Read viewId pairs from text file"""
    with open(pairs_path) as f:
        return [line.strip().split() for line in f if line.strip()]

def get_image_shapes(sfm_data, view_ids):
    """Get image dimensions from SFM data"""
    shapes = {}
    for view in sfm_data['views']:
        if view['viewId'] in view_ids:
            # In a real implementation, you would read the actual image dimensions
            # Here we return a dummy size (you should replace this with actual image reading)
            shapes[view['viewId']] = (1, 1, 1080, 1920)  # (batch, channel, height, width)
    return shapes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Path to input.sfm")
    parser.add_argument('--pairs', required=True, help="Path to pairs file")
    parser.add_argument('--features', required=True, help="Features directory")
    parser.add_argument('--output', required=True, help="Output directory")
    parser.add_argument('--weights', required=True, help="SuperGlue weights path")
    parser.add_argument('--matchThreshold', type=float, default=0.5)
    parser.add_argument('--forceCpu', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and not args.forceCpu else 'cpu'
    
    # Load SFM data for image paths
    with open(args.input) as f:
        sfm_data = json.load(f)
    
    # Initialize SuperGlue
    config = {
        'weights': args.weights,
        'match_threshold': args.matchThreshold,
        'sinkhorn_iterations': 100
    }
    superglue = SuperGlue(config).eval().to(device)

    # Process all pairs
    pairs = read_image_pairs(args.pairs)
    os.makedirs(args.output, exist_ok=True)
    
    # Get image shapes for all views in pairs
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
                'image0': torch.empty(image_shapes[view_id0]).to(device),  # Dummy image tensor
                'image1': torch.empty(image_shapes[view_id1]).to(device)   # Dummy image tensor
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