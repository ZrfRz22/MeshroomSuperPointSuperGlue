import os
import numpy as np
import torch
import json
import struct
import sys
import time
from superglue import SuperGlue

def log_message(message, flush=True):
    print(f"[DEBUG][{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stdout)
    if flush:
        sys.stdout.flush()

def load_all_pairs(pairs_file):
    all_pairs = set()  # Use a set to automatically handle duplicate pairs
    with open(pairs_file) as f:
        for line in f:
            line = line.strip()
            if line:
                image_ids = line.split()
                if len(image_ids) >= 2:
                    for i in range(len(image_ids)):
                        for j in range(i + 1, len(image_ids)):
                            # Ensure consistent ordering within the pair (lexicographically)
                            pair = tuple(sorted((image_ids[i], image_ids[j])))
                            all_pairs.add(pair)
    log_message(f"Loaded {len(all_pairs)} unique image pairs from {pairs_file}")
    return list(all_pairs)

def dpsift_load_features(features_dir, view_id):
    feat_path = os.path.join(features_dir, f"{view_id}.dspsift.feat")
    desc_path = os.path.join(features_dir, f"{view_id}.dspsift.desc")
    conf_path = os.path.join(features_dir, f"{view_id}.confidence.txt")

    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    if not os.path.exists(desc_path):
        raise FileNotFoundError(f"Descriptor file not found: {desc_path}")
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"Confidence file not found: {conf_path}")

    log_message(f"Loading features for view {view_id} from:\n- {feat_path}\n- {desc_path}\n- {conf_path}")

    # Read .feat file (text format: x y scale orientation)
    with open(feat_path, 'r') as f:
        kpts_data = []
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                kpts_data.append([float(parts[0]), float(parts[1])])
    kpts = np.array(kpts_data, dtype=np.float32)

    # Read .desc file (binary format: num_desc, desc_dim, descriptors)
    with open(desc_path, 'rb') as f:
        num_desc = struct.unpack('<I', f.read(4))[0]
        desc_dim = struct.unpack('<I', f.read(4))[0]
        desc = np.fromfile(f, dtype=np.uint8, count=num_desc*desc_dim).reshape(-1, desc_dim)

    # Read confidence scores
    with open(conf_path, 'r') as f:
        scores = np.array([float(line.strip()) for line in f if line.strip()])

    # Validate that all files have consistent number of features
    if len(kpts) != num_desc or len(kpts) != len(scores):
        raise ValueError(f"Mismatch between features: keypoints ({len(kpts)}), descriptors ({num_desc}), scores ({len(scores)})")

    log_message(f"Loaded {len(kpts)} keypoints, {desc.shape} descriptors, and {len(scores)} scores for view {view_id}")
    return kpts, desc, scores

def dpsift_save_matches(output_path, view_id0, view_id1, matches):
    valid_matches = [(i, m) for i, m in enumerate(matches) if m != -1]
    log_message(f"Saving {len(valid_matches)} matches between {view_id0} and {view_id1} to {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"{view_id0} {view_id1}\n")
        f.write("1\n")
        f.write(f"dspsift {len(valid_matches)}\n")
        for i, j in valid_matches:
            f.write(f"{i} {j}\n")
    log_message("Matches saved successfully")

def main(args):
    log_message(f"Starting SuperGlue matching with args: {args}")

    features_dirs = [d for d in args['featuresFolder'] if d.strip()]
    if not features_dirs:
        log_message("ERROR: No valid feature directories provided")
        return
    features_dir = features_dirs[0]

    if not os.path.exists(features_dir):
        log_message(f"ERROR: Feature directory does not exist: {features_dir}")
        return

    device = 'cuda' if torch.cuda.is_available() and not args.get('forceCpu', False) else 'cpu'

    describer_dirs = args['describerTypes']  # Access the list of describer types
    first_describer_dir = describer_dirs[0]  # Access the first describer type in the list
    log_message(f"Using {first_describer_dir} features on {device.upper()}")

    config = {
        'weights': args['weightsType'],
        'match_threshold': args.get('matchThreshold', 0.7),
        'sinkhorn_iterations': args.get('sinkhornIterations', 20)
    }

    model = SuperGlue(config, args['weights']).eval().to(device)
    log_message("Model initialized and set to eval mode")

    pairs = load_all_pairs(args['pairs'])

    os.makedirs(args['output'], exist_ok=True)
    log_message(f"Created output directory: {args['output']}")

    with open(args['input']) as f:
        sfm_data = json.load(f)
        shapes = {view['viewId']: (1, 1, int(view['height']), int(view['width'])) 
                  for view in sfm_data['views']}
    log_message(f"Loaded image shapes for {len(shapes)} views")

    for idx, (id0, id1) in enumerate(pairs):
        try:
            log_message(f"\nProcessing pair {idx+1}/{len(pairs)}: {id0} - {id1}")

            kpts0, desc0, scores0 = dpsift_load_features(features_dir, id0)
            kpts1, desc1, scores1 = dpsift_load_features(features_dir, id1)

            desc0 = torch.from_numpy(desc0.T).float() / 255.0
            desc1 = torch.from_numpy(desc1.T).float() / 255.0

            data = {
                'keypoints0': torch.from_numpy(kpts0).float().unsqueeze(0).to(device),
                'keypoints1': torch.from_numpy(kpts1).float().unsqueeze(0).to(device),
                'descriptors0': desc0.unsqueeze(0).to(device),
                'descriptors1': desc1.unsqueeze(0).to(device),
                'scores0': torch.from_numpy(scores0).float().unsqueeze(0).to(device),
                'scores1': torch.from_numpy(scores1).float().unsqueeze(0).to(device),
                'image0': torch.empty(shapes[id0]).to(device),
                'image1': torch.empty(shapes[id1]).to(device)
            }

            with torch.no_grad():
                pred = model(data)
            matches = pred['matches0'][0].cpu().numpy()
            num_matches = np.sum(matches != -1)
            log_message(f"Found {num_matches} matches between {id0} and {id1}")

            output_path = os.path.join(args['output'], f"{idx}.matches.txt")
            dpsift_save_matches(output_path, id0, id1, matches)

        except Exception as e:
            log_message(f"ERROR processing pair {id0}-{id1}: {str(e)}", flush=True)
            continue

    log_message("\nMatching process completed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SuperGlue feature matching')
    parser.add_argument('--input', required=True, help='Input SfMData file')
    parser.add_argument('--pairs', required=True, help='File containing image pairs')
    parser.add_argument('--featuresFolder', required=True, nargs='+', help='Feature directories')
    parser.add_argument('--output', required=True, help='Output directory for matches')
    parser.add_argument('--weights', required=True, help='Path to SuperGlue weights')
    parser.add_argument('--weightsType', choices=['indoor', 'outdoor'], required=True)
    parser.add_argument('--matchThreshold', type=float, default=0.7)
    parser.add_argument('--sinkhornIterations', type=int, default=20)
    parser.add_argument('--describerTypes', nargs='+', default=['dspsift'])
    parser.add_argument('--forceCpu', action='store_true')
    
    args = parser.parse_args()
    main(vars(args))