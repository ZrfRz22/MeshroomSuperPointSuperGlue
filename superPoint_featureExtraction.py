import os
import json
import cv2
import numpy as np
import torch
import argparse
from superpoint import SuperPoint
import struct
import sys
import time

def log_message(message, flush=True):
    print(f"[DEBUG][{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stdout)
    if flush:
        sys.stdout.flush()

def parse_sfm_file(sfm_path):
    log_message(f"Parsing SFM file: {sfm_path}")
    with open(sfm_path, 'r') as f:
        data = json.load(f)
    views = [(str(v["viewId"]), v["path"]) for v in data["views"]]
    log_message(f"Found {len(views)} views in SFM file")
    return views

def save_features_dpsift(feat_path, desc_path, keypoints, descriptors):
    """Save features in DspSift binary format"""
    num_kps = keypoints.shape[0]
    log_message(f"Saving features to {feat_path} and {desc_path} with {num_kps} keypoints")
    
    with open(feat_path, 'wb') as f:
        f.write(struct.pack('<I', num_kps))
        for kp in keypoints:
            f.write(struct.pack('<4f', kp[0], kp[1], 1.0, 0.0))
    
    with open(desc_path, 'wb') as f:
        f.write(struct.pack('<I', num_kps))
        f.write(struct.pack('<I', descriptors.shape[1]))
        descriptors.tofile(f)
    log_message(f"Features saved successfully")

def extract_and_save(model, image_id, image_path, output_dir, device, describer_type):
    log_message(f"Processing image {image_id} from {image_path}")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log_message(f"ERROR: Failed to read image {image_path}")
        return 0

    log_message(f"Image loaded with shape {img.shape}")
    
    img_tensor = torch.from_numpy(img / 255.).float().unsqueeze(0).unsqueeze(0).to(device)
    log_message(f"Image tensor shape: {img_tensor.shape}")
    
    with torch.no_grad():
        output = model({'image': img_tensor})
        keypoints = torch.flip(output['keypoints'][0], [1]).cpu().numpy()
        descriptors = output['descriptors'][0].t().cpu().numpy()
        descriptors = (descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8) * 512).astype(np.uint8)
    
    log_message(f"Extracted {keypoints.shape[0]} keypoints and {descriptors.shape} descriptors")

    feat_file = os.path.join(output_dir, f"{image_id}.feat")
    desc_file = os.path.join(output_dir, f"{image_id}.desc")
    
    if describer_type == "dspsift":
        save_features_dpsift(feat_file, desc_file, keypoints, descriptors)
    else:
        log_message(f"ERROR: Unsupported describer type {describer_type}")
    
    return keypoints.shape[0]

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--maxKeypoints', type=int, default=-1)
    parser.add_argument('--describerType', choices=['dspsift'], default='dspsift')
    args = parser.parse_args()

    log_message(f"Starting SuperPoint feature extraction with args: {vars(args)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_message(f"Using device: {device}")
    
    config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': args.maxKeypoints,
        'remove_borders': 4,
    }
    log_message(f"Model config: {config}")

    model = SuperPoint(config, args.weights).to(device)
    model.eval()
    log_message("Model initialized and set to eval mode")

    image_list = parse_sfm_file(args.input)
    os.makedirs(args.output, exist_ok=True)
    log_message(f"Created output directory: {args.output}")
    
    total_keypoints = 0
    for image_id, image_path in image_list:
        try:
            num_kps = extract_and_save(model, image_id, image_path, args.output, device, args.describerType)
            total_keypoints += num_kps
        except Exception as e:
            log_message(f"ERROR processing image {image_id}: {str(e)}")
    
    log_message(f"Feature extraction complete. Total keypoints extracted: {total_keypoints}")

if __name__ == "__main__":
    main()