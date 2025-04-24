import os
import json
import cv2
import numpy as np
import torch
import argparse
from superpoint import SuperPoint

### === UTILITY FUNCTIONS === ###
def parse_sfm_file(sfm_path):
    with open(sfm_path, 'r') as f:
        data = json.load(f)
    return [(str(v["viewId"]), v["path"]) for v in data["views"]]


def save_feature_files(feat_path, desc_path, keypoints, descriptors):
    keypoints = np.array(keypoints, dtype=np.float32)
    descriptors = np.array(descriptors, dtype=np.float32)
    keypoints.tofile(feat_path)
    descriptors.tofile(desc_path)


def extract_and_save(model, image_id, image_path, output_dir, device):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return

    img_tensor = torch.from_numpy(img / 255.).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model({'image': img_tensor})  # model outputs a dictionary

        keypoints = output['keypoints'][0]  # Take the first image batch
        scores = output['scores'][0]  # Similarly for scores
        descriptors = output['descriptors'][0]  # Same for descriptors

        # Convert keypoints from (y, x) to (x, y) for saving
        keypoints = torch.flip(keypoints, [1]).cpu().numpy()

        # Since descriptors are already sampled at the keypoints in SuperPoint,
        # they are ready to be saved without additional interpolation.
        descriptors = descriptors.cpu().numpy()

    feat_file = os.path.join(output_dir, f"{image_id}.feat")
    desc_file = os.path.join(output_dir, f"{image_id}.desc")
    save_feature_files(feat_file, desc_file, keypoints, descriptors)


### === MAIN FUNCTION === ###
def superPoint_featureExtraction():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to .sfm file (from CameraInit)")
    parser.add_argument('--output', type=str, required=True, help="Output folder for .feat and .desc files")
    parser.add_argument('--weights', type=str, required=True, help="Path to SuperPoint weights file")
    args = parser.parse_args()

    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SuperPoint(default_config).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    image_list = parse_sfm_file(args.input)
    for image_id, image_path in image_list:
        print(f"[INFO] Processing {image_path}")
        extract_and_save(model, image_id, image_path, args.output, device)

    print("Feature extraction complete!")


if __name__ == "__main__":
    superPoint_featureExtraction()
