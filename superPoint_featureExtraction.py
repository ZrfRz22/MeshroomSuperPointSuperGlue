import os
import json
import cv2
import numpy as np
import torch
import argparse


### === SUPERPOINT ARCHITECTURE === ###
class SuperPointNet(torch.nn.Module):
    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))
        return semi, desc


### === UTILITY FUNCTIONS === ###
def parse_sfm_file(sfm_path):
    with open(sfm_path, 'r') as f:
        data = json.load(f)
    return [(str(v["viewId"]), v["path"]) for v in data["views"]]


def extract_keypoints_from_heatmap(heatmap, confidence_thresh=0.015):
    heatmap = heatmap.squeeze().cpu().numpy()
    heatmap = np.exp(heatmap) / np.sum(np.exp(heatmap), axis=0)
    heatmap = heatmap[:-1, :, :]  # remove dustbin
    prob_map = np.max(heatmap, axis=0)

    keypoints = np.argwhere(prob_map > confidence_thresh)
    keypoints = np.flip(keypoints, axis=1)  # [y, x] -> [x, y]
    scores = prob_map[keypoints[:, 1], keypoints[:, 0]]
    return keypoints, scores


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
        semi, desc = model(img_tensor)
        keypoints, _ = extract_keypoints_from_heatmap(semi)

        # Re-map descriptors to keypoint coordinates
        H, W = img.shape
        desc = torch.nn.functional.interpolate(desc, size=(H, W), mode='bilinear', align_corners=False)
        desc = desc.squeeze().cpu().numpy()
        descriptors = desc[:, keypoints[:, 1], keypoints[:, 0]].T

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

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SuperPointNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    image_list = parse_sfm_file(args.input)
    for image_id, image_path in image_list:
        print(f"[INFO] Processing {image_path}")
        extract_and_save(model, image_id, image_path, args.output, device)

    print("Feature extraction complete!")


if __name__ == "__main__":
    superPoint_featureExtraction()
