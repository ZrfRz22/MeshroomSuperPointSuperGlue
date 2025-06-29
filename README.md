# SuperPoint + SuperGlue Integration into Meshroom Pipeline

## Introduction

This project integrates the **SuperPoint** model for feature extraction and the **SuperGlue** model for feature matching into the **Meshroom** photogrammetry pipeline. 

The goal is to create a hybrid feature pipeline that combines the strengths of both deep learning-based methods (SuperPoint and SuperGlue) and Meshroom’s traditional feature extraction and matching algorithms.By fusing these techniques, the project aims to enhance the robustness and accuracy of 3D reconstruction, especially in challenging environments where traditional methods may struggle, such as low-texture regions, repetitive patterns, inconsistent lighting conditions or significant viewpoint changes. 

This integration enables a more flexible and resilient pipeline that leverages both classical and learning-based approaches to improve feature matching quality across a wider range of real-world scenarios.

---

## Features

This plugin adds a machine learning-based feature matching pipeline to Meshroom with 3 modular nodes:


### 1. SuperPoint

- Loads the pretrained **SuperPoint model**.
- Loads input images.
- Extracts keypoint features from the images.
- Saves extracted features into the output folder.


### 2. SuperGlue

- Loads the pretrained **SuperGlue model**.
- Loads image pairs from Image Matching
- Loads features from the SuperPoint node.
- Matches features across image pairs.
- Saves resulting matches to the output folder.


### 3. Hybrid Combiner

- Loads original features.
- Loads SuperPoint features.
- Loads original matches.
- Loads SuperGlue matches.
- Combines SuperPoint and original features and
- Combines SuperGlue and original matches:
- Resolves duplicates and ensures consistent formatting.
- Saves combined features to output folder.
- Saves combined matches to output folder.

---

## Video Demo and Tutorial

[![Watch the video](https://img.youtube.com/vi/3mHiARhuAfI/hqdefault.jpg)](https://www.youtube.com/watch?v=3mHiARhuAfI)

---

## Installation

* **Download Meshroom 2023.3.0 (Windows Prebuilt Binary)**

  Get the latest release from the official Meshroom GitHub repository:
  [https://github.com/alicevision/meshroom/releases](https://github.com/alicevision/meshroom/releases)

* **Download and Install Anaconda**

  From the official Anaconda website:
  [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

* **Access Command Prompt**

  After installation, open the Anaconda Command Prompt.

* **Clone the GitHub Repository**

```bash
git clone https://github.com/ZrfRz22/MeshroomSuperPointSuperGlue.git
```
  
* **Create and Activate Conda Environment with Python 3.7**

```bash
conda create -n meshroom_env python=3.7
conda activate meshroom_env
```

* **Install Python Dependencies**

```bash
pip install numpy==1.21.6
pip install opencv-python==4.11.0.86
pip install torch==1.13.1
pip install Pillow
pip install pyinstaller
```

* **Navigate to Project Directory**

```bash
cd path/to/your/MeshroomSuperPointSuperGlue
```

* **Copy `MLPlugin` Folder into Meshroom Nodes Folder**

> Windows CMD / Anaconda Prompt:

```cmd
xcopy /E /I MLPlugin "path\to\Meshroom-2023.3.0\lib\meshroom\nodes\MLPlugin"
```

* **Copy the pre-saved pipeline into Meshroom Pipelines Folder**

```bash
copy hybridPhotogrammetry.mg "path\to\Meshroom-2023.3.0\lib\meshroom\pipelines"
```

* **Compile the Custom Executables Using PyInstaller**

```bash
pyinstaller superPoint_featureExtraction.spec
pyinstaller superGlue_featureMatching.spec
pyinstaller hybridFeatureCombiner.spec
pyinstaller featureVisualizer.spec
```

* **Copy the Executables to Meshroom's AliceVision `bin` Folder**

> Windows CMD / Anaconda Prompt:

```cmd
copy dist\superPoint_featureExtraction.exe "path\to\Meshroom-2023.3.0\aliceVision\bin"
copy dist\superGlue_featureMatching.exe "path\to\Meshroom-2023.3.0\aliceVision\bin"
copy dist\hybridFeatureCombiner.exe "path\to\Meshroom-2023.3.0\aliceVision\bin"
copy dist\featureVisualizer.exe "path\to\Meshroom-2023.3.0\aliceVision\bin"
```

* **Launch Meshroom**

```bash
cd path/to/Meshroom
.\Meshroom.exe
```

---

## Usage

### Pipeline Setup
Once Meshroom is launched, you will be greeted with the main user interface.
![image](https://github.com/user-attachments/assets/a3831c87-d4bd-4f8b-8922-24abb9671533)

Navigate to the workspace panel at the bottom of the screen.
![image](https://github.com/user-attachments/assets/819f7408-476d-4756-a683-9a3d81f96229)

Right-click anywhere in the workspace and hover over the `MLPlugin` category

You will find the three main custom nodes:
   * `SuperPointFeatureExtraction`
   * `SuperGlueFeatureMatching`
   * `HybridFeatureCombiner`

![image](https://github.com/user-attachments/assets/8a451984-dc06-4f17-bdb5-44e74af30773)
   
Add all three nodes to the workspace:
![image](https://github.com/user-attachments/assets/47474b9e-a34b-4c1b-8b55-ebd6f5200856)

Duplicate the default `ImageMatching` node to create a second one:

![image](https://github.com/user-attachments/assets/57f36ca1-a8f3-43bd-b966-7d66070f048f)
 
 ---
 
### Node Arrangement
Once all the nodes have been added, arrange the nodes like so:
![image](https://github.com/user-attachments/assets/72286f9a-1931-4709-8ef2-94ee52865929)

Remove all the connections between the Feature Matching node and the Structure from Motion node by right clicking on a connection selecting `Remove`
![image](https://github.com/user-attachments/assets/ca16b5ed-73a8-4323-b81d-5129a0b3b7d6)

Then, connect the nodes in the following layout (The Describer Types can be found by clicking the dropdown button at the bottom of a node to reveal more parameters). Make sure all node connections flow left to right as Meshroom determines the pipeline execution order based on this direction:
![image](https://github.com/user-attachments/assets/9f7fabb0-a5b9-4674-b816-7aa4583e5d90)

Clicking the dropdown button again after making the connections is recommended as to not make the workspace too cluttered, as seen here:
![image](https://github.com/user-attachments/assets/e884a848-67e3-4d61-b319-8c6c6225128c)

Click on the `File` menu at the top → `Save as` to Save the pipeline

![image](https://github.com/user-attachments/assets/844db156-bfdc-48b9-86f5-f074ce2ba24d)

---

###  Execution
Click on the `File` menu at the top → `Import Images` → Select your image folder to import your images.

![image](https://github.com/user-attachments/assets/179d4e90-7fe3-4e53-bc74-3a18bc17d361)

Your imported images will be displayed on your dashboard.
![image](https://github.com/user-attachments/assets/cd8e8f13-677f-4024-8568-f18a05f1957f)

Click the `Start` button on the top toolbar.

![image](https://github.com/user-attachments/assets/0e36dd60-914c-4693-8273-c358e3f251bc)

The pipeline will begin execution automatically.
![image](https://github.com/user-attachments/assets/535c5d49-aa49-4256-8a05-2ce0e650e464)

---

## Configuration

### SuperPoint Configuration
- **max_num_keypoints** controls how many keypoints are extracted from each image. A value of 1000 limits the extraction to 1000 keypoints, which is faster but may miss finer details. Setting it to -1 means no limit, allowing all detected keypoints to be used, which improves accuracy but slows down processing. In a nutshell, the more keypoints you extract, the better your matching can be, but it comes at the cost of speed.

- **nms_radius (Non-Maximum Suppression radius)** determines how close keypoints can be to each other. A smaller radius keeps more keypoints, including ones that are close together, but it may include redundant or noisy points. A higher value removes more closely packed keypoints, helping reduce noise but possibly discarding useful features.

### SuperGlue Configuration
- **match_threshold** sets the minimum confidence for accepting a match. Lower values make the algorithm stricter, resulting in fewer but more reliable matches. A common default is 0.3, which is kept low so that RANSAC has enough matches to filter through.

- **sinkhorn_iterations** defines how many times the Sinkhorn algorithm refines the match scores. More iterations can improve precision but increase processing time. The default is 20, which works well in most cases.

- **ransac_threshold** determines the maximum distance (in pixels) allowed between matched keypoints for them to be considered inliers (valid matches). A smaller value is stricter, accepting only very close matches. This improves robustness but may discard valid matches if images are noisy or distorted.

- **ransac_trials** sets how many times RANSAC will run to find the best transformation model. Higher values like 1000 make the results more reliable but also increase processing time. For noisy datasets, a higher number is recommended to ensure better model fitting.

---

## Recommended Settings

| Use Case         | max\_num\_keypoints | match\_threshold | sinkhorn\_iterations | ransac\_threshold | ransac\_trials |
| ---------------- | ------------------- | ---------------- | -------------------- | ------------------ | --------------- |
| Fast Processing  | 1000                | 0.3              | 20                   | 1.5                | 1000            |
| High Accuracy    | -1                  | 0.3              | 30                   | 1.5                | 3000            |

---

## Additional Tools

### Feature Visualizer Node

The Feature Visualizer is an optional node in the ML Plugin category that allows users to visually and manually inspect each keypoint and each matche from any of the major matching porcesses. It helps debug and better understand the performance of the feature extraction and feature matching nodes in the pipeline.

**How to Use**

1.Navigate to the*workspace section at the bottom, and Right-click in the workspace, go to ML Plugin and select Feature Visualizer

![image](https://github.com/user-attachments/assets/74a47950-b060-4956-af74-34f27efde436)

2. Add it to the workspace, and connect it to any of the macther nodes:
 
![image](https://github.com/user-attachments/assets/57cc8530-adae-49e2-a12e-a5411188e313)

3. Press the "Start" button at the top, and 2 windows should pop up, one for the first image and one for the other
   - Use the "Up" and "Down" arrow keys to cycle through each keypoint pair (Highlighted in red)
   - Use the "Left" and "Right" arrow keys to cycle between image pairs
   - All keypoints of an image are also displayed (coloured in green)
   - Press "r" to toggle between counter-clockwise and clockwise image rotation (In case of key point and image misalignment)
   - Press "f" to flip the image vertcially (In case of key point and image misalignment)

![image](https://github.com/user-attachments/assets/21c3c9c9-0d66-4fb9-ac18-17e3f95ba706)

---

## Disclaimers and Recommendations

### Disclaimer

This plugin was developed without altering the internal components of the official Meshroom application. As such, any issues that originated in Meshroom’s internal system remain unresolved during implementation.

Known issues include, when modifying a pipeline after nodes have already been executed, the following unpredictable behaviors may occur:
- **Node settings may not update properly.** Even after changing a node’s parameters and resetting it, the node may still execute using the previous settings.
- **Pipeline may retain outdated process paths.** If you rearrange, reconnect, or modify nodes, the execution may still follow the old version of the pipeline, ignoring the new changes.

These behaviors are inconsistent and unpredictable. Sometimes the changes apply correctly, and other times, the pipeline behaves as if no modification was made.
It should be made clear that these limitations stem from Meshroom itself and are not directly caused by this plugin.

### Recommendations

To ensure consistent results and avoid unpredictable behavior:

1. Set up the ML hybrid pipeline with all necessary nodes and connections.
2. Select all the nodes in the pipeline and copy them.
3. Then, click `File > New > Photogrammetry Pipeline` to create a new project.
4. You will be given the generic photogrammetry pipeline.
5. Paste the copied ML pipeline into the new workspace.
6. Delete the provided generic pipeline.
7. Recheck the connections of the pasted ML hybrid pipeline.
8. Conduct reconstruction.

This help provide a fresh and reliable environment for each new project session.

---

## Example Screenshots

Execution in Porgress
![image](https://github.com/user-attachments/assets/b8272557-edff-446c-a63e-b4fa17bef728)

Failed or Halted Execution
![image](https://github.com/user-attachments/assets/d906695a-5fb5-424a-8070-dce40ddb9cd3)

Successful Execution Completion
![image](https://github.com/user-attachments/assets/b5d0c844-c088-4095-b646-10c59f18125e)

---

## Lisence
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. 
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

---

## Acknowledgements
This project builds upon the  work of several open-source projects. Credits and ownership go to:

- [**SuperPoint**](https://github.com/magicleap/SuperPointPretrainedNetwork): A real-time interest point detector and descriptor by Magic Leap, originally described in *SuperPoint: Self-Supervised Interest Point Detection and Description*.
- [**SuperGlue**](https://github.com/magicleap/SuperGluePretrainedNetwork): A graph neural network for feature matching by Magic Leap, described in *SuperGlue: Learning Feature Matching with Graph Neural Networks*.
- [**Meshroom**](https://github.com/alicevision/meshroom): A free, open-source 3D reconstruction software based on photogrammetry.
- [**AliceVision**](https://github.com/alicevision/AliceVision): The photogrammetric computer vision framework that powers Meshroom.
