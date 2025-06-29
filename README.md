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

## Installation Script

Ensure Git is installed, first.

[➡️ Click here to download the installation script](Install-MeshroomPlugin.ps1)

After downloading, right-click the file and select “Run with PowerShell” (make sure to run it as Administrator).
Alternatively, open PowerShell as Administrator and run:

```powershell
.\Install-MeshroomPlugin.ps1
```

---

## Manual Installation
If the previously mentioned installation script successfuly installed Meshroom and the plugin, please skip to the "Usage" section, but in the case, that the installation file fails to build the plugin or the installation takes too long, you can follow this manual installation guide as an alternative. 

A YouTube video tutorial for manual installation is also provided below for additional clarity.

[![Watch the video](https://img.youtube.com/vi/r_90XGJAx2E/0.jpg)](https://www.youtube.com/watch?v=r_90XGJAx2E)

**Download Meshroom 2023.3.0 (Windows Prebuilt Binary)**

  Get the latest release from the official Meshroom GitHub repository:
  [https://github.com/alicevision/meshroom/releases](https://github.com/alicevision/meshroom/releases)

**Download and Install Anaconda**

  From the official Anaconda website:
  [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

**Access Command Prompt**

  After installation, open the Anaconda Command Prompt.

**Clone the GitHub Repository**

```bash
git clone https://github.com/ZrfRz22/MeshroomSuperPointSuperGlue.git
```
  
**Create and Activate Conda Environment with Python 3.7**

```bash
conda create -n meshroom_env python=3.7
conda activate meshroom_env
```

**Install Python Dependencies**

```bash
pip install numpy==1.21.6
pip install opencv-python==4.11.0.86
pip install torch==1.13.1
pip install Pillow
pip install pyinstaller
```

**Navigate to Project Directory**

```bash
cd path/to/your/MeshroomSuperPointSuperGlue
```

**Copy `MLPlugin` Folder into Meshroom Nodes Folder**

> Windows CMD / Anaconda Prompt:

```cmd
xcopy /E /I MLPlugin "path\to\Meshroom-2023.3.0\lib\meshroom\nodes\MLPlugin"
```

**Copy the pre-saved pipeline into Meshroom Pipelines Folder**

```bash
copy hybridPhotogrammetry.mg "path\to\Meshroom-2023.3.0\lib\meshroom\pipelines"
```

**Compile the Custom Executables Using PyInstaller**

```bash
pyinstaller superPoint_featureExtraction.spec
pyinstaller superGlue_featureMatching.spec
pyinstaller hybridFeatureCombiner.spec
pyinstaller featureVisualizer.spec
```

**Copy the Executables to Meshroom's AliceVision `bin` Folder**

> Windows CMD / Anaconda Prompt:

```cmd
copy dist\superPoint_featureExtraction.exe "path\to\Meshroom-2023.3.0\aliceVision\bin"
copy dist\superGlue_featureMatching.exe "path\to\Meshroom-2023.3.0\aliceVision\bin"
copy dist\hybridFeatureCombiner.exe "path\to\Meshroom-2023.3.0\aliceVision\bin"
copy dist\featureVisualizer.exe "path\to\Meshroom-2023.3.0\aliceVision\bin"
```

**Launch Meshroom**

```bash
cd path/to/Meshroom
.\Meshroom.exe
```

---

## Usage

### Pipeline Setup
Once Meshroom is launched, you will be greeted with the main user interface.
![image](https://github.com/user-attachments/assets/a3831c87-d4bd-4f8b-8922-24abb9671533)

To set up the pipeline, click on the `File` menu at the top → `New Pipeline` → select `Hybrid Photogrammetry`
![image](https://github.com/user-attachments/assets/08b144ec-accd-4a5e-b713-c0c8b939f81c)

Once clicked, the hybrid pipeline will appear in the workspace
![image](https://github.com/user-attachments/assets/6683fff7-7125-41f8-a919-780d965c71c9)

---

###  Execution
Click on the `File` menu at the top → `Import Images` → Select your image folder to import your images.
![image](https://github.com/user-attachments/assets/0bd5c0f6-b1b6-43f3-8ba9-dfe8de54101d)

Your imported images will be displayed on your dashboard.
![image](https://github.com/user-attachments/assets/2720f69a-976c-4f24-b4c8-bc9082625c43)

Click the `Start` button on the top toolbar.
![image](https://github.com/user-attachments/assets/7e3d4bb1-2a7b-4175-bcb1-150aa2a83e15)

The pipeline will begin execution automatically.
![image](https://github.com/user-attachments/assets/66bf062a-80cf-4eb5-8eb0-35b3695b089f)

When execution successfuly completes, the dashboard will look like this
![image](https://github.com/user-attachments/assets/3a72979a-b86f-46bb-8779-e9a737a04fc7)

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

1. Navigate to the workspace section at the bottom, and Right-click in the workspace, go to ML Plugin and select Feature Visualizer
![image](https://github.com/user-attachments/assets/9f8b9934-d56b-427b-8a95-a70678f4d944)

2. Add it to the workspace, and connect it to any of the matcher nodes:
![image](https://github.com/user-attachments/assets/57cc8530-adae-49e2-a12e-a5411188e313)

3. Press the "Start" button at the top, and 2 windows should pop up, one for the first image and one for the other
   - Use the "Up" and "Down" arrow keys to cycle through each keypoint pair (Highlighted in red)
   - Use the "Left" and "Right" arrow keys to cycle between image pairs
   - All keypoints of an image are also displayed (coloured in green)
   - The matched keypoints of an image pair are also displayed (highlighted in red)
   - Press "a" to toggle between showing all keypoints and showing only matched keypoints

![image](https://github.com/user-attachments/assets/7fad659c-0961-4c58-9f45-cef590dbd94b)

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

To ensure consistent results and avoid unpredictable behavior, ensure to start a new Hybrid Photogrammetry before executing every reconstruction to ensure inconsistent image file paths and configuration parameters between nodes, providing a fresh and reliable environment for each new project session.
![image](https://github.com/user-attachments/assets/08b144ec-accd-4a5e-b713-c0c8b939f81c)

---

## Example Screenshots

Execution in Progress
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
