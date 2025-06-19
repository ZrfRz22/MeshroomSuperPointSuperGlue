# SuperPoint + SuperGlue Integration into Meshroom Pipeline

## Introduction

This project integrates the **SuperPoint** model for feature extraction and the **SuperGlue** model for feature matching into the **Meshroom** photogrammetry pipeline. The goal is to create a **hybrid feature pipeline** that combines the strengths of both deep learning-based methods (SuperPoint and SuperGlue) and Meshroom’s traditional feature extraction and matching algorithms.By fusing these techniques, the project aims to enhance the **robustness and accuracy** of 3D reconstruction, especially in **challenging environments** where traditional methods may struggle, such as low-texture regions, repetitive patterns, inconsistent lighting conditions or significant viewpoint changes. This integration enables a more flexible and resilient pipeline that leverages both classical and learning-based approaches to improve feature matching quality across a wider range of real-world scenarios.

Great! Here's a clean and structured **Installation** section for your README based on those steps:

---

##  Installation

### 1. Environment Setup

1. **Download Meshroom (Windows Prebuilt Binary)**
   - Get the latest release from the official [Meshroom GitHub repository](https://github.com/alicevision/meshroom/releases).

2. **Install Python 3.7**
   - Download Python 3.7 from the [official Python website](https://www.python.org/downloads/release).
   - Ensure you download the correct version for your system (64-bit recommended).

3. **Install Anaconda**
   - Download and install Anaconda from [here](https://www.anaconda.com/products/distribution).
   - Open the **Anaconda Prompt** (CMD).

4. **Create and Activate a New Conda Environment**
   ```bash
   conda create -n meshroom_env python=3.7
   conda activate meshroom_env
   ```

5. **Install Required Dependencies**
   *(Dependencies will be listed below once finalized)*

6. **Navigate to Project Root Folder**

   ```bash
   cd path\to\your\github\project
   ```

7. **Integrate MLPlugin Folder**

   * Copy the `MLPlugin` folder from this repository.
   * Paste it into:
     ```
     path\to\Meshroom\meshroom\nodes
     ```

8. **Compile Executables Using PyInstaller**

   ```bash
   pyinstaller superPoint_featureExtraction.spec
   pyinstaller superGlue_featureMatching.spec
   pyinstaller hybridFeatureCombiner.spec
   ```

9. **Copy Compiled Executables**

   * After compilation, copy the generated `.exe` files into:
     ```
     path\to\Meshroom\aliceVision\bin
     ```

10. **Launch Meshroom**

    ```bash
    .\Meshroom.exe
    ```

---

### 17. Add the SuperPoint and SuperGlue nodes
- Right click the workspace at the bottom, and the SuperPoint node and SuperGlue node can be found in the ML Plugin section.
![image](https://github.com/user-attachments/assets/1e0b35d2-61cf-4087-991a-41fa1556aad0)
![image](https://github.com/user-attachments/assets/a78fa5aa-f1a8-4e88-b518-76124adcd9a8)

### 18. Remove Meshroom's original feature extraction and feature matching nodes
![image](https://github.com/user-attachments/assets/29b4e82f-b51b-4e44-81f9-2a781ab7db89)
![image](https://github.com/user-attachments/assets/60473f83-1c52-4d07-b481-b765b20d99e7)

### 19. Connect the inputs and outputs between nodes
- Follow as specified in the image below
![image](https://github.com/user-attachments/assets/d62f47ed-e9e5-413b-96ac-332943ea8735)

### 20. Import images and start reconstruction


