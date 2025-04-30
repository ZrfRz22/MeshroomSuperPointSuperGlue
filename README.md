## Installation Steps

### 1. Download Prebuilt Meshroom
- Go to the [Meshroom GitHub releases](https://github.com/alicevision/meshroom/releases).
- Download the latest **prebuilt binary** (e.g., `Meshroom-2023.3.0-win64.zip`).
- Extract the contents to a folder, e.g., `C:\Meshroom`.

### 2. Clone This Project
```bash
git clone https://github.com/ZrfRz22/MeshroomSuperPointSuperGlue.git
cd MeshroomSuperPointSuperGlue
```

### 3. Install Anaconda (if not already installed)
- Download and install from: https://www.anaconda.com/products/distribution

### 4. Install Python 3.7 (64-bit) (recommended for compatibility)

Download from:
https://www.python.org/downloads/release/python-370/

---

## Set Up Python Environment

### 5. Open Anaconda Command Prompt

### 6. Create a Conda Environment with Python 3.7
```bash
conda create -n myenv python=3.7
```

### 7. Activate the Environment
```bash
conda activate myenv
```

### 8. Install Required Python Packages
```bash
pip install torch opencv-python numpy matplotlib pyinstaller
```

---

## Build Executables

### 9. Navigate to Project Directory
```bash
cd path\to\MeshroomSuperPointSuperGlue
```

### 10. Build SuperPoint Executable
```bash
pyinstaller superPoint_featureExtraction.spec
```

### 11. Build SuperGlue Executable
```bash
pyinstaller superGlue_featureMatching.spec
```

---

## Integrate with Meshroom

### 12. Go to AliceVision Binary Folder
```
path\to\Meshroom\aliceVision\bin
```

### 13. Copy Executables into AliceVision
- Copy:
  - `dist\superPoint_featureExtraction\superPoint_featureExtraction.exe`
  - `dist\superGlue_featureMatching\superGlue_featureMatching.exe`
- Paste them into:
  ```
  path\to\Meshroom\aliceVision\bin
  ```

### 14. Go to Meshroom `nodes` Folder
```
path\to\Meshroom\meshroom\nodes
```

### 15. Add the Plugin Node
- Copy the `MLPlugin` folder from this repo.
- Paste it into the `nodes` folder:
  ```
  path\to\Meshroom\meshroom\nodes\MLPlugin
  ```

---

## Run Meshroom

### 16. Launch Meshroom with Plugin Support
- Open a new Command Prompt (not Anaconda Prompt, No environment activated).
```bash
cd path\to\Meshroom
.\Meshroom.exe
```

You should now see the `MLPlugin` nodes: `SuperPointFeatureExtraction` and `SuperGlueFeatureMatching` in the Meshroom node graph!

---
