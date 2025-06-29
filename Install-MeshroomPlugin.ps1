# MeshroomSuperPointSuperGlue Installer with Miniconda
param (
    [string]$MeshroomPath = "$env:USERPROFILE\Downloads\Meshroom-2023.3.0",
    [string]$InstallPath = "$env:USERPROFILE\MeshroomSuperPointSuperGlue",
    [string]$MinicondaPath = "$env:USERPROFILE\Miniconda3"
)

# 1. Download Meshroom
Write-Host "Downloading Meshroom..." -ForegroundColor Cyan
$meshroomUrl = "https://github.com/alicevision/meshroom/releases/download/v2023.3.0/Meshroom-2023.3.0-win64.zip"
$zipPath = "$env:TEMP\Meshroom-2023.3.0-win64.zip"
Invoke-WebRequest -Uri $meshroomUrl -OutFile $zipPath

# 2. Extract Meshroom
Write-Host "Extracting Meshroom..." -ForegroundColor Cyan
Expand-Archive -Path $zipPath -DestinationPath (Split-Path $MeshroomPath) -Force
Remove-Item $zipPath

# 3. Install Miniconda
Write-Host "Installing Miniconda..." -ForegroundColor Cyan
$minicondaUrl = "https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Windows-x86_64.exe"
$minicondaInstaller = "$env:TEMP\MinicondaInstaller.exe"
Invoke-WebRequest -Uri $minicondaUrl -OutFile $minicondaInstaller
Start-Process -Wait -FilePath $minicondaInstaller -ArgumentList "/InstallationType=JustMe /AddToPath=1 /RegisterPython=1 /S /D=$MinicondaPath"
Remove-Item $minicondaInstaller

# 4. Clone repository
Write-Host "Cloning SuperPointSuperGlue repository..." -ForegroundColor Cyan
git clone https://github.com/ZrfRz22/MeshroomSuperPointSuperGlue.git $InstallPath

# 5. Set up Conda environment
Write-Host "Creating Conda environment..." -ForegroundColor Cyan
$env:Path += ";$MinicondaPath\Scripts;$MinicondaPath\condabin"
conda create -n meshroom_env python=3.7 -y
conda activate meshroom_env

# 6. Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
pip install numpy==1.21.6 opencv-python==4.11.0.86 torch==1.13.1 Pillow pyinstaller

# 7. Copy plugin files
Write-Host "Installing Meshroom plugin..." -ForegroundColor Cyan
$pluginDest = "$MeshroomPath\lib\meshroom\nodes\MLPlugin"
New-Item -ItemType Directory -Path $pluginDest -Force | Out-Null
Copy-Item -Path "$InstallPath\MLPlugin\*" -Destination $pluginDest -Recurse -Force

# 8. Copy pipeline
Write-Host "Copying pipeline file..." -ForegroundColor Cyan
$pipelineDest = "$MeshroomPath\lib\meshroom\pipelines"
Copy-Item -Path "$InstallPath\hybridPhotogrammetry.mg" -Destination $pipelineDest -Force

# 9. Compile executables
Write-Host "Compiling executables..." -ForegroundColor Cyan
Set-Location $InstallPath
pyinstaller superPoint_featureExtraction.spec
pyinstaller superGlue_featureMatching.spec
pyinstaller hybridFeatureCombiner.spec
pyinstaller featureVisualizer.spec

# 10. Copy executables
$binDest = "$MeshroomPath\aliceVision\bin"
Copy-Item -Path "$InstallPath\dist\*.exe" -Destination $binDest -Force

# 11. Create desktop shortcut
Write-Host "Creating desktop shortcut..." -ForegroundColor Cyan
$shortcutPath = "$env:USERPROFILE\Desktop\Meshroom.lnk"
$WScriptShell = New-Object -ComObject WScript.Shell
$shortcut = $WScriptShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "$MeshroomPath\Meshroom.exe"
$shortcut.WorkingDirectory = $MeshroomPath
$shortcut.Save()

Write-Host "Installation completed successfully!" -ForegroundColor Green
Write-Host "Meshroom shortcut created on your desktop." -ForegroundColor Green