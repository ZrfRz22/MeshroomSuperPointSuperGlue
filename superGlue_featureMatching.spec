block_cipher = None

a = Analysis(
    ['superGlue_featureMatching.py'],  # Path to your Python script
    pathex=[],
    binaries=[],
    datas=[],  # Leave this empty since we're not bundling the weights
    hiddenimports=[],  # Add any hidden imports if needed
    hookspath=[],
    runtime_hooks=[],
    excludes=[]  # Any excluded modules go here
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='superGlue_featureMatching',  # Name of the executable
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)