# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['transcriber.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['opencc', 'moviepy', 'whisper', 'torch', 'numpy', 'psutil', 'soundfile'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['Cache'],  # Exclude the Cache folder
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Echoes',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Add UPX compression settings
    upx_compress=True,
    upx_compress_level=9,
    upx_compress_filter='*'
)