# -*- mode: python ; coding: utf-8 -*-

# Create debug spec file
@'
# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all PaddleOCR submodules and data
paddleocr_datas = collect_data_files('paddleocr')
paddleocr_hidden = collect_submodules('paddleocr')

# Collect paddle submodules
paddle_hidden = collect_submodules('paddle')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=paddleocr_datas + [
        (os.path.expanduser('~/.paddleocr'), '.paddleocr'),
    ],
    hiddenimports=list(set(
        paddleocr_hidden + 
        paddle_hidden + 
        [
            'pyclipper',
            'shapely',
            'shapely.geometry',
            'skimage',
            'skimage.morphology',
            'lmdb',
            'imghdr',
            'paddleocr.tools',
            'paddleocr.tools.infer',
            'paddleocr.ppocr',
            'paddleocr.ppocr.postprocess',
            'paddleocr.ppocr.utils',
        ]
    )),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ChequeOCR_Debug',
    debug=True,  # Enable debug mode
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Show console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None
)
'@ | Out-File -FilePath "cheque_ocr_debug.spec" -Encoding ASCII
