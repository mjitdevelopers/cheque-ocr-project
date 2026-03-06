# ocr_app.spec
# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# Collect all PaddleOCR data files
paddleocr_datas = collect_data_files('paddleocr')
paddle_datas = collect_data_files('paddle')
ppocr_datas = collect_data_files('ppocr')

# Collect dynamic libraries
paddleocr_binaries = collect_dynamic_libs('paddleocr')
paddle_binaries = collect_dynamic_libs('paddle')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=paddleocr_datas + paddle_datas + ppocr_datas,
    hiddenimports=[
        'paddle',
        'paddleocr',
        'paddleocr.ppocr',
        'paddleocr.ppocr.modeling',
        'paddleocr.ppocr.modeling.architectures',
        'paddleocr.ppocr.modeling.backbones',
        'paddleocr.ppocr.modeling.necks',
        'paddleocr.ppocr.modeling.heads',
        'paddleocr.ppocr.modeling.transforms',
        'paddleocr.ppocr.losses',
        'paddleocr.ppocr.optimizer',
        'paddleocr.ppocr.postprocess',
        'paddleocr.ppocr.metrics',
        'paddleocr.ppocr.data',
        'paddleocr.ppocr.utils',
        'paddleocr.tools',
        'paddleocr.tools.infer',
        'paddleocr.tools.infer.predict_det',
        'paddleocr.tools.infer.predict_rec',
        'paddleocr.tools.infer.predict_cls',
        'paddleocr.tools.infer.utility',
        'paddleocr.tools.infer.db_postprocess',
        'paddle',
        'paddle.fluid',
        'paddle.fluid.core',
        'paddle.fluid.contrib',
        'paddle.fluid.contrib.slim',
        'paddle.nn',
        'paddle.tensor',
        'paddle.framework',
        'paddle.optimizer',
        'paddle.fluid.dygraph',
        'paddle.fluid.layers',
        'cv2',
        'skimage',
        'imghdr',
        'lmdb',
        'premailer',
        'cssutils',
        'cssselect',
        'lxml',
        'html5lib',
        'bs4',
        'pyclipper',
        'shapely',
        'shapely.geometry',
        'shapely.ops',
    ],
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
    name='Cheque_OCR_Updater',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging, False for production
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None  # Add your icon here if you have one
)