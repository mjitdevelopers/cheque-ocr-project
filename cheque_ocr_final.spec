# cheque_ocr_final.spec
# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# Forcefully collect all PaddleOCR data
datas = []
binaries = []
hiddenimports = []

# PaddleOCR and its dependencies
for pkg in ['paddleocr', 'paddle', 'cv2', 'PIL', 'numpy', 'skimage', 'pyclipper', 'imgaug']:
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas.extend(pkg_datas)
    binaries.extend(pkg_binaries)
    hiddenimports.extend(pkg_hidden)

# Add specific hidden imports
hiddenimports.extend([
    'paddleocr',
    'paddle',
    'cv2',
    'PIL',
    'PIL._tkinter_finder',
    'numpy',
    'numpy.core',
    'numpy.lib',
    'numpy.fft',
    'numpy.linalg',
    'numpy.random',
    'numpy.ctypeslib',
    'numpy.ma',
    'numpy.matrixlib',
    'numpy.polynomial',
    'numpy.testing',
    'skimage',
    'skimage.feature',
    'skimage.filters',
    'skimage.measure',
    'skimage.morphology',
    'skimage.transform',
    'skimage.segmentation',
    'skimage._shared',
    'pyclipper',
    'imgaug',
    'dbf',
    'dbf.ver_2',
    'dbf.ver_3',
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
])

# Add data files
datas.extend(collect_data_files('paddleocr'))
datas.extend(collect_data_files('paddle'))
datas.extend(collect_data_files('cv2'))
datas.extend(collect_data_files('PIL'))

# Add current directory
datas.append(('.', '.'))

a = Analysis(
    ['app.py'],
    pathex=['D:\\cheque-ocr'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    name='ChequeOCR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
)