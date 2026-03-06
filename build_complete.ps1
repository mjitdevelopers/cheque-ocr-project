@echo off
echo ========================================
echo    Cheque OCR - Final Build
echo ========================================
echo.

REM Activate virtual environment
call D:\cheque-ocr\ocr_env\Scripts\activate

REM Clean
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul
del *.spec 2>nul

REM Set Python path
set PYTHONPATH=D:\cheque-ocr\ocr_env\Lib\site-packages

REM Build with ALL dependencies
pyinstaller --onefile --windowed ^
    --name "ChequeOCR" ^
    --paths "D:\cheque-ocr\ocr_env\Lib\site-packages" ^
    --hidden-import paddleocr ^
    --hidden-import paddle ^
    --hidden-import cv2 ^
    --hidden-import numpy ^
    --hidden-import PIL ^
    --hidden-import skimage ^
    --hidden-import pyclipper ^
    --hidden-import imghdr ^
    --hidden-import imageio ^
    --collect-all paddleocr ^
    --collect-all paddle ^
    --collect-all cv2 ^
    --collect-all numpy ^
    --collect-all PIL ^
    --add-data "D:\cheque-ocr\ocr_env\Lib\site-packages\paddleocr;.\paddleocr" ^
    --add-data "D:\cheque-ocr\ocr_env\Lib\site-packages\paddle;.\paddle" ^
    app.py

echo.
if exist dist\ChequeOCR.exe (
    echo ✅ SUCCESS! EXE created: dist\ChequeOCR.exe
    echo Size: %~z0 bytes
    echo.
    echo Run it: dist\ChequeOCR.exe
) else (
    echo ❌ Build failed. Trying alternative method...
    
    REM Alternative simple build
    pyinstaller --onefile --windowed --name ChequeOCR app.py
)

pause