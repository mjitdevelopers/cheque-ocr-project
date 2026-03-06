@echo off
echo Building ChequeOCR.exe...
pyinstaller --onefile --windowed --name="ChequeOCR" ^
--hidden-import=numpy ^
--hidden-import=paddleocr ^
--hidden-import=paddle ^
--hidden-import=dbf ^
--hidden-import=tkinter ^
--hidden-import=PIL ^
--hidden-import=cv2 ^
--hidden-import=shapely ^
--hidden-import=pyyaml ^
--hidden-import=scipy ^
--hidden-import=skimage ^
--hidden-import=imgaug ^
--hidden-import=lmdb ^
--hidden-import=tqdm ^
--hidden-import=visualdl ^
--hidden-import=rapidfuzz ^
--hidden-import=PyMuPDF ^
--collect-all=paddleocr ^
--collect-all=paddle ^
--collect-all=numpy ^
--collect-all=cv2 ^
--collect-all=PIL ^
app.py

echo.
echo Build complete! Check the 'dist' folder for ChequeOCR.exe
pause   