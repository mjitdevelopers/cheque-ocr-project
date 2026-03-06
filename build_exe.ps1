# build_exe.ps1
Write-Host "Building ChequeOCR Executable..." -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green

# Clean old builds
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }

# Build EXE
pyinstaller --onefile --windowed `
    --name "ChequeOCR" `
    --hidden-import paddleocr `
    --hidden-import cv2 `
    --hidden-import PIL `
    --hidden-import numpy `
    --hidden-import dbf `
    --hidden-import skimage `
    --hidden-import pyclipper `
    --collect-all paddleocr `
    --collect-all cv2 `
    --add-data ".;." `
    app.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ SUCCESS! EXE created at: dist\ChequeOCR.exe" -ForegroundColor Green
    Write-Host "File size: $((Get-Item dist\ChequeOCR.exe).Length / 1MB) MB" -ForegroundColor Yellow
} else {
    Write-Host "`n❌ Build failed!" -ForegroundColor Red
}

Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")