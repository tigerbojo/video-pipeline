@echo off
chcp 65001 >nul 2>&1
title AI 影片自動後製工作流

echo.
echo  ========================================
echo   AI 影片自動後製工作流
echo   啟動中...
echo  ========================================
echo.

cd /d "H:\dev\drnogreasy\video-pipeline"

:: 用完整路徑啟動 Python
start "" /b C:\Python313\python.exe app.py

echo  等待伺服器啟動...
timeout /t 10 /nobreak >nul

:: 開啟瀏覽器
start http://localhost:7860

echo.
echo  ========================================
echo   瀏覽器已開啟 http://localhost:7860
echo   關閉此視窗 = 停止伺服器
echo  ========================================
echo.
pause
