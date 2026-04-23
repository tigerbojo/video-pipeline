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

:: 啟動 Gradio
start /b python app.py

:: 等待伺服器就緒
echo  等待伺服器啟動...
:wait
timeout /t 2 /nobreak >nul
curl -s -o nul -w "%%{http_code}" http://localhost:7860/ 2>nul | findstr "200" >nul
if errorlevel 1 goto wait

:: 開啟瀏覽器
echo.
echo  伺服器就緒！開啟瀏覽器...
echo.
start http://localhost:7860

echo  ========================================
echo   已在背景執行
echo   關閉此視窗即停止伺服器
echo  ========================================
echo.
echo  按任意鍵關閉伺服器...
pause >nul

:: 關閉 python 進程
taskkill /f /im python.exe /fi "WINDOWTITLE eq AI 影片自動後製工作流" >nul 2>&1
