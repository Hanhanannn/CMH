@echo off
chcp 65001 > nul 2>&1
echo 正在下载离线依赖包...
python download_dependencies.py
echo.
echo 下载完成！
pause
