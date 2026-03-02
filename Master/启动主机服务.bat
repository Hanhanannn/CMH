@echo off
chcp 65001 > nul 2>&1

echo ======================================
echo 中药材AI识别系统 - 主机服务启动脚本
echo ======================================
echo.

rem 检查Python是否安装
where python > nul 2>&1
if %errorlevel% neq 0 (
    echo 未检测到Python环境，正在尝试安装...
    rem 这里可以添加自动安装Python的逻辑
    pause
    exit /b 1
)

echo 检测到Python环境，版本：
python --version
echo.

rem 检查并安装依赖
echo 正在检查依赖...
python -c "import pip" > nul 2>&1
if %errorlevel% neq 0 (
    echo 请确保pip已正确安装
    pause
    exit /b 1
)

echo 正在安装依赖...
pip install -r dependencies/online_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if %errorlevel% neq 0 (
    echo 在线安装失败，尝试离线安装...
    pip install --no-index --find-links=dependencies/packages -r dependencies/offline_requirements.txt
    if %errorlevel% neq 0 (
        echo 离线安装失败，请手动安装依赖
        pause
        exit /b 1
    )
)

echo.
echo 依赖安装完成，正在启动同步中心服务...
echo 监控终端将显示同步状态和日志...
echo.
echo ======================================
echo 同步中心服务启动中...
echo ======================================
echo.

rem 启动同步中心服务
python sync_center_server.py

pause
