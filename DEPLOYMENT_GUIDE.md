# 中药材AI识别系统 - 打包部署指南

## 1. 项目概述

本项目是一个中药材AI识别系统，包含两个主要服务：

- **主机服务 (Master)**：负责模型管理、数据同步和中心服务功能
- **从机服务 (Slave)**：负责摄像头管理、图像采集和实时识别功能

## 2. 打包内容

### 2.1 主机服务 (Master) 包含：
- 核心源代码文件 (*.py)
- 配置文件和目录 (config/)
- 数据文件和映射 (data/)
- 依赖管理文件 (dependencies/)
- 前端构建文件 (frontend/)
- 学习引擎模块 (learning_engine/)
- 启动脚本 (启动主机服务.bat)

### 2.2 从机服务 (Slave) 包含：
- 核心源代码文件 (*.py)
- 模型文件 (base_lib/)
- 配置文件和目录 (config/)
- 数据文件和映射 (data/)
- 依赖管理文件 (dependencies/)
- 前端构建文件 (frontend/)
- 学习引擎模块 (learning_engine/)
- FFmpeg 和 MediaMTX 可执行文件
- 各种文档和测试文件

## 3. 部署步骤

### 3.1 环境要求
- Windows 操作系统
- Python 3.8 或更高版本
- 网络连接（用于在线安装依赖）
- 摄像头设备（用于从机服务）

### 3.2 依赖安装

#### 3.2.1 在线安装（推荐）
1. 进入对应服务目录
2. 运行依赖安装命令：
   ```bash
   # 主机服务
   cd Master
   pip install -r dependencies/online_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   
   # 从机服务
   cd Slave
   pip install -r dependencies/online_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

#### 3.2.2 离线安装
1. 确保已下载离线依赖包到 `dependencies/packages/` 目录
2. 运行离线安装命令：
   ```bash
   # 主机服务
   cd Master
   pip install --no-index --find-links=dependencies/packages -r dependencies/offline_requirements.txt
   
   # 从机服务
   cd Slave
   pip install --no-index --find-links=dependencies/packages -r dependencies/offline_requirements.txt
   ```

## 4. 启动方法

### 4.1 主机服务启动
1. 进入 Master 目录
2. 双击运行 `启动主机服务.bat`
3. 或使用命令行：
   ```bash
   python sync_center_server.py
   ```

### 4.2 从机服务启动
1. 进入 Slave 目录
2. 运行主服务：
   ```bash
   python http_server.py
   ```
3. 如果需要 RTSP 流支持：
   - 先启动 MediaMTX：双击运行 `mediamtx.exe`
   - 然后启动主服务

## 5. 服务接口

### 5.1 主机服务接口
- 同步中心服务：默认运行在 5566 端口

### 5.2 从机服务接口
- HTTP API：默认运行在 5567 端口
- MJPEG 流：`http://<IP>:5567/ai/stream/live.mjpg`
- RTSP 流：`rtsp://<IP>:8554/live/cam`（需启动 MediaMTX）

## 6. 配置文件

### 6.1 主机服务配置
- `config/config.json`：主配置文件
- `center_config.json`：同步中心配置

### 6.2 从机服务配置
- `config/config.json`：主配置文件
- `config/app_config.ini`：应用配置
- `config/sync_config.ini`：同步配置

## 7. 常见问题排查

### 7.1 依赖安装失败
- 检查网络连接
- 尝试使用不同的 PyPI 镜像源
- 确认 Python 版本符合要求

### 7.2 服务启动失败
- 检查端口是否被占用
- 确认依赖已正确安装
- 查看日志文件获取详细错误信息

### 7.3 摄像头无法连接
- 检查摄像头是否被其他程序占用
- 确认摄像头驱动已正确安装
- 尝试修改摄像头索引配置

### 7.4 RTSP 流无法访问
- 确认 MediaMTX 已启动
- 检查防火墙是否放行 8554 端口
- 确认 FFmpeg 已正确安装

## 8. 技术支持

- 如有问题，请参考项目中的文档文件
- 或联系系统管理员获取技术支持

---

**版本信息**：V1.0.0
**打包日期**：2026-03-02
**适用平台**：Windows
