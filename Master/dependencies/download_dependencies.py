#!/usr/bin/env python3
"""
依赖下载脚本 - 用于在有网络环境下下载离线依赖包
"""

import os
import subprocess
import sys

def main():
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 依赖存储目录
    packages_dir = os.path.join(current_dir, "packages")
    os.makedirs(packages_dir, exist_ok=True)
    
    # 读取依赖列表
    requirements_file = os.path.join(current_dir, "online_requirements.txt")
    
    if not os.path.exists(requirements_file):
        print(f"未找到依赖文件: {requirements_file}")
        return 1
    
    # 下载依赖
    print("开始下载依赖包...")
    
    try:
        # 使用pip下载依赖
        cmd = [
            sys.executable, "-m", "pip", "download",
            "-r", requirements_file,
            "-d", packages_dir,
            "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
            "--no-cache-dir"
        ]
        
        subprocess.run(cmd, check=True, shell=True if sys.platform == "win32" else False)
        
        print("
依赖包下载完成！")
        print(f"依赖包已存储在: {packages_dir}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"下载失败: {e}")
        return 1
    except Exception as e:
        print(f"发生错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
