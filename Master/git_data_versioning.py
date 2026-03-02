#!/usr/bin/env python3
"""
药材特征数据版本控制脚本
使用Git对药材特征数据库进行版本管理
"""

import os
import subprocess
import sys
from datetime import datetime

class DataVersionController:
    """药材特征数据版本控制器"""
    
    def __init__(self, data_dir="data"):
        """初始化数据版本控制器"""
        self.data_dir = data_dir
        self.git_repo_path = data_dir
        self.logger = self._setup_logger()
        
        print("=== 药材特征数据版本控制器初始化 ===")
    
    def _setup_logger(self):
        """设置日志记录器"""
        def log(message):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
        return log
    
    def _run_git_command(self, cmd, cwd=None):
        """运行Git命令"""
        cwd = cwd or self.git_repo_path
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True
            )
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)
    
    def init_git_repo(self):
        """初始化Git仓库"""
        self.logger(f"=== 初始化Git仓库 ===")
        
        # 检查目录是否存在
        if not os.path.exists(self.git_repo_path):
            self.logger(f"❌ 目录不存在：{self.git_repo_path}")
            return False
        
        # 检查是否已经是Git仓库
        is_repo, _, _ = self._run_git_command("git rev-parse --is-inside-work-tree")
        if is_repo:
            self.logger(f"✅ 目录已经是Git仓库：{self.git_repo_path}")
            return True
        
        # 初始化Git仓库
        success, stdout, stderr = self._run_git_command("git init")
        if success:
            self.logger(f"✅ Git仓库初始化成功：{self.git_repo_path}")
            
            # 创建.gitignore文件
            gitignore_content = """# 忽略临时文件
*.tmp
*.temp
*~\n
# 忽略日志文件
*.log
logs/\n
# 忽略Python缓存文件
__pycache__/
*.pyc\n
# 忽略IDE配置文件
.idea/
.vscode/
*.swp\n
# 忽略环境变量文件
.env
.env.local
.env.*.local\n
# 忽略测试文件
test_*.py
"""
            gitignore_path = os.path.join(self.git_repo_path, ".gitignore")
            with open(gitignore_path, 'w') as f:
                f.write(gitignore_content)
            self.logger(f"✅ .gitignore文件创建成功")
            
            # 第一次提交
            self.commit_changes("初始化药材特征数据库版本控制")
            
            return True
        else:
            self.logger(f"❌ Git仓库初始化失败：{stderr}")
            return False
    
    def commit_changes(self, commit_message):
        """提交数据变更"""
        self.logger(f"=== 提交数据变更 ===")
        
        # 检查是否是Git仓库
        is_repo, _, _ = self._run_git_command("git rev-parse --is-inside-work-tree")
        if not is_repo:
            self.logger(f"❌ 目录不是Git仓库：{self.git_repo_path}")
            return False
        
        # 检查是否有未提交的变更
        success, stdout, _ = self._run_git_command("git status --porcelain")
        if not success:
            self.logger(f"❌ 检查Git状态失败")
            return False
        
        if not stdout.strip():
            self.logger(f"✅ 没有未提交的变更")
            return True
        
        # 添加所有变更
        success, stdout, stderr = self._run_git_command("git add .")
        if not success:
            self.logger(f"❌ 添加变更失败：{stderr}")
            return False
        
        # 提交变更
        success, stdout, stderr = self._run_git_command(f"git commit -m \"{commit_message}\"")
        if success:
            self.logger(f"✅ 数据变更提交成功")
            self.logger(f"   提交信息：{commit_message}")
            return True
        else:
            self.logger(f"❌ 提交变更失败：{stderr}")
            return False
    
    def get_version_history(self):
        """获取版本历史"""
        self.logger(f"=== 获取版本历史 ===")
        
        # 检查是否是Git仓库
        is_repo, _, _ = self._run_git_command("git rev-parse --is-inside-work-tree")
        if not is_repo:
            self.logger(f"❌ 目录不是Git仓库：{self.git_repo_path}")
            return False
        
        # 获取版本历史
        success, stdout, stderr = self._run_git_command("git log --oneline -n 20")
        if success:
            self.logger(f"✅ 版本历史获取成功")
            self.logger(f"   最近20个版本：")
            for line in stdout.strip().split('\n'):
                self.logger(f"   {line}")
            return True
        else:
            self.logger(f"❌ 获取版本历史失败：{stderr}")
            return False
    
    def run(self):
        """执行数据版本控制初始化"""
        self.logger(f"=== 开始数据版本控制初始化流程 ===")
        
        # 1. 初始化Git仓库
        if self.init_git_repo():
            # 2. 提交初始数据
            self.commit_changes("初始化药材特征数据库")
            # 3. 获取版本历史
            self.get_version_history()
        
        self.logger(f"=== 数据版本控制初始化流程完成 ===")

if __name__ == "__main__":
    # 创建数据版本控制器实例
    version_controller = DataVersionController()
    
    # 执行数据版本控制初始化
    version_controller.run()
