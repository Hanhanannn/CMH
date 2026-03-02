#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志管理模块
"""

import os
import logging
from datetime import datetime
from config_manager import ConfigManager

class Logger:
    def __init__(self, name="herb_recognition", log_dir="logs"):
        """初始化日志管理器"""
        # 尝试创建ConfigManager实例，如果失败则使用默认日志级别
        try:
            self.config_manager = ConfigManager()
        except Exception as e:
            self.config_manager = None
        
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 配置日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)  # 设置最低日志级别
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self._get_log_level())
        
        # 创建文件处理器
        log_file = os.path.join(self.log_dir, f"{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        
        # 设置日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def _get_log_level(self):
        """根据配置获取日志级别"""
        # 如果config_manager为None，使用默认日志级别
        if self.config_manager is None:
            return logging.INFO
        
        try:
            log_level = self.config_manager.get_param("system", "log_level", "info")
            level_map = {
                "debug": logging.DEBUG,
                "info": logging.INFO,
                "warning": logging.WARNING,
                "error": logging.ERROR,
                "critical": logging.CRITICAL
            }
            return level_map.get(log_level.lower(), logging.INFO)
        except Exception:
            # 如果获取日志级别失败，使用默认值
            return logging.INFO
    
    def debug(self, message, **kwargs):
        """记录调试信息"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message, **kwargs):
        """记录普通信息"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message, **kwargs):
        """记录警告信息"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message, **kwargs):
        """记录错误信息"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message, **kwargs):
        """记录严重错误信息"""
        self.logger.critical(message, **kwargs)
    
    def update_log_level(self):
        """更新日志级别"""
        log_level = self._get_log_level()
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(log_level)
