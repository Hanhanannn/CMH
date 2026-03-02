import json
import os
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_path="config/config.json"):
        self.logger = logger
        self.config_path = config_path
        self.backup_dir = "config/backup"
        
        # 确保配置目录和备份目录存在
        config_dir = os.path.dirname(self.config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
            self.logger.info(f"创建配置目录：{config_dir}")
        
        os.makedirs(self.backup_dir, exist_ok=True)
        self.logger.info(f"初始化ConfigManager，配置路径：{config_path}，备份目录：{self.backup_dir}")
        
        self.config = self._load_config()  # 加载配置

    # 加载配置文件
    def _load_config(self):
        try:
            if not os.path.exists(self.config_path):
                self.logger.warning(f"配置文件不存在：{self.config_path}，将初始化默认配置")
                # 初始化默认配置
                return self._init_default_config()
            
            self.logger.info(f"加载配置文件：{self.config_path}")
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            self.logger.debug(f"配置文件加载成功，原始配置：{config}")
            # 确保配置完整，补充缺失的配置项
            return self._ensure_config_complete(config)
        except json.JSONDecodeError as e:
            self.logger.error(f"配置文件格式错误：{self.config_path}，错误：{e}")
            self.logger.warning("将使用默认配置")
            return self._init_default_config()
        except Exception as e:
            self.logger.error(f"加载配置文件时发生错误：{self.config_path}，错误：{e}")
            self.logger.warning("将使用默认配置")
            return self._init_default_config()

    # 确保配置完整，补充缺失的配置项
    def _ensure_config_complete(self, config):
        # 默认配置模板
        default = {
            "camera": {"default_index": 0, "resolution": "640x480", "exposure": 500},
            "recognition": {"threshold": 0.75, "topK": 3, "min_trigger_weight": 20, "weight_stable_time": 1000},
            "system": {"auto_start": True, "log_level": "info"},
            "roi": {"type": "rect", "params": {"x": 100, "y": 100, "w": 400, "h": 400}},
            "database": {
                "host": "localhost",
                "port": 3306,
                "user": "root",
                "password": "123456",
                "db": "zhongyaocai_db"
            }
        }
        
        self.logger.info("检查配置完整性")
        config_updated = False
        
        # 补充缺失的配置项
        for section, section_config in default.items():
            if section not in config:
                config[section] = section_config
                config_updated = True
                self.logger.info(f"添加缺失的配置节：{section}，配置：{section_config}")
            else:
                for key, value in section_config.items():
                    if key not in config[section]:
                        config[section][key] = value
                        config_updated = True
                        self.logger.info(f"在配置节 {section} 中添加缺失的配置项：{key} = {value}")
        
        if config_updated:
            self.logger.info("配置已更新，将保存到文件")
            self.logger.debug(f"更新后的配置：{config}")
            # 保存更新后的配置
            self.config = config
            self.save_config()
        else:
            self.logger.info("配置完整，无需更新")
            self.config = config
        
        return config

    # 初始化默认配置
    def _init_default_config(self):
        self.logger.info("初始化默认配置")
        default = {
            "camera": {"default_index": 0, "resolution": "640x480", "exposure": 500},
            "recognition": {"threshold": 0.75, "topK": 3},
            "system": {"auto_start": True, "log_level": "info"},
            "roi": {"type": "rect", "params": {"x": 100, "y": 100, "w": 400, "h": 400}},
            "database": {
                "host": "localhost",
                "port": 3306,
                "user": "root",
                "password": "123456",
                "db": "zhongyaocai_db"
            }
        }
        
        self.logger.debug(f"默认配置：{default}")
        self.config = default
        self.save_config()  # 保存默认配置到文件
        return default

    # 保存配置（自动备份历史版本）
    def save_config(self):
        try:
            # 备份当前配置
            backup_path = os.path.join(
                self.backup_dir,
                f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            self.logger.info(f"保存配置备份：{backup_path}")
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"保存配置文件：{self.config_path}")
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"配置保存成功，配置内容：{self.config}")
        except Exception as e:
            self.logger.error(f"保存配置时发生错误：{e}")

    # 修改参数
    def set_param(self, section, key, value):
        self.logger.info(f"修改配置参数，节：{section}，键：{key}，新值：{value}")
        
        try:
            if section in self.config and key in self.config[section]:
                old_value = self.config[section][key]
                if old_value == value:
                    self.logger.info(f"配置参数未变化，节：{section}，键：{key}，值：{value}")
                    return True
                
                self.logger.debug(f"修改前值：{old_value}，修改后值：{value}")
                self.config[section][key] = value
                self.save_config()  # 自动保存
                self.logger.info(f"配置参数修改成功，节：{section}，键：{key}")
                return True
            elif section in self.config:
                # 如果section存在但key不存在，添加新key
                self.logger.info(f"添加新配置参数，节：{section}，键：{key}，值：{value}")
                self.config[section][key] = value
                self.save_config()
                return True
            else:
                self.logger.warning(f"配置节不存在：{section}，无法修改参数：{key}")
                return False
        except Exception as e:
            self.logger.error(f"修改配置参数时发生错误：{e}")
            return False

    # 获取参数
    def get_param(self, section, key, default=None):
        value = self.config.get(section, {}).get(key, default)
        self.logger.debug(f"获取配置参数，节：{section}，键：{key}，值：{value}")
        return value

    # 获取所有配置
    def get_all_config(self):
        self.logger.info("获取所有配置")
        self.logger.debug(f"所有配置：{self.config}")
        return self.config

    # 回滚到指定版本（简化实现，实际需根据版本号查找对应备份文件）
    def rollback_to_version(self, version):
        self.logger.info(f"尝试回滚配置到版本：{version}")
        try:
            # 简化实现，实际需根据版本号查找对应备份文件
            # 这里假设版本号对应备份文件中的某个标记
            self.logger.warning("回滚功能为简化实现，实际需根据版本号查找对应备份文件")
            # 在实际实现中，应该查找对应的备份文件并加载
            return True
        except Exception as e:
            self.logger.error(f"回滚配置时发生错误：{e}")
            return False
