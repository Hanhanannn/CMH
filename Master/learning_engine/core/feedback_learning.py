from ..utils.db_handler import FeatureDBHandler
from ..utils.feature_extract import FeatureExtractor
from .sync_manager import SyncManager
import sys
import os
import time
# 添加项目根目录到sys.path
try:
    # 获取当前文件的绝对路径
    current_file = os.path.abspath(__file__)
    # 获取当前文件的目录
    current_dir = os.path.dirname(current_file)
    # 获取上一级目录
    parent_dir = os.path.dirname(current_dir)
    # 获取上上级目录
    grandparent_dir = os.path.dirname(parent_dir)
    # 获取项目根目录
    project_root = os.path.dirname(grandparent_dir)
    # 添加项目根目录到sys.path
    sys.path.append(project_root)
except Exception as e:
    print(f"添加项目根目录到sys.path失败：{e}")
    # 如果获取失败，添加当前目录到sys.path
    sys.path.append(os.getcwd())
from logger import Logger

class FeedbackLearning:
    """反馈学习核心类（cmd=201）"""
    def __init__(self):
        self.logger = Logger("FeedbackLearning")
        self.feature_db = FeatureDBHandler()
        self.extractor = FeatureExtractor()
        self.sync_manager = SyncManager()
        # 启动同步服务
        self.sync_manager.start()
        self.logger.info("反馈学习模块初始化完成")

    def learn(self, request_id, correct_plu, image_path):
        """
        执行反馈学习
        :param request_id: 识别请求ID
        :param correct_plu: 正确的PLU码
        :param image_path: 本次识别的原始图像路径
        :return: 学习结果（bool）
        """
        try:
            self.logger.info(f"开始学习，请求ID：{request_id}，PLU码：{correct_plu}，图像路径：{image_path}")
            
            # 1. 检查图像文件是否存在
            if not os.path.exists(image_path):
                self.logger.error(f"图像文件不存在：{image_path}")
                return False
            
            # 2. 提取图像特征
            self.logger.info("正在提取图像特征")
            try:
                feature = self.extractor.extract(image_path)
            except Exception as extract_e:
                self.logger.error(f"特征提取异常：{extract_e}", exc_info=True)
                return False
                
            if feature is None:
                self.logger.error("特征提取失败")
                return False
            self.logger.info(f"图像特征提取成功，特征维度：{len(feature)}")
            
            # 3. 检查是否为新药材首次学习
            try:
                existing_features = self.feature_db.get_features_by_plu(correct_plu)
                is_first_learn = len(existing_features) == 0
            except Exception as db_e:
                self.logger.error(f"获取已有特征失败：{db_e}", exc_info=True)
                return False
            
            if is_first_learn:
                self.logger.info(f"首次学习新药材，PLU码：{correct_plu}，初始化特征数据")
                # 首次学习，作为初始特征
                result = self.feature_db.add_feature(feature, correct_plu)
                if result:
                    self.logger.info(f"新药材特征初始化成功，PLU码：{correct_plu}")
                else:
                    self.logger.error(f"新药材特征初始化失败，PLU码：{correct_plu}")
                    return False
            else:
                self.logger.info(f"已有特征数量：{len(existing_features)}，执行特征迭代优化")
                # 已有特征，执行特征迭代优化
                result = self.feature_db.add_feature(feature, correct_plu)
                if result:
                    self.logger.info("特征迭代优化成功")
                else:
                    self.logger.error("特征迭代优化失败")
                    return False
            
            # 4. 异步触发同步上传（使用线程异步执行，不阻塞主流程）
            self.logger.info("正在异步触发同步上传")
            try:
                import threading
                threading.Thread(target=self._async_sync, args=(correct_plu,), daemon=True).start()
                self.logger.info("异步同步上传触发成功")
            except Exception as sync_e:
                self.logger.error(f"异步同步上传触发失败：{sync_e}")
                # 同步触发失败不影响学习结果
            
            self.logger.info(f"学习完成，请求ID：{request_id}")
            return True
        except Exception as e:
            self.logger.error(f"反馈学习失败：{e}", exc_info=True)
            return False
    
    def _async_sync(self, plu_code):
        """
        异步执行同步上传
        :param plu_code: PLU码
        """
        try:
            # 延迟执行，避免立即占用系统资源
            time.sleep(1)
            # 执行同步上传
            self.sync_manager.trigger_sync(plu_code)
            self.logger.info(f"异步同步上传完成，PLU码：{plu_code}")
        except Exception as e:
            self.logger.error(f"异步同步上传失败：{e}", exc_info=True)
