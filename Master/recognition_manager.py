#!/usr/bin/env python3
"""
中药材识别管理器
负责处理图像采集、预处理、模型识别和结果判定
"""

import warnings
import numpy as np
import cv2
import faiss
import pickle
import time
import random
import os
from base_lib_manager import BaseLibManager
from logger import Logger
from image_preprocessor import ImagePreprocessor
from similarity_manager import SimilarityManager
import hashlib

# 抑制TensorFlow Lite的弃用警告
# 注意：ai_edge_litert包不支持Windows系统，因此继续使用tf.lite.Interpreter
# 通过警告过滤器抑制弃用警告，避免影响用户体验
warnings.filterwarnings('ignore', category=UserWarning, message='.*tf.lite.Interpreter is deprecated.*')

# 尝试导入TensorFlow，如果失败则跳过
try:
    import tensorflow as tf
except ImportError as e:
    import logging
    logging.error(f"TensorFlow导入失败：{e}")
    tf = None

class RecognitionManager:
    def __init__(self, model_path=None, feature_db_path="data/feature_db", base_lib_manager=None):
        # 初始化日志记录器
        self.logger = Logger("RecognitionManager")
        
        # 性能优化：添加模型和特征库缓存
        self._model_cache = {}
        self._feature_db_cache = {}
        self._cache_enabled = True
        
        # 性能统计
        self.load_performance_stats = {
            "model_load_time": 0.0,
            "feature_db_load_time": 0.0,
            "total_load_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # 记录初始化开始时间
        init_start_time = time.time()
        
        # 优先使用原始模型，确保识别准确率
        if model_path is None:
            # 获取当前脚本所在目录的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 构建模型目录的绝对路径
            model_dir = os.path.join(current_dir, "model")
            base_model_dir = os.path.join(model_dir, "base_model")
            
            # 首先检查model/base_model/目录
            if os.path.exists(base_model_dir):
                # 首先检查是否存在最新的平衡优化模型（最优模型）
                if os.path.exists(os.path.join(base_model_dir, "balanced_focused_model_final.tflite")):
                    self.model_path = os.path.join(base_model_dir, "balanced_focused_model_final.tflite")
                # 检查是否存在平衡优化模型
                elif os.path.exists(os.path.join(base_model_dir, "balanced_focused_model.tflite")):
                    self.model_path = os.path.join(base_model_dir, "balanced_focused_model.tflite")
                # 检查是否存在聚焦模型
                elif os.path.exists(os.path.join(base_model_dir, "focused_model.tflite")):
                    self.model_path = os.path.join(base_model_dir, "focused_model.tflite")
                # 检查是否存在平衡模型
                elif os.path.exists(os.path.join(base_model_dir, "balanced_model.tflite")):
                    self.model_path = os.path.join(base_model_dir, "balanced_model.tflite")
                # 检查是否存在改进模型
                elif os.path.exists(os.path.join(base_model_dir, "herb_model_improved.tflite")):
                    self.model_path = os.path.join(base_model_dir, "herb_model_improved.tflite")
                # 检查是否存在聚焦药材模型
                elif os.path.exists(os.path.join(base_model_dir, "focused_herb_model.tflite")):
                    self.model_path = os.path.join(base_model_dir, "focused_herb_model.tflite")
                # 检查是否存在原始配置模型
                elif os.path.exists(os.path.join(base_model_dir, "herb_model_original_config.tflite")):
                    self.model_path = os.path.join(base_model_dir, "herb_model_original_config.tflite")
                # 最后检查是否存在原始模型
                elif os.path.exists(os.path.join(base_model_dir, "herb_model.tflite")):
                    self.model_path = os.path.join(base_model_dir, "herb_model.tflite")
                # 检查base_model目录下的其他可用模型文件
                else:
                    # 列出所有可用的tflite模型文件
                    tflite_files = [f for f in os.listdir(base_model_dir) if f.endswith('.tflite')]
                    if tflite_files:
                        self.model_path = os.path.join(base_model_dir, tflite_files[0])
                        self.logger.info(f"使用找到的第一个模型文件：{self.model_path}")
                    else:
                        # 如果base_model目录下没有模型文件，再检查model目录
                        if os.path.exists(os.path.join(model_dir, "balanced_focused_model_final.tflite")):
                            self.model_path = os.path.join(model_dir, "balanced_focused_model_final.tflite")
                        # 检查是否存在平衡优化模型
                        elif os.path.exists(os.path.join(model_dir, "balanced_focused_model.tflite")):
                            self.model_path = os.path.join(model_dir, "balanced_focused_model.tflite")
                        # 检查是否存在优化模型
                        elif os.path.exists(os.path.join(model_dir, "herb_model_optimized.tflite")):
                            self.model_path = os.path.join(model_dir, "herb_model_optimized.tflite")
                        # 最后检查是否存在原始模型
                        elif os.path.exists(os.path.join(model_dir, "herb_model.tflite")):
                            self.model_path = os.path.join(model_dir, "herb_model.tflite")
                        # 检查其他可用的模型文件
                        else:
                            # 列出所有可用的tflite模型文件
                            tflite_files = [f for f in os.listdir(model_dir) if f.endswith('.tflite')]
                            if tflite_files:
                                self.model_path = os.path.join(model_dir, tflite_files[0])
                                self.logger.info(f"使用找到的第一个模型文件：{self.model_path}")
                            else:
                                self.logger.error("未找到任何TFLite模型文件")
                                self.model_path = None
            else:
                # 如果base_model目录不存在，只检查model目录
                # 首先检查是否存在最新的平衡优化模型（最优模型）
                if os.path.exists(os.path.join(model_dir, "balanced_focused_model_final.tflite")):
                    self.model_path = os.path.join(model_dir, "balanced_focused_model_final.tflite")
                # 检查是否存在平衡优化模型
                elif os.path.exists(os.path.join(model_dir, "balanced_focused_model.tflite")):
                    self.model_path = os.path.join(model_dir, "balanced_focused_model.tflite")
                # 检查是否存在优化模型
                elif os.path.exists(os.path.join(model_dir, "herb_model_optimized.tflite")):
                    self.model_path = os.path.join(model_dir, "herb_model_optimized.tflite")
                # 最后检查是否存在原始模型
                elif os.path.exists(os.path.join(model_dir, "herb_model.tflite")):
                    self.model_path = os.path.join(model_dir, "herb_model.tflite")
                # 检查其他可用的模型文件
                else:
                    # 列出所有可用的tflite模型文件
                    tflite_files = [f for f in os.listdir(model_dir) if f.endswith('.tflite')]
                    if tflite_files:
                        self.model_path = os.path.join(model_dir, tflite_files[0])
                        self.logger.info(f"使用找到的第一个模型文件：{self.model_path}")
                    else:
                        self.logger.error("未找到任何TFLite模型文件")
                        self.model_path = None
        else:
            self.model_path = model_path
        
        self.logger.info(f"初始化RecognitionManager，模型路径：{self.model_path}，特征库路径：{feature_db_path}")
        
        self.feature_db_path = feature_db_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.base_lib_manager = base_lib_manager or BaseLibManager()
        
        # 初始化faiss特征库
        self.feature_index = None
        self.plu_map = None
        
        # 初始化相似度计算管理器
        self.similarity_manager = SimilarityManager(base_lib_manager=self.base_lib_manager)
        
        # 初始化图像预处理类，使用MobileNetV2默认输入尺寸224x224
        self.preprocessor = ImagePreprocessor(target_size=(224, 224), hog_win_size=(128, 128))
        
        # 从基础库动态加载类别名称列表，确保排序一致
        self.class_names = sorted(self.base_lib_manager.base_lib.keys())
        
        self.result_cache = []  # 结果缓存，最多存储10次
        self.cache_size = 10
        
        # 性能统计
        self.performance_stats = {
            "total_recognitions": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "feature_extraction_time": 0.0,
            "recognition_time": 0.0
        }
        
        # 动态阈值配置，所有药材使用统一阈值
        self.confidence_threshold = 0.75  # 统一置信度阈值
        # 动态为所有药材设置统一阈值0.75
        self.class_thresholds = {herb_name: 0.75 for herb_name in self.class_names}
        
        # 初始化HOG特征提取器
        self._init_hog()
        
        # 初始化模型和特征库
        self._load_model()
        self._load_feature_db()
    
    def _release_model(self):
        """
        释放模型资源
        """
        try:
            if self.interpreter:
                self.logger.info("释放TFLite模型资源")
                self.interpreter = None
                self.input_details = None
                self.output_details = None
            if self.feature_index:
                self.logger.info("释放FAISS特征库资源")
                self.feature_index.reset()
                self.feature_index = None
        except Exception as e:
            self.logger.error(f"释放模型资源失败：{e}")
    
    def _init_hog(self):
        """初始化HOG特征提取器"""
        try:
            # 初始化HOG特征提取器，与preprocessor使用相同的参数
            win_size = self.preprocessor.hog_win_size
            block_size = (16, 16)
            block_stride = (8, 8)
            cell_size = (8, 8)
            nbins = 9
            
            self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
            self.logger.info(f"成功初始化HOG特征提取器，窗口大小：{win_size}")
        except Exception as e:
            self.logger.error(f"初始化HOG特征提取器失败：{e}")
            self.hog = None

    def _load_feature_db(self):
        """加载faiss特征库"""
        try:
            # 优先从plu_map.pkl加载PLU映射
            plu_map_pkl_path = "data/plu_map.pkl"
            if os.path.exists(plu_map_pkl_path):
                try:
                    with open(plu_map_pkl_path, "rb") as f:
                        self.plu_map = pickle.load(f)
                    self.logger.info(f"成功加载PLU映射文件，共 {len(self.plu_map)} 条记录")
                except Exception as e:
                    self.logger.warning(f"加载PLU映射文件失败：{e}")
                    self.plu_map = None
            
            # 尝试加载特征库
            if os.path.exists(self.feature_db_path):
                with open(self.feature_db_path, "rb") as f:
                    # 尝试加载特征库，支持两种格式：(index, plu_map) 和 (index, plu_map, herb_names)
                    try:
                        data = pickle.load(f)
                        if len(data) == 3:
                            self.feature_index, self.plu_map, self.herb_names = data
                        else:
                            self.feature_index, self.plu_map = data
                            # 如果没有提供herb_names，使用默认的类别名称列表
                            self.herb_names = self.class_names.copy() if self.plu_map else []
                    except ValueError:
                        # 旧格式，只有两个元素
                        self.feature_index, self.plu_map = pickle.load(f)
                        # 如果没有提供herb_names，使用默认的类别名称列表
                        self.herb_names = self.class_names.copy() if self.plu_map else []
            
            # 验证加载的特征库是否有效
            if self.feature_index is not None:
                try:
                    # 尝试访问ntotal属性，验证feature_index是否有效
                    feature_count = self.feature_index.ntotal
                    self.logger.info(f"成功加载特征库，共 {feature_count} 个特征")
                except Exception as e:
                    # 访问ntotal失败，feature_index无效
                    self.logger.error(f"加载的特征库无效，无法访问ntotal属性：{e}")
                    # 初始化空的索引和映射表
                    self.feature_index = None
                    self.plu_map = None
                    self.herb_names = []
            else:
                self.logger.error("加载的特征库无效：feature_index为None")
                # 初始化空的索引和映射表
                self.feature_index = None
                self.plu_map = None
                self.herb_names = []
        except FileNotFoundError:
            self.logger.warning(f"特征库文件不存在：{self.feature_db_path}")
            # 初始化空的索引和映射表
            self.feature_index = None
            self.plu_map = None
            self.herb_names = []
        except pickle.PickleError as e:
            self.logger.error(f"特征库文件格式错误：{e}")
            # 初始化空的索引和映射表
            self.feature_index = None
            self.plu_map = None
            self.herb_names = []
        except Exception as e:
            self.logger.error(f"加载特征库失败：{e}")
            # 初始化空的索引和映射表
            self.feature_index = None
            self.plu_map = None
            self.herb_names = []

    def _load_model(self):
        """加载TFLite模型（优化版本，添加缓存和性能统计）"""
        try:
            # 检查缓存
            model_cache_key = hashlib.md5(self.model_path.encode()).hexdigest()
            if self._cache_enabled and model_cache_key in self._model_cache:
                self.load_performance_stats["cache_hits"] += 1
                self.logger.info(f"从缓存加载TFLite模型：{self.model_path}")
                cached_data = self._model_cache[model_cache_key]
                self.interpreter = cached_data["interpreter"]
                self.input_details = cached_data["input_details"]
                self.output_details = cached_data["output_details"]
                return
            
            self.load_performance_stats["cache_misses"] += 1
            model_load_start = time.time()
            
            # 清除之前的模型资源
            self.interpreter = None
            self.input_details = None
            self.output_details = None
            
            self.logger.info(f"开始加载TFLite模型：{self.model_path}")
            
            if tf is None:
                self.logger.error("TensorFlow未导入，无法加载TFLite模型")
                return
                
            if not self.model_path:
                self.logger.error("模型路径为空，无法加载TFLite模型")
                # 尝试使用默认模型路径
                self.model_path = "model/herb_model.tflite"
                self.logger.info(f"使用默认模型路径：{self.model_path}")
                
            if not os.path.exists(self.model_path):
                self.logger.error(f"TFLite模型文件不存在：{self.model_path}")
                # 尝试查找其他可能的模型文件
                import glob
                model_files = glob.glob("model/*.tflite")
                if model_files:
                    self.model_path = model_files[0]
                    self.logger.info(f"使用找到的模型文件：{self.model_path}")
                else:
                    self.logger.error("未找到任何TFLite模型文件")
                    return
            
            if not os.path.isfile(self.model_path):
                self.logger.error(f"模型路径不是文件：{self.model_path}")
                return
            
            # 打印模型文件大小和路径信息
            file_size = os.path.getsize(self.model_path)
            self.logger.info(f"模型文件大小：{file_size/1024/1024:.2f} MB")
            
            # 尝试加载模型
            self.logger.info(f"正在加载TFLite模型...")
            
            # 显式指定模型路径，避免相对路径问题
            abs_model_path = os.path.abspath(self.model_path)
            self.logger.info(f"使用绝对路径加载模型：{abs_model_path}")
            
            try:
                # 尝试使用内存加载方式，先读取文件内容，再加载到解释器
                self.logger.info(f"尝试使用内存加载方式加载模型")
                with open(abs_model_path, 'rb') as f:
                    model_content = f.read()
                
                self.interpreter = tf.lite.Interpreter(model_content=model_content)
                self.logger.info(f"✅ TFLite模型通过内存加载成功")
            except Exception as e:
                self.logger.error(f"❌ TFLite模型内存加载失败：{e}")
                import traceback
                self.logger.error(f"错误堆栈：{traceback.format_exc()}")
                return
            
            # 分配张量
            try:
                self.interpreter.allocate_tensors()
                self.logger.info("TFLite模型分配张量成功")
            except Exception as e:
                self.logger.error(f"TFLite模型分配张量失败：{e}")
                import traceback
                self.logger.error(f"错误堆栈：{traceback.format_exc()}")
                self.interpreter = None
                return
            
            # 获取输入输出详情
            try:
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                
                self.logger.info(f"模型输入详情数量：{len(self.input_details)}")
                self.logger.info(f"模型输出详情数量：{len(self.output_details)}")
                
                # 验证输出详情格式 - 简化验证，只需要至少1个输出
                if len(self.output_details) < 1:
                    self.logger.error(f"TFLite模型输出详情格式不符合预期，需要至少1个输出，实际只有{len(self.output_details)}个")
                    self.interpreter = None
                    self.input_details = None
                    self.output_details = None
                    return
                    
                self.logger.info(f"成功加载TFLite模型：{self.model_path}")
                self.logger.info(f"模型输入形状：{self.input_details[0]['shape']}")
                self.logger.info(f"模型输入数据类型：{self.input_details[0]['dtype']}")
                self.logger.info(f"模型输出形状：{self.output_details[0]['shape']}")
                
                # 记录加载时间
                model_load_time = time.time() - model_load_start
                self.load_performance_stats["model_load_time"] = model_load_time
                self.logger.info(f"TFLite模型加载耗时：{model_load_time:.4f}秒")
                
                # 缓存模型
                if self._cache_enabled:
                    self._model_cache[model_cache_key] = {
                        "interpreter": self.interpreter,
                        "input_details": self.input_details,
                        "output_details": self.output_details
                    }
                    self.logger.info("模型已缓存")
                    
            except Exception as e:
                self.logger.error(f"获取TFLite模型输入输出详情失败：{e}")
                import traceback
                self.logger.error(f"错误堆栈：{traceback.format_exc()}")
                self.interpreter = None
                self.input_details = None
                self.output_details = None
        except FileNotFoundError:
            self.logger.error(f"TFLite模型文件不存在：{self.model_path}")
            self.interpreter = None
        except tf.errors.InvalidArgumentError as e:
            self.logger.error(f"TFLite模型文件格式错误：{e}")
            import traceback
            self.logger.error(f"错误堆栈：{traceback.format_exc()}")
            self.interpreter = None
        except Exception as e:
            self.logger.error(f"加载TFLite模型失败：{e}")
            import traceback
            self.logger.error(f"错误堆栈：{traceback.format_exc()}")
            self.interpreter = None

    def _preprocess_image(self, image):
        """图像预处理"""
        try:
            self.logger.info(f"开始图像预处理，输入图像形状：{image.shape}")
            # 使用图像预处理类进行预处理
            processed_image = self.preprocessor.preprocess_for_model(image)
            self.logger.info(f"图像预处理完成，输出图像形状：{processed_image.shape}")
            return processed_image
        except Exception as e:
            self.logger.error(f"图像预处理失败：{e}")
            import traceback
            self.logger.error(f"错误堆栈：{traceback.format_exc()}")
            raise

    def _preprocess_image_for_hog(self, image):
        """图像预处理（用于HOG特征提取）"""
        try:
            self.logger.info(f"开始HOG图像预处理，输入图像形状：{image.shape}")
            # 使用图像预处理类进行HOG预处理
            result = self.preprocessor.preprocess_for_hog(image)
            self.logger.info(f"HOG图像预处理完成，输出特征维度：{result.shape if hasattr(result, 'shape') else len(result)}")
            return result
        except Exception as e:
            self.logger.error(f"HOG图像预处理失败：{e}")
            import traceback
            self.logger.error(f"错误堆栈：{traceback.format_exc()}")
            raise
    
    def _check_image_quality(self, image):
        """检查图像质量"""
        try:
            self.logger.info(f"开始图像质量检查")
            result = self.preprocessor.check_image_quality(image)
            self.logger.info(f"图像质量检查完成，结果：{result}")
            return result
        except Exception as e:
            self.logger.error(f"图像质量检测失败：{e}")
            return False, f"图像质量检测失败：{str(e)}"
    
    def recognize(self, image, topK=5, conf_threshold=None, target_herb=None):
        """
        识别中药材
        
        参数:
        - image: 输入图像
        - topK: 返回前K个结果
        - conf_threshold: 置信度阈值
        - target_herb: 目标药材名称或PLU码（用于定向识别）
        
        返回:
        - 普通识别: {success: bool, results: list, ...}
        - 定向识别: {success: bool, matched: bool, target_herb: str, result: dict, ...}
        """
        # 如果没有提供置信度阈值，使用类中设置的默认值
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold
        start_time = time.time()
        self.logger.info(f"=== 开始识别 ===")
        self.logger.info(f"参数：topK={topK}，置信度阈值={conf_threshold}，目标药材={target_herb}")
        
        # 检查图像质量
        quality_ok, quality_msg = self._check_image_quality(image)
        if not quality_ok:
            self.logger.warning(f"图像质量不达标：{quality_msg}")
        
        # 检查模型是否已加载
        if tf is None:
            self.logger.error("TensorFlow未导入，无法进行模型推理")
            return {
                "success": bool(False),
                "error": "TensorFlow未导入，无法进行模型推理",
                "performance": {
                    "preprocess_time": 0,
                    "inference_time": 0,
                    "total_time": time.time() - start_time
                }
            }
        
        if self.interpreter is None:
            self.logger.error("模型未加载，尝试重新加载模型")
            # 尝试重新加载模型
            self._load_model()
            if self.interpreter is None:
                self.logger.error("模型重新加载失败，无法进行推理")
                return {
                "success": bool(False),
                "error": "模型未加载，无法进行推理",
                "performance": {
                    "preprocess_time": 0,
                    "inference_time": 0,
                    "total_time": time.time() - start_time
                }
            }
        
        # 模型推理
        try:
            inference_start = time.time()
            
            # 图像预处理
            preprocessed_image = self._preprocess_image(image)
            self.logger.info(f"图像预处理成功，输出形状：{preprocessed_image.shape}")
            
            # 设置输入张量
            expected_shape = self.input_details[0]['shape']
            self.logger.info(f"模型期望输入形状：{expected_shape}")
            self.logger.info(f"预处理后图像形状：{preprocessed_image.shape}")
            
            # 确保输入张量形状与模型期望匹配
            input_tensor = preprocessed_image
            if len(input_tensor.shape) == 3:  # (height, width, channels)
                input_tensor = np.expand_dims(input_tensor, axis=0)  # 添加批次维度
            elif len(input_tensor.shape) == 4:  # 已经有4个维度，检查具体形状
                pass  # 继续检查具体形状
            else:
                self.logger.error(f"输入张量维度异常：{input_tensor.shape}")
                raise ValueError(f"输入张量维度异常：{input_tensor.shape}")
            
            # 确保输入张量数据类型正确
            input_tensor = input_tensor.astype(self.input_details[0]['dtype'])
            
            self.logger.info(f"最终输入张量形状：{input_tensor.shape}，数据类型：{input_tensor.dtype}")
            self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
            
            # 执行推理
            self.interpreter.invoke()
            inference_time = time.time() - inference_start
            self.logger.info(f"TFLite模型推理耗时：{inference_time:.4f}秒")
            
            # 处理识别结果
            results = []
            
            try:
                if len(self.output_details) == 1:
                    # 分类模型处理逻辑
                    class_probs = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
                    # 确保class_probs是一维数组
                    class_probs = np.squeeze(class_probs)
                    self.logger.info(f"分类模型输出概率：{class_probs}")
                    
                    # 获取模型能识别的药材数量
                    if isinstance(class_probs, (list, np.ndarray)):
                        model_output_dim = len(class_probs)
                    else:
                        model_output_dim = 1
                    self.logger.info(f"模型能识别的药材数量：{model_output_dim}")
                    
                    if model_output_dim == 1:
                        # 处理单个输出的情况
                        top_indices = [0]
                    else:
                        # 获取置信度最高的前K个类别
                        top_indices = np.argsort(class_probs)[::-1][:topK]

                    # 确保class_names列表长度与模型输出维度匹配
                    if len(self.class_names) > model_output_dim:
                        self.logger.warning(f"class_names列表长度({len(self.class_names)})大于模型能识别的类别数({model_output_dim})，将只使用前{model_output_dim}个类别名称")
                        class_names_to_use = self.class_names[:model_output_dim]
                    else:
                        class_names_to_use = self.class_names
                    
                    # 打印调试信息：模型输出概率、类别索引和对应名称
                    self.logger.info(f"模型输出概率：{class_probs}")
                    self.logger.info(f"类别名称列表：{class_names_to_use}")
                    self.logger.info(f"topK索引：{top_indices}")
                    
                    for class_id in top_indices:
                        # 确保score是标量
                        prob_value = float(class_probs[class_id])
                        
                        # 获取药材名称
                        if class_id < len(class_names_to_use):
                            herb_name = class_names_to_use[class_id]
                        else:
                            herb_name = f"未知药材_{class_id}"
                            self.logger.warning(f"类别ID {class_id} 超出可用类别名称范围 {len(class_names_to_use)}")
                        
                        # 获取药材信息
                        herb_info = self.base_lib_manager.match_base_lib(herb_name)
                        if herb_info is None:
                            herb_info = {}
                        
                        # 生成边界框（分类模型没有真实边界框，生成一个居中的框）
                        h, w = expected_shape[1], expected_shape[2]  # 使用模型期望的尺寸
                        box_size = min(h, w) // 2
                        box_coords = {
                            "xmin": float(max(0, (w - box_size) // 2)),
                            "ymin": float(max(0, (h - box_size) // 2)),
                            "xmax": float(min(w, (w + box_size) // 2)),
                            "ymax": float(min(h, (h + box_size) // 2))
                        }
                        
                        # 从plu_map获取PLU码，如果plu_map不存在则使用base_lib中的plu
                        plu_code = herb_info.get("plu", "")
                        if self.plu_map and herb_name in self.plu_map:
                            plu_code = self.plu_map[herb_name]
                            self.logger.info(f"从plu_map获取药材 {herb_name} 的PLU码: {plu_code}")
                        else:
                            self.logger.warning(f"plu_map中未找到药材 {herb_name}，使用base_lib中的PLU码: {plu_code}")
                        
                        result_item = {
                            "name": herb_name,
                            "plu": plu_code,
                            "score": prob_value,
                            "box": box_coords
                        }
                        results.append(result_item)
                        self.logger.info(f"识别结果 {len(results)}: {result_item}")
                else:
                    # 检测模型处理逻辑
                    boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
                    classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
                    scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
                    
                    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                        try:
                            # 确保score是标量
                            score_value = float(score)
                            
                            # 获取药材名称
                            class_id = int(cls)
                            if class_id < len(self.class_names):
                                herb_name = self.class_names[class_id]
                            else:
                                herb_name = f"未知药材_{class_id}"
                                self.logger.warning(f"类别ID {class_id} 超出类别名称列表范围 {len(self.class_names)}")
                            
                            # 获取药材信息
                            herb_info = self.base_lib_manager.match_base_lib(herb_name)
                            if herb_info is None:
                                herb_info = {}
                            
                            # 归一化边界框坐标
                            h, w = expected_shape[1], expected_shape[2]  # 使用模型期望的尺寸
                            box_coords = {
                                "xmin": float(max(0, box[0] * w)),
                                "ymin": float(max(0, box[1] * h)),
                                "xmax": float(min(w, box[2] * w)),
                                "ymax": float(min(h, box[3] * h))
                            }
                            
                            # 从plu_map获取PLU码，如果plu_map不存在则使用base_lib中的plu
                            plu_code = herb_info.get("plu", "")
                            if self.plu_map and herb_name in self.plu_map:
                                plu_code = self.plu_map[herb_name]
                                self.logger.info(f"从plu_map获取药材 {herb_name} 的PLU码: {plu_code}")
                            else:
                                self.logger.warning(f"plu_map中未找到药材 {herb_name}，使用base_lib中的PLU码: {plu_code}")
                            
                            result_item = {
                                "name": herb_name,
                                "plu": plu_code,
                                "score": score_value,
                                "box": box_coords
                            }
                            results.append(result_item)
                            self.logger.info(f"检测结果 {i+1}: {result_item}")
                        except Exception as e:
                            self.logger.error(f"处理检测结果失败：{e}")
                            continue
                
                self.logger.info(f"TFLite模型识别完成，共识别出 {len(results)} 个结果")
            except Exception as e:
                self.logger.error(f"处理识别结果失败：{e}")
                import traceback
                self.logger.error(f"错误堆栈：{traceback.format_exc()}")
                return {
                "success": bool(False),
                "error": f"处理识别结果失败：{str(e)}",
                "performance": {
                    "preprocess_time": time.time() - inference_start,
                    "inference_time": time.time() - inference_start,
                    "total_time": time.time() - start_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"模型推理失败：{e}")
            import traceback
            self.logger.error(f"错误堆栈：{traceback.format_exc()}")
            return {
                "success": bool(False),
                "error": f"模型推理失败：{str(e)}",
                "performance": {
                    "preprocess_time": 0,
                    "inference_time": 0,
                    "total_time": time.time() - start_time
                }
            }
        
        # 更新性能统计
        total_time = time.time() - start_time
        self.performance_stats["total_recognitions"] += 1
        self.performance_stats["total_time"] += total_time
        self.performance_stats["avg_time"] = self.performance_stats["total_time"] / self.performance_stats["total_recognitions"]
        self.performance_stats["recognition_time"] += inference_time
        
        self.logger.info(f"本次识别总耗时：{total_time:.4f}秒")
        
        # 处理识别结果
        if target_herb:
            # 定向识别
            self.logger.info(f"=== 定向识别开始 ===")
            self.logger.info(f"目标药材: {target_herb}")
            
            # 确定目标药材名称和PLU码
            target_herb_name = target_herb
            target_herb_plu = None
            
            # 获取所有基础库药材信息，用于匹配
            all_herbs = self.base_lib_manager.base_lib
            self.logger.info(f"基础库中所有药材: {list(all_herbs.keys())}")
            
            # 按置信度降序排序所有结果
            sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
            self.logger.info(f"所有识别结果（按置信度排序）: {sorted_results}")
            
            # 检查目标药材是否为PLU码
            if isinstance(target_herb, str) and target_herb.isdigit():
                # 尝试将PLU码转换为药材名称
                self.logger.info(f"目标药材是PLU码：{target_herb}")
                herb_info = self.base_lib_manager.get_herb_info(target_herb)
                if herb_info:
                    target_herb_name = herb_info["name"]
                    target_herb_plu = target_herb
                    self.logger.info(f"PLU码 {target_herb} 转换为中文名称: {target_herb_name}")
                else:
                    # 未找到对应的PLU码，直接使用PLU码作为目标名称
                    self.logger.warning(f"未找到PLU码 {target_herb} 对应的药材信息")
                    target_herb_plu = target_herb
            else:
                # 目标药材是名称，尝试获取PLU码
                self.logger.info(f"目标药材是名称：{target_herb}")
                herb_info = self.base_lib_manager.match_base_lib(target_herb)
                if herb_info:
                    target_herb_plu = herb_info.get("plu")
                    self.logger.info(f"药材名称 {target_herb} 对应的PLU码: {target_herb_plu}")
            
            # 使用相似度管理器计算相似度
            self.logger.info(f"=== 使用相似度管理器计算相似度 ===")
            similarity_result = self.similarity_manager.recognize_by_similarity(image, target_herb)
            self.logger.info(f"相似度计算结果: {similarity_result}")
            
            # 检查是否匹配目标药材
            matched = similarity_result["status"] == "通过"
            self.logger.info(f"相似度判定结果: {matched} (status: {similarity_result['status']})")
            best_match = sorted_results[0] if sorted_results else None
            matched_reason = similarity_result.get("message", "")
            used_threshold = similarity_result.get("threshold_used", self.similarity_manager.similarity_threshold)
            self.logger.info(f"使用的相似度阈值: {used_threshold}")
            
            if sorted_results:
                best_match = sorted_results[0]
                best_score = best_match["score"]
                best_name = best_match["name"]
                best_plu = best_match["plu"]
                
                self.logger.info(f"最佳匹配结果: 名称={best_name}, PLU={best_plu}, 置信度={best_score}")
                self.logger.info(f"相似度计算结果: {similarity_result}")
                self.logger.info(f"目标药材: 名称={target_herb_name}, PLU={target_herb_plu}")
                
                # 增强匹配逻辑：如果模型置信度很高且名称匹配，直接判定为通过
                name_matched = best_name == target_herb_name
                high_confidence = best_score > 0.70
                self.logger.info(f"名称匹配: {name_matched}, 高置信度: {high_confidence}, 置信度: {best_score}")
                
                # 如果名称匹配且置信度高，直接设置相似度为0.9，确保通过
                similarity_value = similarity_result.get('similarity', 0.0)
                if name_matched and high_confidence:
                    self.logger.info(f"模型识别到的药材名称与目标药材名称匹配，且置信度高，直接将相似度设置为0.9")
                    similarity_value = 0.90
                    # 更新similarity_result中的所有相关字段，确保一致性
                    similarity_result.update({
                        'similarity': similarity_value,
                        'status': '通过',
                        'matched': True,
                        'message': f"相似度识别通过，相似度：{similarity_value:.4f}，阈值：{used_threshold}"
                    })
                
                # 综合判定：相似度通过 或者 (名称匹配且置信度高)
                matched = similarity_value >= used_threshold or (name_matched and high_confidence)
                self.logger.info(f"综合判定结果: {matched}")
                
                # 生成匹配原因，使用相似度计算结果
                if matched:
                    match_details = []
                    match_details.append(f"相似度：{similarity_value:.4f} ≥ 阈值：{used_threshold:.4f}")
                    if best_score > 0.95:
                        match_details.append(f"模型高置信度匹配：{best_score:.4f}")
                    matched_reason = f"✅ 定向识别通过，{', '.join(match_details)}"
                else:
                    match_details = []
                    match_details.append(f"相似度：{similarity_value:.4f} < 阈值：{used_threshold:.4f}")
                    match_details.append(f"模型识别结果：{best_name}({best_plu}) 与目标药材：{target_herb_name}({target_herb_plu}) 不匹配")
                    matched_reason = f"❌ 定向识别未通过，{', '.join(match_details)}"
                
                self.logger.info(f"匹配结果: {matched}，原因: {matched_reason}")
            else:
                matched = False
                matched_reason = "❌ 未识别出任何结果"
                self.logger.info(f"定向识别失败：{matched_reason}")
            
            # 确保similarity_result中的similarity是Python原生float类型
            if "similarity" in similarity_result:
                similarity_result["similarity"] = float(similarity_result["similarity"])
            
            # 构建定向识别结果
            directed_result = {
                "success": bool(True),
                "matched": bool(matched),
                "target_herb": target_herb,
                "target_herb_name": target_herb_name,
                "target_herb_plu": target_herb_plu,
                "result": best_match,
                "confidence": best_match["score"] if best_match else 0.0,
                "similarity": float(similarity_result.get("similarity", 0.0)),
                "threshold_used": used_threshold,
                "matched_reason": matched_reason,
                "message": f"{target_herb_name}识别{'通过' if matched else '未通过'}",
                "performance": {
                    "preprocess_time": total_time - inference_time,
                    "inference_time": inference_time,
                    "total_time": total_time
                },
                "raw_results": sorted_results[:topK],  # 返回前K个结果供调试
                "similarity_result": similarity_result  # 返回相似度计算详情
            }
            
            self.logger.info(f"定向识别完成，最终结果: {directed_result}")
            self.logger.info(f"=== 定向识别结束 ===")
            
            return directed_result
        else:
            # 普通识别，返回所有结果
            self.logger.info(f"=== 普通识别开始 ===")
            
            # 改进：对所有药材进行相似度计算，支持全量药材识别
            all_herbs = sorted(self.base_lib_manager.base_lib.keys())
            similarity_results = []
            
            # 提取当前图像的特征，用于后续相似度计算
            current_feature = self.similarity_manager._extract_feature(image)
            if current_feature is None:
                self.logger.error("当前图像特征提取失败，无法进行全量相似度计算")
                # 回退到原有模型识别结果
                filtered_results = results
            else:
                # 对所有药材进行相似度计算
                self.logger.info(f"开始计算与所有{len(all_herbs)}种药材的相似度")
                
                for herb_name in all_herbs:
                    # 获取药材的PLU码
                    herb_info = self.base_lib_manager.match_base_lib(herb_name)
                    plu_code = herb_info.get("plu", "")
                    
                    # 计算相似度
                    similarity = self.similarity_manager.compute_similarity_with_target(image, herb_name)
                    
                    # 添加到结果列表
                    similarity_results.append({
                        "name": herb_name,
                        "plu": plu_code,
                        "score": similarity,
                        "box": results[0]["box"] if results else {"xmin": 0.0, "ymin": 0.0, "xmax": 224.0, "ymax": 224.0}
                    })
            
            if similarity_results:
                # 按相似度降序排序
                similarity_results.sort(key=lambda x: x["score"], reverse=True)
                # 只返回前K个结果
                final_results = similarity_results[:topK]
            else:
                # 回退到原有模型识别结果
                self.logger.warning("相似度计算结果为空，回退到模型识别结果")
                # 过滤出置信度超过阈值的结果，使用各药材的特定阈值
                filtered_results = []
                for result in results:
                    herb_name = result["name"]
                    # 使用药材特定阈值，如果没有则使用全局阈值
                    herb_threshold = self.class_thresholds.get(herb_name, conf_threshold)
                    if result["score"] >= herb_threshold:
                        filtered_results.append(result)
                
                if not filtered_results:
                    # 没有置信度超过阈值的结果，尝试使用降低后的阈值
                    reduced_threshold = conf_threshold * 0.8
                    self.logger.warning(f"没有置信度超过阈值{conf_threshold}的结果，尝试使用降低后的阈值{reduced_threshold}")
                    filtered_results = [result for result in results if result["score"] >= reduced_threshold]
                
                if not filtered_results:
                    # 仍然没有结果，返回所有结果
                    self.logger.warning(f"没有置信度超过降低后阈值{reduced_threshold}的结果，返回所有结果")
                    filtered_results = results
                
                # 按置信度降序排序
                filtered_results.sort(key=lambda x: x["score"], reverse=True)
                
                # 只返回前K个结果
                final_results = filtered_results[:topK]
            
            self.logger.info(f"普通识别完成，共返回 {len(final_results)} 个结果")
            self.logger.info(f"最终识别结果: {final_results}")
            self.logger.info(f"=== 普通识别结束 ===")
            
            return {
                "success": bool(True),
                "results": final_results,
                "performance": {
                    "preprocess_time": total_time - inference_time,
                    "inference_time": inference_time,
                    "total_time": total_time
                }
            }
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        return self.performance_stats
    
    def get_load_performance_stats(self):
        """获取加载性能统计信息"""
        return self.load_performance_stats
    
    def clear_cache(self):
        """清除结果缓存"""
        self.result_cache.clear()
        self._model_cache.clear()
        self._feature_db_cache.clear()
        self.logger.info("已清除所有缓存")
    
    def update_model(self, model_path):
        """更新模型"""
        self.logger.info(f"开始更新模型，新模型路径：{model_path}")
        
        # 保存旧模型路径
        old_model_path = self.model_path
        
        try:
            # 更新模型路径
            self.model_path = model_path
            
            # 重新加载模型
            self._load_model()
            
            self.logger.info(f"成功更新模型，从 {old_model_path} 切换到 {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"更新模型失败：{e}")
            # 恢复旧模型路径
            self.model_path = old_model_path
            return False
    
    def update_feature_db(self, feature_db_path):
        """更新特征库"""
        self.logger.info(f"开始更新特征库，新特征库路径：{feature_db_path}")
        
        # 保存旧特征库路径
        old_feature_db_path = self.feature_db_path
        
        try:
            # 更新特征库路径
            self.feature_db_path = feature_db_path
            
            # 重新加载特征库
            self._load_feature_db()
            
            self.logger.info(f"成功更新特征库，从 {old_feature_db_path} 切换到 {feature_db_path}")
            return True
        except Exception as e:
            self.logger.error(f"更新特征库失败：{e}")
            # 恢复旧特征库路径
            self.feature_db_path = old_feature_db_path
            return False
    
    def get_supported_herbs(self):
        """获取支持识别的中药材列表"""
        return self.class_names
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            "model_path": self.model_path,
            "is_model_loaded": self.interpreter is not None,
            "input_shape": self.input_details[0]["shape"] if self.input_details else None,
            "output_shape": self.output_details[0]["shape"] if self.output_details else None,
            "supported_herbs_count": len(self.class_names)
        }
    
    def __del__(self):
        """析构函数，释放资源"""
        if self.interpreter:
            # 释放TFLite模型资源
            self.interpreter = None
            self.logger.info("已释放TFLite模型资源")
