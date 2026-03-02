#!/usr/bin/env python3
"""
相似度计算管理器
负责处理图像相似度计算
"""

import cv2
import numpy as np
from image_preprocessor import ImagePreprocessor
from base_lib_manager import BaseLibManager
from logger import Logger
import os
import pickle
from learning_engine.utils.db_handler import FeatureDBHandler

class SimilarityManager:
    """相似度计算管理器"""
    
    def __init__(self, base_lib_manager=None):
        """初始化相似度计算管理器"""
        self.logger = Logger("SimilarityManager")
        self.base_lib_manager = base_lib_manager or BaseLibManager()
        
        # 初始化图像预处理类
        self.preprocessor = ImagePreprocessor(target_size=(224, 224))
        
        # 默认相似度阈值，统一设置为0.75
        self.similarity_threshold = 0.75
        
        # 初始化特征提取器
        self.feature_extractor = self._init_feature_extractor()
        
        # 初始化特征数据库，用于获取学习到的特征
        self.feature_db = FeatureDBHandler()
        
        # 加载目标药材图像库
        self.target_images = self._load_target_images()
        
        self.logger.info("相似度计算管理器初始化完成")
    
    def _init_feature_extractor(self):
        """
        初始化特征提取器
        使用颜色直方图特征，不需要HOG特征提取器
        """
        # 颜色直方图特征不需要HOG特征提取器，直接返回None
        self.logger.info("使用颜色直方图特征，无需HOG特征提取器")
        return None
    
    def _load_target_images(self):
        """加载目标药材图像库"""
        target_images = {}
        
        # 检查目标图像目录
        target_image_dir = "target_images"
        if not os.path.exists(target_image_dir):
            self.logger.warning(f"目标图像目录不存在：{target_image_dir}")
            return target_images
        
        # 直接指定要加载的药材图像，避免文件名编码问题
        herb_files = {
            "黄芪": "黄芪.jpg",
            "陈皮": "陈皮.jpg"
        }
        
        # 加载指定的药材图像
        for herb_name, file_name in herb_files.items():
            file_path = os.path.join(target_image_dir, file_name)
            
            # 加载图像
            try:
                self.logger.info(f"尝试加载目标药材图像：{herb_name}，文件路径：{file_path}")
                
                if os.path.exists(file_path):
                    self.logger.info(f"文件存在：{file_path}")
                    
                    # 使用cv2.imdecode和numpy.fromfile来处理中文文件名
                    with open(file_path, 'rb') as f:
                        img_data = np.fromfile(f, dtype=np.uint8)
                        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    
                    self.logger.info(f"cv2.imdecode结果：{image is not None}")
                    
                    if image is not None:
                        self.logger.info(f"图像加载成功，尺寸：{image.shape}")
                        # 预处理图像
                        processed_image = self.preprocessor.preprocess_for_model(image)
                        # 提取特征
                        feature = self._extract_feature(image)
                        target_images[herb_name] = {
                            "image": image,
                            "processed_image": processed_image,
                            "feature": feature,
                            "file_path": file_path
                        }
                        self.logger.info(f"✅ 成功加载目标药材图像：{herb_name}，路径：{file_path}")
                    else:
                        self.logger.warning(f"❌ 图像加载失败（cv2.imdecode返回None）：{file_path}")
                else:
                    self.logger.warning(f"❌ 文件不存在：{file_path}")
            except Exception as e:
                self.logger.error(f"❌ 加载目标药材图像失败 {file_name}：{e}", exc_info=True)
        
        self.logger.info(f"共加载 {len(target_images)} 种目标药材图像")
        return target_images
    
    def _extract_feature(self, image):
        """
        提取图像特征
        使用颜色直方图特征，对药材的颜色特征更敏感
        """
        try:
            # 图像预处理
            resized = cv2.resize(image, (224, 224))  # 统一尺寸
            
            # 转换到HSV色彩空间，更适合颜色分析
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            
            # 计算颜色直方图
            hist_h = cv2.calcHist([hsv], [0], None, [64], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [64], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [64], [0, 256])
            
            # 归一化直方图
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()
            
            # 合并三个通道的直方图特征
            feature = np.concatenate([hist_h, hist_s, hist_v])
            
            # 特征归一化
            feature = feature / (np.linalg.norm(feature) + 1e-7)
            
            self.logger.info(f"颜色直方图特征提取成功，特征维度：{feature.shape[0]}")
            return feature
        except Exception as e:
            self.logger.error(f"提取图像特征失败：{e}")
            import traceback
            self.logger.error(f"错误堆栈：{traceback.format_exc()}")
            return None
    
    def compute_similarity(self, image1=None, image2=None, feature1=None, feature2=None):
        """
        计算相似度
        采用改进的余弦相似度结合欧氏距离的混合计算方法
        支持两种模式：
        1. 两张图像比较：image1和image2都提供
        2. 两个特征比较：feature1和feature2都提供
        
        Args:
            image1: 第一张图像（可选）
            image2: 第二张图像（可选）
            feature1: 第一个特征（可选）
            feature2: 第二个特征（可选）
            
        Returns:
            float: 相似度值，范围0-1，精确到小数点后两位
        """
        try:
            # 特征比较模式
            if feature1 is not None and feature2 is not None:
                # 确保特征是numpy数组
                if isinstance(feature1, list):
                    feature1 = np.array(feature1, dtype=np.float32)
                if isinstance(feature2, list):
                    feature2 = np.array(feature2, dtype=np.float32)
                
                # 处理特征维度，确保是一维向量
                if len(feature1.shape) > 1:
                    feature1 = feature1.flatten()
                if len(feature2.shape) > 1:
                    feature2 = feature2.flatten()
                
                # 处理特征维度不匹配
                if feature1.shape != feature2.shape:
                    self.logger.error(f"特征维度不匹配：{feature1.shape} vs {feature2.shape}")
                    # 调整特征维度，取较小的维度
                    min_dim = min(feature1.shape[0], feature2.shape[0])
                    feature1 = feature1[:min_dim]
                    feature2 = feature2[:min_dim]
                
                # 确保特征数据类型正确
                feature1 = feature1.astype(np.float32)
                feature2 = feature2.astype(np.float32)
                
                # 归一化特征向量
                norm1 = np.linalg.norm(feature1) + 1e-7
                norm2 = np.linalg.norm(feature2) + 1e-7
                feature1_norm = feature1 / norm1
                feature2_norm = feature2 / norm2
                
                # 计算余弦相似度
                cosine_similarity = np.dot(feature1_norm, feature2_norm)
                
                # 计算欧氏距离
                euclidean_distance = np.linalg.norm(feature1_norm - feature2_norm)
                
                # 计算欧氏相似度（归一化到0-1范围）
                euclidean_similarity = 1.0 / (1.0 + euclidean_distance)
                
                # 混合相似度计算：余弦相似度权重70%，欧氏相似度权重30%
                mixed_similarity = 0.7 * cosine_similarity + 0.3 * euclidean_similarity
                
                # 确保相似度在0-1范围内
                mixed_similarity = max(0.0, min(1.0, mixed_similarity))
                
                # 精确到小数点后两位
                mixed_similarity = round(float(mixed_similarity), 2)
                
                self.logger.info(f"混合相似度计算结果：余弦相似度={cosine_similarity:.4f}, 欧氏相似度={euclidean_similarity:.4f}, 混合相似度={mixed_similarity:.4f}")
                return mixed_similarity
            # 图像比较模式
            elif image1 is not None and image2 is not None:
                # 提取两张图像的特征
                feature1 = self._extract_feature(image1)
                feature2 = self._extract_feature(image2)
                
                if feature1 is None or feature2 is None:
                    self.logger.error("图像特征提取失败")
                    raise ValueError("图像特征提取失败")
                
                # 递归调用特征比较模式
                return self.compute_similarity(feature1=feature1, feature2=feature2)
            else:
                self.logger.error("计算相似度失败：必须提供两张图像或两个特征")
                raise ValueError("计算相似度失败：必须提供两张图像或两个特征")
        except ValueError as e:
            self.logger.error(f"计算相似度失败：{e}")
            # 当发生特征提取失败等错误时，返回0.00作为默认值
            return 0.00
        except Exception as e:
            self.logger.error(f"计算相似度失败：{e}")
            import traceback
            self.logger.error(f"错误堆栈：{traceback.format_exc()}")
            # 其他异常情况，返回0.00作为默认值
            return 0.00
    
    def update_target_herb_image(self, herb_name, image):
        """
        更新目标药材图像
        
        Args:
            herb_name: 药材名称
            image: 药材图像
            
        Returns:
            bool: 是否更新成功
        """
        try:
            # 预处理图像
            processed_image = self.preprocessor.preprocess_for_model(image)
            # 提取特征
            feature = self._extract_feature(image)
            
            # 更新目标图像库
            self.target_images[herb_name] = {
                "image": image,
                "processed_image": processed_image,
                "feature": feature,
                "file_path": f"target_images/{herb_name}.jpg"
            }
            
            self.logger.info(f"成功更新目标药材图像：{herb_name}")
            return True
        except Exception as e:
            self.logger.error(f"更新目标药材图像失败：{e}")
            return False
    
    def compute_similarity_with_target(self, image, target_herb):
        """
        计算当前图像与目标药材的相似度
        综合考虑预加载图像和学习到的特征，优化相似度计算逻辑
        
        Args:
            image: 当前图像
            target_herb: 目标药材名称或PLU码
            
        Returns:
            float: 相似度值，范围0-1
        """
        try:
            # 确定目标药材名称和PLU码
            target_herb_name = target_herb
            target_herb_plu = None
            
            # 检查是否为PLU码
            if isinstance(target_herb, str) and target_herb.isdigit():
                # 从PLU码获取药材名称
                self.logger.info(f"目标药材是PLU码：{target_herb}")
                herb_info = self.base_lib_manager.get_herb_info(target_herb)
                if herb_info:
                    target_herb_name = herb_info["name"]
                    target_herb_plu = target_herb
                    self.logger.info(f"PLU码 {target_herb} 转换为中文名称: {target_herb_name}")
                else:
                    self.logger.warning(f"未找到PLU码 {target_herb} 对应的药材信息")
                    target_herb_plu = target_herb
                    target_herb_name = f"未知药材_{target_herb}"
            else:
                # 从药材名称获取PLU码
                self.logger.info(f"目标药材是名称：{target_herb}")
                herb_info = self.base_lib_manager.match_base_lib(target_herb)
                if herb_info:
                    target_herb_plu = herb_info.get("plu")
                    self.logger.info(f"药材名称 {target_herb} 对应的PLU码: {target_herb_plu}")
                else:
                    self.logger.warning(f"未找到药材名称 {target_herb} 对应的信息")
                    # 尝试直接使用名称作为PLU码
                    target_herb_plu = target_herb
            
            self.logger.info(f"计算相似度：当前图像 vs 目标药材 {target_herb_name} (PLU: {target_herb_plu})")
            
            learned_similarities = []
            
            # 2. 与学习到的特征比较
            # 获取学习到的目标药材特征
            learned_features = self.feature_db.get_features_by_plu(target_herb_plu)
            if learned_features and len(learned_features) > 0:
                # 提取当前图像的特征
                current_feature = self._extract_feature(image)
                if current_feature is not None:
                    # 计算当前图像特征与每个学习到的特征的相似度
                    for i, learned_feature in enumerate(learned_features):
                        # 确保learned_feature是numpy数组
                        if isinstance(learned_feature, list):
                            learned_feature_np = np.array(learned_feature)
                        else:
                            learned_feature_np = learned_feature
                        
                        # 直接调用compute_similarity方法，让它来处理特征维度不匹配的情况
                        feature_similarity = self.compute_similarity(feature1=current_feature, feature2=learned_feature_np)
                        learned_similarities.append(feature_similarity)
                        self.logger.info(f"与第{i+1}个学习到的特征的相似度：{feature_similarity:.4f}")
                    
                    # 如果有学习到的特征，提高权重
                    if learned_similarities:
                        self.logger.info(f"学习到的特征数量：{len(learned_similarities)}")
            
            # 3. 计算最终相似度，优化特征融合策略
            all_similarities = learned_similarities
            final_similarity = 0.0
            
            # 添加调试信息
            self.logger.info(f"学习到的相似度列表：{learned_similarities}")
            self.logger.info(f"所有相似度列表：{all_similarities}")
            
            if all_similarities and len(all_similarities) > 0:
                # 计算各种相似度指标
                max_similarity = max(all_similarities)
                avg_similarity = sum(all_similarities) / len(all_similarities)
                median_similarity = np.median(all_similarities)
                
                # 如果有学习到的特征，使用更优的加权融合策略
                if learned_similarities:
                    # 计算学习特征的各项指标
                    max_learned_similarity = max(learned_similarities)
                    avg_learned_similarity = sum(learned_similarities) / len(learned_similarities)
                    median_learned_similarity = np.median(learned_similarities)
                    
                    # 优化加权策略：
                    # - 只使用学习特征，不使用预加载图像
                    final_similarity = 0.5 * max_learned_similarity + 0.3 * avg_learned_similarity + 0.2 * median_learned_similarity
                
                # 确保相似度在0-1范围内
                final_similarity = max(0.0, min(1.0, final_similarity))
            else:
                # 没有找到任何可比较的特征或图像，返回0.00相似度，符合首次识别机制
                self.logger.warning(f"未找到可比较的特征或图像：{target_herb_name}")
                # 首次识别，强制返回0.00相似度，确保准确性
                final_similarity = 0.00
                self.logger.info(f"首次识别，设置默认相似度：{final_similarity}")
            
            # 将numpy float64转换为Python原生float，以便JSON序列化
            final_similarity = float(final_similarity)
            # 确保相似度精确到小数点后两位
            final_similarity = round(final_similarity, 2)
            self.logger.info(f"最终相似度：{final_similarity:.4f}")
            return final_similarity
        except ValueError as e:
            self.logger.error(f"计算与目标药材相似度失败：{e}")
            # 当发生特征提取失败等错误时，直接抛出异常
            raise e
        except Exception as e:
            self.logger.error(f"计算与目标药材相似度失败：{e}")
            # 其他异常情况，返回0.50作为默认值
            return 0.50
    
    def recognize_by_similarity(self, image, target_herb, threshold=None):
        """
        根据相似度进行识别
        
        Args:
            image: 当前图像
            target_herb: 目标药材名称或PLU码
            threshold: 相似度阈值，默认使用self.similarity_threshold
            
        Returns:
            dict: 识别结果，包含similarity和status字段
        """
        try:
            # 使用指定阈值或默认阈值
            used_threshold = threshold or self.similarity_threshold
            
            # 计算相似度
            similarity = self.compute_similarity_with_target(image, target_herb)
            
            # 确定识别结果
            status = "通过" if similarity >= used_threshold else "不通过"
            matched = status == "通过"
            
            self.logger.info(f"相似度识别结果：{status} (相似度：{similarity:.4f}，阈值：{used_threshold})，匹配结果：{matched}")
            
            return {
                "success": bool(True),
                "similarity": similarity,
                "threshold_used": used_threshold,
                "status": status,
                "matched": bool(matched),
                "target_herb": target_herb,
                "message": f"相似度识别{status}，相似度：{similarity:.4f}，阈值：{used_threshold}"
            }
        except ValueError as e:
            self.logger.error(f"相似度识别失败：{e}")
            # 当发生特征提取失败等错误时，返回明确的错误信息
            return {
                "success": False,
                "error": str(e),
                "status": "不通过"
            }
        except Exception as e:
            self.logger.error(f"相似度识别失败：{e}")
            return {
                "success": False,
                "error": str(e),
                "status": "不通过"
            }
    
    def set_similarity_threshold(self, threshold):
        """
        设置相似度阈值
        
        Args:
            threshold: 相似度阈值，范围0-1
        """
        if 0 <= threshold <= 1:
            self.similarity_threshold = threshold
            self.logger.info(f"相似度阈值已设置为：{threshold}")
            return True
        else:
            self.logger.error(f"无效的相似度阈值：{threshold}，必须在0-1之间")
            return False
    
    def add_target_image(self, herb_name, image):
        """
        添加目标药材图像
        
        Args:
            herb_name: 药材名称
            image: 药材图像
            
        Returns:
            bool: 是否添加成功
        """
        try:
            # 预处理图像
            processed_image = self.preprocessor.preprocess_for_model(image)
            # 提取特征
            feature = self._extract_feature(image)
            
            # 添加到目标图像库
            self.target_images[herb_name] = {
                "image": image,
                "processed_image": processed_image,
                "feature": feature
            }
            
            self.logger.info(f"成功添加目标药材图像：{herb_name}")
            return True
        except Exception as e:
            self.logger.error(f"添加目标药材图像失败：{e}")
            return False
    
    def get_target_herbs(self):
        """
        获取所有目标药材名称
        
        Returns:
            list: 目标药材名称列表
        """
        return list(self.target_images.keys())
    
    def save_target_images(self):
        """
        保存目标图像库
        """
        try:
            # 只保存必要的数据
            save_data = {}
            for herb_name, image_data in self.target_images.items():
                save_data[herb_name] = {
                    "feature": image_data["feature"],
                    "file_path": image_data.get("file_path", "")
                }
            
            # 保存到文件
            with open("data/target_images.pkl", "wb") as f:
                pickle.dump(save_data, f)
            
            self.logger.info("目标图像库已保存")
            return True
        except Exception as e:
            self.logger.error(f"保存目标图像库失败：{e}")
            return False
    
    def load_target_images_from_file(self):
        """
        从文件加载目标图像库
        """
        try:
            # 加载保存的特征数据
            with open("data/target_images.pkl", "rb") as f:
                saved_data = pickle.load(f)
            
            # 重新构建目标图像库
            for herb_name, data in saved_data.items():
                # 尝试重新加载图像
                if "file_path" in data and os.path.exists(data["file_path"]):
                    # 使用安全的图像读取方式，支持中文路径
                    try:
                        with open(data["file_path"], 'rb') as f:
                            img_data = np.fromfile(f, dtype=np.uint8)
                            image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                        if image is not None:
                            processed_image = self.preprocessor.preprocess_for_model(image)
                            self.target_images[herb_name] = {
                                "image": image,
                                "processed_image": processed_image,
                                "feature": data["feature"],
                                "file_path": data["file_path"]
                            }
                    except Exception as e:
                        self.logger.error(f"安全读取图像失败 {data['file_path']}：{e}")
            
            self.logger.info(f"从文件加载了 {len(self.target_images)} 种目标药材图像")
            return True
        except Exception as e:
            self.logger.error(f"从文件加载目标图像库失败：{e}")
            return False

def quick_test():
    """快速测试函数，用于验证修复效果"""
    import cv2
    import numpy as np
    
    # 创建测试图像
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.rectangle(image, (56, 56), (168, 168), (0, 255, 255), -1)
    
    # 初始化相似度管理器
    manager = SimilarityManager()
    
    # 测试黄芪相似度
    print("=== 快速测试 ===")
    print("测试黄芪相似度...")
    similarity = manager.compute_similarity_with_target(image, "黄芪")
    print(f"黄芪相似度: {similarity:.4f}")
    
    # 测试PLU码9997相似度
    print("\n测试PLU码9997相似度...")
    similarity_plu = manager.compute_similarity_with_target(image, "9997")
    print(f"PLU码9997相似度: {similarity_plu:.4f}")
    
    # 验证结果
    if similarity > 0.0 and similarity_plu > 0.0:
        print("\n✅ 修复成功：所有相似度值都大于0.0")
    else:
        print("\n❌ 修复失败：存在相似度值为0.0的情况")
    
    return similarity, similarity_plu

if __name__ == "__main__":
    # 运行快速测试
    quick_test()
