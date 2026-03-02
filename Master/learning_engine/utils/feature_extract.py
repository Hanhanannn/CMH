import cv2
import numpy as np

class FeatureExtractor:
    """中药材图像特征提取"""
    @staticmethod
    def extract(image_path):
        """
        从原始图像提取特征向量
        :param image_path: 图像路径
        :return: 一维numpy数组（特征向量）
        """
        # 1. 图像预处理
        # 使用cv2.imdecode和numpy.fromfile来处理中文文件名
        import numpy as np
        with open(image_path, 'rb') as f:
            img_data = np.fromfile(f, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError(f"无法读取图像：{image_path}")
        
        # 统一尺寸
        img = cv2.resize(img, (224, 224))
        
        # 转换到HSV色彩空间，更适合颜色分析
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 2. 特征提取：计算颜色直方图
        hist_h = cv2.calcHist([hsv], [0], None, [64], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [64], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [64], [0, 256])
        
        # 归一化直方图
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # 合并三个通道的直方图特征
        feature = np.concatenate([hist_h, hist_s, hist_v])
        
        # 3. 特征归一化
        feature = feature / (np.linalg.norm(feature) + 1e-7)
        
        return feature
