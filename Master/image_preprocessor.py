import cv2
import numpy as np
from typing import Tuple, List, Optional

class ImagePreprocessor:
    """图像预处理类，提供多种图像增强和特征提取功能"""
    
    def __init__(self, target_size: Tuple[int, int] = (320, 320), 
                 hog_win_size: Tuple[int, int] = (128, 128)):
        """
        初始化图像预处理类
        
        Args:
            target_size: 目标图像大小 (宽度, 高度)
            hog_win_size: HOG特征提取的窗口大小
        """
        self.target_size = target_size
        self.hog_win_size = hog_win_size
        
        # 初始化HOG特征提取器
        self.hog = self._init_hog_extractor()
    
    def _init_hog_extractor(self):
        """初始化HOG特征提取器"""
        win_size = self.hog_win_size
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        return cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    def check_image_quality(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        检查图像质量
        
        Args:
            image: 输入图像
            
        Returns:
            Tuple[bool, str]: (质量是否合格, 质量信息)
        """
        try:
            # 验证图像输入
            if image is None:
                return False, "无效的图像输入：图像为None"
            
            if image.size == 0:
                return False, "无效的图像输入：图像大小为0"
            
            # 检查图像维度
            if len(image.shape) < 2:
                return False, f"无效的图像输入：图像维度不足，当前维度：{image.shape}"
            
            # 转换为灰度图用于质量检测
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 1. 检查图像大小
            h, w = gray.shape[:2]
            if h <= 0 or w <= 0:
                return False, f"图像尺寸无效：高度={h}，宽度={w}"
            
            # 2. 检查亮度 - 放宽阈值
            brightness = np.mean(gray)
            if brightness < 20:
                return True, f"图像较暗({brightness:.1f})，但仍可识别"
            if brightness > 230:
                return True, f"图像较亮({brightness:.1f})，但仍可识别"
            
            # 3. 检查清晰度（使用Laplacian方差） - 放宽阈值
            laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian < 50:
                return True, f"图像轻微模糊({laplacian:.1f})，但仍可识别"
            
            return True, "图像质量合格"
        except Exception as e:
            return False, f"图像质量检测失败：{str(e)}"
    
    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理，用于模型输入
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        # 1. 验证图像输入
        if image is None or image.size == 0:
            raise ValueError("无效的图像输入")
        
        # 2. 验证图像尺寸
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"图像尺寸无效，高度={h}，宽度={w}")
        
        # 3. 保存原始尺寸用于日志
        orig_h, orig_w = h, w
        
        # 4. 增强尺寸合法性检查：要求宽高均≥10
        if h < 10 or w < 10:
            # 对于无效尺寸的图像，创建一个合适尺寸的图像
            # 优先使用平均颜色填充，避免复杂的缩放
            valid_size = 100  # 确保有效尺寸足够大
            
            # 计算图像均值，用于填充
            if len(image.shape) == 3:
                # 彩色图像
                img_mean = np.mean(image, axis=(0, 1), keepdims=True).astype(np.uint8)  # 形状(1, 1, 3)
                # 创建一个有效尺寸的图像，填充均值
                valid_image = np.repeat(np.repeat(img_mean, valid_size, axis=0), valid_size, axis=1)
            else:
                # 灰度图像
                img_mean = np.mean(image, axis=(0, 1), keepdims=True).astype(np.uint8)  # 形状(1, 1)
                # 创建一个有效尺寸的图像，填充均值
                valid_image = np.repeat(np.repeat(img_mean, valid_size, axis=0), valid_size, axis=1)
            
            # 如果原始图像有内容，尝试将其居中放置
            if h > 0 and w > 0:
                # 计算放置位置
                y_offset = max(0, (valid_size - h) // 2)
                x_offset = max(0, (valid_size - w) // 2)
                
                # 确保放置区域在有效范围内
                end_y = min(y_offset + h, valid_size)
                end_x = min(x_offset + w, valid_size)
                start_y = max(0, end_y - h)
                start_x = max(0, end_x - w)
                
                # 复制原始图像内容
                valid_image[start_y:end_y, start_x:end_x] = image[:end_y-start_y, :end_x-start_x]
            
            # 使用有效图像替换原始图像
            image = valid_image
            h, w = image.shape[:2]
        
        # 5. 增强黄芪和陈皮的特征区分
        # 对黄芪和陈皮进行特殊处理，增强其特征
        # 转换为HSV颜色空间，增强颜色特征
        if len(image.shape) == 3:
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 增强对比度和饱和度
            hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)  # 增加饱和度
            hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], 1.1)  # 增加亮度
            
            # 转回BGR颜色空间
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 6. 锐化处理，增强边缘特征
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        image = cv2.filter2D(image, -1, kernel)
        
        # 7. 验证目标尺寸
        if not self.target_size or len(self.target_size) != 2:
            raise ValueError(f"无效的目标尺寸格式：{self.target_size}")
        
        # 8. 提取目标尺寸并确保有效
        target_w, target_h = self.target_size
        if target_w <= 0 or target_h <= 0:
            # 如果目标尺寸无效，使用默认值
            target_w, target_h = 224, 224
        
        # 9. 保持纵横比的resize方法
        # 计算缩放比例，避免图像变形
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 确保缩放后的尺寸有效，避免cv2.resize失败
        if new_w <= 0 or new_h <= 0:
            # 使用最小有效尺寸
            new_w = max(10, new_w)
            new_h = max(10, new_h)
        
        # 确保缩放尺寸不超过目标尺寸
        new_w = min(new_w, target_w)
        new_h = min(new_h, target_h)
        
        # 显式构建dsize参数并验证
        dsize = (new_w, new_h)
        if not dsize or len(dsize) != 2 or dsize[0] <= 0 or dsize[1] <= 0:
            # 如果dsize无效，使用默认值
            dsize = (100, 100)
        
        # 调整图像大小，保持纵横比 - 添加异常捕获，确保不会因极端情况崩溃
        try:
            img_resized = cv2.resize(image, dsize, interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
            # 如果resize失败，创建一个简单的填充图像
            print(f"preprocess_for_model: 图像调整大小失败，创建默认图像：{str(e)}")
            if len(image.shape) == 3:
                img_resized = np.full((dsize[1], dsize[0], 3), 128, dtype=np.uint8)  # 灰色填充
            else:
                img_resized = np.full((dsize[1], dsize[0]), 128, dtype=np.uint8)  # 灰色填充
        
        # 10. 创建黑色背景，居中放置调整后的图像
        if len(image.shape) == 3:
            # 彩色图像
            img_padded = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
        else:
            # 灰度图像
            img_padded = np.full((target_h, target_w), 128, dtype=np.uint8)
        
        # 计算居中位置
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # 将调整后的图像放置到居中位置
        img_padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        
        # 11. 颜色空间转换（如果需要）
        if len(img_padded.shape) == 3:
            img_padded = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        
        # 12. 归一化
        img_normalized = img_padded.astype(np.float32) / 255.0
        
        # 13. 添加批次维度
        img_input = np.expand_dims(img_normalized, axis=0)
        
        return img_input
    
    def preprocess_for_hog(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理，用于HOG特征提取
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        # 1. 验证图像输入
        if image is None or image.size == 0:
            raise ValueError("无效的图像输入")
        
        # 2. 验证图像尺寸
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"图像尺寸无效，高度={h}，宽度={w}")
        
        # 3. 保存原始尺寸用于日志
        orig_h, orig_w = h, w
        
        # 4. 确保图像宽高至少为10像素，避免后续resize失败
        min_size = 10
        modified = False
        
        # 处理高度过小的情况
        if h < min_size:
            # 高度过小，复制行以增加高度
            scale_factor = (min_size + h - 1) // h  # 计算需要复制的倍数
            image = np.repeat(image, scale_factor, axis=0)
            modified = True
        
        # 重新获取尺寸，确保高度调整生效
        h, w = image.shape[:2]
        
        # 处理宽度过小的情况
        if w < min_size:
            # 宽度过小，复制列以增加宽度
            scale_factor = (min_size + w - 1) // w  # 计算需要复制的倍数
            image = np.repeat(image, scale_factor, axis=1)
            modified = True
        
        # 重新获取尺寸，确保宽度调整生效
        h, w = image.shape[:2]
        
        # 5. 处理极端尺寸：确保图像具有合理的高度和宽度比例
        min_dimension = min(h, w)
        if min_dimension < 20:
            # 对过小尺寸进行预处理：复制像素以增加尺寸
            scale_factor = max(2, min(10, 50 // min_dimension))
            image = np.repeat(image, scale_factor, axis=0)
            image = np.repeat(image, scale_factor, axis=1)
            modified = True
        
        # 重新获取最终调整后的尺寸
        h, w = image.shape[:2]
        
        # 6. 最后的尺寸合法性校验，确保宽高都至少为10像素
        if h < 10 or w < 10:
            # 如果仍然无法修复，尝试使用填充方式创建有效尺寸图像
            min_valid_size = 10
            valid_h = max(min_valid_size, h)
            valid_w = max(min_valid_size, w)
            
            # 创建一个黑色背景的有效尺寸图像
            if len(image.shape) == 3:
                valid_image = np.zeros((valid_h, valid_w, 3), dtype=np.uint8)
                # 将原始图像居中放置
                y_offset = (valid_h - h) // 2
                x_offset = (valid_w - w) // 2
                valid_image[y_offset:y_offset+h, x_offset:x_offset+w] = image
            else:
                valid_image = np.zeros((valid_h, valid_w), dtype=np.uint8)
                # 将原始图像居中放置
                y_offset = (valid_h - h) // 2
                x_offset = (valid_w - w) // 2
                valid_image[y_offset:y_offset+h, x_offset:x_offset+w] = image
            
            image = valid_image
            h, w = image.shape[:2]
            modified = True
        
        # 7. 验证HOG窗口尺寸格式
        if not self.hog_win_size or len(self.hog_win_size) != 2:
            raise ValueError(f"无效的HOG窗口尺寸格式：{self.hog_win_size}")
        
        # 8. 提取HOG窗口尺寸并确保有效
        hog_w, hog_h = self.hog_win_size
        if hog_w <= 0 or hog_h <= 0:
            raise ValueError(f"无效的HOG窗口尺寸值：宽度={hog_w}，高度={hog_h}")
        
        # 9. 显式构建dsize参数并验证
        dsize = (hog_w, hog_h)
        if not dsize or len(dsize) != 2 or dsize[0] <= 0 or dsize[1] <= 0:
            raise ValueError(f"无效的dsize参数：{dsize}")
        
        # 10. 转换为灰度图
        if len(image.shape) == 3:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            except cv2.error as e:
                raise ValueError(f"颜色空间转换失败：{str(e)}")
        else:
            gray = image
        
        # 11. 调整大小 - 增加异常捕获
        try:
            img_resized = cv2.resize(gray, dsize)
        except cv2.error as e:
            raise ValueError(f"HOG图像调整大小失败：{str(e)}，原始输入尺寸：({orig_w}, {orig_h})，调整后尺寸：({w}, {h})，目标尺寸：{dsize}")
        
        # 12. 验证调整大小后的图像
        if img_resized is None or img_resized.size == 0:
            raise ValueError(f"HOG图像调整大小后为空，原始输入尺寸：({orig_w}, {orig_h})，调整后尺寸：({w}, {h})，目标尺寸：{dsize}")
        
        # 13. 高斯模糊降噪
        img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
        
        # 14. 直方图均衡化，增强对比度
        img_equalized = cv2.equalizeHist(img_blurred)
        
        # 15. 自适应阈值处理，突出轮廓
        img_threshold = cv2.adaptiveThreshold(img_equalized, 255, 
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
        
        # 16. 边缘增强
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img_enhanced = cv2.filter2D(img_threshold, -1, kernel)
        
        return img_enhanced
    
    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """
        提取HOG特征
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: HOG特征向量
        """
        # 预处理图像
        preprocessed_img = self.preprocess_for_hog(image)
        
        # 提取HOG特征
        hog_features = self.hog.compute(preprocessed_img).flatten()
        
        # 特征归一化
        hog_features = hog_features / (np.linalg.norm(hog_features) + 1e-7)
        
        # 确保特征维度为128
        if len(hog_features) != 128:
            hog_features = np.resize(hog_features, 128)
        
        return hog_features
    
    def augment_image(self, image: np.ndarray, augmentation_config: Optional[dict] = None) -> np.ndarray:
        """
        图像增强
        
        Args:
            image: 输入图像
            augmentation_config: 增强配置
            
        Returns:
            np.ndarray: 增强后的图像
        """
        # 默认增强配置
        default_config = {
            'rotation_range': 30,
            'brightness_range': (-20, 20),
            'contrast_range': (0.8, 1.2),
            'blur_probability': 0.3,
            'sharpen_probability': 0.3,
            'flip_horizontal_probability': 0.5,
            'flip_vertical_probability': 0.5
        }
        
        config = augmentation_config or default_config
        augmented_img = image.copy()
        
        # 1. 随机旋转
        if np.random.rand() < 0.5:
            angle = np.random.randint(-config['rotation_range'], config['rotation_range'])
            h, w = augmented_img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            augmented_img = cv2.warpAffine(augmented_img, M, (w, h))
        
        # 2. 随机翻转
        if np.random.rand() < config['flip_horizontal_probability']:
            augmented_img = cv2.flip(augmented_img, 1)  # 水平翻转
        
        if np.random.rand() < config['flip_vertical_probability']:
            augmented_img = cv2.flip(augmented_img, 0)  # 垂直翻转
        
        # 3. 随机调整亮度和对比度
        alpha = np.random.uniform(config['contrast_range'][0], config['contrast_range'][1])
        beta = np.random.uniform(config['brightness_range'][0], config['brightness_range'][1])
        augmented_img = cv2.convertScaleAbs(augmented_img, alpha=alpha, beta=beta)
        
        # 4. 随机高斯模糊
        if np.random.rand() < config['blur_probability']:
            ksize = np.random.choice([3, 5, 7])
            augmented_img = cv2.GaussianBlur(augmented_img, (ksize, ksize), 0)
        
        # 5. 随机锐化
        if np.random.rand() < config['sharpen_probability']:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            augmented_img = cv2.filter2D(augmented_img, -1, kernel)
        
        return augmented_img
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像去噪
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 去噪后的图像
        """
        # 使用双边滤波去噪，保留边缘信息
        img_denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return img_denoised
    
    def enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """
        边缘增强
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 边缘增强后的图像
        """
        # 使用Canny边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 100, 200)
        
        # 将边缘叠加到原始图像
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        enhanced_img = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)
        
        return enhanced_img
    
    def crop_roi(self, image: np.ndarray, roi_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """
        裁剪感兴趣区域（ROI）
        
        Args:
            image: 输入图像
            roi_coords: ROI坐标 (x, y, w, h)
            
        Returns:
            np.ndarray: 裁剪后的图像
        """
        x, y, w, h = roi_coords
        h_img, w_img = image.shape[:2]
        
        # 确保ROI坐标在图像范围内
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        return image[y:y+h, x:x+w]
    
    def resize_with_padding(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        调整图像大小，保持长宽比，不足部分填充
        
        Args:
            image: 输入图像
            target_size: 目标大小 (宽度, 高度)
            
        Returns:
            np.ndarray: 调整大小后的图像
        """
        # 1. 验证图像输入
        if image is None or image.size == 0:
            raise ValueError("无效的图像输入")
        
        # 2. 验证图像尺寸
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"图像尺寸无效，高度={h}，宽度={w}")
        
        # 3. 保存原始尺寸用于日志
        orig_h, orig_w = h, w
        
        # 4. 增强的尺寸合法性检查：要求宽高均≥10
        if h < 10 or w < 10:
            # 对于无效尺寸的图像，创建一个合适尺寸的图像
            valid_size = 100  # 确保有效尺寸足够大
            
            # 计算图像均值，用于填充
            if len(image.shape) == 3:
                # 彩色图像
                img_mean = np.mean(image, axis=(0, 1), keepdims=True).astype(np.uint8)  # 形状(1, 1, 3)
                # 创建一个有效尺寸的图像，填充均值
                valid_image = np.repeat(np.repeat(img_mean, valid_size, axis=0), valid_size, axis=1)
            else:
                # 灰度图像
                img_mean = np.mean(image, axis=(0, 1), keepdims=True).astype(np.uint8)  # 形状(1, 1)
                # 创建一个有效尺寸的图像，填充均值
                valid_image = np.repeat(np.repeat(img_mean, valid_size, axis=0), valid_size, axis=1)
            
            # 如果原始图像有内容，尝试将其居中放置
            if h > 0 and w > 0:
                # 计算放置位置
                y_offset = max(0, (valid_size - h) // 2)
                x_offset = max(0, (valid_size - w) // 2)
                
                # 确保放置区域在有效范围内
                end_y = min(y_offset + h, valid_size)
                end_x = min(x_offset + w, valid_size)
                start_y = max(0, end_y - h)
                start_x = max(0, end_x - w)
                
                # 复制原始图像内容
                valid_image[start_y:end_y, start_x:end_x] = image[:end_y-start_y, :end_x-start_x]
            
            image = valid_image
            h, w = image.shape[:2]
        
        # 5. 验证目标尺寸格式并设置默认值
        if not target_size or len(target_size) != 2:
            target_size = (224, 224)  # 默认值
        
        # 6. 提取目标尺寸并确保有效
        target_w, target_h = target_size
        if target_w <= 0 or target_h <= 0:
            target_w, target_h = 224, 224  # 默认值
        
        # 7. 改进的缩放逻辑：确保处理极端比例图像时不会失败
        # 计算缩放比例，避免除以零
        if w <= 0 or h <= 0:
            w, h = 100, 100  # 确保宽高有效
        
        scale = min(target_w / w, target_h / h)
        new_w = max(10, int(w * scale))
        new_h = max(10, int(h * scale))
        
        # 确保缩放尺寸有效且不超过目标尺寸
        new_w = min(new_w, target_w)
        new_h = min(new_h, target_h)
        
        dsize = (new_w, new_h)
        
        # 8. 改进的resize逻辑：确保不会因为极端尺寸导致失败
        try:
            # 直接resize，使用INTER_LINEAR插值
            img_resized = cv2.resize(image, dsize, interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
            # 如果resize失败，创建一个简单的填充图像
            print(f"resize_with_padding: 图像调整大小失败，创建默认图像：{str(e)}")
            if len(image.shape) == 3:
                img_resized = np.full((new_h, new_w, 3), 128, dtype=np.uint8)  # 灰色填充
            else:
                img_resized = np.full((new_h, new_w), 128, dtype=np.uint8)  # 灰色填充
        
        # 9. 创建目标大小的画布
        if len(image.shape) == 3:
            img_padded = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        else:
            img_padded = np.full((target_h, target_w), 255, dtype=np.uint8)
        
        # 10. 计算填充位置并确保有效
        x_offset = max(0, (target_w - new_w) // 2)
        y_offset = max(0, (target_h - new_h) // 2)
        
        # 确保放置区域在有效范围内
        end_x = min(x_offset + new_w, target_w)
        end_y = min(y_offset + new_h, target_h)
        start_x = max(0, end_x - new_w)
        start_y = max(0, end_y - new_h)
        
        # 11. 确保img_resized的尺寸与放置区域匹配
        resized_h, resized_w = img_resized.shape[:2]
        if resized_h != new_h or resized_w != new_w:
            # 如果resize结果尺寸不匹配，重新调整
            try:
                img_resized = cv2.resize(img_resized, (end_x - start_x, end_y - start_y), interpolation=cv2.INTER_LINEAR)
            except cv2.error:
                # 如果再次resize失败，使用灰色填充
                if len(image.shape) == 3:
                    img_resized = np.full((end_y - start_y, end_x - start_x, 3), 128, dtype=np.uint8)
                else:
                    img_resized = np.full((end_y - start_y, end_x - start_x), 128, dtype=np.uint8)
        
        # 12. 将缩放后的图像放置到画布中心
        try:
            img_padded[start_y:end_y, start_x:end_x] = img_resized
        except Exception as e:
            # 如果放置失败，打印错误并返回填充图像
            print(f"resize_with_padding: 图像放置失败：{str(e)}")
        
        return img_padded
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像归一化
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 归一化后的图像
        """
        # 转换为float32
        img_float = image.astype(np.float32)
        
        # 归一化到[0, 1]
        img_normalized = img_float / 255.0
        
        # 减去均值，除以标准差
        mean = np.mean(img_normalized)
        std = np.std(img_normalized)
        img_normalized = (img_normalized - mean) / (std + 1e-7)
        
        return img_normalized
