#!/usr/bin/env python3
"""
数据标准化与整理脚本
对633种药材进行全面标准化处理，建立统一的数据标准规范
"""

import os
import json
import pickle
import hashlib
from datetime import datetime

class HerbDataStandardizer:
    """药材数据标准化器"""
    
    def __init__(self):
        """初始化数据标准化器"""
        self.base_lib_dir = "base_lib"
        self.data_dir = "data"
        self.standardized_data = {}
        self.plu_map = {}
        self.features_dim = 512  # 统一特征向量维度
        
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        print("=== 药材数据标准化器初始化 ===")
    
    def generate_plu_code(self, herb_name, index):
        """
        生成10位数字PLU编码
        规则：前2位为大类，中间4位为亚类，后4位为具体品种
        """
        # 简单实现：前2位固定为"01"（中药材大类）
        # 中间4位：使用药材名称的拼音首字母ASCII码转换
        # 后4位：使用索引的4位补零
        
        # 前2位：大类代码
        category = "01"
        
        # 中间4位：亚类代码（使用名称哈希）
        name_hash = hashlib.md5(herb_name.encode('utf-8')).hexdigest()[:4]
        subclass = str(int(name_hash, 16) % 10000).zfill(4)
        
        # 后4位：具体品种代码
        variety = str(index).zfill(4)
        
        plu_code = f"{category}{subclass}{variety}"
        return plu_code
    
    def load_existing_herbs(self):
        """加载现有药材数据"""
        print(f"=== 加载现有药材数据 ===")
        
        # 检查base_lib目录是否存在
        if not os.path.exists(self.base_lib_dir):
            print(f"❌ base_lib目录不存在：{self.base_lib_dir}")
            return
        
        # 加载所有.pkl文件
        herb_files = [f for f in os.listdir(self.base_lib_dir) if f.endswith('.pkl')]
        print(f"发现 {len(herb_files)} 种药材数据文件")
        
        # 按文件名排序（拼音顺序）
        herb_files.sort()
        
        # 加载每种药材数据
        for idx, herb_file in enumerate(herb_files):
            herb_name = os.path.splitext(herb_file)[0]
            herb_path = os.path.join(self.base_lib_dir, herb_file)
            
            try:
                with open(herb_path, 'rb') as f:
                    herb_data = pickle.load(f)
                
                # 生成PLU码
                plu_code = self.generate_plu_code(herb_name, idx + 1)
                
                # 标准化药材数据
                standardized_herb = {
                    "name": herb_name,
                    "plu": plu_code,
                    "index": idx + 1,
                    "original_data": herb_data,
                    "standardized_data": {
                        "chinese_name": herb_name,
                        "latin_name": herb_data.get("latin_name", ""),
                        "morphological_features": herb_data.get("morphological_features", ""),
                        "medicinal_parts": herb_data.get("medicinal_parts", ""),
                        "origin": herb_data.get("origin", ""),
                        "feature_vector_dim": self.features_dim,
                        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "version": "1.0.0"
                    }
                }
                
                self.standardized_data[herb_name] = standardized_herb
                self.plu_map[plu_code] = herb_name
                
                print(f"✅ 标准化药材：{herb_name} -> PLU: {plu_code}")
                
            except Exception as e:
                print(f"❌ 加载药材数据失败：{herb_name}，错误：{str(e)}")
    
    def generate_plu_mapping(self):
        """生成PLU映射文件"""
        print("\n=== 生成PLU映射文件 ===")
        
        plu_mapping = {
            "version": "1.0.0",
            "generated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_herbs": len(self.plu_map),
            "plu_to_name": self.plu_map,
            "name_to_plu": {v: k for k, v in self.plu_map.items()}
        }
        
        # 保存PLU映射
        plu_map_path = os.path.join(self.data_dir, "plu_mapping.json")
        with open(plu_map_path, 'w', encoding='utf-8') as f:
            json.dump(plu_mapping, f, ensure_ascii=False, indent=2)
        
        print(f"✅ PLU映射文件生成成功：{plu_map_path}")
        print(f"   包含 {len(self.plu_map)} 种药材的PLU编码映射")
    
    def save_standardized_data(self):
        """保存标准化后的数据"""
        print("\n=== 保存标准化后的数据 ===")
        
        # 保存标准化药材数据
        standardized_data_path = os.path.join(self.data_dir, "standardized_herbs.pkl")
        with open(standardized_data_path, 'wb') as f:
            pickle.dump(self.standardized_data, f)
        
        print(f"✅ 标准化药材数据保存成功：{standardized_data_path}")
        print(f"   包含 {len(self.standardized_data)} 种标准化药材")
    
    def run(self):
        """执行完整的数据标准化流程"""
        print("=== 开始药材数据标准化流程 ===")
        
        # 1. 加载现有药材数据
        self.load_existing_herbs()
        
        # 2. 生成PLU映射文件
        self.generate_plu_mapping()
        
        # 3. 保存标准化后的数据
        self.save_standardized_data()
        
        print("\n=== 数据标准化流程完成 ===")
        print(f"总处理药材数量：{len(self.standardized_data)}")
        print(f"生成PLU编码数量：{len(self.plu_map)}")
        print(f"数据准确率：99.9%（目标）")

if __name__ == "__main__":
    # 创建数据标准化器实例
    standardizer = HerbDataStandardizer()
    
    # 执行数据标准化流程
    standardizer.run()
