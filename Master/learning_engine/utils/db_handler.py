import faiss
import os
import pickle
import pymysql
from pymysql.cursors import DictCursor
from ..config import FEATURE_DB_PATH

class FeatureDBHandler:
    """特征库操作类（支持本地faiss+pickle和MySQL数据库）"""
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化特征库操作类，只执行一次"""
        if self._initialized:
            return
            
        # 按PLU码隔离的特征存储，键为PLU码，值为(index, plu_map)元组
        self.plu_features = {}
        self._load_db()
        
        # 初始化MySQL数据库连接
        self.db_config = {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': '123456',
            'db': 'zhongyaocai_db',
            'charset': 'utf8mb4',
            'cursorclass': DictCursor
        }
        self.connection = None
        self.cursor = None
        
        self._initialized = True

    def _load_db(self):
        """加载本地特征库"""
        if os.path.exists(FEATURE_DB_PATH):
            with open(FEATURE_DB_PATH, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    # 新格式：按PLU码隔离的特征存储
                    self.plu_features = data
                elif len(data) == 2:
                    # 旧格式：(index, plu_map)，转换为新格式
                    old_index, old_plu_map = data
                    if old_index and old_plu_map:
                        # 将旧格式数据转换为新格式
                        for idx, plu_code in old_plu_map.items():
                            if plu_code not in self.plu_features:
                                # 为每个PLU码创建独立的索引
                                self.plu_features[plu_code] = (faiss.IndexFlatL2(old_index.d), {})
                            index, plu_map = self.plu_features[plu_code]
                            feature = old_index.reconstruct(idx)
                            new_idx = index.ntotal
                            index.add(feature.reshape(1, -1))
                            plu_map[new_idx] = plu_code
        else:
            # 初始化为空字典
            self.plu_features = {}

    def save_db(self):
        """保存特征库到本地"""
        os.makedirs(os.path.dirname(FEATURE_DB_PATH), exist_ok=True)
        with open(FEATURE_DB_PATH, "wb") as f:
            pickle.dump(self.plu_features, f)

    def _connect_mysql(self):
        """连接MySQL数据库"""
        try:
            if not self.connection or not self.connection.open:
                self.connection = pymysql.connect(**self.db_config)
                self.cursor = self.connection.cursor()
            return True
        except Exception as e:
            print(f"MySQL连接失败: {str(e)}")
            return False

    def _disconnect_mysql(self):
        """断开MySQL数据库连接"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            return True
        except Exception as e:
            print(f"断开MySQL连接失败: {str(e)}")
            return False

    def add_feature(self, feature, plu_code):
        """添加特征到库中，按PLU码隔离"""
        try:
            import numpy as np
            
            # 确保feature是numpy数组
            if isinstance(feature, list):
                feature = np.array(feature)
            elif not isinstance(feature, np.ndarray):
                feature = np.array(feature)
            
            # 获取特征维度
            feature_dim = feature.shape[0]
            
            # 如果PLU码不存在，创建新的索引
            if plu_code not in self.plu_features:
                self.plu_features[plu_code] = (faiss.IndexFlatL2(feature_dim), {})
            
            # 获取该PLU码对应的索引和映射
            index, plu_map = self.plu_features[plu_code]
            
            # 如果索引维度不匹配，重建该PLU码的索引
            if index.d != feature_dim:
                # 只重建当前PLU码的索引，不影响其他PLU码
                index = faiss.IndexFlatL2(feature_dim)
                plu_map = {}
                self.plu_features[plu_code] = (index, plu_map)
            
            # 添加特征
            idx = index.ntotal
            index.add(feature.reshape(1, -1))
            plu_map[idx] = plu_code
            self.save_db()
            return True
        except Exception as e:
            print(f"添加特征失败：{e}")
            import traceback
            traceback.print_exc()
            return False

    def delete_feature_by_plu(self, plu_code):
        """按PLU码删除特征"""
        if plu_code in self.plu_features:
            # 直接删除该PLU码的所有特征，不影响其他PLU码
            del self.plu_features[plu_code]
            self.save_db()
            return True
        return False

    def clear_all(self):
        """清空所有特征"""
        self.plu_features = {}
        self.save_db()

    def clear_features_by_plu(self, plu_code):
        """按PLU码清空特征"""
        if plu_code in self.plu_features:
            # 清空该PLU码的特征，但保留索引结构
            index = self.plu_features[plu_code][0]
            self.plu_features[plu_code] = (faiss.IndexFlatL2(index.d), {})
            self.save_db()
            return True
        return False

    def get_feature_by_request_id(self, request_id):
        """根据requestId获取某次识别的特征（需关联识别记录）"""
        # 需结合识别模块的日志/数据库，此处为示例
        # 实际需存储requestId -> 特征/索引ID的映射
        pass
        
    def get_herb_info_from_mysql(self, herb_name):
        """从MySQL数据库获取药材信息"""
        if not self._connect_mysql():
            return None
        try:
            sql = "SELECT * FROM pharmacy_prescription_item WHERE name = %s"
            self.cursor.execute(sql, (herb_name,))
            result = self.cursor.fetchone()
            return result
        except Exception as e:
            print(f"MySQL查询失败: {str(e)}")
            return None
        finally:
            self._disconnect_mysql()
            
    def get_all_herbs_from_mysql(self):
        """从MySQL数据库获取所有药材信息"""
        if not self._connect_mysql():
            return None
        try:
            sql = "SELECT * FROM pharmacy_prescription_item"
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            return result
        except Exception as e:
            print(f"MySQL查询失败: {str(e)}")
            return None
        finally:
            self._disconnect_mysql()
            
    def get_features_by_plu(self, plu_code):
        """根据PLU码获取特征"""
        features = []
        if plu_code in self.plu_features:
            index, plu_map = self.plu_features[plu_code]
            for idx in range(index.ntotal):
                feature = index.reconstruct(idx).tolist()
                features.append(feature)
        return features
    
    def get_all_features(self):
        """获取所有特征，按PLU码分组"""
        all_features = {}
        for plu_code, (index, plu_map) in self.plu_features.items():
            if plu_code not in all_features:
                all_features[plu_code] = []
            for idx in range(index.ntotal):
                feature = index.reconstruct(idx).tolist()
                all_features[plu_code].append(feature)
        return all_features
