from ..utils.db_handler import FeatureDBHandler

class FeatureClean:
    """特征清理（cmd=217）"""
    def __init__(self):
        self.feature_db = FeatureDBHandler()

    def correct_feature(self, wrong_plu, correct_plu, request_id):
        """
        修正错误特征（迁移+删除冗余）
        :param wrong_plu: 错误的PLU码
        :param correct_plu: 正确的PLU码
        :param request_id: 错误识别的requestId
        :return: 修正结果（bool）
        """
        try:
            # 1. 根据requestId获取错误特征的索引ID（需关联识别记录）
            # 此处需补充requestId -> 索引ID的映射逻辑
            # 目前暂不支持按requestId删除单个特征，使用按PLU码清空后重新学习的方式
            # wrong_idx = self.feature_db.get_feature_by_request_id(request_id)
            # if not wrong_idx:
            #     return False

            # 2. 获取错误PLU码的所有特征
            wrong_features = self.feature_db.get_features_by_plu(wrong_plu)
            if not wrong_features:
                return True  # 没有错误特征，直接返回成功

            # 3. 将错误特征迁移到正确PLU码
            for wrong_feature in wrong_features:
                self.feature_db.add_feature(wrong_feature, correct_plu)

            # 4. 删除错误PLU码对应的所有特征
            self.feature_db.delete_feature_by_plu(wrong_plu)
            return True
        except Exception as e:
            print(f"特征清理失败：{e}")
            return False
    
    def optimize_features(self, plu_code, max_features=100):
        """
        优化特征，保留最新的N个特征
        :param plu_code: 药材PLU码
        :param max_features: 保留的最大特征数量
        :return: 优化结果（bool）
        """
        try:
            # 获取该PLU码的所有特征
            features = self.feature_db.get_features_by_plu(plu_code)
            if not features:
                return True  # 没有特征，直接返回成功
            
            # 如果特征数量超过最大值，只保留最新的max_features个
            if len(features) > max_features:
                # 清空现有特征
                self.feature_db.clear_features_by_plu(plu_code)
                # 重新添加最新的max_features个特征
                # 注意：由于特征存储顺序即为添加顺序，最新的特征在列表末尾
                for feature in features[-max_features:]:
                    self.feature_db.add_feature(feature, plu_code)
            
            return True
        except Exception as e:
            print(f"特征优化失败：{e}")
            return False
