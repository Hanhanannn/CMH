import os
import pickle
from logger import Logger

class BaseLibManager:
    def __init__(self, base_lib_path="base_lib"):
        # 初始化日志记录器
        self.logger = Logger("BaseLibManager")
        self.logger.info(f"初始化BaseLibManager，基础库路径：{base_lib_path}")
        
        self.base_lib_path = base_lib_path
        self.base_lib = self._load_base_lib()
        
        # 加载PLU码映射
        plu_map_path = "data/plu_map.pkl"
        if os.path.exists(plu_map_path):
            with open(plu_map_path, "rb") as f:
                self.plu_map = pickle.load(f)
        else:
            self.plu_map = {}
            self.logger.warning("PLU码映射文件不存在，使用空映射")
    
    def _load_base_lib(self):
        """加载基础库数据"""
        base_lib = {}
        
        # 检查基础库目录是否存在
        if not os.path.exists(self.base_lib_path):
            self.logger.warning(f"基础库目录不存在：{self.base_lib_path}")
            return base_lib
        
        # 遍历基础库目录下的所有.pkl文件
        for file_name in os.listdir(self.base_lib_path):
            if file_name.endswith(".pkl"):
                herb_name = file_name[:-4]  # 去掉.pkl后缀
                file_path = os.path.join(self.base_lib_path, file_name)
                
                try:
                    with open(file_path, "rb") as f:
                        herb_data = pickle.load(f)
                        base_lib[herb_name] = herb_data
                    self.logger.debug(f"已加载基础库数据：{herb_name}")
                except Exception as e:
                    self.logger.error(f"加载基础库数据失败 {file_name}：{e}")
        
        self.logger.info(f"共加载 {len(base_lib)} 种药材的基础库数据")
        return base_lib
    
    def match_base_lib(self, herb_name):
        """根据药材名称匹配基础库数据"""
        if herb_name in self.base_lib:
            # 始终使用plu_map中的PLU码，而不是base_lib中的PLU码
            if self.plu_map is not None and herb_name in self.plu_map:
                plu_code = self.plu_map[herb_name]
            else:
                # 如果plu_map为None或药材不在plu_map中，使用base_lib中的PLU码
                plu_code = self.base_lib[herb_name].get("plu", "")
            return {
                "name": herb_name,
                "plu": plu_code,
                "box": self.base_lib[herb_name].get("box", {"xmin": 0.0, "ymin": 0.0, "xmax": 224.0, "ymax": 224.0})
            }
        else:
            # 如果没有找到，返回一个包含PLU码的默认值
            # 优先使用plu_map中的PLU码，如果plu_map为None则使用base_lib中的PLU码
            if self.plu_map is not None and herb_name in self.plu_map:
                plu_code = self.plu_map[herb_name]
            else:
                # 如果plu_map为None或药材不在plu_map中，尝试从base_lib获取PLU码
                # 查找base_lib中是否有该药材的PLU码
                for herb_info in self.base_lib.values():
                    if herb_info.get("name") == herb_name:
                        plu_code = herb_info.get("plu", "")
                        break
                else:
                    plu_code = ""
            return {"plu": plu_code, "name": herb_name}
    
    def add_plu_mapping(self, herb_name, plu_code):
        """添加PLU码映射关系"""
        self.plu_map[herb_name] = plu_code
        # 保存PLU码映射到文件
        plu_map_path = "data/plu_map.pkl"
        try:
            with open(plu_map_path, "wb") as f:
                pickle.dump(self.plu_map, f)
            self.logger.info(f"✅ 添加PLU码映射：{herb_name} → {plu_code}")
        except Exception as e:
            self.logger.error(f"❌ 保存PLU码映射失败：{e}")
    
    def get_herb_info(self, plu_code):
        """根据PLU码获取药材信息，支持各种格式的PLU码"""
        self.logger.info(f"开始根据PLU码获取药材信息：{plu_code}")
        
        original_plu = plu_code
        
        # 1. 首先检查是否是模型支持的10种药材之一（PLU码从9990开始）
        model_supported_herbs = ['白薇', '徐长卿', '细辛', '菊花', '桂枝', '茯苓', '白芷', '黄芪', '柴胡', '陈皮']
        
        # 直接创建模型支持的PLU码映射，确保PLU码与药材名称正确匹配
        model_herb_plu_mapping = {
            "9990": "白薇",
            "9991": "徐长卿",
            "9992": "细辛",
            "9993": "菊花",
            "9994": "桂枝",
            "9995": "茯苓",
            "9996": "白芷",
            "9997": "黄芪",
            "9998": "柴胡",
            "9999": "陈皮"
        }
        
        # 如果是模型支持的PLU码，直接返回对应的药材信息
        if plu_code in model_herb_plu_mapping:
            herb_name = model_herb_plu_mapping[plu_code]
            self.logger.info(f"✅ 直接找到模型支持的药材：{herb_name} (PLU: {plu_code})")
            return {"name": herb_name, "plu": plu_code}
        
        # 2. 支持多种PLU码格式
        if isinstance(plu_code, str):
            # 首先尝试直接匹配
            for herb_name, info in self.base_lib.items():
                if info.get("plu") == plu_code:
                    self.logger.info(f"✅ 根据PLU码 {original_plu} 直接找到药材：{herb_name} (PLU: {info['plu']})")
                    return info
            
            # 3. 尝试移除前导零后匹配
            try:
                stripped_plu = str(int(plu_code))
                if stripped_plu != plu_code:
                    self.logger.info(f"尝试移除前导零后匹配：{plu_code} → {stripped_plu}")
                    for herb_name, info in self.base_lib.items():
                        if str(info.get("plu")) == stripped_plu:
                            self.logger.info(f"✅ 移除前导零后找到药材：{herb_name} (PLU: {info['plu']})")
                            return info
                        # 同时尝试将stripped_plu转换为短PLU码格式（001-010）
                        if f"{int(stripped_plu):03d}" == info.get("plu"):
                            self.logger.info(f"✅ 移除前导零后转换为短PLU码格式找到药材：{herb_name} (PLU: {info['plu']})")
                            return info
            except ValueError:
                pass
            
            # 4. 处理长PLU码（1000-1009）转换为短PLU码（001-010）
            if plu_code.startswith("100") and len(plu_code) == 4:
                try:
                    # 将1000-1009转换为001-010
                    short_plu = f"{int(plu_code) - 999:03d}"
                    self.logger.info(f"✅ 长PLU码 {plu_code} 转换为短PLU码 {short_plu}")
                    
                    # 尝试匹配转换后的短PLU码
                    for herb_name, info in self.base_lib.items():
                        if info.get("plu") == short_plu:
                            self.logger.info(f"✅ 转换后根据PLU码 {short_plu} 找到药材：{herb_name} (PLU: {info['plu']})")
                            return info
                except ValueError as e:
                    self.logger.error(f"❌ 长PLU码转换失败：{plu_code}，错误：{e}")
            
            # 5. 尝试将短PLU码转换为长PLU码匹配
            if len(plu_code) == 3 and plu_code.isdigit():
                try:
                    long_plu = f"{int(plu_code) + 999:04d}"  # 001-010 → 1000-1009
                    self.logger.info(f"尝试将短PLU码转换为长PLU码匹配：{plu_code} → {long_plu}")
                    for herb_name, info in self.base_lib.items():
                        if info.get("plu") == long_plu:
                            self.logger.info(f"✅ 转换为长PLU码后找到药材：{herb_name} (PLU: {info['plu']})")
                            return info
                except ValueError:
                    pass
            
            # 6. 尝试直接匹配数字部分（不考虑格式）
            try:
                plu_num = int(plu_code)
                self.logger.info(f"尝试直接匹配数字部分：{plu_num}")
                for herb_name, info in self.base_lib.items():
                    try:
                        info_plu_num = int(info.get("plu"))
                        if info_plu_num == plu_num:
                            self.logger.info(f"✅ 直接匹配数字部分找到药材：{herb_name} (PLU: {info['plu']})")
                            return info
                    except ValueError:
                        pass
            except ValueError:
                pass
            
            # 7. 尝试匹配plu_map中的映射关系
            reverse_plu_map = {v: k for k, v in self.plu_map.items()}
            if plu_code in reverse_plu_map:
                herb_name = reverse_plu_map[plu_code]
                if herb_name in self.base_lib:
                    # 使用plu_map中的PLU码，而不是base_lib中的PLU码
                    result = self.base_lib[herb_name].copy()
                    result['plu'] = plu_code
                    self.logger.info(f"✅ 通过PLU码映射找到药材：{herb_name} (PLU: {plu_code})")
                    return result
        
        # 8. 如果所有尝试都失败，返回一个包含PLU码的默认信息
        self.logger.warning(f"❌ 未找到PLU码 {original_plu} 对应的药材信息，使用默认信息")
        return {"name": plu_code, "plu": plu_code}
    
    def update_base_lib(self, herb_name, herb_data):
        """更新基础库数据"""
        self.base_lib[herb_name] = herb_data
        file_path = os.path.join(self.base_lib_path, f"{herb_name}.pkl")
        
        try:
            with open(file_path, "wb") as f:
                pickle.dump(herb_data, f)
            self.logger.info(f"已更新基础库数据：{herb_name}")
            return True
        except Exception as e:
            self.logger.error(f"更新基础库数据失败 {herb_name}：{e}")
            return False
    
    def get_all_herbs(self):
        """获取所有药材名称列表"""
        return list(self.base_lib.keys())
    
    def get_all_plu_codes(self):
        """获取所有PLU码列表"""
        return [info.get("plu") for info in self.base_lib.values() if "plu" in info]

# 测试代码
if __name__ == "__main__":
    manager = BaseLibManager()
    print("所有药材名称：", manager.get_all_herbs())
    print("所有PLU码：", manager.get_all_plu_codes())
    print("测试匹配药材：", manager.match_base_lib("茯苓"))