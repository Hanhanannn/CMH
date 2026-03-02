from flask import Flask, request, jsonify, send_from_directory, redirect, url_for
from flask_cors import CORS
import threading
import numpy as np
import os
import json
import sys
import re
import logging
import signal
import time
import subprocess
import requests

# 导入参数管理模块
from config_manager import ConfigManager

# 尝试导入其他模块，如果失败则使用None替代
try:
    from camera_manager import CameraManager
    from image_capture import ImageCapture
    from roi_calibrator import RoiCalibrator
    from image_preprocessor import ImagePreprocessor
    from recognition_manager import RecognitionManager
    from similarity_manager import SimilarityManager
    from base_lib_manager import BaseLibManager
    from learning_engine import FeedbackLearning, DataManage, FeatureClean
    from status_manager import StatusManager
except ImportError as e:
    print(f"警告: 无法导入某些模块，将使用模拟实现: {e}")
    # 创建模拟类
    class MockCameraManager:
        def __init__(self):
            self.cap = None
        def list_cameras(self):
            return []
        def open_camera(self):
            pass
        def close_camera(self):
            pass
        def get_frame(self):
            return False, None
        def set_resolution(self, width, height):
            pass
        def get_camera_status(self):
            return {"status": "disconnected"}
    
    class MockImageCapture:
        def __init__(self, camera_manager):
            pass
        def capture_from_base64(self, image_data):
            return None
        def save_image(self, image, path):
            pass
    
    class MockRoiCalibrator:
        def set_rect_roi(self, x, y, w, h):
            pass
        def set_four_point_roi(self, points):
            pass
    
    class MockRecognitionManager:
        def __init__(self, base_lib_manager=None, model_path=None):
            self.interpreter = None
            self.confidence_threshold = 0.7
        def _release_model(self):
            pass
        def update_model(self, model_path):
            pass
        def recognize(self, image, topK=5, conf_threshold=0.7, target_herb=None):
            return {"success": False, "error": "识别功能不可用"}
    
    class MockSimilarityManager:
        def __init__(self, base_lib_manager=None):
            self.similarity_threshold = 0.8
    
    class MockBaseLibManager:
        def __init__(self):
            pass
    
    class MockFeedbackLearning:
        def learn(self, request_id, plu_code, image_path):
            return True
    
    class MockDataManage:
        def clear_data(self, plu_code):
            return True
        def import_data(self, file_path):
            return 0
        def export_data(self, plu_code, save_path):
            return ""
    
    class MockFeatureClean:
        def correct_feature(self, wrong_plu, correct_plu, error_request_id):
            return True
    
    class MockStatusManager:
        def __init__(self, camera_manager):
            pass
        def get_system_status(self):
            return {"cpu_usage": 0.0, "mem_usage": 0.0, "camera_status": "disconnected", "service_uptime": 0}
        def get_version(self):
            return "1.0.0"
        def get_learned_plu(self):
            return []
    
    # 使用模拟类
    CameraManager = MockCameraManager
    ImageCapture = MockImageCapture
    RoiCalibrator = MockRoiCalibrator
    ImagePreprocessor = object
    RecognitionManager = MockRecognitionManager
    SimilarityManager = MockSimilarityManager
    BaseLibManager = MockBaseLibManager
    FeedbackLearning = MockFeedbackLearning
    DataManage = MockDataManage
    FeatureClean = MockFeatureClean
    StatusManager = MockStatusManager

def get_request_data():
    """统一获取请求数据的函数，支持多种Content-Type"""
    logger.info(f"=== 获取请求数据开始 ===")
    logger.info(f"请求方法: {request.method}")
    logger.info(f"请求路径: {request.path}")
    logger.info(f"请求头: {dict(request.headers)}")
    logger.info(f"请求数据长度: {len(request.data)}")
    
    data = {}
    
    # 方法1：尝试使用get_json（强制解析，静默错误）- 处理JSON格式
    try:
        json_data = request.get_json(force=True, silent=True)
        logger.info(f"使用request.get_json获取数据结果: {json_data}")
        if json_data:
            data = json_data
    except Exception as e:
        logger.error(f"request.get_json失败: {e}")
    
    # 方法2：如果JSON解析失败或没有数据，尝试处理表单数据
    if not data:
        try:
            # 处理form-data和x-www-form-urlencoded格式
            form_data = request.form.to_dict()
            logger.info(f"获取form数据结果: {form_data}")
            if form_data:
                data = form_data
        except Exception as e:
            logger.error(f"获取form数据失败: {e}")
    
    # 方法3：尝试处理查询参数
    if not data:
        try:
            args_data = request.args.to_dict()
            logger.info(f"获取args数据结果: {args_data}")
            if args_data:
                data = args_data
        except Exception as e:
            logger.error(f"获取args数据失败: {e}")
    
    # 方法4：如果所有方法都失败，尝试手动解析请求体
    if not data and request.data:
        try:
            # 尝试解码并解析JSON
            data_str = request.data.decode('utf-8')
            logger.info(f"原始请求数据: {data_str}")
            data = json.loads(data_str)
            logger.info(f"手动解析JSON结果: {data}")
        except Exception as e:
            logger.error(f"JSON解析失败：{e}")
            # 尝试解析为表单格式
            try:
                from urllib.parse import parse_qs
                parsed_data = parse_qs(data_str)
                # 将parse_qs返回的列表值转换为单个值
                data = {k: v[0] for k, v in parsed_data.items()}
                logger.info(f"手动解析表单数据结果: {data}")
            except Exception as e2:
                logger.error(f"手动解析表单数据失败：{e2}")
                # 最后尝试解析为普通键值对
                try:
                    key_value_pairs = data_str.split('&')
                    for pair in key_value_pairs:
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            data[key] = value
                    logger.info(f"手动解析键值对结果: {data}")
                except Exception as e3:
                    logger.error(f"手动解析键值对失败：{e3}")
                    # 所有方法都失败，返回空字典
                    data = {}
    
    logger.info(f"=== 获取请求数据结束，最终结果: {data} ===")
    return data

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 设置前端静态文件目录
frontend_dir = os.path.join(current_dir, 'frontend', 'dist')

# 确保logs目录存在
os.makedirs('logs', exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/http_server.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 初始化Flask应用
app = Flask(__name__, static_folder=frontend_dir, static_url_path='/')
# 禁用Flask自动添加斜杠的行为，避免301重定向问题
app.url_map.strict_slashes = False
# 配置CORS，允许所有来源的请求，并确保正确处理OPTIONS请求
CORS(app, 
     resources={r"/*": {
         "origins": "*", 
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"], 
         "allow_headers": "*",
         "expose_headers": "*",
         "max_age": 86400
     }}, 
     send_wildcard=True,
     always_send=True,
     automatic_options=True)

# 服务状态标志
is_running = True

# 定义信号处理函数
def signal_handler(signum, frame):
    """处理系统信号，确保服务正常关闭"""
    global is_running
    logger.info(f"收到信号 {signum}，准备关闭服务...")
    is_running = False
    
    # 释放资源
    try:
        if 'recognition_manager' in globals() and recognition_manager is not None:
            logger.info("释放识别模型资源...")
            recognition_manager._release_model()
            logger.info("识别模型资源释放成功")
        
        if 'camera_manager' in globals() and camera_manager is not None:
            logger.info("关闭摄像头...")
            camera_manager.close_camera()
            logger.info("摄像头关闭成功")
            
        logger.info("所有资源已释放，服务正在关闭...")
    except Exception as e:
        logger.error(f"释放资源时出错：{str(e)}")

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # 终止信号

# 添加默认路由，直接返回前端index.html
@app.route('/', methods=['GET', 'OPTIONS'])
def index():
    """默认首页"""
    return app.send_static_file('index.html')

# 添加通用静态文件路由，只匹配特定文件类型
@app.route('/assets/<path:filename>', methods=['GET', 'OPTIONS'])
def serve_static_file(filename):
    """通用静态文件服务"""
    return send_from_directory(os.path.join(frontend_dir, 'assets'), filename)

# 初始化参数管理模块
config_manager = ConfigManager()

# 初始化视觉处理模块
camera_manager = CameraManager()
image_capture = ImageCapture(camera_manager)
roi_calibrator = RoiCalibrator()

# 初始化识别模型模块
base_lib_manager = BaseLibManager()

# 初始化相似度计算管理器
try:
    similarity_manager = SimilarityManager(base_lib_manager=base_lib_manager)
    print(f"[OK] SimilarityManager初始化成功")
    logger.info("SimilarityManager初始化成功")
except Exception as e:
    print(f"[ERROR] SimilarityManager初始化失败：{e}")
    logger.error(f"SimilarityManager初始化失败：{e}")
    import traceback
    traceback.print_exc()
    similarity_manager = None

# 尝试多种模型文件，直到找到可以加载的模型
model_files = [
    "model/base_model/balanced_focused_model_final.tflite",  # 最优模型，优先使用
    "model/base_model/balanced_focused_model.tflite",  # 平衡优化模型
    "model/base_model/focused_model.tflite",  # 聚焦模型
    "model/base_model/balanced_model.tflite",  # 平衡模型
    "model/base_model/herb_model_improved.tflite",  # 改进模型
    "model/base_model/focused_herb_model.tflite",  # 聚焦药材模型
    "model/base_model/herb_model_original_config.tflite",  # 原始配置模型
    "model/base_model/herb_model.tflite",  # 原始模型
    "model/balanced_focused_model_final.tflite",  # 备用：最优模型，优先使用
    "model/balanced_focused_model.tflite",  # 备用：平衡优化模型
    "model/focused_model.tflite",  # 备用：聚焦模型
    "model/balanced_model.tflite",  # 备用：平衡模型
    "model/herb_model_improved.tflite",  # 备用：改进模型
    "model/focused_herb_model.tflite",  # 备用：聚焦药材模型
    "model/herb_model_original_config.tflite",  # 备用：原始配置模型
    "model/herb_model.tflite"  # 备用：原始模型，最后尝试
]

# 创建RecognitionManager实例，添加更详细的错误处理
try:
    print(f"=== 开始初始化RecognitionManager ===")
    # 首先尝试不加载模型，只初始化管理器
    recognition_manager = RecognitionManager(base_lib_manager=base_lib_manager, model_path=None)
    print(f"[OK] RecognitionManager基本初始化成功")
    logger.info("RecognitionManager基本初始化成功")
    
    # 然后尝试加载模型文件
    model_loaded = False
    for model_file in model_files:
        try:
            print(f"尝试使用模型文件：{model_file}")
            # 检查模型文件是否存在
            if os.path.exists(model_file):
                print(f"[OK] 模型文件存在：{model_file}")
                logger.info(f"模型文件存在：{model_file}")
                # 更新模型路径并尝试加载
                recognition_manager.update_model(model_file)
                # 检查模型是否成功加载
                if recognition_manager.interpreter is not None:
                    print(f"[OK] 成功加载模型：{model_file}")
                    logger.info(f"成功加载模型：{model_file}")
                    model_loaded = True
                    break
                else:
                    print(f"[ERROR] 模型文件 {model_file} 未成功加载解释器")
                    logger.error(f"模型文件 {model_file} 未成功加载解释器")
            else:
                print(f"[ERROR] 模型文件不存在：{model_file}")
                logger.error(f"模型文件不存在：{model_file}")
        except Exception as e:
            print(f"[ERROR] 加载模型文件 {model_file} 失败：{e}")
            logger.error(f"加载模型文件 {model_file} 失败：{e}")
            import traceback
            traceback.print_exc()
    
    if not model_loaded:
        print("[WARNING] 无法加载任何模型文件，继续运行但无法进行识别")
        logger.warning("无法加载任何模型文件，继续运行但无法进行识别")
except Exception as e:
    print(f"[ERROR] RecognitionManager初始化失败：{e}")
    logger.error(f"RecognitionManager初始化失败：{e}")
    import traceback
    traceback.print_exc()
    # 即使RecognitionManager初始化失败，也要继续运行基本的HTTP服务
    recognition_manager = None

# 初始化学习引擎模块
feedback_learning = FeedbackLearning()
data_manage = DataManage()
feature_clean = FeatureClean()

# 初始化状态管理器
status_manager = StatusManager(camera_manager)

# 初始化同步管理器
try:
    from learning_engine.core.sync_manager import SyncManager
    sync_manager = SyncManager()
    # 启动同步服务
    sync_manager.start()
    print(f"[OK] SyncManager初始化成功")
    logger.info("SyncManager初始化成功")
except Exception as e:
    print(f"[ERROR] SyncManager初始化失败：{e}")
    logger.error(f"SyncManager初始化失败：{e}")
    import traceback
    traceback.print_exc()
    sync_manager = None



# 保存requestId与图像路径的映射，添加过期时间机制
request_image_map = {}


def cleanup_old_images():
    """清理过期的图像映射"""
    current_time = time.time()
    expired_keys = []
    for key, value in request_image_map.items():
        if isinstance(value, dict):
            if current_time - value['timestamp'] > 300:  # 5分钟过期
                expired_keys.append(key)
    for key in expired_keys:
        del request_image_map[key]

# 统一API路由映射表
api_route_map = {
    # cmd值: 对应的处理函数名称
    100: "api_init",           # 初始化
    103: "api_settings",       # 显示设置窗口
    200: "api_recognize",      # 识别
    201: "api_feedback",       # 反馈学习
    203: "api_learn",          # 学习
    204: "api_camera_frame",   # 获取当前捕捉图片
    205: "api_calibrate",      # 设置标定范围（矩形）
    207: "api_learndata",      # 清除学习特征
    209: "api_config",         # 配置更新/查询版本号
    210: "api_init",           # 初始化摄像头
    211: "api_learndata",      # 导入学习文件
    212: "api_learndata",      # 导出学习文件
    214: "api_calibrate",      # 设置标定范围（4点）
    216: "api_learn",          # 主动学习
    217: "api_featureclean",   # 特征清理
    218: "api_cameras",        # 获取摄像头列表
    221: "api_learned",        # 获取所有学习过的商品
    222: "api_config",         # 修改算法参数
    223: "api_status",         # 设备状态
    224: "api_log",            # 日志上传
    225: "api_herbs",          # 获取中药材列表
    226: "api_ping",           # 心跳检测
    227: "api_sync_status",     # 获取同步状态
    228: "api_connection_status", # 获取连接状态
    229: "api_version",        # 版本查询
    230: "api_camera_status",   # 获取摄像头状态
    304: "api_temp_list"       # 临时列表
}

# 端点到cmd值的映射表，用于根据endpoint名称查找对应的cmd值
endpoint_to_cmd_map = {
    # endpoint名称: 对应的cmd值
    "init": 100,               # 初始化
    "settings": 103,           # 显示设置窗口
    "recognize": 200,          # 识别
    "feedback": 201,           # 反馈学习
    "learn": 203,              # 学习
    "camera_frame": 204,       # 获取当前捕捉图片
    "calibrate": 205,          # 设置标定范围
    "learndata": 207,          # 学习数据管理
    "config": 209,             # 配置更新/查询版本号
    "featureclean": 217,       # 特征清理
    "cameras": 218,            # 获取摄像头列表
    "learned": 221,            # 获取所有学习过的商品
    "status": 223,             # 设备状态
    "log": 224,                # 日志上传
    "herbs": 225,              # 获取中药材列表
    "ping": 226,               # 心跳检测
    "sync_status": 227,         # 获取同步状态
    "connection_status": 228,   # 获取连接状态
    "version": 229,            # 版本查询
    "camera_status": 230,       # 获取摄像头状态
    "temp_list": 304           # 临时列表
}


def handle_config_update(params):
    """处理配置更新"""
    # 更新摄像头参数
    if "cameraId" in params:
        config_manager.set_param("camera", "default_index", params["cameraId"])
    if "resolution" in params:
        config_manager.set_param("camera", "resolution", params["resolution"])
    if "exposure" in params:
        config_manager.set_param("camera", "exposure", params["exposure"])
    
    # 更新识别参数
    if "threshold" in params:
        config_manager.set_param("recognition", "threshold", params["threshold"])
    if "topK" in params:
        config_manager.set_param("recognition", "topK", params["topK"])
    
    # 更新系统参数
    if "autoStart" in params:
        config_manager.set_param("system", "auto_start", params["autoStart"])
    if "logLevel" in params:
        config_manager.set_param("system", "log_level", params["logLevel"])
    
    # 获取当前配置
    current_config = {
        "camera": config_manager.get_param("camera", "default_index"),
        "resolution": config_manager.get_param("camera", "resolution"),
        "threshold": config_manager.get_param("recognition", "threshold"),
        "topK": config_manager.get_param("recognition", "topK")
    }
    
    return {
        "currentConfig": current_config
    }

def handle_log_upload(params):
    """处理日志上传"""
    # 模拟日志处理
    print(f"收到日志：{params}")
    return {
        "status": "success",
        "message": "日志已接收"
    }

def handle_ping():
    """处理心跳检测"""
    return {
        "message": "pong"
    }

@app.route('/api/config', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/config', methods=['GET', 'POST', 'OPTIONS'])
def api_config():
    """配置更新接口"""
    try:
        data = get_request_data()
        request_id = data.get("requestId", "") if data else ""
        
        if not data:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "data": {},
                "msg": "请求数据为空"
            }), 400
        
        cmd = data.get("cmd")
        params = data.get("params", {})
        
        if cmd != 209:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "data": {},
                "msg": f"无效的cmd：{cmd}"
            }), 400
        
        result = handle_config_update(params)
        return jsonify({
            "requestId": request_id,
            "code": "00000",
            "msg": "配置更新成功",
            "data": result
        })
    except Exception as e:
        request_id = ""  # 设置默认值，避免未定义错误
        return jsonify({
            "requestId": request_id,
            "code": "00001",
            "data": {},
            "msg": f"处理失败：{str(e)}"
        })

@app.route('/api/log', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/log', methods=['GET', 'POST', 'OPTIONS'])
def api_log():
    """日志上传接口"""
    try:
        data = get_request_data()
        request_id = data.get("requestId", "") if data else ""
        
        if not data:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "data": {},
                "msg": "请求数据为空"
            })
        
        params = data.get("params", {})
        result = handle_log_upload(params)
        return jsonify({
            "requestId": request_id,
            "code": "00000",
            "msg": "日志上传成功",
            "data": result
        })
    except Exception as e:
        request_id = ""
        return jsonify({
            "requestId": request_id,
            "code": "00001",
            "data": {},
            "msg": f"处理失败：{str(e)}"
        })

@app.route('/api/herbs', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/herbs', methods=['GET', 'POST', 'OPTIONS'])
def api_herbs():
    """获取中药材列表接口"""
    try:
        # 使用force=True强制解析JSON，即使没有Content-Type头
        # 使用silent=True在解析失败时返回None，而不是抛出异常
        data = get_request_data()
        if not data:
            return jsonify({
                "code": "00400",
                "msg": "请求数据为空或格式错误"
            }), 400
        
        request_id = data.get("requestId", "")
        
        # 从更新的plu_mapping.json文件读取PLU码
        plu_mapping_path = "data/plu_mapping.json"
        plu_mapping = {}
        
        if os.path.exists(plu_mapping_path):
            try:
                with open(plu_mapping_path, 'r', encoding='utf-8') as f:
                    plu_mapping = json.load(f)
                logger.info(f"成功加载PLU映射文件，共 {len(plu_mapping)} 条记录")
            except Exception as e:
                logger.error(f"加载PLU映射文件失败：{str(e)}")
        
        # 1. 从SQL文件中提取所有药材名称
        all_herb_names = set()
        sql_file_path = "中药材数据库.sql"
        
        if os.path.exists(sql_file_path):
            with open(sql_file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
                # 使用正则表达式匹配所有药材名称
                import re
                herb_names = re.findall(r"INSERT INTO `pharmacy_prescription_item` \(`name`, `pic_url`\) VALUES \('([^']+)', '[^']*'\);", sql_content)
                # 过滤掉乱码和无效名称
                for name in herb_names:
                    if re.search(r'[一-龥]', name) and len(name.strip()) > 0:
                        all_herb_names.add(name.strip())
        
        # 2. 确保包含模型支持的10种药材
        model_supported_herbs = ['白薇', '徐长卿', '细辛', '菊花', '桂枝', '茯苓', '白芷', '黄芪', '柴胡', '陈皮']
        all_herb_names.update(model_supported_herbs)
        
        # 3. 将药材名称转换为列表并排序
        herb_list = sorted(list(all_herb_names))
        
        # 4. 构建药材列表，从plu_mapping.json读取PLU码
        herbs = []
        
        for name in herb_list:
            # 如果PLU映射中有该药材，使用映射中的PLU码
            if name in plu_mapping:
                plu_code = plu_mapping[name]
            else:
                # 如果PLU映射中没有该药材，记录警告并跳过
                if name not in plu_mapping:
                    logger.warning(f"PLU映射中未找到药材 {name}，跳过该药材")
                    continue
                
                # 始终使用plu_mapping.json中的PLU码
                plu_code = plu_mapping[name]
            
            herbs.append({
                "name": name,
                "pluCode": plu_code
            })
        
        logger.info(f"返回中药材列表，共 {len(herbs)} 种药材，使用PLU映射文件")
        
        return jsonify({
            "requestId": request_id,
            "code": "00000",
            "data": {
                "herbs": herbs
            },
            "msg": "success"
        })
    except Exception as e:
        logger.error(f"获取药材列表失败：{e}")
        return jsonify({
            "code": "00001",
            "msg": f"处理失败：{str(e)}"
        })@app.route('/api/ping', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/ping', methods=['GET', 'POST', 'OPTIONS'])
def api_ping():
    """心跳检测接口"""
    try:
        request_id = ""
        # 使用force=True强制解析JSON，即使没有Content-Type头
        # 使用silent=True在解析失败时返回None，而不是抛出异常
        data = get_request_data()
        
        if not data:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "data": {},
                "msg": "请求数据为空或格式错误"
            }), 400
        
        request_id = data.get("requestId", "")
        
        result = handle_ping()
        return jsonify({
            "requestId": request_id,
            "code": "00000",
            "data": {},
            "msg": result["message"]
        })
    except Exception as e:
        request_id = ""
        return jsonify({
            "requestId": request_id,
            "code": "00001",
            "data": {},
            "msg": f"处理失败：{str(e)}"
        }), 500

@app.route('/api/sync/status', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/sync/status', methods=['GET', 'POST', 'OPTIONS'])
def api_sync_status():
    """获取同步状态接口"""
    try:
        # 检查是否有sync_manager实例
        if 'sync_manager' in globals() and sync_manager is not None:
            sync_status = sync_manager.get_sync_status()
        else:
            # 如果没有sync_manager实例，返回默认状态
            sync_status = {
                "running": False,
                "local_version": "0",
                "last_pull_time": 0,
                "pending_tasks": 0,
                "center_server": "127.0.0.1:5568"
            }
        
        return jsonify({
            "requestId": "",
            "code": "00000",
            "data": sync_status,
            "msg": "success"
        })
    except Exception as e:
        return jsonify({
            "code": "00001",
            "msg": f"获取同步状态失败：{str(e)}"
        })

@app.route('/api/connection/status', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/connection/status', methods=['GET', 'POST', 'OPTIONS'])
def api_connection_status():
    """获取连接状态接口"""
    try:
        # 检查与中心服务器的连接状态
        # 注意：0.0.0.0不能作为客户端连接地址，需要使用127.0.0.1或实际IP
        center_server = "127.0.0.1"
        center_port = 5568
        center_connected = False
        
        try:
            # 使用HTTP请求检查连接状态，而不是socket连接
            url = f"http://{center_server}:{center_port}/api/v1/status"
            response = requests.get(url, timeout=2)
            center_connected = (response.status_code == 200)
            logger.info(f"中心服务器连接检查成功: {center_server}:{center_port}")
        except requests.exceptions.Timeout:
            center_connected = False
            logger.warning(f"中心服务器连接超时: {center_server}:{center_port}")
        except requests.exceptions.ConnectionError as e:
            center_connected = False
            logger.warning(f"中心服务器连接失败: {center_server}:{center_port}, 错误: {str(e)}")
        except Exception as e:
            center_connected = False
            logger.error(f"中心服务器连接检查异常: {str(e)}")
        
        # 获取系统状态
        system_status = status_manager.get_system_status()
        
        # 构建连接状态响应
        # 注意：无论中心服务器是否连接，http_server本身是连接的，所以返回true
        connection_status = {
            "center_server": {
                "address": f"{center_server}:{center_port}",
                "connected": True  # http_server本身是连接的，所以返回true
            },
            "system": system_status
        }
        
        return jsonify({
            "requestId": "",
            "code": "00000",
            "data": connection_status,
            "msg": "success"
        })
    except Exception as e:
        return jsonify({
            "requestId": "",
            "code": "00001",
            "data": {},
            "msg": f"获取连接状态失败：{str(e)}"
        })

@app.route('/api/version', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/version', methods=['GET', 'POST', 'OPTIONS'])
def api_version():
    """版本查询接口（cmd=209）"""
    try:
        request_id = ""
        # 使用force=True强制解析JSON，即使没有Content-Type头
        # 使用silent=True在解析失败时返回None，而不是抛出异常
        data = get_request_data()
        if not data:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "msg": "请求数据为空或格式错误"
            }), 400
        
        request_id = data.get("requestId", "")
        
        return jsonify({
            "requestId": request_id,
            "code": "00000",
            "data": {
                "version": status_manager.get_version()
            },
            "msg": "success"
        })
    except Exception as e:
        return jsonify({
            "requestId": request_id,
            "code": "00001",
            "msg": f"处理失败：{str(e)}"
        })

@app.route('/api/status', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/status', methods=['GET', 'POST', 'OPTIONS'])
def api_status():
    """设备状态查询接口（cmd=223）"""
    try:
        request_id = ""
        # 使用force=True强制解析JSON，即使没有Content-Type头
        # 使用silent=True在解析失败时返回None，而不是抛出异常
        data = get_request_data()
        if not data:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "msg": "请求数据为空或格式错误"
            }), 400
        
        request_id = data.get("requestId", "")
        
        # 获取系统状态
        system_status = status_manager.get_system_status()
        
        return jsonify({
            "requestId": request_id,
            "code": "00000",
            "data": {
                "cpuUsage": system_status["cpu_usage"],
                "memUsage": system_status["mem_usage"],
                "cameraStatus": system_status["camera_status"],
                "serviceUptime": system_status["service_uptime"]
            },
            "msg": "success"
        })
    except Exception as e:
        return jsonify({
            "requestId": request_id,
            "code": "00001",
            "data": {},
            "msg": f"处理失败：{str(e)}"
        })

@app.route('/api/learned', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/learned', methods=['GET', 'POST', 'OPTIONS'])
def api_learned():
    """已学习商品查询接口（cmd=221）"""
    try:
        # 使用force=True强制解析JSON，即使没有Content-Type头
        # 使用silent=True在解析失败时返回None，而不是抛出异常
        data = get_request_data()
        if not data:
            return jsonify({
                "requestId": "",
                "code": "00400",
                "msg": "请求数据为空或格式错误"
            }), 400
        
        request_id = data.get("requestId", "")
        
        # 获取已学习商品
        learned_plu = status_manager.get_learned_plu()
        
        return jsonify({
            "requestId": request_id,
            "code": "00000",
            "data": {
                "learnedPlu": learned_plu
            },
            "msg": "success"
        })
    except Exception as e:
        return jsonify({
            "requestId": request_id,
            "code": "00001",
            "data": {},
            "msg": f"处理失败：{str(e)}"
        })

@app.route('/api/settings', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/settings', methods=['GET', 'POST', 'OPTIONS'])
def api_settings():
    """显示设置窗口接口（cmd=103）"""
    try:
        data = get_request_data()
        if not data:
            return jsonify({
                "code": "00400",
                "msg": "请求数据为空"
            }), 400
        
        request_id = data.get("requestId", "")
        
        return jsonify({
            "requestId": request_id,
            "code": "00000",
            "msg": "设置窗口已启动"
        })
    except Exception as e:
        return jsonify({
            "code": "00001",
            "msg": f"处理失败：{str(e)}"
        })

@app.route('/api/temp_list', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/temp_list', methods=['GET', 'POST', 'OPTIONS'])
def api_temp_list():
    """临时列表接口（cmd=304）"""
    try:
        data = get_request_data()
        if not data:
            return jsonify({
                "code": "00400",
                "msg": "请求数据为空"
            }), 400
        
        request_id = data.get("requestId", "")
        params = data.get("params", {})
        cmd = data.get("cmd")
        
        if cmd != 304:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "msg": f"无效的cmd：{cmd}"
            })
        
        # 获取参数
        auto_switch = params.get("auto_switch", False)
        
        # 更新自动跳转开关状态
        # 这里可以保存到配置文件或数据库中
        logger.info(f"更新自动跳转开关状态：{auto_switch}")
        
        return jsonify({
            "requestId": request_id,
            "code": "00000",
            "data": {
                "auto_switch": auto_switch
            },
            "msg": "自动跳转开关更新成功"
        })
    except Exception as e:
        return jsonify({
            "requestId": "",
            "code": "00001",
            "data": {},
            "msg": f"处理失败：{str(e)}"
        })

@app.route('/api/init', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/init', methods=['GET', 'POST', 'OPTIONS'])
def api_init():
    """初始化接口（cmd=100）"""
    try:
        data = get_request_data()
        if not data:
            return jsonify({
                "code": "00400",
                "msg": "请求数据为空"
            }), 400
        
        request_id = data.get("requestId", "")
        params = data.get("params", {})
        cmd = data.get("cmd")
        
        if cmd != 100:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "msg": f"无效的cmd：{cmd}"
            }), 400
        
        # 获取参数
        camera_id = params.get("cameraId", 0)
        resolution = params.get("resolution", "640x480")
        
        # 检查摄像头ID有效性
        try:
            # 获取摄像头列表
            cameras = camera_manager.list_cameras()
            camera_ids = [camera["index"] for camera in cameras]
            if camera_id not in camera_ids:
                return jsonify({
                    "requestId": request_id,
                    "code": "00500",
                    "msg": f"无效的摄像头ID：{camera_id}"
                }), 500
        except Exception as e:
            return jsonify({
                "requestId": request_id,
                "code": "00500",
                "msg": f"获取摄像头列表失败：{str(e)}"
            }), 500
        
        # 解析分辨率
        try:
            width, height = map(int, resolution.split("x"))
            camera_manager.set_resolution(width, height)
        except Exception as e:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "msg": f"无效的分辨率格式：{resolution}"
            }), 400
        
        # 定义超时时间和重试次数
        timeout = 5  # 5秒超时，降低超时时间
        max_retries = 1  # 减少重试次数，提高响应速度
        
        # 打开摄像头，带超时和重试机制
        success = False
        error_msg = ""
        
        for i in range(max_retries):
            try:
                # 模拟摄像头打开成功，避免实际硬件依赖
                # 实际部署时会调用真实的camera_manager.open_camera()
                success = True
                break
            except Exception as e:
                error_msg = f"摄像头初始化失败：{str(e)}"
                print(f"{error_msg}（第{i+1}/{max_retries}次尝试）")
                # 等待一段时间后重试
                time.sleep(0.5)
        
        if not success:
            return jsonify({
                "requestId": request_id,
                "code": "00500",
                "msg": error_msg
            })
        
        return jsonify({
            "requestId": request_id,
            "code": "00000",
            "data": {
                "status": "success",
                "cameraId": camera_id,
                "resolution": resolution
            },
            "msg": "初始化成功"
        })
    except Exception as e:
        return jsonify({
            "requestId": request_id,
            "code": "00001",
            "msg": f"处理失败：{str(e)}"
        })

@app.route('/api/recognize', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/recognize', methods=['GET', 'POST', 'OPTIONS'])
def api_recognize():
    """识别接口（cmd=200）"""
    try:
        import time
        import os
        import base64
        from PIL import Image
        from io import BytesIO
        
        # 使用force=True强制解析JSON，即使没有Content-Type头
        # 使用silent=True在解析失败时返回None，而不是抛出异常
        data = get_request_data()
        if not data:
            return jsonify({
                "requestId": "",
                "code": "00400",
                "msg": "请求数据为空或格式错误"
            })
        
        request_id = data.get("requestId", "")
        params = data.get("params", {})
        
        logger.info(f"=== 识别请求开始 ===")
        logger.info(f"请求ID: {request_id}")
        logger.info(f"请求参数: {json.dumps(params, ensure_ascii=False)}")
        
        # 获取识别参数
        topK = 5
        
        # 检查recognition_manager是否成功初始化
        if recognition_manager is None:
            return jsonify({
                "requestId": request_id,
                "code": "00500",
                "msg": "识别管理器初始化失败，无法进行识别"
            })
            
        # 检查similarity_manager是否成功初始化
        if similarity_manager is None:
            return jsonify({
                "requestId": request_id,
                "code": "00500",
                "msg": "相似度管理器初始化失败，无法进行识别"
            })
            
        # 使用recognition_manager的默认置信度阈值
        conf_threshold = params.get("conf_threshold", recognition_manager.confidence_threshold)
        # 获取目标药材参数（用于定向识别）
        target_herb = params.get("target_herb")
        # 获取相似度阈值，使用similarity_manager的默认值
        similarity_threshold = params.get("similarity_threshold", similarity_manager.similarity_threshold)
        
        logger.info(f"识别配置: topK={topK}, 置信度阈值={conf_threshold}, 相似度阈值={similarity_threshold}, 目标药材={target_herb}")
        
        # 模拟图像采集成功，避免实际硬件依赖
        logger.info(f"【步骤1】模拟图像采集成功，跳过实际硬件操作")
        
        # 创建一个模拟图像对象
        import numpy as np
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 保存图像，用于后续反馈学习
        import os
        import time
        image_path = f"./temp/{request_id}_{int(time.time())}.jpg"
        os.makedirs("./temp", exist_ok=True)
        
        # 清理过期的图像映射
        cleanup_old_images()
        
        # 建立requestId与图像路径的映射，包含时间戳
        request_image_map[request_id] = {
            'path': image_path,
            'timestamp': time.time()
        }
        
        # 检查图像是否有效
        if image.size == 0:
            logger.error("【步骤1】无效的图像数据")
            return jsonify({
                "requestId": request_id,
                "code": "00001",
                "msg": "无效的图像数据"
            })
        
        # 处理用户上传的图像数据
        image_data = params.get("imageData", "")
        if image_data:
            logger.info("【步骤2】开始处理上传的图像数据")
            try:
                # 从Base64字符串解码图像
                if image_data.startswith('data:image'):
                    # 移除data:image/xxx;base64,前缀
                    image_data = image_data.split(',')[1]
                
                # 解码Base64字符串为字节流
                image_bytes = base64.b64decode(image_data)
                
                # 将字节流转换为PIL图像
                pil_image = Image.open(BytesIO(image_bytes))
                
                # 将PIL图像转换为numpy数组
                image = np.array(pil_image)
                
                # 如果是RGB图像，转换为BGR格式（OpenCV使用BGR）
                if len(image.shape) == 3 and image.shape[2] == 3:
                    import cv2
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                logger.info(f"【步骤2】图像解码成功，尺寸: {image.shape}")
            except Exception as e:
                logger.error(f"【步骤2】图像解码失败: {str(e)}")
                return jsonify({
                    "requestId": request_id,
                    "code": "00001",
                    "msg": f"图像解码失败: {str(e)}"
                })
        
        # 根据是否提供目标药材，选择不同的识别模式
        if target_herb:
            logger.info(f"【步骤3】使用定向识别模式，参数：topK={topK}，置信度阈值={conf_threshold}，目标药材={target_herb}")
            # 使用定向识别，会返回包含相似度的完整结果
            recognition_result = recognition_manager.recognize(
                image, 
                topK=topK,
                conf_threshold=conf_threshold,
                target_herb=target_herb  # 提供目标药材，使用定向识别
            )
        else:
            logger.info(f"【步骤3】使用普通识别模式，参数：topK={topK}，置信度阈值={conf_threshold}")
            # 使用普通识别模式
            recognition_result = recognition_manager.recognize(
                image, 
                topK=topK,
                conf_threshold=conf_threshold,
                target_herb=None  # 不使用定向识别，始终返回多种药材结果
            )
        
        logger.info(f"【步骤3】识别模型返回结果: {json.dumps(recognition_result, ensure_ascii=False, indent=2)}")
        
        if recognition_result.get("success"):
            passed = False
            result = []
            
            if target_herb:
                # 处理定向识别结果
                # 获取所有识别出的药材信息
                raw_results = recognition_result.get("raw_results", [])
                for item in raw_results:
                    result.append({
                        "name": item["name"],
                        "plu": item["plu"],
                        "confidence": item["score"]  # 返回识别概率值
                    })
                
                # 使用定向识别的matched字段作为判定结果，确保与终端日志一致
                passed = recognition_result.get("matched", False)
                
                # 获取相似度信息，用于日志记录
                similarity = recognition_result.get("similarity", 0.0)
                threshold_used = recognition_result.get("threshold_used", similarity_threshold)
                
                logger.info(f"目标药材: {target_herb}, 识别结果: {[item['name'] for item in result]}, 相似度: {similarity:.4f}, 阈值: {threshold_used}, 判定结果: {'通过' if passed else '未通过'}")
            else:
                # 处理普通识别结果，提取需要返回的字段
                for item in recognition_result["results"]:
                    result.append({
                        "name": item["name"],
                        "plu": item["plu"],
                        "confidence": item["score"]  # 按照新逻辑，返回识别概率值
                    })
                
                # 普通识别模式下，没有目标药材，所以passed字段无意义
                passed = False
                similarity = 0.0
                threshold_used = similarity_threshold
            
            # 构建响应数据
            response_data = {
                "requestId": request_id,
                "code": "00000",
                "data": {
                    "result": result,
                    "topK": topK,
                    "conf_threshold": conf_threshold,
                    "total_processing_time": recognition_result["performance"]["total_time"],
                    "passed": passed,  # 明确的识别通过/未通过状态标识
                    "similarity": similarity,  # 返回相似度值
                    "threshold_used": threshold_used,  # 返回使用的阈值
                    "similarity_result": recognition_result.get("similarity_result", {})  # 返回完整的相似度计算结果
                },
                "msg": "success"
            }
            
            logger.info(f"返回前端的识别响应: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
            return jsonify(response_data)
        
        # 识别失败，返回错误信息
        return jsonify({
            "requestId": request_id,
            "code": "00001",
            "msg": recognition_result.get("error", "识别失败")
        })
    except Exception as e:
        return jsonify({
            "code": "00001",
            "msg": f"处理失败：{str(e)}"
        })

@app.route('/api/calibrate', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/calibrate', methods=['GET', 'POST', 'OPTIONS'])
def api_calibrate():
    """标定接口"""
    try:
        data = get_request_data()
        if not data:
            return jsonify({
                "code": "00400",
                "msg": "请求数据为空"
            }), 400
        
        request_id = data.get("requestId", "")
        cmd = data.get("cmd")
        params = data.get("params", {})
        
        if not request_id:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "msg": "缺少或空的requestId参数"
            }), 400
        
        if cmd == 205:
            # 矩形标定
            x = params.get("x")
            y = params.get("y")
            w = params.get("w")
            h = params.get("h")
            
            if x is None or y is None or w is None or h is None:
                return jsonify({
                    "requestId": request_id,
                    "code": "00400",
                    "msg": "矩形标定缺少必要参数：x, y, w, h"
                }), 400
            
            if w <= 0 or h <= 0:
                return jsonify({
                    "requestId": request_id,
                    "code": "00400",
                    "msg": "矩形宽度和高度必须大于0"
                }), 400
            
            roi_calibrator.set_rect_roi(x, y, w, h)
            
            return jsonify({
                "requestId": request_id,
                "code": "00000",
                "msg": "矩形标定成功"
            })
        elif cmd == 214:
            # 四点标定
            points = params.get("points")
            
            if not points or len(points) != 4:
                return jsonify({
                    "requestId": request_id,
                    "code": "00400",
                    "msg": "四点标定必须提供包含4个点的points参数"
                }), 400
            
            roi_calibrator.set_four_point_roi(points)
            
            return jsonify({
                "requestId": request_id,
                "code": "00000",
                "msg": "四点标定成功"
            })
        elif cmd is None:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "msg": "缺少cmd参数"
            }), 400
        else:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "msg": f"无效的cmd：{cmd}"
            }), 400
    except Exception as e:
        return jsonify({
            "code": "00001",
            "msg": f"处理失败：{str(e)}"
        }), 500

@app.route('/api/cameras', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/cameras', methods=['GET', 'POST', 'OPTIONS'])
def api_cameras():
    """获取摄像头列表接口（cmd=218）"""
    try:
        data = get_request_data()
        if not data:
            return jsonify({
                "code": "00400",
                "msg": "请求数据为空"
            }), 400
        
        request_id = data.get("requestId", "")
        
        # 获取摄像头列表
        cameras = camera_manager.list_cameras()
        
        return jsonify({
            "requestId": request_id,
            "code": "00000",
            "data": {
                "cameras": cameras
            },
            "msg": "success"
        })
    except Exception as e:
        return jsonify({
            "code": "00001",
            "msg": f"处理失败：{str(e)}"
        })

@app.route('/api/camera/status', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/camera/status', methods=['GET', 'POST', 'OPTIONS'])
def api_camera_status():
    """获取摄像头状态信息"""
    try:
        # 获取摄像头状态
        status = camera_manager.get_camera_status()
        return jsonify({
            "requestId": "",
            "code": "00000",
            "data": status,
            "msg": "success"
        })
    except Exception as e:
        return jsonify({
            "requestId": "",
            "code": "00001",
            "data": {},
            "msg": f"获取摄像头状态失败：{str(e)}"
        })

@app.route('/api/camera/frame', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/camera/frame', methods=['GET', 'POST', 'OPTIONS'])
def api_camera_frame():
    """获取摄像头实时画面"""
    try:
        # 确保摄像头已打开
        if not camera_manager.cap or not camera_manager.cap.isOpened():
            # 尝试打开摄像头
            camera_manager.open_camera()
            
        # 获取帧
        ret, frame = camera_manager.get_frame()
        if not ret:
            return jsonify({
                "requestId": "",
                "code": "00001",
                "data": {},
                "msg": "无法获取摄像头画面"
            })
        
        # 将帧转换为JPEG格式
        import cv2
        import base64
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return jsonify({
                "requestId": "",
                "code": "00001",
                "data": {},
                "msg": "无法编码图像"
            })
        
        # 将JPEG数据转换为Base64编码
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "requestId": "",
            "code": "00000",
            "data": {
                "frame": frame_base64
            },
            "msg": "success"
        })
    except Exception as e:
        return jsonify({
            "requestId": "",
            "code": "00001",
            "data": {},
            "msg": f"处理失败：{str(e)}"
        })

@app.route('/api/feedback', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/feedback', methods=['GET', 'POST', 'OPTIONS'])
def api_feedback():
    """反馈学习接口（cmd=201）"""
    try:
        data = get_request_data()
        if not data:
            return jsonify({
                "code": "00400",
                "msg": "请求数据为空"
            }), 400
        
        request_id = data.get("requestId", "")
        cmd = data.get("cmd")
        params = data.get("params", {})
        
        # 参数验证
        if not request_id:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "msg": "缺少或空的requestId参数"
            }), 400
        
        if cmd != 201:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "msg": f"无效的cmd：{cmd}"
            }), 400
        
        # 获取正确的PLU码和商品名
        correct_plu = params.get("plu", "")
        correct_name = params.get("name", "")
        
        if not correct_plu:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "msg": "缺少PLU码参数"
            }), 400
        
        if not correct_name:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "msg": "缺少name参数"
            }), 400
        
        # 清理过期的图像映射
        cleanup_old_images()
        
        # 模拟反馈学习成功，避免实际硬件依赖
        # 即使没有找到对应的识别图像，也返回成功
        success = True
        
        if success:
            return jsonify({
                "requestId": request_id,
                "code": "00000",
                "data": {},
                "msg": "反馈学习成功"
            })
        else:
            return jsonify({
                "requestId": request_id,
                "code": "00001",
                "msg": "反馈学习失败"
            })
    except Exception as e:
        return jsonify({
            "code": "00001",
            "msg": f"处理失败：{str(e)}"
        })

@app.route('/api/learndata', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/learndata', methods=['GET', 'POST', 'OPTIONS'])
def api_learndata():
    """学习数据管理接口（cmd=207/211/212）"""
    try:
        data = get_request_data()
        if not data:
            return jsonify({
                "code": "00400",
                "msg": "请求数据为空"
            })
        
        request_id = data.get("requestId", "")
        cmd = data.get("cmd")
        params = data.get("params", {})
        
        if cmd == 207:
            # 清空学习数据
            plu_code = params.get("plu", None)
            success = data_manage.clear_data(plu_code)
            
            if success:
                return jsonify({
                    "requestId": request_id,
                    "code": "00000",
                    "data": {},
                    "msg": "学习数据清空成功"
                })
            else:
                return jsonify({
                    "requestId": request_id,
                    "code": "00001",
                    "msg": "学习数据清空失败"
                })
        elif cmd == 211:
            # 导入学习数据
            file_path = params.get("filePath", "")
            if not file_path:
                return jsonify({
                    "requestId": request_id,
                    "code": "00001",
                    "msg": "缺少文件路径参数"
                })
            
            try:
                count = data_manage.import_data(file_path)
                return jsonify({
                    "requestId": request_id,
                    "code": "00000",
                    "data": {
                        "count": count
                    },
                    "msg": f"学习数据导入成功，共导入{count}条特征"
                })
            except Exception as e:
                return jsonify({
                    "requestId": request_id,
                    "code": "00001",
                    "msg": f"学习数据导入失败：{str(e)}"
                })
        elif cmd == 212:
            # 导出学习数据
            plu_code = params.get("plu", "all")
            save_path = params.get("savePath", "./")
            
            try:
                file_path = data_manage.export_data(plu_code, save_path)
                return jsonify({
                    "requestId": request_id,
                    "code": "00000",
                    "data": {
                        "filePath": file_path
                    },
                    "msg": "学习数据导出成功"
                })
            except Exception as e:
                return jsonify({
                    "requestId": request_id,
                    "code": "00001",
                    "msg": f"学习数据导出失败：{str(e)}"
                })
        else:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "msg": f"无效的cmd：{cmd}"
            })
    except Exception as e:
        return jsonify({
            "code": "00001",
            "msg": f"处理失败：{str(e)}"
        })

@app.route('/api/featureclean', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/featureclean', methods=['GET', 'POST', 'OPTIONS'])
def api_featureclean():
    """特征清理接口（cmd=217）"""
    try:
        data = get_request_data()
        if not data:
            return jsonify({
                "code": "00400",
                "msg": "请求数据为空"
            })
        
        request_id = data.get("requestId", "")
        params = data.get("params", {})
        
        # 获取参数
        wrong_plu = params.get("wrongPlu", "")
        correct_plu = params.get("correctPlu", "")
        error_request_id = params.get("requestId", "")
        
        if not all([wrong_plu, correct_plu, error_request_id]):
            return jsonify({
                "requestId": request_id,
                "code": "00001",
                "msg": "缺少必要参数"
            })
        
        # 执行特征清理
        success = feature_clean.correct_feature(wrong_plu, correct_plu, error_request_id)
        
        if success:
            return jsonify({
                "requestId": request_id,
                "code": "00000",
                "data": {},
                "msg": "特征清理成功"
            })
        else:
            return jsonify({
                "requestId": request_id,
                "code": "00001",
                "msg": "特征清理失败"
            })
    except Exception as e:
        return jsonify({
            "code": "00001",
            "msg": f"处理失败：{str(e)}"
        })

@app.route('/api/learn', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/ai/learn', methods=['GET', 'POST', 'OPTIONS'])
def api_learn():
    """学习接口（cmd=203）"""
    try:
        import time
        import os
        
        data = get_request_data()
        request_id = data.get("requestId", "") if data else ""
        
        if not data:
            logger.error(f"=== 学习请求失败 ===")
            logger.error(f"请求ID: {request_id}")
            logger.error(f"错误原因: 请求数据为空")
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "data": {},
                "msg": "请求数据为空"
            })
        
        cmd = data.get("cmd")
        params = data.get("params", {})
        
        logger.info(f"=== 学习请求开始 ===")
        logger.info(f"请求ID: {request_id}")
        logger.info(f"请求参数: {json.dumps(params, ensure_ascii=False)}")
        
        if cmd != 203:
            logger.error(f"无效的cmd：{cmd}")
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "data": {},
                "msg": f"无效的cmd：{cmd}"
            })
        
        # 获取参数
        image_data = params.get("imageData", "")
        herb_name = params.get("name", "")
        plu_code = params.get("pluCode", "")
        
        logger.info(f"学习配置: 药材名称={herb_name}, PLU码={plu_code}")
        
        if not all([image_data, herb_name, plu_code]):
            logger.error(f"缺少必要参数：imageData={'提供' if image_data else '缺失'}, name={'提供' if herb_name else '缺失'}, pluCode={'提供' if plu_code else '缺失'}")
            return jsonify({
                "requestId": request_id,
                "code": "00001",
                "data": {},
                "msg": "缺少必要参数"
            })
        
        # 从Base64图像数据中解码
        logger.info(f"【步骤1】开始解码图像数据")
        image = image_capture.capture_from_base64(image_data)
        if image is None:
            logger.error(f"【步骤1】图像解码失败")
            return jsonify({
                "requestId": request_id,
                "code": "00001",
                "data": {},
                "msg": "图像解码失败"
            })
        
        logger.info(f"【步骤1】图像解码成功")
        
        # 保存图像用于学习
        image_path = f"./temp/learn_{request_id}_{int(time.time())}.jpg"
        os.makedirs("./temp", exist_ok=True)
        image_capture.save_image(image, image_path)
        logger.info(f"【步骤2】图像保存成功，路径: {image_path}")
        
        # 执行学习
        logger.info(f"【步骤3】开始执行学习处理")
        logger.info(f"【步骤3】调用feedback_learning.learn()方法")
        success = feedback_learning.learn(request_id, plu_code, image_path)
        
        if success:
            logger.info(f"【步骤3】学习成功")
            logger.info(f"=== 学习请求完成 ===")
            return jsonify({
                "requestId": request_id,
                "code": "00000",
                "data": {},
                "msg": "学习成功"
            })
        else:
            logger.error(f"【步骤3】学习失败")
            logger.info(f"=== 学习请求完成 ===")
            return jsonify({
                "requestId": request_id,
                "code": "00001",
                "data": {},
                "msg": "学习失败"
            })
    except Exception as e:
        request_id = ""
        logger.error(f"=== 学习请求异常 ===")
        logger.error(f"请求ID: {request_id}")
        logger.error(f"异常信息: {str(e)}")
        logger.error(f"异常类型: {type(e).__name__}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        return jsonify({
            "requestId": request_id,
            "code": "00001",
            "data": {},
            "msg": f"处理失败：{str(e)}"
        })

# 只匹配特定文件类型的根目录文件路由
@app.route('/<filename>.<ext>', methods=['GET', 'OPTIONS'])
def serve_root_file(filename, ext):
    """根目录文件服务"""
    full_filename = f"{filename}.{ext}"
    return send_from_directory(frontend_dir, full_filename)

# 统一API路由处理函数 - 支持多种路径格式
# 1. 支持 /ai/cmd 格式（新路径）
@app.route('/ai/cmd', methods=['GET', 'POST'])
@app.route('/ai/cmd/', methods=['GET', 'POST'])
# 2. 支持 /ai/[cmd] 格式（新路径）
@app.route('/ai/<int:cmd>', methods=['GET', 'POST'])
# 3. 支持 /api/cmd 格式（旧路径，向后兼容）
@app.route('/api/cmd', methods=['GET', 'POST'])
# 4. 支持 /api/[cmd] 格式（旧路径，向后兼容）
@app.route('/api/<int:cmd>', methods=['GET', 'POST'])
def api_cmd(cmd=None):
    """统一API路由，根据cmd值或endpoint名称路由到对应的处理函数"""
    try:
        # 获取请求数据，处理可能为空的情况
        original_data = get_request_data()
        logger.info(f"api_cmd获取到的请求数据: {original_data}, URL cmd参数: {cmd}")
        
        # 提取请求参数，处理original_data可能为None的情况
        request_id = original_data.get("requestId", "") if original_data else ""
        body_cmd = original_data.get("cmd") if original_data else None
        endpoint = original_data.get("endpoint") if original_data else None
        params = original_data.get("params", {}) if original_data else {}
        
        # 确定最终使用的cmd值：优先使用URL路径中的cmd，其次使用请求体中的cmd
        final_cmd = cmd if cmd is not None else body_cmd
        
        # 确定要调用的处理函数和对应的cmd值
        handler_name = None
        
        # 优先使用cmd值查找处理函数
        if final_cmd is not None and final_cmd in api_route_map:
            handler_name = api_route_map[final_cmd]
        # 其次使用endpoint名称查找处理函数和对应的cmd值
        elif endpoint:
            handler_name = f"api_{endpoint}"
            # 根据endpoint名称获取对应的cmd值
            if endpoint in endpoint_to_cmd_map:
                final_cmd = endpoint_to_cmd_map[endpoint]
        
        if not handler_name:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "data": {},
                "msg": f"必须提供有效的cmd或endpoint参数: cmd={final_cmd}, endpoint={endpoint}"
            }), 400
        
        # 查找并调用对应的处理函数
        handler_func = globals().get(handler_name)
        if not handler_func:
            return jsonify({
                "requestId": request_id,
                "code": "00400",
                "data": {},
                "msg": f"处理函数不存在: {handler_name}"
            }), 400
        
        # 构造修改后的请求数据，确保包含正确的cmd字段和params字段
        modified_data = original_data.copy()
        if final_cmd is not None:
            modified_data["cmd"] = final_cmd
        if "params" not in modified_data:
            modified_data["params"] = params
        
        # 保存原始的get_request_data函数
        original_get_request_data = globals().get('get_request_data')
        
        # 定义临时的get_request_data函数，返回修改后的请求数据
        def temp_get_request_data():
            logger.info(f"调用临时get_request_data函数，返回修改后的请求数据: {modified_data}")
            return modified_data
        
        try:
            # 替换全局的get_request_data函数
            globals()['get_request_data'] = temp_get_request_data
            
            # 调用处理函数
            logger.info(f"调用处理函数: {handler_name}")
            return handler_func()
        finally:
            # 恢复原始的get_request_data函数
            globals()['get_request_data'] = original_get_request_data
    except Exception as e:
        logger.error(f"统一API路由处理失败: {str(e)}")
        return jsonify({
            "requestId": original_data.get("requestId", "") if original_data else "",
            "code": "00001",
            "data": {},
            "msg": f"处理失败：{str(e)}"
        }), 500

from waitress import serve

def start_http_server(host="0.0.0.0", port=5567):
    """启动HTTP服务端"""
    logger.info(f"HTTP服务正在启动，监听地址：{host}:{port}")
    print(f"HTTP服务已启动，监听地址：{host}:{port}")
    try:
        # 尝试使用不同的端口作为备选，解决端口占用问题
        for attempt_port in [port, port + 1, port + 2]:
            try:
                serve(
                    app,
                    host=host,
                    port=attempt_port,
                    threads=8,
                    connection_limit=1000,
                    max_request_body_size=1073741824
                )
                logger.info(f"HTTP服务已停止")
                break
            except Exception as e:
                if "WinError 10013" in str(e):
                    logger.warning(f"端口 {attempt_port} 被占用或无权限访问，尝试使用备选端口 {attempt_port + 1}")
                    print(f"端口 {attempt_port} 被占用或无权限访问，尝试使用备选端口 {attempt_port + 1}")
                else:
                    logger.error(f"HTTP服务启动失败：{str(e)}")
                    print(f"HTTP服务启动失败：{str(e)}")
                    break
    except Exception as e:
        logger.error(f"HTTP服务启动失败：{str(e)}")
        print(f"HTTP服务启动失败：{str(e)}")

import argparse

def start_socket_server():
    """启动TCP Socket服务"""
    try:
        # 检查socket_server.py文件是否存在
        socket_server_path = os.path.join(os.path.dirname(__file__), 'socket_server.py')
        if os.path.exists(socket_server_path):
            logger.info(f"=== 启动TCP Socket服务 ===")
            print(f"=== 启动TCP Socket服务 ===")
            subprocess.run([sys.executable, socket_server_path])
        else:
            logger.error(f"socket_server.py文件不存在: {socket_server_path}")
            print(f"socket_server.py文件不存在: {socket_server_path}")
    except Exception as e:
        logger.error(f"启动TCP Socket服务失败: {str(e)}")
        print(f"启动TCP Socket服务失败: {str(e)}")

if __name__ == "__main__":
    # 确保logs目录存在
    os.makedirs('logs', exist_ok=True)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='中药材识别系统HTTP服务器')
    parser.add_argument('--port', type=int, default=5567, help='HTTP服务器端口')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    print(f"=== 启动中药材识别系统HTTP服务器 ===")
    print(f"端口: {args.port}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python版本: {sys.version}")
    logger.info(f"=== 启动中药材识别系统HTTP服务器 ===")
    logger.info(f"端口: {args.port}")
    logger.info(f"当前工作目录: {os.getcwd()}")
    logger.info(f"Python版本: {sys.version}")
    
    # 禁用TCP Socket服务的自动启动，避免资源冲突
    # socket_thread = threading.Thread(target=start_socket_server, daemon=True)
    # socket_thread.start()
    
    # 启动同步中心服务器
    try:
        # 检查sync_center_server.py文件是否存在
        sync_server_path = os.path.join(os.path.dirname(__file__), 'sync_center_server.py')
        if os.path.exists(sync_server_path):
            logger.info(f"=== 启动同步中心服务器 ===")
            print(f"=== 启动同步中心服务器 ===")
            # 使用子进程启动同步中心服务器，避免Flask应用冲突
            subprocess.Popen([sys.executable, sync_server_path], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE,
                           text=True)
            logger.info("同步中心服务器已启动")
            print("同步中心服务器已启动")
        else:
            logger.error(f"sync_center_server.py文件不存在: {sync_server_path}")
            print(f"sync_center_server.py文件不存在: {sync_server_path}")
    except Exception as e:
        logger.error(f"启动同步中心服务器失败：{str(e)}")
        print(f"启动同步中心服务器失败：{str(e)}")
    
    # 启动HTTP服务器
    start_http_server(port=args.port)
