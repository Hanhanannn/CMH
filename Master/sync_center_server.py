from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import time
import shutil
import threading
import gzip
from datetime import datetime
import hashlib
import uuid

app = Flask(__name__)
# 配置CORS，允许所有来源的请求
CORS(app, 
     resources={r"/*": {
         "origins": "*", 
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"], 
         "allow_headers": "*"
     }}, 
     send_wildcard=True,
     always_send=True,
     automatic_options=True)

class SyncCenterServer:
    """医馆中心节点服务"""
    def __init__(self, config_path="center_config.json"):
        self.config = self._load_config(config_path)
        self._ensure_storage_structure()
        self.tenants = self.config["tenants"]
        self.running = False
        self.backup_thread = None
    
    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            # 直接使用center_config.json作为配置文件
            config_path = "center_config.json"
            
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"配置文件不存在：{config_path}")
            return self._get_default_config()
        except json.JSONDecodeError:
            print(f"配置文件格式错误：{config_path}")
            # 尝试使用默认配置
            return self._get_default_config()
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            "server": {
                "port": 5568,
                "host": "0.0.0.0",
                "debug": False
            },
            "storage": {
                "base_path": "D:\\SFAICenter",
                "backup_enabled": True,
                "backup_interval": 86400,
                "max_backups": 7
            },
            "tenants": {},
            "security": {
                "enable_encryption": True,
                "allow_cors": True,
                "api_key_required": False
            },
            "advanced": {
                "log_level": "INFO",
                "max_request_size": 524288000,  # 增加到500MB
                "request_timeout": 60
            }
        }
    
    def _ensure_storage_structure(self):
        """确保存储结构存在"""
        try:
            # 创建基础存储目录
            os.makedirs(self.config["storage"]["base_path"], exist_ok=True)
            
            # 为每个租户创建目录
            for tenant_id in self.config["tenants"]:
                tenant_dir = self._get_tenant_dir(tenant_id)
                os.makedirs(tenant_dir, exist_ok=True)
                os.makedirs(os.path.join(tenant_dir, "backups"), exist_ok=True)
        except Exception as e:
            print(f"创建存储结构失败：{e}")
    
    def _get_tenant_dir(self, tenant_id):
        """获取租户目录"""
        return os.path.join(self.config["storage"]["base_path"], tenant_id)
    
    def _get_tenant_data_file(self, tenant_id):
        """获取租户数据文件路径"""
        return os.path.join(self._get_tenant_dir(tenant_id), "merged.sfd")
    
    def _get_tenant_log_file(self, tenant_id):
        """获取租户日志文件路径"""
        return os.path.join(self._get_tenant_dir(tenant_id), "sync_log.json")
    
    def _authenticate_tenant(self, tenant_id, secret_key):
        """认证租户身份"""
        if tenant_id not in self.tenants:
            return False, "租户不存在"
        
        tenant = self.tenants[tenant_id]
        if not tenant["enabled"]:
            return False, "租户已禁用"
        
        if tenant["secret_key"] != secret_key:
            return False, "密钥错误"
        
        return True, "认证成功"
    
    def _get_tenant_data(self, tenant_id):
        """获取租户数据"""
        data_file = self._get_tenant_data_file(tenant_id)
        if not os.path.exists(data_file):
            return {"version": 0, "data": {}}
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取租户数据失败：{e}")
            return {"version": 0, "data": {}}
    
    def _save_tenant_data(self, tenant_id, data):
        """保存租户数据"""
        data_file = self._get_tenant_data_file(tenant_id)
        try:
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"保存租户数据失败：{e}")
            return False
    
    def _log_sync_event(self, tenant_id, device_id, event_type, details):
        """记录同步事件"""
        log_file = self._get_tenant_log_file(tenant_id)
        
        # 生成日志条目
        log_entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "device_id": device_id,
            "event_type": event_type,
            "details": details
        }
        
        # 在终端输出学习信息保存同步日志
        print(f"\n[学习信息同步日志]")
        print(f"时间：{log_entry['datetime']}")
        print(f"租户ID：{tenant_id}")
        print(f"设备ID：{device_id}")
        print(f"事件类型：{event_type}")
        print(f"版本：{details.get('version', 'N/A')}")
        print(f"数据类型：{details.get('data_type', 'N/A')}")
        print(f"PLU码：{details.get('plu_code', 'N/A')}")
        print("[同步完成]\n")
        
        # 保存日志到文件
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except Exception as e:
                print(f"读取同步日志失败：{e}")
        
        logs.append(log_entry)
        
        # 只保留最近1000条日志
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存同步日志失败：{e}")
    
    def _merge_incremental_data(self, tenant_data, incremental_data):
        """合并增量数据"""
        try:
            # 首先检查incremental_data是否为字典类型
            if not isinstance(incremental_data, dict):
                print(f"增量数据格式错误：不是字典类型，实际类型：{type(incremental_data).__name__}")
                return tenant_data
            
            if incremental_data["type"] == "incremental":
                plu_code = incremental_data["plu_code"]
                features = incremental_data["features"]
                
                # 确保features是列表类型
                if not isinstance(features, list):
                    print(f"特征数据格式错误：不是列表类型，实际类型：{type(features).__name__}")
                    return tenant_data
                
                # 更新或添加特征
                if plu_code not in tenant_data["data"]:
                    tenant_data["data"][plu_code] = []
                
                # 合并特征，避免重复
                existing_features = tenant_data["data"][plu_code]
                for feature in features:
                    if feature not in existing_features:
                        existing_features.append(feature)
            elif incremental_data["type"] == "full":
                # 全量数据直接替换
                full_features = incremental_data["features"]
                # 确保full_features是字典类型
                if isinstance(full_features, dict):
                    tenant_data["data"] = full_features
                else:
                    print(f"全量特征数据格式错误：不是字典类型，实际类型：{type(full_features).__name__}")
            
            # 更新版本号
            tenant_data["version"] = max(tenant_data["version"], int(time.time() * 1000))
            
            return tenant_data
        except KeyError as e:
            print(f"合并增量数据失败：缺少必要字段 {e}")
            return tenant_data
        except Exception as e:
            print(f"合并增量数据失败：{e}")
            return tenant_data
    
    def _start_backup_thread(self):
        """启动备份线程"""
        if self.config["storage"]["backup_enabled"] and not self.backup_thread:
            self.backup_thread = threading.Thread(target=self._backup_loop)
            self.backup_thread.daemon = True
            self.backup_thread.start()
    
    def _backup_loop(self):
        """备份循环"""
        while self.running:
            self._perform_backup()
            time.sleep(self.config["storage"]["backup_interval"])
    
    def _perform_backup(self):
        """执行备份"""
        try:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for tenant_id in self.tenants:
                data_file = self._get_tenant_data_file(tenant_id)
                if os.path.exists(data_file):
                    backup_file = os.path.join(
                        self._get_tenant_dir(tenant_id),
                        "backups",
                        f"backup_{current_time}.sfd"
                    )
                    shutil.copy2(data_file, backup_file)
                    
                    # 清理旧备份
                    self._clean_old_backups(tenant_id)
        except Exception as e:
            print(f"执行备份失败：{e}")
    
    def _clean_old_backups(self, tenant_id):
        """清理旧备份"""
        backup_dir = os.path.join(self._get_tenant_dir(tenant_id), "backups")
        backups = []
        
        for filename in os.listdir(backup_dir):
            if filename.startswith("backup_") and filename.endswith(".sfd"):
                file_path = os.path.join(backup_dir, filename)
                mtime = os.path.getmtime(file_path)
                backups.append((mtime, file_path))
        
        # 按修改时间排序，保留最新的max_backups个
        backups.sort(reverse=True)
        for mtime, file_path in backups[self.config["storage"]["max_backups"]:]:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"删除旧备份失败：{e}")

    def _handle_sync_request_data(self, data, secret_key_header):
        """处理同步请求数据（业务逻辑）"""
        # 检查是否为加密数据
        if data.get("encrypted"):
            try:
                # 这里需要添加解密逻辑
                from Crypto.Cipher import AES
                from Crypto.Util.Padding import pad, unpad
                import base64
                
                # 获取加密数据
                encrypted_data = data.get("data")
                if not encrypted_data:
                    return jsonify({"success": False, "message": "加密数据不能为空"}), 400
                
                # 获取身份信息
                tenant_id = data.get("tenant_id")
                device_id = data.get("device_id")
                
                # 身份认证
                auth_success, auth_msg = self._authenticate_tenant(tenant_id, secret_key_header)
                if not auth_success:
                    return jsonify({"success": False, "message": auth_msg}), 403
                
                # 生成解密密钥（与子机保持一致，使用secret_key生成）
                hash_object = hashlib.sha256(secret_key_header.encode('utf-8'))
                key = hash_object.digest()[:16]
                iv = b'1234567890123456'  # 固定IV
                
                try:
                    # 解密数据
                    base64_decoded = base64.b64decode(encrypted_data)
                    cipher = AES.new(key, AES.MODE_CBC, iv)
                    decrypted_data_raw = cipher.decrypt(base64_decoded)
                    decrypted_data = unpad(decrypted_data_raw, AES.block_size)
                    decrypted_str = decrypted_data.decode('utf-8')
                    decrypted_json = json.loads(decrypted_str)
                except Exception as decrypt_e:
                    print(f"解密过程失败：{decrypt_e}")
                    import traceback
                    traceback.print_exc()
                    return jsonify({"success": False, "message": f"解密过程失败：{str(decrypt_e)}"}), 500
                
                # 处理解密后的同步数据
                sync_data = decrypted_json.get("data")
                if not sync_data:
                    return jsonify({"success": False, "message": "解密后的数据中没有同步数据"}), 400
                
                if not isinstance(sync_data, dict):
                    return jsonify({"success": False, "message": "同步数据必须是字典类型"}), 400
                
                # 获取当前租户数据
                tenant_data = self._get_tenant_data(tenant_id)
                
                # 合并增量数据
                merged_data = self._merge_incremental_data(tenant_data, sync_data)
                
                # 保存合并后的数据
                if self._save_tenant_data(tenant_id, merged_data):
                    # 记录日志
                    log_details = {
                        "version": merged_data["version"],
                        "data_type": sync_data.get("type", "unknown"),
                        "plu_code": sync_data.get("plu_code", "unknown")
                    }
                    self._log_sync_event(tenant_id, device_id, "upload", log_details)
                    
                    return jsonify({
                        "success": True,
                        "message": "加密数据处理成功",
                        "version": merged_data["version"]
                    })
                else:
                    return jsonify({"success": False, "message": "保存数据失败"}), 500
            except Exception as e:
                print(f"处理加密数据失败：{e}")
                import traceback
                traceback.print_exc()
                return jsonify({"success": False, "message": f"处理加密数据失败：{str(e)}"}), 500
        
        # 非加密数据处理
        # 提取身份信息
        tenant_id = data.get("tenant_id")
        device_id = data.get("device_id")
        secret_key = secret_key_header # 使用传入的header
        
        # 身份认证
        auth_success, auth_msg = self._authenticate_tenant(tenant_id, secret_key)
        if not auth_success:
            return jsonify({"success": False, "message": auth_msg}), 403
        
        # 处理同步数据
        sync_data = data.get("data")
        if not sync_data:
            return jsonify({"success": False, "message": "无效的同步数据"}), 400
        
        if not isinstance(sync_data, dict):
            return jsonify({"success": False, "message": "同步数据必须是字典类型"}), 400
        
        # 获取当前租户数据
        tenant_data = self._get_tenant_data(tenant_id)
        
        # 合并增量数据
        merged_data = self._merge_incremental_data(tenant_data, sync_data)
        
        # 保存合并后的数据
        if self._save_tenant_data(tenant_id, merged_data):
            # 记录日志
            log_details = {
                "version": merged_data["version"],
                "data_type": sync_data.get("type", "unknown"),
                "plu_code": sync_data.get("plu_code", "unknown")
            }
            self._log_sync_event(tenant_id, device_id, "upload", log_details)
            
            return jsonify({
                "success": True,
                "message": "数据上传成功",
                "version": merged_data["version"]
            })
        else:
            return jsonify({"success": False, "message": "保存数据失败"}), 500
    
    def start(self):
        """启动服务器"""
        self.running = True
        self._start_backup_thread()
        
        # 注册路由
        self._register_routes()
        
        # 启动Flask服务器
        app.run(
            host='0.0.0.0',
            port=self.config["server"]["port"],
            debug=self.config["server"]["debug"]
        )
    
    def stop(self):
        """停止服务器"""
        self.running = False
        if self.backup_thread:
            self.backup_thread.join()
    
    def _register_routes(self):
        """注册路由"""
        @app.route('/ai/v1/status', methods=['GET'])
        def status():
            """获取服务器状态"""
            return jsonify({
                "success": True,
                "message": "同步中心服务运行正常",
                "timestamp": time.time(),
                "version": "1.0.0"
            })
        
        @app.route('/ai/v1/sync/upload', methods=['POST'])
        def upload_data():
            """接收终端上传的增量数据"""
            try:
                # 直接在终端输出收到上传请求的日志
                print(f"\n[上传请求日志]")
                print(f"时间：{datetime.now().isoformat()}")
                print(f"请求路径：{request.path}")
                print(f"请求方法：{request.method}")
                print(f"客户端IP：{request.remote_addr}")
                print("[开始处理上传请求]\n")
                
                # 验证请求大小
                max_size = self.config.get("advanced", {}).get("max_request_size", 524288000)
                if request.content_length and request.content_length > max_size:
                    return jsonify({"success": False, "message": "请求大小超过限制"}), 413
                
                # 处理GZIP压缩
                if request.headers.get('Content-Encoding') == 'gzip':
                    try:
                        compressed_data = request.get_data()
                        json_data = gzip.decompress(compressed_data)
                        data = json.loads(json_data)
                    except Exception as e:
                        print(f"GZIP解压失败：{e}")
                        return jsonify({"success": False, "message": f"GZIP解压失败：{e}"}), 400
                else:
                    data = request.get_json()
                
                if not data:
                    return jsonify({"success": False, "message": "无效的请求数据"}), 400
                
                # 调用业务处理逻辑
                return self._handle_sync_request_data(data, request.headers.get("X-Secret-Key"))
            
            except Exception as e:
                print(f"处理上传请求失败：{e}")
                return jsonify({"success": False, "message": str(e)}), 500

        @app.route('/ai/v1/sync/upload_chunk', methods=['POST'])
        def upload_chunk():
            """上传分片"""
            try:
                upload_id = request.form.get('upload_id')
                chunk_index = request.form.get('chunk_index')
                total_chunks = request.form.get('total_chunks')
                file = request.files.get('file')
                
                if not all([upload_id, chunk_index is not None, total_chunks, file]):
                    return jsonify({"success": False, "message": "缺少必要参数"}), 400

                temp_dir = os.path.join("temp", "uploads", upload_id)
                os.makedirs(temp_dir, exist_ok=True)
                
                chunk_path = os.path.join(temp_dir, str(chunk_index))
                file.save(chunk_path)
                
                return jsonify({"success": True, "message": "分片上传成功"})
            except Exception as e:
                print(f"分片上传失败: {e}")
                return jsonify({"success": False, "message": str(e)}), 500

        @app.route('/ai/v1/sync/merge_chunks', methods=['POST'])
        def merge_chunks():
            """合并分片"""
            try:
                data = request.get_json()
                upload_id = data.get('upload_id')
                chunk_count = int(data.get('total_chunks'))
                
                temp_dir = os.path.join("temp", "uploads", upload_id)
                if not os.path.exists(temp_dir):
                    return jsonify({"success": False, "message": "上传会话不存在"}), 404
                
                merged_file_path = os.path.join(temp_dir, "merged_data")
                with open(merged_file_path, 'wb') as outfile:
                    for i in range(chunk_count):
                        chunk_path = os.path.join(temp_dir, str(i))
                        if not os.path.exists(chunk_path):
                            return jsonify({"success": False, "message": f"缺失分片 {i}"}), 400
                        with open(chunk_path, 'rb') as infile:
                            shutil.copyfileobj(infile, outfile)
                
                with open(merged_file_path, 'rb') as f:
                    content_data = f.read()
                
                shutil.rmtree(temp_dir)
                
                try:
                    # 尝试GZIP解压
                    try:
                        json_data = gzip.decompress(content_data)
                        request_data = json.loads(json_data)
                    except:
                        request_data = json.loads(content_data)
                except Exception as e:
                     return jsonify({"success": False, "message": f"数据解析失败: {e}"}), 400
                
                return self._handle_sync_request_data(request_data, request.headers.get("X-Secret-Key"))
                
            except Exception as e:
                print(f"合并分片失败: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({"success": False, "message": str(e)}), 500
        
        @app.route('/ai/v1/sync/pull', methods=['GET'])
        def pull_data():
            """处理终端拉取数据请求"""
            try:
                # 提取身份信息
                tenant_id = request.args.get("tenant_id")
                secret_key = request.headers.get("X-Secret-Key")
                local_version = int(request.args.get("local_version", 0))
                
                # 身份认证
                auth_success, auth_msg = self._authenticate_tenant(tenant_id, secret_key)
                if not auth_success:
                    return jsonify({"success": False, "message": auth_msg}), 403
                
                # 获取当前租户数据
                tenant_data = self._get_tenant_data(tenant_id)
                
                # 检查是否有更新
                if tenant_data["version"] <= local_version:
                    return jsonify({
                        "success": True,
                        "has_update": False,
                        "message": "当前已是最新版本"
                    })
                
                # 准备返回数据
                response_data = {
                    "success": True,
                    "has_update": True,
                    "version": tenant_data["version"],
                    "data": {
                        "type": "full",
                        "features": tenant_data["data"]
                    },
                    "timestamp": time.time()
                }
                
                return jsonify(response_data)
            
            except Exception as e:
                print(f"处理拉取请求失败：{e}")
                return jsonify({"success": False, "message": str(e)}), 500
        
        @app.route('/ai/v1/sync/check-update', methods=['GET'])
        def check_update():
            """检查是否有更新"""
            try:
                # 提取身份信息
                tenant_id = request.args.get("tenant_id")
                secret_key = request.headers.get("X-Secret-Key")
                local_version = int(request.args.get("local_version", 0))
                
                # 身份认证
                auth_success, auth_msg = self._authenticate_tenant(tenant_id, secret_key)
                if not auth_success:
                    return jsonify({"success": False, "message": auth_msg}), 403
                
                # 获取当前租户数据
                tenant_data = self._get_tenant_data(tenant_id)
                
                return jsonify({
                    "success": True,
                    "has_update": tenant_data["version"] > local_version,
                    "latest_version": tenant_data["version"],
                    "local_version": local_version
                })
            
            except Exception as e:
                print(f"处理检查更新请求失败：{e}")
                return jsonify({"success": False, "message": str(e)}), 500

# 启动服务
if __name__ == '__main__':
    server = SyncCenterServer()
    server.running = True
    server._start_backup_thread()
    server._register_routes()
    app.run(
            host='0.0.0.0',
            port=server.config["server"]["port"],
            debug=server.config["server"]["debug"]
        )