#!/usr/bin/env python3
"""
更新PLU码映射脚本
"""

import os
import pickle
import json

# 读取plu_mapping.json文件
plu_mapping_path = "data/plu_mapping.json"
if not os.path.exists(plu_mapping_path):
    print(f"❌ 未找到 {plu_mapping_path} 文件")
    exit(1)

# 读取JSON文件
with open(plu_mapping_path, "r", encoding="utf-8") as f:
    plu_mapping = json.load(f)

# 提取name_to_plu映射
name_to_plu = plu_mapping.get("name_to_plu", {})

# 保存为pickle格式
plu_map_path = "data/plu_map.pkl"
with open(plu_map_path, "wb") as f:
    pickle.dump(name_to_plu, f)

print(f"✅ 已更新PLU码映射文件：{plu_map_path}")
print(f"包含 {len(name_to_plu)} 种药材的PLU码映射")
print("\nPLU码映射内容：")
for herb_name, plu_code in name_to_plu.items():
    print(f"  {herb_name} -> {plu_code}")
