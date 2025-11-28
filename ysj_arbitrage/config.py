# config.py

import os

# ADB 路径 (用户指定)
ADB_PATH = r"D:\scrcpy\adb.exe"

# 目录配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGES_DIR = os.path.join(BASE_DIR, "input_images")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DB_PATH = os.path.join(BASE_DIR, "phones.db")
RESULTS_DIR = os.path.join(BASE_DIR, "results")  # 保存识别结果

# 利润阈值 (元)
PROFIT_THRESHOLD_HIGH = 200
PROFIT_THRESHOLD_LOW = 50

# 价格上浮比例 (含配件)
PRICE_MARKUP = 1.03

# 闲鱼包名
XIANYU_PACKAGE = "com.taobao.idlefish"

