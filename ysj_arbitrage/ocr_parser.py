from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
import os
import tempfile
import time
from datetime import datetime
from .db_manager import insert_or_update_price
from .config import INPUT_IMAGES_DIR, PRICE_MARKUP

# 机型纠错映射：尽量覆盖常见 OCR 误识别
MODEL_CORRECTIONS = {
    "OPPO": [
        (r'(?:reno|no)\s*14\s*pro\+', "Reno14 Pro+"),
        (r'(?:reno|no)\s*14\s*pro', "Reno14 Pro"),
        (r'(?:reno|no)\s*14', "Reno14"),
        (r'(?:reno|no)\s*15\s*pro', "Reno15 Pro"),
        (r'(?:reno|no)\s*15', "Reno15"),
        (r'pkz110', "Reno14 Pro"),
        (r'pla110', "Reno14"),
    ],
    "HUAWEI": [
        (r'p70u|p70ultra|pura70u', "Pura70 Ultra"),
        (r'p70pro\+', "Pura70 Pro+"),
        (r'p70pro', "Pura70 Pro"),
        (r'p70', "Pura70"),
    ],
    "GLOBAL": [
        (r'pura70ultra', "Pura70 Ultra"),
        (r'pura70pro\+', "Pura70 Pro+"),
        (r'pura70pro', "Pura70 Pro"),
        (r'pura70', "Pura70"),
    ],
}

CONDITION_COLUMNS = ['充新', '靓机', '小花', '大花', '外爆', '内爆']
# 只使用充新价格
TARGET_CONDITION = '充新'

class QuoteParser:
    def __init__(self):
        # 初始化OCR模型，使用 PP-OCRv5_server 模型（与 logic_manager 中的 table_ocr 保持一致）
        # use_gpu=False 默认，如果安装了 gpu 版本可以设置为 True
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            ocr_version='PP-OCRv5'
        )

    def _preprocess_table_image(self, image_path):
        """
        对报价单表格图片进行预处理，提升 OCR 识别率
        返回预处理后的图片路径（可能是临时文件）
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"[Preprocess] Failed to read image: {image_path}")
                return image_path
            
            height, width = img.shape[:2]
            
            # 1. 裁剪掉可能的边缘干扰（保留顶部，因为表头在顶部）
            # 不裁剪顶部，确保表头完整
            crop_bottom = int(height * 0.98)  # 只裁剪底部2%
            img = img[:crop_bottom, :]
            
            # 2. 增强对比度（表格通常需要更强的对比度）
            # 使用 LAB 色彩空间增强亮度
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            # 对亮度通道进行 CLAHE（限制对比度自适应直方图均衡化）
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # 3. 轻微锐化（提升文字边缘清晰度）
            sharpen_kernel = np.array([[0, -1, 0],
                                       [-1, 5, -1],
                                       [0, -1, 0]])
            enhanced = cv2.filter2D(enhanced, -1, sharpen_kernel)
            
            # 4. 如果图片太小，适当放大（表头文字通常较小）
            if width < 1200:
                scale = 1200 / width
                new_width = int(width * scale)
                new_height = int(enhanced.shape[0] * scale)
                enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # 5. 保存预处理后的图片到临时文件
            fd, temp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            cv2.imwrite(temp_path, enhanced)
            print(f"[Preprocess] 预处理完成，保存到: {temp_path}")
            return temp_path
            
        except Exception as e:
            print(f"[Preprocess] 预处理失败: {e}")
            import traceback
            traceback.print_exc()
            return image_path  # 失败时返回原图

    def parse_image(self, image_path, brand=None, stop_event=None, *, batch_id=None, source="ocr", date_key=None):
        """解析报价单图片并入库 (重构版：支持复杂表格)"""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return

        # 检查停止信号
        if stop_event and stop_event.is_set():
            print("OCR stopped by user.")
            return

        print(f"Starting OCR for {image_path}...")
        if not batch_id:
            batch_id = f"ocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not date_key:
            date_key = datetime.now().strftime('%Y-%m-%d')
        if not source:
            source = "ocr"
        
        # 预处理图片
        preprocessed_path = self._preprocess_table_image(image_path)
        temp_file_created = (preprocessed_path != image_path)
        
        try:
            # 使用预处理后的图片进行 OCR
            result = self.ocr.ocr(preprocessed_path)
        except Exception as e:
            print(f"OCR Error: {e}")
            return
        finally:
            # 清理临时文件
            if temp_file_created and os.path.exists(preprocessed_path):
                try:
                    os.remove(preprocessed_path)
                    print(f"[Preprocess] 已清理临时文件: {preprocessed_path}")
                except Exception as e:
                    print(f"[Preprocess] 清理临时文件失败: {e}")
        
        if not result or not result[0]:
            print("No text found.")
            return

        # result[0] 是一个列表，每个元素是 [box, (text, confidence)]
        # box 是 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        raw_data = result[0]
        print(f"OCR raw block count: {len(raw_data)}")
        print(f"OCR raw data type: {type(raw_data)}")
        if isinstance(raw_data, dict):
            print(f"OCR raw data keys: {list(raw_data.keys())}")
            for key in ['rec_texts', 'rec_scores', 'rec_polys', 'dt_polys', 'rec_boxes']:
                if key in raw_data:
                    value = raw_data[key]
                    try:
                        length = len(value)
                    except Exception:
                        length = 'N/A'
                    sample = value[0] if hasattr(value, '__getitem__') and length not in ['N/A', 0] else None
                    print(f"  {key} -> type: {type(value)}, len: {length}, sample: {sample}")
        elif isinstance(raw_data, list) and raw_data:
            print(f"OCR first entry sample: {raw_data[0]}")
        else:
            attr_list = [attr for attr in dir(raw_data) if not attr.startswith("_")]
            print(f"OCR raw data attrs: {attr_list}")
            for attr in ['rec_texts', 'rec_scores', 'rec_polys', 'dt_polys', 'rec_boxes']:
                value = getattr(raw_data, attr, None)
                if value is not None:
                    try:
                        length = len(value)
                    except Exception:
                        length = 'N/A'
                    sample = value[0] if hasattr(value, '__getitem__') and length not in ['N/A', 0] else None
                    print(f"  attr {attr} -> type: {type(value)}, len: {length}, sample: {sample}")
        
        normalized_blocks = self.normalize_raw_blocks(raw_data)
        print(f"Normalized block count: {len(normalized_blocks)}")
        if not normalized_blocks:
            print("No normalized OCR blocks available.")
            return
        
        # 1. 提取所有文本块并按 Y 坐标聚类成行
        lines = self.sort_and_group_lines(normalized_blocks)
        print(f"Grouped lines: {len(lines)}")
        
        # 2. 解析表格结构
        # 尝试找到表头行（包含“容量”、“靓机”等关键词）
        header_mapping = self.identify_headers(lines)
        print(f"Header mapping result: {header_mapping}")

        # 根据所有文本块构建型号区域（用于根据 Y 轴位置推断行所属型号）
        model_regions = self._build_model_regions(normalized_blocks, brand)
        
        # 3. 遍历每一行提取数据
        current_model = ""
        
        for idx, line_blocks in enumerate(lines):
            if stop_event and stop_event.is_set(): return

            # 这一行的所有文本内容
            row_text = " ".join([b['text'] for b in line_blocks])
            print(f"[Line {idx}] {row_text}")
            
            # 跳过表头行本身（支持多种表头格式）
            # 表头特征：第一行且不包含配置格式，或者包含表头关键词
            is_first_line = idx == 0
            has_config_format = bool(re.search(r'\d+\+\d+', row_text))
            has_header_keywords = "容量" in row_text or "靓机" in row_text or "充新" in row_text or \
                                 "Reno系列" in row_text or "系列/型号" in row_text or "网络型号" in row_text
            
            if (is_first_line and not has_config_format and len(line_blocks) >= 5) or \
               (has_header_keywords and not has_config_format):
                print(f"[Line {idx}] 跳过表头行: {row_text}")
                continue
            
            # 计算当前行的垂直范围
            row_y_min, row_y_max, row_center = self._get_row_bounds(line_blocks)

            # 根据预先扫描的“型号区域”判断本行所属型号
            region_model = self._find_model_from_regions(row_center, model_regions)
            if region_model and region_model != current_model:
                print(f"[Line {idx}] 区域匹配型号 -> {region_model} (row_center={row_center})")
                current_model = region_model
            elif region_model:
                current_model = region_model

            # 使用正则快速匹配型号（兜底逻辑，防止 OCR 漏掉关键字）
            regex_model = self._match_model_name(row_text)
            if regex_model:
                current_model = regex_model
                print(f"[Line {idx}] Regex 匹配型号 -> {current_model}")

            # 根据 X 坐标判断列归属
            row_data = self.map_blocks_to_columns(line_blocks, header_mapping)
            
            # 更新型号：更准确地识别型号，区分 Reno15 和 Reno15 Pro
            # 型号特征：包含 "Reno"、"Pro" 等关键词，且不包含配置格式
            # 检查整行文本，寻找型号信息
            row_text_lower = row_text.lower()
            
            # 检查是否包含型号关键词（支持 OCR 识别不完整的情况，如 "eno" 代替 "reno"）
            has_model_keywords = any(kw in row_text_lower for kw in ['reno', 'eno', 'pro', '5g', 'plv', 'plw'])
            
            # 检查第一列是否是型号（不包含配置格式，不是纯数字价格）
            first_block = line_blocks[0]
            first_text = first_block['text']
            is_config = bool(re.search(r'\d+\+\d+', first_text))
            is_price = bool(re.search(r'^\d{3,5}$', first_text))
            
            print(f"[Line {idx}] 型号识别检查: has_model_keywords={has_model_keywords}, is_config={is_config}, first_text='{first_text}'")
            
            # 如果第一列看起来像型号，或者整行包含型号关键词但没有配置格式
            if has_model_keywords and not is_config:
                # 尝试从整行提取型号（可能包含多个文本块，如 "Reno15 Pro 5G"）
                # 优先从第一列提取，如果第一列是型号的一部分
                potential_model = ""
                
                # 如果第一列不是配置、不是价格，且看起来像型号，则可能是型号
                # 放宽条件：只要第一列不包含配置格式、不是纯数字、不是表头关键词，就可能是型号
                if not is_config and not is_price and len(first_text) > 2:
                    if "系列" not in first_text and "型号" not in first_text and "等级" not in first_text:
                        # 检查第一列是否包含型号特征（数字+字母的组合，如 "eno15Pro5G"）
                        first_lower = first_text.lower()
                        # 如果包含数字和字母，或者包含型号关键词，就认为是型号
                        has_digit = bool(re.search(r'\d', first_text))
                        has_letter = bool(re.search(r'[a-z]', first_lower))
                        has_model_kw = any(kw in first_lower for kw in ['reno', 'eno', 'pro', '5g', 'plv', 'plw'])
                        
                        if (has_digit and has_letter) or has_model_kw:
                            potential_model = first_text
                            print(f"[Line {idx}] 从第一列提取型号: '{potential_model}'")
                
                # 如果第一列不是型号，尝试从整行提取（可能型号被分割成多个块）
                if not potential_model:
                    # 收集所有非配置、非价格的文本块（在配置之前的所有文本块）
                    model_parts = []
                    config_found = False
                    for block in line_blocks:
                        text = block['text']
                        # 如果遇到配置格式，停止收集型号部分
                        if re.search(r'\d+\+\d+', text):
                            config_found = True
                            break
                        # 如果遇到价格数字，也停止（价格在配置之后）
                        if re.search(r'^\d{3,5}$', text):
                            break
                        if "系列" not in text and "型号" not in text and "等级" not in text:
                            # 检查是否包含型号关键词（支持 OCR 识别不完整的情况）
                            if any(kw in text.lower() for kw in ['reno', 'eno', 'pro', '5g', 'plv', 'plw']):
                                model_parts.append(text)
                    
                    if model_parts:
                        # 合并型号部分（通常型号在配置之前）
                        # 合并所有部分，确保能识别完整的型号（如 "Reno15 Pro 5G"）
                        potential_model = " ".join(model_parts)
                
                # 清理和规范化型号文本
                if potential_model:
                    print(f"[Line {idx}] 提取到 potential_model: '{potential_model}'")
                    # 移除可能的网络型号（如 PLV110、PLW110），这些不是型号的一部分
                    potential_model = re.sub(r'\b(PLV|PLW)\d+\b', '', potential_model, flags=re.IGNORECASE)
                    potential_model = re.sub(r'\s+', ' ', potential_model).strip()
                    print(f"[Line {idx}] 清理后 potential_model: '{potential_model}'")
                    
                    # 如果清理后变成空字符串，说明可能只包含网络型号，尝试从原始文本重新提取
                    if not potential_model:
                        print(f"[Line {idx}] 警告：清理后 potential_model 为空，尝试从第一列重新提取")
                        if not is_config and not is_price and len(first_text) > 2:
                            if "系列" not in first_text and "型号" not in first_text and "等级" not in first_text:
                                potential_model = first_text
                                print(f"[Line {idx}] 从第一列重新提取: '{potential_model}'")
                    
                    # 关键：检查是否明确包含 "Pro"
                    # 如果合并后的文本包含 "pro"，且包含 "reno15" 或 "eno15"（OCR可能漏掉R），则应该是 "Reno15 Pro"
                    # 否则，如果只包含 "reno15" 或 "eno15"，则是 "Reno15"
                    model_lower = potential_model.lower()
                    has_pro = 'pro' in model_lower
                    # 放宽条件：支持 "reno15" 或 "eno15"（OCR可能漏掉开头的R）
                    has_reno15 = ('reno' in model_lower or 'eno' in model_lower) and '15' in model_lower
                    
                    print(f"[Line {idx}] 型号检查: has_pro={has_pro}, has_reno15={has_reno15}, model_lower='{model_lower}'")
                    
                    if has_reno15:
                        if has_pro:
                            # 明确是 Reno15 Pro
                            current_model = potential_model
                        else:
                            # 明确是 Reno15（不包含 Pro）
                            current_model = potential_model
                        print(f"[Line {idx}] Updated current_model -> {current_model} (from row: {row_text}, has_pro={has_pro})")
                    else:
                        # 即使没有明确匹配到 reno15，如果包含型号关键词，也尝试更新（可能是其他型号）
                        if any(kw in model_lower for kw in ['reno', 'eno', 'pro', '15']):
                            current_model = potential_model
                            print(f"[Line {idx}] Updated current_model -> {current_model} (from row: {row_text}, fallback match)")
                        else:
                            print(f"[Line {idx}] potential_model '{potential_model}' 不包含型号关键词，不更新 current_model")
                else:
                    print(f"[Line {idx}] 未提取到 potential_model")
            
            # 提取配置
            config = ""
            config_idx = -1
            # 优先看是否有一列被映射为 "容量"
            if row_data.get('容量'):
                config = row_data.get('容量')
                config_idx = 0  # 如果来自列映射，默认认为在最左侧
            else:
                # 否则正则查找
                for idx_block, block in enumerate(line_blocks):
                    if re.search(r'\d+\+\d+', block['text']):
                        config = block['text']
                        config_idx = idx_block
                        break

            # 如果仍然没有识别到配置，尝试识别 “16T / 14t / 16  T” 这类缺少 “+1” 的格式
            if not config:
                for idx_block, block in enumerate(line_blocks):
                    fallback_match = re.search(r'(\d{2})\s*(?:t|T)\b', block['text'])
                    if fallback_match:
                        config = f"{fallback_match.group(1)}+1T"
                        config_idx = idx_block
                        print(f"[Line {idx}] 配置兜底：从 '{block['text']}' 识别为 '{config}'")
                        break
                else:
                    # 如果逐块未识别，再尝试整行兜底
                    fallback_text = " ".join(b['text'] for b in line_blocks)
                    fallback_match = re.search(r'(\d{2})\s*(?:t|T)\b', fallback_text)
                    if fallback_match:
                        config = f"{fallback_match.group(1)}+1T"
                        config_idx = 0
                        print(f"[Line {idx}] 配置兜底（整行）：从 '{fallback_text}' 识别为 '{config}'")
            
            # 如果这一行没有配置，也没有价格，可能只是一个分类标题行，跳过
            # 但是，如果这行更新了 current_model，说明它是型号行，应该保留 current_model 并跳过
            has_any_condition_price = any(row_data.get(col) for col in CONDITION_COLUMNS)
            if not config and not has_any_condition_price:
                # 检查这行是否更新了 current_model（通过检查是否包含型号关键词）
                row_text_lower = row_text.lower()
                has_model_keywords = any(kw in row_text_lower for kw in ['reno', 'eno', 'pro', '5g', 'plv', 'plw'])
                if has_model_keywords:
                    # 这可能是型号行，已经更新了 current_model，跳过即可
                    print(f"[Line {idx}] Skip: 型号行（无配置/价格）, current_model={current_model}")
                else:
                    # 不是型号行，也没有配置和价格，跳过
                    print(f"[Line {idx}] Skip: no config/price, current_model={current_model}")
                continue
                
            # 只提取"充新"价格（配置后的第一个价格列）
            chongxin_price = None
            price_text = row_data.get(TARGET_CONDITION)
            if price_text:
                normalized_price = self.extract_price_value(price_text)
                if normalized_price:
                    chongxin_price = normalized_price
            
            # 如果表头映射失败，尝试从配置后的第一个数字提取（充新价格）
            if not chongxin_price and config:
                # 配置后的第一个数字就是充新价格
                if config_idx >= 0:
                    # 先在配置所在的块之后寻找
                    if config_idx + 1 < len(line_blocks):
                        for i in range(config_idx + 1, len(line_blocks)):
                            text = line_blocks[i]['text']
                            price_match = re.search(r'(\d{3,4})', text)
                            if price_match:
                                price_value = price_match.group(1)
                                price_int = int(price_value)
                                if 1000 <= price_int <= 9999:
                                    chongxin_price = price_value
                                    print(f"[Line {idx}] 从配置后提取充新价格: {chongxin_price} (来自文本: {text})")
                                    break
                    # 如果配置所在块已经包含价格（例如 16T 2700 2600...），在同一块内查找
                    if not chongxin_price:
                        text = line_blocks[config_idx]['text']
                        candidates = re.findall(r'(\d{3,4})', text)
                        for candidate in candidates:
                            price_int = int(candidate)
                            if price_int >= 100:  # 跳过配置中的 "16"
                                chongxin_price = candidate
                                print(f"[Line {idx}] 从配置所在块提取充新价格: {chongxin_price} (来自文本: {text})")
                                break
            
            # 最终入库：只保存充新价格，并应用价格上浮
            if current_model and config and chongxin_price:
                model_clean = self.normalize_model_text(brand, current_model)
                if not model_clean:
                    print(f"[Line {idx}] Skip insert: unable to normalize model '{current_model}'")
                    continue
                
                # 验证价格合理性（充新价格应该在 1000-10000 之间）
                try:
                    price_float = float(chongxin_price)
                    if price_float < 1000 or price_float > 10000:
                        print(f"[Line {idx}] 价格不合理，跳过: {chongxin_price}")
                        continue
                except ValueError:
                    print(f"[Line {idx}] 价格格式错误，跳过: {chongxin_price}")
                    continue
                
                # 直接传充新价格给数据库函数，让 db_manager 统一计算收货价（避免双重计算）
                print(f"Parsed Table: {brand} | {model_clean} | {config} | {TARGET_CONDITION} | 充新价={chongxin_price}")
                # 传入充新价格，db_manager 会计算：充新价 × 1.03 = 收货价
                line_confidence = None
                if line_blocks:
                    confidences = [block.get('confidence') for block in line_blocks if block.get('confidence') is not None]
                    if confidences:
                        line_confidence = max(confidences)
                insert_or_update_price(
                    brand,
                    model_clean,
                    config,
                    TARGET_CONDITION,
                    price_float,
                    source=source,
                    batch_id=batch_id,
                    date_key=date_key,
                    confidence=line_confidence,
                    status="active",
                    raw_price=price_float,
                    markup_factor=PRICE_MARKUP,
                )
            else:
                missing_info = []
                if not current_model:
                    missing_info.append("model")
                if not config:
                    missing_info.append("config")
                if not chongxin_price:
                    missing_info.append("充新价格")
                print(f"[Line {idx}] Missing data -> {', '.join(missing_info)}")

    def normalize_raw_blocks(self, raw_data):
        """
        兼容 PaddleOCR 不同版本的输出，将其转换为统一结构:
        [{'text': str, 'box': [[x1,y1],...], 'confidence': float}, ...]
        """
        blocks = []
        if isinstance(raw_data, list):
            for item in raw_data:
                if not item or len(item) < 2:
                    continue
                box = item[0]
                text = item[1][0]
                score = item[1][1] if len(item[1]) > 1 else 1.0
                quad = self._normalize_box(box)
                if quad:
                    blocks.append({'text': text, 'box': quad, 'confidence': score})
        else:
            rec_texts = self._get_field(raw_data, 'rec_texts')
            rec_scores = self._get_field(raw_data, 'rec_scores')
            rec_boxes = self._get_field(raw_data, 'rec_polys')
            if rec_boxes is None:
                rec_boxes = self._get_field(raw_data, 'dt_polys')
            if rec_boxes is None:
                rec_boxes = self._get_field(raw_data, 'rec_boxes')

            rec_texts = self._to_list(rec_texts)
            rec_scores = self._to_list(rec_scores)
            rec_boxes = self._to_list(rec_boxes)

            if rec_texts and rec_boxes:
                min_len = min(len(rec_texts), len(rec_boxes))
                if len(rec_texts) != len(rec_boxes):
                    print(f"Warning: rec_texts({len(rec_texts)}) != rec_boxes({len(rec_boxes)}), truncating to {min_len}")
                for idx in range(min_len):
                    text = rec_texts[idx]
                    if isinstance(text, (list, tuple)):
                        text = "".join([str(t) for t in text])
                    box = rec_boxes[idx]
                    quad = self._normalize_box(box)
                    if not quad:
                        continue
                    score = rec_scores[idx] if rec_scores and idx < len(rec_scores) else 1.0
                    try:
                        score = float(score)
                    except Exception:
                        score = 1.0
                    blocks.append({'text': str(text), 'box': quad, 'confidence': score})
        return blocks

    def _get_field(self, obj, key):
        """从对象或字典中获取字段"""
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key)
        return None

    def _to_list(self, value):
        """将 numpy 数组或其它结构转为 list"""
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return value

    def _normalize_box(self, box):
        """将多种 box 表示转换为四点坐标"""
        if box is None:
            return None
        if isinstance(box, np.ndarray):
            box = box.tolist()
        if isinstance(box, (list, tuple)):
            if len(box) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in box):
                return [[float(pt[0]), float(pt[1])] for pt in box]
            if len(box) == 8:
                return [
                    [float(box[0]), float(box[1])],
                    [float(box[2]), float(box[3])],
                    [float(box[4]), float(box[5])],
                    [float(box[6]), float(box[7])],
                ]
        return None

    def sort_and_group_lines(self, raw_blocks, y_threshold=15):
        """
        根据 Y 坐标将文本块聚类成行。
        raw_blocks: List of {'text','box','confidence'}
        y_threshold: 同一行的高度容差
        """
        # 按 Y 坐标排序
        sorted_data = sorted(raw_blocks, key=lambda b: b['box'][0][1] if b['box'] else 0)
        
        lines = []
        current_line = []
        current_y = -100
        
        for item in sorted_data:
            box = item['box']
            text = item['text']
            y = box[0][1] if box else 0
            x = box[0][0] if box else 0
            
            # 计算中心 Y 坐标可能更准，这里简单用 top Y
            center_y = (box[0][1] + box[3][1]) / 2 if box else y
            
            if not current_line:
                current_line.append({'text': text, 'x': x, 'y': center_y, 'box': box})
                current_y = center_y
            else:
                if abs(center_y - current_y) < y_threshold:
                    current_line.append({'text': text, 'x': x, 'y': center_y, 'box': box})
                else:
                    # 新的一行
                    # 对上一行按 X 排序
                    current_line.sort(key=lambda b: b['x'])
                    lines.append(current_line)
                    
                    current_line = [{'text': text, 'x': x, 'y': center_y, 'box': box}]
                    current_y = center_y
        
        if current_line:
            current_line.sort(key=lambda b: b['x'])
            lines.append(current_line)
            
        return lines

    def identify_headers(self, lines):
        """
        识别表头，返回各列的 X 坐标范围
        返回: {'充新': (x_start, x_end), '容量': ...}
        """
        mapping = {}
        print("Identifying headers...") # 调试日志
        
        # 尝试识别表头行：第一行通常是表头，即使 OCR 识别错误
        # 表头特征：包含多个文本块，且不包含配置格式（如 12+256）
        for line_idx, line in enumerate(lines):
            row_text = " ".join([b['text'] for b in line])
            print(f"Checking header line {line_idx}: {row_text}") # 调试日志
            
            # 检查是否是表头行：
            # 1. 包含关键词
            # 2. 或者第一行且包含多个文本块（通常是表头）
            # 3. 不包含配置格式（如 12+256）
            has_keywords = "容量" in row_text or "靓机" in row_text or "充新" in row_text or "小花" in row_text or \
                          "Reno系列" in row_text or "系列/型号" in row_text or "网络型号" in row_text
            has_config_format = bool(re.search(r'\d+\+\d+', row_text))
            is_first_line = line_idx == 0
            
            if has_keywords or (is_first_line and len(line) >= 5 and not has_config_format):
                print(f"Header line found (line {line_idx}): {row_text}")
                # 这是一个表头行
                for i, block in enumerate(line):
                    text = block['text']
                    normalized = self.normalize_header_text(text)
                    if not normalized:
                        continue
                    # 跳过"型号列"和"网络型号"的映射（这些列不需要提取价格）
                    if normalized in ['型号列', '网络型号']:
                        continue
                    # 记录每一列的中心点或大致范围
                    x_start = block['box'][0][0]
                    x_end = block['box'][1][0]
                    
                    # 扩展一点范围
                    mapping[normalized] = (x_start - 10, x_end + 10)
                    
                    # 处理包含多个列名的文本块（如果 OCR 把它们识别在一起）
                    # 例如 "充新 靓机"
                    
        # 如果没找到具体表头，打印警告
        if not mapping:
            print("Warning: No headers found! Will use fallback price extraction.")
        else:
            print(f"Found headers: {list(mapping.keys())}")
            
        return mapping

    def normalize_header_text(self, text):
        """
        将 OCR 识别出来的列标题映射为规范字段名
        """
        if not text:
            return None
        cleaned = text.replace(" ", "").replace("：", "").replace("价", "")
        header_map = {
            '容量': ['容量', '配置', '内存'],
            '靓机': ['靓机', '靚机', '靓機', '靓机（原保）', '靓机（无原保）', '靓机原保', '靓机无原保'],
            '充新': ['充新', '全新', '准新'],
            '小花': ['小花', '小划', '小瑕', '小花机'],
            '大花': ['大花', '大划', '大瑕', '大花机'],
            '外爆': ['外爆', '外爆可测'],
            '内爆': ['内爆', '内爆可测', '内暴'],
            '型号列': ['ren系列', 'reno系列', '系列/型号', '系列型号', '型号'],
            '网络型号': ['网络型号', '网络', '型号'],
        }
        for canonical, keywords in header_map.items():
            for kw in keywords:
                if kw in cleaned.lower() or kw in cleaned:
                    return canonical
        return None

    def map_blocks_to_columns(self, line_blocks, header_mapping):
        """
        将一行的文本块映射到对应的列名
        """
        row_data = {}
        if not header_mapping:
            return row_data
            
        for block in line_blocks:
            x_center = (block['box'][0][0] + block['box'][1][0]) / 2
            text = block['text']
            
            # 检查该块落在哪个列的范围内
            matched_col = None
            min_dist = float('inf')
            
            for col_name, (x_min, x_max) in header_mapping.items():
                # 简单的范围检查
                # 或者计算到列中心的距离
                col_center = (x_min + x_max) / 2
                dist = abs(x_center - col_center)
                
                # 如果在范围内，或者距离很近
                # 这里简化：如果在范围内就直接命中
                if x_min <= x_center <= x_max:
                    matched_col = col_name
                    break
                
                # 记录最近的列，防止稍微偏离
                if dist < min_dist and dist < 50: # 50像素容差
                    min_dist = dist
                    matched_col = col_name
            
            if matched_col:
                row_data[matched_col] = text
                
        return row_data

    def normalize_model_text(self, brand, raw_model):
        """
        对 OCR 识别出的型号进行纠错，尽量映射到标准型号名称
        关键：准确区分 Reno15 和 Reno15 Pro
        """
        if not raw_model:
            return ""
        cleaned = raw_model
        cleaned = cleaned.replace("（", "(").replace("）", ")")
        cleaned = re.sub(r'[，,、·]', ' ', cleaned)
        # 先检查是否包含 Pro，再移除 5G（避免误判）
        has_pro = 'pro' in cleaned.lower()
        cleaned = re.sub(r'(?i)5g', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        normalized = re.sub(r'[^a-z0-9+]+', '', cleaned.lower())
        
        brand_key = (brand or "GLOBAL").upper()
        candidates = MODEL_CORRECTIONS.get(brand_key, []) + MODEL_CORRECTIONS.get("GLOBAL", [])
        
        # 优先匹配更具体的模式（Pro 版本优先）
        # 先匹配 Pro 版本，再匹配普通版本
        pro_patterns = [(p, t) for p, t in candidates if 'pro' in t.lower()]
        normal_patterns = [(p, t) for p, t in candidates if 'pro' not in t.lower()]
        
        # 如果原始文本包含 Pro，优先匹配 Pro 模式
        if has_pro:
            for pattern, target in pro_patterns:
                if re.search(pattern, normalized):
                    print(f"Model normalized: '{raw_model}' -> '{target}' (matched Pro pattern)")
                    return target
            # 如果 Pro 模式没匹配到，但文本明确包含 Pro，尝试直接构造
            if 'reno' in normalized and '15' in normalized:
                print(f"Model normalized: '{raw_model}' -> 'Reno15 Pro' (has Pro keyword)")
                return "Reno15 Pro"
        else:
            # 如果原始文本不包含 Pro，优先匹配普通版本
            for pattern, target in normal_patterns:
                if re.search(pattern, normalized):
                    print(f"Model normalized: '{raw_model}' -> '{target}' (matched normal pattern)")
                    return target
            # 如果普通模式没匹配到，但文本不包含 Pro，尝试直接构造
            if 'reno' in normalized and '15' in normalized:
                print(f"Model normalized: '{raw_model}' -> 'Reno15' (no Pro keyword)")
                return "Reno15"
        
        # 如果都没匹配到，尝试所有模式
        for pattern, target in candidates:
            if re.search(pattern, normalized):
                print(f"Model normalized: '{raw_model}' -> '{target}' (fallback match)")
                return target
        
        # 没有匹配到映射时，返回简单的首字母大写结果
        print(f"Model normalized: '{raw_model}' -> '{cleaned}' (no pattern match)")
        return cleaned

    def _build_model_regions(self, blocks, brand):
        """从所有文本块中构建型号区域，便于根据 Y 轴位置推断每一行所属型号"""
        raw_regions = []
        for block in blocks:
            text = block.get('text')
            box = block.get('box')
            if not text or not box:
                continue
            raw_model = self._match_model_name(text)
            if not raw_model:
                continue
            normalized = self.normalize_model_text(brand, raw_model)
            y_values = [pt[1] for pt in box]
            if not y_values:
                continue
            y_min = min(y_values)
            y_max = max(y_values)
            center = (y_min + y_max) / 2
            height = max(5, y_max - y_min)
            raw_regions.append({
                'raw': raw_model,
                'model': normalized,
                'center': center,
                'height': height
            })
            print(f"[ModelRegion] '{text}' -> {normalized}, center={center:.1f}, height={height:.1f}")

        if not raw_regions:
            return []

        # 根据中心位置排序，并为每个区域计算上下界（与前后区域中心点的中点）
        raw_regions.sort(key=lambda r: r['center'])
        regions = []
        for idx, region in enumerate(raw_regions):
            prev_center = raw_regions[idx - 1]['center'] if idx > 0 else None
            next_center = raw_regions[idx + 1]['center'] if idx + 1 < len(raw_regions) else None
            lower = -float('inf') if prev_center is None else (prev_center + region['center']) / 2
            upper = float('inf') if next_center is None else (next_center + region['center']) / 2
            region['lower'] = lower
            region['upper'] = upper
            regions.append(region)
            print(f"[ModelRegionBound] {region['model']} -> range=({lower:.1f}, {upper:.1f})")

        return regions

    def _get_row_bounds(self, line_blocks):
        """计算一行文本块的垂直范围"""
        y_values = []
        for block in line_blocks:
            box = block.get('box')
            if box:
                y_values.extend([pt[1] for pt in box])
        if not y_values:
            return None, None, None
        y_min = min(y_values)
        y_max = max(y_values)
        center = (y_min + y_max) / 2
        return y_min, y_max, center

    def _find_model_from_regions(self, row_center, regions):
        """根据行中心位置匹配最近的型号区域"""
        if row_center is None or not regions:
            return None
        closest = None
        min_dist = float('inf')
        for region in regions:
            dist = abs(row_center - region['center'])
            if dist < min_dist:
                min_dist = dist
                closest = region
        # 优先根据明确的范围匹配
        for region in regions:
            if region['lower'] <= row_center <= region['upper']:
                print(f"[ModelRegionMatch] row_center={row_center:.1f} -> {region['model']} (within [{region['lower']:.1f}, {region['upper']:.1f}])")
                return region['model']

        # 如果不在任何范围内，则退回到最近的区域
        closest_region = min(regions, key=lambda r: abs(row_center - r['center']))
        print(f"[ModelRegionFallback] row_center={row_center:.1f} -> {closest_region['model']} (dist={abs(row_center - closest_region['center']):.1f})")
        return closest_region['model']

    def _match_model_name(self, text):
        """
        使用正则在整行文本中匹配型号（支持 OCR 错误，如缺少 R）
        例如: eno15Pro5G -> Reno15 Pro
        """
        if not text:
            return None
        # 将文本统一成小写，便于匹配
        lowered = text.lower()
        # 预处理：移除非字母数字字符，防止 OCR 的杂讯影响匹配
        normalized = re.sub(r'[^a-z0-9]+', ' ', lowered)
        # 正则匹配：eno/reno + 数字 + 可选的 pro + 可选的 5g（支持没有空格的情况）
        pattern = re.compile(r'(?:re|e)no\s*\d+\s*(?:pro)?(?:\s*5g)?', re.IGNORECASE)
        match = pattern.search(normalized)
        if match:
            raw_model = match.group(0)
            raw_model = raw_model.replace(" ", "")
            # 如果以 eno 开头，说明 OCR 少了 R，补上
            if raw_model.startswith("eno"):
                raw_model = "r" + raw_model
            print(f"[Regex] 在文本 '{text}' 中匹配到型号: '{raw_model}'")
            return raw_model
        return None

    def extract_price_value(self, text):
        """从 OCR 文本中提取价格数字，返回字符串形式的金额（只提取合理的价格：1000-9999）"""
        if not text:
            return None
        cleaned = str(text).replace(" ", "")
        # 提取3-4位数字（价格范围通常是1000-9999）
        match = re.search(r'(\d{3,4})', cleaned)
        if match:
            price_str = match.group(1)
            price_int = int(price_str)
            # 验证是合理的价格（1000-9999）
            if 1000 <= price_int <= 9999:
                return price_str
        return None

    def extract_price_from_line(self, line_blocks):
        """在整行文本块中兜底提取一个价格"""
        for block in line_blocks:
            price_value = self.extract_price_value(block['text'])
            if price_value:
                return price_value
        return None

def process_input_images(stop_event=None):
    """处理 input_images 目录下的所有图片"""
    parser = QuoteParser()
    
    if not os.path.exists(INPUT_IMAGES_DIR):
        os.makedirs(INPUT_IMAGES_DIR)
        print(f"Created directory: {INPUT_IMAGES_DIR}")
        return

    files = os.listdir(INPUT_IMAGES_DIR)
    if not files:
        print(f"No images found in {INPUT_IMAGES_DIR}")
        return

    # 统计信息
    total_files = 0
    processed_files = 0
    total_records = 0
    failed_files = []
    
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_files = len(image_files)
    print(f"[Import] 找到 {total_files} 张图片待处理")

    def _infer_date_key(filename):
        base = os.path.splitext(filename)[0]
        match = re.search(r'(20\d{2})(\d{2})(\d{2})', base)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month}-{day}"
        match = re.search(r'(\d{1,2})月(\d{1,2})', base)
        if match:
            year = datetime.now().strftime('%Y')
            month = int(match.group(1))
            day = int(match.group(2))
            return f"{year}-{month:02d}-{day:02d}"
        return datetime.now().strftime('%Y-%m-%d')
    
    for filename in image_files:
        # 检查停止信号
        if stop_event and stop_event.is_set():
            print("[Import] 处理被用户停止")
            break

        file_path = os.path.join(INPUT_IMAGES_DIR, filename)
        print(f"[Import] 正在处理: {filename}...")
        
        try:
            # 尝试从文件名提取品牌信息
            brand = None
            if "_" in filename:
                possible_brand = filename.split("_")[0]
                if possible_brand.upper() in ["OPPO", "VIVO", "HUAWEI", "HONOR", "XIAOMI"]:
                    brand = possible_brand.upper()
            elif "OPPO" in filename.upper(): brand = "OPPO"
            elif "VIVO" in filename.upper(): brand = "vivo"
            elif "HUAWEI" in filename.upper(): brand = "HUAWEI"
            elif "HONOR" in filename.upper(): brand = "HONOR"
            elif "XIAOMI" in filename.upper(): brand = "XIAOMI"
            
            # 默认品牌
            if not brand: brand = "Unknown"
            
            # 记录处理前的记录数（用于统计）
            from .db_manager import get_all_prices
            records_before = len(get_all_prices())
            
            date_key = _infer_date_key(filename)
            batch_id = f"{os.path.splitext(filename)[0]}_{int(time.time())}"
            parser.parse_image(
                file_path,
                brand,
                stop_event,
                batch_id=batch_id,
                source="ocr",
                date_key=date_key
            )
            
            # 记录处理后的记录数
            records_after = len(get_all_prices())
            records_added = records_after - records_before
            total_records += records_added
            
            if records_added > 0:
                print(f"[Import] {filename} 处理完成，新增 {records_added} 条记录")
            else:
                print(f"[Import] {filename} 处理完成，但未新增记录（可能已存在或识别失败）")
            
            processed_files += 1
            
        except Exception as e:
            import traceback
            print(f"[Import] 处理 {filename} 时出错: {e}")
            traceback.print_exc()
            failed_files.append(filename)
    
    # 打印统计信息
    print(f"[Import] ========== 导入统计 ==========")
    print(f"[Import] 总文件数: {total_files}")
    print(f"[Import] 成功处理: {processed_files}")
    print(f"[Import] 失败文件: {len(failed_files)}")
    if failed_files:
        print(f"[Import] 失败文件列表: {', '.join(failed_files)}")
    print(f"[Import] 新增记录数: {total_records}")
    print(f"[Import] ==============================")

if __name__ == "__main__":
    # 测试代码
    from .db_manager import init_db
    init_db()
    process_input_images()
