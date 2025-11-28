import os
import re
import tempfile
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import subprocess
import sys
import json
from datetime import datetime
from paddleocr import PaddleOCR
from .db_manager import get_price_info
from .config import PROFIT_THRESHOLD_HIGH, PROFIT_THRESHOLD_LOW, RESULTS_DIR

class ArbitrageLogic:
    def __init__(self):
        # server 模型用于报价单 / 表格
        self.table_ocr = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            ocr_version='PP-OCRv5'
        )
        # mobile 模型用于闲鱼截图（使用官方 PP-OCRv3 mobile 模型，首次调用会自动下载到本地缓存）
        self.mobile_ocr = PaddleOCR(
            lang='ch',
            use_angle_cls=False,
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            text_detection_model_name="PP-OCRv3_mobile_det",
            text_recognition_model_name="PP-OCRv3_mobile_rec",
            text_det_box_thresh=0.25,
            text_det_limit_side_len=1536,
            text_det_limit_type="max"
        )

    def analyze_screenshot(self, screenshot_path, target_model, xml_path=None):
        """
        识别闲鱼截图或解析界面层级文本，提取价格，并与数据库对比
        返回一个列表，包含: {'title': str, 'price': float, 'profit': float, 'level': str}
        """
        print(f"Analyzing screenshot: {screenshot_path} for {target_model}")
        lines = []
        line_meta = None
        structured_prices = []

        if xml_path and os.path.exists(xml_path):
            lines, line_meta, structured_prices = self._extract_lines_from_xml(xml_path)
            if lines:
                print(f"[Hierarchy] Extracted {len(lines)} text nodes from {xml_path}")

        if not lines and screenshot_path and os.path.exists(screenshot_path):
            lines = self._run_ocr_with_fallback(screenshot_path)

        if not lines:
            return []

        # 简单的行合并与商品识别逻辑
        # 闲鱼列表页通常是：图片 -> 标题 -> 价格 -> 其他信息
        # 这里简化为：寻找包含 target_model 关键词的文本块，以及其附近的数字
        
        found_items = []
        
        # 获取目标机型的参考收购价 (取最高价作为基准，或者平均价)
        db_prices = get_price_info(target_model)
        # db_prices: [(config, condition, price), ...]
        if not db_prices:
            print(f"No database price found for {target_model}")
            return []
            
        # 简单策略：取所有配置中的最低收购价作为保守参考，或者取最高
        # 为了不错过机会，我们可以取一个合理的基准，例如 "靓机" 的价格
        # 这里简单取最大值，实际逻辑需根据配置匹配
        max_purchase_price = max([p[2] for p in db_prices])
        print(f"[Analyze] 参考收购价: {max_purchase_price}, 目标机型: {target_model}, 识别文本数: {len(lines)}")
        
        # 先提取所有可能的价格（用于调试和备用匹配）
        all_prices = []
        for i, (text, score) in enumerate(lines):
            price_match = re.search(r'[¥￥]?\s*(\d{3,5})', text)
            if price_match:
                possible_price = float(price_match.group(1))
                if possible_price > 50:  # 降低阈值，先收集所有价格
                    all_prices.append((i, possible_price, text))
        
        print(f"[Analyze] 找到 {len(all_prices)} 个可能的价格: {all_prices[:5]}...")  # 只显示前5个
        
        for i, (text, score) in enumerate(lines):
            # 标题匹配（宽松匹配）
            # 将 model 拆分为单词
            keywords = target_model.split()
            match_count = sum(1 for k in keywords if k.lower() in text.lower())
            
            if match_count >= len(keywords) * 0.6: # 匹配度超过 60%
                print(f"[Analyze] 匹配到机型文本 (行{i}): {text[:50]}...")
                # 尝试在当前行或后续几行找价格
                # 闲鱼价格通常以 "¥" 开头，或者纯数字
                price = None

                if structured_prices and line_meta and i < len(line_meta):
                    title_meta = line_meta[i]
                    price_entry = self._match_structured_price(structured_prices, title_meta)
                    if price_entry:
                        price = price_entry["value"]
                        print(f"[Analyze] 使用结构化价格: {price} (来自: {price_entry.get('text','')[:30]})")
                
                if price is None:
                    # 搜索范围：当前行及后5行（扩大搜索范围）
                    search_range = lines[i:min(i+6, len(lines))]
                    for p_text, p_score in search_range:
                        # 匹配价格格式 ¥1234 或 1234
                        # 排除包含 "人想要" 等非价格数字
                        if "人想要" in p_text or "浏览" in p_text or "小时前" in p_text:
                            continue
                            
                        price_match = re.search(r'[¥￥]?\s*(\d{3,5})', p_text)
                        if price_match:
                            possible_price = float(price_match.group(1))
                            # 降低价格阈值，从 500 降到 50
                            if possible_price >= 50:
                                price = possible_price
                                print(f"[Analyze] 找到价格: {price} (来自文本: {p_text[:30]}...)")
                                break
                
                if price:
                    profit = max_purchase_price - price
                    level = "normal"
                    if profit >= PROFIT_THRESHOLD_HIGH:
                        level = "high"
                    elif profit >= PROFIT_THRESHOLD_LOW:
                        level = "medium"
                    else:
                        level = "none"
                    
                    item = {
                        "title": text,
                        "price": price,
                        "ref_price": max_purchase_price,
                        "profit": profit,
                        "level": level
                    }
                    found_items.append(item)
                    print(f"[Analyze] Found potential item: {item}")
                else:
                    print(f"[Analyze] 匹配到机型但未找到价格 (行{i}): {text[:50]}...")
        
        if not found_items:
            print(f"[Analyze] 未找到匹配商品。识别文本示例: {[lines[i][0][:30] for i in range(min(10, len(lines)))]}")
        
        return found_items

    def _run_ocr_with_fallback(self, screenshot_path):
        """直接使用 CLI 执行 OCR，跳过 API 调用以节省时间"""
        print(f"[OCR:cli] 直接使用 CLI 识别: {screenshot_path}")
        cli_lines = self._run_cli_ocr(screenshot_path)
        if cli_lines:
            print(f"[OCR:cli] Success with {len(cli_lines)} text blocks")
            return cli_lines
        return []

    def _extract_lines(self, result):
        """
        将 PaddleOCR 的输出统一为 [(text, score), ...] 形式，兼容 list / OCRResult
        """
        if not result:
            return []
        
        # 新版本可能返回 OCRResult 列表
        first_page = result[0]
        lines = []
        
        if isinstance(first_page, list):
            for entry in first_page:
                if not entry or len(entry) < 2:
                    continue
                text_info = entry[1]
                if isinstance(text_info, (list, tuple)) and text_info:
                    text = text_info[0]
                    score = text_info[1] if len(text_info) > 1 else 1.0
                else:
                    text = str(text_info)
                    score = 1.0
                lines.append((text, float(score)))
            return lines
        
        # PaddleOCR>=3.0 可能返回 OCRResult 对象
        rec_texts = self._to_list(getattr(first_page, "rec_texts", None))
        rec_scores = self._to_list(getattr(first_page, "rec_scores", None))
        
        if rec_texts:
            for idx, text in enumerate(rec_texts):
                score = rec_scores[idx] if rec_scores and idx < len(rec_scores) else 1.0
                lines.append((str(text), float(score)))
        return lines

    def _run_cli_ocr(self, image_path):
        """
        直接调用 paddleocr CLI，从 stdout 解析结果（避免 --save_path 导致的文件冲突）
        """
        cmd = [
            sys.executable, "-u", "-m", "paddleocr", "ocr",  # -u 禁用输出缓冲
            "-i", image_path,
            "--lang", "ch",
            "--text_detection_model_name", "PP-OCRv3_mobile_det",
            "--text_recognition_model_name", "PP-OCRv3_mobile_rec",
            "--text_det_box_thresh", "0.3",
            "--text_det_limit_side_len", "1536"
        ]
        try:
            # 使用 Popen 以便实时获取输出
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # 行缓冲
                universal_newlines=True
            )
            stdout, stderr = proc.communicate(timeout=180)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            print("[OCR:cli] CLI timeout after 180s")
            return []
        except Exception as exc:
            print(f"[OCR:cli] Failed to run CLI: {exc}")
            return []

        output = stdout.strip() if stdout else ""
        stderr_output = stderr.strip() if stderr else ""
        
        # 如果 stdout 为空，检查 stderr（某些情况下输出可能在 stderr）
        if not output and stderr_output:
            # 尝试从 stderr 中提取结果
            output = stderr_output
            print(f"[OCR:cli] Using stderr as output (returncode={returncode})")
        
        if not output:
            if returncode != 0:
                print(f"[OCR:cli] CLI exited with code {returncode}")
                if stderr_output:
                    print(f"[OCR:cli] STDERR (first 500 chars): {stderr_output[:500]}")
            else:
                print("[OCR:cli] No output from CLI (stdout and stderr both empty)")
            return []

        # 从 stdout 中提取 JSON 结果（PaddleOCR CLI 输出 Python dict 格式）
        # 找到最后一个包含 'res' 的字典
        json_start = output.rfind("{'res':")
        if json_start == -1:
            json_start = output.rfind('{"res":')
        if json_start == -1:
            # 尝试解析整个输出
            json_start = 0

        json_str = output[json_start:]
        
        try:
            # 使用 ast.literal_eval 安全解析 Python dict
            import ast
            data = ast.literal_eval(json_str)
        except:
            try:
                # 如果失败，尝试用 eval（因为输出包含 numpy array）
                data = eval(json_str)
            except:
                # 最后尝试正则提取 rec_texts
                import re
                match = re.search(r"'rec_texts':\s*\[(.*?)\]", output, re.DOTALL)
                if match:
                    texts_str = match.group(1)
                    texts = re.findall(r"'([^']+)'", texts_str)
                    lines = [(t.strip(), 1.0) for t in texts if t.strip()]
                    if lines:
                        print(f"[OCR:cli] Extracted {len(lines)} texts via regex fallback")
                        # 保存原始 OCR 文本（regex fallback 模式）
                        self._save_ocr_text(image_path, lines, texts, None)
                        return lines
                print(f"[OCR:cli] Failed to parse output, first 500 chars: {output[:500]}")
                return []

        page = data.get("res", {}) if isinstance(data, dict) else {}
        rec_texts = page.get("rec_texts") or []
        rec_scores = page.get("rec_scores") or []
        
        lines = []
        for idx, text in enumerate(rec_texts):
            text = str(text).strip()
            if not text:
                continue
            score = rec_scores[idx] if rec_scores and idx < len(rec_scores) else 1.0
            # 处理 numpy 类型
            if hasattr(score, 'item'):
                score = score.item()
            lines.append((text, float(score)))
        
        if not lines:
            print("[OCR:cli] No valid text extracted from JSON output")
            return []
        
        # 保存原始 OCR 文本
        self._save_ocr_text(image_path, lines, rec_texts, rec_scores)
        
        return lines
    
    def _save_ocr_text(self, image_path, lines, rec_texts, rec_scores):
        """保存原始 OCR 识别文本到文件"""
        try:
            # 确保结果目录存在
            os.makedirs(RESULTS_DIR, exist_ok=True)
            
            # 生成文件名：基于截图文件名和时间戳
            image_name = os.path.basename(image_path)
            image_base = os.path.splitext(image_name)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            txt_file = os.path.join(RESULTS_DIR, f"ocr_{image_base}_{timestamp}.txt")
            
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(f"OCR识别结果 - {image_name}\n")
                f.write(f"识别时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"识别文本块数量: {len(lines)}\n")
                f.write("=" * 80 + "\n\n")
                
                # 保存所有识别文本（带置信度）
                f.write("识别文本列表（带置信度）:\n")
                f.write("-" * 80 + "\n")
                for idx, (text, score) in enumerate(lines, 1):
                    f.write(f"{idx:3d}. [{score:.3f}] {text}\n")
                
                # 保存原始 rec_texts 和 rec_scores（如果有）
                if rec_texts:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("原始识别结果 (rec_texts):\n")
                    f.write("-" * 80 + "\n")
                    # 处理 rec_texts 可能是列表或字符串的情况
                    if isinstance(rec_texts, list):
                        for idx, text in enumerate(rec_texts, 1):
                            if isinstance(text, str):
                                score = rec_scores[idx-1] if rec_scores and isinstance(rec_scores, (list, tuple)) and idx-1 < len(rec_scores) else 1.0
                                if hasattr(score, 'item'):
                                    score = score.item()
                                f.write(f"{idx:3d}. [{float(score):.3f}] {text}\n")
                            else:
                                f.write(f"{idx:3d}. {text}\n")
            
            print(f"[OCR:cli] 原始文本已保存到: {txt_file}")
        except Exception as e:
            print(f"[OCR:cli] 保存原始文本失败: {e}")
            import traceback
            traceback.print_exc()

    def _to_list(self, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return list(value)
        try:
            # numpy array 等
            return list(value)
        except TypeError:
            return None

    def _extract_lines_from_xml(self, xml_path):
        """从 uiautomator 层级文件提取文本与价格结构化数据"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"[Hierarchy] Failed to parse {xml_path}: {e}")
            return [], None, []

        entries, parent_children = self._collect_xml_entries(root)
        skip_packages = {"com.android.systemui"}
        skip_keywords = ["系统通知", "usb 调试", "蓝牙开启", "nfc", "振铃器", "性能模式"]
        currency_symbols = {"¥", "￥"}

        lines = []
        line_meta = []
        entry_to_line = {}

        for entry in entries:
            text = entry["text"]
            if not text:
                continue
            text = text.strip()
            if not text:
                continue
            pkg = entry["package"]
            if pkg in skip_packages:
                continue
            lower_text = text.lower()
            if any(keyword in lower_text for keyword in skip_keywords):
                continue
            entry_to_line[entry["id"]] = len(lines)
            line_meta.append({
                "entry_id": entry["id"],
                "order": entry["order"],
                "center_y": entry["center_y"],
            })
            lines.append((text, 1.0))

        price_entries = []
        for entry in entries:
            text = (entry["text"] or "").strip()
            if text not in currency_symbols:
                continue
            siblings = parent_children.get(entry["parent_id"], [])
            try:
                idx = siblings.index(entry)
            except ValueError:
                continue

            for next_sibling in siblings[idx + 1:]:
                candidate_text = (next_sibling["text"] or "").strip().replace(" ", "")
                if not candidate_text:
                    continue
                match = re.match(r"(\d{2,5})", candidate_text)
                if not match:
                    continue
                value = float(match.group(1))
                if value < 50:
                    continue
                price_entry = {
                    "value": value,
                    "order": next_sibling["order"],
                    "center_y": next_sibling["center_y"],
                    "text": next_sibling["text"],
                    "line_index": entry_to_line.get(next_sibling["id"]),
                    "used": False,
                }
                price_entries.append(price_entry)
                break

        if price_entries:
            price_entries.sort(key=lambda x: x["order"])
            sample_prices = [f"{int(p['value'])}" for p in price_entries[:5]]
            print(f"[Hierarchy] 结构化价格候选: {sample_prices}")

        return lines, line_meta if line_meta else None, price_entries

    def _collect_xml_entries(self, root):
        """遍历 XML，收集节点文本、顺序及父子关系"""
        entries = []
        parent_children = {}
        entry_id_counter = 0

        def traverse(node, parent_id):
            nonlocal entry_id_counter
            current_id = entry_id_counter
            entry_id_counter += 1
            text = node.attrib.get("text") or node.attrib.get("content-desc") or ""
            entry = {
                "id": current_id,
                "parent_id": parent_id,
                "text": text,
                "package": node.attrib.get("package", ""),
                "bounds": node.attrib.get("bounds"),
                "center_y": self._parse_bounds_center(node.attrib.get("bounds")),
                "order": len(entries),
            }
            entries.append(entry)
            parent_children.setdefault(parent_id, []).append(entry)

            for child in list(node):
                traverse(child, current_id)

        traverse(root, None)
        return entries, parent_children

    def _parse_bounds_center(self, bounds_str):
        if not bounds_str:
            return None
        try:
            parts = bounds_str.replace("[", "").replace("]", ",").split(",")
            if len(parts) < 4:
                return None
            y1 = float(parts[1])
            y2 = float(parts[3])
            return (y1 + y2) / 2.0
        except Exception:
            return None

    def _match_structured_price(self, structured_prices, title_meta, max_vertical_gap=220):
        """根据标题节点元数据匹配最近的价格节点"""
        if not structured_prices or not title_meta:
            return None

        title_order = title_meta.get("order")
        title_center = title_meta.get("center_y")

        best_entry = None
        best_gap = None

        for entry in structured_prices:
            if entry.get("used"):
                continue
            if title_order is not None and entry["order"] <= title_order:
                continue

            gap = None
            if title_center is not None and entry.get("center_y") is not None:
                gap = abs(entry["center_y"] - title_center)
                if gap > max_vertical_gap:
                    continue
            elif title_order is not None:
                gap = entry["order"] - title_order
                if gap > 80:
                    continue
            else:
                gap = entry["order"]

            if best_entry is None or (gap is not None and gap < best_gap):
                best_entry = entry
                best_gap = gap

        if best_entry:
            best_entry["used"] = True
        return best_entry

    def _prepare_candidate_images(self, screenshot_path):
        """生成原图和多种预处理图像，供 OCR 逐个尝试"""
        candidates = [screenshot_path]
        temp_files = []

        variant = self._enhance_for_ocr(screenshot_path, aggressive=False)
        if variant:
            candidates.append(variant)
            temp_files.append(variant)
            print(f"[OCR] Added enhanced image: {variant}")

        variant_zoom = self._enhance_for_ocr(screenshot_path, aggressive=False, zoom=True)
        if variant_zoom:
            candidates.append(variant_zoom)
            temp_files.append(variant_zoom)
            print(f"[OCR] Added zoom-enhanced image: {variant_zoom}")

        variant_strong = self._enhance_for_ocr(screenshot_path, aggressive=True)
        if variant_strong:
            candidates.append(variant_strong)
            temp_files.append(variant_strong)
            print(f"[OCR] Added aggressive enhanced image: {variant_strong}")

        return candidates, temp_files

    def _enhance_for_ocr(self, screenshot_path, aggressive=False, zoom=False):
        """增强图像对比度并根据宽度自适应缩放，便于 OCR 检测"""
        try:
            img = cv2.imread(screenshot_path)
            if img is None:
                print(f"[OCR] Failed to read image: {screenshot_path}")
                return None

            height, width = img.shape[:2]

            crop_top = int(height * 0.06)
            crop_bottom = int(height * 0.99)
            img = img[crop_top:crop_bottom, :]

            if aggressive:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = cv2.equalizeHist(l)
                lab = cv2.merge((l, a, b))
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
                sharpen_kernel = np.array([[0, -1, 0],
                                           [-1, 5, -1],
                                           [0, -1, 0]])
                enhanced = cv2.filter2D(enhanced, -1, sharpen_kernel)
                gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                enhanced = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            else:
                enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
                enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)

            height, width = enhanced.shape[:2]
            target_width = 1400 if zoom else 1100
            if zoom:
                scale = target_width / width if width < target_width else 1.3
            else:
                scale = target_width / width if (width < target_width or width > target_width + 200) else 1.0
            if scale != 1.0:
                enhanced = cv2.resize(enhanced, (int(width * scale), int(height * scale)))

            fd, temp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            cv2.imwrite(temp_path, enhanced)
            return temp_path
        except Exception as e:
            print(f"Enhance image failed: {e}")
            return None

