# -*- coding: utf-8 -*-
import re
import pandas as pd
import sys
from typing import Dict, Optional, Tuple

print("=" * 60)
print("开始处理演示机价格表...")
print("=" * 60)

# 1. 读取词典文件
print("\n[1/6] 正在读取词典文件...")
try:
    # 尝试使用openpyxl引擎
    try:
        df_model_dict = pd.read_excel('机型词典.xlsx', engine='openpyxl')
    except:
        df_model_dict = pd.read_excel('机型词典.xlsx')
    
    print(f"✓ 机型词典: {len(df_model_dict)} 条记录")
    print(f"  列名: {df_model_dict.columns.tolist()}")
    
    try:
        df_condition_dict = pd.read_excel('成色词典.xlsx', engine='openpyxl')
    except:
        df_condition_dict = pd.read_excel('成色词典.xlsx')
    
    print(f"✓ 成色词典: {len(df_condition_dict)} 条记录")
    print(f"  列名: {df_condition_dict.columns.tolist()}")
    
except Exception as e:
    print(f"✗ 读取词典文件出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. 构建匹配字典
print("\n[2/6] 正在构建匹配字典...")
model_dict = {}
for idx, row in df_model_dict.iterrows():
    try:
        model_name = str(row.get('机型', '')).strip()
        if model_name and model_name != 'nan' and model_name:
            model_dict[model_name] = {
                '品牌': str(row.get('品牌', '')).strip() if pd.notna(row.get('品牌')) else '',
                '分类': str(row.get('分类', '')).strip() if pd.notna(row.get('分类')) else '手机',
                '机型': model_name,
                '配置': str(row.get('配置', '')).strip() if pd.notna(row.get('配置')) else '',
                '颜色': str(row.get('颜色', '')).strip() if pd.notna(row.get('颜色')) else ''
            }
    except Exception as e:
        continue

print(f"✓ 已加载 {len(model_dict)} 个机型")

# 3. 读取原始文本并提取日期
print("\n[3/6] 正在读取原始文本...")
input_filename = '新演示手机收货价10月21.txt'
try:
    with open(input_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"✓ 读取了 {len(lines)} 行")
    
    # 从文件名提取日期信息
    date_match = re.search(r'(\d+)月(\d+)', input_filename)
    if date_match:
        month = int(date_match.group(1))
        day = int(date_match.group(2))
        # 假设年份为2024（可以根据需要调整）
        extracted_date = f"2024-{month:02d}-{day:02d}"
    else:
        # 如果无法从文件名提取，使用默认日期
        extracted_date = "2024-10-21"
    print(f"✓ 提取的日期: {extracted_date}")
except Exception as e:
    print(f"✗ 读取文本文件出错: {e}")
    sys.exit(1)

# 4. 解析函数
def extract_config(text: str) -> Optional[str]:
    """提取配置信息"""
    patterns = [
        r'(\d+)\+(\d+)([GT]?B?)',
        r'(\d+)\s*\+\s*(\d+)([GT]?B?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            gb = match.group(3) if match.group(3) else 'G'
            return f"{match.group(1)}+{match.group(2)}{gb}"
    return None

def extract_color(text: str) -> Optional[str]:
    """提取颜色信息"""
    colors = ['红色', '黑色', '白色', '蓝色', '绿色', '紫色', '棕色', '金色', '灰色', '星云灰', '红', '黑', '白', '蓝', '绿', '紫', '棕', '金', '灰']
    found = []
    for color in colors:
        if color in text:
            found.append(color)
    return ' '.join(found) if found else None

def match_condition(text: str) -> Dict:
    """匹配成色"""
    result = {'部位': '', '数量': '', '程度': '', '类型': ''}
    
    # 特殊成色：完美
    if '完美' in text and ('整机' in text or '成色完美' in text):
        return result
    
    # 匹配部位
    parts = ['屏幕', '外观', '尾插', '摄像头', '后壳', '后摄像头', '小屏', '大屏', '中框', '手机壳', '屏']
    for part in parts:
        if part in text:
            result['部位'] = part
            break
    
    # 匹配数量
    quantity_patterns = [
        (r'(\d+)[-到](\d+)处', lambda m: f"{m.group(1)}-{m.group(2)}处"),
        (r'(\d+)处', lambda m: f"{m.group(1)}处"),
        (r'三处以上', '3处以上'),
        (r'3-5处', '3-5处'),
        (r'2处', '2处'),
        (r'1处', '1处'),
    ]
    
    for pattern, replacement in quantity_patterns:
        if isinstance(pattern, str):
            if pattern in text:
                result['数量'] = replacement if isinstance(replacement, str) else replacement
                break
        else:
            match = re.search(pattern, text)
            if match:
                result['数量'] = replacement(match) if callable(replacement) else replacement
                break
    
    # 匹配程度
    degrees = ['轻微', '严重', '厉害', '明显', '大']
    for degree in degrees:
        if degree in text:
            result['程度'] = degree
            break
    
    # 匹配类型
    types = ['磨损', '磕碰', '老化', '大花', '掉皮', '掉漆', '夹痕', '划伤', '烧屏', '花']
    for ctype in types:
        if ctype in text:
            result['类型'] = ctype
            break
    
    return result

def extract_price(text: str) -> Optional[Tuple[int, bool]]:
    """提取价格"""
    # 绝对价格
    price_match = re.search(r'(\d{3,5})', text)
    if price_match:
        return (int(price_match.group(1)), False)
    
    # 相对价格
    relative_match = re.search(r'[-少](\d+)', text)
    if relative_match:
        return (int(relative_match.group(1)), True)
    
    return None

# 5. 解析文本
print("\n[4/6] 正在解析文本...")
results = []
brands = ["华为", "荣耀", "vivo", "OPPO", "一加", "iQOO", "小米", "红米", "三星", "魅族", "苹果", "iPhone"]

current_context = None
current_base_price = None
line_count = 0

for i, line in enumerate(lines):
    line = line.strip()
    if not line or len(line) < 3:
        continue
    
    # 跳过注释行
    if any(x in line for x in ['10月', '求演示机', '真诚合作', '感谢', '磕碰和磨损']):
        continue
    
    line_count += 1
    if line_count % 50 == 0:
        print(f"  已处理 {line_count} 行...")
    
    # 提取每行左起30个字符（用于品牌和机型匹配）
    line_prefix = line[:30] if len(line) > 30 else line
    
    # 尝试匹配品牌，记录品牌位置（在前30字符范围内）
    brand = None
    brand_end_pos = 0
    for b in brands:
        brand_pos = line_prefix.find(b)
        if brand_pos != -1:
            brand = b
            brand_end_pos = brand_pos + len(b)
            break
    
    # 尝试匹配机型（从品牌之后开始搜索，支持大小写不敏感）
    # 只在前30字符范围内匹配机型
    matched_model = None
    best_match_length = 0
    
    # 确定搜索范围：如果找到品牌，从品牌之后开始；否则从前30字符开始
    # 如果品牌和机型连在一起（没有空格），需要跳过可能的连字符或直接开始搜索
    if brand and brand_end_pos < len(line_prefix):
        # 跳过品牌后的空格、连字符等
        search_start = brand_end_pos
        while search_start < len(line_prefix) and line_prefix[search_start] in [' ', '-', '_']:
            search_start += 1
        search_text = line_prefix[search_start:]
    else:
        search_text = line_prefix
    
    # 华为品牌特殊映射规则（按优先级顺序）
    if brand == '华为':
        # 1. XT映射到"Mate XT 非凡大师"（最高优先级，因为XT是完整词）
        if re.search(r'\bXT\b', search_text, re.IGNORECASE):
            sorted_models = sorted(model_dict.keys(), key=len, reverse=True)
            for model_name in sorted_models:
                model_lower = model_name.lower()
                if 'mate xt' in model_lower or 'matext' in model_lower or '非凡大师' in model_name:
                    if len(model_name) > best_match_length:
                        matched_model = model_dict[model_name].copy()
                        best_match_length = len(model_name)
                        if brand:
                            matched_model['品牌'] = brand
                        break
        
        # 2. P70系列映射到Pura70系列（在X之前，避免P70X被误匹配）
        if not matched_model:
            # 注意：P70U的U指的是Ultra，应映射到Pura70Ultra
            p70_patterns = [
                (r'P70U', ['pura70ultra', 'pura 70ultra', 'pura70 ultra']),
                (r'P70Ultra', ['pura70ultra', 'pura 70ultra', 'pura70 ultra']),
                (r'P70PRO\+', ['pura70pro+', 'pura 70pro+', 'pura70 pro+']),
                (r'P70PRO', ['pura70pro', 'pura 70pro', 'pura70 pro']),
                (r'P70Pro', ['pura70pro', 'pura 70pro', 'pura70 pro']),
                (r'P70\+', ['pura70+', 'pura 70+']),
                (r'P70', ['pura70', 'pura 70']),
            ]
            
            for p_pattern, pura_variants in p70_patterns:
                if re.search(p_pattern, search_text, re.IGNORECASE):
                    sorted_models = sorted(model_dict.keys(), key=len, reverse=True)
                    for model_name in sorted_models:
                        model_lower = model_name.lower()
                        for pura_variant in pura_variants:
                            if pura_variant in model_lower:
                                if len(model_name) > best_match_length:
                                    matched_model = model_dict[model_name].copy()
                                    best_match_length = len(model_name)
                                    if brand:
                                        matched_model['品牌'] = brand
                                    break
                        if matched_model:
                            break
                    if matched_model:
                        break
        
        # 3. X映射到MateX系列（MateX, MateX6等），但排除XT（XT已单独处理）
        # 匹配规则：文本中有独立的X（不是XT的一部分），且不是P70X等
        if not matched_model:
            x_match = re.search(r'\bX\b', search_text, re.IGNORECASE)
            if x_match and not re.search(r'\bXT\b', search_text, re.IGNORECASE):
                # 检查X前面是否有P70等，避免误匹配
                x_pos = x_match.start()
                before_x = search_text[:x_pos].strip()
                if not re.search(r'P70|P60|P50', before_x[-10:] if len(before_x) > 10 else before_x, re.IGNORECASE):
                    sorted_models = sorted(model_dict.keys(), key=len, reverse=True)
                    for model_name in sorted_models:
                        model_lower = model_name.lower()
                        if 'matex' in model_lower and 'xt' not in model_lower:
                            if len(model_name) > best_match_length:
                                matched_model = model_dict[model_name].copy()
                                best_match_length = len(model_name)
                                if brand:
                                    matched_model['品牌'] = brand
                                break
        
        # 4. mate映射到Mate系列（排除MateX和MateXT）
        if not matched_model and re.search(r'\bmate\b', search_text, re.IGNORECASE):
            sorted_models = sorted(model_dict.keys(), key=len, reverse=True)
            for model_name in sorted_models:
                model_lower = model_name.lower()
                # 匹配Mate系列，但排除MateX和MateXT（已单独处理）
                if model_lower.startswith('mate') and 'matex' not in model_lower:
                    if len(model_name) > best_match_length:
                        matched_model = model_dict[model_name].copy()
                        best_match_length = len(model_name)
                        if brand:
                            matched_model['品牌'] = brand
                        break
    
    # 5. NOVA系列映射（适用于所有品牌，但主要是华为）
    # 匹配NOVA系列：NOVA9, NOVA11se, NOVA12等
    if not matched_model and re.search(r'\bNOVA', search_text, re.IGNORECASE):
        # 提取NOVA后的完整标识（如NOVA9, NOVA11se, NOVA12等）
        nova_match = re.search(r'\bNOVA(\d+[a-z]*|)', search_text, re.IGNORECASE)
        if nova_match:
            nova_suffix = nova_match.group(1) if nova_match.group(1) else ''
            sorted_models = sorted(model_dict.keys(), key=len, reverse=True)
            for model_name in sorted_models:
                model_lower = model_name.lower()
                if 'nova' in model_lower:
                    # 如果提取到NOVA数字，优先匹配相同数字的
                    if nova_suffix:
                        if f'nova{nova_suffix.lower()}' in model_lower or f'nova {nova_suffix.lower()}' in model_lower:
                            if len(model_name) > best_match_length:
                                matched_model = model_dict[model_name].copy()
                                best_match_length = len(model_name)
                                if brand:
                                    matched_model['品牌'] = brand
                                break
                    else:
                        # 如果没有数字，匹配所有NOVA系列
                        if len(model_name) > best_match_length:
                            matched_model = model_dict[model_name].copy()
                            best_match_length = len(model_name)
                            if brand:
                                matched_model['品牌'] = brand
                            break
    
    # 如果还没有通过映射找到机型，继续正常匹配流程
    if not matched_model:
        # 对机型词典按长度排序（从长到短），优先匹配更长的机型名称
        sorted_models = sorted(model_dict.keys(), key=len, reverse=True)
        
        for model_name in sorted_models:
            # 大小写不敏感匹配
            model_lower = model_name.lower()
            search_lower = search_text.lower()
            
            # 尝试精确匹配或部分匹配
            if model_lower in search_lower:
                # 找到匹配位置
                match_pos = search_lower.find(model_lower)
                if match_pos != -1:
                    # 检查是否是合理的匹配位置
                    # 允许匹配位置在开头，或者前面是空格/标点
                    is_valid_match = True
                    if match_pos > 0:
                        prev_char = search_text[match_pos - 1]
                        # 如果前面是字母数字，可能不是完整匹配（但允许+和-，因为可能是配置信息）
                        if prev_char.isalnum() and prev_char not in ['+', '-']:
                            is_valid_match = False
                    
                    # 如果匹配到更长的机型名称，更新匹配结果
                    if len(model_name) > best_match_length and is_valid_match:
                        matched_model = model_dict[model_name].copy()
                        best_match_length = len(model_name)
                        if brand:
                            matched_model['品牌'] = brand
    
    # 如果找到机型，确保品牌信息正确
    if matched_model and brand:
        matched_model['品牌'] = brand
    
    # 检查是否在前30字符范围内找到了品牌和机型
    # 只有在前30字符范围内同时找到品牌和机型，才标记为已分类
    # 否则标记为未分类
    found_brand_in_prefix = brand is not None
    found_model_in_prefix = matched_model is not None
    
    # 提取配置和颜色
    config = extract_config(line)
    color = extract_color(line)
    
    # 提取价格
    price_info = extract_price(line)
    
    # 匹配成色
    condition_info = match_condition(line)
    
    # 如果找到机型，更新上下文
    # 只有在前30字符范围内同时找到品牌和机型时，才标记为已分类
    if matched_model:
        # 检查是否在前30字符范围内找到了品牌和机型
        if found_brand_in_prefix and found_model_in_prefix:
            classification_status = '已分类'
        else:
            # 虽然找到了机型，但不在前30字符范围内，标记为未分类
            classification_status = '未分类'
        current_context = matched_model.copy()
        if config:
            current_context['配置'] = config
        if color:
            current_context['颜色'] = color
        current_base_price = None
        
        if price_info:
            price, is_relative = price_info
            if not is_relative:
                current_base_price = price
                result = {
                    '品牌': current_context.get('品牌', ''),
                    '分类': current_context.get('分类', '手机'),
                    '机型': current_context.get('机型', ''),
                    '配置': current_context.get('配置', ''),
                    '颜色': current_context.get('颜色', ''),
                    '部位': condition_info['部位'],
                    '数量': condition_info['数量'],
                    '程度': condition_info['程度'],
                    '类型': condition_info['类型'],
                    '价格': price,
                    '日期': extracted_date,
                    '分类状态': classification_status
                }
                results.append(result)
    elif current_context:
        # 没有机型但有机型上下文，可能是价格行
        price_info = extract_price(line)
        if price_info:
            price, is_relative = price_info
            if is_relative and current_base_price:
                price = current_base_price - price
            elif not is_relative:
                current_base_price = price
            
            condition_info = match_condition(line)
            
            result = {
                '品牌': current_context.get('品牌', ''),
                '分类': current_context.get('分类', '手机'),
                '机型': current_context.get('机型', ''),
                '配置': current_context.get('配置', ''),
                '颜色': current_context.get('颜色', ''),
                '部位': condition_info['部位'],
                '数量': condition_info['数量'],
                '程度': condition_info['程度'],
                '类型': condition_info['类型'],
                '价格': price,
                '日期': extracted_date,
                '分类状态': '已分类'
            }
            results.append(result)
    else:
        # 没有匹配到机型，检查是否在前30字符范围内找到了品牌和机型
        # 如果在前30字符范围内没有同时找到品牌和机型，标记为未分类
        if not (found_brand_in_prefix and found_model_in_prefix):
            # 在前30字符范围内未找到品牌或机型，标记为未分类
            if price_info:
                price, is_relative = price_info
                if not is_relative:
                    # 创建未分类记录（有价格）
                    result = {
                        '品牌': brand if brand else '',
                        '分类': '手机',
                        '机型': '',
                        '配置': config if config else '',
                        '颜色': color if color else '',
                        '部位': condition_info['部位'],
                        '数量': condition_info['数量'],
                        '程度': condition_info['程度'],
                        '类型': condition_info['类型'],
                        '价格': price,
                        '日期': extracted_date,
                        '分类状态': '未分类'
                    }
                    results.append(result)
            elif brand or config or color or any(condition_info.values()):
                # 即使没有价格，但如果提取到品牌、配置、颜色或成色信息，也输出未分类记录
                result = {
                    '品牌': brand if brand else '',
                    '分类': '手机',
                    '机型': '',
                    '配置': config if config else '',
                    '颜色': color if color else '',
                    '部位': condition_info['部位'],
                    '数量': condition_info['数量'],
                    '程度': condition_info['程度'],
                    '类型': condition_info['类型'],
                    '价格': '',
                    '日期': extracted_date,
                    '分类状态': '未分类'
                }
                results.append(result)
        else:
            # 在前30字符范围内找到了品牌和机型，但当前行没有匹配到机型（可能是价格行）
            # 这种情况已经在current_context分支处理了，这里不需要额外处理
            pass

print(f"✓ 解析完成，提取了 {len(results)} 条记录")

# 6. 生成输出
print("\n[5/6] 正在生成输出文件...")
if results:
    df_output = pd.DataFrame(results)
    
    # 确保字段顺序：品牌、分类、机型、配置、颜色、部位、数量、程度、类型、价格、日期、分类状态
    column_order = ['品牌', '分类', '机型', '配置', '颜色', '部位', '数量', '程度', '类型', '价格', '日期', '分类状态']
    # 只保留存在的列
    existing_columns = [col for col in column_order if col in df_output.columns]
    df_output = df_output[existing_columns]
    
    df_output = df_output.sort_values(['品牌', '机型', '价格'], ascending=[True, True, False])
    
    output_file = '演示机价格表_标准化.csv'
    df_output.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"✓ 已成功导出到【{output_file}】")
    print(f"\n前10条记录预览:")
    print(df_output.head(10).to_string())
    print(f"\n总计: {len(df_output)} 条记录")
    
    # 统计分类状态
    if '分类状态' in df_output.columns:
        status_counts = df_output['分类状态'].value_counts()
        print(f"\n分类状态统计:")
        for status, count in status_counts.items():
            print(f"  {status}: {count} 条")
else:
    print("⚠️ 没有提取到任何记录")

print("\n[6/6] 完成！")
print("=" * 60)

