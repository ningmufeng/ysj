import re
import pandas as pd
from typing import List, Dict, Optional, Tuple

# 1. 读取词典文件
print("正在读取词典文件...")
try:
    df_model_dict = pd.read_excel('机型词典.xlsx')
    print(f"机型词典: {len(df_model_dict)} 条记录")
    print(f"机型词典列: {df_model_dict.columns.tolist()}")
    print(df_model_dict.head(10))
    
    df_condition_dict = pd.read_excel('成色词典.xlsx')
    print(f"\n成色词典: {len(df_condition_dict)} 条记录")
    print(f"成色词典列: {df_condition_dict.columns.tolist()}")
    print(df_condition_dict.head(10))
except Exception as e:
    print(f"读取词典文件出错: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 2. 构建匹配字典
print("\n正在构建匹配字典...")

# 机型词典：构建以机型名称为键的字典
model_dict = {}
for _, row in df_model_dict.iterrows():
    model_name = str(row.get('机型', '')).strip()
    if model_name and model_name != 'nan':
        model_dict[model_name] = {
            '品牌': str(row.get('品牌', '')).strip(),
            '分类': str(row.get('分类', '')).strip(),
            '机型': model_name,
            '配置': str(row.get('配置', '')).strip(),
            '颜色': str(row.get('颜色', '')).strip()
        }

# 成色词典：构建匹配规则
condition_dict = {}
for _, row in df_condition_dict.iterrows():
    part = str(row.get('部位', '')).strip()
    quantity = str(row.get('数量', '')).strip()
    degree = str(row.get('程度', '')).strip()
    condition_type = str(row.get('类型', '')).strip()
    
    # 构建关键词匹配
    key = f"{part}_{quantity}_{degree}_{condition_type}"
    condition_dict[key] = {
        '部位': part if part != 'nan' else '',
        '数量': quantity if quantity != 'nan' else '',
        '程度': degree if degree != 'nan' else '',
        '类型': condition_type if condition_type != 'nan' else ''
    }

print(f"机型词典已加载 {len(model_dict)} 条")
print(f"成色词典已加载 {len(df_condition_dict)} 条")

# 3. 读取原始文本
print("\n正在读取原始文本...")
with open('新演示手机收货价10月21.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 4. 解析文本
print("正在解析文本...")
results = []

# 品牌关键词
brands = ["华为", "荣耀", "vivo", "OPPO", "一加", "iQOO", "小米", "红米", "三星", "魅族", "苹果", "iPhone"]

# 当前上下文（用于处理跨行信息）
current_model_info = None
current_base_price = None

def match_model(text: str) -> Optional[Dict]:
    """匹配机型"""
    # 尝试匹配品牌
    brand = None
    for b in brands:
        if b in text:
            brand = b
            break
    
    # 尝试匹配机型词典中的机型
    best_match = None
    best_score = 0
    
    for model_name, model_data in model_dict.items():
        # 简单的包含匹配
        if model_name in text:
            score = len(model_name)
            if score > best_score:
                best_score = score
                best_match = model_data.copy()
                if brand:
                    best_match['品牌'] = brand
    
    return best_match

def match_condition(text: str) -> Dict:
    """匹配成色"""
    result = {
        '部位': '',
        '数量': '',
        '程度': '',
        '类型': ''
    }
    
    # 特殊成色：完美、整机完美
    if '完美' in text and ('整机' in text or '成色完美' in text):
        return result  # 返回空值
    
    # 匹配部位
    parts = ['屏幕', '外观', '尾插', '摄像头', '后壳', '后摄像头', '小屏', '大屏', '中框', '手机壳']
    for part in parts:
        if part in text:
            result['部位'] = part
            break
    
    # 匹配数量
    quantity_patterns = [
        (r'(\d+)[-到](\d+)处', lambda m: f"{m.group(1)}-{m.group(2)}处"),
        (r'(\d+)处', lambda m: f"{m.group(1)}处"),
        (r'1处', '1处'),
        (r'2处', '2处'),
        (r'3处', '3处'),
        (r'3-5处', '3-5处'),
        (r'三处', '3处'),
        (r'三处以上', '3处以上'),
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
    types = ['磨损', '磕碰', '老化', '大花', '掉皮', '掉漆', '夹痕', '划伤', '烧屏']
    for ctype in types:
        if ctype in text:
            result['类型'] = ctype
            break
    
    return result

def extract_price(text: str) -> Optional[Tuple[int, bool]]:
    """提取价格，返回(价格, 是否为相对价格)"""
    # 匹配绝对价格（3-5位数字）
    price_match = re.search(r'(\d{3,5})', text)
    if price_match:
        return (int(price_match.group(1)), False)
    
    # 匹配相对价格（如-50, -100等）
    relative_match = re.search(r'[-少](\d+)', text)
    if relative_match:
        return (int(relative_match.group(1)), True)
    
    return None

def extract_config(text: str) -> Optional[str]:
    """提取配置信息"""
    # 匹配配置格式：如 12+256G, 8+128, 16+512G等
    config_pattern = r'(\d+)\+(\d+)([GT]?B?)'
    match = re.search(config_pattern, text, re.IGNORECASE)
    if match:
        return f"{match.group(1)}+{match.group(2)}{match.group(3) if match.group(3) else 'G'}"
    return None

def extract_color(text: str) -> Optional[str]:
    """提取颜色信息"""
    colors = ['红色', '黑色', '白色', '蓝色', '绿色', '紫色', '棕色', '金色', '灰色', '星云灰']
    for color in colors:
        if color in text:
            return color
    
    # 处理多个颜色（如"黑 蓝 白"）
    color_list = []
    for color in colors:
        if color in text:
            color_list.append(color)
    if color_list:
        return ' '.join(color_list)
    
    return None

# 解析每一行
for i, line in enumerate(lines):
    line = line.strip()
    if not line or len(line) < 3:
        continue
    
    # 跳过注释行
    if line.startswith('10月') or '求演示机' in line or '真诚合作' in line or '感谢' in line:
        continue
    
    # 尝试匹配机型
    model_info = match_model(line)
    
    # 提取配置
    config = extract_config(line)
    if config and model_info:
        model_info['配置'] = config
    
    # 提取颜色
    color = extract_color(line)
    if color and model_info:
        model_info['颜色'] = color
    
    # 提取价格
    price_info = extract_price(line)
    
    # 匹配成色
    condition_info = match_condition(line)
    
    # 如果有机型信息，创建记录
    if model_info:
        current_model_info = model_info.copy()
        current_base_price = None
        
        if price_info:
            price, is_relative = price_info
            if not is_relative:
                current_base_price = price
                # 创建记录
                result = {
                    '品牌': model_info.get('品牌', ''),
                    '分类': model_info.get('分类', '手机'),
                    '机型': model_info.get('机型', ''),
                    '配置': model_info.get('配置', ''),
                    '颜色': model_info.get('颜色', ''),
                    '部位': condition_info['部位'],
                    '数量': condition_info['数量'],
                    '程度': condition_info['程度'],
                    '类型': condition_info['类型'],
                    '价格': price
                }
                results.append(result)
    elif current_model_info:
        # 没有机型信息，但之前有上下文，可能是价格行
        price_info = extract_price(line)
        if price_info:
            price, is_relative = price_info
            if is_relative and current_base_price:
                price = current_base_price - price
            elif not is_relative:
                current_base_price = price
            
            condition_info = match_condition(line)
            
            result = {
                '品牌': current_model_info.get('品牌', ''),
                '分类': current_model_info.get('分类', '手机'),
                '机型': current_model_info.get('机型', ''),
                '配置': current_model_info.get('配置', ''),
                '颜色': current_model_info.get('颜色', ''),
                '部位': condition_info['部位'],
                '数量': condition_info['数量'],
                '程度': condition_info['程度'],
                '类型': condition_info['类型'],
                '价格': price
            }
            results.append(result)

print(f"\n解析完成，共提取 {len(results)} 条记录")

# 5. 生成输出文件
if results:
    df_output = pd.DataFrame(results)
    
    # 按品牌、机型、价格排序
    df_output = df_output.sort_values(['品牌', '机型', '价格'], ascending=[True, True, False])
    
    # 导出Excel
    output_file = '演示机价格表_标准化.xlsx'
    df_output.to_excel(output_file, index=False)
    print(f"\n✅ 已成功导出到【{output_file}】")
    print(f"\n前10条记录预览:")
    print(df_output.head(10).to_string())
    print(f"\n总计: {len(df_output)} 条记录")
else:
    print("⚠️ 没有提取到任何记录，请检查文本格式或词典匹配规则。")

