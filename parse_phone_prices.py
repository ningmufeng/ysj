import re
import pandas as pd

# 1️⃣ 读取原始文本
# 建议先把你哥发的那份文本粘贴到同目录下的 "raw.txt" 文件中
with open("raw.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2️⃣ 定义品牌关键词（你可以随时扩充）
brands = ["华为", "荣耀", "vivo", "OPPO", "一加", "iQOO", "小米", "红米", "三星"]

# 3️⃣ 正则规则匹配：
# 先匹配品牌+型号+规格，然后在同一行中查找所有成色-价格对
# 成色描述必须包含关键字（完美|磨|损|化|磕|处|花），匹配完整的成色描述
rows = []
brand_model_pattern = re.compile(
    r'(?P<brand>' + '|'.join(brands) + r')?'
    r'(?P<model>[A-Za-z0-9一-龥+ ]{1,20})'
    r'(?P<spec>[0-9]{3,4}G)?'
)

# 成色-价格对的正则：匹配包含关键字（完美|磨|损|化|磕|处|花）的成色描述和价格
# 允许描述中包含数字（如"2处"、"3-5处"），但不匹配到价格数字（3-5位）
condition_price_pattern = re.compile(
    r'(?P<condition>.*?[完美磨损伤化磕处花].*?)(?=\s*\d{3,5})\s*(?P<price>\d{3,5})'
)

for line in text.split('\n'):
    line = line.strip()
    if not line:
        continue
    
    # 查找品牌+型号+规格
    brand_model_match = brand_model_pattern.search(line)
    if not brand_model_match:
        continue
    
    base_data = brand_model_match.groupdict()
    brand = base_data.get('brand', '').strip() if base_data.get('brand') else ''
    model = base_data.get('model', '').strip() if base_data.get('model') else ''
    spec = base_data.get('spec', '').strip() if base_data.get('spec') else ''
    
    if not model:
        continue
    
    # 在型号之后的文本中查找所有成色-价格对
    remaining_text = line[brand_model_match.end():]
    
    # 查找所有成色-价格对
    condition_matches = list(condition_price_pattern.finditer(remaining_text))
    
    if condition_matches:
        # 为每个成色-价格对创建一条记录
        for cond_match in condition_matches:
            data = {
                'brand': brand,
                'model': model,
                'spec': spec,
                'condition': cond_match.group('condition').strip(),
                'price': cond_match.group('price').strip()
            }
            rows.append(data)
    else:
        # 如果没有找到成色-价格对，但找到了型号，也记录一条（可能只有价格）
        price_match = re.search(r'\d{3,5}', remaining_text)
        if price_match:
            data = {
                'brand': brand,
                'model': model,
                'spec': spec,
                'condition': '',
                'price': price_match.group().strip()
            }
            rows.append(data)

# 4️⃣ 转换成DataFrame并导出Excel
if rows:
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "brand": "品牌",
        "model": "型号",
        "spec": "规格",
        "condition": "成色",
        "price": "收货价"
    })
    df.to_excel("演示机收货价_结构化.xlsx", index=False)
    print("✅ 已成功导出到【演示机收货价_结构化.xlsx】")
    print(df.head(10))
else:
    print("⚠️ 没匹配到任何条目，请检查原始文本格式或扩展正则规则。")
