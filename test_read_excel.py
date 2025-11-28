import pandas as pd
import sys

print("开始测试读取Excel文件...")

try:
    print("\n1. 读取机型词典...")
    df_model = pd.read_excel('机型词典.xlsx', engine='openpyxl')
    print(f"   成功！行数: {len(df_model)}, 列数: {len(df_model.columns)}")
    print(f"   列名: {df_model.columns.tolist()}")
    print(f"   前5行:")
    print(df_model.head(5))
    
    print("\n2. 读取成色词典...")
    df_condition = pd.read_excel('成色词典.xlsx', engine='openpyxl')
    print(f"   成功！行数: {len(df_condition)}, 列数: {len(df_condition.columns)}")
    print(f"   列名: {df_condition.columns.tolist()}")
    print(f"   前5行:")
    print(df_condition.head(5))
    
    print("\n测试完成！")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

