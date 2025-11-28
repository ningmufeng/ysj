import pandas as pd
import sys

print("=" * 60)
print("机型词典.xlsx")
print("=" * 60)
try:
    df_model = pd.read_excel('机型词典.xlsx')
    print("列名:", df_model.columns.tolist())
    print("行数:", len(df_model))
    print("\n前20行数据:")
    print(df_model.head(20).to_string())
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("成色词典.xlsx")
print("=" * 60)
try:
    df_condition = pd.read_excel('成色词典.xlsx')
    print("列名:", df_condition.columns.tolist())
    print("行数:", len(df_condition))
    print("\n前20行数据:")
    print(df_condition.head(20).to_string())
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

