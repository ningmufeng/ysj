import pandas as pd

# 查看机型词典
print("=" * 60)
print("机型词典.xlsx")
print("=" * 60)
try:
    df_model = pd.read_excel('机型词典.xlsx')
    print("列名:", df_model.columns.tolist())
    print("行数:", len(df_model))
    print("\n前15行数据:")
    print(df_model.head(15))
    print("\n数据类型:")
    print(df_model.dtypes)
except Exception as e:
    print(f"读取机型词典出错: {e}")

print("\n" + "=" * 60)
print("成色词典.xlsx")
print("=" * 60)
try:
    df_condition = pd.read_excel('成色词典.xlsx')
    print("列名:", df_condition.columns.tolist())
    print("行数:", len(df_condition))
    print("\n前15行数据:")
    print(df_condition.head(15))
    print("\n数据类型:")
    print(df_condition.dtypes)
except Exception as e:
    print(f"读取成色词典出错: {e}")

print("\n" + "=" * 60)
print("收货价.xlsx")
print("=" * 60)
try:
    df_price = pd.read_excel('收货价.xlsx')
    print("列名:", df_price.columns.tolist())
    print("行数:", len(df_price))
    print("\n前10行数据:")
    print(df_price.head(10))
except Exception as e:
    print(f"读取收货价出错: {e}")

