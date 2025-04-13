import pandas as pd

# 替换为你的 parquet 文件路径（可以是相对路径或绝对路径）
parquet_file = "stallion.parquet"

# 使用 pandas 读取 parquet 文件
try:
    df = pd.read_parquet(parquet_file)
except ImportError as e:
    print("请先安装 pyarrow 或 fastparquet:")
    print("pip install pyarrow")
    raise e

# 显示基础信息
print("📊 数据集基本信息：")
print("-" * 40)
print(f"行数 × 列数: {df.shape}")
print("字段列表:")
print(df.columns.tolist())

# 显示前几行数据
print("\n🔍 前 5 行数据预览：")
print("-" * 40)
print(df.head())

# 可选：显示列的数据类型
print("\n🧠 各字段数据类型：")
print("-" * 40)
print(df.dtypes)