import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# 起始时间设置为 2022年1月1日
start_time = datetime(2022, 1, 1)

# 每15分钟一个点，一天96条，一周 96×7 = 672
interval_minutes = 15
n_intervals = 96 * 7  # 一周数据

# 生成时间戳列表
timestamps = [start_time + timedelta(minutes=i * interval_minutes) for i in range(n_intervals)]

# 模拟风场和涡轮编号
wind_farms = ["A", "B"]
turbines = ["T1", "T2", "T3"]

# 存储所有记录
records = []

np.random.seed(42)  # 保证可复现

# 遍历所有风场、涡轮、时间点组合
for wf in wind_farms:
    for tid in turbines:
        for ts in timestamps:
            wind_speed = np.random.normal(8, 2)  # 模拟风速
            power_output = max(0, wind_speed * np.random.uniform(10, 20))  # 简单线性乘风速
            wind_direction = np.random.randint(0, 360)
            temperature = np.random.normal(20, 5)

            records.append({
                "timestamp": ts,
                "wind_farm": wf,
                "turbine_id": tid,
                "wind_speed": round(wind_speed, 2),
                "power_output": round(power_output, 2),
                "wind_direction": wind_direction,
                "temperature": round(temperature, 2)
            })

# 创建 DataFrame
df = pd.DataFrame(records)

# 保存到 CSV 文件
os.makedirs("data", exist_ok=True)
df.to_csv("data/wind_power.csv", index=False)

print(f"✅ 已生成模拟风力数据：{len(df)} 条记录，保存至 data/wind_power.csv")