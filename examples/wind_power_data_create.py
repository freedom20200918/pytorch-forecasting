import pandas as pd
import numpy as np
from datetime import datetime, timedelta

start_time = datetime(2022, 1, 1)
hours = 7 * 24  # 至少一周的数据

timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
power = np.abs(np.random.normal(100, 20, size=hours))
wind_speed = np.random.normal(8, 2, size=hours)
wind_direction = np.random.randint(180, 360, size=hours)

df = pd.DataFrame({
    "timestamp": timestamps,
    "power": np.round(power, 2),
    "wind_speed": np.round(wind_speed, 2),
    "wind_direction": wind_direction
})


import os

# 确保 data 文件夹存在
os.makedirs("data", exist_ok=True)

df.to_csv("wind_power.csv", index=False)




