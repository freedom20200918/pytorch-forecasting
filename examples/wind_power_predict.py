import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer  # ✅ 正确的归一化方式

# ✅ 加载风力数据
df = pd.read_csv("data/wind_power.csv", parse_dates=["timestamp"])
df["time_idx"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // (15 * 60)
df["time_idx"] = df["time_idx"].astype(int)
df["month"] = df["timestamp"].dt.month.astype("str").astype("category")
df["wind_speed_log"] = np.log(df["wind_speed"] + 1e-8)

# ✅ 参数设置（保持和训练一致）
max_encoder_length = 96
max_prediction_length = 96
training_cutoff = df["time_idx"].max() - max_prediction_length

# ✅ 构造训练集对象（结构需与训练完全一致）
training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="power_output",
    group_ids=["wind_farm", "turbine_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["wind_farm", "turbine_id"],
    time_varying_known_reals=["time_idx", "wind_speed", "temperature"],
    time_varying_unknown_reals=["power_output", "wind_speed_log"],
    target_normalizer=GroupNormalizer(groups=["wind_farm", "turbine_id"]),  # ✅ 必须与训练保持一致
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# ✅ 构造验证集（未来时间窗口）
validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

# ✅ 加载 Optuna study，提取最佳模型路径
with open("wind_power_study.pkl", "rb") as f:
    study = pickle.load(f)

best_trial = study.best_trial
if "best_model_path" not in best_trial.user_attrs:
    raise RuntimeError("❌ best_model_path 不存在，请确认训练中使用 save_checkpoints=True 且未被剪枝")
best_model_path = best_trial.user_attrs["best_model_path"]
print(f"✅ 加载最佳模型路径：{best_model_path}")

# ✅ 加载训练好的模型并预测
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

predictions, x = best_tft.predict(
    val_dataloader,
    mode="prediction",
    return_x=True,
    return_index=True
)

# ✅ 构造预测结果表
pred_df = pd.DataFrame({
    "time_idx": x["decoder_time_idx"][0].numpy(),
    "timestamp": x["decoder_time"][0].numpy(),
    "predicted_power": predictions[0].detach().numpy()
})
pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])

# ✅ 保存预测结果
pred_df.to_csv("predicted_24h_wind_power.csv", index=False)
print("✅ 预测结果已保存到 predicted_24h_wind_power.csv")

# ✅ 可视化预测
plt.figure(figsize=(12, 5))
plt.plot(pred_df["timestamp"], pred_df["predicted_power"], marker="o", label="Predicted Power")
plt.title("Predicted Wind Power Output - Next 24 Hours (15-min interval)")
plt.xlabel("Time")
plt.ylabel("Power Output (kW)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("predicted_24h_wind_power.png")
plt.show()
