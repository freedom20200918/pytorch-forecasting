# 文件名：wind_power_train.py

import pandas as pd
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import SMAPE

# 加载风电数据
df = pd.read_csv("data/wind_power.csv", parse_dates=["timestamp"])


# 创建时间索引（以小时为单位）并转换为整数类型
df["time_idx"] = ((df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 3600).astype(int)
df["group"] = "wind_turbine_1"  # 简化为单个风机

# 加入 log_power 避免为零出错
df["log_power"] = np.log(df["power"] + 1e-6)

# 设置训练/验证集分割点
training_cutoff = df["time_idx"].max() - 24
max_encoder_length = 24  # 使用过去 24 小时信息
max_prediction_length = 6  # 预测未来 6 小时

# 构建训练数据集
training = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="power",
    group_ids=["group"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["time_idx", "wind_speed", "wind_direction"],
    time_varying_unknown_reals=["power", "log_power"],
    target_normalizer=NaNLabelEncoder(),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# 验证数据集
validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

# 加载器
train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

# 初始化 Temporal Fusion Transformer 模型
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,
    loss=SMAPE(),
    log_interval=10,
    reduce_on_plateau_patience=4
)

# PyTorch Lightning 训练器
trainer = Trainer(
    max_epochs=30,
    accelerator="cpu",
    callbacks=[EarlyStopping(monitor="val_loss", patience=5)]
)

# 模型训练
trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

# 执行预测
preds, index = tft.predict(val_loader, return_index=True)
print("预测结果前 5 条：")
print(preds[:5])
