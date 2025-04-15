import os
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pandas.errors import SettingWithCopyWarning

# ========== 禁用 GPU（适用于 Mac） ==========
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
if torch.backends.mps.is_available():
    torch.backends.mps.is_available = lambda: False

# ========== PyTorch Forecasting ==========
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    GroupNormalizer
)
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

warnings.simplefilter("error", category=SettingWithCopyWarning)

# ========== 加载风力数据 ==========
df = pd.read_csv("data/wind_power.csv", parse_dates=["timestamp"])

# 每15分钟为1个时间步
df["time_idx"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // (15 * 60)
df["time_idx"] = df["time_idx"].astype(int)

# 添加类别变量
df["month"] = df["timestamp"].dt.month.astype("str").astype("category")
df["wind_speed_log"] = np.log(df["wind_speed"] + 1e-8)

# ========== 定义模型参数 ==========
max_encoder_length = 96  # 24小时历史数据
max_prediction_length = 96  # 预测未来24小时（15分钟 × 96）

training_cutoff = df["time_idx"].max() - max_prediction_length

# ========== 构建 TimeSeries 数据集 ==========
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
    target_normalizer=GroupNormalizer(groups=["wind_farm", "turbine_id"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

# ========== 构建 DataLoader ==========
train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

# ========== 回调设置 ==========
early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger("lightning_logs", name="wind_power")

# ========== 超参数优化（Optuna） ==========
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path="optuna_wind",
    n_trials=30,
    max_epochs=20,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 64),
    hidden_continuous_size_range=(8, 64),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.001, 0.1),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(
        limit_train_batches=30,
        log_every_n_steps=1,
        enable_progress_bar=True,
        accelerator="cpu"
    ),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False
)

# ✅ 保存 study 到文件
import pickle
with open("wind_power_study.pkl", "wb") as f:
    pickle.dump(study, f)
print("✅ 已保存调参 study 到 wind_power_study.pkl")










