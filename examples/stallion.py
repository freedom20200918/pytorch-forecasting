import os

# 禁用 CUDA GPU（通常在 mac 上不影响，但保险起见）
# 设置环境变量，确保不使用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 禁用 MPS 后端（非常关键）
# 设置环境变量，防止使用 Apple 的 Metal Performance Shaders 后端
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

# 强制使用 CPU 模式
import torch
if torch.backends.mps.is_available():
    # 如果 MPS 可用，则将其设置为不可用，强制使用 CPU
    torch.backends.mps.is_available = lambda: False
    torch.device("cpu")

import pickle
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
from pandas.errors import SettingWithCopyWarning

from pytorch_forecasting import (
    GroupNormalizer,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)

warnings.simplefilter("error", category=SettingWithCopyWarning)

# 获取示例数据
data = get_stallion_data()

# 将日期转换为月份，并转换为类别类型
data["month"] = data.date.dt.month.astype("str").astype("category")
# 计算成交量的对数，避免出现 0 值
data["log_volume"] = np.log(data.volume + 1e-8)

# 创建时间索引，表示每个月的序号
data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
data["time_idx"] -= data["time_idx"].min()  # 使时间索引从 0 开始
# 计算每个 SKU 的平均成交量
data["avg_volume_by_sku"] = data.groupby(
    ["time_idx", "sku"], observed=True
).volume.transform("mean")
# 计算每个代理的平均成交量
data["avg_volume_by_agency"] = data.groupby(
    ["time_idx", "agency"], observed=True
).volume.transform("mean")
# data = data[lambda x: (x.sku == data.iloc[0]["sku"]) & (x.agency == data.iloc[0]["agency"])] # noqa: E501
# 定义特殊日期
special_days = [
    "easter_day",
    "good_friday",
    "new_year",
    "christmas",
    "labor_day",
    "independence_day",
    "revolution_day_memorial",
    "regional_games",
    "fifa_u_17_world_cup",
    "football_gold_cup",
    "beer_capital",
    "music_fest",
]
# 将特殊日期转换为类别类型
data[special_days] = (
    data[special_days].apply(lambda x: x.map({0: "", 1: x.name})).astype("category")
)

# 设置训练截止时间，留出最后 6 个月作为验证集
training_cutoff = data["time_idx"].max() - 6
max_encoder_length = 36  # 最大编码长度
max_prediction_length = 6  # 最大预测长度

# 创建训练数据集
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],  # 训练数据集的时间范围
    time_idx="time_idx",  # 时间索引字段
    target="volume",  # 目标变量
    group_ids=["agency", "sku"],  # 分组 ID
    min_encoder_length=max_encoder_length // 2,  # 允许的最小编码长度
    max_encoder_length=max_encoder_length,  # 最大编码长度
    min_prediction_length=1,  # 最小预测长度
    max_prediction_length=max_prediction_length,  # 最大预测长度
    static_categoricals=["agency", "sku"],  # 静态分类变量
    static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],  # 静态实数变量
    time_varying_known_categoricals=["special_days", "month"],  # 时间变化的已知分类变量
    variable_groups={
        "special_days": special_days
    },  # 将特殊日期分组为一个变量
    time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],  # 时间变化的已知实数变量
    time_varying_unknown_categoricals=[],  # 时间变化的未知分类变量
    time_varying_unknown_reals=[
        "volume",
        "log_volume",
        "industry_volume",
        "soda_volume",
        "avg_max_temp",
        "avg_volume_by_agency",
        "avg_volume_by_sku",
    ],  # 时间变化的未知实数变量
    target_normalizer=GroupNormalizer(
        groups=["agency", "sku"], transformation="softplus", center=False
    ),  # 使用 softplus 进行目标归一化
    add_relative_time_idx=True,  # 添加相对时间索引
    add_target_scales=True,  # 添加目标缩放
    add_encoder_length=True,  # 添加编码长度
)

# 创建验证数据集
validation = TimeSeriesDataSet.from_dataset(
    training, data, predict=True, stop_randomization=True
)
batch_size = 64  # 批次大小
# 创建训练数据加载器
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
# 创建验证数据加载器
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size, num_workers=0
)

# 保存数据集
training.save("t raining.pkl")
validation.save("validation.pkl")

# 设置早停回调
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)
# 设置学习率监控
lr_logger = LearningRateMonitor()
# 创建 TensorBoard 日志记录器
logger = TensorBoardLogger(save_dir="lightning_logs", log_graph=True)
# 创建 Trainer 实例，负责训练过程
trainer = pl.Trainer(
    max_epochs=100,  # 最大训练轮数
    accelerator="cpu",  # 使用 CPU 进行训练
    gradient_clip_val=0.1,  # 梯度裁剪值
    limit_train_batches=30,  # 限制训练批次数量
    # val_check_interval=20,
    # limit_val_batches=1,
    # fast_dev_run=True,
    logger=logger,  # 日志记录器
    # profiler=True,
    callbacks=[lr_logger, early_stop_callback],  # 回调函数列表
)

# 创建 Temporal Fusion Transformer 模型
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,  # 学习率
    hidden_size=16,  # 隐藏层大小
    attention_head_size=1,  # 注意力头的数量
    dropout=0.1,  # dropout 概率
    hidden_continuous_size=8,  # 隐藏连续变量的大小
    output_size=7,  # 输出大小
    loss=QuantileLoss(),  # 损失函数
    log_interval=10,  # 日志记录间隔
    log_val_interval=1,  # 验证日志记录间隔
    reduce_on_plateau_patience=3,  # 学习率调度的耐心值
)
print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

# # find optimal learning rate
# # remove logging and artificial epoch size
# tft.hparams.log_interval = -1
# tft.hparams.log_val_interval = -1
# trainer.limit_train_batches = 1.0
# # run learning rate finder
# res = Tuner(trainer).lr_find(
#     tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, min_lr=1e-5, max_lr=1e2 # noqa: E501
# )
# print(f"suggested learning rate: {res.suggestion()}")
# fig = res.plot(show=True, suggest=True)
# fig.show()
# tft.hparams.learning_rate = res.suggestion()

# trainer.fit(
#     tft,
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader,
# )

# # make a prediction on entire validation set
# preds, index = tft.predict(val_dataloader, return_index=True, fast_dev_run=True)

# 调优超参数
study = optimize_hyperparameters(
    train_dataloader,  # 训练数据加载器
    val_dataloader,  # 验证数据加载器
    model_path="optuna_test",  # 模型保存路径
    n_trials=200,  # 试验次数
    max_epochs=50,  # 最大训练轮数
    gradient_clip_val_range=(0.01, 1.0),  # 梯度裁剪值范围
    hidden_size_range=(8, 128),  # 隐藏层大小范围
    hidden_continuous_size_range=(8, 128),  # 隐藏连续变量大小范围
    attention_head_size_range=(1, 4),  # 注意力头数量范围
    learning_rate_range=(0.001, 0.1),  # 学习率范围
    dropout_range=(0.1, 0.3),  # dropout 概率范围
    trainer_kwargs=dict(limit_train_batches=30),  # Trainer 的参数
    reduce_on_plateau_patience=4,  # 学习率调度的耐心值
    use_learning_rate_finder=False,  # 是否使用学习率查找器
)
with open("test_study.pkl", "wb") as fout:
    pickle.dump(study, fout)

# profile speed
# profile(
#     trainer.fit,
#     profile_fname="profile.prof",
#     model=tft,
#     period=0.001,
#     filter="pytorch_forecasting",
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader,
# )
