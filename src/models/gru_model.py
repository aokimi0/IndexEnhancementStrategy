"""GRU 时序 alpha 模型（CPU 友好）。

模型结构与参数量约束：
    * 1 层单向 GRU，``hidden_size=32``，``dropout=0.1``；
    * 输出端为 ``hidden_size -> 1`` 的线性回归头；
    * 输入维度为特征数 ``F``，参数量约为 ``3 * 32 * (F + 32) + 33`` ≈ 5K（远小于 50K 上限）。

为每条 (股票, 交易日) 样本构造过去 ``sequence_length`` 天的滚动特征窗口，缺失或不足处零填充。
特征重要性使用 *permutation importance*（shuffle 单个特征后观测 RMSE 增量）。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import AlphaModelBase


class _GRURegressor(nn.Module):
    """单层 GRU + 线性回归头。"""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 32,
        dropout: float = 0.1,
    ) -> None:
        """初始化网络。

        Args:
            input_dim: 输入特征维度。
            hidden_size: GRU 隐藏单元数。
            dropout: GRU 输出后的 dropout 比例。
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算并返回最后一个时间步的回归输出。

        Args:
            x: 形状为 ``(batch, seq_len, n_features)`` 的张量。

        Returns:
            torch.Tensor: 形状为 ``(batch,)`` 的预测向量。
        """
        output, _ = self.gru(x)
        last = self.dropout(output[:, -1, :])
        return self.head(last).squeeze(-1)


class GRUAlphaModel(AlphaModelBase):
    """基于 GRU 时序网络的 alpha 模型，固定在 CPU 上训练与推断。"""

    def __init__(
        self,
        feature_columns: list[str],
        label_column: str = "label_excess_return_20d",
        train_months: int = 12,
        min_train_rows: int = 2000,
        sequence_length: int = 20,
        hidden_size: int = 32,
        dropout: float = 0.1,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 20,
        early_stop_patience: int = 3,
        val_ratio: float = 0.1,
        random_state: int = 42,
        permutation_repeats: int = 1,
    ) -> None:
        """初始化 GRU 模型与训练参数。

        Args:
            feature_columns: 特征列。
            label_column: 标签列。
            train_months: 滚动窗口月数。
            min_train_rows: 最小训练样本数。
            sequence_length: 时序窗口长度（默认 20 个交易日）。
            hidden_size: GRU 隐藏单元数。
            dropout: dropout 比例。
            batch_size: 训练 batch 大小。
            learning_rate: AdamW 学习率。
            weight_decay: AdamW 权重衰减。
            max_epochs: 最大训练 epoch 数。
            early_stop_patience: early stopping 等待 epoch 数。
            val_ratio: 用于 early stopping 的内部验证集比例。
            random_state: 随机种子。
            permutation_repeats: 每个特征 permutation importance 的重复次数。
        """
        super().__init__(
            feature_columns=feature_columns,
            label_column=label_column,
            train_months=train_months,
            min_train_rows=min_train_rows,
        )
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.early_stop_patience = early_stop_patience
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.permutation_repeats = permutation_repeats
        self._device = torch.device("cpu")
        self._sequences: np.ndarray | None = None
        self._panel_size: int = 0

    def prepare(self, panel: pd.DataFrame) -> None:
        """为整个面板预生成 (n_rows, sequence_length, n_features) 序列缓存。

        Args:
            panel: 已经过 :func:`src.models.base.prepare_panel` 处理的面板。
        """
        self._panel_size = len(panel)
        self._sequences = self._build_sequences(panel)

    def _build_sequences(self, panel: pd.DataFrame) -> np.ndarray:
        """为面板每一行构造其过去 ``sequence_length`` 天的特征滚动窗口。

        Args:
            panel: 含 ``trade_date``、``ts_code`` 与特征列的面板，索引应为连续整数。

        Returns:
            np.ndarray: 形状 ``(n_rows, sequence_length, n_features)`` 的 ``float32`` 数组，
                行序与 ``panel.index`` 对齐；不足 ``sequence_length`` 天时在序列前端零填充。
        """
        n_rows = len(panel)
        n_features = len(self.feature_columns)
        sequences = np.zeros(
            (n_rows, self.sequence_length, n_features), dtype=np.float32
        )
        feature_array = panel[self.feature_columns].fillna(0.0).to_numpy(
            dtype=np.float32
        )
        for _, group in panel.groupby("ts_code", sort=False):
            indices = group.index.to_numpy()
            group_features = feature_array[indices]
            n = len(indices)
            for i in range(n):
                start = max(0, i - self.sequence_length + 1)
                seq = group_features[start : i + 1]
                pad_len = self.sequence_length - seq.shape[0]
                if pad_len > 0:
                    pad = np.zeros((pad_len, n_features), dtype=np.float32)
                    seq = np.concatenate([pad, seq], axis=0)
                sequences[indices[i]] = seq
        return sequences

    def fit_predict_batch(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        compute_importance: bool = True,
    ) -> dict[str, Any]:
        """对训练子集训练 GRU，并对测试子集打分。

        Args:
            train_frame: 训练子集。
            test_frame: 测试子集。
            compute_importance: 是否计算 permutation importance。代价较高，OOF 场景建议关闭。

        Returns:
            dict[str, Any]: 含 ``predictions``（``np.ndarray``）与 ``importance``
                （``dict[str, float]``）；当 ``compute_importance=False`` 时重要性为零字典。

        Raises:
            RuntimeError: 当未先调用 :meth:`prepare` 构建序列缓存时抛出。
        """
        if self._sequences is None:
            raise RuntimeError(
                "GRUAlphaModel.fit_predict_batch 必须先调用 prepare(panel) 构建序列缓存。"
            )
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        train_idx = train_frame.index.to_numpy()
        test_idx = test_frame.index.to_numpy()
        x_train_full = self._sequences[train_idx]
        y_train_full = train_frame[self.label_column].to_numpy(dtype=np.float32)
        x_test = self._sequences[test_idx]
        y_test = test_frame[self.label_column].fillna(0.0).to_numpy(dtype=np.float32)

        x_train, y_train, x_val, y_val = self._split_train_val(
            x_train_full, y_train_full, ratio=self.val_ratio
        )
        model = self._train(x_train, y_train, x_val, y_val)
        predictions = self._infer(model, x_test).astype(np.float64)
        if compute_importance and len(x_test) > 0:
            importance = self._permutation_importance(model, x_test, y_test)
        else:
            importance = {name: 0.0 for name in self.feature_columns}
        return {"predictions": predictions, "importance": importance}

    def _split_train_val(
        self,
        x: np.ndarray,
        y: np.ndarray,
        ratio: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """随机切分训练集为 (train, val) 两部分。

        Args:
            x: 训练特征序列。
            y: 训练标签。
            ratio: 验证集占比。

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                ``(x_train, y_train, x_val, y_val)``。当样本数过少时验证集与训练集相同。
        """
        n = len(x)
        if ratio <= 0 or n < 10:
            return x, y, x, y
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(n)
        n_val = max(1, int(n * ratio))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        return x[train_idx], y[train_idx], x[val_idx], y[val_idx]

    def _train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> _GRURegressor:
        """训练 GRU 并按 early stopping 选择最佳权重。

        Args:
            x_train: 训练特征序列，形如 (n_train, seq_len, n_features)。
            y_train: 训练标签，形如 (n_train,)。
            x_val: 验证特征序列。
            y_val: 验证标签。

        Returns:
            _GRURegressor: 训练完成且加载了最佳验证权重的模型实例（CPU）。
        """
        input_dim = x_train.shape[-1]
        model = _GRURegressor(
            input_dim=input_dim,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        ).to(self._device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()
        train_dataset = TensorDataset(
            torch.from_numpy(x_train).float(),
            torch.from_numpy(y_train).float(),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        val_tensor_x = torch.from_numpy(x_val).float().to(self._device)
        val_tensor_y = torch.from_numpy(y_val).float().to(self._device)

        best_val_loss = math.inf
        best_state: dict[str, torch.Tensor] | None = None
        bad_epochs = 0
        for _ in range(self.max_epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_pred = model(val_tensor_x)
                val_loss = float(criterion(val_pred, val_tensor_y).item())
            if val_loss + 1e-6 < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.early_stop_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    @torch.no_grad()
    def _infer(self, model: _GRURegressor, x: np.ndarray) -> np.ndarray:
        """按 batch 批量推断并返回 numpy 数组。

        Args:
            model: 已训练的 GRU。
            x: 输入序列 (n, seq_len, n_features)。

        Returns:
            np.ndarray: 长度为 ``n`` 的预测向量。
        """
        model.eval()
        if len(x) == 0:
            return np.zeros(0, dtype=np.float32)
        tensor = torch.from_numpy(x).float().to(self._device)
        outputs: list[torch.Tensor] = []
        for i in range(0, len(tensor), self.batch_size):
            batch = tensor[i : i + self.batch_size]
            outputs.append(model(batch).cpu())
        return torch.cat(outputs).numpy()

    def _permutation_importance(
        self,
        model: _GRURegressor,
        x: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        """对测试集做 permutation importance：shuffle 单特征后观测 RMSE 增量。

        Args:
            model: 已训练的 GRU。
            x: 测试特征序列。
            y: 测试标签（NaN 已填 0）。

        Returns:
            dict[str, float]: 特征名 -> 重要性得分（值越大越重要）。
        """
        baseline_pred = self._infer(model, x)
        baseline_rmse = float(np.sqrt(np.mean((baseline_pred - y) ** 2)))
        rng = np.random.default_rng(self.random_state)
        importance: dict[str, float] = {}
        for feature_idx, feature_name in enumerate(self.feature_columns):
            delta_sum = 0.0
            for _ in range(max(1, self.permutation_repeats)):
                shuffled = x.copy()
                perm = rng.permutation(shuffled.shape[0])
                shuffled[:, :, feature_idx] = shuffled[perm, :, feature_idx]
                pred = self._infer(model, shuffled)
                rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
                delta_sum += rmse - baseline_rmse
            importance[feature_name] = delta_sum / max(1, self.permutation_repeats)
        return importance
