from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:                                                                     
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except Exception:                                                               
    torch = None                            
    nn = None                            
    DataLoader = None                            
    TensorDataset = None                            
    _TORCH_AVAILABLE = False

from src.config import DeepLearningConfig, TrainingConfig


def torch_available() -> bool:

    return _TORCH_AVAILABLE


def _ensure_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the deep-learning models. Install with "
            "`pip install torch`."
        )


                                                                             
                     
                                                                             


def _make_lstm_module(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    num_classes: int,
):
    _ensure_torch()

    class LstmClassifier(nn.Module):                              
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):                          
            output, _ = self.rnn(x)
            return self.fc(self.dropout(output[:, -1, :]))

    return LstmClassifier()


def _make_gru_module(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    num_classes: int,
):
    _ensure_torch()

    class GruClassifier(nn.Module):                              
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):                          
            output, _ = self.rnn(x)
            return self.fc(self.dropout(output[:, -1, :]))

    return GruClassifier()


def _make_cnn_module(
    input_size: int,
    channels: int,
    kernel_size: int,
    dropout: float,
    num_classes: int,
):
    _ensure_torch()

    class Cnn1dClassifier(nn.Module):                              
        def __init__(self) -> None:
            super().__init__()
            padding = max(kernel_size // 2, 1)
            self.conv1 = nn.Conv1d(input_size, channels, kernel_size=kernel_size, padding=padding)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(channels, num_classes)

        def forward(self, x):                          
                                                                         
            x = x.transpose(1, 2)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            x = self.dropout(x)
            return self.fc(x)

    return Cnn1dClassifier()


def build_model(model_name: str, input_size: int, dl_cfg: DeepLearningConfig, num_classes: int):
    name = model_name.lower()
    if name == "lstm":
        return _make_lstm_module(
            input_size, dl_cfg.hidden_size, dl_cfg.num_layers, dl_cfg.dropout, num_classes
        )
    if name == "gru":
        return _make_gru_module(
            input_size, dl_cfg.hidden_size, dl_cfg.num_layers, dl_cfg.dropout, num_classes
        )
    if name == "cnn1d":
        return _make_cnn_module(
            input_size, dl_cfg.cnn_channels, dl_cfg.cnn_kernel_size, dl_cfg.dropout, num_classes
        )
    raise ValueError(f"Unknown deep-learning model: {model_name}")


                                                                             
         
                                                                             


@dataclass
class TrainingHistory:
    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    best_val_loss: float
    train_time_sec: float = 0.0
    inference_time_sec: float = 0.0


def _set_torch_seed(seed: int) -> None:
    if not _TORCH_AVAILABLE:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DeepSequenceClassifier:

    def __init__(
        self,
        model_name: str,
        input_size: int,
        dl_cfg: DeepLearningConfig,
        train_cfg: TrainingConfig,
        num_classes: int = 2,
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        _ensure_torch()
        _set_torch_seed(seed)
        self.model_name = model_name
        self.input_size = input_size
        self.dl_cfg = dl_cfg
        self.train_cfg = train_cfg
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(model_name, input_size, dl_cfg, num_classes).to(self.device)
        self.history: Optional[TrainingHistory] = None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainingHistory:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.dl_cfg.learning_rate)

        train_loader = self._make_loader(x_train, y_train, shuffle=True)
        val_loader = self._make_loader(x_val, y_val, shuffle=False)

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(self.model.state_dict())
        patience = self.train_cfg.early_stopping_patience
        no_improve = 0

        start = time.perf_counter()
        for epoch in range(1, self.train_cfg.max_epochs + 1):
            train_losses.append(self._run_epoch(train_loader, criterion, optimizer, train=True))
            val_losses.append(self._run_epoch(val_loader, criterion, optimizer=None, train=False))
            if val_losses[-1] < best_val - 1e-6:
                best_val = val_losses[-1]
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        train_time = time.perf_counter() - start

        self.model.load_state_dict(best_state)
        self.history = TrainingHistory(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            best_val_loss=best_val,
            train_time_sec=train_time,
        )
        return self.history

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        loader = self._make_loader(x, np.zeros(len(x), dtype=np.int64), shuffle=False)
        outputs: list[np.ndarray] = []
        start = time.perf_counter()
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                outputs.append(logits.argmax(dim=1).cpu().numpy())
        if self.history is not None:
            self.history.inference_time_sec = time.perf_counter() - start
        return np.concatenate(outputs) if outputs else np.empty((0,), dtype=np.int64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        loader = self._make_loader(x, np.zeros(len(x), dtype=np.int64), shuffle=False)
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                probs = torch.softmax(self.model(batch_x), dim=1)
                outputs.append(probs.cpu().numpy())
        return (
            np.concatenate(outputs, axis=0)
            if outputs
            else np.empty((0, self.num_classes), dtype=np.float32)
        )

    def _make_loader(self, x: np.ndarray, y: np.ndarray, shuffle: bool):
        x_t = torch.tensor(np.asarray(x, dtype=np.float32))
        y_t = torch.tensor(np.asarray(y, dtype=np.int64))
        dataset = TensorDataset(x_t, y_t)
        return DataLoader(
            dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def _run_epoch(
        self,
        loader,
        criterion,
        optimizer=None,
        train: bool = True,
    ) -> float:
        if train:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0.0
        total_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            if train:
                optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                if train:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            total_count += batch_x.size(0)
        return total_loss / max(total_count, 1)


def fit_dl_model(
    model_name: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    dl_cfg: DeepLearningConfig,
    train_cfg: TrainingConfig,
) -> Tuple[DeepSequenceClassifier, TrainingHistory]:

    classifier = DeepSequenceClassifier(
        model_name=model_name,
        input_size=x_train.shape[-1],
        dl_cfg=dl_cfg,
        train_cfg=train_cfg,
        seed=seed,
    )
    history = classifier.fit(x_train, y_train, x_val, y_val)
    return classifier, history
