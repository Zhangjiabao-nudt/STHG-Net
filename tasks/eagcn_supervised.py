import os
from operator import lshift
from typing import Union, IO, Optional, Any

from lightning.pytorch.callbacks import ModelCheckpoint
from networkx import edges, adjacency_matrix
from scipy.signal.windows import kaiser
from typing_extensions import Self, overload

import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.fabric.utilities.types import _PATH, _MAP_LOCATION_TYPE
from torch.cpu.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils.data.function import build_adjacency_matrix
import utils.metrics
import utils.losses
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch import LightningModule
import argparse
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from pygments.lexer import default
from pytorch_lightning.utilities import rank_zero_info
import Models, tasks
import utils.callbacks
import utils.data
import utils.email
import utils.logging
from lightning.pytorch import Trainer, seed_everything, loggers as pl_loggers


class EagcnForecastTask(LightningModule):
    def __init__(
        self,
        # long_model: nn.Module,
        # short_model: nn.Module,
        adj,
        model: nn.Module,
        regressor="linear",
        loss="mse",
        pre_len: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat: list=None ,
        default_root_path: str=None,
        exp_dir: str=None,
        **kwargs
    ):
        super(EagcnForecastTask, self).__init__()
        # self.long_sequence_model = long_model
        # self.short_term_model = short_model
        self.model = model
        self.feat= feat
        self.root_path = default_root_path
        self.exp_dir = exp_dir
        # self.device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.save_hyperparameters(ignore=['model', 'adj'])
        self.adj = torch.from_numpy(adj).float()

        # self.regressor = (
        #     nn.Linear(
        #         self.model.hyperparameters.get("hidden_dim")
        #         or self.model.hyperparameters.get("output_dim"),
        #         self.hparams.pre_len,
        #     )
        #     if regressor == "linear"
        #     else regressor
        # )
        self._loss = loss

        self.configure_save()

    # GRU 已撤销
    # OVPGCN
    def forward(self, batch):
        x, y, current_index= batch["x"], batch["y"], batch["current_index"]
        x_short, x_long = batch["x_short"], batch["x_long"]
        # (batch_size, seq_len, num_nodes)
        # (batch_size, num_nodes, hidden_dim)
        x = x.unsqueeze(0).permute(1, 2, 3, 0).contiguous()
        x_short = x_short.unsqueeze(0).permute(1, 2, 3, 0).contiguous()
        x_long = x_long.unsqueeze(0).permute(1, 2, 3, 0).contiguous()
        print(x.shape, x_short.shape, x_long.shape, "forward")
        hidden = self.model(x, x_short, x_long, self.adj)

        return hidden

    def on_train_start(self) -> None:
        for name, param in self.named_parameters():
            if param.grad is None:
                print(f"未使用的参数: {name}")

    def shared_step(self, batch):
        # (batch_size, seq_len/pre_len, num_nodes)
        # x, y, current_index = batch["x"], batch["y"], batch["current_index"]
        y = batch["y"]
        if self.adj.device != y.device:
            self.adj = self.adj.to(y.device)
            print(self.adj.device)
        num_nodes = batch['x'].size(2)
        with autocast():  # 自动转换为 FP16
            predictions = self(batch)
        # print("prediction, y", predictions.shape, y.shape)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes)).contiguous()
        y = y.reshape((-1, y.size(2))).contiguous()
        return predictions, y

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return utils.losses.masked_mse(inputs, targets)
            # return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch):
        torch.cuda.empty_cache()
        predictions, y = self.shared_step(batch)
        print(y.shape, predictions.shape)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        self.log("lr", self.optimizers().param_groups[0]['lr'])
        print(f"train_loss：{self._loss}", loss.item())
        return loss

    def test_step(self, batch):
        predictions, y = self.shared_step(batch)

        predictions = predictions * self.feat[3] + self.feat[2]
        y = y * self.feat[3] + self.feat[2]

        # predictions = predictions * self.feat[0] + self.feat[1]
        # y = y * self.feat[0] + self.feat[1]

        # predictions = predictions * self.feat[0]
        # y = y * self.feat[0]
        loss = self.loss(predictions, y)
        print('validation_loss', loss)
        rmse = utils.metrics.root_mean_squared_error(predictions, y)
        # rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        # mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        mae = utils.metrics.mean_absolute_error(predictions, y)
        accuracy = utils.metrics.accuracy(predictions, y)
        r2 = utils.metrics.r2(predictions, y)
        explained_variance = utils.metrics.explained_variance(predictions, y)
        rmpe = utils.metrics.masked_mape(predictions, y)
        metrics = {
            "val_loss": loss,
            "RMSE": rmse,
            "MAE": mae,
            "accuracy": accuracy,
            "R2": r2,
            "ExplainedVar": explained_variance,
            'rmpe': rmpe,
        }
        for key, value in metrics.items():
            self.log(key, value, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        return predictions, y

    def validation_step(self, batch):
        predictions, y = self.shared_step(batch)
        # mean std
        predictions = predictions * self.feat[3] + self.feat[2]
        y = y * self.feat[3] + self.feat[2]
        # min max
        # predictions = predictions * self.feat[0] + self.feat[1]
        # y = y * self.feat[0] + self.feat[1]

        # predictions = predictions * self.feat[0]
        # y = y * self.feat[0]
        loss = self.loss(predictions, y)
        print('validation_loss', loss)
        rmse = utils.metrics.root_mean_squared_error(predictions, y)
        # rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        # mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        mae = utils.metrics.mean_absolute_error(predictions, y)
        # accuracy = utils.metrics.accuracy(predictions, y)
        r2 = utils.metrics.r2(predictions, y)
        # explained_variance = utils.metrics.explained_variance(predictions, y)
        mape = utils.metrics.masked_mape(predictions, y)
        metrics = {
            "val_loss": loss,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            'mape': mape,
        }
        for key, value in metrics.items():
            self.log(key, value, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)

        return predictions.reshape(batch['y'].size()), y.reshape(batch['y'].size())


    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # 初始周期长度
            T_mult=2,  # 周期倍增系数
            eta_min=1e-4
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parser.add_argument("--loss", type=str, default="mse")
        return parser

    def configure_save(self):
        save_dir = os.path.join(self.root_path, self.exp_dir)
        os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def set_trainer_kwargs(callback, **kwargs):
        r"""
        Default kwargs used when initializing pl.Trainer
        """
        print(kwargs)
        logger = []
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=kwargs.get('log_path'))
        csv_logger = pl_loggers.CSVLogger(save_dir=kwargs.get('log_path'))
        logger += [tb_logger, csv_logger]

        ret = dict(
            callbacks=callback,
            # log
            logger=logger,
            log_every_n_steps=1,
            # save
            default_root_dir=kwargs.get('root_dir'),
            # ddp
            accelerator="gpu",
            # accelerator="cpu",
            # strategy=DDPStrategy(find_unused_parameters=False),
            strategy=DDPStrategy(find_unused_parameters=True),
            # strategy=ApexDDPStrategy(find_unused_parameters=False, delay_allreduce=True),
            # optimization
            max_epochs=kwargs.get('max_epochs'),
            check_val_every_n_epoch=1,
            # gradient_clip_val=1.0,
            # NVIDIA amp
            # misc
            inference_mode=False,
        )

        # ret.update(kwargs)
        return ret




