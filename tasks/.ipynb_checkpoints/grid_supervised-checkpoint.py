import os

import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch import device
from torch.cpu.amp import autocast

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


class GridForecastTask(LightningModule):
    def __init__(
            self,
            # long_model: nn.Module,
            # short_model: nn.Module，
            mask,
            model: nn.Module,
            regressor="linear",
            loss="mse",
            pre_len: int = 3,
            learning_rate: float = 1e-3,
            weight_decay: float = 1.5e-3,
            feat: list = None,
            adj_list: list = None,
            default_root_path: str = None,
            exp_dir: str = None,
            use_normalize: int = 1,
            batch_size:int=10,
            # **kwargs
    ):

        super(GridForecastTask, self).__init__()

        self.model = model
        self.feat = feat  # [max, min, mean, std]
        self.adj_list = adj_list
        self.root_path = default_root_path
        self.exp_dir = exp_dir
        self.save_hyperparameters(ignore=['model'])
        self.use_normalize = use_normalize
        self._loss = loss
        self.configure_save()
        self.mask = mask
        self._device = None

    def forward(self, batch, verbose=False):
        x, y, current_index = batch["x"], batch["y"], batch["current_index"]
        if verbose:
            print("inputs:")
            print(f"x.shape = {x.shape}")
            for key, val in batch.items():
                if hasattr(val, "shape"):
                    print(f"{key}.shape = {val.shape}")
        hidden = self.model(x)
        return hidden

    def shared_step(self, batch):
        # (batch_size, seq_len/pre_len, num_nodes)
        # x, y, current_index = batch["x"], batch["y"], batch["current_index"]
        # print(x.shape,y.shape)
        # num_nodes = batch['x'].size(2)
        y = batch["y"]
        with autocast():  # 自动转换为 FP16
            predictions = self(batch)

        # print(predictions.shape, y.shape)
        # print("prediction, y", predictions.shape, y.shape)
        return predictions, y

    def loss(self, inputs, targets):
        if self._device is None:
            self._device = inputs.device
        if self._loss == "mse":
            return utils.losses.masked_mse(inputs, targets, self.mask.to(self._device))
        if self._loss == "mse_with_regularizer":
            print(self._loss)
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self.mask.to(self._device))
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch):
        # torch.cuda.empty_cache()
        predictions, y = self.shared_step(batch)
        # print(predictions.shape, y.shape)
        loss = self.loss(predictions, y)
        print("training loss:", loss.item())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        # torch.cuda.empty_cache()
        predictions, y = self.shared_step(batch)
        predictions = predictions * self.feat[3] + self.feat[2]
        y = y * self.feat[3] + self.feat[2]
        loss = self.loss(predictions, y)
        rmse = utils.metrics.root_mean_squared_error(predictions, y, self.mask.to(self._device))
        mae = utils.metrics.mean_absolute_error(predictions, y, self.mask.to(self._device))
        accuracy = utils.metrics.grid_accuracy(predictions, y, self.mask.to(self._device))
        r2 = utils.metrics.r2(predictions, y, self.mask.to(self._device))
        rmpe = utils.metrics.root_mean_position_error(predictions, y, self.mask.to(self._device))
        explained_variance = utils.metrics.explained_variance(predictions, y, self.mask.to(self._device))
        metrics = {
            "val_loss": loss,
            "RMSE": rmse,
            "MAE": mae,
            "accuracy": accuracy,
            "R2": r2,
            "ExplainedVar": explained_variance,
            'rmpe': rmpe,
        }
        for key, val in metrics.items():
            self.log(key, val, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return predictions.reshape(batch['y'].size()), y.reshape(batch['y'].size())

    def test_step(self, batch):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

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
        """
        Default kwargs used when initializing pl.Trainer
        """
        # print(kwargs)
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
            # accelerator="gpu",
            accelerator="cpu",
            strategy=DDPStrategy(find_unused_parameters=False),
            # strategy=ApexDDPStrategy(find_unused_parameters=False, delay_allreduce=True),
            # optimization
            max_epochs=kwargs.get('max_epochs'),
            check_val_every_n_epoch=2,
            gradient_clip_val=1.0,
            # NVIDIA amp
            inference_mode=False,
        )

        # ret.update(kwargs)
        return ret

