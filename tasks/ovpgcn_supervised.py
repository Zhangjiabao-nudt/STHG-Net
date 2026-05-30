import os
import numpy as np
from networkx import adjacency_matrix
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ConstantLR, SequentialLR
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
import utils.callbacks
import utils.data
import utils.email
import utils.logging
from lightning.pytorch import Trainer, seed_everything, loggers as pl_loggers


class AdaptiveCosineWR(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, base_T, T_mult_factor=1.5, eta_min=1e-4):

        # 必须显式初始化自定义属性
        self.base_T = int(base_T)
        self.T_mult_factor = float(T_mult_factor)
        self.current_restarts = 0  # 关键修复点
        self.best_loss = float('inf')
        self.last_val_loss = float('inf')
        super().__init__(optimizer, T_0=base_T, T_mult=1, eta_min=eta_min)

    def step(self, epoch=None):
        # 动态调整逻辑
        if self.last_val_loss < self.best_loss * 0.98:
            self.T_mult = 1
            self.best_loss = self.last_val_loss
        else:
            self.T_mult = self.T_mult_factor

        # 调用父类方法前确保属性存在
        if not hasattr(self, 'current_restarts'):
            self.current_restarts = 0

        super().step(epoch)

        # 周期计数器更新
        self.current_restarts += self.T_cur // self.T_i


class OvpgcnForecastTask(LightningModule):
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
        mask: list = None,
        **kwargs
    ):
        super(OvpgcnForecastTask, self).__init__()
        # self.long_sequence_model = long_model
        # self.short_term_model = short_model
        self.adj_list, self.per_adj = adj
        self.model = model
        self.feat= feat
        self.mask = mask
        self.root_path = default_root_path
        self.exp_dir = exp_dir
        self.mae_record = []
        self.rmse_record = []
        # self.device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # self.save_hyperparameters(ignore=['model', 'adj'])
        self.save_hyperparameters(ignore=['model', 'mask', 'feat', 'adj'])
        # self.adj = torch.from_numpy(adj).float()

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
        # adjacency_matrix = batch["adjacency_matrix"]
        # mask = self.mask[current_index.cpu()]
        # mask = mask.float().to(current_index.device)
        B, pre_len, num_nodes, _ = y.size()
        adj = []
        per_adj = []
        for index, i in enumerate(current_index):
            print(self.adj_list[i])
            adj_i = torch.load(self.adj_list[i]).float().to(x.device)
            # per_adj_i = torch.load(self.per_adj[i]).float().to(x.device)
            adj.append(adj_i)
            # per_adj.append(per_adj_i)

        # torch.cuda.empty_cache()
        # with open(
        #         '/media/jameszhang/Elements/4070/Summary of Graph Neural Network Code and Graph Theory/Summary of Graph Neural Network Code and Graph Theory/Model_0222/data_record.txt',
        #         'a') as f:  # 使用 'a' 模式以追加内容到文件
        #     print('path', self.adj_list[i], 'path_long', self.long_adj_list[j], 'path_idx', current_index, 'path_long_idx', long_index,
        #           file=f)

        prediction_all = []
        for index, i in enumerate(current_index):
            x_batch_one = x[index]
            x_short_batch, x_long_batch = x_short[index], x_long[index]
            x_batch_one = x_batch_one.unsqueeze(0)

            batch_size, seq_len, num_nodes, _ = x_batch_one.size()
            x_batch_one = x_batch_one.view(batch_size, seq_len, num_nodes*num_nodes, 1).contiguous()
            x_short_batch = x_short_batch.view(batch_size, -1, num_nodes*num_nodes, 1).contiguous()
            x_long_batch = x_long_batch.view(batch_size,-1, num_nodes*num_nodes, 1).contiguous()
            torch.cuda.empty_cache()
            per_adj_i = build_adjacency_matrix(x_batch_one[0, :, :, 0].permute(1, 0))
            # adj_i = torch.load(self.adj_list[i]).float().to(x.device)
            torch.cuda.empty_cache()
            print("ovpgcn_supervised", per_adj_i.shape, x_short_batch.shape, x_long_batch.shape, x_batch_one.shape)
            hidden = self.model(x_batch_one, x_short_batch, x_long_batch, adj[index], adj[index], adj[index], per_adj_i)
            # hidden = self.model(x_batch_one, x_short_batch, x_long_batch, adj_i, adj_i, adj_i, per_adj_i)
            hidden = hidden[:, ..., 0].reshape(batch_size, pre_len, num_nodes, num_nodes)
            print(hidden.shape)
            prediction_all.append(hidden)
            # print(prediction_all[0].shape)
        del adj, per_adj
        # print(torch.stack(prediction_all, dim=0).shape)
        prediction_all = torch.stack(prediction_all, dim=0)
        return prediction_all[0]

    def on_train_start(self) -> None:
        for name, param in self.named_parameters():
            if param.grad is None:
                print(f"未使用的参数: {name}")

    def shared_step(self, batch):
        # (batch_size, seq_len/pre_len, num_nodes)
        # x, y, current_index = batch["x"], batch["y"], batch["current_index"]
        current_index = batch["current_index"]
        mask = self.mask[current_index.cpu()]
        mask = mask.float().to(current_index.device)

        y = batch["y"]

        num_nodes = batch['x'].size(2)
        with autocast():  # 自动转换为 FP16
            predictions = self(batch)
        # print("prediction, y", predictions.shape, y.shape)
        # predictions = predictions.transpose(1, 2).reshape((-1, num_nodes)).contiguous()
        # y = y.reshape((-1, y.size(2))).contiguous()
        return predictions, y, mask

    def loss(self, inputs, targets, mask):
        if self._loss == "mse":
            return utils.losses.masked_mse(inputs, targets, mask)
            # return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, mask)
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch):
        torch.cuda.empty_cache()
        predictions, y, mask = self.shared_step(batch)
        print(y.shape, predictions.shape)
        loss = self.loss(predictions, y, mask)
        self.log("train_loss", loss)
        self.log("lr", self.optimizers().param_groups[0]['lr'])
        print(f"train_loss：{self._loss}", loss.item())
        return loss

    def test_step(self, batch):
        predictions, y, mask = self.shared_step(batch)
        time = batch['time'][0].to(torch.float).cpu().numpy()
        # mean std
        predictions = predictions * self.feat[3] + self.feat[2]
        y = y * self.feat[3] + self.feat[2]

        result = [
            predictions.detach().float().cpu().numpy(),  # prediction
            y.detach().float().cpu().numpy(),  # target
            mask.detach().float().cpu().numpy(),  # mask
        ]

        # 使用字典结构保存为 NPZ 文件
        np.savez_compressed(
            os.path.join(self.save_dir_result, str(time) +'.npz'),
            prediction=result[0],
            target=result[1],
            mask=result[2]
        )
        # min max
        # predictions = predictions * self.feat[0] + self.feat[1]
        # y = y * self.feat[0] + self.feat[1]
        # max
        # predictions = predictions * self.feat[0]
        # y = y * self.feat[0]
        loss = self.loss(y, predictions, mask)
        print('validation_loss', loss)
        rmse = utils.metrics.root_mean_squared_error(y, predictions, mask)
        # rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        # mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        mae = utils.metrics.mean_absolute_error(y, predictions, mask)
        # accuracy = utils.metrics.accuracy(predictions, y)
        r2 = utils.metrics.r2(y, predictions, mask)
        # explained_variance = utils.metrics.explained_variance(predictions, y)
        mape = utils.metrics.masked_mape(y, predictions, mask)
        mae_time = utils.metrics.mean_absolute_error_time(y, predictions, mask)
        rmse_time = utils.metrics.root_mean_squared_error_time(y, predictions, mask)
        metrics = {
            "val_loss": loss,
            "RMSE": rmse,
            "MAE": mae,
            # "accuracy": accuracy,
            "R2": r2,
            # "ExplainedVar": explained_variance,
            'mape': mape,
        }
        self.mae_record.append(mae_time.detach().cpu().numpy())
        self.rmse_record.append(rmse_time.detach().cpu().numpy())
        for key, value in metrics.items():
            self.log(key, value, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        return predictions.reshape(batch['y'].size()), y.reshape(batch['y'].size())
        
    def validation_step(self, batch):
        predictions, y, mask = self.shared_step(batch)
        # mean std
        predictions = predictions * self.feat[3] + self.feat[2]
        y = y * self.feat[3] + self.feat[2]
        # min max
        # predictions = predictions * self.feat[0] + self.feat[1]
        # y = y * self.feat[0] + self.feat[1]

        # predictions = predictions * self.feat[0]
        # y = y * self.feat[0]
        loss = self.loss(y, predictions, mask)
        print('validation_loss', loss)
        rmse = utils.metrics.root_mean_squared_error(y, predictions, mask)
        # rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        # mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        mae = utils.metrics.mean_absolute_error(y, predictions, mask)
        # accuracy = utils.metrics.accuracy(predictions, y)
        r2 = utils.metrics.r2(y, predictions, mask)
        # explained_variance = utils.metrics.explained_variance(predictions, y)
        mape = utils.metrics.masked_mape(y, predictions, mask)
        metrics = {
            "val_loss": loss,
            "RMSE": rmse,
            "MAE": mae,
            # "R2": r2,
            'mape': mape,
        }
        for key, value in metrics.items():
            self.log(key, value, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)

        return predictions.reshape(batch['y'].size()), y.reshape(batch['y'].size())

    def on_test_epoch_end(self):
        # 确保保存目录存在
        save_dir = "./ovpgcn_metrics_results"  # 替换为你的特定路径
        save_dir = os.path.join(self.save_dir, save_dir)
        os.makedirs(os.path.join(save_dir, save_dir), exist_ok=True)

        # 将列表转换为numpy数组
        mae_array = np.array(self.mae_record)
        rmse_array = np.array(self.rmse_record)

        # 保存到特定路径
        np.save(os.path.join(save_dir, "mae_record.npy"), mae_array)
        np.save(os.path.join(save_dir, "rmse_record.npy"), rmse_array)
        
    def configure_optimizers(self):

        # optimizer = torch.optim.AdamW(
        #     self.parameters(),
        #     lr=self.hparams.learning_rate,
        #     weight_decay=self.hparams.weight_decay,
        # )
        # # optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        #
        # scheduler = CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=10,  # 初始周期长度
        #     T_mult=2,  # 周期倍增系数
        #     eta_min=1e-4
        # )
        # return [optimizer], [scheduler]
        # 初始化优化器
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # 第一阶段：前5个epoch固定学习率
        warmup_scheduler = ConstantLR(
            optimizer,
            factor=1.0,  # 保持初始学习率不变
            total_iters=5
        )

        adaptive_scheduler = AdaptiveCosineWR(
            optimizer,
            base_T=10,  # 初始周期长度
            T_mult_factor=1.2,  # 周期延长系数
            eta_min=1e-4
        )

        # 组合调度器
        combined_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, adaptive_scheduler],
            milestones=[5]  # 15个epoch后切换
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": combined_scheduler,
                "interval": "epoch",
                "monitor": "val_loss"  # 需要监控验证损失
            }
        }

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parser.add_argument("--loss", type=str, default="mse")
        return parser

    def configure_save(self):
        self.save_dir = os.path.join(self.root_path, self.exp_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        save_dir_result = "./ovpgcn_results_120-30"  # 替换为你的特定路径
        self.save_dir_result = os.path.join(self.save_dir, save_dir_result)
        os.makedirs(os.path.join(self.save_dir_result), exist_ok=True)
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




