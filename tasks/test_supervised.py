import os
import torch.optim
import torch.nn as nn
from torch.cpu.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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


class TestForecastTask(LightningModule):
    def __init__(
        self,
        # long_model: nn.Module,
        # short_model: nn.Module,
        model: nn.Module,
        regressor="linear",
        loss="mse",
        pre_len: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat: list=None ,
        adj_list: list=None,
        default_root_path: str=None,
        exp_dir: str=None,
        **kwargs
    ):

        super(TestForecastTask, self).__init__()

        # self.long_sequence_model = long_model
        # self.short_term_model = short_model
        self.model = model
        self.feat= feat
        self.adj_list, self.long_adj_list = adj_list
        self.root_path = default_root_path
        self.exp_dir = exp_dir
        self.top_k = kwargs.get("top_k", 5)
        self.threshold = kwargs.get("threshold", 0.9)
        # self.device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.save_hyperparameters(ignore=['model'])

        self.regressor = (
            nn.Linear(
                self.model.hyperparameters.get("hidden_dim")
                or self.model.hyperparameters.get("output_dim"),
                self.hparams.pre_len,
            )
            if regressor == "linear"
            else regressor
        )
        self._loss = loss

        self.configure_save()

    # 使用adj
    def forward(self, batch, verbose=False):

        # (batch_size, seq_len, num_nodes)
        x, y, current_index, long_index= batch["x"], batch["y"], batch["current_index"], batch["long_index"]
        # print(type(x), type(y), type(current_index))  # 查看变量类型
        # print(current_index, current_index.shape)
        if verbose:
            print("inputs:")
            print(f"x.shape = {x.shape}")
            for key, val in batch.items():
                if hasattr(val, "shape"):
                    print(f"{key}.shape = {val.shape}")

        adj = []
        long_adj = []
        for index, i in enumerate(current_index):
            print(self.adj_list[i])
            adj_i = torch.load(self.adj_list[i]).float().to(x.device)
            adj.append(adj_i)

        for index, i in enumerate(long_index):
            print(self.long_adj_list[i])
            adj_i = torch.load(self.long_adj_list[i]).float().to(x.device)
            long_adj.append(adj_i)

        prediction_all = []
        for index, i in enumerate(zip(current_index, long_index)):
            x_batch_one = x[index]
            x_batch_one = x_batch_one.unsqueeze(0)
            batch_size, pre_len, num_nodes = x_batch_one.size()
            # print("num_nodes:",num_nodes)
            # (batch_size, num_nodes, hidden_dim)
            hidden = self.model(x_batch_one, adj[index], long_adj[index])
            # (batch_size * num_nodes, hidden_dim)
            hidden = hidden.reshape((-1, hidden.size(2)))
            # (batch_size * num_nodes, pre_len)
            if self.regressor is not None:
                predictions = self.regressor(hidden)
            else:
                predictions = hidden
            # print("output:",predictions.shape)
            predictions = predictions.reshape((batch_size, num_nodes, -1))
            prediction_all.append(predictions)

        del adj, long_adj
        return torch.stack(prediction_all, dim=0)

    # GRU 已撤销
    # TSGN
    # def forward(self, batch):
    #     x, y, current_index= batch["x"], batch["y"], batch["current_index"]
    #     # (batch_size, seq_len, num_nodes)
    #     batch_size, seq_len, num_nodes = x.size()
    #     edges_indexs = []
    #     with torch.no_grad():
    #         for i in range(batch_size):
    #             edges_index = generate_edge_index(x[i].permute(1, 0), top_k=self.top_k, threshold=self.threshold, device=x.device)
    #         edges_indexs.append(edges_index)
    #     # (batch_size, num_nodes, hidden_dim)
    #     x = x.unsqueeze(0).permute(1, 2, 3, 0).contiguous()
    #     hidden = self.model(x, edges_index)
    #     # (batch_size * num_nodes, hidden_dim)
    #     # hidden = hidden.reshape((-1, hidden.size(2)))
    #     # (batch_size * num_nodes, pre_len)
    #     # if self.regressor is not None:
    #     #     predictions = self.regressor(hidden)
    #     # else:
    #     #     predictions = hidden
    #     # predictions = predictions.reshape((batch_size, num_nodes, -1))
    #
    #     # return predictions
    #     return hidden[:,...,0]

    def on_train_start(self) -> None:
        for name, param in self.named_parameters():
            if param.grad is None:
                print(f"未使用的参数: {name}")

    def shared_step(self, batch):
        # (batch_size, seq_len/pre_len, num_nodes)
        # x, y, current_index = batch["x"], batch["y"], batch["current_index"]
        num_nodes = batch['x'].size(2)
        y = batch["y"]
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
        rmpe = utils.metrics.masked_mape(predictions, y)
        metrics = {
            "val_loss": loss,
            "RMSE": rmse,
            "MAE": mae,
            # "accuracy": accuracy,
            "R2": r2,
            # "ExplainedVar": explained_variance,
            'rmpe': rmpe,
        }
        for key, value in metrics.items():
            self.log(key, value, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        # ExplainedVar,MAE,R2,RMSE,accuracy,epoch,step,train_loss,val_loss
        # self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        # self.log('RMSE', rmse, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        # self.log('MAE', mae, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        # self.log('accuracy', accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, logger=True)
        # self.log('R2', r2, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, logger=True)
        # self.log('ExplainedVar', explained_variance, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, logger=True)
        # self.log('rmpe', rmpe, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        # self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        return predictions.reshape(batch['y'].size()), y.reshape(batch['y'].size())

    # def test_step(self, batch):
    #     pass

    def configure_optimizers(self):
        # return torch.optim.Adam(
        #     self.parameters(),
        #     lr=self.hparams.learning_rate,
        #     weight_decay=self.hparams.weight_decay,
        # )
        # return torch.optim.AdamW(
        #     self.parameters(),
        #     lr=self.hparams.learning_rate,
        #     weight_decay=self.hparams.weight_decay,
        # )
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
            check_val_every_n_epoch=2,
            gradient_clip_val=1.0,
            # NVIDIA amp
            # misc
            inference_mode=False,
        )

        # ret.update(kwargs)
        return ret


    def load_from_path(self, ckpt_dict):
        self.regressor.load_state_dict(ckpt_dict)



