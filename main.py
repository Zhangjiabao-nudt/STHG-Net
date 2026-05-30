import argparse
import os
import traceback
from pathlib import Path

import yaml
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from pygments.lexer import default
from pytorch_lightning.utilities import rank_zero_info
import Models, tasks
import utils.callbacks
import utils.data
import utils.email
import utils.logging
from lightning.pytorch import Trainer, seed_everything, loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch
# from utils.callbacks import DelayedEarlyStopping

# 推荐组合设置
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True  # 启用 cuDNN 自动调优

DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
    "sst": {'feat': "data/sstData-4-16/sstData.csv", "adj": "data/sstData-4-16/sstw.csv"},
}


def get_model(args, dm):
    model = None
    if args.model_name == "GCN":
        model = Models.GCN(adj=dm.adj, input_dim=args.seq_len, output_dim=args.hidden_dim)
    if args.model_name == "GRU":
        model = Models.GRU(input_dim=dm.adj.shape[0], hidden_dim=args.hidden_dim)
    if args.model_name == "TGCN":
        model = Models.TGCN(adj_shape=dm.adj.shape[0], hidden_dim=args.hidden_dim, d_model=args.d_model, seq_len=args.seq_len)
    if args.model_name == "CNN":
        model = Models.CNN(input_channels=1, output_timesteps=args.pre_len)
    if args.model_name == "ConvLSTM":
        model = Models.ConvLSTM(input_channels=1, output_timesteps=args.pre_len, hidden_channels=args.hidden_channels)
    if args.model_name == "CNNGRU":
        model = Models.CNNGRU(input_channels=1, output_timesteps=args.pre_len)
    if args.model_name == "TSGN":
        model = Models.TSGN(gcn_dims=args.gcn_dim, lstm_hidden=args.lstm_hidden, pred_steps=args.pre_len, aggrs=args.aggrs)
    if args.model_name == "OVPGCN":
        print("OVPGCN")
        model = Models.OVPGCN(in_feats_st=dm.in_feat_st, in_feats_bc=dm.in_feat_bc, in_feats_pdm=dm.in_feat_pdm,
                              hidden_feats=args.hidden_dim, pred_len=args.pre_len, out_feats=args.pre_len*dm.channels)
    if args.model_name == "EAGCN":
        print("EAGCN")
        model = Models.EAGCN(num_nodes=dm.adj.shape[0], input_dims=args.pre_len,
                             hidden_dim=args.hidden_dim, num_layers=args.num_layers, pred_steps=args.pre_len)
    if args.model_name == "AATGCN":
        print("AATGCN")
        model = Models.AATGCN(input_steps=args.seq_len, output_steps=args.pre_len, hidden_dim=args.hidden_dim,
                              features_dim=1)
    return model


def get_task(args, model, dm):
    # task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
    #     model=model, feat=dm.feat, adj_list=dm.adjacency_matrix(), pre_len=args.pre_len, learning_rate=args.learning_rate,
    #     weight_decay=args.weight_decay,default_root_path=args.root_dir, exp_dir = args.log_path,loss=args.loss,
    # )
    # task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
    #     model=model, feat=dm.feat, pre_len=args.pre_len, learning_rate=args.learning_rate, mask=dm.mask,
    #     weight_decay=args.weight_decay, default_root_path=args.root_dir, exp_dir=args.log_path, loss=args.loss,
    # )
    # task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
    #     model=model, feat=dm.feat, pre_len=args.pre_len, learning_rate=args.learning_rate, adj=dm.adj,
    #     weight_decay=args.weight_decay,default_root_path=args.root_dir, exp_dir = args.log_path,loss=args.loss,
    # )
    """ grid graph OVPGCN """
    task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
        model=model, pre_len=args.pre_len, learning_rate=args.learning_rate, mask=dm.mask, feat=dm.feat,adj=dm.adjacency_matrix,
        weight_decay=args.weight_decay, default_root_path=args.root_dir, exp_dir=args.log_path, loss=args.loss,
    )

    """ grid graph AATGCN"""
    # task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
    #     model=model, pre_len=args.pre_len, learning_rate=args.learning_rate, mask=dm.mask, feat=dm.feat,adj=dm.adjacency_matrix(),
    #     log=dm.log_u, weight_decay=args.weight_decay, default_root_path=args.root_dir, exp_dir=args.log_path, loss=args.loss,
    # )
    return task


def get_callbacks(args):
    # checkpoint_callback = ModelCheckpoint(monitor="train_loss")
    # ---- 初始化早停回调 ----
    # early_stop = DelayedEarlyStopping(
    #     monitor="val_loss",  # 监控的指标
    #     patience=15,  # 允许无改进的最大epoch数
    #     start_epoch=1
    # )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoint-{args.settings}",  # 保存路径
        filename="best-{epoch}-{val_loss:.2f}",  # 文件名格式
        monitor="val_loss",  # 监控指标
        mode="min",  # 最小化模式
        save_top_k=5,  # 只保留最好的1个
        auto_insert_metric_name=False,  # 简化文件名
        save_last=True,  # 同时保存last.ckpt
        every_n_epochs=1,  # 每epoch检查
        save_weights_only=False  # 保存完整模型
    )

    periodic_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_path, "periodic_checkpoints"),
        filename="periodic-{epoch:03d}",
        every_n_epochs=3,
        save_top_k=-1,  # 保存所有每5个epoch的模型
        save_on_train_epoch_end=False,)
    
    early_stop = EarlyStopping(
        monitor="val_loss_epoch",  # 监控的指标
        patience=15,  # 允许无改进的最大epoch数
        mode='min',
        verbose=True
    )
    kwargs = {'adj_mask_dir' : os.path.join(args.data_root,'data_gradient/gradient_mask_list.npz'), 'adj_number' : 200,
              'grid': 'grid', }
    # kwargs = {'adj_mask_dir' : os.path.join(args.data_root,'data_gradient/gradient_mask.npz'), 'adj_number' : 200,
    #           'grid': args.settings, }
    plot_validation_predictions_callback = utils.callbacks.PlotValidationPredictionsCallback(monitor=["train_loss", "val_loss"], **kwargs)
    callbacks = [
        checkpoint_callback,
        periodic_checkpoint_callback,
        # plot_validation_predictions_callback,
        early_stop,
    ]
    return callbacks


def main_grid(args):
    current_file = Path(__file__).resolve()
    args.data_root = r"/home/jameszhang/桌面/SST/data"
    args.root_dir = os.path.join(current_file.parent, 'experiments')
    args.log_path = os.path.join(args.root_dir, args.log_path)
    dm = utils.data.GridSpatioTemporalDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pre_len=args.pre_len,
        # move_step=10,
        normalize=args.normalize,
    )
    dm.prepare_data()
    dm.setup()
    model = get_model(args, dm)
    task = get_task(args, model, dm)
    callbacks = get_callbacks(args)

    trainer_args = task.set_trainer_kwargs(callbacks, **vars(args))
    trainer = Trainer(**trainer_args)
    trainer.fit(task, dm)
    results = trainer.validate(datamodule=dm)
    return results

def main_ovpgcn(args):
    current_file = Path(__file__).resolve()
    args.data_root = r"/home/jameszhang/桌面/SST/data_two"
    args.data_root = r"/root/autodl-tmp/SST"
    args.root_dir = os.path.join(current_file.parent, 'experiments')
    args.log_path = os.path.join(args.root_dir, args.log_path)
    dm = utils.data.OvpgcnDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pre_len=args.pre_len,
        normalize=args.normalize,
    )
    dm.prepare_data()
    dm.setup()
    model = get_model(args, dm)
    task = get_task(args, model, dm)
    callbacks = get_callbacks(args)

    trainer_args = task.set_trainer_kwargs(callbacks, **vars(args))
    # trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer = Trainer(**trainer_args)
    trainer.fit(task, datamodule=dm)
    results = trainer.validate(datamodule=dm, ckpt_path='last')
    return results


def main_eagcn(args):
    current_file = Path(__file__).resolve()
    args.root_dir = os.path.join(current_file.parent, 'experiments')
    args.log_path = os.path.join(args.root_dir, args.log_path)

    dm = utils.data.EagcnDataModule(
        data_root=r"/home/jameszhang/桌面/SST/data",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pre_len=args.pre_len,
        normalize=args.normalize,
    )
    dm.prepare_data()
    dm.setup()
    model = get_model(args, dm)
    task = get_task(args, model, dm)
    callbacks = get_callbacks(args)

    trainer_args = task.set_trainer_kwargs(callbacks, **vars(args))
    # trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer = Trainer(**trainer_args)
    trainer.fit(task, datamodule=dm)
    results = trainer.validate(datamodule=dm, ckpt_path='last')
    return results

def main_supervised(args):
    current_file = Path(__file__).resolve()
    args.data_root = r"/home/jameszhang/桌面/SST/data"
    args.root_dir = os.path.join(current_file.parent, 'experiments')
    args.log_path = os.path.join(args.root_dir, args.log_path)
    with open(current_file.parent /'config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    tsgn = config.get("tsgn", None)
    if tsgn:
        args.gcn_dim = tsgn.get("gcn_dims", None)
        args.lstm_hidden = tsgn.get("lstm_hidden", None)
        args.aggrs = tsgn.get("aggrs", None)
    dm = utils.data.SpatioTemporalDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pre_len=args.pre_len,
        normalize=args.normalize,
    )
    dm.prepare_data()
    dm.setup()
    model = get_model(args, dm)
    task = get_task(args, model, dm)
    callbacks = get_callbacks(args)

    trainer_args = task.set_trainer_kwargs(callbacks, **vars(args))
    # trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer = Trainer(**trainer_args)
    trainer.fit(task, dm)
    results = trainer.validate(datamodule=dm)
    return results

def main_aatgcn(args):
    current_file = Path(__file__).resolve()
    args.data_root = r"/home/jameszhang/桌面/SST/data" # r"/root/autodl-tmp/SST"
    args.root_dir = os.path.join(current_file.parent, 'experiments')
    args.log_path = os.path.join(args.root_dir, args.log_path)

    dm = utils.data.AatgcnDataModule(
        data_root= args.data_root,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pre_len=args.pre_len,
        normalize=args.normalize,
    )
    dm.prepare_data()
    dm.setup()
    model = get_model(args, dm)
    task = get_task(args, model, dm)
    callbacks = get_callbacks(args)

    trainer_args = task.set_trainer_kwargs(callbacks, **vars(args))
    # trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer = Trainer(**trainer_args)
    trainer.fit(task, datamodule=dm)
    results = trainer.validate(datamodule=dm)
    return results

def main(args):
    rank_zero_info(vars(args))
    results = globals()["main_" + args.settings](args)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--data", type=str, help="The name of the dataset", choices=("shenzhen", "losloop", "sst"), default="sst"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("GCN", "GRU", "TGCN", 'CNN', "CNNGRU", "ConvLSTM", "TSGN", "ConvLSTM", "TSGN", "OVPGCN", "EAGCN", "AATGCN"),
        default="TGCN",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised",'grid', 'ovpgcn', 'eagcn', 'test', 'aatgcn'),
        default="supervised",
    )

    parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")
    parser.add_argument("--send_email", "--email", action="store_true", help="Send email when finished")
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--gpus", type=int, choices=(int(0),int(1)), default=int(1))

    temp_args, _ = parser.parse_known_args()

    parser = getattr(utils.data, temp_args.settings.capitalize() + "DataModule").add_data_specific_arguments(parser)
    parser = getattr(Models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.capitalize() + "ForecastTask").add_task_specific_arguments(parser)


    args = parser.parse_args()

    try:
        results = main(args)
    except:  # noqa: E722
        traceback.print_exc()
        # if args.send_email:
        #     tb = traceback.format_exc()
        #     subject = "[Email Bot][❌] " + "-".join([args.settings, args.model_name, args.data])
        #     utils.email.send_email(tb, subject)
        exit(-1)

    # if args.send_email:
    #     subject = "[Email Bot][✅] " + "-".join([args.settings, args.model_name, args.data])
    #     utils.email.send_experiment_results_email(args, results, subject=subject)
