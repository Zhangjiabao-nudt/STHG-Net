import argparse
import os
import traceback
from pathlib import Path
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from networkx import adjacency_matrix
from pygments.lexer import default
from pytorch_lightning.utilities import rank_zero_info
import Models, tasks
import utils.callbacks
import utils.data
import utils.email
import utils.logging
from lightning.pytorch import Trainer, seed_everything, loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
# from tasks import (SupervisedForecastTask, TestForecastTask, GridForecastTask,
#                    ChgnnForecastTask)
from tasks import OvpgcnForecastTask, ChgnnForecastTask

# 推荐组合设置
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True  # 启用 cuDNN 自动调优

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

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
        model = Models.TGCN(adj=dm.adj, hidden_dim=args.hidden_dim,d_model=args.d_model, seq_len=args.seq_len)
    if args.model_name == "CNN":
        model = Models.CNN(input_channels=1, output_timesteps=args.pre_len)
    if args.model_name == "ConvLSTM":
        model = Models.ConvLSTM(input_channels=1, output_timesteps=args.pre_len, hidden_channels=args.hidden_channels)
    if args.model_name == "CNNGRU":
        model = Models.CNNGRU(input_channels=1, output_timesteps=args.pre_len)
    if args.model_name == "CHGNN":
        model = Models.CHGNN(adj_shape=dm.adj.shape[0], hidden_dim=args.hidden_dim, d_model=args.d_model, seq_len=args.seq_len)
    if args.model_name == 'OVPGCN':
        model = Models.OVPGCN(in_feats_st=dm.in_feat_st, in_feats_bc=dm.in_feat_bc, in_feats_pdm=dm.in_feat_pdm,
                              hidden_feats=args.hidden_dim, pred_len=args.pre_len, out_feats=args.pre_len*dm.channels)
    return model


def get_task(args, model, dm):
    # graph
    # task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
    #     model=model, feat=dm.feat, adj_list=dm.adjacency_matrix(), pre_len=args.pre_len, learning_rate=args.learning_rate,
    #     weight_decay=args.weight_decay,default_root_path=args.root_dir, exp_dir = args.log_path,loss=args.loss,
    # )
    """grid"""
    # task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
    #     model=model, feat=dm.feat, pre_len=args.pre_len, learning_rate=args.learning_rate, mask=dm.mask,
    #     weight_decay=args.weight_decay, default_root_path=args.root_dir, exp_dir=args.log_path, loss=args.loss,
    # )
    """ grid graph """
    # task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
    #     model=model, pre_len=args.pre_len, learning_rate=args.learning_rate, mask=dm.mask, adj_list=dm.adjacency_matrix,feat=dm.feat,
    #     weight_decay=args.weight_decay, default_root_path=args.root_dir, exp_dir=args.log_path, loss=args.loss,
    # )
    """ OVPGCN """
    task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
        model=model, pre_len=args.pre_len, learning_rate=args.learning_rate, mask=dm.mask, feat=dm.feat,adj=dm.adjacency_matrix(),
        weight_decay=args.weight_decay, default_root_path=args.root_dir, exp_dir=args.log_path, loss=args.loss,
    )
    return task


def get_callbacks(args):
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoint-{args.settings}",  # 保存路径
        filename="best-{epoch}-{val_loss:.2f}",  # 文件名格式
        monitor="val_loss",  # 监控指标
        mode="min",  # 最小化模式
        save_top_k=1,  # 只保留最好的1个
        auto_insert_metric_name=False,  # 简化文件名
        save_last=True,  # 同时保存last.ckpt
        every_n_epochs=1,  # 每epoch检查
        save_weights_only=False  # 保存完整模型
    )
    # kwargs = {'adj_mask_dir' : os.path.join(args.data_root,'data_gradient/gradient_mask.npz'), 'adj_number' : 200,
    #           'grid': args.settings, }
    kwargs = {'adj_mask_dir' : os.path.join(args.data_root,'data_gradient/gradient_mask_list.npz'), 'adj_number' : 200,
              'grid': 'grid', }
    plot_validation_predictions_callback = utils.callbacks.PlotValidationPredictionsCallback(monitor="val_loss", **kwargs)
    plot_test_predictions_callback = utils.callbacks.PlotTestPredictionsCallback(monitor="val_loss", **kwargs)
    callbacks = [
        checkpoint_callback,
        # plot_validation_predictions_callback,
        # plot_test_predictions_callback,
    ]
    return callbacks


# def main_supervised(args):
#     """
#     测试已训练好的模型
#     :param args: 包含所有配置参数的命名空间对象（需与训练时参数一致）
#     :return: 测试结果字典
#     """
#     current_file = Path(__file__).resolve()
#     args.data_root = r"/home/jameszhang/桌面/SST/data"
#     args.root_dir = os.path.join(current_file.parent, 'experiments')
#     args.log_path = os.path.join(args.root_dir, args.log_path)
#     args.ckpt_path = os.path.join(args.root_dir, args.log_path, f'lightning_logs/{args.ckpt_path}')
#     dm = utils.data.SpatioTemporalDataModule(
#         data_root= args.data_root,
#         seq_len=30,
#         pre_len=7,
#         normalize=args.normalize,
#     )
#     dm.prepare_data()
#     dm.setup(stage="test")
#     callbacks = get_callbacks(args)
#
#     # model = get_model(args, dm)
#     # 加载并修复检查点
#     # checkpoint = torch.load(ckpt_path, map_location='cpu')
#     # for k, v in checkpoint.items():
#     #     print(k)
#     # modified_dict = {}
#     # regressor ={}
#     # # 遍历原始字典
#     # for key, value in checkpoint['state_dict'].items():
#     #     if key.startswith('model.'):
#     #         new_key = key.replace('model.', '')
#     #         modified_dict[new_key] = value
#     #     elif key.startswith('regressor.'):
#     #         new_key = key.replace('regressor.', '')
#     #         regressor[new_key] = value
#     # model.load_state_dict(modified_dict)
#     # task = get_task(args, model, dm)
#     # task.load_from_path(regressor)
#
#     if args.ckpt_path.endswith(".ckpt"):
#         # 方式1：通过类直接调用load_from_checkpoint（修复错误）
#         task = SupervisedForecastTask.load_from_checkpoint(
#             checkpoint_path=args.ckpt_path,
#             model=get_model(args, dm),  # 必须重新实例化模型
#             dm=None,
#             strict=True  # 确保权重与模型结构完全匹配
#         )
#     trainer = Trainer(
#         **task.set_trainer_kwargs(callbacks, **vars(args))
#     )
#     # 执行测试
#     trainer.test(task, datamodule=dm)
#     # 执行训练
#     # trainer.fit(task, datamodule=dm, ckpt_path=args.ckpt_path)
#     # 执行验证+画图
#     # trainer.validate(datamodule=dm)
#     results = trainer.validate(datamodule=dm, ckpt_path=args.ckpt_path)
#     return results


# def main_grid(args):
#     args.data_root = r"/home/jameszhang/桌面/SST/data"
#     current_file = Path(__file__).resolve()
#     args.root_dir = os.path.join(current_file.parent, 'experiments')
#     args.log_path = os.path.join(args.root_dir, args.log_path)
#     args.ckpt_path = os.path.join(args.root_dir, args.log_path, f'lightning_logs/{args.ckpt_path}')
#     dm = utils.data.GridSpatioTemporalDataModule(
#         data_root=r"/home/jameszhang/桌面/SST/data",
#         batch_size=args.batch_size,
#         seq_len=30,
#         pre_len=7,
#         move_step=10,
#         normalize=args.normalize,
#     )
#     dm.prepare_data()
#     dm.setup(stage="test")
#     callbacks = get_callbacks(args)
#     if args.ckpt_path.endswith(".ckpt"):
#         # 方式1：通过类直接调用load_from_checkpoint（修复错误）
#         task = GridForecastTask.load_from_checkpoint(
#             checkpoint_path=args.ckpt_path,
#             model=get_model(args, dm),  # 必须重新实例化模型
#             dm=None,
#             strict=True  # 确保权重与模型结构完全匹配
#         )
#     trainer = Trainer(
#         **task.set_trainer_kwargs(callbacks, **vars(args))
#     )
#     # 执行测试
#     # trainer.test(task, datamodule=dm)
#     # 执行训练
#     trainer.fit(task, datamodule=dm, ckpt_path=args.ckpt_path)


def main(args):
    rank_zero_info(vars(args))
    results = globals()["main_" + args.settings](args)
    return results

def main_test(args):
    args.data_root = r"/home/jameszhang/桌面/SST/data"
    current_file = Path(__file__).resolve()
    args.root_dir = os.path.join(current_file.parent, 'experiments')
    args.log_path = os.path.join(args.root_dir, args.log_path)
    args.ckpt_path = os.path.join(args.root_dir, args.log_path, f'lightning_logs/{args.ckpt_path}')
    # dm = utils.data.SpatioTemporalDataModuleTest(
    #     data_root=args.data_root,
    #     seq_len=args.seq_len,
    #     pre_len=args.pre_len,
    #     normalize=args.normalize,
    # )
    dm = utils.data.SpatioTemporalDataModuleOVPGCN(
        data_root=args.data_root,
        seq_len=args.seq_len,
        pre_len=args.pre_len,
        normalize=args.normalize,
    )
    dm.prepare_data()
    dm.setup(stage="test")
    # callbacks = get_callbacks(args)
    # if args.ckpt_path.endswith(".ckpt"):
    #     方式1：通过类直接调用load_from_checkpoint（修复错误）
        # task = TestForecastTask.load_from_checkpoint(
        #     checkpoint_path=args.ckpt_path,
        #     model=get_model(args, dm),  # 必须重新实例化模型
        #     dm=None,
        #     strict=True  # 确保权重与模型结构完全匹配
        # )
    # trainer = Trainer(
    #     **task.set_trainer_kwargs(callbacks, **vars(args))
    # )
    # 执行测试
    # trainer.test(task, datamodule=dm)
    # 执行训练
    # trainer.fit(task, datamodule=dm, ckpt_path=args.ckpt_path)
    # 执行验证+画图
    # # trainer.validate(datamodule=dm)
    # # results = trainer.validate(datamodule=dm, ckpt_path=args.ckpt_path, model=task)
    #
    # # return results

def main_ovpgcn(args):
    # args.data_root = r"E:\PaperWork\SST"
    # args.data_root = r"/home/jameszhang/桌面/SST/data"
    args.data_root = r"/root/autodl-tmp/SST"
    current_file = Path(__file__).resolve()
    args.root_dir = os.path.join(current_file.parent, 'experiments')
    args.log_path = os.path.join(args.root_dir, args.log_path)
    args.ckpt_path = os.path.join(args.root_dir, args.log_path, f'lightning_logs/{args.ckpt_path}')
    dm = utils.data.SpatioTemporalDataModuleOVPGCN(
        data_root=args.data_root,
        seq_len=args.seq_len,
        pre_len=args.pre_len,
        batch_size=args.batch_size,
        normalize=args.normalize,
    )
    dm.prepare_data()
    dm.setup(stage="test")
    callbacks = get_callbacks(args)

    if args.ckpt_path.endswith(".ckpt"):
        # 方式1：通过类直接调用load_from_checkpoint（修复错误）
        task = OvpgcnForecastTask.load_from_checkpoint(
            checkpoint_path=args.ckpt_path,
            model=get_model(args, dm),  # 必须重新实例化模型
            dm=None,
            strict=True,  # 确保权重与模型结构完全匹配
            adj=dm.adjacency_matrix,
            feat=dm.feat, weight_decay=args.weight_decay,
            mask=dm.mask, default_root_path=args.root_dir, exp_dir=args.log_path,
        )
    trainer = Trainer(
        **task.set_trainer_kwargs(callbacks, **vars(args))
    )
    trainer.test(task, datamodule=dm, ckpt_path=args.ckpt_path)
    # trainer.fit(task, datamodule=dm, ckpt_path=args.ckpt_path)

def main_chgnn(args):
    # args.data_root = r"/home/jameszhang/桌面/SST/data_two"
    args.data_root = r"/root/autodl-tmp/SST"
    # args.data_root = r"/home/jameszhang/桌面/SST/data"
    current_file = Path(__file__).resolve()
    args.root_dir = os.path.join(current_file.parent, 'experiments')
    args.log_path = os.path.join(args.root_dir, args.log_path)
    args.ckpt_path = os.path.join(args.root_dir, args.log_path, f'lightning_logs/{args.ckpt_path}')
    dm = utils.data.SpatioTemporalDataModuleCHGNN(
        data_root=args.data_root,
        seq_len=args.seq_len,
        pre_len=args.pre_len,
        batch_size=args.batch_size,
        normalize=args.normalize,
    )
    dm.prepare_data()
    dm.setup(stage="test")
    # batch = next(iter(dm.train_dataloader()))
    # model = get_model(args, dm)
    # task = get_task(args, model, dm)
    callbacks = get_callbacks(args)

    # trainer_args = task.set_trainer_kwargs(callbacks, **vars(args))
    # trainer = Trainer(**trainer_args)
    # trainer.fit(task, dm)
    # results = trainer.validate(datamodule=dm)
    # return results

    if args.ckpt_path.endswith(".ckpt"):
        # 方式1：通过类直接调用load_from_checkpoint（修复错误）
        task = ChgnnForecastTask.load_from_checkpoint(
            checkpoint_path=args.ckpt_path,
            model=get_model(args, dm),  # 必须重新实例化模型
            dm=None,
            strict=True,  # 确保权重与模型结构完全匹配
            adj_list=dm.adjacency_matrix,
            feat=dm.feat, weight_decay=args.weight_decay,
            mask=dm.mask, default_root_path=args.root_dir, exp_dir=args.log_path
        )
    trainer = Trainer(
        **task.set_trainer_kwargs(callbacks, **vars(args))
    )
    # trainer.fit(task, datamodule=dm)
    # trainer.test(task, datamodule=dm)
    trainer.test(task, datamodule=dm, ckpt_path=args.ckpt_path)
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)

    # parser.add_argument(
    #     "--data", type=str, help="The name of the dataset", choices=("shenzhen", "losloop", "sst"), default="sst"
    # )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("GCN", "GRU", "TGCN", 'CNN', "CNNGRU", "ConvLSTM", "CHGNN", 'OVPGCN'),
        default="TGCN",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised",'grid', 'test', "chgnn", 'ovpgcn'),
        default="supervised",
    )

    parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")
    parser.add_argument('--ckpt_path', type=str, default=None, help="Path to the checkpoint path")
    parser.add_argument("--send_email", "--email", action="store_true", help="Send email when finished")
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--gpus", type=int, choices=(int(0),int(1)), default=int(1))

    temp_args, _ = parser.parse_known_args()

    """1"""
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


# python test.py --model_name TGCN --max_epochs 200 --learning_rate 0.001 --weight_decay 0.0005 --batch_size 1 --loss mse
# --settings supervised --gpus 1 --log_path test_TGCN
# --ckpt_path version_13/checkpoints/epoch=35-step=30384.ckpt --pre_len 7 --hidden_dim 128
# epoch=26-step=22788.ckpt

"""
--model_name TGCN --max_epochs 108 --learning_rate 0.001 --weight_decay 0.0005 --batch_size 1 --loss mse --settings test 
--gpus 1 --log_path test_TGCN --ckpt_path version_30/checkpoints/epoch=107-step=113940.ckpt --pre_len 7 --hidden_dim 128

python main.py --model_name ConvLSTM --max_epochs 200 --learning_rate 0.001 --weight_decay 0.0015
--batch_size 1 --loss mse --settings grid  --gpus 1 --log_path test_ConvLSTM --pre_len 7 --ckpt_path version_16/checkpoints/epoch=33-step=28696.ckpt

--model_name
CHGNN
--max_epochs
97
--learning_rate
0.001
--weight_decay
0.0015
--batch_size
1
--loss
mse
--settings
chgnn
--gpus
1
--log_path
test_CHGNN
--pre_len
1
--seq_len
7
--hidden_dim
128
--hidden_dim
64
--d_model
16
--ckpt_path
version_2/checkpoints/epoch=95-step=203136.ckpt
"""

"""
--model_name OVPGCN 
--max_epochs 99 
--learning_rate 0.001 
--weight_decay 0.0015 
--batch_size 1 
--loss mse 
--settings ovpgcn --gpus 1 
--log_path test_OVPGCN_grid --pre_len 30 --hidden_dim 64 --seq_len 120 --pred_steps 30 --batch_size 1
"""