import os
import argparse

from fontTools.misc.plistlib import end_integer
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from networkx import adjacency_matrix
from pyarrow import float32
# from triton.language import dtype

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # global environment
import re
from os import mkdir
import numpy as np
import pytorch_lightning as pl
from lightning.pytorch import LightningDataModule
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch
from tqdm import tqdm
from utils.data.function import compute_mmap_statistics, compute_yearly_gradient, compute_daily_gradient, \
    compute_and_save_graph, \
    compute_and_save_adjacency, generate_adjacency_matrix_np, build_adjacency_matrix
from utils.data.function_other_model import generate_static_adj, generate_pearson_adj


def dict_collate(batch):
    """处理字典列表的合并"""
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }


def is_path_empty(path):
    try:
        return len(os.listdir(path)) == 0
    except Exception as e:
        print(f"Error: {e}")
        return False


def load_data_from_file(filename):
    # 使用正则表达式匹配文件名中的str和int部分
    try:
        match = re.match(r'(.+)_(\d+)_(.+)\.npy', filename)
    except Exception as e:
        print(filename)
        raise ValueError("Filename does not match the expected format")

    # 提取str和int部分
    str_part1 = match.group(1)
    int_part = int(match.group(2))
    str_part2 = match.group(3)

    return str_part1, int_part, str_part2


class NPZFileLoaderOVPGCN:
    def __init__(self, data_root, seq_len, pre_len):
        self.data_root = Path(data_root)
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.time_delta = np.timedelta64(1, 'D')
        self.data_dir = self.data_root / 'all_npz'
        # self.data_dir = self.data_root / 'all_data'
        self.gradient_data_year_dir = self.data_root / 'data_gradient_year'
        self.gradient_data_dir = self.data_root / 'data_gradient'
        self.gradient_data_half_year_dir = self.data_root / 'data_gradient_half_year'
        if not self.gradient_data_half_year_dir.exists(): mkdir(self.gradient_data_half_year_dir)
        if not self.gradient_data_year_dir.exists(): mkdir(self.gradient_data_year_dir)
        if not self.gradient_data_dir.exists(): mkdir(self.gradient_data_dir)
        self.spatio = 'spatio'
        self.temporal = 'temporal'

        # 加载并预处理数据
        self.file_list = sorted([f for f in self.data_dir.glob("*.npz") if f.stem.isdigit()])
        self.time_stamps = [int(f.stem) for f in self.file_list]
        self.start_year = 1992
        self.end_year = 2020
        self.start_idx = 0
        self.end_idx = int(len(self.file_list))
        self.move_step = 10
        self.valid_gradient = False

        # 验证时间连续性
        # self._validate_time_sequence()

        # 创建内存映射文件
        self.mmap_path = self.data_dir / "sst_mmap.npy"
        if not self.mmap_path.exists():
            self._create_memmap('npz')
        self.sst_mmap = np.load(
            self.mmap_path,
            mmap_mode='r', allow_pickle=True)

        # 创建metric文件
        self.metric_path = self.data_root / "sst_features.npz"
        if not self.metric_path.exists():
            # memory is enough
            # self._create_metric()
            # memory is not enough
            compute_mmap_statistics(self.mmap_path, self.metric_path, safe_chunk_mb=3000)
        self.feat = np.load(self.metric_path)

        # 创建年梯度文件
        self.temporal_dir = self.gradient_data_year_dir / self.temporal
        if not self.temporal_dir.exists():
            mkdir(self.temporal_dir)
            compute_yearly_gradient(self.sst_mmap, self.temporal_dir, self.start_year, self.end_year,
                                    spatio_temporal=False)

        self.spatio_dir = self.gradient_data_year_dir / self.spatio
        if not self.spatio_dir.exists():
            mkdir(self.spatio_dir)
            compute_yearly_gradient(self.sst_mmap, self.spatio_dir, self.start_year, self.end_year)

        self.temporal_gradient_file_list = sorted([f for f in self.temporal_dir.glob("*_*.npy")])
        self.spatio_gradient_file_list = sorted([f for f in self.spatio_dir.glob("*_*.npy")])

        self.spatio_path = self.spatio_dir / 'mmap.npy'
        if not self.spatio_path.exists():
            self._create_memmap('spatio')
        self.temporal_path = self.temporal_dir / 'mmap.npy'
        if not self.temporal_path.exists():
            self._create_memmap('temporal')

        self.spatio_gradient = np.load(self.spatio_path, mmap_mode='r', allow_pickle=True)
        self.temporal_gradient = np.load(self.temporal_path, mmap_mode='r', allow_pickle=True)

        self.valid_indices = self._calculate_indices()
        self.valid_indices_half = self._calculate_indices_half()
        # 创建日梯度文件
        if is_path_empty(self.gradient_data_dir):
            compute_daily_gradient(self.sst_mmap, self.gradient_data_dir, self.valid_indices, self.seq_len,
                                   self.pre_len, self.time_stamps)

        self.gradient_file_list = sorted([f for f in self.gradient_data_dir.glob("*_*_*.npy")])

        self.gradient_mmap_path = self.gradient_data_dir / "mmap.npy"
        if not self.gradient_mmap_path.exists():
            self._create_memmap('gradient')
        self.gradient = np.load(self.gradient_mmap_path, mmap_mode='r', allow_pickle=True)
        # print(self.gradient.shape)

        """对每一个seq_len+pre_len应用对应的mask"""
        self.mask_dir = self.gradient_data_dir / 'gradient_mask_list.npz'
        if not self.mask_dir.exists():
            self.get_mask(self.mask_dir)
        self.mask_list = np.load(self.mask_dir)["mask"]
        print(type(self.mask_list), len(self.mask_list), self.mask_list.shape)

        # 创建长时间梯度文件
        if is_path_empty(self.gradient_data_half_year_dir):
            compute_daily_gradient(self.sst_mmap, self.gradient_data_half_year_dir, self.valid_indices_half, 182,
                                   0, self.time_stamps)

        self.gradient_file_list = sorted([f for f in self.gradient_data_half_year_dir.glob("*_*_*.npy")])

        self.gradient_mmap_path = self.gradient_data_half_year_dir / "mmap.npy"
        if not self.gradient_mmap_path.exists():
            self._create_memmap('gradient')
        self.gradient_long = np.load(self.gradient_mmap_path, mmap_mode='r', allow_pickle=True)

        if self.valid_gradient:
            # 验证梯度文件是否生成正确
            self._validate_gradient()

        # 寻找最大nan值mask——使用统一的mask矩阵以实现节点数据的读取
        self.mask_dir = self.gradient_data_dir / 'gradient_mask.npz'
        if not self.mask_dir.exists():
            self.find_mask_max()
        self.max_mask = np.load(self.mask_dir)["mask"]

        print(np.sum(self.max_mask), self.max_mask.shape)

        """对每一个seq_len+pre_len应用对应的mask"""
        self.mask_dir = self.gradient_data_dir / 'gradient_mask_list.npz'
        if not self.mask_dir.exists():
            self.get_mask(self.mask_dir)
        self.mask_list = np.load(self.mask_dir)["mask"]
        print(type(self.mask_list), len(self.mask_list), self.mask_list.shape)


        # self.adj_dir = self.gradient_data_dir / 'adj.npz'
        # if not self.adj_dir.exists():
        #     generate_adjacency_matrix_np(self.max_mask, file_path=self.adj_dir)
        # self.adjacency_matrix = np.load(self.adj_dir)['adj']
        # print(self.adjacency_matrix.shape)
        print(self.sst_mmap.shape)

        """generate static adj"""
        self.static_adj_path = self.data_root / 'OVPGCN_static_adj'
        if not self.static_adj_path.exists(): mkdir(self.static_adj_path)
        if is_path_empty(self.static_adj_path):
            generate_static_adj(self.valid_indices, self.mask_list,
                                     self.time_stamps, self.static_adj_path, self.seq_len+self.pre_len)
        self.static_adj_file_list = sorted([f for f in self.static_adj_path.glob("*.pth") if f.stem.isdigit()])

        """generate pearson adj"""
        self.pearson_path = self.data_root / 'OVPGCN_pearson'
        if not self.pearson_path.exists(): mkdir(self.pearson_path)
        # if is_path_empty(self.pearson_path):
        #     generate_pearson_adj(self.sst_mmap, self.valid_indices, self.time_stamps,
        #                         self.pearson_path, self.seq_len+self.pre_len)
        self.pearson_list = sorted([f for f in self.pearson_path.glob("*.pth") if f.stem.isdigit()])

        # print(self.static_adj_file_list, self.pearson_list)

    def _validate_gradient(self):
        for idx, file in enumerate(tqdm(self.gradient_file_list, desc='处理进度')):
            _, file_data_name, _ = load_data_from_file(str(file))
            index_file = self.valid_indices[idx]
            data2 = self.sst_mmap[index_file:index_file + self.seq_len + self.pre_len + 1]
            daily_data = np.where(data2 == 0, np.nan, data2)
            gradient_3d = np.gradient(daily_data, edge_order=1)
            if np.isnan(gradient_3d[1]).all() and np.isnan(gradient_3d[2]).all():
                raise "all is nan"
            temperature_gradient_magnitude = np.nanmean(np.sqrt(gradient_3d[1] ** 2 + gradient_3d[2] ** 2), axis=0)
            mask1 = ~np.isnan(temperature_gradient_magnitude)
            mask = ~np.isnan(self.gradient[idx])
            print(np.sum(~mask1))
            print(np.sum(~mask))
            data1 = self.gradient[idx][mask]
            tem = temperature_gradient_magnitude[mask1]
            if not np.allclose(self.gradient[idx], temperature_gradient_magnitude, rtol=1e-05, atol=1e-08,
                               equal_nan=True):
                raise ValueError("data is not equal")

            if not np.equal(data1, tem).all():
                raise ValueError("data is not equal")
        print("验证完成：梯度生成无异常")

    def _calculate_indices(self):
        """计算有效索引范围"""
        return range(
            self.start_idx,
            self.end_idx - (self.seq_len + self.pre_len) + 1,
            self.move_step
        )

    def _calculate_indices_half(self):
        return range(
            self.start_idx,
            self.end_idx - 182,
            182,
        )

    def find_mask_max(self):
        max_mask = None
        for i, f in enumerate(tqdm(self.gradient_file_list, desc="寻找最大mask")):
            daily_gradient = self.gradient[i]

            if i == 0:
                max_mask = ~np.isnan(daily_gradient)
            else:
                current_non_nan = ~np.isnan(daily_gradient)
                max_mask = max_mask & current_non_nan  # 逻辑与操作
        print(np.sum(max_mask))
        np.savez(f'{self.gradient_data_dir}/gradient_mask.npz', mask=max_mask)

    def _validate_time_sequence(self):
        # 实现时间连续性检查
        dates = np.array(
            [np.datetime64(f'{str(date)[:4]}-{str(date)[4:6]}-{str(date)[6:8]}', 'D') for date in self.time_stamps])
        # print(dates)
        # 计算相邻日期差是否为1天（向量化操作）
        diffs = np.diff(dates)
        cond = np.logical_or(diffs == self.time_delta, diffs == self.time_delta + self.time_delta)
        if not (np.all(cond)):
            raise ValueError("时间序列不连续或间隔不一致")

    def _create_memmap(self, file_type):
        # 预扫描所有文件
        shapes = []
        dtypes = []
        if file_type == 'npz':
            mmap_path = str(self.mmap_path)
            for f in self.file_list:
                data = np.load(f)['sst']
                shapes.append(data.shape)
                dtypes.append(data.dtype)
            file_list = self.file_list
        elif file_type == 'temporal':
            mmap_path = str(self.temporal_path)
            for f in self.temporal_gradient_file_list:
                data = np.load(f)
                shapes.append(data.shape)
                dtypes.append(data.dtype)
            file_list = self.temporal_gradient_file_list
        elif file_type == 'spatio':
            mmap_path = str(self.spatio_path)
            for f in self.spatio_gradient_file_list:
                data = np.load(f)
                shapes.append(data.shape)
                dtypes.append(data.dtype)
            file_list = self.spatio_gradient_file_list
        elif file_type == 'gradient':
            mmap_path = str(self.gradient_mmap_path)
            for f in self.gradient_file_list:
                data = np.load(f)
                shapes.append(data.shape)
                dtypes.append(data.dtype)
            file_list = self.gradient_file_list
        else:
            raise f'{file_type} is not supported'

        unique_shapes = set(shapes)
        print(unique_shapes)
        if len(unique_shapes) != 1:
            raise ValueError(f"发现{len(unique_shapes)}种不同数据维度")

        # 确定统一类型
        final_dtype = np.result_type(*dtypes)
        target_shape = (len(file_list), *shapes[0])

        # 创建内存映射
        mmap_arr = open_memmap(
            mmap_path,
            dtype=final_dtype,
            mode='w+',
            shape=target_shape
        )
        # 带进度条的写入
        for i, f in enumerate(tqdm(file_list, desc="创建内存映射")):
            if file_type == 'npz':
                data = np.load(f)['sst'].astype(final_dtype)
            else:
                data = np.load(f).astype(final_dtype)
            mmap_arr[i] = data
            if i % 100 == 0:
                mmap_arr.flush()
        mmap_arr.flush()
        del mmap_arr

    def _create_metric(self, mmap):
        self.feat_max_val = np.max(mmap)  # 根据实际需求调整标准化策略
        self.feat_min_val = np.min(mmap)
        self.feat_mean = np.mean(mmap)
        self.feat_std = np.std(mmap)
        np.savez(self.metric_path, max=self.feat_max_val, min=self.feat_min_val,
                 mean=self.feat_mean, std=self.feat_std)

    def get_mask(self, file_path):
        mask_list = []
        for i, f in enumerate(tqdm(self.gradient_file_list, desc="得到mask list")):
            daily_gradient = self.gradient[i]
            daily_gradient = ~np.isnan(daily_gradient)
            daily_gradient = daily_gradient.reshape(1, daily_gradient.shape[0], daily_gradient.shape[1])
            mask_list.append(daily_gradient)
        mask_list = np.concatenate(mask_list)
        # print(mask_list.shape)
        np.savez(file_path, mask=mask_list)



class SpatioTemporalDataset(Dataset):
    def __init__(self, mmap_data, spatio_gradient, temporal_gradient, start_idx, end_idx, seq_len,
                 pre_len, features, device,  gradient_mmap, max_mask, move_step=10, normalize=2,
                 test_list=None, time_list=None):
        self.mmap_data = mmap_data
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.normalize = normalize
        self.spatio_gradient = spatio_gradient
        self.temporal_gradient = temporal_gradient
        self.device = device
        self.gradient_mmap = gradient_mmap
        # 需要添加到config
        self.use_spatio = True
        self.valid_mask = max_mask
        self._init_features(features)
        # 索引计算
        self.move_step = move_step
        self.valid_indices = range(
            start_idx,
            end_idx - (self.seq_len + self.pre_len) + 1,
            move_step
        )
        self.test_list = test_list
        self.time_delta = [180, 30, 0]
        self.time_list = time_list

    def _init_features(self, features):
        """初始化标准化参数并预存到GPU"""
        self.feat_max_val = np.round(features["max"], 4)
        self.feat_min_val = np.round(features["min"], 4)
        self.feat_mean = np.round(features["mean"], 4)
        self.feat_std = np.round(features["std"], 4)

    def __len__(self):
        if self.test_list is None:
            return len(self.valid_indices)
        else:
            return len(self.test_list)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx] if self.test_list is None else self.valid_indices[self.test_list[idx]]
        # print(actual_idx)
        # 读取时间窗口数据
        # adj = torch.load(self.adj_file_list[idx])
        window = self.mmap_data[actual_idx:actual_idx + self.seq_len + self.pre_len, 600:800, 400:600]
        window_long = self.mmap_data[actual_idx - 180:actual_idx, 600:800, 400:600]
        window_short = self.mmap_data[actual_idx - 30:actual_idx, 600:800, 400:600]
        # index = int(actual_idx / 365)
        # if self.use_spatio:
        #     gradient = self.spatio_gradient[index]
        # else:
        #     gradient = self.temporal_gradient[index]

        # 标准化处理
        # 2. 数据标准化（在CPU进行）
        if self.normalize == 0:
            print('max initial')
            window = window / self.feat_max_val
            window_long = window_long / self.feat_max_val
            window_short = window_short / self.feat_max_val
        elif self.normalize == 1:
            print('min max initial')
            window = (window - self.feat_min_val) / self.feat_max_val
            window_long = (window_long - self.feat_min_val) / self.feat_max_val
            window_short = (window_short - self.feat_min_val) / self.feat_max_val
        elif self.normalize == 2:
            print('mean std initial')
            window = (window - self.feat_mean) / self.feat_std
            window_long = (window_long - self.feat_mean) / self.feat_std
            window_short = (window_short - self.feat_mean) / self.feat_std


        x = window[:self.seq_len]
        y = window[self.seq_len:self.seq_len + self.pre_len]
        # adj = build_adjacency_matrix(torch.from_numpy(x.reshape(-1, 200*200).transpose(1, 0)).float())

        # 直接通过三维数组索引提取有效节点
        # x = x[:, self.valid_mask]
        # x_short = window_short[:, self.valid_mask]
        # x_long = window_long[:, self.valid_mask]
        # y = y[:, self.valid_mask]

        x_short = window_short[:, :, :]
        x_long = window_long[:, :, :]
        # x = x[None, :, :]
        # y = y[None, :, :]
        # print(x.shape, x_short.shape, x_long.shape)
        # print("ovpgcn_data:", x.shape, x_short.shape, x_long.shape)
        return {
            'x': torch.from_numpy(x).float(),
            'x_short': torch.from_numpy(x_short).float(),
            'x_long': torch.from_numpy(x_long).float(),
            'y': torch.from_numpy(y).float(),
            'current_index': torch.tensor(actual_idx // self.move_step).int(),
            # 'adjacency_matrix': torch.from_numpy(adj).float(),
            'long_index': torch.tensor(actual_idx // 182).int(),
            'time': self.time_list[actual_idx + self.seq_len:actual_idx + self.seq_len + self.pre_len],
            # 'mask': mask_matrix,
            # 'gradient': gradient,
            # 'adj_indices': adj.indices(),  # 稀疏张量的坐标 (2, nnz)  # 报错未处理重复点
            # 'adj_values': adj.values(),  # 稀疏张量的值 (nnz,)
            # 'adj_size': torch.tensor(adj.shape)  # 原始形状 (2,)
        }


class SpatioTemporalDataModuleOVPGCN(LightningDataModule):
    def __init__(
            self,
            data_root: str,
            batch_size: int = 1,
            seq_len: int = 7,
            pre_len: int = 3,
            split_ratio: float = 0.8,
            normalize: int = 1,
            bc: int = 30,
            pdm: int = 180,
            channles: int = 1,
            **kwargs
    ):
        super().__init__()
        self.in_feat_pdm = None
        self.in_feat_bc = None
        self.in_feat_st = None
        self.feat = None
        self.adj = None
        self.mask = None
        self.data_root = data_root
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        self.bc = bc
        self.pdm = pdm
        self.channels = channles

        # 延迟初始化
        self.file_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

    def prepare_data(self):
        # 初始化文件加载器（仅在主进程执行）
        if not self.file_loader:
            self.file_loader = NPZFileLoaderOVPGCN(
                self.data_root,
                self.seq_len,
                self.pre_len
            )
        # 数据大小
        # self.adj = self.file_loader.adjacency_matrix
        self.feat = self._init_features(self.file_loader.feat)
        self.in_feat_st = int(self.seq_len * self.channels) # channels
        self.in_feat_bc = int(self.bc * 1 * self.channels)
        self.in_feat_pdm = int(self.pdm * self.channels)
        self.mask = torch.tensor(self.file_loader.mask_list).float()

    def _init_features(self, features):
        """初始化标准化参数并预存到GPU"""
        feat_max_val = np.round(features["max"], 4)
        feat_min_val = np.round(features["min"], 4)
        feat_mean = np.round(features["mean"], 4)
        feat_std = np.round(features["std"], 4)
        print(feat_max_val, feat_min_val, feat_mean, feat_std)
        return [feat_max_val, feat_min_val, feat_mean, feat_std]

    def setup(self, stage: str = None):
        # 划分训练验证集
        total_days = int(len(self.file_loader.file_list))
        train_size = int(total_days * self.split_ratio)
        # train_size = total_days
        # test_list = np.load(os.path.join(self.data_root, 'low.npy'))
        # print(test_list.shape)
        # print(test_list)
        if stage == 'test':
            self.test_dataset = SpatioTemporalDataset(
                mmap_data=self.file_loader.sst_mmap,
                start_idx=train_size,
                # start_idx=180,
                end_idx=total_days,
                seq_len=self.seq_len,
                pre_len=self.pre_len,
                normalize=self.normalize,
                spatio_gradient=self.file_loader.spatio_gradient,
                temporal_gradient=self.file_loader.temporal_gradient,
                features=self.file_loader.feat,
                device=self.device,
                gradient_mmap=self.file_loader.gradient,
                move_step=self.file_loader.move_step,
                max_mask=self.file_loader.max_mask,
                # test_list=test_list,
                time_list=self.file_loader.time_stamps,
            )
        self.train_dataset = SpatioTemporalDataset(
            mmap_data=self.file_loader.sst_mmap,
            start_idx=180,
            # end_idx=train_size,
            end_idx=total_days,
            seq_len=self.seq_len,
            pre_len=self.pre_len,
            normalize=self.normalize,
            spatio_gradient=self.file_loader.spatio_gradient,
            temporal_gradient=self.file_loader.temporal_gradient,
            features=self.file_loader.feat,
            device=self.device,
            gradient_mmap=self.file_loader.gradient,
            move_step=self.file_loader.move_step,
            max_mask=self.file_loader.max_mask,
            time_list=self.file_loader.time_stamps,
        )

        self.val_dataset = SpatioTemporalDataset(
            mmap_data=self.file_loader.sst_mmap,
            start_idx=train_size,
            end_idx=total_days,
            seq_len=self.seq_len,
            pre_len=self.pre_len,
            normalize=self.normalize,
            spatio_gradient=self.file_loader.spatio_gradient,
            temporal_gradient=self.file_loader.temporal_gradient,
            features=self.file_loader.feat,
            device=self.device,
            gradient_mmap=self.file_loader.gradient,
            move_step=self.file_loader.move_step,
            max_mask=self.file_loader.max_mask,
            # test_list=test_list,
            time_list=self.file_loader.time_stamps,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            persistent_workers=True,
            pin_memory=True,
            # collate_fn = dict_collate  # 显式指定合并函数
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            persistent_workers=True,
            pin_memory=True,
            # collate_fn=dict_collate  # 显式指定合并函数
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

    @property
    def adjacency_matrix(self):
        return self.file_loader.static_adj_file_list, self.file_loader.pearson_list

    @property
    def feat_max_val(self):
        return self.file_loader.feat_max_val

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=7)
        parser.add_argument("--seq_len", type=int, default=30)
        parser.add_argument("--pre_len", type=int, default=7)
        parser.add_argument("--split_ratio", type=float, default=0.8)
        parser.add_argument("--normalize", type=int, default=2)
        parser.add_argument("--channels", type=int, default=1)
        parser.add_argument("--bc", type=int, default=30)
        parser.add_argument("--pdm", type=int, default=180)
        return parser



# 使用示例
if __name__ == "__main__":
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)  # 设置启动方法为 spawn
    dm = SpatioTemporalDataModuleOVPGCN(
        # data_root=r"E:\PaperWork\SST",
        data_root=r"/home/jameszhang/桌面/SST/data",
        batch_size=7,
        seq_len=30,
        pre_len=7
    )
    dm.prepare_data()
    # sample = np.load(dm.file_loader.file_list[0])['sst']
    # sample = np.where(sample==0,np.nan,sample)
    # print(sample.shape, np.sum(np.isnan(sample)) / 1200/1400)
    dm.setup()
    # batch = next(iter(dm.train_dataloader()))
    batch = next(iter(dm.val_dataloader()))
    # # 检查单个样本类型
    # sample = dm.train_dataset[0]
    # print("Sample type:", type(sample))  # 应输出 <class 'dict'>
    # print("Sample keys:", sample.keys())  # 应显示 ['x', 'y', 'current_index']
    # # 获取一个batch 29021
    # adj = torch.load(adj_list[batch['current_index'][0]])
    # print("adj dtype:", adj.float().dtype)  # 输出应为 torch.float64 或 torch.float32
    # print("input x dtype:", batch['x'][0].dtype)
    # print(f"Input shape: {batch['x'][0].T.shape}")
    # print(f"Target shape: {batch['y'].shape}")
    # print(torch.sparse.mm(adj.float(), batch['x'][0].T))

    # adj_indices = batch['adj_indices']  # list of (2, nnz_i)
    # adj_values = batch['adj_values']  # list of (nnz_i,)
    # adj_sizes = batch['adj_size']  # (batch_size, 2)
    # adj_i = torch.sparse_coo_tensor(
    #     indices=adj_indices,
    #     values=adj_values,
    #     size=tuple(adj_sizes.tolist())
    # ).to(batch['x'].device)
    # print(adj_i.shape)

    # feat = np.load(r"/home/jameszhang/桌面/SST/data/sst_features.npz")
    # for key, value in feat.items():
    #     print(key, value)