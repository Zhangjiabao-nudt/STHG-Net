import os
import argparse
from networkx import adjacency_matrix
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
    compute_and_save_adjacency


# from function import compute_mmap_statistics, compute_yearly_gradient, compute_daily_gradient, \
#     compute_and_save_graph, \
#     compute_and_save_adjacency


class GridNPZFileLoader:
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.time_delta = np.timedelta64(1, 'D')
        self.data_dir = self.data_root / 'all_npz'

        # 加载并预处理数据
        self.file_list = sorted([f for f in self.data_dir.glob("*.npz") if f.stem.isdigit()])
        self.time_stamps = [int(f.stem) for f in self.file_list]
        self.start_year = 1992
        self.end_year = 2020
        self.gradient_data_dir = self.data_root / 'data_gradient'
        # 验证时间连续性
        self._validate_time_sequence()

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

        self.mask_dir = self.gradient_data_dir / 'gradient_mask.npz'
        if not self.mask_dir.exists():
            self.find_mask_max()
        self.max_mask = np.load(self.mask_dir)["mask"]

        print(np.sum(self.max_mask), self.max_mask.shape)

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
        # print(unique_shapes)
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


class GridSpatioTemporalDataset(Dataset):
    def __init__(self, mmap_data, start_idx, end_idx, seq_len, mask, batch_size,
                 pre_len, features, move_step=10, normalize=2):
        self.mmap_data = mmap_data
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.normalize = normalize
        # self.device = device
        # 需要添加到config
        self._init_features(features)
        # 索引计算
        self.batch_size = batch_size
        self.valid_indices = range(
            start_idx,
            end_idx - (self.seq_len + self.pre_len) + 1,
            move_step
        )
        self.mask = torch.from_numpy(mask).float()
        self.mask = self.mask.unsqueeze(0).unsqueeze(0)
        # self.mask = self.mask.view(1, 1, self.mask.shape[0], self.mask.shape[1])  # 增加B,T维度
        # self.mask = self.mask.expand(self.batch_size, self.pre_len, self.mask.shape[2],
        #                              self.mask.shape[3])  # 广播到(B,T,H,W)

    def _init_features(self, features):
        """初始化标准化参数并预存到GPU"""
        self.feat_max_val = np.round(features["max"], 4)
        self.feat_min_val = np.round(features["min"], 4)
        self.feat_mean = np.round(features["mean"], 4)
        self.feat_std = np.round(features["std"], 4)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        window = self.mmap_data[actual_idx:actual_idx + self.seq_len + self.pre_len, 600:800, 400:600]


        # 标准化处理
        # 2. 数据标准化（在CPU进行）
        if self.normalize == 0:
            window = window / self.feat_max_val
        elif self.normalize == 1:
            window = (window - self.feat_min_val) / self.feat_max_val
        elif self.normalize == 2:
            print("mean std initial")
            window = (window - self.feat_mean) / self.feat_std

        x = window[:self.seq_len, :, :, np.newaxis]
        # y = window[self.seq_len:self.seq_len + self.pre_len, :, : , np.newaxis]
        y = window[self.seq_len:self.seq_len + self.pre_len]
        # print(x.shape, y.shape)
        return {
            'x': torch.from_numpy(x).float(),
            'y': torch.from_numpy(y).float(),
            'current_index': torch.tensor(idx).int(),
        }


class GridSpatioTemporalDataMLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True, **kwargs):
        # 自定义 collate 函数处理字典结构
        def collate_fn(batch):
            collated = {
                'x': torch.stack([item['x'] for item in batch]),  # 堆叠 x [batch_size, 1, seq_len, nodes]
                'y': torch.stack([item['y'] for item in batch]),  # 堆叠 y [batch_size, 1, pre_len, nodes]
                'current_index': torch.stack([item['current_index'] for item in batch])  # [batch_size]
                # 如需启用其他字段，按相同模式处理
            }
            return collated

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,  # 关键：绑定自定义 collate 函数
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            **kwargs
        )

    def __iter__(self):
        return super().__iter__()


class GridSpatioTemporalDataModule(LightningDataModule):
    def __init__(
            self,
            data_root: str,
            batch_size: int = 1,
            seq_len: int = 7,
            pre_len: int = 3,
            split_ratio: float = 0.8,
            normalize: int = 2,
            move_step: int = 10,
            **kwargs
    ):
        super().__init__()
        self.mask = None
        self.data_root = data_root
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        self.move_step = move_step

        # 延迟初始化
        self.file_loader = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        # 初始化文件加载器（仅在主进程执行）
        if not self.file_loader:
            self.file_loader = GridNPZFileLoader(
                self.data_root,
            )
        # 数据大小
        self.feat = self._init_features(self.file_loader.feat)


    def _init_features(self, features):
        """初始化标准化参数并预存到GPU"""
        feat_max_val = np.round(features["max"], 4)
        feat_min_val = np.round(features["min"], 4)
        feat_mean = np.round(features["mean"], 4)
        feat_std = np.round(features["std"], 4)
        return [feat_max_val, feat_min_val, feat_mean, feat_std]

    def setup(self, stage: str = None):
        # 划分训练验证集
        total_days = int(len(self.file_loader.file_list))
        train_size = int(total_days * self.split_ratio)

        self.train_dataset = GridSpatioTemporalDataset(
            mmap_data=self.file_loader.sst_mmap,
            start_idx=0,
            end_idx=train_size,
            seq_len=self.seq_len,
            pre_len=self.pre_len,
            normalize=self.normalize,
            features=self.file_loader.feat,
            move_step=self.move_step,
            mask=self.file_loader.max_mask,
            batch_size=self.batch_size,
        )
        self.mask = self.train_dataset.mask
        self.val_dataset = GridSpatioTemporalDataset(
            mmap_data=self.file_loader.sst_mmap,
            start_idx=train_size,
            end_idx=total_days,
            seq_len=self.seq_len,
            pre_len=self.pre_len,
            normalize=self.normalize,
            features=self.file_loader.feat,
            move_step=self.move_step,
            mask=self.file_loader.max_mask,
            batch_size=self.batch_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
            pin_memory=False,
        )

    # # @property
    # def adjacency_matrix(self):
    #     return self.file_loader.adjacency_list

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
        return parser


# 使用示例
if __name__ == "__main__":
    dm = GridSpatioTemporalDataModule(
        # data_root=r"E:\PaperWork\SST",
        # data_root=r"/home/jameszhang/桌面/SST/data",
        data_root=r"/root/autodl-tmp/SST",
        batch_size=10,
        seq_len=30,
        pre_len=7,
        move_step=5,
    )
    dm.prepare_data()
    # sample = np.load(dm.file_loader.file_list[0])['sst']
    # sample = np.where(sample==0,np.nan,sample)
    # print(sample.shape, np.sum(np.isnan(sample)) / 1200/1400)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    # # 检查单个样本类型
    # sample = dm.train_dataset[0]
    # print("Sample type:", type(sample))  # 应输出 <class 'dict'>
    # print("Sample keys:", sample.keys())  # 应显示 ['x', 'y', 'current_index']
    # # 获取一个batch 29021
    print(f"Input shape: {batch['x'].shape}")
    print(f"Target shape: {batch['y'].shape}")
    # print(torch.where(torch.isnan(batch['x'])))
