import re
import numpy as np
import matplotlib.pyplot as plt
import cmaps
from sympy.physics.units import current

from utils.callbacks.base import BestEpochCallback
from mpl_toolkits.basemap import Basemap
import torch
import numpy as np
import matplotlib.pyplot as plt
import cmaps
import torch
from mpl_toolkits.basemap import Basemap
from utils.callbacks.base import BestEpochCallback  # 假设保留基础类

# 运行在验证集上

class PlotValidationPredictionsCallback(BestEpochCallback):
    def __init__(self, monitor="", mode="min", **kwargs):
        super(PlotValidationPredictionsCallback, self).__init__(monitor=monitor, mode=mode)
        self.ground_truths = []
        self.predictions = []
        self.time = []
        self.masks = []
        # if self.kwargs is not None:

        self.mask_list = np.load(kwargs.get('adj_mask_dir'))['mask']
        self.adj_number = kwargs.get('adj_number')
        self.graph = np.full((self.adj_number, self.adj_number), np.nan)
        self.grid = kwargs.get('grid')

    def on_fit_start(self, trainer, pl_module):
        self.ground_truths.clear()
        self.predictions.clear()
        self.time.clear()
        pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): # delete data_loader_idx
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        # if trainer.current_epoch != self.best_epoch:
        #     return
        # self.ground_truths.clear()
        # self.predictions.clear()
        predictions, y = outputs
        # print(predictions[0:30], y[0:30])
        predictions = predictions.detach().to(torch.float).cpu().numpy()
        y = y.detach().to(torch.float).cpu().numpy()

        current_index = batch["current_index"]
        mask = self.mask_list[current_index.cpu()]
        self.masks.append(mask)

        # y is consist of the number which can be used in seriese data and the formula is len(train_data) - seq_len - pre_len,
        #  seq_len,number node 
        # (batch_size, seq_len/pre_len, num_nodes)
        print(y.shape, predictions.shape)
        # time_one =
        self.time.append(batch['time'][0].to(torch.float).cpu().numpy())
        # self.masks.append(batch['mask'][0].to(torch.float).cpu().numpy())
        self.ground_truths.append(y[:, 0, :])
        self.predictions.append(predictions[:, 0, :])
        # print(len(self.predictions), len(self.ground_truths))
        # self.ground_truths.append(y[:, 0, :, :])
        # self.predictions.append(predictions[:, 0, :, :])

    def on_fit_end(self, trainer, pl_module):
        # batch_size * step  may = the train data number
        # This indicate that the sequence of time series
        ground_truth = np.concatenate(self.ground_truths, 0)
        predictions = np.concatenate(self.predictions, 0)
        # print(predictions.shape, ground_truth.shape)

        tensorboard = pl_module.logger.experiment
        # for node_idx in range(ground_truth.shape[0]):
        #     plt.clf()
        #     plt.rcParams["font.family"] = "Times New Roman"
        #     fig = plt.figure(figsize=(30, 15), dpi=300)
        #     plt.plot(
        #         ground_truth[node_idx, :],
        #
        #         color="dimgray",
        #         linestyle="-",
        #         label="Ground truth",
        #     )
        #     plt.plot(
        #         predictions[node_idx, :],
        #         color="deepskyblue",
        #         linestyle="-",
        #         label="Predictions",
        #     )
        #     plt.legend(loc="best", fontsize=10)
        #     plt.xlabel("Time")
        #     plt.ylabel("SST Data")
        #     tensorboard.add_figure(
        #         "Prediction result of node " + str(node_idx),
        #         fig,
        #         global_step=len(trainer.train_dataloader) * self.best_epoch,
        #         close=True,
        #     )

        for number in range(ground_truth.shape[0]):
            GraphDataTruth = self.GraphDataCovertGraphWithLand(ground_truth[number], self.masks[number], 30., 40., 120., 130.,0.05)
            GraphDataPrediction = self.GraphDataCovertGraphWithLand(predictions[number], self.masks[number], 30., 40., 120., 130.,0.05)
            GraphDataTruth = np.where(GraphDataTruth == 0, np.nan, GraphDataTruth)
            GraphDataPrediction = np.where(GraphDataPrediction < 0.5, np.nan, GraphDataPrediction)
            fig = self.plotFuncTwo(30., 40., 120., 130., 200, 200, GraphDataTruth, GraphDataPrediction, self.time[number])
            tensorboard.add_figure(
                "Prediction time: " + str(self.time[number]),
                # "prediction node" + str(number),
                fig,
                global_step=len(trainer.train_dataloader) * self.best_epoch,
                close=True,
            )

    def GraphDataCovertGraphWithLand(self, data, mask, llat, rlat, llon, rlon, precision):
        """
        _summary_
                使用陆地数据网格点的转化
        Args:
            data (_type_): 一维数据矩阵
            llat (_type_): _description_
            rlat (_type_): _description_
            llon (_type_): _description_
            rlon (_type_): _description_
            precision (_type_): _description_
        
        Returns:
            _type_: _description_
        """   
        # length = data.shape[0]
        # lat = int((rlat - llat) / precision)
        # lon = int((rlon - llon) / precision)
        # Graph = np.zeros((lat,lon))
        # if length == lat*lon:
        #     for i in range(lat):
        #         for j in range(lon):
        #             Graph[i][j] = data[i*lon+j]
        #     return Graph[::-1]
        # else:
        #     print("The amount is wrong.")
        #     return None
        # regruster_data = np.full_like(self.graph, np.nan)
        # 使用 mask 来填充 reconstructed_graph
        # regruster_data[self.mask] = data
        if self.grid == 'grid':
            data[~mask.astype(bool)] = np.nan
            return data[::-1]
        else:
            regruster_data = np.full_like(self.graph, np.nan)
            # 使用 mask 来填充 reconstructed_graph
            regruster_data[self.mask] = data
            return regruster_data[::-1]

    def plotFuncTwo(self, llat, rlat, llon, rlon, numlat, numlon, data_plt_sla1, data_plt_sla2, time):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        count = 0
        # 显示第一个图像并添加颜色条
        # im1 = ax1.imshow(data_plt_sla1, cmap=cmaps.NCV_jaisnd, interpolation='nearest')
        # ax1.set_title('Original Sea Surface Temperature', fontsize='large')
        # cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical')
        # cbar1.set_label('Temperature', fontsize='large')
        #
        # # 显示第二个图像并添加颜色条
        # im2 = ax2.imshow(data_plt_sla2, cmap=cmaps.NCV_jaisnd, interpolation='nearest')
        # ax2.set_title('Prediction Sea Surface Temperature', fontsize='large')
        # cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical')
        # cbar2.set_label('Temperature', fontsize='large')
        # plt.show()
        for ax in [ax1, ax2]:
            map = Basemap(projection='cyl', llcrnrlat=llat, urcrnrlat=rlat, llcrnrlon=llon, urcrnrlon=rlon, resolution='i', ax=ax)
            map.drawmapboundary()
            map.drawstates()
            map.drawcoastlines()

            lons, lats = map.makegrid(numlon, numlat)
            lats = lats[::-1]
            x, y = map(lons, lats)

            map.drawparallels(np.arange(llat, rlat+1, 10), labels=[1, 0, 0, 0], fontsize=8)
            map.drawmeridians(np.arange(llon, rlon+1, 10), labels=[0, 0, 0, 1], fontsize=8)

            # cmap = plt.get_cmap('NCV_jaisnd')
            if count == 0:
                # shade = map.contourf(x, y, data_plt_sla1, extend='both', cmap=cmap)
                shade1 = map.pcolormesh(x, y, data_plt_sla1, shading='auto', cmap=cmaps.NCV_jaisnd)
                count = 1
                ax1.legend(['Ground_Truth'], loc='best', fontsize=10)
                # ax1.set_xlabel("Lon")
                # ax1.set_ylabel("Lat")
            else:
                shade = map.pcolormesh(x, y, data_plt_sla2, shading='auto', cmap=cmaps.NCV_jaisnd)
                ax2.legend(['Prediction'], loc='best', fontsize=10)
                # ax2.set_xlabel("Lon")
                # ax2.set_ylabel("Lat")
        # 添加全局标题
        # fig.suptitle(f'Sea Level Anomaly Comparison', fontsize=12, y=0.92)  # y参数控制垂直位置

        # 在figure左侧添加全局颜色条
        cbar_ax = fig.add_axes([0.04, 0.25, 0.02, 0.5])  # 调整参数：[左, 下, 宽, 高]
        cbar = fig.colorbar(shade1, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=6)

        # plt.show()
        plt.savefig(f'/home/jameszhang/桌面/SST/result_png/CHGNN/{str(time)}.jpg')
        return fig


class PlotTestPredictionsCallback(BestEpochCallback):
    """用于测试集的预测结果可视化回调类"""

    def __init__(self, **kwargs):
        # 测试阶段不需要监控指标来确定最佳epoch
        super(PlotTestPredictionsCallback, self).__init__(monitor="", mode="min")
        self.ground_truths = []
        self.predictions = []
        self.time = []
        self.masks = []

        # 从参数获取必要配置
        self.mask_list = np.load(kwargs.get('adj_mask_dir'))['mask']
        self.adj_number = kwargs.get('adj_number')
        self.graph = np.full((self.adj_number, self.adj_number), np.nan)
        self.grid = kwargs.get('grid')
        # 添加测试结果保存路径
        # self.save_dir = kwargs.get('save_dir', './test_results/plots')
        self.save_dir = kwargs.get('save_dir', './test_results/ovpgcn_plots_30')
        # 创建保存目录
        import os
        os.makedirs(self.save_dir, exist_ok=True)

    def on_test_start(self, trainer, pl_module):
        """测试开始时初始化列表"""
        self.ground_truths.clear()
        self.predictions.clear()
        self.time.clear()
        self.masks.clear()
        pl_module.eval()  # 确保模型处于评估模式

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """测试批次结束时收集数据"""
        # 从输出中获取预测值和真实值
        predictions, y = outputs

        # 转换为numpy数组
        predictions = predictions.detach().to(torch.float).cpu().numpy()
        y = y.detach().to(torch.float).cpu().numpy()

        # 获取当前批次的索引和掩码
        current_index = batch["current_index"]
        mask = self.mask_list[current_index.cpu()]
        self.masks.append(mask)

        # 收集时间信息、真实值和预测值
        self.time.append(batch['time'][0].to(torch.float).cpu().numpy())
        self.ground_truths.append(y[:, 29, :])
        self.predictions.append(predictions[:, 29, :])

    def on_test_end(self, trainer, pl_module):
        """测试结束时处理并可视化所有数据"""
        # 拼接所有批次的数据
        ground_truth = np.concatenate(self.ground_truths, 0)
        predictions = np.concatenate(self.predictions, 0)

        # 获取tensorboard日志器
        tensorboard = pl_module.logger.experiment

        # 为每个时间步绘制并保存图像
        for number in range(ground_truth.shape[0]):
            # 处理数据，转换为带陆地掩码的网格格式
            GraphDataTruth = self.GraphDataCovertGraphWithLand(
                ground_truth[number],
                self.masks[number],
                30., 40., 120., 130., 0.05
            )
            GraphDataPrediction = self.GraphDataCovertGraphWithLand(
                predictions[number],
                self.masks[number],
                30., 40., 120., 130., 0.05
            )

            # 处理无效值
            GraphDataTruth = np.where(GraphDataTruth == 0, np.nan, GraphDataTruth)
            GraphDataPrediction = np.where(GraphDataPrediction < 0.5, np.nan, GraphDataPrediction)

            # 绘制图像
            fig = self.plotFuncTwo(
                30., 40., 120., 130., 200, 200,
                GraphDataTruth, GraphDataPrediction,
                self.time[number]
            )

            # 添加到tensorboard
            tensorboard.add_figure(
                f"Test Prediction time: {str(self.time[number])}",
                fig,
                global_step=trainer.global_step,
                close=True,
            )

    def GraphDataCovertGraphWithLand(self, data, mask, llat, rlat, llon, rlon, precision):
        """将一维数据转换为带陆地掩码的网格格式"""
        if self.grid == 'grid':
            # 网格数据处理
            data[~mask.astype(bool)] = np.nan
            return data[::-1]
        else:
            # 图数据处理
            regruster_data = np.full_like(self.graph, np.nan)
            regruster_data[self.mask] = data
            return regruster_data[::-1]

    def plotFuncTwo(self, llat, rlat, llon, rlon, numlat, numlon, data_plt_sla1, data_plt_sla2, time):
        """绘制真实值和预测值的对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        count = 0

        for ax in [ax1, ax2]:
            # 创建底图
            map = Basemap(
                projection='cyl',
                llcrnrlat=llat,
                urcrnrlat=rlat,
                llcrnrlon=llon,
                urcrnrlon=rlon,
                resolution='i',
                ax=ax
            )
            map.drawmapboundary()
            map.drawstates()
            map.drawcoastlines()

            # 创建经纬度网格
            lons, lats = map.makegrid(numlon, numlat)
            lats = lats[::-1]
            x, y = map(lons, lats)

            # 绘制经纬线
            map.drawparallels(
                np.arange(llat, rlat + 1, 10),
                labels=[1, 0, 0, 0],
                fontsize=8
            )
            map.drawmeridians(
                np.arange(llon, rlon + 1, 10),
                labels=[0, 0, 0, 1],
                fontsize=8
            )

            # 绘制数据
            if count == 0:
                shade1 = map.pcolormesh(
                    x, y, data_plt_sla1,
                    shading='auto',
                    cmap=cmaps.NCV_jaisnd
                )
                ax.set_title('Ground Truth', fontsize=12)
                count = 1
            else:
                shade = map.pcolormesh(
                    x, y, data_plt_sla2,
                    shading='auto',
                    cmap=cmaps.NCV_jaisnd
                )
                ax.set_title('Prediction', fontsize=12)

        # 添加全局颜色条
        cbar_ax = fig.add_axes([0.04, 0.25, 0.02, 0.5])  # 位置：[左, 下, 宽, 高]
        cbar = fig.colorbar(shade1, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label('Temperature', fontsize=8)

        # 添加全局标题
        fig.suptitle(f'Test Prediction at Time: {time}', fontsize=14, y=0.92)

        # 保存图像到指定目录
        save_path = f"{self.save_dir}/{str(time)}.jpg"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

