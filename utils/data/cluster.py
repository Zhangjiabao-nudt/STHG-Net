import cmaps
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


def update_max_min(current_min, current_max, value):
    current_max = current_max if current_max > value else value
    current_min = current_min if current_min < value else value
    return current_min, current_max


class HyperGraphConstructor:
    def __init__(self, data):
        # 数据预处理
        self.cluster_data = data  # 用于画图
        self.data = data * 10
        self.data = np.where(np.isnan(self.data), -327, self.data)  # 用于聚类

        # 预计算常用值
        self.rows, self.cols = data.shape
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1),
                           (-1, -1)]  # 上下左右斜对角八个方向 (1,1), (1, -1), (-1, 1), (-1,-1)

        # 状态存储优化
        self.visited = np.zeros_like(data, dtype=bool)
        self.cluster_record = np.full_like(self.data, fill_value=-1, dtype=int)  # 记录点的簇类别
        self.sub_graphs = []
        self.clusters_metric = []  # 记录簇的特征，暂未用到

        # 阈值相关优化
        self.gradient_threshold = self.get_threshold()
        self.condition_threshold = 1  # 0,1,2  1 ：区分结果为空白地图
        self.num_cluster_threshold = 2  # 表明一个簇内部最少的节点数量
        self.idx = 0  # 索引记录簇

    def is_within_threshold(self, current_min, current_max, value):
        if self.condition_threshold == 0:
            return abs(current_max - value) <= self.gradient_threshold
        current_min_change, current_max_change = update_max_min(current_min, current_max, value)
        if self.condition_threshold == 1:
            return abs(current_max_change - current_min_change) <= self.gradient_threshold
        if self.condition_threshold == 2:
            return abs(current_max_change - current_min_change) <= current_min_change
        return False

    def classify_sub_graphs(self):
        start_points = np.argwhere(~self.visited & (self.data != -327))

        for (x, y) in start_points:
            if self.visited[x, y]:
                continue
            # sub_graph = self._explore_sub_graph(x, y)
            sub_graph = self._bfs_region_growing(x, y)
            if sub_graph and len(sub_graph) >= self.num_cluster_threshold:
                # print(len(sub_graph))
                self.sub_graphs.append(sub_graph)
                self._update_cluster_record(sub_graph)
                self.idx += 1
                # self.visited[x, y] = True
            else:
                self._visited_revoke(sub_graph)

        # 优化孤立点
        self._reassign_orphans()

        # 还原
        # false_coordinates = np.where(self.visited == False)
        # # 使用zip将行和列的索引组合成坐标对
        # false_points = np.array(false_coordinates, dtype=int).T
        # # 处理非nan未连结的数据
        # for point_x, point_y in false_points:
        #     if self.data[point_x, point_y] != -327:
        #         self._explore_sub_graph_add(point_x, point_y)
        #         self.visited[point_x, point_y] = True
        # else:
        #     print(self.data[point_x, point_y])

    def _visited_revoke(self, sub_graph):
        x_coords, y_coords = np.array(sub_graph, dtype=int).T
        self.visited[x_coords, y_coords] = False

    def _explore_sub_graph_add(self, x, y):
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.data.shape[0] and 0 <= ny < self.data.shape[1]:
                idx = self.cluster_record[nx, ny]
                if idx != -1 and self.data[nx, ny] != -327:
                    self.sub_graphs[idx].append([nx, ny])

    def find_max(self, sub_graph_points):
        sub_graph_data = np.full_like(self.data, fill_value=-329)
        x_coords, y_coords = np.array(sub_graph_points, dtype=int).T
        sub_graph_values = self.data[x_coords, y_coords]
        sub_graph_data[x_coords, y_coords] = sub_graph_values
        return np.max(sub_graph_data), np.min(sub_graph_data)

    def _explore_sub_graph(self, x, y):
        stack = [(x, y)]
        sub_graph_points = []
        current_Id = 0
        current_max = 0
        current_min = 0

        while stack:
            x, y = stack.pop()
            if self.visited[x, y]:
                continue
            self.visited[x, y] = True
            sub_graph_points.append((x, y))
            self.cluster_record[x, y] = self.idx

            if current_Id == 0:
                current_max, current_min = self.find_max(sub_graph_points)
                current_Id = 1
            else:
                current_min, current_max = update_max_min(current_min=current_min, current_max=current_max,
                                                          value=self.data[x, y])

            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.data.shape[0] and 0 <= ny < self.data.shape[1] and not self.visited[nx, ny]:
                    # print(a)
                    if self.is_within_threshold(current_min, current_max, self.data[nx, ny]):
                        stack.append([nx, ny])

        if len(sub_graph_points) <= self.num_cluster_threshold:
            x_coords, y_coords = np.array(sub_graph_points, dtype=int).T
            self.visited[x_coords, y_coords] = False
            self.cluster_record[x_coords, y_coords] = -1
            return None
        return sub_graph_points

    def _update_cluster_record(self, sub_graph):
        x_coords, y_coords = np.array(sub_graph, dtype=int).T
        self.cluster_record[x_coords, y_coords] = self.idx

    def _bfs_region_growing(self, x, y):
        queue = deque()
        queue.append((x, y))
        sub_graph = []
        current_max = current_min = self.data[x, y]

        while queue:
            x, y = queue.popleft()
            if self.visited[x, y]:
                continue

            # update state
            self.visited[x, y] = True
            sub_graph.append([x, y])
            val = self.data[x, y]
            current_max = max(current_max, val)
            current_min = min(current_min, val)
            # self.cluster_record[x, y] = self.idx

            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.data.shape[0] and 0 <= ny < self.data.shape[1] and not self.visited[nx, ny]:
                    if self.is_within_threshold(current_min, current_max, self.data[nx, ny]):
                        queue.append((nx, ny))
        return sub_graph
        # return sub_graph if len(sub_graph) >= self.num_cluster_threshold else None

    def _reassign_orphans(self):
        """优化的孤立点处理"""
        orphans = np.argwhere(~self.visited & (self.data != -327))
        print(f"orphans: {len(orphans)}")
        for x, y in orphans:
            # 查找最近邻簇
            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.rows and 0 <= ny < self.cols:
                    cid = self.cluster_record[nx, ny]
                    self.sub_graphs[cid].append([x, y])
                    self.cluster_record[x, y] = cid
                    self.visited[x, y] = True

    def plot_sub_graphs(self):
        num_sub_graphs = len(self.sub_graphs)
        cols = int(np.ceil(np.sqrt(num_sub_graphs)))
        rows = int(np.ceil(num_sub_graphs / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = axes.flatten()

        for i, sub_graph in enumerate(self.sub_graphs):
            ax = axes[i]
            sub_graph_data = np.ma.masked_where(~np.isin(self.data, [self.data[x, y] for x, y in sub_graph]), self.data)
            # ax.imshow(self.data, cmap='hot', interpolation='nearest')
            ax.imshow(sub_graph_data, cmap='coolwarm', alpha=0.5)
            ax.set_title(f'sub_graph {i + 1}', fontsize='large')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_sub_graphs_mask_origin(self):
        num_sub_graphs = len(self.sub_graphs)
        cols = int(np.ceil(np.sqrt(num_sub_graphs)))
        rows = int(np.ceil(num_sub_graphs / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = axes.flatten()
        for i, sub_graph in enumerate(self.sub_graphs):
            ax = axes[i]
            sub_graph_data = np.zeros_like(self.data)
            # 将 sub_graph 转换为两个数组，分别包含 x 和 y 坐标
            x_coords, y_coords = np.array(sub_graph, dtype=int).T
            # 使用 NumPy 的高级索引来直接选择子图数据
            sub_graph_values = self.data[x_coords, y_coords]
            # 使用 NumPy 的高级索引来将子图数据赋值到正确的位置
            sub_graph_data[x_coords, y_coords] = sub_graph_values
            # for x, y in sub_graph:
            # sub_graph_data[x, y] = data[x, y]
            ax.imshow(sub_graph_data, cmap='coolwarm')
            ax.set_title(f'sub_graph {i + 1}')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_original_image(self):
        plt.figure(figsize=(8, 8))
        plt.imshow(self.cluster_data, cmap=cmaps.NCV_jaisnd, interpolation='nearest')
        plt.title('Original Sea Surface Temperature', fontsize='large')
        plt.colorbar()
        plt.show()

    def plot_all_sub_graphs(self):
        num_sub_graphs = len(self.sub_graphs)
        cols = int(np.ceil(np.sqrt(num_sub_graphs)))
        rows = int(np.ceil(num_sub_graphs / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = axes.flatten()

        for i, sub_graph in enumerate(self.sub_graphs):
            ax = axes[i]
            # print(len(sub_graph))
            sub_graph_data = np.zeros_like(self.cluster_data)
            x_coords, y_coords = np.array(sub_graph, dtype=int).T
            sub_graph_values = self.cluster_data[x_coords, y_coords]
            sub_graph_data[x_coords, y_coords] = sub_graph_values
            ax.imshow(sub_graph_data, cmap=cmaps.NCV_jaisnd, interpolation='nearest')
            ax.set_title(f'sub_graph {i + 1}', fontsize='large')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    """
       # def plot_sub_graph_boundaries(self):
    #     boundary_mask = np.zeros_like(self.data, dtype=bool)
    #     directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    #     for sub_graph in self.sub_graphs:
    #         for x, y in sub_graph:
    #             for dx, dy in directions:
    #                 nx, ny = x + dx, y + dy
    #                 if 0 <= nx < self.data.shape[0] and 0 <= ny < self.data.shape[1]:
    #                     if (nx, ny) not in sub_graph:
    #                         boundary_mask[x, y] = True

    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(self.data, cmap='hot', interpolation='nearest')
    #     plt.imshow(boundary_mask, cmap='cool', alpha=0.5)
    #     plt.title('sub_graph Boundaries on Original Image', fontsize='large')
    #     plt.colorbar()
    #     plt.show()
    def plot_sub_graph_boundaries(self):
        num_sub_graphs = len(self.sub_graphs)
        cols = int(np.ceil(np.sqrt(num_sub_graphs)))
        rows = int(np.ceil(num_sub_graphs / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = axes.flatten()

        for i, sub_graph in enumerate(self.sub_graphs):
            ax = axes[i]
            sub_graph_data = np.zeros_like(self.data)
            x_coords, y_coords = np.array(sub_graph, dtype=int).T
            sub_graph_values = self.data[x_coords, y_coords]
            sub_graph_data[x_coords, y_coords] = sub_graph_values

            # 绘制子图区域
            ax.imshow(sub_graph_data, cmap='hot', interpolation='nearest')

            # 创建一个用于绘制轮廓的矩阵
            boundary_mask = np.zeros_like(self.data, dtype=bool)
            for x, y in sub_graph:
                boundary_mask[x, y] = True

            # 计算轮廓
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                boundary_mask &= ~np.roll(boundary_mask, dx, axis=0 if dx != 0 else 1)
                boundary_mask &= ~np.roll(boundary_mask, dy, axis=1 if dy != 0 else 0)

            # 绘制轮廓线
            ax.contour(boundary_mask, colors='blue', linewidths=2)

            ax.set_title(f'sub_graph {i + 1} Boundary', fontsize='large')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()
    """

    def visualize_sub_graph(self, index):
        """可视化指定子图 (示例)"""
        # import matplotlib.pyplot as plt
        if index >= len(self.sub_graphs):
            raise IndexError("sub_graph index out of range")
        # mask = np.zeros_like(self.data, dtype=bool)
        # x_coords, y_coords = np.array(self.sub_graphs[index]).T
        # mask[x_coords, y_coords] = True

        x_coords, y_coords = np.array(self.sub_graphs[index], dtype=int).T
        if len(x_coords) == 0:
            raise ValueError("sub_graph is empty")

        plt.figure(figsize=(10, 6))
        plt.imshow(self.cluster_data, cmap=cmaps.NCV_jaisnd)
        plt.colorbar(label='SST Value')
        plt.scatter(y_coords, x_coords, s=3, c='red', alpha=0.7)
        plt.title(f"sub_graph {index} (Size: {len(x_coords)})")
        plt.show()

    def get_threshold(self):
        cols = np.nanmean(self.cluster_data, axis=0)
        rows = np.nanmean(self.cluster_data, axis=1)
        # print(cols, rows)
        # return min(np.nanmin(cols), np.nanmin(rows)) * 10.0
        # return min(np.nanmin(cols), np.nanmin(rows)) * 5
        # return min(np.nanmin(cols), np.nanmin(rows)) * 2.5
        return 1
        # return min(np.nanmin(cols), np.nanmin(rows)) * 20
    def get_sub_graphs(self):
        return self.sub_graphs

    def save_visualize_sub_graphs(self, path, index):
        if index >= len(self.sub_graphs):
            raise IndexError("sub_graph index out of range")

        x_coords, y_coords = np.array(self.sub_graphs[index], dtype=int).T
        if len(x_coords) == 0:
            raise ValueError("sub_graph is empty")

        plt.figure(figsize=(10, 6))
        plt.imshow(self.cluster_data, cmap=cmaps.NCV_jaisnd)
        plt.colorbar(label='SST Value')
        plt.scatter(y_coords, x_coords, s=1, c='red', alpha=0.7)
        plt.title(f"sub_graph {index} (Size: {len(x_coords)})")
        plt.savefig(path / f"sub_graph_{index}.png")
        plt.close()
        # plt.show()


if __name__ == "__main__":
    # 示例数据
    # data = np.random.rand(10, 10) * 10  # 10x10的随机海表面温度数据
    threshold = 0.11  # 阈值设定
    data = np.load(r'E:\PaperWork\SST\data_gradient\mmap.npy',
    # data = np.load(r"/home/jameszhang/桌面/SST/data/data_gradient/mmap.npy",
                   mmap_mode='r', allow_pickle=True)
    one = data[0, 600:800, 400:600][::-1][::5, ::5]
    sst = HyperGraphConstructor(data[0, 600:800, 400:600][::-1][::5, ::5])
    sst.classify_sub_graphs()
    # sst.plot_sub_graph_boundaries()
    # sst.classify_sub_graphs()
    print(len(sst.sub_graphs))
    ans = []
    for sg in sst.sub_graphs:
        ans.append(len(sg))
    print(f'clusters:{len(ans)}, max:{max(ans)}, '
          f'mean:{np.mean(ans)}, 0.25:{np.percentile(ans, 25)}, 0.75:{np.percentile(ans, 75)}')
    n = len(ans)
    print(ans[int(n * 0.75)])
    sst.visualize_sub_graph(5)
    print(11.47 * 156, 63 * 20.65, 221 * 11.25, 86 * 16.85)
# clusters:86, max:542, mean:16.848837209302324, 0.25:4.0, 0.75:10.75 1 * \theta
# clusters:63, max:592, mean:20.650793650793652, 0.25:3.5, 0.75:15.5 fix = 0.1
# clusters:156, max:178, mean:11.256410256410257, 0.25:3.75, 0.75:10.25 0.5 * \theta
# clusters:221, max:493, mean:11.46606334841629, 0.25:5.0, 0.75:11.0 0.25 * \theta


# sters:33, max:907, mean:39.36363636363637, 0.25:2.0, 0.75:19.0 2 * \theta
