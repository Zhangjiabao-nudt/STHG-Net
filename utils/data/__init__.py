# from spatotemporal_data import SpatioTemporalNPYDataModule
from utils.data.cluster import HyperGraphConstructor
from utils.data.function import (calculate_temperature_gradient_spatio, compute_mmap_statistics, compute_yearly_gradient, \
    compute_daily_gradient, compute_and_save_graph, compute_and_save_adjacency, compute_half_year_gradient, \
    generate_edge_index, generate_adjacency_matrix_np)
                                 # generate_static_adjacency_matrix)
# from utils.data.function_other_model import generate_static_adj, generate_pearson_adj


# from utils.data.adjacency_generation import grid_hypergraph_to_graph
# from utils.data.graph_conv import calculate_laplacian_with_self_loop
# from utils.data.spatotemporal_data import SpatioTemporalDataModule
# from utils.data.grid_spatiotemporal_data import GridSpatioTemporalDataModule
from utils.data.saptiotemporal_ovpgcn import SpatioTemporalDataModuleOVPGCN
# from utils.data.spatiotemporal_EAGCN import SpatioTemporalDataModuleEAGCN
# from utils.data.spatiotemporal_AATGCN import SpatioTemporalDataModuleAATGCN
# from utils.data.SpatioTemporalDataModelTest import SpatioTemporalDataModuleTest
from utils.data.spatiotemporal_CHGNN import SpatioTemporalDataModuleCHGNN

# SupervisedDataModule = SpatioTemporalDataModule
# GridDataModule = GridSpatioTemporalDataModule
OvpgcnDataModule = SpatioTemporalDataModuleOVPGCN
# EagcnDataModule = SpatioTemporalDataModuleEAGCN
# AatgcnDataModule = SpatioTemporalDataModuleAATGCN
# TestDataModule = SpatioTemporalDataModuleTest
ChgnnDataModule = SpatioTemporalDataModuleCHGNN

# __all__ = [
#     "SupervisedDataModule",
#     "SpatioTemporalDataModule",
#     "GridSpatioTemporalDataModule",
#     "GridDataModule",
#     "OvpgcnDataModule",
#     "SpatioTemporalDataModuleOVPGCN",
#     "EagcnDataModule",
#     "SpatioTemporalDataModuleEAGCN",
#     "AatgcnDataModule",
#     "SpatioTemporalDataModuleAATGCN",
#     'TestDataModule',
#     'SpatioTemporalDataModuleTest',
#     'SpatioTemporalDataModuleCHGNN',
#     'ChgnnDataModule'
# ]

__all__ = [
    # "SupervisedDataModule",
    # "SpatioTemporalDataModule",
    # "GridSpatioTemporalDataModule",
    # "GridDataModule",
    "OvpgcnDataModule",
    "SpatioTemporalDataModuleOVPGCN",
    # "EagcnDataModule",
    # "SpatioTemporalDataModuleEAGCN",
    # "AatgcnDataModule",
    # "SpatioTemporalDataModuleAATGCN",
    # 'TestDataModule',
    # 'SpatioTemporalDataModuleTest',
    'SpatioTemporalDataModuleCHGNN',
    'ChgnnDataModule'
]