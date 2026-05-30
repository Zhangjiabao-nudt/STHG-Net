# from tasks.supervised import SupervisedForecastTask
# from tasks.grid_supervised import GridForecastTask
from tasks.ovpgcn_supervised import OvpgcnForecastTask
# from tasks.eagcn_supervised import EagcnForecastTask
# from tasks.aatgcn_supervised import AatgcnForecastTask
# from tasks.test_supervised import TestForecastTask
from tasks.chgnn_supervised import ChgnnForecastTask

# __all__ = ["SupervisedForecastTask", 'GridForecastTask', "OvpgcnForecastTask", "EagcnForecastTask"
#     , "AatgcnForecastTask", 'TestForecastTask', 'chgnn_supervised']

__all__ = ["OvpgcnForecastTask", 'ChgnnForecastTask']