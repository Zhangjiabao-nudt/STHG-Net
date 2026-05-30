import logging
from typing import Union
from lightning.pytorch.loggers.logger import Logger

# def format_logger(logger, fmt="\033[31m[%(asctime)s %(levelname)s]\033[0m%(message)s"):
#     handler = logger.handlers[0]
#     formatter = logging.Formatter(fmt)
#     handler.setFormatter(formatter)
#
#
# def output_logger_to_file(logger, output_path, fmt="[%(asctime)s %(levelname)s]%(message)s"):
#     handler = logging.FileHandler(output_path, encoding="UTF-8")
#     formatter = logging.Formatter(fmt)
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     return logger



def format_logger(
        logger: Union[Logger, list],
        fmt: str = "\033[31m[%(asctime)s %(levelname)s]\033[0m%(message)s"
) -> None:
    """适配 PyTorch Lightning 2.0+ 的日志格式设置"""
    # 处理多logger情况
    loggers = logger if isinstance(logger, list) else [logger]

    for pl_logger in loggers:
        # 获取底层实际logger对象
        _logger = logging.getLogger(pl_logger.name)

        # 清除现有控制台handler
        for handler in _logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                _logger.removeHandler(handler)

        # 添加新的格式处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(fmt))
        _logger.addHandler(console_handler)


def output_logger_to_file(
        logger: Union[Logger, list],
        output_path: str,
        fmt: str = "[%(asctime)s %(levelname)s]%(message)s"
) -> Logger:
    """适配 PyTorch Lightning 2.0+ 的文件日志记录"""
    loggers = logger if isinstance(logger, list) else [logger]

    for pl_logger in loggers:
        _logger = logging.getLogger(pl_logger.name)

        # 添加文件handler
        file_handler = logging.FileHandler(output_path, encoding="UTF-8")
        file_handler.setFormatter(logging.Formatter(fmt))
        _logger.addHandler(file_handler)

    return logger