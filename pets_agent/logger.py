import logging
from pathlib import Path
from datetime import datetime

# 模块级标志，确保 handler 只添加一次
_initialized = False


def setup_logger(name: str = "pets_agent", log_dir: str = "logs") -> logging.Logger:
    """设置日志系统"""
    global _initialized

    # 项目根目录
    project_root = Path(__file__).parent.parent
    log_path = project_root / log_dir
    log_path.mkdir(exist_ok=True)

    # 日志文件名（按日期）
    log_file = log_path / f"{datetime.now().strftime('%Y%m%d')}.log"

    # 创建 logger
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.INFO)

    # 只在首次调用时添加 handler
    if not _initialized:
        # 清除可能存在的重复 handler
        _logger.handlers.clear()

        # 文件 handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        # 控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 格式化
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        _logger.addHandler(file_handler)
        _logger.addHandler(console_handler)

        _initialized = True

    return _logger


# 全局 logger 实例
logger = setup_logger()
