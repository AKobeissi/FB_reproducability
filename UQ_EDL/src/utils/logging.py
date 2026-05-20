import logging
import sys
import pathlib
from datetime import datetime


def get_logger(name: str, log_file: str | pathlib.Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file is not None:
        log_file = pathlib.Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def timestamped_run_dir(base: str | pathlib.Path, name: str) -> pathlib.Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pathlib.Path(base) / f"{name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
