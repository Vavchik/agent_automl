"""Р1. Логи в консоль и в файл agent.log рядом с task.md."""
import logging
import sys
from pathlib import Path


def setup_logger(task_dir: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("automl")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if task_dir is not None:
        try:
            file_handler = logging.FileHandler(Path(task_dir) / "agent.log", encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        except OSError:
            pass
    return logger


log = logging.getLogger("automl")
