"""
Модуль единого логгера для AutoML Agent (ml_core/logger.py).

Обеспечивает:
1. Вывод в консоль с информативными статусами и форматированием.
2. Сохранение подробных логов в файл automl_agent.log (в корне проекта и каталоге задачи).
3. Перехват логов всех модулей: agent.llm, agent.autoresearch, ml_core.trainer, ml_core.submission_guard.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Optional


class CustomFormatter(logging.Formatter):
    """Форматтер логов для красивого вывода в консоль."""

    FORMAT = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
    DATE_FORMAT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        formatter = logging.Formatter(self.FORMAT, datefmt=self.DATE_FORMAT)
        return formatter.format(record)


def setup_logger(
    task_dir: Optional[Path | str] = None,
    log_filename: str = "automl_agent.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Настраивает и возвращает корневой логгер пайплайна.
    Дублирует все события в файл и на консоль.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Очищаем старые хэндлеры во избежание дублирования
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(CustomFormatter())
    root_logger.addHandler(console_handler)

    # 2. File Handler (в корне проекта)
    root_log_path = Path(log_filename)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_file_handler = logging.FileHandler(root_log_path, mode="a", encoding="utf-8")
    root_file_handler.setLevel(logging.DEBUG)
    root_file_handler.setFormatter(file_formatter)
    root_logger.addHandler(root_file_handler)

    # 3. File Handler в каталоге задачи (если указан)
    if task_dir is not None:
        task_log_path = Path(task_dir) / log_filename
        task_file_handler = logging.FileHandler(task_log_path, mode="a", encoding="utf-8")
        task_file_handler.setLevel(logging.DEBUG)
        task_file_handler.setFormatter(file_formatter)
        root_logger.addHandler(task_file_handler)

    return logging.getLogger("automl_agent")
