"""Р4. Исполнение кода от LLM в отдельном процессе с таймаутом.

Код должен определить:
def transform_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]

Все агрегаты и статистики считаются ТОЛЬКО на train_df и проецируются на test_df.
Зависший или упавший код не роняет основной процесс.
"""
from __future__ import annotations

import inspect
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

import config

RUNNER = """
import inspect
import sys
import numpy as np
import pandas as pd

data = pd.read_pickle(sys.argv[1])
train_df = data["train"]
test_df = data["test"]

ns = {"np": np, "pd": pd}
exec(open(sys.argv[2], encoding="utf-8").read(), ns)

if "transform_features" not in ns:
    raise NameError("Функция 'transform_features' не найдена в сгенерированном коде")

fn = ns["transform_features"]
sig = inspect.signature(fn)
params_count = len(sig.parameters)

if params_count == 1:
    train_out = fn(train_df.copy())
    test_out = fn(test_df.copy())
else:
    res = fn(train_df.copy(), test_df.copy())
    if not isinstance(res, (tuple, list)) or len(res) != 2:
        raise TypeError("transform_features должна вернуть кортеж (train_df, test_df)")
    train_out, test_out = res

if not isinstance(train_out, pd.DataFrame) or not isinstance(test_out, pd.DataFrame):
    raise TypeError("Оба возвращенных объекта должны быть pandas DataFrame")

if len(train_out) != len(train_df):
    raise ValueError(f"Число строк в train изменилось: {len(train_out)} != {len(train_df)}")

if len(test_out) != len(test_df):
    raise ValueError(f"Число строк в test изменилось: {len(test_out)} != {len(test_df)}")

pd.to_pickle({"train": train_out, "test": test_out}, sys.argv[3])
"""


def run_feature_code(code: str, train_df: pd.DataFrame, test_df: pd.DataFrame | None = None,
                     timeout: float = config.SANDBOX_TIMEOUT_SEC):
    is_single = test_df is None
    if test_df is None:
        test_df = train_df.iloc[:0].copy()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pd.to_pickle({"train": train_df, "test": test_df}, tmp / "in.pkl")
        (tmp / "code.py").write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-c", RUNNER, str(tmp / "in.pkl"), str(tmp / "code.py"), str(tmp / "out.pkl")],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, f"Таймаут выполнения {timeout:.0f} с"
        if proc.returncode != 0:
            return None, proc.stderr.strip()[-2000:]
        out = pd.read_pickle(tmp / "out.pkl")
        if is_single:
            return out["train"], None
        return (out["train"], out["test"]), None
