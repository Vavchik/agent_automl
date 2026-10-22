"""Р4. Исполнение кода от LLM в отдельном процессе с таймаутом.

Код должен определить `def transform_features(df) -> DataFrame`. df — признаки train и test
вместе (без таргета), поэтому любые агрегаты считаются одинаково для обеих выборок.
Зависший или упавший код не роняет основной процесс.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

import config

RUNNER = """
import sys
import numpy as np
import pandas as pd
df = pd.read_pickle(sys.argv[1])
ns = {"np": np, "pd": pd}
exec(open(sys.argv[2], encoding="utf-8").read(), ns)
out = ns["transform_features"](df.copy())
if not isinstance(out, pd.DataFrame):
    raise TypeError("transform_features должна вернуть DataFrame")
if len(out) != len(df):
    raise ValueError(f"Число строк изменилось: {len(out)} != {len(df)}")
out.to_pickle(sys.argv[3])
"""


def run_feature_code(code: str, df: pd.DataFrame,
                     timeout: float = config.SANDBOX_TIMEOUT_SEC) -> tuple[pd.DataFrame | None, str | None]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        df.to_pickle(tmp / "in.pkl")
        (tmp / "code.py").write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-c", RUNNER, str(tmp / "in.pkl"), str(tmp / "code.py"), str(tmp / "out.pkl")],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, f"Таймаут {timeout:.0f} с"
        if proc.returncode != 0:
            return None, proc.stderr.strip()[-1500:]
        return pd.read_pickle(tmp / "out.pkl"), None
