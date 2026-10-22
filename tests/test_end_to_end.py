"""Р1: solution.py на синтетических задачах разных типов всегда оставляет валидный сабмит."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from bench import make_tasks

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("make", make_tasks.ALL, ids=lambda f: f.__name__)
def test_solution_on_synthetic_task(tmp_path, make):
    task_md = make(n=1500, out_root=tmp_path)
    env = dict(os.environ, AUTOML_FAST="1", AUTOML_TIME_LIMIT="150", AUTOML_N_JOBS="2")
    env.pop("OPENAI_API_KEY", None)
    subprocess.run([sys.executable, str(ROOT / "solution.py"), str(task_md)], env=env, check=True, timeout=600)
    sub = pd.read_csv(task_md.parent / "submission.csv")
    sample = pd.read_csv(task_md.parent / "sample_submission.csv")
    assert list(sub.columns) == list(sample.columns)
    assert len(sub) == len(sample) and not sub.isna().any().any()
    report = json.loads((task_md.parent / "report.json").read_text(encoding="utf-8"))
    assert report["best_score"] is not None


def test_fallback_when_pipeline_crashes(tmp_path):
    task_md = make_tasks.binary_auc(n=300, out_root=tmp_path)
    (task_md.parent / "test.csv").write_text("broken\n", encoding="utf-8")   # ломаем test
    subprocess.run([sys.executable, str(ROOT / "solution.py"), str(task_md)], timeout=300)
    assert (task_md.parent / "submission.csv").exists()
