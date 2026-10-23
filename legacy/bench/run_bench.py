"""Р5. Стенд: прогоняет solution.py на всех задачах и собирает таблицу.

python bench/run_bench.py            # быстрый режим (AUTOML_FAST=1), публичные + синтетические задачи
python bench/run_bench.py --full     # полный режим (долго)
Результат: таблица в консоли и bench/results.csv (дописывается, с датой — видно динамику).
Для синтетических задач считается настоящий скор по скрытым ответам.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from task.metrics import score  # noqa: E402


def true_score(task_dir: Path, report: dict) -> float | None:
    answers = ROOT / "bench" / "answers" / f"{task_dir.name}.csv"
    sub_path = task_dir / "submission.csv"
    if not answers.exists() or not sub_path.exists():
        return None
    ans, sub = pd.read_csv(answers), pd.read_csv(sub_path)
    id_col, target = ans.columns
    merged = ans.merge(sub, on=id_col, suffixes=("", "_pred"))
    metric = report["metric"]
    if report["output"] == "proba_per_class":
        classes = sorted(c for c in sub.columns if c != id_col)
        codes = merged[target].map({c: i for i, c in enumerate(classes)}).to_numpy()
        return score(metric, codes, merged[classes].to_numpy())
    return score(metric, merged[target].to_numpy(), merged[f"{target}_pred"].to_numpy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--timeout", type=float, default=3 * 3600)
    parser.add_argument("--only", default="", help="запускать только задачи, в пути которых есть эта подстрока")
    args = parser.parse_args()

    if not (ROOT / "bench" / "tasks").exists():
        subprocess.run([sys.executable, str(ROOT / "bench" / "make_tasks.py")], check=True)
    tasks = sorted((ROOT / "given_data").glob("*/task.md")) + sorted((ROOT / "bench" / "tasks").glob("*/task.md"))
    tasks = [t for t in tasks if args.only in str(t)]
    env = dict(os.environ, AUTOML_FAST="0" if args.full else "1")

    rows = []
    for task_md in tasks:
        print(f"=== {task_md.parent.name}")
        start = time.time()
        proc = subprocess.run([sys.executable, str(ROOT / "solution.py"), str(task_md)], env=env,
                              capture_output=True, text=True, timeout=args.timeout)
        report_path = task_md.parent / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        crashed = proc.returncode != 0 or not report
        if crashed:
            print(proc.stdout[-3000:], proc.stderr[-3000:])
        rows.append({
            "date": time.strftime("%Y-%m-%d %H:%M"), "task": task_md.parent.name, "mode": "full" if args.full else "fast",
            "ok": not crashed and (task_md.parent / "submission.csv").exists(),
            "metric": report.get("metric"), "cv_baseline": report.get("baseline_score"),
            "cv_best": report.get("best_score"), "true_score": true_score(task_md.parent, report) if report else None,
            "accepted": report.get("accepted"), "validation": report.get("validation"),
            "seconds": round(time.time() - start), "llm_tokens": report.get("llm_tokens"),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    out = ROOT / "bench" / "results.csv"
    table.to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
