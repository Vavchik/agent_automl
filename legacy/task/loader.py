"""Р2. Понимание задачи: файлы, таргет, id, тип задачи, метрика, формат ответа.

Сначала эвристики по файлам и тексту, потом (если доступна LLM) task/llm_parser.py
дополняет то, что эвристики не нашли. Всё, что решено здесь, лежит в TaskSpec.

Что улучшать (Р2):
- надёжнее определять метрику и формат ответа по тексту и sample_submission;
- задачи с несколькими таргетами, нестандартные имена файлов, другой разделитель;
- проверить на синтетических задачах из bench/tasks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from core.contracts import TaskSpec
from runtime.logger import log
from task.metrics import METRICS, detect_metric


def find_file(task_dir: Path, name: str) -> Optional[Path]:
    direct = task_dir / name
    if direct.exists():
        return direct
    found = sorted(task_dir.rglob(name))
    return found[0] if found else None


def read_dictionary(dict_path: Optional[Path]) -> tuple[dict[str, str], dict[str, str]]:
    """Возвращает (роли колонок, описания колонок) из data_dictionary.csv."""
    roles, descriptions = {}, {}
    if dict_path is None:
        return roles, descriptions
    try:
        dd = pd.read_csv(dict_path)
    except Exception as e:
        log.warning(f"Не удалось прочитать словарь: {e}")
        return roles, descriptions
    name_col = next((c for c in ("feature", "column", "name") if c in dd.columns), dd.columns[0])
    for _, row in dd.iterrows():
        feat = str(row[name_col])
        if "role" in dd.columns and pd.notna(row["role"]):
            roles[feat] = str(row["role"]).strip().lower()
        desc = row.get("description_ru", row.get("description", ""))
        unit = row.get("unit", "")
        text = str(desc) if pd.notna(desc) else ""
        if pd.notna(unit) and str(unit).strip():
            text += f" [{unit}]"
        if text:
            descriptions[feat] = text
    return roles, descriptions


def detect_task_type(y: pd.Series, metric: Optional[str]) -> str:
    n_unique = y.nunique(dropna=True)
    if metric is not None and METRICS[metric].family == "regression":
        return "regression"
    if metric is not None and METRICS[metric].family == "classification":
        return "binary" if n_unique == 2 else "multiclass"
    # метрики нет — решаем по таргету
    if pd.api.types.is_numeric_dtype(y):
        is_integer = bool((y.dropna() % 1 == 0).all())
        if not is_integer or n_unique > 20:
            return "regression"
    return "binary" if n_unique == 2 else "multiclass"


def default_metric(task_type: str) -> str:
    return {"regression": "rmse", "binary": "roc_auc", "multiclass": "macro_f1"}[task_type]


def detect_output(task_type: str, metric: str, sample: pd.DataFrame) -> str:
    if task_type == "regression":
        return "value"
    if METRICS[metric].needs == "proba":
        return "proba_per_class" if sample.shape[1] > 2 else "proba"
    return "label"


def load_task(task_md: Path | str) -> tuple[TaskSpec, pd.DataFrame, pd.DataFrame]:
    task_md = Path(task_md).resolve()
    task_dir = task_md.parent
    text = task_md.read_text(encoding="utf-8")

    paths = {name: find_file(task_dir, f"{name}.csv") for name in ("train", "test", "sample_submission")}
    missing = [n for n, p in paths.items() if p is None]
    if missing:
        raise FileNotFoundError(f"Не найдены файлы {missing} в {task_dir}")
    dict_path = find_file(task_dir, "data_dictionary.csv")

    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
    sample = pd.read_csv(paths["sample_submission"])
    roles, descriptions = read_dictionary(dict_path)

    id_col = next((c for c, r in roles.items() if r in ("id", "identifier") and c in test.columns), sample.columns[0])
    target_col = next((c for c, r in roles.items() if r == "target" and c in train.columns), None)
    if target_col is None:
        candidates = [c for c in sample.columns[1:] if c in train.columns]
        target_col = candidates[0] if candidates else [c for c in train.columns if c not in test.columns][0]

    metric = detect_metric(text)
    task_type = detect_task_type(train[target_col], metric)
    if metric is None or METRICS[metric].family != ("regression" if task_type == "regression" else "classification"):
        metric = default_metric(task_type)
    if metric == "f1" and task_type == "multiclass":
        metric = "macro_f1"

    spec = TaskSpec(
        task_dir=task_dir,
        text=text,
        train_path=paths["train"],
        test_path=paths["test"],
        sample_sub_path=paths["sample_submission"],
        dict_path=dict_path,
        target_col=target_col,
        id_col=id_col,
        task_type=task_type,
        metric=metric,
        output=detect_output(task_type, metric, sample),
        submission_columns=list(sample.columns),
        feature_descriptions=descriptions,
    )
    log.info(f"Задача: {task_type}, метрика {metric}, таргет '{target_col}', id '{id_col}', ответ: {spec.output}")
    log.info(f"train {train.shape}, test {test.shape}")
    return spec, train, test
