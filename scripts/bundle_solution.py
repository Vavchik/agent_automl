#!/usr/bin/env python3
"""
Скрипт сборки автономного архива solution.zip для платформы Boosters (Альфа Хак).

Требования платформы:
- В корне архива только файлы, БЕЗ папок:
  * solution.py — точка входа;
  * requirements.txt — зависимости;
- При проверке в каталог задачи копируются ТОЛЬКО файлы из корня архива.

Данный скрипт объединяет всю модульную кодовую базу в один автономный,
проверенный solution.py и упаковывает solution.zip.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"


def build_unified_solution(out_path: Path) -> None:
    """Генерирует единый, самодостаточный solution.py со всеми компонентами."""

    header = '''"""Единая точка входа для платформы Boosters (Альфа Хак: AutoML).
Запуск: python3 solution.py task.md
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
import optuna
from openai import OpenAI

'''

    # Читаем компоненты
    def read_clean(rel_path: str) -> str:
        lines = (ROOT / rel_path).read_text(encoding="utf-8").splitlines()
        clean_lines = []
        for line in lines:
            # Убираем внутренние импорты между нашими модулями
            if re.match(r"^(from|import)\s+(core|runtime|task|ml|agent|validation|config)($|[\s\.])", line.strip()):
                continue
            if line.startswith("from __future__ import annotations"):
                continue
            clean_lines.append(line)
        return "\n".join(clean_lines).strip()

    # Секция 1: Конфигурация
    sec_config = '''
# =========================================================================== #
# 1. КОНФИГУРАЦИЯ И ПАРАМЕТРЫ
# =========================================================================== #
def _env(name: str, default):
    value = os.getenv(name)
    if value is None:
        return default
    return type(default)(value)

FAST = os.getenv("AUTOML_FAST") == "1"
SEED = 42

TIME_LIMIT_SEC = _env("AUTOML_TIME_LIMIT", 3 * 3600.0)
SAFETY_MARGIN_SEC = _env("AUTOML_SAFETY_MARGIN", 60.0 if FAST else 15 * 60.0)
FAST_MAX_TRAIN_ROWS = 20_000

N_FOLDS = 3 if FAST else 5
ACCEPT_K = _env("AUTOML_ACCEPT_K", 1.0)

RESEARCH_MODELS = ["lgbm"]
FINAL_MODELS = ["lgbm", "catboost", "xgb"]
FINAL_TIME_FACTOR = 6.0
N_ESTIMATORS = 300 if FAST else 3000
LEARNING_RATE = 0.1 if FAST else 0.05
EARLY_STOPPING_ROUNDS = 50 if FAST else 100
N_JOBS = _env("AUTOML_N_JOBS", 8)

MAX_RESEARCH_ITERS = 4 if FAST else 40
SANDBOX_TIMEOUT_SEC = 120 if FAST else 600
TUNER_TIME_SEC = 60 if FAST else 20 * 60

LLM_TOKEN_BUDGET = _env("AUTOML_LLM_TOKEN_BUDGET", 150_000)
LLM_TIMEOUT_SEC = 90

# Псевдоним config для внутренней совместимости
config = sys.modules[__name__]
sys.modules['config'] = sys.modules[__name__]
'''

    # Секция 2: Runtime (Logger, Clock, Fallback)
    sec_runtime = '''
# =========================================================================== #
# 2. RUNTIME (Logger, Clock, Fallback)
# =========================================================================== #
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

class Clock:
    def __init__(self, limit_sec: float = TIME_LIMIT_SEC, margin_sec: float = SAFETY_MARGIN_SEC):
        self.start = time.time()
        self.limit_sec = limit_sec
        self.margin_sec = margin_sec

    def elapsed(self) -> float:
        return time.time() - self.start

    def remaining(self) -> float:
        return self.limit_sec - self.margin_sec - self.elapsed()

    def has(self, seconds: float) -> bool:
        return self.remaining() > seconds

def _find_file_fallback(task_dir: Path, name: str) -> Path | None:
    direct = task_dir / name
    if direct.exists():
        return direct
    found = sorted(task_dir.rglob(name))
    return found[0] if found else None

def write_fallback(task_md: Path) -> Path:
    task_dir = Path(task_md).resolve().parent
    out = task_dir / "submission.csv"
    sample_path = _find_file_fallback(task_dir, "sample_submission.csv")
    sample = pd.read_csv(sample_path)
    try:
        test = pd.read_csv(_find_file_fallback(task_dir, "test.csv"))
        train = pd.read_csv(_find_file_fallback(task_dir, "train.csv"))
        id_col = sample.columns[0]
        sub = pd.DataFrame({id_col: test[id_col]})
        for col in sample.columns[1:]:
            if col in train.columns:
                target = train[col].dropna()
                is_float_sample = pd.api.types.is_float_dtype(sample[col])
                if pd.api.types.is_numeric_dtype(target) and (target.nunique() > 20 or is_float_sample):
                    sub[col] = float(target.median())
                else:
                    sub[col] = target.mode().iloc[0]
            else:
                sub[col] = sample[col].iloc[0]
        sub.to_csv(out, index=False)
    except Exception:
        sample.to_csv(out, index=False)
    return out
'''

    # Секция 3: Контракты данных
    sec_contracts = '''
# =========================================================================== #
# 3. КОНТРАКТЫ ДАННЫХ
# =========================================================================== #
@dataclass
class TaskSpec:
    task_dir: Path
    text: str
    train_path: Path
    test_path: Path
    sample_sub_path: Path
    dict_path: Optional[Path]
    target_col: str
    id_col: str
    task_type: str                 # "regression" | "binary" | "multiclass"
    metric: str
    output: str                    # "value" | "label" | "proba" | "proba_per_class"
    submission_columns: list[str]
    time_col: Optional[str] = None
    group_col: Optional[str] = None
    feature_descriptions: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @property
    def is_classification(self) -> bool:
        return self.task_type in ("binary", "multiclass")

@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    X_test: pd.DataFrame
    test_ids: pd.Series
    cat_cols: list[str]

@dataclass
class ModelConfig:
    models: list[str] = field(default_factory=lambda: ["lgbm"])
    params: dict[str, dict] = field(default_factory=dict)
    target_transform: Optional[str] = None

    def copy(self) -> "ModelConfig":
        return ModelConfig(list(self.models), {k: dict(v) for k, v in self.params.items()}, self.target_transform)

@dataclass
class ExperimentResult:
    name: str
    score: float
    fold_scores: list[float]
    oof: np.ndarray
    test_pred: np.ndarray
    classes: Optional[np.ndarray]
    seconds: float
    feature_importance: dict[str, float] = field(default_factory=dict)

Folds = list[tuple[np.ndarray, np.ndarray]]
'''

    # Секция 4: Реестр метрик
    sec_metrics = '''
# =========================================================================== #
# 4. РЕЕСТР МЕТРИК
# =========================================================================== #
@dataclass(frozen=True)
class Metric:
    fn: Callable
    higher_is_better: bool
    family: str          # "regression" | "classification"
    needs: str           # "value" | "label" | "proba"

def _rmse(y, p):
    return float(np.sqrt(skm.mean_squared_error(y, p)))

def _rmsle(y, p):
    return float(np.sqrt(skm.mean_squared_log_error(np.clip(y, 0, None), np.clip(p, 0, None))))

def _roc_auc(y, p):
    if p.ndim == 2:
        return float(skm.roc_auc_score(y, p, multi_class="ovr", labels=np.arange(p.shape[1])))
    return float(skm.roc_auc_score(y, p))

def _logloss(y, p):
    labels = np.arange(p.shape[1]) if p.ndim == 2 else [0, 1]
    return float(skm.log_loss(y, p, labels=labels))

METRICS: dict[str, Metric] = {
    "rmse": Metric(_rmse, False, "regression", "value"),
    "mse": Metric(lambda y, p: float(skm.mean_squared_error(y, p)), False, "regression", "value"),
    "mae": Metric(lambda y, p: float(skm.mean_absolute_error(y, p)), False, "regression", "value"),
    "rmsle": Metric(_rmsle, False, "regression", "value"),
    "mape": Metric(lambda y, p: float(skm.mean_absolute_percentage_error(y, p)), False, "regression", "value"),
    "r2": Metric(lambda y, p: float(skm.r2_score(y, p)), True, "regression", "value"),
    "macro_f1": Metric(lambda y, p: float(skm.f1_score(y, p, average="macro")), True, "classification", "label"),
    "weighted_f1": Metric(lambda y, p: float(skm.f1_score(y, p, average="weighted")), True, "classification", "label"),
    "f1": Metric(lambda y, p: float(skm.f1_score(y, p)), True, "classification", "label"),
    "accuracy": Metric(lambda y, p: float(skm.accuracy_score(y, p)), True, "classification", "label"),
    "roc_auc": Metric(_roc_auc, True, "classification", "proba"),
    "pr_auc": Metric(lambda y, p: float(skm.average_precision_score(y, p)), True, "classification", "proba"),
    "logloss": Metric(_logloss, False, "classification", "proba"),
}

METRIC_PATTERNS: list[tuple[str, str]] = [
    (r"macro[\s_-]?f1|f1[\s_-]?macro|макро[\s-]?f1", "macro_f1"),
    (r"weighted[\s_-]?f1|f1[\s_-]?weighted|взвешенн\w* f1", "weighted_f1"),
    (r"pr[\s_-]?auc|average[\s_]precision", "pr_auc"),
    (r"roc[\s_-]?auc|\bauc\b|gini|джини", "roc_auc"),
    (r"log[\s_-]?loss|кросс-энтропи|cross[\s-]?entropy", "logloss"),
    (r"rmsle", "rmsle"),
    (r"rmse|root mean squared", "rmse"),
    (r"\bmae\b|mean absolute error|средняя абсолютная", "mae"),
    (r"\bmape\b", "mape"),
    (r"\bmse\b", "mse"),
    (r"\br2\b|r²|r\^2|детерминации", "r2"),
    (r"\bf1\b", "f1"),
    (r"accuracy|доля правильных", "accuracy"),
]

def detect_metric(text: str) -> Optional[str]:
    low = text.lower()
    start = low.find("метрик")
    if start < 0:
        start = low.find("metric")
    region = low[start:] if start >= 0 else low
    best: tuple[int, int, str] | None = None
    for order, (pattern, name) in enumerate(METRIC_PATTERNS):
        match = re.search(pattern, region)
        if match and (best is None or (match.start(), order) < best[:2]):
            best = (match.start(), order, name)
    return best[2] if best else None

def score(metric: str, y_true, pred) -> float:
    return METRICS[metric].fn(np.asarray(y_true), np.asarray(pred))

def improvement(metric: str, new: float, old: float) -> float:
    sign = 1.0 if METRICS[metric].higher_is_better else -1.0
    return sign * (new - old)
'''

    # Секция 5: LLM Client
    sec_llm = '''
# =========================================================================== #
# 5. LLM КЛИЕНТ (Qwen3.8-27B)
# =========================================================================== #
class LLM:
    def __init__(self):
        self.model = os.getenv("OPENAI_MODEL", "qwen/qwen3.8-27b")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://changellenge.up.railway.app/v1")
        self.backup_url = "https://api.alfahack-gateway26.ru/v1"
        self.enabled = bool(os.getenv("OPENAI_API_KEY"))
        self.used_tokens = 0
        self.failures = 0
        self._client = None

    def available(self) -> bool:
        return self.enabled and self.used_tokens < LLM_TOKEN_BUDGET and self.failures < 3

    def _get_client(self, url: str) -> OpenAI:
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=url, timeout=LLM_TIMEOUT_SEC, max_retries=1)

    def ask(self, system: str, user: str, max_tokens: int = 1500, temperature: float = 0.3) -> str | None:
        if not self.available():
            return None
        endpoints = [self.base_url]
        if self.base_url != self.backup_url:
            endpoints.append(self.backup_url)
        for ep in endpoints:
            try:
                client = self._get_client(ep)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                self.failures = 0
                text = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                self.used_tokens += usage.total_tokens if usage else (len(system) + len(user) + len(text)) // 3
                return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            except Exception as e:
                log.warning(f"Ошибка запроса к LLM ({ep}): {e}")
        self.failures += 1
        return None

    def ask_json(self, system: str, user: str, max_tokens: int = 1000) -> dict | None:
        text = self.ask(system, user, max_tokens=max_tokens, temperature=0.1)
        if not text:
            return None
        match = re.search(r"\{.*\}", text, flags=re.S)
        try:
            return json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            return None

def extract_code(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
    code = match.group(1) if match else text
    return code if "def transform_features" in code else None

llm = LLM()
'''

    # Секция 6: Загрузка задачи и типы
    sec_task = read_clean("task/dtypes.py") + "\n\n" + read_clean("task/loader.py") + "\n\n" + read_clean("task/llm_parser.py") + "\n\n" + read_clean("task/submission.py")

    # Секция 7: Валидация
    sec_validation = read_clean("validation/splits.py") + "\n\n" + read_clean("validation/acceptance.py") + "\n\n" + read_clean("validation/adversarial.py")

    # Секция 8: ML-ядро
    sec_ml = read_clean("ml/preprocess.py") + "\n\n" + read_clean("ml/models.py") + "\n\n" + read_clean("ml/postprocess.py") + "\n\n" + read_clean("ml/trainer.py") + "\n\n" + read_clean("ml/tuner.py") + "\n\n" + read_clean("ml/ensemble.py")

    # Секция 9: Агент и AutoResearch
    sec_agent = read_clean("agent/sandbox.py") + "\n\n" + read_clean("agent/memory.py") + "\n\n" + read_clean("agent/actions.py") + "\n\n" + read_clean("agent/planner.py") + "\n\n" + read_clean("agent/research_loop.py")

    # Секция 10: Пайплайн и точка входа main
    sec_pipeline = read_clean("core/pipeline.py") + '''

def main() -> None:
    task_md = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md").resolve()
    try:
        run(task_md)
    except BaseException:
        traceback.print_exc()
        if not (task_md.parent / "submission.csv").exists():
            path = write_fallback(task_md)
            print(f"Пайплайн упал до первого сабмита, записан аварийный ответ: {path}")

if __name__ == "__main__":
    main()
'''

    full_code = header + sec_config + sec_runtime + sec_contracts + sec_metrics + sec_llm + "\n\n" + sec_task + "\n\n" + sec_validation + "\n\n" + sec_ml + "\n\n" + sec_agent + "\n\n" + sec_pipeline

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_code, encoding="utf-8")
    print(f"✓ Собран единый solution.py: {out_path} ({len(full_code.splitlines())} строк)")


def package_zip(solution_py: Path, requirements_txt: Path, out_zip: Path) -> None:
    """Упаковывает строго в корень архива solution.py и requirements.txt."""
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(solution_py, arcname="solution.py")
        z.write(requirements_txt, arcname="requirements.txt")
    print(f"✓ Упакован {out_zip} ({out_zip.stat().st_size} байт):")
    with zipfile.ZipFile(out_zip, "r") as z:
        for info in z.infolist():
            print(f"   • {info.filename} ({info.file_size} байт)")


def main():
    dist_dir = ROOT / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    unified_solution = dist_dir / "solution.py"
    req_file = ROOT / "requirements.txt"
    out_zip = ROOT / "solution.zip"

    build_unified_solution(unified_solution)
    shutil.copy(req_file, dist_dir / "requirements.txt")
    package_zip(unified_solution, req_file, out_zip)
    print("\n✓ Готово к отправке на платформу Boosters!")


if __name__ == "__main__":
    main()
