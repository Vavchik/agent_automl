"""Р1. Все настройки агента в одном месте.

Быстрый режим для отладки и стенда: AUTOML_FAST=1 (подвыборка, меньше деревьев и итераций).
Любое значение можно переопределить переменной окружения, не меняя код.
"""
import os


def _env(name: str, default):
    value = os.getenv(name)
    if value is None:
        return default
    return type(default)(value)


FAST = os.getenv("AUTOML_FAST") == "1"
SEED = 42

# --- Время (лимит платформы: 3 часа на задачу) ---
TIME_LIMIT_SEC = _env("AUTOML_TIME_LIMIT", 3 * 3600.0)
SAFETY_MARGIN_SEC = _env("AUTOML_SAFETY_MARGIN", 60.0 if FAST else 15 * 60.0)

# --- Данные ---
FAST_MAX_TRAIN_ROWS = 20_000          # в быстром режиме обучаемся на подвыборке

# --- Валидация (Р5) ---
N_FOLDS = 3 if FAST else 5
ACCEPT_K = _env("AUTOML_ACCEPT_K", 1.0)  # гипотеза принимается, если прирост > K * шум между фолдами

# --- Модели (Р3) ---
RESEARCH_MODELS = ["lgbm"]                    # быстрые модели для проверки гипотез
FINAL_MODELS = ["lgbm", "catboost", "xgb"]    # бленд на лучших признаках в конце
FINAL_TIME_FACTOR = 6.0                       # финал ~ во столько раз дольше одного эксперимента
N_ESTIMATORS = 300 if FAST else 3000  # верхняя граница, реально останавливает early stopping
LEARNING_RATE = 0.1 if FAST else 0.05
EARLY_STOPPING_ROUNDS = 50 if FAST else 100
N_JOBS = _env("AUTOML_N_JOBS", 8)

# --- Исследование (Р4) ---
MAX_RESEARCH_ITERS = 4 if FAST else 40
SANDBOX_TIMEOUT_SEC = 120 if FAST else 600
TUNER_TIME_SEC = 60 if FAST else 20 * 60

# --- LLM (Р4) ---
LLM_TOKEN_BUDGET = _env("AUTOML_LLM_TOKEN_BUDGET", 150_000)  # на один запуск задачи
LLM_TIMEOUT_SEC = 90
