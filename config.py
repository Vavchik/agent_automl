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
FAST_MAX_TRAIN_ROWS = _env("AUTOML_FAST_MAX_TRAIN_ROWS", 200_000)          # используем полные данные

# --- Валидация (Р5) ---
N_FOLDS = 3 if FAST else 5
ACCEPT_K = _env("AUTOML_ACCEPT_K", 1.0)  # гипотеза принимается, если прирост > K * шум между фолдами

# --- Модели (Р3) ---
def _detect_final_models():
    models = ["lgbm"]
    try:
        import catboost  # noqa: F401
        models.append("catboost")
    except ImportError:
        pass
    try:
        import xgboost  # noqa: F401
        models.append("xgb")
    except ImportError:
        pass
    return models

RESEARCH_MODELS = ["lgbm"]                    # быстрые модели для проверки гипотез
FINAL_MODELS = _detect_final_models()          # бленд на лучших признаках в конце
FINAL_TIME_FACTOR = 6.0                       # финал ~ во столько раз дольше одного эксперимента
N_ESTIMATORS = 1000 if FAST else 3000  # верхняя граница, реально останавливает early stopping
LEARNING_RATE = 0.08 if FAST else 0.05
EARLY_STOPPING_ROUNDS = 50 if FAST else 100
N_JOBS = _env("AUTOML_N_JOBS", 8)

# AutoGluon best_quality на наших фолдах (Р3). 0 — выключить.
AUTOGLUON_TIME_SEC = _env("AUTOML_AG_TIME", 0.0 if FAST else 40 * 60.0)

# --- Исследование: дерево решений (Р4) ---
MAX_RESEARCH_ITERS = _env("AUTOML_MAX_ITERS", 8 if FAST else 60)
MAX_CHILDREN = 4          # сколько раз можно расширять один узел, потом поиск откатывается к другим
EXPLORE_PROB = 0.3        # вероятность расширить не лучший узел, а один из топ-3 (исследование веток)
KEEP_NODES = 8            # для скольких лучших узлов храним данные (остальные нельзя расширять)
MAX_DEBUG_ATTEMPTS = 2    # сколько раз LLM чинит упавший код по traceback
SANDBOX_TIMEOUT_SEC = 120 if FAST else 600
TUNER_TIME_SEC = 60 if FAST else 15 * 60

# --- LLM (Р4) ---
# На платформе ключ, адрес и модель приходят из переменных окружения OPENAI_* (как в baseline.py).
LLM_MODEL_DEFAULT = "qwen/qwen3.8-27b"
LLM_BASE_URL_DEFAULT = "https://changellenge.up.railway.app/v1"
LLM_BACKUP_URL = "https://api.alfahack-gateway26.ru/v1"
LLM_TOKEN_BUDGET = _env("AUTOML_LLM_TOKEN_BUDGET", 200_000)  # на один запуск задачи
LLM_TIMEOUT_SEC = 120
LLM_NO_THINK = True       # просим Qwen3 не тратить токены на скрытые рассуждения (/no_think)
