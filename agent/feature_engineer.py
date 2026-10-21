"""
Модуль генерации признаков (Feature Engineer) на базе LLM Qwen3.8-27B.

Отвечает за:
1. Формирование обогащенного промпта с доменным контекстом (task.md),
   словарем признаков (data_dictionary.csv), историей рефлексии и топ-важными фичами.
2. Генерацию чистой Python-функции:
   `def transform_features(df: pd.DataFrame) -> pd.DataFrame:`
3. Безопасное тестирование трансформации в памяти или песочнице.
4. Доменные fallback-эвристики на случай недоступности LLM или исчерпания токенов.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from agent.llm import ask_llm, extract_code

logger = logging.getLogger("agent.feature_engineer")


FEATURE_SYSTEM_PROMPT = """Ты — Senior Kaggle Grandmaster и ведущий ML-инженер, эксперт в Feature Engineering для табличных данных.
Твоя цель — придумать и написать чистый Python-код, создающий НОВЫЕ сильные доменные признаки для табличной модели (LightGBM/CatBoost).

Строгие правила:
1. Код должен содержать ровно одну функцию с сигнатурой:
   def transform_features(df: pd.DataFrame) -> pd.DataFrame:
2. Внутри функции:
   - Делай `df = df.copy()` в самом начале.
   - НЕ пытайся использовать целевую переменную (таргет)! Ее нет в тестовых данных.
   - Создавай 3-7 новых мощных признаков:
     * GroupBy агрегации (ОБЯЗАТЕЛЬНО для высокого скора):
       mean_val = df.groupby('cat_col')['num_col'].transform('mean')
       df['f_diff_num_cat'] = df['num_col'] - mean_val
       df['f_ratio_num_cat'] = df['num_col'] / (mean_val.abs() + 1e-5)
     * Доменные соотношения и взаимодействия:
       df['f_ratio_a_b'] = df['col_a'] / (df['col_b'].abs() + 1e-5)
       df['f_mult_a_b'] = df['col_a'] * df['col_b']
     * Частотное кодирование категорий:
       df['f_freq_cat'] = df['cat_col'].astype(str).map(df['cat_col'].astype(str).value_counts(normalize=True)).fillna(0)
   - Защищай деление от нуля: `a / (b.abs() + 1e-5)`.
   - Заполняй возможные inf и nan: `df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)`.
   - Возвращай преобразованный `df`.
3. Никаких побочных эффектов (print, сохранение на диск, импорт запрещенных библиотек).
4. Разрешены только `pandas as pd` и `numpy as np`.
5. Верни ответ в формате:
   Краткое обоснование гипотезы (2-3 предложения)
   ```python
   def transform_features(df: pd.DataFrame) -> pd.DataFrame:
       ...
   ```
"""


def _extract_python_code(llm_response: str) -> str:
    """Извлекает блок python-кода из markdown-ответа LLM."""
    match = re.search(r"```(?:python)?\s*([\s\S]*?)```", llm_response)
    if match:
        code = match.group(1).strip()
    else:
        code = llm_response.strip()

    # Проверяем наличие функции
    if "def transform_features" not in code:
        raise ValueError("В ответе LLM не найдена функция def transform_features(df)")
    return code


def _extract_hypothesis_description(llm_response: str) -> str:
    """Извлекает текстовое описание гипотезы до блока с кодом."""
    parts = re.split(r"```(?:python)?", llm_response)
    if parts and parts[0].strip():
        # Берем первые 200 символов описания
        desc = parts[0].strip().replace("\n", " ")
        return desc[:250]
    return "Доменный Feature Engineering от LLM"


def generate_feature_code(
    task_description: str,
    feature_dictionary: dict[str, str],
    metric_name: str,
    current_best_score: float,
    top_features: list[tuple[str, float]],
    reflection_history: str = "",
    iteration: int = 1,
) -> tuple[str, str]:
    """
    Запрашивает у LLM гипотезу и код для генерации признаков.
    Возвращает (код_функции, описание_гипотезы).
    """
    # Собираем топ важных признаков
    top_f_str = ", ".join([f"{name} (imp: {imp:.3f})" for name, imp in top_features[:10]]) or "нет данных"

    # Собираем словарь признаков (до 25 штук чтобы экономить токены)
    dict_items = list(feature_dictionary.items())[:25]
    dict_str = "\n".join([f"- `{col}`: {desc}" for col, desc in dict_items]) or "Словарь отсутствует"

    user_prompt = f"""### Контекст задачи:
{task_description[:800]}

### Описание доступных признаков (словарь):
{dict_str}

### Текущее состояние модели:
- Целевая метрика: {metric_name}
- Текущий лучший CV-скор: {current_best_score:.5f}
- Топ-10 самых сильных признаков базовой модели: {top_f_str}

### История предыдущих попыток (Рефлексия):
{reflection_history}

### Итерация #{iteration}:
Сформулируй новую доменную гипотезу и напиши функцию `def transform_features(df: pd.DataFrame) -> pd.DataFrame:`.
Помни: новые признаки должны логически объединять сильнейшие признаки с другими доменными факторами.
"""

    try:
        response = ask_llm(
            system_prompt=FEATURE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=1500,
        )
        code = _extract_python_code(response)
        desc = _extract_hypothesis_description(response)
        return code, desc
    except Exception as e:
        logger.warning(f"Ошибка вызова LLM для Feature Engineering: {e}. Используем доменный fallback.")
        return generate_heuristic_features(top_features, iteration)


def generate_heuristic_features(
    top_features: list[tuple[str, float]],
    iteration: int = 1,
) -> tuple[str, str]:
    """
    Детерминированный эвристический генератор признаков (Kaggle Baseline Hacks),
    когда LLM недоступен или исчерпан лимит токенов.
    """
    feat_names = [f[0] for f in top_features[:6]]

    if len(feat_names) >= 2 and iteration == 1:
        f1, f2 = feat_names[0], feat_names[1]
        code = f"""def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        if '{f1}' in df.columns and '{f2}' in df.columns:
            # Отношение и произведение двух важнейших числовых признаков
            v1 = pd.to_numeric(df['{f1}'], errors='coerce').fillna(0)
            v2 = pd.to_numeric(df['{f2}'], errors='coerce').fillna(0)
            df['f_ratio_{f1}_{f2}'] = v1 / (np.abs(v2) + 1e-5)
            df['f_mult_{f1}_{f2}'] = v1 * v2
            df['f_diff_{f1}_{f2}'] = v1 - v2
    except Exception:
        pass
    return df
"""
        return code, f"Взаимодействие топ-2 признаков {f1} и {f2} (ratio, mult, diff)"

    if len(feat_names) >= 1 and iteration == 2:
        f1 = feat_names[0]
        code = f"""def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        if '{f1}' in df.columns:
            v1 = pd.to_numeric(df['{f1}'], errors='coerce').fillna(0)
            df['f_log_{f1}'] = np.log1p(np.maximum(0, v1))
            df['f_sqrt_{f1}'] = np.sqrt(np.maximum(0, v1))
    except Exception:
        pass
    return df
"""
        return code, f"Нелинейные преобразования (log1p, sqrt) для признака {f1}"

    # Generic fallback
    num_cols = [f[0] for f in top_features[:4]]
    if len(num_cols) >= 2:
        c1, c2 = num_cols[0], num_cols[1]
        code = f"""def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        v1 = pd.to_numeric(df['{c1}'], errors='coerce').fillna(0)
        v2 = pd.to_numeric(df['{c2}'], errors='coerce').fillna(0)
        df['f_num_sum'] = v1 + v2
        df['f_num_diff'] = v1 - v2
        df['f_num_ratio'] = v1 / (v2.abs() + 1e-5)
    except Exception:
        pass
    return df
"""
        return code, f"Линейные комбинации и отношения признаков {c1} и {c2}"

    return "def transform_features(df: pd.DataFrame) -> pd.DataFrame:\n    return df.copy()", "Пустая трансформация"


def execute_feature_transform(
    code_str: str,
    df_train: Any,
    df_test: Optional[Any] = None,
    target_col: Optional[str] = None,
) -> tuple[Any, Optional[Any], Optional[str]]:
    """
    Безопасно исполняет сгенерированный код трансформации на копиях train и test.
    - Изолирует целевую переменную target_col (защита от Target Leakage и KeyError на test).
    - Преобразует Categorical типы в object для защиты от TypeError в Pandas 3.0.
    - Автоматически очищает inf и nan в новых числовых признаках.
    Возвращает (new_train_df, new_test_df, error_str).
    """
    if pd is None:
        return df_train, df_test, "Pandas is not installed"

    local_namespace: dict[str, Any] = {
        "np": np,
        "pd": pd,
    }

    try:
        # Компилируем и выполняем определение функции
        exec(code_str, local_namespace)
        transform_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = local_namespace.get("transform_features")

        if not callable(transform_fn):
            return df_train, df_test, "Объект transform_features не является вызываемой функцией"

        # Изолируем таргет, чтобы функция LLM не могла опереться на него (Data Leakage)
        target_series = None
        tr_input = df_train.copy()
        if target_col and target_col in tr_input.columns:
            target_series = tr_input[target_col].copy()
            tr_input = tr_input.drop(columns=[target_col])

        # Защита от Pandas CategoricalDtype setitem/fillna ошибок:
        # Конвертируем категориальные колонки в object для трансформации
        for col in tr_input.select_dtypes(include=["category"]).columns:
            tr_input[col] = tr_input[col].astype(object)

        te_input = None
        if df_test is not None:
            te_input = df_test.copy()
            if target_col and target_col in te_input.columns:
                te_input = te_input.drop(columns=[target_col])
            for col in te_input.select_dtypes(include=["category"]).columns:
                te_input[col] = te_input[col].astype(object)

        # Применяем трансформацию к train
        new_tr = transform_fn(tr_input)
        if not isinstance(new_tr, pd.DataFrame):
            return df_train, df_test, "transform_features должна возвращать pd.DataFrame"

        if len(new_tr) != len(df_train):
            return df_train, df_test, f"Количество строк изменилось: {len(new_tr)} != {len(df_train)}"

        # Применяем к test при наличии
        new_te = None
        if te_input is not None:
            new_te = transform_fn(te_input)
            if len(new_te) != len(df_test):
                return df_train, df_test, f"Количество строк в test изменилось: {len(new_te)} != {len(df_test)}"

        # Возвращаем таргет в train выборку
        if target_series is not None and target_col:
            new_tr[target_col] = target_series

        # Проверяем, что число колонок не уменьшилось относительно исходных
        if len(new_tr.columns) < len(df_train.columns):
            return df_train, df_test, "Количество признаков уменьшилось (удалены базовые колонки)"

        # Санитизируем созданные признаки (inf / -inf -> 0.0, nan -> 0.0)
        base_cols = set(df_train.columns)
        for col in new_tr.columns:
            if col not in base_cols:
                if pd.api.types.is_numeric_dtype(new_tr[col]):
                    new_tr[col] = new_tr[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    if new_te is not None and col in new_te.columns:
                        new_te[col] = new_te[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return new_tr, new_te, None

    except Exception as e:
        logger.error(f"Ошибка исполнения кода признаков: {e}", exc_info=True)
        return df_train, df_test, f"{type(e).__name__}: {str(e)}"
