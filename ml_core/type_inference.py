"""
Стадия 1: Базовая предобработка и определение типов колонок (regex, to_numeric).

Пайплайн:
    1. Очистка типичного мусора в строках (запятые, пробелы, валюты, проценты).
    2. Агрессивная проверка на число (pd.to_numeric(errors='coerce')).
    3. Аналогичная проверка на дату (pd.to_datetime(errors='coerce')).
    4. Оставшиеся колонки: категория (мало уникальных) или текст.

Статистика и отчёт для LLM живут в `data_profiler.py`.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# Константы
# --------------------------------------------------------------------------- #

# Типы колонок (используются также в data_profiler.py)
NUMERIC = "numeric"
DATETIME = "datetime"
CATEGORY = "category"
TEXT = "text"
EMPTY = "empty"

# Доля уникальных значений от числа строк, ниже которой колонка считается категорией
CATEGORY_THRESHOLD = 0.1

# Токены, которые считаются пропуском
_NA_TOKENS = {"", "nan", "none", "null", "n/a", "na", "#n/a", "-", "—"}

# Знаки валют, проценты и частые текстовые обозначения валют
_CURRENCY_RE = re.compile(r"[$€£¥₽₴₸₹₩%]|(?<![^\W\d_])(?:руб|rub|usd|eur|грн)\.?(?![^\W\d_])", re.I)
# Любые пробельные символы (включая неразрывные и тонкие)
_SPACES_RE = re.compile(r"\s+")
# Тысячные разделители-запятые: 1,234,567 или 1,234,567.89 (не может начинаться с 0, т.к. 0,001 - это дробь)
_THOUSANDS_COMMA_RE = re.compile(r"^[+-]?[1-9]\d{0,2}(,\d{3})+(\.\d+)?$")
# Дата вида 12.03.2021 / 12/03/2021 / 12-03-21 (для определения порядка день/месяц)
_DMY_RE = re.compile(r"^(\d{1,2})([./-])(\d{1,2})[./-]\d{2,4}")


# --------------------------------------------------------------------------- #
# Шаг 1: очистка мусора
# --------------------------------------------------------------------------- #

def _normalize_missing(col: pd.Series) -> pd.Series:
    """Строки-пустышки ('', 'nan', 'null', '-', а также "' '" в кавычках) превращает в настоящие NaN."""
    def is_na_token(v: Any) -> bool:
        if not isinstance(v, str):
            return False
        cleaned = v.strip().strip("'\"").strip().lower()
        return cleaned in _NA_TOKENS

    return col.mask(col.map(is_na_token))


def _clean_number_text(txt: pd.Series) -> pd.Series:
    """
    Чистит строки, которые потенциально являются числами:
    убирает валюты/проценты и пробелы, заменяет запятую на точку.
    """
    txt = txt.str.replace("\u2212", "-", regex=False)      # unicode-минус
    txt = txt.str.replace(_CURRENCY_RE, "", regex=True)     # валюты и %
    txt = txt.str.replace(_SPACES_RE, "", regex=True)       # пробелы в числах
    # 1,234,567.89 -> 1234567.89 (запятая как разделитель тысяч)
    is_thousands = txt.str.match(_THOUSANDS_COMMA_RE)
    txt = txt.where(~is_thousands, txt.str.replace(",", "", regex=False))
    # 3,14 -> 3.14 (запятая как десятичный разделитель)
    return txt.str.replace(",", ".", regex=False)


# --------------------------------------------------------------------------- #
# Шаги 2-3: проверка на число и дату
# --------------------------------------------------------------------------- #

def _try_numeric(col: pd.Series) -> Optional[pd.Series]:
    """
    Возвращает числовую версию колонки, если to_numeric(coerce) не породил
    новых NaN (т.е. все непустые значения удалось распарсить), иначе None.
    """
    mask = col.notna()
    cleaned = _clean_number_text(col[mask].astype(str))
    converted = pd.to_numeric(cleaned, errors="coerce")

    if converted.isna().sum() != 0:  # появились новые NaN -> не число
        return None
    return converted.reindex(col.index).astype(float)


def _detect_dayfirst(txt: pd.Series) -> bool:
    """
    Угадывает порядок «день-месяц» в датах вида 12.03.2021 / 12/03/2021.
    Если первое число где-то > 12 -> день первый; если второе > 12 -> месяц первый;
    при неоднозначности считаем день первым для '.' и месяц первым для остальных разделителей.
    """
    parts = txt.str.extract(_DMY_RE).dropna()
    if parts.empty:
        return False
    first, second = parts[0].astype(int), parts[2].astype(int)
    if (first > 12).any():
        return True
    if (second > 12).any():
        return False
    return bool((parts[1] == ".").mean() > 0.5)


def _try_datetime(col: pd.Series) -> Optional[pd.Series]:
    """То же самое, что и для числа, но через pd.to_datetime(coerce)."""
    mask = col.notna()
    txt = col[mask].astype(str).str.strip()
    dayfirst = _detect_dayfirst(txt)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            converted = pd.to_datetime(txt, errors="coerce", format="mixed", dayfirst=dayfirst)
        except (TypeError, ValueError):  # pandas < 2.0 не знает format="mixed"
            converted = pd.to_datetime(txt, errors="coerce", dayfirst=dayfirst)

    if converted.isna().sum() != 0:
        return None
    return converted.reindex(col.index)


# --------------------------------------------------------------------------- #
# Шаг 4: основной пайп определения типов
# --------------------------------------------------------------------------- #

def _infer_column(col: pd.Series, category_threshold: float) -> tuple[pd.Series, str]:
    """Определяет тип одной колонки и возвращает (преобразованная колонка, тип)."""
    n_rows = len(col)

    # Уже типизированные колонки
    if pd.api.types.is_bool_dtype(col):
        return col.astype("category"), CATEGORY
    if pd.api.types.is_numeric_dtype(col):
        return col, NUMERIC if col.notna().any() else EMPTY
    if pd.api.types.is_datetime64_any_dtype(col):
        return col, DATETIME if col.notna().any() else EMPTY
    if isinstance(col.dtype, pd.CategoricalDtype):
        return col, CATEGORY if col.notna().any() else EMPTY

    # Строковые / object колонки
    col = _normalize_missing(col)
    if col.notna().sum() == 0:
        return col, EMPTY

    numeric = _try_numeric(col)
    if numeric is not None:
        return numeric, NUMERIC

    dates = _try_datetime(col)
    if dates is not None:
        return dates, DATETIME

    if col.nunique(dropna=True) <= category_threshold * n_rows:
        return col.astype("category"), CATEGORY
    return col, TEXT


def infer_types(
    df: pd.DataFrame,
    category_threshold: float = CATEGORY_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Определяет типы колонок и приводит данные к этим типам.

    Args:
        df: исходный датасет (не изменяется).
        category_threshold: колонка - категория, если nunique < threshold * n_rows.

    Returns:
        (преобразованный DataFrame, {колонка: тип}).
    """
    converted: dict[str, pd.Series] = {}
    column_types: dict[str, str] = {}
    for name in df.columns:
        converted[name], column_types[name] = _infer_column(df[name], category_threshold)
    return pd.DataFrame(converted, index=df.index), column_types
