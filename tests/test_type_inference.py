"""
Набор тестов для модуля ml_core.type_inference.

Проверяет:
1. Нормализацию пропусков и строковых пустышек (_normalize_missing).
2. Очистку мусора в числовых строках (_clean_number_text).
3. Агрессивную проверку на числа (_try_numeric).
4. Проверку на дату и определение формата день/месяц (_try_datetime, _detect_dayfirst).
5. Разделение категорий и текста по порогу кардинальности (_infer_column).
6. Комплексный инференс датафрейма (infer_types).
7. Тесты на реальных бенчмарк-датасетах (Telco Churn, Titanic, Bike Sharing).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml_core.type_inference import (
    CATEGORY,
    DATETIME,
    EMPTY,
    NUMERIC,
    TEXT,
    _clean_number_text,
    _detect_dayfirst,
    _infer_column,
    _normalize_missing,
    _try_datetime,
    _try_numeric,
    infer_types,
)

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# =========================================================================== #
# 1. Тесты нормализации пропусков
# =========================================================================== #

class TestMissingValuesNormalization:
    """Тестирование превращения псевдо-пустых строковых токенов в NaN."""

    def test_normalize_empty_and_na_tokens(self) -> None:
        raw_series = pd.Series(["", "   ", "nan", "NaN", "None", "null", "NULL", "n/a", "N/A", "#n/a", "-", "—"])
        normalized = _normalize_missing(raw_series)
        assert normalized.isna().all(), "Все псевдо-пустые токены должны стать NaN"

    def test_normalize_preserves_valid_strings(self) -> None:
        valid_strings = pd.Series(["apple", "none_of_the_above", "n/a_code", "positive", "valid text", "0"])
        normalized = _normalize_missing(valid_strings)
        assert normalized.notna().all(), "Обычные строки не должны превращаться в NaN"
        assert list(normalized) == list(valid_strings)

    def test_normalize_preserves_existing_types(self) -> None:
        numeric_series = pd.Series([1, 2, np.nan, 4])
        normalized = _normalize_missing(numeric_series)
        assert normalized.iloc[2] is np.nan or pd.isna(normalized.iloc[2])
        assert normalized.iloc[0] == 1


# =========================================================================== #
# 2. Тесты очистки чисел от мусора
# =========================================================================== #

class TestCleanNumberText:
    """Тестирование очистки строк от знаков валют, пробелов и запятых."""

    def test_currency_and_percent_cleaning(self) -> None:
        s = pd.Series(["$100", "€250.50", "300 ₽", "450 руб.", "500 usd", "15%"])
        cleaned = _clean_number_text(s)
        assert list(cleaned) == ["100", "250.50", "300", "450", "500", "15"]

    def test_spaces_removal(self) -> None:
        # Обычные пробелы, неразрывные (\u00a0) и тонкие (\u2009)
        s = pd.Series(["1 000 000", "1\u00a0250", "3\u2009500.75"])
        cleaned = _clean_number_text(s)
        assert list(cleaned) == ["1000000", "1250", "3500.75"]

    def test_comma_replacement(self) -> None:
        # Запятая как десятичный разделитель
        s_decimal = pd.Series(["3,14", "0,001", "-12,5"])
        assert list(_clean_number_text(s_decimal)) == ["3.14", "0.001", "-12.5"]

        # Запятая как разделитель тысяч (US формат: 1,234,567.89)
        s_thousands = pd.Series(["1,234,567", "1,234,567.89", "-1,000.50"])
        assert list(_clean_number_text(s_thousands)) == ["1234567", "1234567.89", "-1000.50"]

    def test_unicode_minus(self) -> None:
        s = pd.Series(["\u221242.5", "\u2212 100 ₽"])
        cleaned = _clean_number_text(s)
        assert list(cleaned) == ["-42.5", "-100"]


# =========================================================================== #
# 3. Тесты проверки на число (_try_numeric)
# =========================================================================== #

class TestTryNumeric:
    """Тестирование функции агрессивного приведения к числам."""

    def test_clean_numbers(self) -> None:
        s = pd.Series(["10", "20.5", "-30", "0"])
        result = _try_numeric(s)
        assert result is not None
        assert np.allclose(result, [10.0, 20.5, -30.0, 0.0])

    def test_dirty_numbers_with_currency_and_spaces(self) -> None:
        s = pd.Series(["100 000 ₽", "  $ 250,50 ", " 15% ", "-50,0 руб."])
        result = _try_numeric(s)
        assert result is not None
        assert np.allclose(result, [100000.0, 250.5, 15.0, -50.0])

    def test_numeric_with_valid_nans(self) -> None:
        s = pd.Series(["10", np.nan, "20", None, "30.5"])
        result = _try_numeric(s)
        assert result is not None
        assert result.isna().sum() == 2
        assert result.iloc[0] == 10.0
        assert result.iloc[2] == 20.0
        assert result.iloc[4] == 30.5

    def test_reject_mixed_or_text_column(self) -> None:
        # Колонка, где есть хотя бы одно нечисловое значение, не должна становиться числовой
        s = pd.Series(["10", "20", "invalid_text", "40"])
        assert _try_numeric(s) is None

        s_pure_text = pd.Series(["apple", "banana", "cherry"])
        assert _try_numeric(s_pure_text) is None


# =========================================================================== #
# 4. Тесты проверки на дату (_try_datetime, _detect_dayfirst)
# =========================================================================== #

class TestTryDatetime:
    """Тестирование функции распознавания дат."""

    def test_iso_dates(self) -> None:
        s = pd.Series(["2023-01-01", "2023-05-15", "2023-12-31"])
        result = _try_datetime(s)
        assert result is not None
        assert result.dt.year.tolist() == [2023, 2023, 2023]
        assert result.dt.month.tolist() == [1, 5, 12]
        assert result.dt.day.tolist() == [1, 15, 31]

    def test_russian_dmy_format_with_dots(self) -> None:
        s = pd.Series(["01.02.2023", "15.03.2023", "28.04.2023"])
        assert _detect_dayfirst(s) is True
        result = _try_datetime(s)
        assert result is not None
        assert result.dt.day.tolist() == [1, 15, 28]
        assert result.dt.month.tolist() == [2, 3, 4]

    def test_detect_dayfirst_disambiguation(self) -> None:
        # Первое число > 12 -> точно день первый
        s_day_first = pd.Series(["15/01/2021", "25/02/2021"])
        assert _detect_dayfirst(s_day_first) is True

        # Второе число > 12 -> точно месяц первый (US format)
        s_month_first = pd.Series(["01/15/2021", "02/25/2021"])
        assert _detect_dayfirst(s_month_first) is False

    def test_datetime_with_nans(self) -> None:
        s = pd.Series(["2023-01-01", np.nan, "2023-01-03"])
        result = _try_datetime(s)
        assert result is not None
        assert result.isna().sum() == 1
        assert result.iloc[0].year == 2023

    def test_reject_non_dates(self) -> None:
        s = pd.Series(["2023-01-01", "not_a_date", "2023-01-03"])
        assert _try_datetime(s) is None


# =========================================================================== #
# 5. Тесты классификации колонок (_infer_column, infer_types)
# =========================================================================== #

class TestColumnInference:
    """Тестирование разграничения Category / Text / Numeric / Datetime."""

    def test_boolean_is_category(self) -> None:
        col = pd.Series([True, False, True, True])
        _, ctype = _infer_column(col, category_threshold=0.1)
        assert ctype == CATEGORY

    def test_empty_column(self) -> None:
        col = pd.Series([None, np.nan, "", "nan", "-"])
        _, ctype = _infer_column(col, category_threshold=0.1)
        assert ctype == EMPTY

    def test_category_threshold_logic(self) -> None:
        # Датасет из 100 строк. Порог 0.1 (до 10 уникальных = категория).
        # 1. Колонка с 3 уникальными значениями -> CATEGORY
        cat_col = pd.Series(["A", "B", "C"] * 33 + ["A"])
        _, ctype = _infer_column(cat_col, category_threshold=0.1)
        assert ctype == CATEGORY

        # 2. Колонка с 50 уникальными значениями -> TEXT
        text_col = pd.Series([f"user_{i}" for i in range(50)] * 2)
        _, ctype = _infer_column(text_col, category_threshold=0.1)
        assert ctype == TEXT

    def test_custom_category_threshold(self) -> None:
        # 15 уникальных на 100 строк (15%)
        col = pd.Series([f"val_{i % 15}" for i in range(100)])
        # При пороге 0.1 (10%) -> TEXT
        _, type_default = _infer_column(col, category_threshold=0.1)
        assert type_default == TEXT
        # При повышенном пороге 0.2 (20%) -> CATEGORY
        _, type_relaxed = _infer_column(col, category_threshold=0.2)
        assert type_relaxed == CATEGORY

    def test_infer_types_dataframe_end_to_end(self) -> None:
        df = pd.DataFrame({
            "num_dirty": ["1 000 ₽", "2 500 ₽", "3 000 ₽", "4 200 ₽"],
            "date_str": ["01.01.2023", "02.01.2023", "03.01.2023", "04.01.2023"],
            "category_col": ["active", "active", "churn", "churn"],
            "text_col": ["comment A", "comment B", "comment C", "comment D"],
            "empty_col": ["", " - ", "None", np.nan],
        })

        clean_df, types = infer_types(df, category_threshold=0.5)

        assert types["num_dirty"] == NUMERIC
        assert types["date_str"] == DATETIME
        assert types["category_col"] == CATEGORY
        assert types["text_col"] == TEXT
        assert types["empty_col"] == EMPTY

        # Проверка типов в самом датафрейме
        assert pd.api.types.is_numeric_dtype(clean_df["num_dirty"])
        assert pd.api.types.is_datetime64_any_dtype(clean_df["date_str"])
        assert clean_df["num_dirty"].iloc[0] == 1000.0

    def test_original_df_is_not_mutated(self) -> None:
        original = pd.DataFrame({"col": [" 100 ₽ ", " 200 ₽ "]})
        original_copy = original.copy(deep=True)
        infer_types(original)
        pd.testing.assert_frame_equal(original, original_copy)


# =========================================================================== #
# 6. Тесты на реальных датасетах из data/raw/
# =========================================================================== #

class TestRealDatasetsInference:
    """Проверка инференса типов на реальных датасетах из data/raw/."""

    @pytest.mark.skipif(not (DATA_RAW_DIR / "telco_churn.csv").exists(), reason="telco_churn.csv не найден")
    def test_telco_churn_dirty_numeric_column(self) -> None:
        """
        В Telco Customer Churn колонка TotalCharges содержит строковые пробелы ' '.
        type_inference обязан очистить их и определить как NUMERIC, а не TEXT.
        """
        df = pd.read_csv(DATA_RAW_DIR / "telco_churn.csv")
        clean_df, types = infer_types(df)

        assert types["TotalCharges"] == NUMERIC, (
            f"TotalCharges должен определяться как NUMERIC, а не {types['TotalCharges']}"
        )
        assert pd.api.types.is_numeric_dtype(clean_df["TotalCharges"])
        # Проверяем, что MonthlyCharges и tenure тоже числовые
        assert types["MonthlyCharges"] == NUMERIC
        assert types["tenure"] == NUMERIC
        # Категории
        assert types["Churn"] == CATEGORY
        assert types["Contract"] == CATEGORY

    @pytest.mark.skipif(not (DATA_RAW_DIR / "titanic.csv").exists(), reason="titanic.csv не найден")
    def test_titanic_dataset_types(self) -> None:
        df = pd.read_csv(DATA_RAW_DIR / "titanic.csv")
        clean_df, types = infer_types(df)

        assert types["age"] == NUMERIC or types.get("Age") == NUMERIC
        assert types["fare"] == NUMERIC or types.get("Fare") == NUMERIC
        assert types["sex"] == CATEGORY or types.get("Sex") == CATEGORY
        # Имя пассажира высококардинальное -> text
        name_col = "name" if "name" in types else "Name"
        assert types[name_col] == TEXT

    @pytest.mark.skipif(not (DATA_RAW_DIR / "bike_sharing.csv").exists(), reason="bike_sharing.csv не найден")
    def test_bike_sharing_datetime_column(self) -> None:
        df = pd.read_csv(DATA_RAW_DIR / "bike_sharing.csv")
        clean_df, types = infer_types(df)

        date_col = "dteday" if "dteday" in types else "datetime"
        if date_col in types:
            assert types[date_col] == DATETIME
            assert pd.api.types.is_datetime64_any_dtype(clean_df[date_col])
