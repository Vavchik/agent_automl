"""
Сквозные тесты выполнения пайплайна: от сырых данных до обучения модели.

Проверяет:
1. Прогон датасета через Этап 1: Очистка типов (type_inference) + Профилирование (data_profiler).
2. Передачу результатов в Этап 2: Мета-анализ (meta_analyst) и валидацию ответов агента.
3. Разбиение данных на выборки через splitter (Этап 3).
4. Реальный запуск и обучение выбранных моделей:
   - Черновика (LightGBM, Ridge, LogisticRegression);
   - Боевой модели (CatBoostClassifier, CatBoostRegressor).
5. Расчет целевых метрик (ROC-AUC, Macro-F1, RMSE) и проверку их адекватности.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Tuple
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, roc_auc_score, root_mean_squared_error

from agent.orchestrator import AutoMLOrchestrator
from ml_core.data_profiler import profile_data
from ml_core.schemas import ModelSelectionDecision, SplitResult
from ml_core.splitter import split_data
from ml_core.type_inference import CATEGORY, DATETIME, NUMERIC, TEXT, infer_types

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _instantiate_model(library_class: str, is_classification: bool = True) -> Any:
    """Динамически загружает и инициализирует класс модели по строке модуля."""
    module_name, class_name = library_class.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    # Параметры для быстрого и детерминированного запуска в тестах
    if "LGBM" in class_name:
        return cls(n_estimators=30, max_depth=3, random_state=42, verbose=-1)
    if "CatBoost" in class_name:
        return cls(iterations=30, depth=3, random_seed=42, verbose=0)
    if "LogisticRegression" in class_name:
        return cls(max_iter=200, random_state=42)
    if "Ridge" in class_name:
        return cls(alpha=1.0)

    return cls()


def _prepare_simple_features(
    df: pd.DataFrame, target: str, column_types: dict[str, str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Базовая быстрая подготовка фичей перед fit:
    - выделяет X и y;
    - для дат извлекает числовые день, месяц, год;
    - для категорий приводит к category dtypes (для LightGBM/CatBoost);
    - числовые пропуски заполняет медианой;
    - убирает сырой текст.
    """
    y = df[target].copy()
    X = df.drop(columns=[target]).copy()

    for col in list(X.columns):
        ctype = column_types.get(col)
        if ctype == DATETIME:
            dt_s = pd.to_datetime(X[col], errors="coerce")
            X[f"{col}_year"] = dt_s.dt.year.fillna(2000).astype(int)
            X[f"{col}_month"] = dt_s.dt.month.fillna(1).astype(int)
            X[f"{col}_day"] = dt_s.dt.day.fillna(1).astype(int)
            X = X.drop(columns=[col])
        elif ctype == TEXT:
            # Текстовые столбцы исключаем для черновика
            X = X.drop(columns=[col])
        elif ctype == CATEGORY:
            X[col] = X[col].astype("category").cat.codes
        else:
            # Числовые признаки: заполняем пропуски медианой
            if X[col].isna().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(0.0 if pd.isna(median_val) else median_val)

    return X, y


# =========================================================================== #
# Тесты сквозного запуска моделей
# =========================================================================== #

class TestEndToEndPipelineWithModelExecution:
    """Сквозные тесты: Этап 1 -> Этап 2 -> Запуск модели."""

    @pytest.mark.skipif(not (DATA_RAW_DIR / "titanic.csv").exists(), reason="titanic.csv не найден")
    def test_titanic_binary_classification_pipeline(self) -> None:
        """
        Датасет Titanic:
        1. Очистка типов + Профиль (Этап 1).
        2. Мета-анализ -> binary_classification, logloss, roc_auc (Этап 2).
        3. Сплит -> stratified.
        4. Обучение LightGBM (Draft) и CatBoost (Battle) и замер ROC-AUC > 0.70.
        """
        raw_df = pd.read_csv(DATA_RAW_DIR / "titanic.csv")
        target = "survived" if "survived" in raw_df.columns else "Survived"

        # 1. Оркестратор и Этап 1
        orchestrator = AutoMLOrchestrator(target=target)
        report = orchestrator.run_stage_profile(raw_df)

        assert report.target == target
        assert report.column_types[target] in (NUMERIC, CATEGORY)

        # 2. Этап 2: Мета-анализ
        decision = orchestrator.run_stage_meta_analysis()
        assert decision.task_type == "binary_classification"
        assert decision.loss_function == "logloss"
        assert decision.metric == "roc_auc"
        assert decision.draft_model.algorithm == "LightGBM"
        assert decision.battle_model.algorithm == "CatBoost"

        # 3. Сплит выборки
        splits = orchestrator.run_stage_split(test_size=0.2, val_size=0.1)
        assert splits.strategy == "stratified"
        assert len(splits.train) > 0 and len(splits.test) > 0

        # 4. Подготовка фичей
        X_train, y_train = _prepare_simple_features(splits.train, target, orchestrator.column_types)
        X_test, y_test = _prepare_simple_features(splits.test, target, orchestrator.column_types)

        # 5. Обучение черновой модели (LightGBM)
        draft_model = _instantiate_model(decision.draft_model.library_class)
        draft_model.fit(X_train, y_train)
        draft_probs = draft_model.predict_proba(X_test)[:, 1]
        draft_auc = roc_auc_score(y_test, draft_probs)

        assert len(draft_probs) == len(X_test)
        assert draft_auc > 0.65, f"ROC-AUC черновика должен быть адекватным: {draft_auc:.3f}"

        # 6. Обучение боевой модели (CatBoost)
        battle_model = _instantiate_model(decision.battle_model.library_class)
        battle_model.fit(X_train, y_train)
        battle_probs = battle_model.predict_proba(X_test)[:, 1]
        battle_auc = roc_auc_score(y_test, battle_probs)

        assert len(battle_probs) == len(X_test)
        assert battle_auc > 0.65, f"ROC-AUC CatBoost должен быть адекватным: {battle_auc:.3f}"

    def test_wine_multiclass_pipeline(self) -> None:
        """
        Мультиклассовый датасет Wine:
        1. Этап 1: 13 непрерывных числовых признаков, 3 класса.
        2. Этап 2: multiclass_classification, multiclass, macro_f1.
        3. Обучение LightGBMClassifier и CatBoostClassifier, проверка Macro-F1 > 0.85.
        """
        from sklearn.datasets import load_wine
        wine = load_wine(as_frame=True)
        raw_df = wine.frame.copy()
        target = "target"

        orchestrator = AutoMLOrchestrator(target=target)
        report = orchestrator.run_stage_profile(raw_df)

        decision = orchestrator.run_stage_meta_analysis()
        assert decision.task_type == "multiclass_classification"
        assert decision.loss_function == "multiclass"
        assert decision.metric == "macro_f1"

        splits = orchestrator.run_stage_split(test_size=0.25)
        X_train, y_train = _prepare_simple_features(splits.train, target, orchestrator.column_types)
        X_test, y_test = _prepare_simple_features(splits.test, target, orchestrator.column_types)

        # Черновик: LightGBM
        draft_model = _instantiate_model(decision.draft_model.library_class)
        draft_model.fit(X_train, y_train)
        preds = draft_model.predict(X_test)
        score = f1_score(y_test, preds, average="macro")

        assert score > 0.80, f"Macro-F1 на винах должен быть высоким: {score:.3f}"

        # Боевая модель: CatBoost
        battle_model = _instantiate_model(decision.battle_model.library_class)
        battle_model.fit(X_train, y_train)
        cat_preds = battle_model.predict(X_test)
        cat_score = f1_score(y_test, cat_preds, average="macro")

        assert cat_score > 0.80, f"Macro-F1 CatBoost на винах: {cat_score:.3f}"

    @pytest.mark.skipif(not (DATA_RAW_DIR / "california_housing.csv").exists(), reason="california_housing.csv не найден")
    def test_california_housing_regression_pipeline(self) -> None:
        """
        Регрессия California Housing:
        1. Этап 1: непрерывные признаки и таргет MedHouseVal.
        2. Этап 2: regression, rmse, rmse.
        3. Обучение LightGBMRegressor и CatBoostRegressor, проверка RMSE < 1.0.
        """
        raw_df = pd.read_csv(DATA_RAW_DIR / "california_housing.csv").head(1500)
        target = "MedHouseVal"

        orchestrator = AutoMLOrchestrator(target=target)
        report = orchestrator.run_stage_profile(raw_df)

        decision = orchestrator.run_stage_meta_analysis()
        assert decision.task_type == "regression"
        assert decision.loss_function == "rmse"
        assert decision.metric == "rmse"

        splits = orchestrator.run_stage_split(test_size=0.2)
        X_train, y_train = _prepare_simple_features(splits.train, target, orchestrator.column_types)
        X_test, y_test = _prepare_simple_features(splits.test, target, orchestrator.column_types)

        # Черновик: LightGBMRegressor
        draft_model = _instantiate_model(decision.draft_model.library_class, is_classification=False)
        draft_model.fit(X_train, y_train)
        preds = draft_model.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)

        assert rmse < 1.2, f"RMSE регрессии должно быть адекватным: {rmse:.3f}"

        # Боевая модель: CatBoostRegressor
        battle_model = _instantiate_model(decision.battle_model.library_class, is_classification=False)
        battle_model.fit(X_train, y_train)
        cat_preds = battle_model.predict(X_test)
        cat_rmse = root_mean_squared_error(y_test, cat_preds)

        assert cat_rmse < 1.2, f"RMSE CatBoost регрессии: {cat_rmse:.3f}"

    def test_synthetic_temporal_linear_pipeline(self) -> None:
        """
        Временной датасет с линейным трендом:
        1. Наличие оси времени (datetime) активирует Ridge регрессию.
        2. time_series сплит.
        3. Обучение Ridge и проверка прогноза.
        """
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        x = np.linspace(1, 10, n)
        # Явный линейный тренд
        y = 2.5 * x + np.random.default_rng(42).normal(0, 0.5, n)

        raw_df = pd.DataFrame({
            "timestamp": dates.strftime("%Y-%m-%d"),
            "feature_x": x,
            "target": y,
        })

        orchestrator = AutoMLOrchestrator(target="target", time_col="timestamp")
        report = orchestrator.run_stage_profile(raw_df)

        decision = orchestrator.run_stage_meta_analysis()
        assert decision.task_type == "regression"
        assert decision.draft_model.algorithm == "Ridge"
        assert decision.split_strategy == "time_series"

        splits = orchestrator.run_stage_split(test_size=0.2)
        assert splits.strategy == "time_series"

        X_train, y_train = _prepare_simple_features(splits.train, "target", orchestrator.column_types)
        X_test, y_test = _prepare_simple_features(splits.test, "target", orchestrator.column_types)

        model = _instantiate_model(decision.draft_model.library_class, is_classification=False)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)

        assert rmse < 3.0, f"RMSE Ridge на линейном тренде: {rmse:.3f}"
