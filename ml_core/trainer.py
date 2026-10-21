"""
Модуль кросс-валидационного обучения (CrossValidatedTrainer).

Реализует:
1. 5-Fold кросс-валидацию для LightGBM, CatBoost и HistGradientBoosting.
2. Сбор Out-of-Fold (OOF) предсказаний и вероятностей.
3. Блендинг / усреднение предсказаний на тесте по 5 фолдам (Fold Averaging).
4. Расчет честной OOF-метрики (Macro F1, RMSE и др.).
5. Учет важности признаков (Feature Importances).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Optional, Sequence, Union
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from ml_core.evaluator import compute_metric, evaluate_oof

logger = logging.getLogger("ml_core.trainer")


@dataclass
class TrainingResult:
    """Результат кросс-валидационного обучения."""
    oof_score: float
    oof_preds: np.ndarray
    test_preds: np.ndarray
    metric_name: str
    oof_probs: Optional[np.ndarray] = None
    test_probs: Optional[np.ndarray] = None
    classes: Optional[np.ndarray] = None
    feature_importances: dict[str, float] = field(default_factory=dict)
    models: list[Any] = field(default_factory=list)


def _prepare_matrix(
    df: Any,
    cat_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Преобразует DataFrame или массив в числовую матрицу, кодируя категории для LightGBM/Sklearn.
    """
    if pd is not None and isinstance(df, pd.DataFrame):
        feature_names = list(df.columns)
        X_df = df.copy()
        for col in feature_names:
            if not pd.api.types.is_numeric_dtype(X_df[col]):
                # Кодируем любые строковые/категориальные колонки в числовые коды
                X_df[col] = X_df[col].astype("category").cat.codes.replace(-1, np.nan)
            else:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
        return X_df.to_numpy(dtype=float), feature_names

    arr = np.asarray(df, dtype=float)
    feature_names = [f"f_{i}" for i in range(arr.shape[1])] if arr.ndim == 2 else []
    return arr, feature_names


class _NumpyDummyEstimator:
    """Чистый NumPy fallback на случай отсутствия LightGBM, CatBoost и scikit-learn."""
    def __init__(self, is_classification: bool = False, random_state: int = 42):
        self.is_classification = is_classification
        self.random_state = random_state
        self.classes_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None
        self.intercept_: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        y = np.asarray(y)
        n_samples = len(X)
        if self.is_classification:
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            Y_onehot = np.zeros((n_samples, n_classes))
            for i, c in enumerate(self.classes_):
                Y_onehot[y == c, i] = 1.0
            X_ext = np.column_stack([np.ones(n_samples), X])
            alpha = 1.0
            A = X_ext.T @ X_ext + alpha * np.eye(X_ext.shape[1])
            try:
                W = np.linalg.solve(A, X_ext.T @ Y_onehot)
            except np.linalg.LinAlgError:
                W = np.linalg.pinv(A) @ (X_ext.T @ Y_onehot)
            self.intercept_ = W[0]
            self.weights_ = W[1:]
        else:
            y_float = np.asarray(y, dtype=float)
            X_ext = np.column_stack([np.ones(n_samples), X])
            alpha = 1.0
            A = X_ext.T @ X_ext + alpha * np.eye(X_ext.shape[1])
            try:
                w = np.linalg.solve(A, X_ext.T @ y_float)
            except np.linalg.LinAlgError:
                w = np.linalg.pinv(A) @ (X_ext.T @ y_float)
            self.intercept_ = float(w[0])
            self.weights_ = w[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        if self.is_classification:
            probs = self.predict_proba(X)
            idx = np.argmax(probs, axis=1)
            return self.classes_[idx]
        else:
            return self.intercept_ + X @ self.weights_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_classification:
            raise ValueError("predict_proba is only available for classification")
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        logits = self.intercept_ + X @ self.weights_
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probs

    @property
    def feature_importances_(self) -> np.ndarray:
        if self.weights_ is None:
            return np.zeros(0)
        if self.weights_.ndim == 2:
            return np.mean(np.abs(self.weights_), axis=1)
        return np.abs(self.weights_)


class CrossValidatedTrainer:
    """
    Универсальный тренер с 5-Fold кросс-валидацией и усреднением тестовых предсказаний.
    """

    def __init__(
        self,
        task_type: str = "multiclass_classification",
        metric: str = "macro_f1",
        algorithm: str = "LightGBM",
        random_state: int = 42,
    ):
        self.task_type = task_type
        self.metric = metric
        self.algorithm = algorithm
        self.random_state = random_state
        self.is_classification = "class" in task_type.lower()

    def _get_base_estimator(self, n_estimators: int = 150, max_depth: int = 6) -> Any:
        """Инстанцирует модель с учетом доступных библиотек в окружении."""
        algo = self.algorithm.lower()

        # 1. Попытка инициализировать LightGBM
        if "lightgbm" in algo or "lgbm" in algo:
            try:
                import lightgbm as lgb
                if self.is_classification:
                    return lgb.LGBMClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=self.random_state,
                        verbose=-1,
                        n_jobs=-1,
                    )
                else:
                    return lgb.LGBMRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=self.random_state,
                        verbose=-1,
                        n_jobs=-1,
                    )
            except ImportError:
                pass

        # 2. Попытка инициализировать CatBoost
        if "catboost" in algo:
            try:
                import catboost as cb
                if self.is_classification:
                    return cb.CatBoostClassifier(
                        iterations=n_estimators,
                        depth=min(max_depth, 8),
                        random_seed=self.random_state,
                        verbose=0,
                        thread_count=-1,
                    )
                else:
                    return cb.CatBoostRegressor(
                        iterations=n_estimators,
                        depth=min(max_depth, 8),
                        random_seed=self.random_state,
                        verbose=0,
                        thread_count=-1,
                    )
            except ImportError:
                pass

        # 3. Fallback на быстрый Scikit-learn HistGradientBoosting
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
            if self.is_classification:
                return HistGradientBoostingClassifier(
                    max_iter=n_estimators,
                    max_depth=max_depth,
                    random_state=self.random_state,
                )
            else:
                return HistGradientBoostingRegressor(
                    max_iter=n_estimators,
                    max_depth=max_depth,
                    random_state=self.random_state,
                )
        except ImportError:
            pass

        # 4. Fallback на линейные модели sklearn
        try:
            from sklearn.linear_model import LogisticRegression, Ridge
            if self.is_classification:
                return LogisticRegression(max_iter=500, random_state=self.random_state)
            return Ridge(alpha=1.0)
        except ImportError:
            pass

        # 5. Чистый NumPy fallback (когда scikit-learn / LightGBM не установлены)
        return _NumpyDummyEstimator(
            is_classification=self.is_classification,
            random_state=self.random_state,
        )

    def _get_estimators_for_blend(self, n_estimators: int = 150, max_depth: int = 6) -> list[Any]:
        """Возвращает список моделей для ансамблирования (LightGBM + CatBoost)."""
        estimators = []
        # 1. LightGBM
        try:
            import lightgbm as lgb
            if self.is_classification:
                estimators.append(lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=self.random_state,
                    verbose=-1,
                    n_jobs=-1,
                ))
            else:
                estimators.append(lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=self.random_state,
                    verbose=-1,
                    n_jobs=-1,
                ))
        except ImportError:
            pass

        # 2. CatBoost
        try:
            import catboost as cb
            if self.is_classification:
                estimators.append(cb.CatBoostClassifier(
                    iterations=n_estimators,
                    depth=min(max_depth, 8),
                    random_seed=self.random_state,
                    verbose=0,
                    thread_count=-1,
                ))
            else:
                estimators.append(cb.CatBoostRegressor(
                    iterations=n_estimators,
                    depth=min(max_depth, 8),
                    random_seed=self.random_state,
                    verbose=0,
                    thread_count=-1,
                ))
        except ImportError:
            pass

        if not estimators:
            estimators.append(self._get_base_estimator(n_estimators=n_estimators, max_depth=max_depth))
        return estimators

    def train_cv(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Optional[Any] = None,
        folds: Optional[list[tuple[Any, Any]]] = None,
        n_estimators: int = 150,
        max_depth: int = 6,
    ) -> TrainingResult:
        """
        Запускает 5-Fold кросс-валидацию:
        - обучает модель (или бленд моделей) на каждом фолде;
        - собирает OOF предсказания;
        - усредняет предсказания на тесте по всем фолдам.
        """
        # Преобразуем матрицы
        X_tr_mat, feat_names = _prepare_matrix(X_train)
        if pd is not None and isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_arr = np.asarray(y_train.to_numpy(dtype=object if self.is_classification else float))
        else:
            y_arr = np.asarray(y_train)

        n_samples = len(X_tr_mat)
        has_test = X_test is not None
        if has_test:
            X_te_mat, _ = _prepare_matrix(X_test)
            n_test = len(X_te_mat)
        else:
            X_te_mat = np.zeros((0, X_tr_mat.shape[1]))
            n_test = 0

        # Определение классов
        classes = np.unique(y_arr) if self.is_classification else None
        n_classes = len(classes) if classes is not None else 0

        # Если фолды не переданы, создаем KFold / StratifiedKFold
        if folds is None:
            from ml_core.splitter import get_cv_folds
            strategy = "stratified" if self.is_classification else "kfold"
            dummy_df = pd.DataFrame({"target": y_arr}) if pd is not None else y_arr
            folds = get_cv_folds(dummy_df, target="target" if pd is not None else "", strategy=strategy, n_splits=5)

        # Выделяем массивы под OOF и тест
        if self.is_classification:
            if n_classes > 2:
                oof_probs = np.zeros((n_samples, n_classes), dtype=float)
                test_probs_acc = np.zeros((n_test, n_classes), dtype=float) if has_test else None
            else:
                oof_probs = np.zeros((n_samples, 2), dtype=float)
                test_probs_acc = np.zeros((n_test, 2), dtype=float) if has_test else None
            oof_preds = np.empty(n_samples, dtype=object)
        else:
            oof_preds = np.zeros(n_samples, dtype=float)
            oof_probs = None
            test_preds_acc = np.zeros(n_test, dtype=float) if has_test else None

        models = []
        importances_acc = np.zeros(X_tr_mat.shape[1], dtype=float)
        is_blend = self.algorithm.lower() in ("blend", "ensemble")

        # Цикл по фолдам
        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            X_f_train, y_f_train = X_tr_mat[tr_idx], y_arr[tr_idx]
            X_f_val, y_f_val = X_tr_mat[val_idx], y_arr[val_idx]

            fold_estimators = (
                self._get_estimators_for_blend(n_estimators=n_estimators, max_depth=max_depth)
                if is_blend
                else [self._get_base_estimator(n_estimators=n_estimators, max_depth=max_depth)]
            )

            fold_val_preds_list = []
            fold_val_probs_list = []
            fold_test_preds_list = []
            fold_test_probs_list = []

            for model in fold_estimators:
                model.fit(X_f_train, y_f_train)
                models.append(model)

                if self.is_classification and hasattr(model, "predict_proba"):
                    probs_val = model.predict_proba(X_f_val)
                    fold_val_probs_list.append(probs_val)
                    if has_test:
                        fold_test_probs_list.append(model.predict_proba(X_te_mat))
                else:
                    preds_val = np.ravel(model.predict(X_f_val))
                    fold_val_preds_list.append(preds_val)
                    if has_test:
                        fold_test_preds_list.append(np.ravel(model.predict(X_te_mat)))

                if hasattr(model, "feature_importances_"):
                    importances_acc += np.asarray(model.feature_importances_, dtype=float) / (len(folds) * len(fold_estimators))

            # Агрегируем предсказания по моделям на фолде
            if self.is_classification:
                if fold_val_probs_list:
                    blend_val_probs = np.mean(fold_val_probs_list, axis=0)
                    oof_probs[val_idx] = blend_val_probs
                    oof_preds[val_idx] = np.array([classes[i] for i in np.argmax(blend_val_probs, axis=1)])
                if has_test and fold_test_probs_list:
                    blend_test_probs = np.mean(fold_test_probs_list, axis=0)
                    test_probs_acc += blend_test_probs / len(folds)
            else:
                blend_val_preds = np.mean(fold_val_preds_list, axis=0)
                oof_preds[val_idx] = blend_val_preds
                if has_test and fold_test_preds_list:
                    blend_test_preds = np.mean(fold_test_preds_list, axis=0)
                    test_preds_acc += blend_test_preds / len(folds)

        # Итоговая OOF метрика
        oof_score = evaluate_oof(
            y_true=y_arr,
            y_oof_pred=oof_preds,
            y_oof_proba=oof_probs if self.is_classification else None,
            metric=self.metric,
        )

        # Тестовые предсказания
        if has_test:
            if self.is_classification:
                test_preds = np.array([classes[i] for i in np.argmax(test_probs_acc, axis=1)])
                test_probs = test_probs_acc
            else:
                test_preds = test_preds_acc
                test_probs = None
        else:
            test_preds = np.zeros(0)
            test_probs = None

        feat_importances_dict = {}
        if feat_names and len(feat_names) == len(importances_acc):
            for name, imp in sorted(zip(feat_names, importances_acc), key=lambda x: -x[1]):
                feat_importances_dict[name] = float(imp)

        return TrainingResult(
            oof_score=oof_score,
            oof_preds=oof_preds,
            test_preds=test_preds,
            metric_name=self.metric,
            oof_probs=oof_probs,
            test_probs=test_probs,
            classes=classes,
            feature_importances=feat_importances_dict,
            models=models,
        )
