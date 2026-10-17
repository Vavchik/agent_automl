"""
Схемы данных ML-ядра.

Содержит типизированные структуры для обмена данными между модулями:
- `DataProfileReport`: отчет профилировщика о типах, пропусках, распределениях и корреляциях.
- `ModelInfo`: описание алгоритма, класса библиотеки и инструмента тюнинга.
- `ModelSelectionDecision`: решение агента по черновой и боевой модели, лоссу и метрике.
- `SplitResult`: контейнер для train/val/test выборок и параметров разбиения.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


# --------------------------------------------------------------------------- #
# 1. Профиль датасета (Стадия 1)
# --------------------------------------------------------------------------- #

@dataclass
class DataProfileReport:
    """Отчёт о датасете: типы, пропуски, уникальные значения, распределения, head, корреляции."""

    n_rows: int
    column_types: dict[str, str]
    missing_share: dict[str, float]
    nunique: dict[str, int]
    distributions: dict[str, dict[str, Any]]
    head: list[dict[str, Any]]
    target: Optional[str] = None
    target_correlation: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-сериализуемый словарь."""
        return {
            "n_rows": self.n_rows,
            "column_types": self.column_types,
            "missing_share": self.missing_share,
            "nunique": self.nunique,
            "distributions": self.distributions,
            "head": self.head,
            "target": self.target,
            "target_correlation": self.target_correlation,
        }

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kwargs)

    def to_text(self) -> str:
        """Компактное текстовое представление для подстановки в промпт LLM."""
        lines = [f"## Общая информация\nКоличество строк (N): {self.n_rows}", "\n## Колонки"]
        for col, ctype in self.column_types.items():
            mark = " (target)" if col == self.target else ""
            lines.append(
                f"- {col}{mark}: type={ctype}, "
                f"missing={self.missing_share[col]:.1%}, nunique={self.nunique[col]} (N/U={self.n_rows/max(self.nunique[col], 1):.1f})"
            )
        lines.append("\n## Распределения")
        for col, dist in self.distributions.items():
            lines.append(f"- {col}: {json.dumps(dist, ensure_ascii=False)}")
        lines.append("\n## Первые строки")
        for row in self.head:
            lines.append(json.dumps(row, ensure_ascii=False))
        if self.target is not None:
            lines.append(f"\n## Корреляция с таргетом '{self.target}'")
            for col, info in self.target_correlation.items():
                value = info["value"]
                value_str = "n/a" if value is None else f"{value:.3f}"
                lines.append(f"- {col}: {value_str} ({info['method']})")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2. Выбор модели и метрики (Мета-анализ)
# --------------------------------------------------------------------------- #

@dataclass
class ModelInfo:
    """Информация об алгоритме (библиотека, класс, описание, инструмент тюнинга)."""

    algorithm: str
    library_class: str
    description: str = ""
    tuning_tool: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "algorithm": self.algorithm,
            "library_class": self.library_class,
            "description": self.description,
        }
        if self.tuning_tool:
            d["tuning_tool"] = self.tuning_tool
        return d


@dataclass
class ModelSelectionDecision:
    """Решение агента по выбору модели, функции потерь и целевой метрики."""

    task_type: str
    draft_model: ModelInfo
    battle_model: ModelInfo
    loss_function: str
    metric: str
    reasoning: str
    split_strategy: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "task_type": self.task_type,
            "draft_model": self.draft_model.to_dict(),
            "battle_model": self.battle_model.to_dict(),
            "loss_function": self.loss_function,
            "metric": self.metric,
            "reasoning": self.reasoning,
        }
        if self.split_strategy:
            res["split_strategy"] = self.split_strategy
        return res

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelSelectionDecision:
        task_type = data.get("task_type", "binary_classification")
        is_cls = "class" in str(task_type).lower()

        draft = data.get("draft_model", {})
        if isinstance(draft, str):
            draft = {"algorithm": draft}

        battle = data.get("battle_model", {})
        if isinstance(battle, str):
            battle = {"algorithm": battle}

        draft_algo = draft.get("algorithm", "LightGBM")
        draft_class = draft.get("library_class", "")
        if not draft_class:
            if "Ridge" in draft_algo:
                draft_class = "sklearn.linear_model.Ridge"
            elif "Logistic" in draft_algo:
                draft_class = "sklearn.linear_model.LogisticRegression"
            else:
                draft_class = "lightgbm.LGBMClassifier" if is_cls else "lightgbm.LGBMRegressor"

        battle_algo = battle.get("algorithm", "CatBoost")
        battle_class = battle.get("library_class", "")
        if not battle_class:
            if "Ridge" in battle_algo:
                battle_class = "sklearn.linear_model.Ridge"
            elif "Logistic" in battle_algo:
                battle_class = "sklearn.linear_model.LogisticRegression"
            elif "TabNet" in battle_algo:
                battle_class = "pytorch_tabnet.tab_model.TabNetClassifier" if is_cls else "pytorch_tabnet.tab_model.TabNetRegressor"
            else:
                battle_class = "catboost.CatBoostClassifier" if is_cls else "catboost.CatBoostRegressor"

        return cls(
            task_type=task_type,
            draft_model=ModelInfo(
                algorithm=draft_algo,
                library_class=draft_class,
                description=draft.get("description", ""),
            ),
            battle_model=ModelInfo(
                algorithm=battle_algo,
                library_class=battle_class,
                description=battle.get("description", ""),
                tuning_tool=battle.get("tuning_tool", "Optuna"),
            ),
            loss_function=data.get("loss_function", "logloss" if is_cls else "rmse"),
            metric=data.get("metric", "roc_auc" if is_cls else "rmse"),
            reasoning=data.get("reasoning", ""),
            split_strategy=data.get("split_strategy"),
        )


# --------------------------------------------------------------------------- #
# 3. Результат разбиения датасета (Стадия 4)
# --------------------------------------------------------------------------- #

@dataclass
class SplitResult:
    """Набор выборок после разбиения датасета."""

    train: pd.DataFrame
    test: pd.DataFrame
    val: Optional[pd.DataFrame] = None
    strategy: str = "stratified"
    target: str = ""
    time_col: Optional[str] = None
    group_col: Optional[str] = None

    def summary(self) -> dict[str, Any]:
        """Краткая сводка по размерам выборок."""
        return {
            "strategy": self.strategy,
            "target": self.target,
            "train_shape": list(self.train.shape),
            "test_shape": list(self.test.shape),
            "val_shape": list(self.val.shape) if self.val is not None else None,
        }


# --------------------------------------------------------------------------- #
# 4. Результат оценки модели (Стадия 6 / 8 / 10)
# --------------------------------------------------------------------------- #

@dataclass
class EvalResult:
    """Результат оценки модели для принятия решения об эскалации."""

    val_score: float          # Метрика на val
    train_score: float        # Метрика на train
    dummy_score: float        # Метрика наивного бейзлайна
    metric_name: str          # "roc_auc" / "macro_f1" / "rmse"
    threshold: float = 0.0    # Порог пользователя (--target-metric-threshold)

    @property
    def is_better_than_dummy(self) -> bool:
        """Модель значительно лучше наивного бейзлайна."""
        if self.metric_name in ("roc_auc", "macro_f1", "accuracy", "r2"):
            return self.val_score > self.dummy_score + 0.05
        return self.val_score < self.dummy_score * 0.95  # Для RMSE/MAE: меньше = лучше

    @property
    def is_overfit(self) -> bool:
        """Переобучение: разрыв train/val > 15%."""
        if self.metric_name in ("roc_auc", "macro_f1", "accuracy", "r2"):
            return (self.train_score - self.val_score) > 0.15
        return (self.val_score - self.train_score) / (abs(self.train_score) + 1e-9) > 0.15

    @property
    def tier1_passed(self) -> bool:
        """Tier 1 успешен: модель адекватна, без переобучения, удовлетворяет порогу."""
        if self.metric_name in ("roc_auc", "macro_f1", "accuracy", "r2"):
            threshold_ok = self.val_score >= self.threshold
        else:
            threshold_ok = (self.val_score <= self.threshold) if self.threshold > 0.0 else True

        return (
            self.is_better_than_dummy
            and not self.is_overfit
            and threshold_ok
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "val_score": self.val_score,
            "train_score": self.train_score,
            "dummy_score": self.dummy_score,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "is_better_than_dummy": self.is_better_than_dummy,
            "is_overfit": self.is_overfit,
            "tier1_passed": self.tier1_passed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalResult:
        return cls(
            val_score=float(data["val_score"]),
            train_score=float(data["train_score"]),
            dummy_score=float(data["dummy_score"]),
            metric_name=str(data["metric_name"]),
            threshold=float(data.get("threshold", 0.0)),
        )

