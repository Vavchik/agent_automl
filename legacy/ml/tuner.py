"""Р3. Подбор гиперпараметров LightGBM через Optuna в пределах бюджета времени.

Каждый trial — полный прогон по тем же фолдам, поэтому скор сравним с остальными экспериментами.
Что улучшать (Р3): подбор на подвыборке/меньшем числе фолдов, pruning, тюнинг CatBoost.
"""
from __future__ import annotations

import optuna

import config
from core.contracts import Dataset, Folds, ModelConfig, TaskSpec
from ml.trainer import train_cv
from runtime.logger import log
from task.metrics import METRICS


def tune_lgbm(ds: Dataset, spec: TaskSpec, folds: Folds, base_cfg: ModelConfig,
              time_budget_sec: float, max_trials: int = 40) -> dict | None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        params = {**base_cfg.params.get("lgbm", {}), **params}
        cfg = ModelConfig(models=["lgbm"], params={"lgbm": params}, target_transform=base_cfg.target_transform)
        return train_cv(ds, spec, folds, cfg, name=f"optuna#{trial.number}").score

    direction = "maximize" if METRICS[spec.metric].higher_is_better else "minimize"
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=config.SEED))
    study.optimize(objective, n_trials=max_trials, timeout=time_budget_sec, catch=(Exception,))
    if not study.trials or study.best_trial is None:
        return None
    log.info(f"Optuna: {len(study.trials)} trials, лучший {spec.metric} = {study.best_value:.5f}")
    return {**base_cfg.params.get("lgbm", {}), **study.best_params}
