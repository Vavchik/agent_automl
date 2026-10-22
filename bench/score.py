"""Р5. Перевод метрик в баллы лидерборда (приложение 3 кейса).

Бейзлайн B — результат baseline.py организаторов на той же выборке.
Пример: python bench/score.py --rmse 40.1 --b-rmse 55.0 --f1 0.61 --b-f1 0.45
"""
import argparse


def s_reg(rmse: float, b_rmse: float) -> float:
    return 100 * max(0.0, 1 - rmse / b_rmse)


def s_cls(f1: float, b_f1: float) -> float:
    return 100 * max(0.0, (f1 - b_f1) / (1 - b_f1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rmse", type=float, required=True)
    p.add_argument("--b-rmse", type=float, required=True)
    p.add_argument("--f1", type=float, required=True)
    p.add_argument("--b-f1", type=float, required=True)
    a = p.parse_args()
    reg, cls = s_reg(a.rmse, a.b_rmse), s_cls(a.f1, a.b_f1)
    print(f"S_reg = {reg:.2f}, S_cls = {cls:.2f}, public score = {(reg + cls) / 2:.2f}")
