"""
Тесты для модуля генерации признаков (agent/feature_engineer.py).
"""

import pytest
from agent.feature_engineer import (
    _extract_python_code,
    _extract_hypothesis_description,
    generate_heuristic_features,
    execute_feature_transform,
)


def test_extract_python_code():
    llm_resp = """
    Я предлагаю создать отношение между x1 и x2.
    ```python
    def transform_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ratio'] = df['x1'] / (df['x2'] + 1)
        return df
    ```
    Это улучшит качество.
    """
    code = _extract_python_code(llm_resp)
    assert "def transform_features" in code
    assert "df['ratio']" in code

    desc = _extract_hypothesis_description(llm_resp)
    assert "отношение" in desc


def test_heuristic_features_generation():
    top_f = [("feature_a", 10.5), ("feature_b", 8.2)]
    code, desc = generate_heuristic_features(top_f, iteration=1)
    assert "def transform_features" in code
    assert "feature_a" in code
    assert "feature_b" in code
    assert "Взаимодействие" in desc


def test_execute_feature_transform():
    pd = pytest.importorskip("pandas")
    df_train = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0], "target": [0, 1, 0]})
    df_test = pd.DataFrame({"x1": [7.0, 8.0], "x2": [9.0, 10.0]})

    code = """
def transform_features(df):
    df = df.copy()
    df['sum_x'] = df['x1'] + df['x2']
    return df
"""
    new_tr, new_te, err = execute_feature_transform(code, df_train, df_test)
    assert err is None
    assert "sum_x" in new_tr.columns
    assert "sum_x" in new_te.columns
    assert len(new_tr) == 3
    assert len(new_te) == 2


def test_execute_feature_transform_error():
    pd = pytest.importorskip("pandas")
    df_train = pd.DataFrame({"x1": [1.0, 2.0]})
    bad_code = """
def transform_features(df):
    raise RuntimeError("Intentional error")
"""
    new_tr, new_te, err = execute_feature_transform(bad_code, df_train)
    assert err is not None
    assert "RuntimeError" in err
