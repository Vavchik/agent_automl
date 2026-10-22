"""Р4: песочница и LLM без ключей."""
import pandas as pd

from agent.llm import LLM, extract_code
from agent.sandbox import run_feature_code

DF = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 0.0, 1.0]})


def test_sandbox_adds_columns():
    code = "def transform_features(df):\n    df['r'] = df['a'] / (df['b'].abs() + 1e-6)\n    return df\n"
    out, err = run_feature_code(code, DF)
    assert err is None and "r" in out.columns


def test_sandbox_reports_errors():
    out, err = run_feature_code("def transform_features(df):\n    return df['nope']\n", DF)
    assert out is None and "nope" in err


def test_sandbox_timeout():
    code = "import time\ndef transform_features(df):\n    time.sleep(10)\n    return df\n"
    out, err = run_feature_code(code, DF, timeout=1)
    assert out is None and "Таймаут" in err


def test_llm_disabled_without_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert LLM().ask("s", "u") is None


def test_extract_code():
    text = "Гипотеза.\n```python\ndef transform_features(df):\n    return df\n```"
    assert extract_code(text).startswith("def transform_features")
