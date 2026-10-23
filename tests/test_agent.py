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


def test_extract_code_from_xml_tags():
    from agent.llm import extract_tag
    text = "<hypothesis>Площадь на этаж</hypothesis>\n<code>\ndef transform_features(df):\n    return df\n</code>"
    assert extract_tag(text, "hypothesis") == "Площадь на этаж"
    assert extract_code(text).startswith("def transform_features")


def _node(i, parent, score, improving=True, cols=()):
    import numpy as np
    from agent.tree import Node
    from core.contracts import Dataset, ExperimentResult, ModelConfig
    X = pd.DataFrame({c: [1.0, 2.0] for c in ("a",) + tuple(cols)})
    ds = Dataset(X, pd.Series([1.0, 2.0]), X.copy(), pd.Series([1, 2]), [])
    res = ExperimentResult(f"n{i}", score, [score], np.zeros(2), np.zeros(2), None, 1.0)
    return Node(i, parent, "features", f"h{i}", res, ds, ModelConfig(), improving=improving, added_cols=list(cols))


def test_tree_backtracks_from_dead_ends_and_finds_merge():
    from agent.tree import SolutionTree
    tree = SolutionTree(_node(0, None, 10.0), "rmse")
    tree.add(_node(1, 0, 9.0, cols=("f1",)))
    tree.add(_node(2, 0, 9.5, cols=("f2",)))
    tree.add(_node(3, 1, 9.8, improving=False, cols=("f1", "f3")))
    tree.best_id = 1
    expandable = [n.id for n in tree.expandable()]
    assert 3 not in expandable                      # узел без прироста — тупик
    assert expandable[0] == 1                       # лучший узел первым
    assert [n.id for n in tree.merge_sources(tree.nodes[1])] == [2]   # признаки соседней ветки
