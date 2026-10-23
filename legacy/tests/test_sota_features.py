import numpy as np
import pandas as pd
import pytest

from agent.llm import extract_code, extract_hypothesis, extract_thinking
from agent.solution_tree import SolutionTree
from core.contracts import Dataset, ExperimentResult, ModelConfig, TaskSpec
from ml.postprocess import fit_class_weights, honest_class_weights
from ml.preprocess import compute_oof_target_encoding
from task.dtypes import build_schema_json


def test_xml_extraction():
    text = """
    <thinking>
    We should create ratio of features a and b to capture density.
    </thinking>
    <hypothesis>
    Ratio a/b improves separation of minority classes.
    </hypothesis>
    <python_code>
    def transform_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df['ratio'] = train_df['a'] / (train_df['b'].abs() + 1e-6)
        test_df['ratio'] = test_df['a'] / (test_df['b'].abs() + 1e-6)
        return train_df, test_df
    </python_code>
    """
    code = extract_code(text)
    assert code is not None and "def transform_features" in code
    hypothesis = extract_hypothesis(text)
    assert "Ratio a/b improves" in hypothesis
    thinking = extract_thinking(text)
    assert "density" in thinking


def test_solution_tree_backtracking():
    ds0 = Dataset(
        X=pd.DataFrame({"a": [1, 2, 3]}),
        y=pd.Series([0, 1, 0]),
        X_test=pd.DataFrame({"a": [4, 5]}),
        test_ids=pd.Series([0, 1]),
        cat_cols=[],
    )
    cfg0 = ModelConfig(models=["lgbm"])
    res0 = ExperimentResult("baseline", score=0.80, fold_scores=[0.80], oof=np.array([0, 1, 0]),
                            test_pred=np.array([0, 1]), classes=None, seconds=1.0)

    tree = SolutionTree(ds0, cfg0, res0)
    assert tree.best_id == 0
    assert tree.get_exploration_node().id == 0

    # Узел 1: успешный (0.85 > 0.80)
    c1 = tree.add_candidate(0, 1, "features", "hyp1", ds0, cfg0)
    res1 = ExperimentResult("1:features", score=0.85, fold_scores=[0.85], oof=np.array([0, 1, 0]),
                            test_pred=np.array([0, 1]), classes=None, seconds=1.0)
    tree.record_result(c1.id, res1, accepted=True, reason="gain +0.05")
    assert tree.best_id == c1.id
    assert tree.get_exploration_node().id == c1.id

    # Узел 2: ветвимся от узла 1, но он неудачный (0.81 < 0.85)
    c2 = tree.add_candidate(c1.id, 2, "features", "hyp2", ds0, cfg0)
    res2 = ExperimentResult("2:features", score=0.81, fold_scores=[0.81], oof=np.array([0, 1, 0]),
                            test_pred=np.array([0, 1]), classes=None, seconds=1.0)
    tree.record_result(c2.id, res2, accepted=False, reason="degrade -0.04")
    # Backtracking: лучший узел по-прежнему c1!
    assert tree.best_id == c1.id
    assert tree.get_exploration_node().id == c1.id

    # Узел 3: падение при исполнении
    c3 = tree.add_candidate(c1.id, 3, "features", "hyp3", ds0, cfg0)
    tree.record_failure(c3.id, "KeyError: 'foo'")
    assert tree.best_id == c1.id
    assert tree.get_exploration_node().id == c1.id


def test_schema_json_generation():
    from pathlib import Path
    spec = TaskSpec(
        task_dir=Path("."),
        text="Sample task",
        train_path=Path("train.csv"),
        test_path=Path("test.csv"),
        sample_sub_path=Path("sample.csv"),
        dict_path=None,
        target_col="target",
        id_col="id",
        task_type="multiclass",
        metric="macro_f1",
        output="label",
        submission_columns=["id", "target"],
    )
    ds = Dataset(
        X=pd.DataFrame({"cat": ["A", "B", "A", "C"], "num": [10.0, 20.0, 15.0, 100.0]}),
        y=pd.Series(["Worms", "Normal", "Normal", "Generic"]),
        X_test=pd.DataFrame({"cat": ["A", "B"], "num": [12.0, 18.0]}),
        test_ids=pd.Series([0, 1]),
        cat_cols=["cat"],
    )
    schema_str = build_schema_json(ds, spec, importance={"num": 0.8, "cat": 0.2})
    assert '"task": "multiclass"' in schema_str
    assert '"metric": "macro_f1"' in schema_str
    assert "classes_distribution" in schema_str
    assert "top_features" in schema_str


def test_oof_target_encoding():
    X_tr = pd.DataFrame({"city": ["Msk", "Spb", "Msk", "Kzn", "Spb", "Msk"]})
    y_tr = np.array([1, 0, 1, 0, 1, 0])
    X_te = pd.DataFrame({"city": ["Msk", "Spb", "Nov"]})
    folds = [(np.array([0, 1, 2, 3]), np.array([4, 5])),
             (np.array([2, 3, 4, 5]), np.array([0, 1]))]
    enc_tr, enc_te = compute_oof_target_encoding(X_tr, y_tr, X_te, ["city"], folds, "binary")
    assert "city__target_enc" in enc_tr.columns
    assert len(enc_tr) == len(X_tr)
    assert len(enc_te) == len(X_te)
    assert not enc_tr.isna().any().any()
    assert not enc_te.isna().any().any()


def test_nelder_mead_class_weights():
    rng = np.random.RandomState(42)
    # Создаем несбалансированные вероятности
    y = np.array([0] * 80 + [1] * 15 + [2] * 5)
    proba = rng.dirichlet(alpha=[5, 2, 0.5], size=len(y))
    weights = fit_class_weights(proba, y, "macro_f1")
    assert len(weights) == 3
    assert np.all(weights > 0)
