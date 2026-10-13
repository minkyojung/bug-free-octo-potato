import pytest

from src.baseline import notebook_split_auc


def test_reproduces_notebook_scores(data):
    """reference/Mini-Project_Baseline.ipynb 실행 결과 (Target_1 0.5099, Target_2 0.9422)."""
    train, _ = data
    auc = notebook_split_auc(train)
    assert auc["Target_1"] == pytest.approx(0.5099, abs=5e-5)
    assert auc["Target_2"] == pytest.approx(0.9422, abs=5e-5)
