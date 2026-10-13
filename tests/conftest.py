import pytest

from src.data import RAW_DIR


@pytest.fixture(scope="session")
def data():
    if not (RAW_DIR / "train.csv").exists():
        pytest.skip("data/raw/에 제공 데이터가 없습니다.")
    from src.data import load_data

    return load_data()
