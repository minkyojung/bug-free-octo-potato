from dataclasses import replace

from src.public_plan import ROUNDS
from src.screening import FINAL_V3, Config2


def test_rounds_match_preregistration():
    """#15 사전 등록: 1회차 = v3, 2회차 = KOSPI는 v3 + 최근 비중, 건강식품은 v2."""
    assert ROUNDS[1] == {"Target_1": FINAL_V3["Target_1"], "Target_2": FINAL_V3["Target_2"]}
    assert ROUNDS[2]["Target_1"] == replace(FINAL_V3["Target_1"], decay=True)
    assert ROUNDS[2]["Target_2"] == Config2(("weekday", "traffic", "weekday_dev"), clip=True)
    assert set(ROUNDS) == {1, 2}  # 3·4회차는 등록 전까지 없음
