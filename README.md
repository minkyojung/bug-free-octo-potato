# bug-free-octo-potato

데이터분석 미니프로젝트: KOSPI·건강식품 지수 D+3 상승 확률 예측. 과제 분석과 전체 계획은 [docs/PLAN.md](docs/PLAN.md)에 있다.

## 준비
1. 제공 파일을 배치한다 (git에 포함하지 않음).
   - `data/raw/`: `train.csv`, `test_public.csv`, `test_private.csv`, `sample_submission_public.csv`, `sample_submission_private.csv`, 변수 설명서 xlsx
   - `reference/`: 과제 PDF, `Mini-Project_Baseline.ipynb`
2. 환경을 만든다 (Python 3.11).
   ```bash
   uv venv .venv --python 3.11
   uv pip install --python .venv/bin/python -r requirements.txt
   ```

## 실행
```bash
.venv/bin/python -m src.baseline   # 베이스라인: 자체 채점 + 제출 파일 생성
.venv/bin/python -m pytest -q      # 테스트
```
- 채점 결과: `outputs/experiments/<run>_cv5.csv`, `<run>_holdout200.csv`
- 제출 파일: `outputs/submissions/<run>/public_<반_이름>_<회차>.csv`, `private_<반_이름>.csv`
  - 파일명 표기는 `src/submission.py`의 `SUBMITTER`에서 바꾼다.

## 구조
| 경로 | 역할 |
|---|---|
| `src/data.py` | 데이터 로딩, 무결성 검사 (Time_step 연속성, Public = Private 앞 100행 등) |
| `src/validation.py` | 자체 채점기: 시계열 5-fold(100일씩, gap 3일) + 마지막 200일 holdout |
| `src/submission.py` | 제출 파일 생성과 형식 검사 |
| `src/baseline.py` | 강사 베이스라인 재구현 (노트북 점수를 그대로 재현) |
| `tests/` | 분할 누수 방지, 제출 형식, 베이스라인 재현 테스트 |
