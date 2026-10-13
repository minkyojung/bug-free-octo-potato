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
.venv/bin/python -m src.experiments Target_1   # 피처 사다리 (Target_2도 동일)
.venv/bin/python -m src.experiments final      # features_v1 채점 + 제출 파일
.venv/bin/python -m src.backlog Target_1        # 후보 백로그 실행 (Target_2도 동일)
.venv/bin/python -m src.backlog final           # features_v2 채점 + 제출 파일
.venv/bin/python -m src.screening Target_1      # 독립 시험 → 합치기 → 최종 확인 (Target_2도 동일)
.venv/bin/python -m src.screening final         # features_v3 채점 + 제출 파일
.venv/bin/python -m src.public_plan round 1     # Public 1회차 파일 (2회차도 동일)
.venv/bin/python -m src.public_plan final 1 2   # 최종 Private: KOSPI 1회차 후보 + 건강식품 2회차 후보 (Public 결과로 선택)
.venv/bin/python -m src.modeling Target_1      # 모델 비교·조정·앙상블 (Target_2도 동일)
.venv/bin/python -m src.hypotheses Target_1    # 사전 등록 가설 검증 (Target_2도 동일)
.venv/bin/python -m pytest -q      # 테스트
(cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb)  # EDA 재실행
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
| `src/eda.py` | 단변량 AUC, 요일 내 AUC, 우연 기준선(순환 이동) |
| `src/plotting.py` | 보고서용 그림 스타일 (한글 폰트, 색상) |
| `src/features.py` | 과거 방향 파생변수 묶음 (미국 지수 변화, 트래픽, 요일 대비 이상도 등) |
| `src/models.py` | 측정 모델(로지스틱, LightGBM)과 순위 평균 |
| `src/experiments.py` | 피처 사다리 실험과 최종 채점·제출 |
| `src/modeling.py` | 모델 비교·조정·앙상블 실험 (덩어리 4) |
| `src/hypotheses.py` | 사전 등록 가설 사다리 (덩어리 4.5) |
| `src/augment.py`, `src/levers.py` | 형제 정답 증강·최근 구간 학습 실험 (덩어리 4.6) |
| `src/backlog.py` | 사전 등록 후보 실행과 최종 v2 채점·제출 (덩어리 4.7) |
| `src/screening.py` | 독립 시험·합치기·최종 확인과 최종 v3 채점·제출 (덩어리 4.8) |
| `src/public_plan.py` | Public 제출 계획: 회차별 파일과 최종 Private 조합 |
| `notebooks/01_eda.ipynb` | 데이터 이해 (EDA). 그림은 `outputs/figures/` |
| `tests/` | 분할·피처 누수 방지, 제출 형식, 베이스라인 재현, 통계 함수 테스트 |
| `docs/EXPERIMENTS.md` | 실험 일지 |
