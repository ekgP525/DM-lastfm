# Last.fm 음악 추천 — 랜덤워크 추천의 인기 편향 측정과 보정

SCE331 Data Mining (2026-2) Team 10

## Overview

- HetRec 2011 Last.fm 2K 청취 기록으로 사용자별 Top-10 아티스트 추천
- 행렬 기반(Item-based CF, MF)과 그래프 기반(RWR) 추천을 같은 split·같은 지표로 비교
- 주 질문: RWR이 인기 아티스트로 더 쏠리는가(H1), 인기도 보정 RWR이 정확도를 크게 잃지 않고 롱테일을 늘리는가(H2), MinHash-LSH 근사 이웃이 Item-CF의 정확도를 유지하는가(H3)

## Team

| 이름 | 학번 | 담당 |
| --- | --- | --- |
| 박채연 | | |
| 진주영 | | |
| 이다혜 | | |

## Repository layout

```
scripts/
  download_data.py     데이터 다운로드, 압축 해제, 접근일·sha256 기록
  data_stats.py        보고서 데이터 장용 통계 → results/data_stats.json
  run_experiment.py    방법 × seed 실행 → results/metrics_*.json, per_user_*.csv
src/lastfm_rec/
  data.py              로딩, log(1+plays), 사용자 필터, ID 인코딩, 친구·태그 인코딩
  split.py             사용자별 train/val/test 분할(seed 고정)
  graph.py             학습 데이터만으로 RWR 그래프 구성(G1/G2/G3), 전이 행렬
  metrics.py           Recall@K, NDCG@K, coverage, 평균 인기도, tail 비율, paired bootstrap
  models/
    base.py            Recommender 인터페이스(fit, score, recommend)
    baselines.py       인기도 순, 무작위
    stubs.py           ItemCF, MinHashLSH, MF, RWR (구현 예정)
tests/                 분할·leakage·지표 단위 테스트
data/raw/              원본 데이터(커밋하지 않음)
results/               실행 결과(커밋하지 않음)
notebooks/             EDA, 그림
```

## Environment / setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

- 개발 확인 환경: Python 3.13, numpy 2.5, scipy 1.18, pandas 3.0
- 최종 실험 환경(CPU, 메모리)은 results/metrics_*.json의 env 필드와 함께 여기에 기록

## Data sources and acquisition

- HetRec 2011 Last.fm 2K, GroupLens 배포 [1]
- 비상업적 용도 한정. 원본은 repo에 포함하지 않음

```bash
python scripts/download_data.py   # data/raw/hetrec2011-lastfm-2k/, data/raw/ACCESS_DATE.txt
```

## Preprocessing

- 청취 횟수 → log(1 + plays)
- 아티스트 5명 미만 사용자 제외
- 태그: 연도 2005~2011 범위 밖 행 제거, 평가 쌍(같은 사용자–같은 아티스트)에 달린 태그는 그래프에서 제외
- 필터 전후 규모: `python scripts/data_stats.py`

## Algorithms

| 알고리즘 | 강의 | 역할 |
| --- | --- | --- |
| Item-based CF | DM_09 Recom1 | 아티스트 코사인 유사도 이웃 가중합 |
| Regularized MF (bias, negative sampling) | DM_10 Recom2 | 잠재 요인 내적 + bias |
| Random Walk with Restart | DM_11 Link Analysis | 사용자 seed 이분 그래프 근접도 |
| MinHash + LSH | DM_04 LSH | Item-CF 이웃 후보 근사 |

- RWR은 teleport 집합이 사용자 한 명인 Personalized PageRank와 같은 식
- 인기도 보정(점수 ÷ (학습 청취자 수 + 1)^β)과 그래프 변형 G1~G3는 RWR의 변형·ablation이며 별도 알고리즘이 아님
- Baseline: 인기도 순, 무작위

## Experiment commands / configurations

```bash
# 하이퍼파라미터 선택: train으로 학습, validation으로 평가
python scripts/run_experiment.py --method popularity --stage val
python scripts/run_experiment.py --method rwr --stage val --params '{"restart": 0.3, "beta": 0.0}' --tag _c03

# 최종 평가: train+val로 학습, test로 평가 (확정한 값으로 한 번)
python scripts/run_experiment.py --method rwr --stage test --params '{"restart": 0.3, "beta": 0.5}'
```

- seed: 0, 1, 2, 3, 4 (분할과 모델 난수 모두)
- 분할: 사용자별 test 20%, 나머지 중 validation 10%

## Evaluation

- Recall@10, NDCG@10: 사용자 평균, seed 5개 평균 ± 표준편차
- 평가 범위 2종: 전체 평가 쌍(all), 학습 청취자 1명 이상인 아티스트만(reachable)
- Coverage, 평균 인기도(학습 청취자 수), tail 비율(head = 학습 청취 건수 누적 20% 상위 아티스트)
- 방법 간 차이: 사용자 단위 paired bootstrap 95% 신뢰구간 (`metrics.paired_bootstrap`, per_user_*.csv 입력)

## Reproducing tables / figures

- 표·그림 생성 순서는 구현 완료 후 기재

## References and external code

[1] I. Cantador, P. Brusilovsky, and T. Kuflik, "2nd Workshop on Information Heterogeneity and Fusion in Recommender Systems (HetRec 2011)," in *Proc. 5th ACM Conf. Recommender Systems (RecSys 2011)*, 2011. Dataset: https://grouplens.org/datasets/hetrec-2011/

- 외부 code 재사용 시 원본 링크와 license를 여기에 기재

## AI-use statement

- 2026-11-13, Anthropic Claude(Claude Opus 5.5)로 repo 초기 구조와 공통 코드(data.py, split.py, graph.py, metrics.py, models/base.py, models/baselines.py, scripts/, tests/)의 초안을 작성함
- 단위 테스트(합성 데이터)로 동작을 확인함. 실제 데이터 실행 결과 검증과 이후 수정 내역은 팀이 추가 기재
