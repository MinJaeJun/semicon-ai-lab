# 데이터 출처와 합성 방침

## 한 줄 요약

이 저장소의 모든 데이터는 **저장소 안의 코드가 생성한 합성 데이터**다.
실제 기업 데이터, 타인이 만든 데이터셋, 외부에서 받은 파일을 포함하지 않는다.

## 무엇을 참고했나

데이터의 **구조**는 공개된 반도체 공정 지식과 기술 문헌을 참고했다. 구체적으로는

- 300mm 웨이퍼의 계측 사이트 배치 관행 (중심 1점 + 동심원상 다점, 존 구분)
- 신뢰성 시험 항목의 표준 구성 (HTOL, TDDB, NBTI, HCI, EM, ESD, TC, LTOL)
- FinFET 소자 파라미터의 일반적 항목 (Fin Width/Height, Gate Length, Gate Ox, Vt, Idsat, SS)
- FDC(Fault Detection & Classification) 가 다루는 설비 신호의 통상적 범주
  (레이저 파워/주파수, 냉각 유량, 세정 압력, 진동 등)
- 플래시 메모리 마모 지표의 표준 항목 (P/E Cycle, TBW, Bad Block, UECC)

이는 특정 회사의 자료가 아니라 해당 분야에서 널리 통용되는 도메인 상식이다.

## 무엇을 참고하지 않았나

**수치는 어느 것도 실측값을 옮겨오지 않았다.** 모든 값은 `common/synth.py` 의
분포 함수와 각 프로젝트 `generate.py` 의 파라미터에서 생성된다.

스펙 한계, 목표값, 분포 중심, 표준편차, 상관 구조, 열화 시나리오 — 전부 이 저장소에서
정의한 값이다. 어떤 값이 어디서 왔는지는 각 `generate.py` 상단 주석과 상수 정의에
그대로 적혀 있다. 외부 파일을 읽어들이는 코드는 없다.

## 왜 합성 데이터인가

실제 공정 데이터는 공개될 수 없다. 그렇다고 지나치게 단순한 공개 데이터를 쓰면
정작 풀고 싶은 문제(계측 편향, 교란, 비단조 응답, 희소 불량)가 사라진다.

합성 데이터에는 실제 데이터에 없는 장점도 하나 있다. **정답을 안다는 것.**

- YieldLens 는 진짜 원인 인자가 5개라는 걸 알고 있으므로, 검출 알고리즘의
  precision/recall 을 정확히 잴 수 있다.
- DiceGuard 는 열화를 3건 주입했으므로, 추세 판정 구간을 며칠로 잡아야 하는지를
  추측이 아니라 실측으로 답할 수 있다.
- RelyLab 은 문제 설비가 2대라는 걸 알고 있으므로, 귀인 절차의 오탐률을 낼 수 있다.

실제 데이터로는 이런 평가를 할 수 없다. 정답이 없기 때문이다.

## 재현성

모든 생성기는 프로젝트 이름에서 결정적으로 시드를 만든다 (`common/synth.py: rng_for`).
같은 코드를 돌리면 같은 데이터가 나온다. 데이터 파일을 지우고 `python build_all.py --data`
를 돌리면 바이트 단위로 동일하게 복원된다.

## 데이터 규모

| 프로젝트 | 파일 | 행 | 열 |
|---|---|---:|---:|
| yieldlens | wafer_history.csv.gz | 12,000 | 84 |
| yieldlens | wafer_history_holdout.csv | 1,080 | 73 |
| etchpilot | {poly,contact,via,pad}_sites.csv | 24,360 | 21 |
| etchpilot | {...}_recipe_master.csv | 48 | 8~23 |
| diceguard | dicing_fdc.csv.gz | 60,000 | 38 |
| cellhealth | device_health.csv.gz | 200,000 | 18 |
| relylab | wafer_params.csv | 1,200 | 34 |
| relylab | reliability_tests.csv | 9,600 | 8 |
| relylab | tool_history.csv | 1,200 | 10 |

합계 약 30만 행. 압축 후 약 15MB.

## 라이선스

데이터와 코드 모두 MIT. 자유롭게 쓰되, 이 데이터는 합성이므로 실제 공정 판단의
근거로 쓰면 안 된다.
