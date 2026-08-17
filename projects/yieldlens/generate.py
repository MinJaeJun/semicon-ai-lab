"""YieldLens — 다단 공정 웨이퍼 수율 데이터셋 생성기.

24개 대표 스텝을 거치는 웨이퍼 12,000장(500 Lot x 24 Wafer)을 합성한다.

이 데이터셋이 일부러 갖고 있는 어려움
------------------------------------
1. 계측이 전수가 아니다. Response 약 22%, Defect 약 8% 만 값이 있다.
2. 그 계측이 무작위가 아니다. 위험해 보이는 웨이퍼를 골라 쟀기 때문에
   "계측된 웨이퍼만 보면 라인이 실제보다 나빠 보인다" (MNAR).
3. 진짜 원인 인자는 38개 Response 중 4개, 8개 Defect 중 1개뿐이다.
   나머지는 아무리 움직여도 수율과 무관하다.
4. 원인 인자는 단조가 아니라 U자다. 최적 구간이 있고 양쪽으로 벗어나면 나빠진다.
   그래서 단순 상관계수나 선형 모델로는 "어느 쪽으로 조정해야 하는지"를 못 낸다.

실행: python generate.py
출력: data/wafer_history.csv, data/wafer_history_holdout.csv, data/schema.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.synth import clip_to_spec, lot_bias, mnar_mask, rng_for, u_shape_response  # noqa: E402
from common.validate import (  # noqa: E402
    Report,
    check_measure_rate,
    check_mnar,
    check_no_duplicate_ids,
    check_relation,
    check_shape,
    check_sum_identity,
    check_u_shape,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

N_STEPS = 24
LOTS_TRAIN = 500
LOTS_HOLDOUT = 45
WAFERS_PER_LOT = 24

# 스텝별 장비. 실제 팹처럼 스텝마다 쓸 수 있는 장비가 정해져 있다.
TOOL_POOL = {
    "ETCH": ["ETCH-01", "ETCH-02", "ETCH-03"],
    "DEPO": ["CVD-01", "CVD-02", "PVD-01"],
    "LITHO": ["SCAN-01", "SCAN-02", "SCAN-03"],
    "CMP": ["CMP-01", "CMP-02", "CMP-03"],
    "DIFF": ["FURN-01", "FURN-02", "FURN-03"],
    "IMPL": ["IMP-01", "IMP-02", "IMP-03"],
    "CLEAN": ["WET-01", "WET-02", "WET-03"],
}
STEP_FAMILY = [
    "DIFF", "LITHO", "ETCH", "CLEAN", "DEPO", "CMP", "LITHO", "IMPL",
    "CLEAN", "DEPO", "ETCH", "CMP", "LITHO", "ETCH", "CLEAN", "DEPO",
    "IMPL", "DIFF", "LITHO", "ETCH", "CMP", "CLEAN", "DEPO", "ETCH",
]
assert len(STEP_FAMILY) == N_STEPS
CHAMBERS = ["CH1", "CH2", "CH3", "CH4"]

# 스텝별 Response 개수 (모든 스텝에 R1, 일부에 R2/R3)
R2_STEPS = [3, 5, 8, 10, 12, 14, 17, 19, 21, 23]
R3_STEPS = [5, 12, 19, 23]
D_STEPS = [2, 6, 9, 13, 16, 18, 20, 24]

FAIL_MODES = ["BRIDGE", "VOID", "PARTICLE", "SHORT", "OPEN"]

# 진짜 원인 인자. (컬럼, 영향 불량모드, 최적 중심, 곡률, 비대칭)
DRIVERS: list[tuple[str, str, float, float, float]] = [
    ("Step07_R1", "BRIDGE", 48.0, 0.0130, 0.22),
    ("Step14_R2", "VOID", 71.5, 0.0098, -0.18),
    ("Step20_R1", "SHORT", 34.0, 0.0155, 0.30),
    ("Step23_R3", "OPEN", 62.0, 0.0042, 0.10),
]
DEFECT_DRIVER = ("Step09_D1", "PARTICLE", 0.62)  # (컬럼, 불량모드, 선형 기울기)

# 신호 대 잡음비 조절. DRIVER_GAIN 을 낮추고 잡음을 키울수록 문제가 어려워진다.
# 현재 값은 유효 인자의 Spearman rho 가 0.20~0.45 에 들어오도록 맞춘 것이다.
DRIVER_GAIN = 0.60
NOISE_SHAPE = 2.0
NOISE_SCALE = 1.00

# 유효 인자의 관측 분포 중심을 최적점에서 이만큼(sigma 배수) 밀어둔다.
# 0 이면 완전 대칭 U가 되어 Spearman 상관이 0으로 죽는다 - 즉 "상관분석으로는
# 아무것도 안 보이는데 실제로는 원인인" 인자가 된다. 현업 데이터는 보통
# 공정이 최적점에서 한쪽으로 치우쳐 돌아가므로 약간의 오프셋이 있다.
CENTER_OFFSET_SIGMA = 0.5

# Lot 바이어스가 유효 인자 계측값을 최적점 밖으로 미는 세기.
# 이 값이 크면 "나쁜 Lot 은 인자도 같이 틀어져 있다" 는 실제 현상이 강해진다.
DRIFT_COUPLING = 0.60

R_MEASURE_RATE = 0.22
D_MEASURE_RATE = 0.08


def _response_columns() -> list[str]:
    cols = [f"Step{n:02d}_R1" for n in range(1, N_STEPS + 1)]
    cols += [f"Step{n:02d}_R2" for n in R2_STEPS]
    cols += [f"Step{n:02d}_R3" for n in R3_STEPS]
    return sorted(cols, key=lambda c: (int(c[4:6]), c[-2:]))


def _defect_columns() -> list[str]:
    return [f"Step{n:02d}_D1" for n in D_STEPS]


def _step_center(col: str, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """(center, sigma, lsl, usl) — 스텝마다 다른 계측 스케일을 갖게 한다."""
    for c, _, ctr, _, _ in DRIVERS:
        if c == col:
            sigma = 6.2
            return ctr + CENTER_OFFSET_SIGMA * sigma, sigma, ctr - 19.0, ctr + 21.0
    base = 20.0 + rng.random() * 60.0
    sigma = 3.0 + rng.random() * 2.6
    return base, sigma, base - 3.3 * sigma, base + 3.3 * sigma


def build(n_lots: int, lot_offset: int, tag: str) -> tuple[pd.DataFrame, dict]:
    rng = rng_for("yieldlens", tag)
    n = n_lots * WAFERS_PER_LOT

    lot_idx = np.repeat(np.arange(n_lots), WAFERS_PER_LOT)
    slot = np.tile(np.arange(1, WAFERS_PER_LOT + 1), n_lots)
    lot_id = np.array([f"L{lot_offset + i + 1:04d}" for i in lot_idx])
    wafer_id = np.array([f"{l}-{s:02d}" for l, s in zip(lot_id, slot)])

    lb = lot_bias(n_lots, rng)[lot_idx]
    # 특정 슬롯(캐리어 위치)이 구조적으로 불리하다 — 실제로 엣지 슬롯에서 흔하다
    slot_bias = np.where(np.isin(slot, [1, 2, 23, 24]), 0.26, 0.0)
    wafer_bias = lb + slot_bias + rng.normal(0, 0.14, n).clip(-0.35, None)

    frame: dict[str, np.ndarray] = {
        "Wafer_ID": wafer_id,
        "Lot_ID": lot_id,
        "Slot": slot,
    }

    # ---- 장비 배정: Model/Tool 은 Lot 단위 고정, Chamber 만 Wafer 단위 변동
    tool_effect: dict[str, float] = {}
    for fam, tools in TOOL_POOL.items():
        for i, t in enumerate(tools):
            tool_effect[t] = [0.0, 0.16, 0.34][i]
    for step in range(1, N_STEPS + 1):
        fam = STEP_FAMILY[step - 1]
        pool = TOOL_POOL[fam]
        lot_tool = rng.choice(pool, size=n_lots)
        tool = lot_tool[lot_idx]
        cham = rng.choice(CHAMBERS, size=n)
        frame[f"Step{step:02d}_Config"] = np.array(
            [f"S{step:02d}|{t}|{c}" for t, c in zip(tool, cham)]
        )
        wafer_bias = wafer_bias + np.array([tool_effect[t] for t in tool]) * 0.05

    # ---- Response / Defect 참값 생성 (계측 마스킹 전)
    rcols, dcols = _response_columns(), _defect_columns()
    truth: dict[str, np.ndarray] = {}
    spec: dict[str, tuple[float, float]] = {}
    driver_cols = {c for c, *_ in DRIVERS}
    for col in rcols:
        ctr, sig, lsl, usl = _step_center(col, rng)
        # Lot 이 나쁘면 계측값도 같이 흔들린다. 이게 교란(confounding)을 만든다.
        # 유효 인자는 "Lot 이 나빠지면 최적점에서 멀어지는" 방향으로 고정한다.
        # 무관 인자는 방향을 무작위로 줘서 가짜 상관이 섞이게 둔다.
        sign = 1.0 if col in driver_cols else float(rng.choice([-1.0, 1.0]))
        drift = wafer_bias * sig * (DRIFT_COUPLING if col in driver_cols else 0.35) * sign
        v = rng.normal(ctr, sig, n) + drift
        truth[col] = v
        spec[col] = (round(lsl, 2), round(usl, 2))
    for col in dcols:
        lam = np.clip(1.6 + wafer_bias * 2.2, 0.05, None)
        truth[col] = rng.poisson(lam).astype(float)

    # ---- 불량률
    #
    # 여기서 신호 대 잡음비를 일부러 낮게 잡는다. 원인 인자를 그대로 꽂으면
    # Spearman rho 가 0.9 를 넘어버리는데, 그런 데이터는 아무 모델이나 붙여도
    # 풀리기 때문에 프로젝트로서 의미가 없다. 실제 양산 데이터에서 단일 인자가
    # 갖는 설명력은 rho 0.2~0.4 수준이다. 그 수준에 맞춰 잡음을 얹는다.
    fr = {m: np.full(n, 0.0) for m in FAIL_MODES}
    for col, mode, ctr, curv, asym in DRIVERS:
        fr[mode] += u_shape_response(truth[col], ctr, 0.20, curv * DRIVER_GAIN, asym)
    dcol, dmode, dslope = DEFECT_DRIVER
    fr[dmode] += 0.42 + dslope * DRIVER_GAIN * truth[dcol]
    for m in FAIL_MODES:
        # 감마 잡음: 오른쪽 꼬리가 두꺼워 가끔 튀는 웨이퍼가 나온다
        fr[m] += 0.25 + 0.9 * wafer_bias + rng.gamma(NOISE_SHAPE, NOISE_SCALE, n)

    total = np.sum([fr[m] for m in FAIL_MODES], axis=0)
    # 총 불량률을 현실적인 범위로 눌러 넣는다 (양품률 하한 62%)
    scale = np.minimum(1.0, 38.0 / np.maximum(total, 1e-9))
    for m in FAIL_MODES:
        fr[m] = np.round(fr[m] * scale, 2)
    total = np.sum([fr[m] for m in FAIL_MODES], axis=0)
    yield_pct = np.round(100.0 - total, 2)
    # 반올림 잔차를 가장 큰 불량모드에 흡수시켜 합=100 을 정확히 맞춘다
    resid = np.round(100.0 - (yield_pct + total), 2)
    biggest = max(FAIL_MODES, key=lambda m: fr[m].mean())
    fr[biggest] = np.round(fr[biggest] - resid, 2)

    # ---- 계측 마스킹 (MNAR): 위험도가 높을수록 잴 확률이 올라간다
    risk = wafer_bias + (100.0 - yield_pct) / 25.0
    for col in rcols:
        mask = mnar_mask(n, R_MEASURE_RATE, risk, rng, risk_gain=2.2)
        v = clip_to_spec(truth[col], *spec[col], rng=rng, hard_ratio=0.85)
        out = np.where(mask, np.round(v, 2), np.nan)
        frame[col] = out
    for col in dcols:
        mask = mnar_mask(n, D_MEASURE_RATE, risk, rng, risk_gain=3.1)
        frame[col] = np.where(mask, truth[col], np.nan)

    frame["Yield_pct"] = yield_pct
    for m in FAIL_MODES:
        frame[f"FR_{m}"] = fr[m]
    # Fail bit count: 불량률과 느슨하게만 연결된다 (같은 불량률도 비트 수는 다르다)
    for m in FAIL_MODES:
        lam = np.maximum(fr[m], 0.02) * rng.uniform(12, 46) * rng.lognormal(0, 0.55, n)
        frame[f"FB_{m}"] = np.round(lam).astype(int)

    df = pd.DataFrame(frame)

    schema = {
        "dataset": "YieldLens wafer_history",
        "grain": "1행 = 1 Wafer (1 Lot = 24 Wafer, 1 Wafer = 1,000 Die 가정)",
        "steps": N_STEPS,
        "id_columns": ["Wafer_ID", "Lot_ID", "Slot"],
        "config_format": "S{step:02d}|{TOOL}|{CH} — Tool 은 Lot 단위 고정, Chamber 는 Wafer 단위 변동",
        "response_columns": rcols,
        "defect_columns": dcols,
        "targets": {
            "yield": "Yield_pct",
            "fail_rates": [f"FR_{m}" for m in FAIL_MODES],
            "fail_bits": [f"FB_{m}" for m in FAIL_MODES],
        },
        "identity": "Yield_pct + sum(FR_*) == 100.0",
        "measure_rate_target": {"response": R_MEASURE_RATE, "defect": D_MEASURE_RATE},
        "ground_truth_drivers": [
            {"factor": c, "target": f"FR_{m}", "shape": "u_shape", "optimum": ctr}
            for c, m, ctr, _, _ in DRIVERS
        ]
        + [{"factor": dcol, "target": f"FR_{dmode}", "shape": "monotonic_increasing"}],
        "spec_limits": {k: list(v) for k, v in spec.items()},
    }
    return df, schema


def validate(df: pd.DataFrame, rows: int) -> Report:
    rep = Report(f"yieldlens ({rows} rows)")
    rcols, dcols = _response_columns(), _defect_columns()
    check_shape(rep, df, rows, 3 + N_STEPS + len(rcols) + len(dcols) + 1 + 2 * len(FAIL_MODES))
    check_no_duplicate_ids(rep, df, "Wafer_ID")
    check_sum_identity(rep, df, ["Yield_pct"] + [f"FR_{m}" for m in FAIL_MODES], 100.0, tol=1e-6)
    check_measure_rate(rep, df, rcols, 18.0, 27.0)
    check_measure_rate(rep, df, dcols, 5.0, 12.0)
    check_mnar(rep, df, "Step09_D1", "Yield_pct", min_gap=0.8)
    for col, mode, *_ in DRIVERS:
        check_relation(rep, df, col, f"FR_{mode}", min_abs_rho=0.25, max_abs_rho=0.72)
        check_u_shape(rep, df, col, f"FR_{mode}", min_gain=0.015)
    check_relation(rep, df, DEFECT_DRIVER[0], f"FR_{DEFECT_DRIVER[1]}", min_abs_rho=0.35, max_abs_rho=0.78)
    return rep


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    DATA.mkdir(exist_ok=True)

    train, schema = build(LOTS_TRAIN, 0, "train")
    rep = validate(train, LOTS_TRAIN * WAFERS_PER_LOT)
    print(rep.render())
    rep.raise_if_failed()
    train.to_csv(DATA / "wafer_history.csv.gz", index=False, encoding="utf-8")

    hold, _ = build(LOTS_HOLDOUT, LOTS_TRAIN, "holdout")
    target_cols = ["Yield_pct"] + [f"FR_{m}" for m in FAIL_MODES] + [f"FB_{m}" for m in FAIL_MODES]
    hold_blind = hold.drop(columns=target_cols)
    hold_blind.to_csv(DATA / "wafer_history_holdout.csv", index=False, encoding="utf-8")
    hold[["Wafer_ID"] + target_cols].to_csv(DATA / "holdout_answers.csv", index=False, encoding="utf-8")

    schema["files"] = {
        "wafer_history.csv": list(train.shape),
        "wafer_history_holdout.csv": list(hold_blind.shape),
        "holdout_answers.csv": [len(hold), len(target_cols) + 1],
    }
    (DATA / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\ntrain   {train.shape}  -> data/wafer_history.csv")
    print(f"holdout {hold_blind.shape}  -> data/wafer_history_holdout.csv (타깃 제거)")


if __name__ == "__main__":
    main()
