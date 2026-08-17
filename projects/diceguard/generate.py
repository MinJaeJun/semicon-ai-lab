"""DiceGuard — 레이저 다이싱 공정 FDC/불량 데이터셋 생성기.

장비 6대(LD-01~LD-06)가 120일간 스트립 60,000개를 가공한 이력을 만든다.

이 데이터셋이 노리는 문제
------------------------
"스펙 안에 있는데 서서히 나빠지는 중" 을 잡아내는 것이다.

그래서 일부러 이렇게 설계했다.

* 열화는 스펙을 넘지 않는다. LD-03 의 세정 유량과 LD-05 의 레이저 파워 효율은
  120일에 걸쳐 천천히 떨어지지만 마지막 날까지도 관리 한계 안에 있다.
  단발성 스펙 위반 감지로는 절대 안 잡힌다.
* 짧은 구간만 보면 안 보인다. 일별 변동이 커서 최근 30일만 잘라 회귀하면
  기울기가 유의하지 않게 나온다. 전 구간을 봐야 보인다.
* 원인과 결과가 강하게 얽혀 있다. 레이저 파워를 내리면 커프 폭이 좁아진다.
  단순 상관으로 원인을 찾으면 "커프 폭을 조치하라" 는 실행 불가능한 답이 나온다.
  커프 폭은 사람이 돌릴 수 있는 손잡이가 아니다.
* 치명 불량은 희소하다. MicroCrack 은 전체의 0.3% 수준이다.

실행: python generate.py
출력: data/dicing_fdc.csv, data/equipment_meta.csv, data/schema.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.synth import rng_for  # noqa: E402
from common.validate import Report, check_categories, check_no_duplicate_ids, check_shape  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

N_ROWS = 60_000
N_DAYS = 120
START = pd.Timestamp("2026-01-06")
MACHINES = ["LD-01", "LD-02", "LD-03", "LD-04", "LD-05", "LD-06"]
PRODUCTS = ["PKG_A", "PKG_B", "PKG_C", "PKG_D", "PKG_E"]
SHIFTS = ["DAY", "SWING", "NIGHT"]
RECIPES = [f"RCP_{i}" for i in range(1, 10)]
OPERATORS = [f"OP_{i}" for i in range(1, 29)]

# 사람이 돌릴 수 있는 설비 설정값 = 원인 후보
ACTIONABLE = {
    "Laser_Power_W": (18.50, 0.055),
    "Laser_Frequency_kHz": (100.0, 0.62),
    "Power_Efficiency_pct": (94.90, 0.075),
    "Beam_Center_Offset_um": (0.00, 0.115),
    "Feed_Speed_mm_s": (250.0, 0.72),
    "Focus_Offset_um": (0.00, 0.205),
    "Head_Temp_C": (28.40, 0.62),
    "Coolant_Flow_lpm": (12.60, 0.235),
    "Clean_Flow_lpm": (8.40, 0.185),
    "Clean_Pressure_bar": (2.35, 0.062),
    "Clean_Time_s": (14.0, 0.42),
    "Vibration_mm_s": (0.85, 0.085),
}
# 가공 결과로 측정되는 값 = 감시지표. 조치 대상이 아니다.
MONITORING = ["Kerf_Width_um", "Groove_Depth_um", "Coat_Thickness_um", "Coat_Roughness_nm"]

DEFECTS = ["Chipping", "MicroCrack", "Particle", "CoatResidue", "EdgeBurn"]
NG_CODE = {"Chipping": "CHIP", "MicroCrack": "CRACK", "Particle": "PART", "CoatResidue": "COAT", "EdgeBurn": "BURN"}

# 주입할 열화 시나리오: (장비, 변수, 120일 누적 변화량(시그마 배수), 시작 시점 비율)
DEGRADATION = [
    ("LD-03", "Clean_Flow_lpm", -2.4, 0.15),
    ("LD-05", "Power_Efficiency_pct", -2.5, 0.30),
    ("LD-02", "Vibration_mm_s", +2.3, 0.45),
]


def build() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = rng_for("diceguard", "v1")
    n = N_ROWS

    day = np.sort(rng.integers(0, N_DAYS, n))
    ts = START + pd.to_timedelta(day, unit="D") + pd.to_timedelta(rng.integers(0, 24 * 60, n), unit="m")
    machine = rng.choice(MACHINES, n)
    frac = day / (N_DAYS - 1)

    df = pd.DataFrame(
        {
            "Timestamp": ts,
            "Day_Index": day,
            "Machine_ID": machine,
            "Product_ID": rng.choice(PRODUCTS, n),
            "Shift": rng.choice(SHIFTS, n),
            "Lot_ID": [f"LOT{v:05d}" for v in rng.integers(1, 4200, n)],
            "Strip_ID": [f"STRIP{v:05d}" for v in rng.integers(1, 60000, n)],
            "Recipe_ID": rng.choice(RECIPES, n),
            "Operator_ID": rng.choice(OPERATORS, n),
        }
    )
    df["Strip_ID"] = [f"STRIP{i + 1:05d}" for i in range(n)]

    # 장비 고유 오프셋 + 제품/레시피 오프셋
    mach_off = {m: rng.normal(0, 0.45) for m in MACHINES}
    prod_off = {p: rng.normal(0, 0.30) for p in PRODUCTS}

    for col, (center, sigma) in ACTIONABLE.items():
        base = rng.normal(center, sigma, n)
        base += np.array([mach_off[m] for m in df["Machine_ID"]]) * sigma * 0.35
        base += np.array([prod_off[p] for p in df["Product_ID"]]) * sigma * 0.18
        # 주간 사이클 (PM 직후 좋아졌다가 서서히 나빠짐)
        base += np.sin(2 * np.pi * day / 7.0) * sigma * 0.12
        df[col] = base

    # 열화 주입: 스펙을 넘지 않는 선에서 천천히 민다
    degr_info = []
    for mach, col, sigma_shift, start_frac in DEGRADATION:
        sel = (df["Machine_ID"] == mach).to_numpy()
        prog = np.clip((frac - start_frac) / max(1e-9, 1 - start_frac), 0, 1)
        # 포화형 곡선. 초중반에 대부분 진행되고 후반에는 완만해진다.
        # 이러면 "최근 30일만 잘라 보면 기울기가 유의하지 않다" 는 현상이 실제로 생긴다.
        shift = sigma_shift * ACTIONABLE[col][1] * (prog**0.55)
        df.loc[sel, col] = df.loc[sel, col].to_numpy() + shift[sel]
        degr_info.append(
            {
                "machine": mach,
                "variable": col,
                "total_shift": round(float(sigma_shift * ACTIONABLE[col][1]), 4),
                "in_sigma": sigma_shift,
                "onset_day": int(start_frac * N_DAYS),
            }
        )

    # 감시지표: 설비 설정값의 결과로 계산된다 (원인 -> 결과)
    df["Kerf_Width_um"] = (
        26.0
        + 0.92 * (df["Laser_Power_W"] - 18.5) * 9.0
        - 0.020 * (df["Feed_Speed_mm_s"] - 250.0)
        + 0.55 * df["Beam_Center_Offset_um"].abs()
        + rng.normal(0, 0.42, n)
    )
    df["Groove_Depth_um"] = (
        31.5
        + 1.55 * (df["Power_Efficiency_pct"] - 94.9)
        - 0.028 * (df["Feed_Speed_mm_s"] - 250.0)
        + rng.normal(0, 0.55, n)
    )
    df["Coat_Thickness_um"] = 2.60 - 0.085 * (df["Clean_Flow_lpm"] - 8.4) + rng.normal(0, 0.075, n)
    df["Coat_Roughness_nm"] = (
        118.0 - 6.4 * (df["Clean_Pressure_bar"] - 2.35) * 4 + 9.0 * df["Vibration_mm_s"] + rng.normal(0, 5.5, n)
    )

    # 불량: 원인 변수에서 직접 나온다 (감시지표를 거치지 않는 경로도 있다)
    risk = {
        "Chipping": 2.6 * np.abs(df["Focus_Offset_um"]) + 1.9 * np.abs(df["Beam_Center_Offset_um"]) + 0.02 * (df["Feed_Speed_mm_s"] - 250),
        "MicroCrack": 3.4 * np.maximum(0, 94.9 - df["Power_Efficiency_pct"]) + 1.1 * df["Vibration_mm_s"],
        "Particle": 2.9 * np.maximum(0, 8.4 - df["Clean_Flow_lpm"]) + 0.9 * np.maximum(0, 2.35 - df["Clean_Pressure_bar"]) * 4,
        "CoatResidue": 3.1 * np.maximum(0, 8.4 - df["Clean_Flow_lpm"]) + 0.6 * np.maximum(0, 14.0 - df["Clean_Time_s"]),
        "EdgeBurn": 2.2 * np.maximum(0, df["Laser_Power_W"] - 18.5) * 9 + 0.8 * np.maximum(0, df["Head_Temp_C"] - 28.4),
    }
    base_rate = {"Chipping": 0.030, "MicroCrack": 0.003, "Particle": 0.055, "CoatResidue": 0.024, "EdgeBurn": 0.011}
    total_fail = np.zeros(n)
    for d in DEFECTS:
        z = risk[d].to_numpy() if hasattr(risk[d], "to_numpy") else np.asarray(risk[d])
        z = (z - np.median(z)) / (np.std(z) + 1e-9)
        p = np.clip(base_rate[d] * np.exp(0.95 * z), 0, 0.65)
        flag = rng.random(n) < p
        cnt = np.where(flag, rng.poisson(1.6, n) + 1, 0)
        df[f"{d}_Flag"] = flag.astype(int)
        df[f"{d}_Count"] = cnt
        total_fail += cnt

    df["Fail_Die"] = total_fail.astype(int)
    df["Yield_pct"] = np.round(np.clip(100.0 - total_fail * rng.uniform(0.20, 0.34, n), 88.0, 100.0), 2)

    ng = np.full(n, "OK", dtype=object)
    for d in DEFECTS:  # 뒤에 오는 불량이 우선순위를 덮어쓴다 (심각도 순)
        ng = np.where(df[f"{d}_Flag"].to_numpy() == 1, NG_CODE[d], ng)
    df["NG_Code"] = ng

    for c in list(ACTIONABLE) + MONITORING:
        df[c] = df[c].round(4)

    meta = pd.DataFrame(
        [
            {
                "Machine_ID": m,
                "Install_Year": int(rng.integers(2019, 2025)),
                "Laser_Source": rng.choice(["UV-355", "UV-266", "GRN-532"]),
                "PM_Interval_days": int(rng.choice([30, 45, 60])),
                "Strips_Processed": int((df["Machine_ID"] == m).sum()),
            }
            for m in MACHINES
        ]
    )

    schema = {
        "dataset": "DiceGuard laser dicing FDC",
        "grain": "1행 = 스트립 1장 가공 이력",
        "period": {"start": str(START.date()), "days": N_DAYS},
        "machines": MACHINES,
        "actionable_fdc": list(ACTIONABLE),
        "monitoring_response": MONITORING,
        "defects": DEFECTS,
        "targets": ["Fail_Die", "Yield_pct", "NG_Code"],
        "causal_note": "설비 설정값(actionable) -> 감시지표(monitoring) -> 불량. 단, 불량은 감시지표를 거치지 않고 설정값에서 직접 발현되는 경로도 있다.",
        "injected_degradation": degr_info,
        "defect_base_rate": base_rate,
    }
    return df, meta, schema


def validate(df: pd.DataFrame, schema: dict) -> Report:
    rep = Report("diceguard")
    n_cols = 9 + len(ACTIONABLE) + len(MONITORING) + 2 * len(DEFECTS) + 3
    check_shape(rep, df, N_ROWS, n_cols)
    check_no_duplicate_ids(rep, df, "Strip_ID")
    check_categories(rep, df, "Machine_ID", set(MACHINES))
    check_categories(rep, df, "NG_Code", {"OK"} | set(NG_CODE.values()))

    # 열화가 실제로 들어갔는지: 해당 장비의 전반부 vs 후반부 평균 차이
    for d in schema["injected_degradation"]:
        sub = df[df["Machine_ID"] == d["machine"]]
        early = sub[sub["Day_Index"] < N_DAYS * 0.25][d["variable"]].mean()
        late = sub[sub["Day_Index"] >= N_DAYS * 0.75][d["variable"]].mean()
        moved = late - early
        ok = np.sign(moved) == np.sign(d["total_shift"]) and abs(moved) > abs(d["total_shift"]) * 0.35
        rep.add(f"degradation:{d['machine']}/{d['variable']}", bool(ok), f"전반->후반 {moved:+.4f} (주입 {d['total_shift']:+.4f})")

    # 열화 변수가 관리한계(평균 +- 3시그마)를 벗어나지 않아야 한다
    for d in schema["injected_degradation"]:
        col = d["variable"]
        center, sigma = ACTIONABLE[col]
        sub = df[df["Machine_ID"] == d["machine"]][col]
        daily = sub.groupby(df.loc[sub.index, "Day_Index"]).mean()
        within = bool(((daily - center).abs() < 3.0 * sigma).mean() > 0.97)
        rep.add(f"within_spec:{col}", within, f"일평균이 +-3sigma 안에 있는 비율 {((daily - center).abs() < 3 * sigma).mean():.3f}")

    # 희소 불량이 실제로 희소한지
    mc = float(df["MicroCrack_Flag"].mean())
    rep.add("rare_defect", 0.0005 <= mc <= 0.02, f"MicroCrack 발생률 {mc:.4%}")
    # 커프 폭과 레이저 파워가 강하게 엮여 있어야 한다 (교란 시나리오의 핵심)
    r = float(df["Laser_Power_W"].corr(df["Kerf_Width_um"]))
    rep.add("kerf_confound", abs(r) > 0.5, f"corr(Laser_Power, Kerf_Width) = {r:+.3f}")
    return rep


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    DATA.mkdir(exist_ok=True)
    df, meta, schema = build()
    rep = validate(df, schema)
    print(rep.render())
    rep.raise_if_failed()
    df.to_csv(DATA / "dicing_fdc.csv.gz", index=False, encoding="utf-8")
    meta.to_csv(DATA / "equipment_meta.csv", index=False, encoding="utf-8")
    schema["shape"] = list(df.shape)
    (DATA / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{df.shape} -> data/dicing_fdc.csv")


if __name__ == "__main__":
    main()
