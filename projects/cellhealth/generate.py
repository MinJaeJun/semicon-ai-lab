"""CellHealth — 모바일 스토리지(UFS/eMMC) 필드 헬스 데이터셋 생성기.

출하된 디바이스 200,000대에서 2회차에 걸쳐 수집한 헬스 텔레메트리를 만든다.

이 데이터셋의 성질
-----------------
* 마모는 선형이 아니다. pe_cycle 이 정격의 70% 를 넘어가면 uecc 가 지수적으로 증가한다.
  선형 회귀로 잔여수명을 추정하면 위험 구간을 과소평가하게 된다.
* 벤더/제품군마다 정격 P/E 수명이 다르다. 같은 pe_cycle 값이라도
  TLC 128GB 와 TLC 512GB 의 의미가 다르다. 정규화 없이 비교하면 틀린다.
* 불량은 극소수다. rtbb > 0 인 디바이스가 3% 내외, uecc > 0 이 1% 내외다.
  전체 평균을 보면 아무 일도 없어 보인다.
* 1회차와 2회차 사이에 실제로 마모가 진행된다. 같은 ufsid 를 추적할 수 있다.

실행: python generate.py
출력: data/device_health.csv, data/column_dictionary.csv, data/schema.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.synth import rng_for  # noqa: E402
from common.validate import Report, check_categories, check_range, check_shape  # noqa: E402

HERE = Path(__file__).resolve().parent

# gzip 헤더에 압축 시각이 박히면 같은 데이터를 다시 만들어도 파일 바이트가 달라진다.
# mtime 을 0 으로 고정해야 재생성 결과가 바이트 단위로 동일해진다.
GZIP = {"method": "gzip", "mtime": 0}

DATA = HERE / "data"

N_DEVICES = 100_000
ROUNDS = [1, 2]  # 같은 디바이스를 두 번 수집한다

VENDORS = {
    "HYX": {"share": 0.46, "prefix": "H9T", "rated_pe": 3000, "quality": 1.00},
    "SMS": {"share": 0.34, "prefix": "KLU", "rated_pe": 2800, "quality": 0.96},
    "MCR": {"share": 0.20, "prefix": "MTF", "rated_pe": 2400, "quality": 0.90},
}
FAMILIES = {"UFS3.1": 0.52, "UFS2.2": 0.30, "eMMC5.1": 0.18}
DENSITIES = {128: 0.34, 256: 0.41, 512: 0.20, 1024: 0.05}

COLUMN_DICT = [
    ("device_uid", "디바이스 고유 ID (수집 회차 간 동일)", "string"),
    ("collect_round", "수집 회차 (1차 / 2차)", "int"),
    ("collect_date", "수집 일자", "date"),
    ("vendor", "메모리 제조사 코드", "string"),
    ("product_family", "제품군 (UFS3.1 / UFS2.2 / eMMC5.1)", "string"),
    ("part_no", "제품 Part Number", "string"),
    ("density_gb", "공칭 용량 (GB)", "int"),
    ("fw_version", "펌웨어 버전", "string"),
    ("storage_used_gb", "실사용 용량 (GB)", "float"),
    ("storage_util", "총 용량 대비 사용률 (0~1)", "float"),
    ("pe_cycle", "블록 Program/Erase 반복 횟수", "int"),
    ("pe_ratio", "정격 P/E 수명 대비 소모율 (pe_cycle / rated_pe)", "float"),
    ("tbw_gb", "누적 기록량 Total Bytes Written (GB)", "int"),
    ("power_on_hours", "누적 전원 인가 시간", "int"),
    ("temp_max_c", "관측 최고 온도 (섭씨)", "int"),
    ("rtbb", "Run Time Bad Block — 사용 중 발생한 불량 블록 수", "int"),
    ("uecc", "Uncorrectable ECC — 정정 불가 비트 에러 발생 횟수", "int"),
    ("health_pct", "컨트롤러가 보고하는 잔여 수명 (%)", "int"),
]


def build() -> tuple[pd.DataFrame, dict]:
    rng = rng_for("cellhealth", "v1")
    nd = N_DEVICES

    vendor = rng.choice(list(VENDORS), nd, p=[v["share"] for v in VENDORS.values()])
    family = rng.choice(list(FAMILIES), nd, p=list(FAMILIES.values()))
    density = rng.choice(list(DENSITIES), nd, p=list(DENSITIES.values()))
    uid = np.array([f"{rng.integers(0x1000, 0xFFFF):04x}{i:08x}" for i in range(nd)])

    rated = np.array([VENDORS[v]["rated_pe"] for v in vendor], dtype=float)
    quality = np.array([VENDORS[v]["quality"] for v in vendor])
    # eMMC 는 정격 수명이 낮다
    rated = rated * np.where(family == "eMMC5.1", 0.62, 1.0)

    part_no = np.array(
        [f"{VENDORS[v]['prefix']}{d:04d}{rng.integers(10, 99):02d}" for v, d in zip(vendor, density)]
    )
    fw = np.array([f"{rng.integers(1, 5)}.{rng.integers(0, 9)}.{rng.integers(0, 9)}" for _ in range(nd)])

    # 사용 강도: 로그정규. 헤비유저가 소수 존재한다.
    intensity = rng.lognormal(0.0, 0.62, nd)
    frames = []
    for rnd in ROUNDS:
        months = 9 * rnd  # 1차 9개월, 2차 18개월 사용 시점
        poh = (months * 30 * 24 * rng.uniform(0.28, 0.55, nd) * intensity).astype(int)
        pe = (rated * 0.09 * months / 9 * intensity * rng.uniform(0.7, 1.35, nd)).astype(int)
        pe_ratio = pe / rated
        tbw = (pe * density * 0.55 * rng.uniform(0.8, 1.25, nd)).astype(int)

        util = np.clip(rng.beta(2.4, 2.0, nd) * (0.55 + 0.45 * intensity / intensity.max()), 0.02, 0.99)
        used = np.round(util * density, 2)
        tmax = np.clip(rng.normal(46 + 6 * intensity, 5, nd), 28, 92).astype(int)

        # 마모 -> 불량. 정격 대비 70% 를 넘으면 급격히 나빠진다.
        stress = np.maximum(0.0, pe_ratio - 0.70) / 0.30
        lam_rtbb = (0.02 + 2.8 * stress**2.1) / quality * (1 + 0.012 * np.maximum(0, tmax - 60))
        rtbb = rng.poisson(np.clip(lam_rtbb, 0, 60))
        lam_uecc = (0.004 + 1.6 * stress**2.6) / quality
        uecc = rng.poisson(np.clip(lam_uecc, 0, 40))

        health = np.clip(np.round(100 * (1 - pe_ratio) - rtbb * 0.35 - uecc * 0.8), 0, 100).astype(int)

        frames.append(
            pd.DataFrame(
                {
                    "device_uid": uid,
                    "collect_round": rnd,
                    "collect_date": pd.Timestamp("2026-03-15") + pd.Timedelta(days=180 * (rnd - 1)),
                    "vendor": vendor,
                    "product_family": family,
                    "part_no": part_no,
                    "density_gb": density,
                    "fw_version": fw,
                    "storage_used_gb": used,
                    "storage_util": np.round(util, 4),
                    "pe_cycle": pe,
                    "pe_ratio": np.round(pe_ratio, 4),
                    "tbw_gb": tbw,
                    "power_on_hours": poh,
                    "temp_max_c": tmax,
                    "rtbb": rtbb,
                    "uecc": uecc,
                    "health_pct": health,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True).sort_values(["device_uid", "collect_round"]).reset_index(drop=True)
    schema = {
        "dataset": "CellHealth mobile storage field telemetry",
        "grain": "1행 = 디바이스 1대 x 수집 회차 1회",
        "devices": int(nd),
        "rounds": ROUNDS,
        "rows": int(len(df)),
        "vendors": {k: {"rated_pe": v["rated_pe"], "share": v["share"]} for k, v in VENDORS.items()},
        "families": FAMILIES,
        "densities": list(DENSITIES),
        "wearout_model": "pe_ratio > 0.70 부터 rtbb/uecc 가 멱함수적으로 증가 (지수 2.1 / 2.6)",
        "columns": [{"name": n, "description": d, "type": t} for n, d, t in COLUMN_DICT],
    }
    return df, schema


def validate(df: pd.DataFrame) -> Report:
    rep = Report("cellhealth")
    check_shape(rep, df, N_DEVICES * len(ROUNDS), len(COLUMN_DICT))
    check_categories(rep, df, "vendor", set(VENDORS))
    check_categories(rep, df, "product_family", set(FAMILIES))
    check_range(rep, df, "storage_util", 0.0, 1.0)
    check_range(rep, df, "health_pct", 0, 100)

    dup = int(df.duplicated(["device_uid", "collect_round"]).sum())
    rep.add("unique:(device_uid,round)", dup == 0, f"중복 {dup}건")

    # 2회차가 1회차보다 마모가 진행돼 있어야 한다
    piv = df.pivot_table(index="device_uid", columns="collect_round", values="pe_cycle", aggfunc="first")
    worse = float((piv[2] > piv[1]).mean())
    rep.add("wear_progresses", worse > 0.95, f"2회차 pe_cycle 증가 비율 {worse:.3f}")

    # 불량은 희소해야 한다
    rr, ur = float((df["rtbb"] > 0).mean()), float((df["uecc"] > 0).mean())
    rep.add("rare_rtbb", 0.005 <= rr <= 0.12, f"rtbb>0 비율 {rr:.4f}")
    rep.add("rare_uecc", 0.001 <= ur <= 0.06, f"uecc>0 비율 {ur:.4f}")

    # 마모 임계 위/아래에서 불량률이 확 갈려야 한다
    lo = float((df.loc[df["pe_ratio"] <= 0.70, "uecc"] > 0).mean())
    hi = float((df.loc[df["pe_ratio"] > 0.70, "uecc"] > 0).mean())
    rep.add("wearout_knee", hi > lo * 5 and hi > 0.02, f"pe_ratio<=0.7 {lo:.4f} vs >0.7 {hi:.4f}")
    return rep


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    DATA.mkdir(exist_ok=True)
    df, schema = build()
    rep = validate(df)
    print(rep.render())
    rep.raise_if_failed()
    df.to_csv(DATA / "device_health.csv.gz", index=False, encoding="utf-8", compression=GZIP)
    pd.DataFrame(COLUMN_DICT, columns=["column", "description", "type"]).to_csv(
        DATA / "column_dictionary.csv", index=False, encoding="utf-8"
    )
    (DATA / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{df.shape} -> data/device_health.csv")


if __name__ == "__main__":
    main()
