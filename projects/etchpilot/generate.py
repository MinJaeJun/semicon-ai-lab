"""EtchPilot — Etch 레시피 평가 데이터셋 생성기.

4개 공정(Poly / Contact / Via / Pad)에 대해 Base + Rev1~Rev11 총 12개 레시피
리비전을 돌린 웨이퍼별 29-site 계측 결과를 만든다.

이 데이터셋의 핵심 구조
----------------------
* 레시피는 한 번에 여러 파라미터를 바꾸지 않는다. 리비전마다 1~3개만 바뀐다.
  그래야 "이 파라미터를 바꿨더니 품질이 어떻게 변했나" 를 나중에 되짚을 수 있다.
* 같은 레시피라도 웨이퍼 위치에 따라 결과가 다르다. 중심-엣지 프로파일이 있고,
  Extreme Edge 로 갈수록 CD 가 벌어지고 Defect 이 늘어난다.
* 장비/챔버마다 고유 바이어스가 있다. 같은 레시피를 A 챔버와 D 챔버에서 돌리면
  결과가 다르다. 이걸 모르면 "레시피가 나쁘다" 와 "챔버가 나쁘다" 를 구분 못 한다.
* Depth 단위가 공정마다 다르다 (Poly 는 Å, 나머지는 nm). 실제로 자주 겪는 함정이다.

실행: python generate.py
출력: data/{process}_sites.csv, data/{process}_recipe_master.csv, data/schema.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.synth import SiteGrid, radial_profile, rng_for  # noqa: E402
from common.validate import Report, check_categories, check_no_duplicate_ids, check_range, check_shape  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# 29-site 배치. 16팀식 25-site(1/4/10/10)와 달리 Mid 를 6개로 늘리고
# Edge/ExtEdge 를 11개씩 둬서 각도 해상도를 높였다.
GRID = SiteGrid(
    zones=(
        ("Center", 1, 0.00),
        ("Mid", 6, 0.52),
        ("Edge", 11, 0.84),
        ("Extreme Edge", 11, 0.96),
    )
)
SITES = GRID.build()

REVISIONS = ["Base"] + [f"Rev{i}" for i in range(1, 12)]  # 12개

TOOLS = {
    "AE-3200": ["CH-A", "CH-B"],
    "PX-7500": ["CH-C", "CH-D"],
}

STAGE_PARAMS = ["Time_s", "Source_RF_W", "Bias_RF_W", "Pressure_mT", "Gas_Flow_sccm"]
PARAM_BASE = {
    "Time_s": (62.0, 7.0, 1),
    "Source_RF_W": (950.0, 70.0, 0),
    "Bias_RF_W": (205.0, 30.0, 0),
    "Pressure_mT": (16.5, 3.2, 1),
    "Gas_Flow_sccm": (135.0, 18.0, 0),
    "Gas2_Flow_sccm": (52.0, 9.0, 0),
}

PROCESSES: dict[str, dict] = {
    "poly": {
        "wafers": 216,
        "stages": ["S1", "S2"],
        "extra_param": "Gas2_Flow_sccm",  # 2가스 공정 -> stage 당 6개
        "depth_col": "Depth_A",
        "depth_unit": "angstrom",
        "center": {"top_cd": 38.4, "mid_cd": 39.8, "bottom_cd": 37.1, "depth": 1665.0},
        "sigma": {"top_cd": 0.62, "mid_cd": 0.78, "bottom_cd": 1.42, "depth": 46.0},
        "edge_delta": {"top_cd": 2.4, "mid_cd": 1.6, "bottom_cd": 1.2, "depth": 68.0},
        "spec": {"top_cd": (36.6, 40.4), "mid_cd": (37.8, 42.0), "bottom_cd": (34.2, 40.4), "depth": (1580.0, 1760.0)},
    },
    "contact": {
        "wafers": 204,
        "stages": ["S1", "S2", "S3"],
        "extra_param": None,
        "depth_col": "Depth_nm",
        "depth_unit": "nm",
        "center": {"top_cd": 71.5, "mid_cd": 69.2, "bottom_cd": 64.8, "depth": 412.0},
        "sigma": {"top_cd": 1.35, "mid_cd": 1.62, "bottom_cd": 2.85, "depth": 21.0},
        "edge_delta": {"top_cd": 3.1, "mid_cd": 2.6, "bottom_cd": 3.4, "depth": 17.0},
        "spec": {"top_cd": (68.4, 75.2), "mid_cd": (65.6, 72.8), "bottom_cd": (59.4, 70.6), "depth": (376.0, 452.0)},
    },
    "via": {
        "wafers": 228,
        "stages": ["S1", "S2", "S3", "S4"],
        "extra_param": None,
        "depth_col": "Depth_nm",
        "depth_unit": "nm",
        "center": {"top_cd": 96.0, "mid_cd": 92.4, "bottom_cd": 86.5, "depth": 985.0},
        "sigma": {"top_cd": 1.9, "mid_cd": 2.4, "bottom_cd": 4.1, "depth": 58.0},
        "edge_delta": {"top_cd": 4.2, "mid_cd": 3.6, "bottom_cd": 4.8, "depth": 44.0},
        "spec": {"top_cd": (91.5, 100.8), "mid_cd": (86.8, 97.6), "bottom_cd": (78.0, 94.5), "depth": (880.0, 1090.0)},
    },
    "pad": {
        "wafers": 192,
        "stages": ["S1"],
        "extra_param": None,
        "depth_col": "Depth_nm",
        "depth_unit": "nm",
        "center": {"top_cd": 2150.0, "mid_cd": 2105.0, "bottom_cd": 2040.0, "depth": 3250.0},
        "sigma": {"top_cd": 26.0, "mid_cd": 32.0, "bottom_cd": 58.0, "depth": 130.0},
        "edge_delta": {"top_cd": 58.0, "mid_cd": 48.0, "bottom_cd": 66.0, "depth": 105.0},
        "spec": {
            "top_cd": (2075.0, 2225.0),
            "mid_cd": (2015.0, 2195.0),
            "bottom_cd": (1900.0, 2170.0),
            "depth": (3000.0, 3500.0),
        },
    },
}

ANOMALIES = ["polymer_residue", "arc_event", "pressure_spike", "endpoint_late", "esc_temp_drift"]
CD_KEYS = ("top_cd", "mid_cd", "bottom_cd", "depth")


def recipe_master(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    params = list(STAGE_PARAMS) + ([cfg["extra_param"]] if cfg["extra_param"] else [])
    cols = [f"{st}_{p}" for st in cfg["stages"] for p in params]
    cur = {}
    for c in cols:
        base, spread, nd = PARAM_BASE[c.split("_", 1)[1]]
        cur[c] = round(float(base + rng.normal(0, spread * 0.12)), nd)
    rows = []
    for vi, rev in enumerate(REVISIONS):
        changed: list[str] = []
        if vi > 0:
            k = int(rng.integers(1, 4))
            changed = sorted(rng.choice(cols, size=min(k, len(cols)), replace=False).tolist())
            for c in changed:
                base, spread, nd = PARAM_BASE[c.split("_", 1)[1]]
                cur[c] = round(float(cur[c] + rng.normal(0, spread * 0.45)), nd)
        rows.append({"Recipe_Version": rev, "Rev_No": vi, "Changed_Params": "|".join(changed), **cur})
    return pd.DataFrame(rows)


def build(name: str, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = rng_for("etchpilot", name)
    master = recipe_master(cfg, rng)
    pcols = [c for c in master.columns if c not in ("Recipe_Version", "Rev_No", "Changed_Params")]

    # 레시피 파라미터를 z-score 로 바꾸고, 항목별 민감도 벡터를 만든다.
    # sqrt(len) 로 나눠 내적 분산이 대략 1이 되게 한다 -- 안 그러면 파라미터가 많은
    # 공정일수록 품질이 미쳐 날뛴다.
    pz = master.set_index("Rev_No")[pcols]
    pz = (pz - pz.mean()) / pz.std(ddof=0).replace(0, 1)
    sens = {k: rng.normal(0, 1, len(pcols)) / np.sqrt(len(pcols)) for k in CD_KEYS}

    # 장비/챔버 바이어스
    combos = [(t, c) for t, chs in TOOLS.items() for c in chs]
    tool_bias = {tc: rng.normal(0, 0.42) for tc in combos}

    n_waf = cfg["wafers"]
    assign = rng.permutation(n_waf)
    rows = []
    sample = 0
    for wi in range(n_waf):
        tool, cham = combos[assign[wi] % len(combos)]
        rev_no = int(assign[wi] % len(REVISIONS))
        rev = REVISIONS[rev_no]
        wafer_id = f"W{wi + 1:03d}"
        z = pz.loc[rev_no].to_numpy()
        bias = tool_bias[(tool, cham)]
        # 웨이퍼 단위 오프셋 (같은 레시피/장비여도 장마다 다르다)
        w_off = {k: rng.normal(0, cfg["sigma"][k] * 0.5) for k in CD_KEYS}

        for site_id, zone, rfrac, ang in SITES:
            sample += 1
            vals = {}
            for k in CD_KEYS:
                sg = cfg["sigma"][k]
                recipe_eff = float(z @ sens[k]) * sg * 0.85
                vals[k] = (
                    cfg["center"][k]
                    + recipe_eff
                    + radial_profile(np.array(rfrac), cfg["edge_delta"][k])
                    + w_off[k]
                    + bias * sg * 0.7
                    + rng.normal(0, sg)
                )
            passes = {k: bool(cfg["spec"][k][0] <= vals[k] <= cfg["spec"][k][1]) for k in CD_KEYS}
            n_fail = sum(not v for v in passes.values())
            lam = 0.09 + 0.5 * rfrac**3 + 0.14 * n_fail
            dcount = int(rng.poisson(lam))
            sev = round(float(dcount * rng.uniform(1.8, 6.5)), 2) if dcount else 0.0
            anomaly = str(rng.choice(ANOMALIES, p=[0.55, 0.12, 0.13, 0.10, 0.10])) if (dcount and rng.random() < 0.5) else ""

            rows.append(
                {
                    "Sample_No": sample,
                    "Recipe_Version": rev,
                    "Rev_No": rev_no,
                    "Wafer_ID": wafer_id,
                    "Tool_ID": tool,
                    "Chamber_ID": cham,
                    "Site_ID": site_id,
                    "Zone": zone,
                    "Radius_frac": rfrac,
                    "Angle_deg": ang,
                    "Top_CD_nm": round(vals["top_cd"], 2),
                    "Mid_CD_nm": round(vals["mid_cd"], 2),
                    "Bottom_CD_nm": round(vals["bottom_cd"], 2),
                    cfg["depth_col"]: round(vals["depth"], 1),
                    "Top_CD_Pass": passes["top_cd"],
                    "Mid_CD_Pass": passes["mid_cd"],
                    "Bottom_CD_Pass": passes["bottom_cd"],
                    "Depth_Pass": passes["depth"],
                    "Defect_Count": dcount,
                    "Defect_Severity": sev,
                    "Anomaly_Event": anomaly,
                }
            )

    df = pd.DataFrame(rows)
    meta = {
        "process": name,
        "wafers": n_waf,
        "sites_per_wafer": GRID.size,
        "rows": len(df),
        "revisions": len(REVISIONS),
        "stages": cfg["stages"],
        "recipe_params": len(pcols),
        "depth_column": cfg["depth_col"],
        "depth_unit": cfg["depth_unit"],
        "spec": {k: list(v) for k, v in cfg["spec"].items()},
        "tool_chamber": {f"{t}/{c}": int((df["Tool_ID"].eq(t) & df["Chamber_ID"].eq(c)).sum() // GRID.size) for t, c in combos},
    }
    return df, master, meta


def validate(name: str, df: pd.DataFrame, cfg: dict, meta: dict) -> Report:
    rep = Report(f"etchpilot/{name}")
    check_shape(rep, df, cfg["wafers"] * GRID.size, 21)
    check_no_duplicate_ids(rep, df, "Sample_No")
    check_categories(rep, df, "Zone", {"Center", "Mid", "Edge", "Extreme Edge"})
    check_categories(rep, df, "Recipe_Version", set(REVISIONS))
    for k, col in zip(("top_cd", "mid_cd", "bottom_cd"), ("Top_CD_nm", "Mid_CD_nm", "Bottom_CD_nm")):
        lo, hi = cfg["center"][k] - 12 * cfg["sigma"][k], cfg["center"][k] + 12 * cfg["sigma"][k]
        check_range(rep, df, col, lo, hi)

    # 엣지로 갈수록 Defect 이 늘어야 한다
    zmean = df.groupby("Zone")["Defect_Count"].mean()
    rep.add(
        "edge_defect_gradient",
        bool(zmean["Extreme Edge"] > zmean["Center"]),
        f"Center {zmean['Center']:.3f} -> ExtEdge {zmean['Extreme Edge']:.3f}",
    )
    # 레시피 리비전에 따라 Pass Rate 가 실제로 움직여야 한다
    pr = df.groupby("Rev_No")[["Top_CD_Pass", "Depth_Pass"]].mean().mean(axis=1)
    rep.add("recipe_effect", bool(pr.max() - pr.min() > 0.10), f"Rev별 Pass Rate 폭 {pr.max() - pr.min():.3f}")
    # 전체 Pass Rate 가 극단으로 쏠리지 않아야 한다
    overall = float(df[["Top_CD_Pass", "Mid_CD_Pass", "Bottom_CD_Pass", "Depth_Pass"]].to_numpy().mean())
    rep.add("pass_rate_band", 0.35 <= overall <= 0.92, f"{overall:.3f}")
    return rep


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    DATA.mkdir(exist_ok=True)
    schema = {
        "dataset": "EtchPilot etch recipe evaluation",
        "grain": "1행 = 1 Wafer의 1 Site 계측 결과",
        "site_layout": {
            "total": GRID.size,
            "zones": [{"zone": z, "count": c, "radius_frac": r} for z, c, r in GRID.zones],
            "sites": [{"site_id": s, "zone": z, "radius_frac": r, "angle_deg": a} for s, z, r, a in SITES],
        },
        "revisions": REVISIONS,
        "tools": TOOLS,
        "processes": {},
    }
    for name, cfg in PROCESSES.items():
        df, master, meta = build(name, cfg)
        rep = validate(name, df, cfg, meta)
        print(rep.render())
        rep.raise_if_failed()
        df.to_csv(DATA / f"{name}_sites.csv", index=False, encoding="utf-8")
        master.to_csv(DATA / f"{name}_recipe_master.csv", index=False, encoding="utf-8")
        schema["processes"][name] = meta
    (DATA / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(m["rows"] for m in schema["processes"].values())
    print(f"\n총 {total:,}행 / 4공정 -> {DATA}")


if __name__ == "__main__":
    main()
