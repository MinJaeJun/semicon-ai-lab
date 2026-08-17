"""분석 산출물과 원본 데이터를 대시보드용 단일 JSON 으로 압축한다.

브라우저에 30만 행을 통째로 올릴 수는 없다. 화면에 실제로 필요한 것만 남긴다.
- 산점도는 층화 샘플링으로 2,000점 이하
- 시계열은 일 단위 집계본
- 표는 상위 N개
결과는 site/data.json 하나. 정적 호스팅에 그대로 올라간다.

실행: python site/build_site_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
P = ROOT / "projects"


def r(x, n=4):
    if isinstance(x, (list, tuple)):
        return [r(v, n) for v in x]
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), n)


def sample(df: pd.DataFrame, cols: list[str], n: int, seed: int = 1) -> list[list]:
    d = df[cols].dropna()
    if len(d) > n:
        d = d.sample(n, random_state=seed)
    return [[r(v) for v in row] for row in d.to_numpy()]


def bin_profile(df: pd.DataFrame, x: str, y: str, bins: int = 12) -> dict:
    d = df[[x, y]].dropna()
    q = pd.qcut(d[x], bins, duplicates="drop")
    g = d.groupby(q, observed=True).agg(x=(x, "mean"), y=(y, "mean"), n=(y, "size"))
    return {"x": r(g["x"].tolist()), "y": r(g["y"].tolist()), "n": [int(v) for v in g["n"]]}


# --------------------------------------------------------------- yieldlens
def yieldlens() -> dict:
    d = P / "yieldlens"
    df = pd.read_csv(d / "data" / "wafer_history.csv.gz")
    schema = json.loads((d / "data" / "schema.json").read_text(encoding="utf-8"))
    scr = pd.read_csv(d / "outputs" / "screening.csv")
    spec = pd.read_csv(d / "outputs" / "specificity.csv", index_col=0)
    win = json.loads((d / "outputs" / "control_windows.json").read_text(encoding="utf-8"))
    rep = json.loads((d / "outputs" / "model_report.json").read_text(encoding="utf-8"))
    truth = {x["factor"] for x in schema["ground_truth_drivers"]}

    drivers = []
    for w in win:
        f, t = w["factor"], w["target"]
        drivers.append(
            {
                **{k: w[k] for k in ("factor", "target", "shape", "optimum", "recommended_low", "recommended_high", "current_median")},
                "profile": bin_profile(df, f, t),
                "points": sample(df, [f, t], 900, seed=hash(f) % 999),
            }
        )

    sp = spec.reset_index().rename(columns={"index": "factor"})
    return {
        "meta": {
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
            "lots": int(df["Lot_ID"].nunique()),
            "tests": int(len(scr)),
            "n_significant": int(scr["significant"].sum()),
            "n_sig_factors": int(scr[scr["significant"]]["factor"].nunique()),
        },
        "detection": rep["driver_detection"],
        "model": {k: rep[k] for k in ("mae_mean", "r2_mean", "top30_worst_precision")},
        "ablation": rep["ablation_without_missing_indicator"],
        "specificity": [
            {"factor": row.factor, "rho_mode": r(row.rho_mode), "rho_yield": r(row.rho_yield),
             "spec": r(row.specificity), "truth": row.factor in truth}
            for row in sp.itertuples()
        ],
        "drivers": drivers,
        "measure_rate": {
            "response": r(df[schema["response_columns"]].notna().to_numpy().mean() * 100, 2),
            "defect": r(df[schema["defect_columns"]].notna().to_numpy().mean() * 100, 2),
        },
        "yield_hist": np.histogram(df["Yield_pct"], bins=40)[0].tolist(),
        "yield_edges": r(np.histogram(df["Yield_pct"], bins=40)[1].tolist(), 2),
    }


# --------------------------------------------------------------- etchpilot
def etchpilot() -> dict:
    d = P / "etchpilot"
    schema = json.loads((d / "data" / "schema.json").read_text(encoding="utf-8"))
    model = json.loads((d / "outputs" / "model_report.json").read_text(encoding="utf-8"))
    rank = pd.read_csv(d / "outputs" / "recipe_ranking.csv")
    ofat = json.loads((d / "outputs" / "ofat_recommendation.json").read_text(encoding="utf-8"))

    procs = {}
    for name, meta in schema["processes"].items():
        sites = pd.read_csv(d / "data" / f"{name}_sites.csv")
        dcol = meta["depth_column"]
        best = rank[rank["process"] == name].iloc[0]
        worst = rank[rank["process"] == name].iloc[-1]
        wafer_map = {}
        for label, rev in (("best", best["Rev_No"]), ("worst", worst["Rev_No"])):
            sub = sites[sites["Rev_No"] == rev]
            agg = sub.groupby(["Site_ID", "Zone", "Radius_frac", "Angle_deg"], as_index=False).agg(
                cd=("Top_CD_nm", "mean"), defect=("Defect_Count", "mean"),
                pass_rate=("Top_CD_Pass", "mean"),
            )
            wafer_map[label] = {
                "rev": str(rank[(rank["process"] == name) & (rank["Rev_No"] == rev)]["Recipe_Version"].iloc[0]),
                "sites": [
                    {"id": t.Site_ID, "zone": t.Zone, "rf": r(t.Radius_frac, 3), "ang": r(t.Angle_deg, 1),
                     "cd": r(t.cd, 2), "defect": r(t.defect, 3), "pass": r(t.pass_rate, 3)}
                    for t in agg.itertuples()
                ],
            }
        zone = sites.groupby("Zone").agg(cd=("Top_CD_nm", "mean"), defect=("Defect_Count", "mean")).reindex(
            ["Center", "Mid", "Edge", "Extreme Edge"]
        )
        procs[name] = {
            "meta": {k: meta[k] for k in ("wafers", "rows", "recipe_params", "depth_unit")},
            "stages": len(meta["stages"]),
            "metrics": model[name],
            "ranking": [
                {"rev": t.Recipe_Version, "score": r(t.quality_score, 2), "pass": r(t.pass_rate, 3),
                 "unif": r(t.uniformity, 3), "prox": r(t.target_proximity, 3), "rank": int(t.rank)}
                for t in rank[rank["process"] == name].itertuples()
            ],
            "wafer_map": wafer_map,
            "zone_profile": {"zones": list(zone.index), "cd": r(zone["cd"].tolist()), "defect": r(zone["defect"].tolist())},
            "ofat": ofat.get(name, [])[:6],
            "depth_col": dcol,
        }
    return {"processes": procs, "sites_per_wafer": schema["site_layout"]["total"], "revisions": schema["revisions"]}


# --------------------------------------------------------------- diceguard
def diceguard() -> dict:
    d = P / "diceguard"
    df = pd.read_csv(d / "data" / "dicing_fdc.csv.gz")
    schema = json.loads((d / "data" / "schema.json").read_text(encoding="utf-8"))
    sweep = pd.read_csv(d / "outputs" / "window_sensitivity.csv")
    trend = pd.read_csv(d / "outputs" / "trend_detection.csv")
    hi = pd.read_csv(d / "outputs" / "health_index.csv")
    attr = json.loads((d / "outputs" / "defect_attribution.json").read_text(encoding="utf-8"))
    act = schema["actionable_fdc"]
    injected = {(x["machine"], x["variable"]) for x in schema["injected_degradation"]}

    daily = df.groupby(["Machine_ID", "Day_Index"], as_index=False)[act].mean()
    series = {}
    for mach, var in sorted(injected):
        sub = daily[daily["Machine_ID"] == mach].sort_values("Day_Index")
        center, sigma = float(df[var].median()), float(df[var].std())
        series[f"{mach}|{var}"] = {
            "machine": mach, "variable": var,
            "day": [int(v) for v in sub["Day_Index"]],
            "value": r(sub[var].tolist(), 4),
            "center": r(center, 4), "ucl": r(center + 3 * sigma, 4), "lcl": r(center - 3 * sigma, 4),
        }
    # 대조군: 열화 없는 정상 변수 하나
    calm = daily[daily["Machine_ID"] == "LD-01"].sort_values("Day_Index")
    c0, s0 = float(df["Head_Temp_C"].median()), float(df["Head_Temp_C"].std())
    series["LD-01|Head_Temp_C"] = {
        "machine": "LD-01", "variable": "Head_Temp_C", "day": [int(v) for v in calm["Day_Index"]],
        "value": r(calm["Head_Temp_C"].tolist(), 4), "center": r(c0, 4),
        "ucl": r(c0 + 3 * s0, 4), "lcl": r(c0 - 3 * s0, 4),
    }

    worst = hi.sort_values("health_index").drop_duplicates("Machine_ID").sort_values("health_index")
    ng = df["NG_Code"].value_counts()
    return {
        "meta": {"rows": int(len(df)), "machines": int(df["Machine_ID"].nunique()), "days": schema["period"]["days"]},
        "injected": schema["injected_degradation"],
        "sweep": sweep.to_dict("records"),
        "detected": [
            {"machine": t.machine, "variable": t.variable, "slope": r(t.slope_per_day, 6),
             "total": r(t.total_change, 4), "tau": r(t.kendall_tau, 3), "p": float(t.p_kendall),
             "injected": bool(t.is_injected)}
            for t in trend[(trend["window"] == "full") & (trend["significant"])].itertuples()
        ],
        "series": series,
        "health": [
            {"machine": t.Machine_ID, "variable": t.variable, "hi": r(t.health_index, 2),
             "level": r(t.level, 2), "urgency": r(t.urgency_U, 3), "trend": bool(t.trend_significant)}
            for t in worst.itertuples()
        ],
        "attribution": attr,
        "ng_dist": {str(k): int(v) for k, v in ng.items()},
        "kerf_confound": {
            "corr": r(df["Laser_Power_W"].corr(df["Kerf_Width_um"]), 3),
            "points": sample(df, ["Laser_Power_W", "Kerf_Width_um"], 1200, seed=7),
        },
    }


# --------------------------------------------------------------- cellhealth
def cellhealth() -> dict:
    d = P / "cellhealth"
    df = pd.read_csv(d / "data" / "device_health.csv.gz")
    model = json.loads((d / "outputs" / "wearout_model.json").read_text(encoding="utf-8"))
    q = json.loads((d / "outputs" / "query_results.json").read_text(encoding="utf-8"))
    knee = model["uecc_rate_by_pe_ratio"]
    vend = df.groupby("vendor").agg(pe=("pe_cycle", "mean"), uecc=("uecc", "mean"), n=("uecc", "size"))
    return {
        "meta": {"rows": int(len(df)), "devices": int(df["device_uid"].nunique()), "rounds": 2},
        "model": model["models"],
        "positive_rate": model["positive_rate"],
        "knee": {"bins": list(knee), "rate": r(list(knee.values()), 5)},
        "queries": [
            {"q": x["question"], "cols": x["response"].get("columns_used", []),
             "matched": x["response"].get("rows_matched"), "result": x["response"].get("result", [])[:4],
             "filters": x["response"].get("filters_applied", []), "agg": x["response"].get("aggregation")}
            for x in q
        ],
        "vendors": [{"vendor": i, "pe": r(v.pe, 1), "uecc": r(v.uecc, 4), "n": int(v.n)} for i, v in vend.iterrows()],
        "round_shift": {
            "r1": r(df[df["collect_round"] == 1]["pe_ratio"].mean(), 4),
            "r2": r(df[df["collect_round"] == 2]["pe_ratio"].mean(), 4),
        },
        "scatter": sample(df[df["uecc"] >= 0], ["pe_ratio", "uecc"], 1500, seed=3),
    }


# --------------------------------------------------------------- relylab
def relylab() -> dict:
    d = P / "relylab"
    params = pd.read_csv(d / "data" / "wafer_params.csv")
    tools = pd.read_csv(d / "data" / "tool_history.csv")
    comp = json.loads((d / "outputs" / "model_comparison.json").read_text(encoding="utf-8"))
    attr = pd.read_csv(d / "outputs" / "tool_attribution.csv")
    schema = json.loads((d / "data" / "schema.json").read_text(encoding="utf-8"))
    return {
        "meta": {"wafers": int(len(params)), "lots": int(params["Lot_ID"].nunique()),
                 "tests": int(len(schema["reliability_tests"])) * int(len(params)),
                 "overall_pass": r(params["overall_pass"].mean(), 4)},
        "per_test": comp["per_test"],
        "detection": comp["tool_detection"],
        "hidden": comp["hidden_effect"],
        "attribution": [
            {"module": t.module, "tool": t.tool, "lots": int(t.n_lots), "pass": r(t.pass_rate, 4),
             "other": r(t.other_tools_pass_rate, 4), "delta": r(t.delta, 4), "p": float(t.p_value),
             "flagged": bool(t.flagged), "truth": bool(t.is_true_rogue)}
            for t in attr.itertuples()
        ],
        "rogue": list(schema["ground_truth_rogue_tools"]),
        "modules": schema["module_tools"],
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    data = {
        "generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%d"),
        "repo": "https://github.com/MinJaeJun/semicon-ai-lab",
        "yieldlens": yieldlens(),
        "etchpilot": etchpilot(),
        "diceguard": diceguard(),
        "cellhealth": cellhealth(),
        "relylab": relylab(),
    }
    out = SITE / "data.json"
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{out}  {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
