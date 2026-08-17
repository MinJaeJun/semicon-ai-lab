"""EtchPilot — 레시피 품질 예측 + 순위화 + 파라미터 조정 추천.

하는 일
------
1. Site 단위 예측기: 레시피 파라미터 + 사이트 좌표(반경/각도/존) + 장비/챔버 를 넣고
   Top/Mid/Bottom CD 와 Depth 를 예측한다. 웨이퍼 평균이 아니라 사이트를 맞춘 뒤
   집계하는 이유는, 같은 평균이라도 산포가 다르면 전혀 다른 레시피이기 때문이다.
2. 품질 점수: Pass Rate 60% + Uniformity 25% + 목표 근접도 15%.
   Uniformity 는 사이트 간 변동계수를 쓴다.
3. 레시피 순위: 실측 기준으로 12개 리비전을 줄 세운다.
4. OFAT 추천: 파라미터를 하나씩만 흔들어 예측 점수가 얼마나 오르는지 본다.
   현장에서 실제로 쓰는 방식이다. 한 번에 여러 개를 바꾸면 원인을 못 가린다.

평가는 리비전 단위 LeaveOneGroupOut 으로 한다. 같은 리비전 웨이퍼가 train/test 에
섞이면 "레시피를 외운" 모델이 되어 성능이 뻥튀기된다.

실행: python analyze.py
출력: outputs/model_report.json, recipe_ranking.csv, ofat_recommendation.json, REPORT.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"

TARGETS = ["Top_CD_nm", "Mid_CD_nm", "Bottom_CD_nm"]
ZONE_CODE = {"Center": 0, "Mid": 1, "Edge": 2, "Extreme Edge": 3}

W_PASS, W_UNIF, W_PROX = 0.60, 0.25, 0.15
UNIF_TOLERANCE_PCT = 3.0  # 이 변동계수를 넘으면 Uniformity 점수 0
PROX_TOLERANCE = 0.05  # 목표 대비 +-5% 벗어나면 근접도 0


def load(process: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    schema = json.loads((DATA / "schema.json").read_text(encoding="utf-8"))
    sites = pd.read_csv(DATA / f"{process}_sites.csv")
    master = pd.read_csv(DATA / f"{process}_recipe_master.csv")
    return sites, master, schema["processes"][process]


def feature_frame(sites: pd.DataFrame, master: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    pcols = [c for c in master.columns if c not in ("Recipe_Version", "Rev_No", "Changed_Params")]
    df = sites.merge(master[["Rev_No"] + pcols], on="Rev_No", how="left")
    df["zone_code"] = df["Zone"].map(ZONE_CODE)
    df["angle_sin"] = np.sin(np.deg2rad(df["Angle_deg"]))
    df["angle_cos"] = np.cos(np.deg2rad(df["Angle_deg"]))
    df["tool_code"] = pd.factorize(df["Tool_ID"])[0]
    df["chamber_code"] = pd.factorize(df["Chamber_ID"])[0]
    feats = pcols + ["Radius_frac", "zone_code", "angle_sin", "angle_cos", "tool_code", "chamber_code"]
    return df, feats


def evaluate(df: pd.DataFrame, feats: list[str], target: str) -> dict:
    """리비전 단위 LOGO. 못 본 레시피에 대해 얼마나 맞추는지가 진짜 성능이다."""
    x, y, g = df[feats], df[target].to_numpy(), df["Rev_No"].to_numpy()
    logo = LeaveOneGroupOut()
    maes, r2s = [], []
    for tr, te in logo.split(x, y, g):
        m = RandomForestRegressor(n_estimators=180, min_samples_leaf=4, n_jobs=-1, random_state=11)
        m.fit(x.iloc[tr], y[tr])
        p = m.predict(x.iloc[te])
        maes.append(float(np.mean(np.abs(p - y[te]))))
        ss_tot = float(((y[te] - y[te].mean()) ** 2).sum())
        r2s.append(1 - float(((y[te] - p) ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan)
    return {
        "target": target,
        "cv": "LeaveOneGroupOut by Rev_No",
        "mae_mean": round(float(np.mean(maes)), 4),
        "mae_worst": round(float(np.max(maes)), 4),
        "r2_mean": round(float(np.nanmean(r2s)), 4),
        "baseline_mae": round(float(np.mean(np.abs(y - np.median(y)))), 4),
    }


def quality_score(g: pd.DataFrame, spec: dict, depth_col: str) -> pd.Series:
    """웨이퍼 한 장의 종합 품질 점수 (0~100)."""
    pass_cols = ["Top_CD_Pass", "Mid_CD_Pass", "Bottom_CD_Pass", "Depth_Pass"]
    pass_rate = float(g[pass_cols].to_numpy().mean())

    unif_scores = []
    prox_scores = []
    for key, col in zip(("top_cd", "mid_cd", "bottom_cd", "depth"), TARGETS + [depth_col]):
        v = g[col].to_numpy(float)
        cv_pct = float(v.std(ddof=0) / max(abs(v.mean()), 1e-9) * 100)
        unif_scores.append(max(0.0, 1.0 - cv_pct / UNIF_TOLERANCE_PCT))
        lo, hi = spec[key]
        target = (lo + hi) / 2
        err = abs(v.mean() - target) / max(abs(target), 1e-9)
        prox_scores.append(max(0.0, 1.0 - err / PROX_TOLERANCE))

    score = 100.0 * (W_PASS * pass_rate + W_UNIF * float(np.mean(unif_scores)) + W_PROX * float(np.mean(prox_scores)))
    return pd.Series(
        {
            "pass_rate": round(pass_rate, 4),
            "uniformity": round(float(np.mean(unif_scores)), 4),
            "target_proximity": round(float(np.mean(prox_scores)), 4),
            "defect_per_site": round(float(g["Defect_Count"].mean()), 4),
            "quality_score": round(score, 2),
        }
    )


def ofat(df: pd.DataFrame, feats: list[str], master: pd.DataFrame, spec: dict, depth_col: str, base_rev: int) -> list[dict]:
    """파라미터 하나씩만 흔들어보고 예상 개선폭을 낸다."""
    pcols = [c for c in master.columns if c not in ("Recipe_Version", "Rev_No", "Changed_Params")]
    models = {}
    for t in TARGETS + [depth_col]:
        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=4, n_jobs=-1, random_state=11)
        m.fit(df[feats], df[t])
        models[t] = m

    base_rows = df[df["Rev_No"] == base_rev].copy()
    if base_rows.empty:
        return []
    template = base_rows.drop_duplicates("Site_ID").copy()

    def predict_score(frame: pd.DataFrame) -> float:
        pred = frame.copy()
        for t, m in models.items():
            pred[t] = m.predict(frame[feats])
        for key, col in zip(("top_cd", "mid_cd", "bottom_cd", "depth"), TARGETS + [depth_col]):
            lo, hi = spec[key]
            pred[{"top_cd": "Top_CD_Pass", "mid_cd": "Mid_CD_Pass", "bottom_cd": "Bottom_CD_Pass", "depth": "Depth_Pass"}[key]] = (
                pred[col].between(lo, hi)
            )
        pred["Defect_Count"] = frame["Defect_Count"].to_numpy()
        return float(quality_score(pred, spec, depth_col)["quality_score"])

    base_score = predict_score(template)
    out = []
    for p in pcols:
        cur = float(template[p].iloc[0])
        step = max(abs(cur) * 0.06, 1e-6)
        for direction, delta in (("증가", step), ("감소", -step)):
            trial = template.copy()
            trial[p] = cur + delta
            s = predict_score(trial)
            out.append(
                {
                    "parameter": p,
                    "direction": direction,
                    "current": round(cur, 3),
                    "proposed": round(cur + delta, 3),
                    "base_score": round(base_score, 2),
                    "predicted_score": round(s, 2),
                    "gain": round(s - base_score, 2),
                }
            )
    return sorted(out, key=lambda d: -d["gain"])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    schema = json.loads((DATA / "schema.json").read_text(encoding="utf-8"))

    all_metrics: dict[str, list] = {}
    all_rank: list[pd.DataFrame] = []
    all_ofat: dict[str, list] = {}

    for process in schema["processes"]:
        sites, master, meta = load(process)
        depth_col = meta["depth_column"]
        spec = meta["spec"]
        df, feats = feature_frame(sites, master)

        metrics = [evaluate(df, feats, t) for t in TARGETS + [depth_col]]
        all_metrics[process] = metrics
        print(f"[{process}] " + "  ".join(f"{m['target']} MAE={m['mae_mean']:.3f}(base {m['baseline_mae']:.3f})" for m in metrics))

        rank = (
            df.groupby(["Rev_No", "Recipe_Version", "Wafer_ID"], as_index=False)
            .apply(lambda g: quality_score(g, spec, depth_col), include_groups=False)
            .groupby(["Rev_No", "Recipe_Version"], as_index=False)
            .mean(numeric_only=True)
            .sort_values("quality_score", ascending=False)
        )
        rank.insert(0, "process", process)
        rank["rank"] = np.arange(1, len(rank) + 1)
        all_rank.append(rank)

        worst_rev = int(rank.iloc[-1]["Rev_No"])
        all_ofat[process] = ofat(df, feats, master, spec, depth_col, worst_rev)[:6]

    ranking = pd.concat(all_rank, ignore_index=True)
    ranking.to_csv(OUT / "recipe_ranking.csv", index=False, encoding="utf-8")
    (OUT / "model_report.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "ofat_recommendation.json").write_text(json.dumps(all_ofat, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# EtchPilot 분석 리포트", "", "## 1. Site 단위 예측 성능 (미학습 레시피 대상)", ""]
    lines += ["| 공정 | 타깃 | MAE | 기준선 MAE | 개선율 | R2 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for process, ms in all_metrics.items():
        for m in ms:
            imp = (1 - m["mae_mean"] / m["baseline_mae"]) * 100 if m["baseline_mae"] else 0
            lines.append(
                f"| {process} | {m['target']} | {m['mae_mean']:.3f} | {m['baseline_mae']:.3f} | {imp:+.1f}% | {m['r2_mean']:.3f} |"
            )
    lines += ["", "기준선은 '전체 중앙값으로 찍기'다. LeaveOneGroupOut 이라 모델은 평가 대상 레시피를", "한 번도 본 적이 없다.", "", "## 2. 레시피 품질 순위 (실측)", ""]
    lines += ["| 공정 | 순위 | 리비전 | 품질점수 | Pass Rate | Uniformity | 목표근접 |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for process in schema["processes"]:
        sub = ranking[ranking["process"] == process]
        for _, r in pd.concat([sub.head(2), sub.tail(1)]).iterrows():
            lines.append(
                f"| {process} | {int(r['rank'])} | {r['Recipe_Version']} | {r['quality_score']:.2f} "
                f"| {r['pass_rate']:.3f} | {r['uniformity']:.3f} | {r['target_proximity']:.3f} |"
            )
    lines += ["", "## 3. 최하위 레시피에 대한 OFAT 조정 추천", ""]
    for process, recs in all_ofat.items():
        if not recs:
            continue
        lines.append(f"**{process}** (기준 점수 {recs[0]['base_score']:.2f})")
        lines.append("")
        lines.append("| 파라미터 | 방향 | 현재 | 제안 | 예상 점수 | 개선 |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for r in recs[:4]:
            lines.append(
                f"| `{r['parameter']}` | {r['direction']} | {r['current']} | {r['proposed']} "
                f"| {r['predicted_score']:.2f} | {r['gain']:+.2f} |"
            )
        lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
