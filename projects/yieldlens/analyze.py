"""YieldLens — 원인 인자 스크리닝 + 권장 운전구간 도출 + 수율 예측.

파이프라인
---------
1. 스크리닝 : 46개 계측 인자 x 6개 타깃을 전수 Spearman 검정하고
              Benjamini-Hochberg FDR 보정으로 다중검정 거짓양성을 걷어낸다.
2. 형태 판정: 살아남은 인자에 2차/1차 적합을 붙여 U자인지 단조인지 가른다.
              U자면 최적 중심과 권장 구간을, 단조면 방향만 낸다.
3. 수율 예측: HistGradientBoosting 으로 Yield_pct 를 맞춘다.
              결측을 채우지 않고 그대로 넘긴다. 결측 자체가 정보이기 때문이다
              (계측 안 한 웨이퍼는 애초에 안전해 보였던 웨이퍼다).
4. 현업 활용: 절대 오차보다 "위험 웨이퍼 상위 N장을 얼마나 잘 골라내는가" 로 평가한다.
              엔지니어가 실제로 쓰는 방식이 그렇다.

실행: python analyze.py
출력: outputs/screening.csv, control_windows.json, model_report.json, REPORT.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"

RHO_MIN = 0.15
Q_MAX = 0.05
N_MIN = 300


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH 절차로 p값을 q값(FDR)으로 바꾼다."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def screen(df: pd.DataFrame, features: list[str], targets: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        for t in targets:
            m = df[[f, t]].dropna()
            if len(m) < N_MIN:
                continue
            rho, p = stats.spearmanr(m[f], m[t])
            if not np.isfinite(rho):
                continue
            rows.append({"factor": f, "target": t, "n": len(m), "spearman": rho, "p_value": p})
    res = pd.DataFrame(rows)
    res["q_value"] = benjamini_hochberg(res["p_value"].to_numpy())
    res["r2"] = res["spearman"] ** 2
    res["significant"] = (res["spearman"].abs() >= RHO_MIN) & (res["q_value"] <= Q_MAX)
    return res.sort_values("spearman", key=np.abs, ascending=False).reset_index(drop=True)


def specificity(scr: pd.DataFrame) -> pd.DataFrame:
    """인자가 '특정 불량모드'에 붙었는지, 그냥 '전체 수율'에 붙었는지 가른다.

    Lot 건강도가 나쁘면 거의 모든 계측값과 거의 모든 불량률이 같이 나빠진다.
    그래서 상관계수만 보면 무관한 인자도 전부 유의하게 나온다 (교란).

    진짜 원인 인자는 특정 불량모드와의 상관이 전체 수율과의 상관보다 강하다.
    교란으로만 엮인 인자는 반대로 전체 수율 쪽이 더 강하다.
    이 차이(specificity)의 부호로 둘을 가른다.
    """
    mode = scr[scr["target"] != "Yield_pct"]
    idx = mode.groupby("factor")["spearman"].apply(lambda s: s.abs().idxmax())
    best = mode.loc[idx].set_index("factor")
    ycorr = scr[scr["target"] == "Yield_pct"].set_index("factor")["spearman"].abs()
    out = pd.DataFrame(
        {
            "best_target": best["target"],
            "rho_mode": best["spearman"].abs(),
            "rho_yield": ycorr,
            "n": best["n"],
        }
    ).dropna()
    out["specificity"] = out["rho_mode"] - out["rho_yield"]
    out["causal_candidate"] = out["specificity"] > 0
    return out.sort_values("specificity", ascending=False)


def control_window(df: pd.DataFrame, factor: str, target: str, n_bins: int = 10) -> dict:
    """2차 적합으로 최적 중심을 잡고, 분위 구간별 실측 불량률로 권장 범위를 낸다."""
    m = df[[factor, target]].dropna()
    x, y = m[factor].to_numpy(float), m[target].to_numpy(float)
    ss = float(((y - y.mean()) ** 2).sum())
    c2, c1 = np.polyfit(x, y, 2), np.polyfit(x, y, 1)
    r2q = 1 - float(((y - np.polyval(c2, x)) ** 2).sum()) / ss
    r2l = 1 - float(((y - np.polyval(c1, x)) ** 2).sum()) / ss

    bins = pd.qcut(m[factor], n_bins, duplicates="drop")
    prof = m.groupby(bins, observed=True)[target].agg(["mean", "count"])
    best_bin = prof["mean"].idxmin()
    lo, hi = float(best_bin.left), float(best_bin.right)

    shape = "u_shape" if (r2q - r2l) >= 0.01 and c2[0] > 0 else ("monotonic" if abs(r2l) > 0.01 else "flat")
    optimum = float(-c2[1] / (2 * c2[0])) if c2[0] != 0 else float("nan")
    if shape != "u_shape" or not (x.min() <= optimum <= x.max()):
        optimum = float(m.loc[m[target].idxmin(), factor])

    return {
        "factor": factor,
        "target": target,
        "shape": shape,
        "quad_r2": round(r2q, 4),
        "linear_r2": round(r2l, 4),
        "optimum": round(optimum, 3),
        "recommended_low": round(lo, 3),
        "recommended_high": round(hi, 3),
        "observed_min": round(float(x.min()), 3),
        "observed_max": round(float(x.max()), 3),
        "current_median": round(float(np.median(x)), 3),
        "best_bin_fail_rate": round(float(prof["mean"].min()), 3),
        "worst_bin_fail_rate": round(float(prof["mean"].max()), 3),
        "n_measured": int(len(m)),
    }


def fit_yield_model(df: pd.DataFrame, features: list[str]) -> tuple[HistGradientBoostingRegressor, dict]:
    """Lot 단위 GroupKFold. 같은 Lot 이 train/valid 에 섞이면 성능이 뻥튀기된다."""
    x = df[features]
    y = df["Yield_pct"].to_numpy()
    groups = df["Lot_ID"].to_numpy()

    gkf = GroupKFold(n_splits=5)
    maes, r2s, top_hits = [], [], []
    for tr, va in gkf.split(x, y, groups):
        model = HistGradientBoostingRegressor(
            max_iter=320, learning_rate=0.06, max_depth=6, min_samples_leaf=40, random_state=7
        )
        model.fit(x.iloc[tr], y[tr])
        pred = model.predict(x.iloc[va])
        maes.append(float(np.mean(np.abs(pred - y[va]))))
        ss_res = float(((y[va] - pred) ** 2).sum())
        ss_tot = float(((y[va] - y[va].mean()) ** 2).sum())
        r2s.append(1 - ss_res / ss_tot)
        # 예측 하위 30장 중 실제 하위 10% 에 든 비율
        k = 30
        worst_pred = np.argsort(pred)[:k]
        cut = np.percentile(y[va], 10)
        top_hits.append(float((y[va][worst_pred] <= cut).mean()))

    final = HistGradientBoostingRegressor(
        max_iter=320, learning_rate=0.06, max_depth=6, min_samples_leaf=40, random_state=7
    )
    final.fit(x, y)
    metrics = {
        "cv": "GroupKFold(5) by Lot_ID",
        "mae_mean": round(float(np.mean(maes)), 4),
        "mae_std": round(float(np.std(maes)), 4),
        "r2_mean": round(float(np.mean(r2s)), 4),
        "r2_std": round(float(np.std(r2s)), 4),
        "top30_worst_precision": round(float(np.mean(top_hits)), 4),
        "n_features": len(features),
        "n_rows": int(len(df)),
    }
    return final, metrics


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "wafer_history.csv.gz")
    schema = json.loads((DATA / "schema.json").read_text(encoding="utf-8"))

    measured = schema["response_columns"] + schema["defect_columns"]
    targets = schema["targets"]["fail_rates"] + ["Yield_pct"]

    # ---------- 1. 스크리닝
    scr = screen(df, measured, targets)
    scr.to_csv(OUT / "screening.csv", index=False, encoding="utf-8")
    sig = scr[scr["significant"]]
    print(f"스크리닝: {len(scr)}개 조합 검정 -> 유의 {len(sig)}건")
    print(sig.head(12).to_string(index=False))

    # ---------- 1b. 교란 제거: 특이도로 진짜 원인 후보만 남긴다
    spec = specificity(scr)
    spec.to_csv(OUT / "specificity.csv", encoding="utf-8")
    cand = spec[spec["causal_candidate"]]
    truth = {d["factor"] for d in schema["ground_truth_drivers"]}
    hit = truth & set(cand.index)
    detection = {
        "n_significant_raw": int(sig["factor"].nunique()),
        "n_causal_candidates": int(len(cand)),
        "ground_truth_drivers": sorted(truth),
        "detected": sorted(hit),
        "missed": sorted(truth - set(cand.index)),
        "false_positives": sorted(set(cand.index) - truth),
        "precision": round(len(hit) / max(len(cand), 1), 3),
        "recall": round(len(hit) / max(len(truth), 1), 3),
    }
    print(f"\n특이도 필터: 유의 인자 {detection['n_significant_raw']}개 -> 인과 후보 {len(cand)}개")
    print(f"  정답 {sorted(truth)}")
    print(f"  검출 {detection['detected']}  | 누락 {detection['missed']}  | 오탐 {detection['false_positives']}")
    print(f"  precision {detection['precision']}  recall {detection['recall']}")

    # ---------- 2. 권장 운전구간 (인과 후보에 대해서만)
    windows = []
    for f, row in cand.iterrows():
        windows.append(control_window(df, f, row["best_target"]))
    # 같은 인자가 여러 타깃에 걸리면 상관이 가장 강한 것만 남긴다
    seen: dict[str, dict] = {}
    for w in windows:
        key = w["factor"]
        if key not in seen or w["quad_r2"] > seen[key]["quad_r2"]:
            seen[key] = w
    windows = sorted(seen.values(), key=lambda w: -w["quad_r2"])
    (OUT / "control_windows.json").write_text(
        json.dumps(windows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- 3. 수율 예측
    # 계측 여부 자체를 피처로 넣는다. MNAR 이라 "안 쟀다" 가 정보다.
    feat = df[measured].copy()
    for c in measured:
        feat[f"{c}__measured"] = df[c].notna().astype(int)
    feat["n_measured"] = df[measured].notna().sum(axis=1)
    model_df = pd.concat([df[["Lot_ID", "Yield_pct"]], feat], axis=1)
    features = [c for c in feat.columns]
    _, metrics = fit_yield_model(model_df, features)
    print(f"\n수율 예측: MAE {metrics['mae_mean']:.3f} / R2 {metrics['r2_mean']:.3f} "
          f"/ 위험 Top30 정밀도 {metrics['top30_worst_precision']:.1%}")

    # 계측 여부 피처를 뺀 대조군 — MNAR 정보가 실제로 도움이 되는지 확인
    _, metrics_plain = fit_yield_model(model_df, measured)
    metrics["ablation_without_missing_indicator"] = {
        "mae_mean": metrics_plain["mae_mean"],
        "r2_mean": metrics_plain["r2_mean"],
        "top30_worst_precision": metrics_plain["top30_worst_precision"],
    }
    metrics["driver_detection"] = detection
    (OUT / "model_report.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- 4. 리포트
    lines = [
        "# YieldLens 분석 리포트",
        "",
        f"- 데이터: {len(df):,}행 x {df.shape[1]}열, Lot {df['Lot_ID'].nunique()}개",
        f"- 계측 인자 {len(measured)}개 x 타깃 {len(targets)}개 = {len(scr)}회 검정",
        f"- 유의 판정 기준: |rho| >= {RHO_MIN}, BH q <= {Q_MAX}, n >= {N_MIN}",
        "",
        "## 1. 검출된 원인 인자",
        "",
        "| 인자 | 타깃 | n | Spearman | r2 | q |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, r in sig[sig["target"] != "Yield_pct"].head(10).iterrows():
        lines.append(
            f"| `{r['factor']}` | {r['target']} | {r['n']:,} | {r['spearman']:+.3f} "
            f"| {r['r2']:.3f} | {r['q_value']:.1e} |"
        )
    lines += [
        "",
        "## 1b. 교란 제거 (specificity)",
        "",
        f"상관만 보면 {detection['n_significant_raw']}개 인자가 유의하다. Lot 건강도가 나쁘면",
        "계측값도 불량률도 같이 나빠지기 때문에 무관한 인자까지 전부 딸려 나온다.",
        "특정 불량모드 상관에서 전체 수율 상관을 뺀 값(specificity)이 양수인 것만 남기면:",
        "",
        f"- 인과 후보 **{detection['n_causal_candidates']}개**",
        f"- 정답 대비 precision **{detection['precision']}**, recall **{detection['recall']}**",
        f"- 검출: {', '.join('`' + x + '`' for x in detection['detected']) or '-'}",
        f"- 누락: {', '.join('`' + x + '`' for x in detection['missed']) or '없음'}",
        f"- 오탐: {', '.join('`' + x + '`' for x in detection['false_positives']) or '없음'}",
        "",
        "## 2. 권장 운전구간",
        "",
        "| 인자 | 형태 | 최적 중심 | 권장 구간 | 현재 중앙값 | 조정 방향 |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for w in windows[:10]:
        direction = "유지"
        if w["current_median"] > w["recommended_high"]:
            direction = "낮춰야 함"
        elif w["current_median"] < w["recommended_low"]:
            direction = "높여야 함"
        lines.append(
            f"| `{w['factor']}` | {w['shape']} | {w['optimum']} "
            f"| {w['recommended_low']} ~ {w['recommended_high']} | {w['current_median']} | {direction} |"
        )
    lines += [
        "",
        "## 3. 수율 예측 성능",
        "",
        f"- 교차검증: {metrics['cv']}",
        f"- MAE {metrics['mae_mean']:.3f} +- {metrics['mae_std']:.3f} %p",
        f"- R2 {metrics['r2_mean']:.3f} +- {metrics['r2_std']:.3f}",
        f"- 위험 웨이퍼 Top30 정밀도 {metrics['top30_worst_precision']:.1%}",
        "",
        "결측 지시자 제거 시(대조군):",
        f"- MAE {metrics_plain['mae_mean']:.3f} / R2 {metrics_plain['r2_mean']:.3f} "
        f"/ Top30 정밀도 {metrics_plain['top30_worst_precision']:.1%}",
        "",
        "R2 절대값이 낮은 것은 데이터가 그렇게 설계됐기 때문이다. 계측률이 22% 라",
        "웨이퍼 한 장의 수율을 정확히 맞추는 건 애초에 불가능하다. 이 모델의 쓰임새는",
        "절대값 예측이 아니라 **검사 우선순위 정렬**이고, 그 지표가 Top30 정밀도다.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
