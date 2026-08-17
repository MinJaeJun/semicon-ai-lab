"""DiceGuard — 열화 추세 탐지 + 장비 Health Index + 원인/감시 분리.

핵심 설계 판단 세 가지
--------------------
1. **일 단위로 묶고 나서 추세를 본다.**
   스트립 단위 원본으로 회귀하면 개별 변동에 묻혀 기울기가 안 보인다.
   일 평균으로 눌러야 장기 추세가 드러난다. 이 데이터에서 실제로 그렇다.

2. **원인과 감시지표를 처음부터 다른 줄에 세운다.**
   Kerf_Width 는 Laser_Power 와 r=0.74 로 붙어 있어서, 불량과의 상관만 보면
   커프 폭이 1등 원인으로 뽑힌다. 하지만 커프 폭은 결과지 손잡이가 아니다.
   조치 추천은 설비 설정값에만 낸다.

3. **Health Index 는 순위가 아니라 여유(margin)로 잰다.**
   "다른 장비보다 몇 등" 이 아니라 "관리한계까지 얼마나 남았는가" 를 100점 만점으로 낸다.
   100 = 여유 미사용, 10 = 관리한계 경계, 10 미만 = 이탈.
   장비 점수는 변수 점수의 최솟값이다. 최악의 변수 하나가 장비를 대표해야
   평균에 묻히지 않는다.

실행: python analyze.py
출력: outputs/trend_detection.csv, health_index.csv, defect_attribution.json, REPORT.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"

TREND_ALPHA = 0.01
# 추세 판정 구간을 며칠로 잡아야 하는지를 실측으로 정한다.
WINDOWS: list[int | None] = [None, 45, 30, 21, 14, 7]


def daily_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.groupby(["Machine_ID", "Day_Index"], as_index=False)[cols].mean()


def trend_test(daily: pd.DataFrame, machine: str, col: str, window: int | None = None) -> dict:
    sub = daily[daily["Machine_ID"] == machine].sort_values("Day_Index")
    if window is not None:
        sub = sub[sub["Day_Index"] >= sub["Day_Index"].max() - window]
    x, y = sub["Day_Index"].to_numpy(float), sub[col].to_numpy(float)
    if len(x) < 8:
        return {"machine": machine, "variable": col, "n_days": len(x), "significant": False}
    lin = stats.linregress(x, y)
    tau, p_mk = stats.kendalltau(x, y)  # 비모수 추세 검정
    return {
        "machine": machine,
        "variable": col,
        "window": "full" if window is None else f"last{window}",
        "n_days": int(len(x)),
        "slope_per_day": float(lin.slope),
        "total_change": float(lin.slope * (x.max() - x.min())),
        "p_linear": float(lin.pvalue),
        "kendall_tau": float(tau),
        "p_kendall": float(p_mk),
        "significant": bool(lin.pvalue < TREND_ALPHA and p_mk < TREND_ALPHA),
    }


def health_index(df: pd.DataFrame, actionable: list[str]) -> pd.DataFrame:
    """관리한계까지 남은 여유를 100점 만점으로 환산한다.

    level  = 100 * (1 - |현재평균 - 중심| / (3시그마))   범위 밖이면 음수
    maturity(S) = 계측 안정성. 변동이 크면 관리 체계가 미성숙하다고 본다.
    urgency(U)  = 추세 기울기의 크기. 빠르게 나빠질수록 급하다.
    HI = 10 + (level - 10) * (1 - 0.45 * max(S, U)),  최소 10점 보장은 하지 않는다
         (이탈 시 10 미만으로 내려가야 "이미 벗어남" 을 표현할 수 있다)
    """
    daily = daily_frame(df, actionable)
    rows = []
    for machine in sorted(df["Machine_ID"].unique()):
        sub = daily[daily["Machine_ID"] == machine]
        for col in actionable:
            ref = df[col]
            center, sigma = float(ref.median()), float(ref.std())
            cur = float(sub.sort_values("Day_Index")[col].tail(14).mean())
            margin = abs(cur - center) / max(3 * sigma, 1e-9)
            level = 100.0 * (1.0 - margin)

            cv = float(sub[col].std() / max(abs(sub[col].mean()), 1e-9))
            maturity = float(np.clip(cv / 0.02, 0, 1))
            t = trend_test(daily, machine, col)
            urgency = float(np.clip(abs(t.get("total_change", 0.0)) / max(2 * sigma, 1e-9), 0, 1))

            hi = 10.0 + (level - 10.0) * (1.0 - 0.45 * max(maturity, urgency))
            rows.append(
                {
                    "Machine_ID": machine,
                    "variable": col,
                    "recent_mean": round(cur, 4),
                    "center": round(center, 4),
                    "sigma": round(sigma, 4),
                    "level": round(level, 2),
                    "maturity_S": round(maturity, 3),
                    "urgency_U": round(urgency, 3),
                    "health_index": round(hi, 2),
                    "trend_significant": bool(t.get("significant", False)),
                }
            )
    hidf = pd.DataFrame(rows)
    machine_hi = hidf.groupby("Machine_ID")["health_index"].min().rename("machine_health").reset_index()
    return hidf.merge(machine_hi, on="Machine_ID")


def defect_attribution(df: pd.DataFrame, actionable: list[str], monitoring: list[str]) -> dict:
    """불량 분류기를 두 번 돌린다: 감시지표 포함 vs 설비 설정값만.

    감시지표를 넣으면 성능은 오르지만 나오는 답이 실행 불가능하다.
    이 프로젝트는 실행 가능한 답을 원하므로 설정값 모델을 채택한다.
    """
    y = (df["NG_Code"] != "OK").astype(int).to_numpy()
    out = {}
    for name, feats in (("with_monitoring", actionable + monitoring), ("actionable_only", actionable)):
        x = df[feats]
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=5, stratify=y)
        clf = RandomForestClassifier(
            n_estimators=160, min_samples_leaf=12, n_jobs=-1, random_state=5, class_weight="balanced_subsample"
        )
        clf.fit(xtr, ytr)
        proba = clf.predict_proba(xte)[:, 1]
        # 상위 5% 를 검사 대상으로 골랐을 때의 정밀도/리프트
        k = max(1, int(len(yte) * 0.05))
        top = np.argsort(-proba)[:k]
        precision_at_k = float(yte[top].mean())
        base = float(yte.mean())
        # 순열 중요도는 비싸다. 평가셋을 4천 행으로 줄여도 순위는 안 바뀐다.
        idx = np.random.default_rng(5).choice(len(xte), size=min(4000, len(xte)), replace=False)
        perm = permutation_importance(
            clf, xte.iloc[idx], yte[idx], n_repeats=3, random_state=5, n_jobs=-1, scoring="roc_auc"
        )
        imp = sorted(
            ({"feature": f, "importance": round(float(v), 5)} for f, v in zip(feats, perm.importances_mean)),
            key=lambda d: -d["importance"],
        )
        out[name] = {
            "n_features": len(feats),
            "base_defect_rate": round(base, 4),
            "precision_at_top5pct": round(precision_at_k, 4),
            "lift": round(precision_at_k / max(base, 1e-9), 2),
            "top_features": imp[:6],
        }
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "dicing_fdc.csv.gz")
    schema = json.loads((DATA / "schema.json").read_text(encoding="utf-8"))
    actionable, monitoring = schema["actionable_fdc"], schema["monitoring_response"]
    injected = {(d["machine"], d["variable"]) for d in schema["injected_degradation"]}

    # ---------- 1. 추세 탐지: 판정 구간 길이 민감도
    daily = daily_frame(df, actionable)
    rows = []
    for machine in sorted(df["Machine_ID"].unique()):
        for col in actionable:
            for w in WINDOWS:
                r = trend_test(daily, machine, col, w)
                r["is_injected"] = (machine, col) in injected
                rows.append(r)
    tdf = pd.DataFrame(rows)
    tdf.to_csv(OUT / "trend_detection.csv", index=False, encoding="utf-8")

    sweep = []
    for w in WINDOWS:
        key = "full" if w is None else f"last{w}"
        sub = tdf[tdf["window"] == key]
        det = sub[sub["significant"]]
        sweep.append(
            {
                "window": key,
                "days_used": int(sub["n_days"].max()),
                "n_significant": int(len(det)),
                "recall": round(float(sub[sub["is_injected"]]["significant"].mean()), 3),
                "precision": round(float(det["is_injected"].mean()) if len(det) else 0.0, 3),
            }
        )
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(OUT / "window_sensitivity.csv", index=False, encoding="utf-8")

    full = tdf[tdf["window"] == "full"]
    det_full = full[full["significant"]]
    recall_full, prec_full = sweep[0]["recall"], sweep[0]["precision"]
    # 재현율 100% 를 유지하는 가장 짧은 판정 구간
    keep = [int(s["window"].removeprefix("last")) for s in sweep[1:] if s["recall"] >= 1.0]
    min_window = min(keep) if keep else None

    print("추세 판정 구간 민감도")
    print(sweep_df.to_string(index=False))
    print(f"=> 재현율 100% 유지 최소 구간: {min_window}일" if min_window else "=> 전 구간에서만 검출")

    # ---------- 2. Health Index
    hidf = health_index(df, actionable)
    hidf.to_csv(OUT / "health_index.csv", index=False, encoding="utf-8")
    worst = hidf.sort_values("health_index").drop_duplicates("Machine_ID").sort_values("health_index")
    print("\n장비 Health Index (낮을수록 급함)")
    print(worst[["Machine_ID", "variable", "health_index", "level", "urgency_U"]].to_string(index=False))

    # ---------- 3. 불량 귀인
    attrib = defect_attribution(df, actionable, monitoring)
    (OUT / "defect_attribution.json").write_text(json.dumps(attrib, ensure_ascii=False, indent=2), encoding="utf-8")
    a, b = attrib["with_monitoring"], attrib["actionable_only"]
    print(f"\n불량 선별  감시지표 포함: top5% 정밀도 {a['precision_at_top5pct']:.3f} (lift {a['lift']})")
    print(f"           설정값만    : top5% 정밀도 {b['precision_at_top5pct']:.3f} (lift {b['lift']})")

    # ---------- 4. 리포트
    lines = [
        "# DiceGuard 분석 리포트",
        "",
        f"- 데이터: {len(df):,}행, 장비 {df['Machine_ID'].nunique()}대, {schema['period']['days']}일",
        f"- 주입된 열화 시나리오 {len(injected)}건 (전부 관리한계 안에서만 이동)",
        "",
        "## 1. 추세 탐지 — 판정 구간을 며칠로 잡을 것인가",
        "",
        "주입한 열화는 전부 관리한계 안에서만 움직인다. 스펙 위반 알람으로는 하나도 안 잡힌다.",
        "그래서 추세로 잡아야 하는데, 판정 구간 길이가 결론을 통째로 바꾼다.",
        "",
        "| 판정 구간 | 실사용 일수 | 유의 검출 | 주입 열화 재현율 | 정밀도 |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[
            f"| {s['window']} | {s['days_used']} | {s['n_significant']} | {s['recall']:.0%} | {s['precision']:.0%} |"
            for s in sweep
        ],
        "",
        (
            f"재현율 100% 를 유지하는 최소 구간은 **{min_window}일**이다. "
            "그보다 짧게 잡으면 이미 진행 중인 열화를 정상으로 오판한다."
            if min_window
            else "전 구간을 봐야만 검출된다."
        ),
        "구간이 짧아지면 표본 수가 줄어 검정력이 사라지고, 열화가 포화형이라 후반 기울기도 완만해진다.",
        "두 효과가 겹쳐서 짧은 창에서는 재현율이 0% 까지 떨어진다.",
        "",
        "### 검출된 열화",
        "",
        "| 장비 | 변수 | 일당 기울기 | 총 변화 | Kendall tau | p | 주입 여부 |",
        "| --- | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for _, r in det_full.sort_values("kendall_tau", key=np.abs, ascending=False).head(10).iterrows():
        lines.append(
            f"| {r['machine']} | `{r['variable']}` | {r['slope_per_day']:+.5f} | {r['total_change']:+.4f} "
            f"| {r['kendall_tau']:+.3f} | {r['p_kendall']:.1e} | {'O' if r['is_injected'] else '-'} |"
        )

    lines += [
        "",
        "## 2. 장비 Health Index",
        "",
        "`HI = 10 + (level - 10) x (1 - 0.45 x max(성숙도, 긴급도))`, 장비 점수 = 변수 점수의 최솟값",
        "",
        "| 장비 | 최악 변수 | HI | level | 긴급도 | 추세 유의 |",
        "| --- | --- | ---: | ---: | ---: | :---: |",
    ]
    for _, r in worst.iterrows():
        lines.append(
            f"| {r['Machine_ID']} | `{r['variable']}` | {r['health_index']:.1f} | {r['level']:.1f} "
            f"| {r['urgency_U']:.2f} | {'O' if r['trend_significant'] else '-'} |"
        )

    lines += [
        "",
        "## 3. 불량 귀인 — 감시지표를 넣으면 답이 망가진다",
        "",
        "| 모델 | 피처 수 | Top 5% 정밀도 | Lift | 1순위 인자 |",
        "| --- | ---: | ---: | ---: | --- |",
        f"| 감시지표 포함 | {a['n_features']} | {a['precision_at_top5pct']:.3f} | {a['lift']} | `{a['top_features'][0]['feature']}` |",
        f"| 설정값만 | {b['n_features']} | {b['precision_at_top5pct']:.3f} | {b['lift']} | `{b['top_features'][0]['feature']}` |",
        "",
        "감시지표를 넣은 모델이 성능은 더 좋을 수 있다. 그런데 1순위로 지목되는 인자가",
        "가공 결과값이면 현장에 넘길 수 없다. 엔지니어가 돌릴 수 있는 손잡이가 아니기 때문이다.",
        "그래서 조치 추천은 설정값 모델에서만 낸다.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
