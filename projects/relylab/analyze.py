"""RelyLab — 신뢰성 불합격의 원인 가르기: 설계 마진인가, 특정 설비인가.

세 단계로 답한다.

1. **파라미터만으로 얼마나 설명되는가**
   소자 파라미터 14개로 시험별 합격 여부를 예측한다. 여기서 안 잡히는 부분이
   "파라미터로는 안 보이는 원인" 이고, 그게 설비일 가능성이 있다.

2. **설비를 넣으면 얼마나 좋아지는가**
   같은 모델에 설비 배정을 추가로 넣고 성능 차이를 잰다.
   유의하게 좋아지면 설비 원인이 실재한다는 뜻이다.

3. **어느 설비인가**
   모듈별로 설비 간 합격률을 비교한다. 표본이 Lot 단위라 웨이퍼를 독립으로 보면
   유의성이 과대평가된다. 그래서 Lot 평균으로 집계한 뒤 검정한다.

실행: python analyze.py
출력: outputs/tool_attribution.csv, model_comparison.json, REPORT.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    schema = json.loads((DATA / "schema.json").read_text(encoding="utf-8"))
    p = pd.read_csv(DATA / "wafer_params.csv")
    r = pd.read_csv(DATA / "reliability_tests.csv")
    t = pd.read_csv(DATA / "tool_history.csv")
    return p, r, t, schema


def cv_auc(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> float:
    """Lot 단위 GroupKFold AUC. 같은 Lot 이 양쪽에 섞이면 성능이 부풀려진다."""
    if len(np.unique(y)) < 2:
        return float("nan")
    aucs = []
    for tr, te in GroupKFold(n_splits=5).split(x, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = RandomForestClassifier(n_estimators=250, min_samples_leaf=5, n_jobs=-1, random_state=9)
        m.fit(x.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(x.iloc[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    params, tests, tools, schema = load()
    spec_cols = list(schema["specs"])
    modules = list(schema["module_tools"])
    rogue_truth = set(schema["ground_truth_rogue_tools"])

    base = params[["Wafer_ID", "Lot_ID"] + spec_cols].merge(tools[["Wafer_ID"] + modules], on="Wafer_ID")
    tool_dummies = pd.get_dummies(base[modules], prefix=modules, drop_first=False).astype(int)

    # ---------- 1~2. 파라미터만 vs 파라미터+설비
    comparison = []
    for test in schema["reliability_tests"]:
        sub = tests[tests["test"] == test][["Wafer_ID", "passed"]]
        d = base.merge(sub, on="Wafer_ID")
        y = (~d["passed"].astype(bool)).astype(int).to_numpy()  # 불합격=1
        g = d["Lot_ID"].to_numpy()
        x_p = d[spec_cols]
        x_pt = pd.concat([d[spec_cols].reset_index(drop=True), tool_dummies.reset_index(drop=True)], axis=1)
        a_p, a_pt = cv_auc(x_p, y, g), cv_auc(x_pt, y, g)
        comparison.append(
            {
                "test": test,
                "fail_rate": round(float(y.mean()), 4),
                "auc_params_only": round(a_p, 4),
                "auc_params_plus_tools": round(a_pt, 4),
                "gain": round(a_pt - a_p, 4),
                "sensitive_params": schema["reliability_tests"][test]["params"],
            }
        )
        print(f"{test:6s} 불합격률 {y.mean():.3f}  AUC 파라미터 {a_p:.3f} -> +설비 {a_pt:.3f} ({a_pt - a_p:+.3f})")

    comp_df = pd.DataFrame(comparison)
    (OUT / "model_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 3. 설비 귀인 (Lot 단위 집계 후 검정)
    wafer_pass = params[["Wafer_ID", "Lot_ID", "overall_pass"]]
    merged = tools[["Wafer_ID"] + modules].merge(wafer_pass, on="Wafer_ID")
    rows = []
    for module in modules:
        lot_level = merged.groupby(["Lot_ID", module], as_index=False)["overall_pass"].mean()
        for tool in sorted(lot_level[module].unique()):
            a = lot_level[lot_level[module] == tool]["overall_pass"]
            b = lot_level[lot_level[module] != tool]["overall_pass"]
            if len(a) < 3 or len(b) < 3:
                continue
            t_stat, p_val = stats.mannwhitneyu(a, b, alternative="less")
            rows.append(
                {
                    "module": module,
                    "tool": tool,
                    "n_lots": int(len(a)),
                    "pass_rate": round(float(a.mean()), 4),
                    "other_tools_pass_rate": round(float(b.mean()), 4),
                    "delta": round(float(a.mean() - b.mean()), 4),
                    "p_value": float(p_val),
                    "flagged": bool(p_val < 0.05),
                    "is_true_rogue": tool in rogue_truth,
                }
            )
    attrib = pd.DataFrame(rows).sort_values("delta")
    attrib.to_csv(OUT / "tool_attribution.csv", index=False, encoding="utf-8")

    flagged = attrib[attrib["flagged"]]
    tp = int(flagged["is_true_rogue"].sum())
    detection = {
        "n_tools_tested": int(len(attrib)),
        "n_flagged": int(len(flagged)),
        "true_rogue_tools": sorted(rogue_truth),
        "flagged_tools": flagged["tool"].tolist(),
        "precision": round(tp / max(len(flagged), 1), 3),
        "recall": round(tp / max(len(rogue_truth), 1), 3),
    }
    print(f"\n설비 귀인: {len(attrib)}대 검정 -> {len(flagged)}대 플래그")
    print(f"  정답 {detection['true_rogue_tools']} / 플래그 {detection['flagged_tools']}")
    print(f"  precision {detection['precision']} recall {detection['recall']}")

    # 문제 설비가 파라미터 판정만으로는 안 잡힌다는 것을 수치로
    hidden = []
    for tool in sorted(rogue_truth):
        module = next(m for m, ts in schema["module_tools"].items() if tool in ts)
        p0 = schema["module_params"][module][0]
        sub = params.merge(tools[["Wafer_ID", module]], on="Wafer_ID")
        hit = sub[sub[module] == tool]
        rest = sub[sub[module] != tool]
        hidden.append(
            {
                "tool": tool,
                "module": module,
                "param": p0,
                "in_spec_rate_on_tool": round(float(hit[f"{p0}__in_spec"].mean()), 4),
                "in_spec_rate_others": round(float(rest[f"{p0}__in_spec"].mean()), 4),
                "reliability_pass_on_tool": round(float(hit["overall_pass"].mean()), 4),
                "reliability_pass_others": round(float(rest["overall_pass"].mean()), 4),
            }
        )

    lines = [
        "# RelyLab 분석 리포트",
        "",
        f"- 웨이퍼 {len(params):,}장 (Lot {params['Lot_ID'].nunique()}개), 신뢰성 시험 {len(tests):,}건",
        f"- 전 시험 합격 웨이퍼 {params['overall_pass'].mean():.1%}",
        "",
        "## 1. 파라미터만으로는 부족하다",
        "",
        "| 시험 | 불합격률 | AUC (파라미터) | AUC (+설비) | 개선 | 민감 파라미터 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for c in comparison:
        lines.append(
            f"| {c['test']} | {c['fail_rate']:.3f} | {c['auc_params_only']:.3f} | {c['auc_params_plus_tools']:.3f} "
            f"| {c['gain']:+.3f} | {', '.join('`' + p + '`' for p in c['sensitive_params'])} |"
        )
    lines += [
        "",
        f"평균 개선폭 {comp_df['gain'].mean():+.3f}. 설비 정보를 넣으면 예측이 좋아진다는 것은",
        "파라미터 계측만으로는 안 보이는 원인이 설비 쪽에 남아 있다는 뜻이다.",
        "",
        "## 2. 어느 설비인가",
        "",
        "웨이퍼를 독립 표본으로 보면 안 된다. 설비는 Lot 단위로 배정되므로 Lot 평균으로 집계한 뒤",
        "Mann-Whitney U 로 단측 검정했다.",
        "",
        "| 모듈 | 설비 | Lot 수 | 합격률 | 타 설비 | 차이 | p | 플래그 | 실제 문제 설비 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | :---: |",
    ]
    for _, r in attrib.head(6).iterrows():
        lines.append(
            f"| {r['module']} | `{r['tool']}` | {r['n_lots']} | {r['pass_rate']:.3f} | {r['other_tools_pass_rate']:.3f} "
            f"| {r['delta']:+.3f} | {r['p_value']:.1e} | {'O' if r['flagged'] else '-'} | {'O' if r['is_true_rogue'] else '-'} |"
        )
    lines += [
        "",
        f"검출 성능: precision **{detection['precision']}**, recall **{detection['recall']}** "
        f"({detection['n_flagged']}대 플래그 / 정답 {len(rogue_truth)}대)",
        "",
        "## 3. 왜 파라미터 판정으로는 못 잡는가",
        "",
        "| 설비 | 모듈 | 파라미터 | 스펙내 비율(해당 설비) | 스펙내 비율(타 설비) | 신뢰성 합격률(해당) | (타 설비) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for h in hidden:
        lines.append(
            f"| `{h['tool']}` | {h['module']} | `{h['param']}` | {h['in_spec_rate_on_tool']:.3f} "
            f"| {h['in_spec_rate_others']:.3f} | {h['reliability_pass_on_tool']:.3f} | {h['reliability_pass_others']:.3f} |"
        )
    lines += [
        "",
        "파라미터 스펙내 비율은 크게 다르지 않은데 신뢰성 합격률은 확연히 갈린다.",
        "인라인 계측 판정만 보고 있으면 이 설비는 영원히 정상으로 남는다.",
        "설비 배정 이력과 신뢰성 결과를 엮어야만 드러난다.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "model_comparison.json").write_text(
        json.dumps({"per_test": comparison, "tool_detection": detection, "hidden_effect": hidden}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
