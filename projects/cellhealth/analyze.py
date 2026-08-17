"""CellHealth — 자연어 질의 엔진 + 마모 위험 예측.

두 개를 만든다.

1) 자연어 질의 엔진
   LLM 을 부르지 않는다. 컬럼 사전과 규칙만으로 질문을 파싱해 집계 계획을 세우고
   pandas 로 실행한다. 이유는 정확성이다. 품질 데이터에서 숫자가 틀리면
   도구로서 가치가 0 이 된다. LLM 에 표를 통째로 넘기면 그 순간 검증이 불가능해진다.
   같은 이유로 엔진은 항상 "어떤 컬럼을 썼고, 몇 행을 봤는지" 를 같이 돌려준다.

2) 마모 위험 예측
   pe_ratio 0.70 부근에 무릎이 있다. 선형 모델은 이 지점을 못 잡는다.
   트리 모델과 선형 모델을 같은 조건에서 비교해 그 차이를 수치로 보여준다.
   평가는 PR-AUC 로 한다. 양성이 0.7% 뿐이라 ROC-AUC 는 낙관적으로 보인다.

실행: python analyze.py
출력: outputs/query_results.json, wearout_model.json, REPORT.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"

# 자연어 표현 -> 컬럼. 사전은 데이터와 함께 배포되므로 사람이 검수할 수 있다.
SYNONYMS: dict[str, list[str]] = {
    "pe_cycle": ["pe 사이클", "pe사이클", "p/e", "pe cycle", "소거 횟수", "프로그램 소거"],
    "pe_ratio": ["소모율", "수명 소모", "정격 대비", "pe ratio"],
    "uecc": ["uecc", "정정 불가", "비트 에러", "bit error"],
    "rtbb": ["rtbb", "불량 블록", "배드블록", "bad block"],
    "tbw_gb": ["tbw", "기록량", "총 쓰기", "written"],
    "storage_util": ["사용률", "점유율", "utilization"],
    "storage_used_gb": ["사용 용량", "사용량", "used"],
    "health_pct": ["잔여 수명", "헬스", "health"],
    "temp_max_c": ["온도", "최고 온도", "temp"],
    "power_on_hours": ["전원 시간", "가동 시간", "poh"],
    "density_gb": ["용량", "density"],
    "vendor": ["벤더", "제조사", "회사"],
    "product_family": ["제품군", "패밀리", "family"],
    "collect_round": ["회차", "차수", "round"],
}
AGGS: dict[str, list[str]] = {
    "mean": ["평균", "average", "avg"],
    "median": ["중앙값", "median"],
    "max": ["최대", "최고", "max"],
    "min": ["최소", "min"],
    "sum": ["합계", "총합", "sum"],
    "count": ["개수", "건수", "몇 개", "몇개", "count"],
}
COMPARATORS = [
    (r"(\d+(?:\.\d+)?)\s*(?:이상|초과|넘는|보다\s*큰|>=|>)", "ge"),
    (r"(\d+(?:\.\d+)?)\s*(?:이하|미만|보다\s*작은|<=|<)", "le"),
]


def resolve_column(text: str) -> str | None:
    t = text.lower()
    best, best_len = None, 0
    for col, words in SYNONYMS.items():
        for w in [col] + words:
            if w.lower() in t and len(w) > best_len:
                best, best_len = col, len(w)
    return best


def parse(question: str) -> dict:
    """질문 -> 실행 계획. 못 알아들으면 None 을 채워서 그대로 반환한다."""
    q = question.lower()
    plan: dict = {"question": question, "agg": None, "metric": None, "group_by": None, "filters": []}

    for agg, words in AGGS.items():
        if any(w in q for w in words):
            plan["agg"] = agg
            break

    # "벤더별", "제품군별" 같은 표현에서 그룹 축을 뽑는다
    m = re.search(r"([가-힣a-z_]+)\s*별", q)
    if m:
        plan["group_by"] = resolve_column(m.group(1))

    # 조건: "pe_cycle 이 2000 이상" 형태
    for pattern, op in COMPARATORS:
        for mm in re.finditer(pattern, q):
            head = q[: mm.start()]
            col = resolve_column(head[-30:]) or resolve_column(head)
            if col:
                plan["filters"].append({"column": col, "op": op, "value": float(mm.group(1))})

    # 집계 대상 지표: 그룹/필터에 안 쓰인 컬럼 중 마지막으로 언급된 것
    used = {plan["group_by"]} | {f["column"] for f in plan["filters"]}
    cand = [(q.rfind(w.lower()), c) for c, ws in SYNONYMS.items() for w in [c] + ws if w.lower() in q and c not in used]
    if cand:
        plan["metric"] = max(cand)[1]
    if plan["agg"] == "count":
        plan["metric"] = plan["metric"] or "device_uid"
    return plan


def execute(df: pd.DataFrame, plan: dict) -> dict:
    """계획을 실행하고, 무엇을 어떻게 계산했는지 함께 돌려준다."""
    work = df
    applied = []
    for f in plan["filters"]:
        col, op, val = f["column"], f["op"], f["value"]
        work = work[work[col] >= val] if op == "ge" else work[work[col] <= val]
        applied.append(f"{col} {'>=' if op == 'ge' else '<='} {val:g}")

    agg, metric, gb = plan["agg"] or "mean", plan["metric"], plan["group_by"]
    if metric is None:
        return {"ok": False, "reason": "집계 대상 컬럼을 특정하지 못했다", "plan": plan}

    if agg == "count":
        result = work.groupby(gb).size().rename("count").reset_index() if gb else pd.DataFrame({"count": [len(work)]})
    elif gb:
        result = work.groupby(gb)[metric].agg(agg).reset_index()
    else:
        result = pd.DataFrame({metric: [getattr(work[metric], agg)()]})

    return {
        "ok": True,
        "columns_used": sorted({c for c in [metric, gb] if c} | {f["column"] for f in plan["filters"]}),
        "filters_applied": applied,
        "rows_scanned": int(len(df)),
        "rows_matched": int(len(work)),
        "aggregation": agg,
        "result": json.loads(result.round(4).to_json(orient="records")),
    }


def wearout_model(df: pd.DataFrame) -> dict:
    """정정 불가 에러(uecc>0) 발생 여부를 맞춘다. 양성 0.7% 의 불균형 문제."""
    feats = ["pe_ratio", "pe_cycle", "tbw_gb", "storage_util", "power_on_hours", "temp_max_c", "density_gb", "rtbb"]
    x = df[feats].to_numpy(float)
    y = (df["uecc"] > 0).astype(int).to_numpy()
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=3, stratify=y)

    out = {"positive_rate": round(float(y.mean()), 5), "features": feats, "models": {}}

    sc = StandardScaler().fit(xtr)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(xtr), ytr)
    p_lr = lr.predict_proba(sc.transform(xte))[:, 1]

    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.07, random_state=3).fit(xtr, ytr)
    p_gb = gb.predict_proba(xte)[:, 1]

    for name, p in (("logistic", p_lr), ("gradient_boosting", p_gb)):
        k = max(1, int(len(yte) * 0.02))
        top = np.argsort(-p)[:k]
        out["models"][name] = {
            "pr_auc": round(float(average_precision_score(yte, p)), 4),
            "roc_auc": round(float(roc_auc_score(yte, p)), 4),
            "precision_at_top2pct": round(float(yte[top].mean()), 4),
            "lift": round(float(yte[top].mean() / max(yte.mean(), 1e-9)), 2),
        }

    # 무릎 지점 실측
    bins = pd.cut(df["pe_ratio"], [0, 0.4, 0.55, 0.65, 0.70, 0.75, 0.85, 1.0, 99])
    knee = df.groupby(bins, observed=True).apply(lambda g: (g["uecc"] > 0).mean(), include_groups=False)
    out["uecc_rate_by_pe_ratio"] = {str(k): round(float(v), 5) for k, v in knee.items()}
    return out


QUESTIONS = [
    "벤더별 평균 PE 사이클은?",
    "제품군별 uecc 평균은?",
    "pe_cycle 이 2000 이상인 디바이스 개수는?",
    "소모율이 0.7 이상인 디바이스의 평균 rtbb 는?",
    "벤더별 잔여 수명 중앙값은?",
    "온도 70 이상인 디바이스의 최대 uecc 는?",
    "회차별 평균 소모율은?",
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "device_health.csv.gz")

    results = []
    for q in QUESTIONS:
        plan = parse(q)
        res = execute(df, plan)
        results.append({"question": q, "plan": plan, "response": res})
        head = res["result"][:3] if res["ok"] else res["reason"]
        print(f"Q: {q}\n   -> {head}")
    (OUT / "query_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    answered = sum(1 for r in results if r["response"]["ok"])

    model = wearout_model(df)
    (OUT / "wearout_model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    lr, gb = model["models"]["logistic"], model["models"]["gradient_boosting"]
    print(f"\n마모 예측  로지스틱 PR-AUC {lr['pr_auc']}  |  GBM PR-AUC {gb['pr_auc']} (lift {gb['lift']})")

    lines = [
        "# CellHealth 분석 리포트",
        "",
        f"- 데이터: {len(df):,}행 ({df['device_uid'].nunique():,}대 x {df['collect_round'].nunique()}회차)",
        f"- 자연어 질의 {len(QUESTIONS)}건 중 {answered}건 해석 성공",
        "",
        "## 1. 자연어 질의 엔진",
        "",
        "LLM 을 부르지 않는다. 컬럼 사전 + 규칙 파서로 질문을 실행 계획으로 바꾸고 pandas 로 집계한다.",
        "답변에는 항상 사용한 컬럼, 적용한 필터, 스캔/매칭 행 수를 같이 붙인다. 검증 가능해야 하기 때문이다.",
        "",
        "| 질문 | 사용 컬럼 | 매칭 행 | 결과(일부) |",
        "| --- | --- | ---: | --- |",
    ]
    for r in results:
        res = r["response"]
        if not res["ok"]:
            lines.append(f"| {r['question']} | - | - | 해석 실패: {res['reason']} |")
            continue
        preview = ", ".join(
            f"{list(d.values())[0]}={list(d.values())[-1]}" if len(d) > 1 else f"{list(d.values())[0]}"
            for d in res["result"][:3]
        )
        lines.append(
            f"| {r['question']} | {', '.join('`' + c + '`' for c in res['columns_used'])} "
            f"| {res['rows_matched']:,} | {preview} |"
        )

    lines += [
        "",
        "## 2. 마모 위험 예측 (uecc 발생 여부)",
        "",
        f"양성 비율 {model['positive_rate']:.2%}. 이 정도 불균형에서는 ROC-AUC 가 낙관적으로 보이므로 PR-AUC 를 쓴다.",
        "",
        "| 모델 | PR-AUC | ROC-AUC | 상위 2% 정밀도 | Lift |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| 로지스틱 회귀 | {lr['pr_auc']} | {lr['roc_auc']} | {lr['precision_at_top2pct']:.3f} | {lr['lift']} |",
        f"| Gradient Boosting | {gb['pr_auc']} | {gb['roc_auc']} | {gb['precision_at_top2pct']:.3f} | {gb['lift']} |",
        "",
        "## 3. 마모 무릎 지점",
        "",
        "| pe_ratio 구간 | uecc 발생률 |",
        "| --- | ---: |",
    ]
    for k, v in model["uecc_rate_by_pe_ratio"].items():
        lines.append(f"| {k} | {v:.2%} |")
    lines += [
        "",
        "정격 대비 70% 를 넘는 순간 발생률이 자릿수 단위로 뛴다. 잔여수명을 선형 외삽하면 안 되는 이유다.",
        "",
        (
            f"다만 이 데이터에서는 두 모델의 PR-AUC 차이가 크지 않다 "
            f"(로지스틱 {lr['pr_auc']} vs GBM {gb['pr_auc']}). "
            "핵심 인자인 pe_ratio 하나가 위험도를 거의 단조적으로 결정하기 때문이다. "
            "무릎이 있다고 해서 자동으로 트리 모델이 이기는 것은 아니라는 뜻이고, "
            "실제로 확인해 보기 전에는 단정하면 안 된다."
        ),
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
