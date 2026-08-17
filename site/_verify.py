"""사이트 정적 검증: data.json 구조와 app.js 가 참조하는 키가 맞는지 확인한다.

브라우저 없이 할 수 있는 최대한을 한다.
1) data.json 파싱 + 필수 키 존재
2) app.js 에서 참조하는 D.<project>.<path> 경로가 실제로 존재하는지
3) 차트에 넘기는 배열 길이가 0 이 아닌지
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8")

data = json.loads((HERE / "data.json").read_text(encoding="utf-8"))
app = (HERE / "app.js").read_text(encoding="utf-8")

fails: list[str] = []

# JS 내장 메서드/속성은 데이터 키가 아니다. 검사 대상에서 뺀다.
JS_BUILTIN = {
    "map", "filter", "find", "forEach", "reduce", "slice", "length", "sort", "join",
    "push", "concat", "some", "every", "at", "includes", "indexOf", "toFixed",
    "toLocaleString", "replace", "split", "querySelector", "querySelectorAll",
    "appendChild", "append", "textContent", "innerHTML", "keys", "values", "entries",
}


def dig(obj, path: list[str]):
    cur = obj
    for k in path:
        if isinstance(cur, dict):
            if k not in cur:
                return None, False
            cur = cur[k]
        else:
            return None, False
    return cur, True


# --- 1. app.js 안의 참조 수집. const y = D.yieldlens 같은 별칭까지 따라간다.
alias: dict[str, str] = {}
for m in re.finditer(r"const\s+([A-Za-z_$][\w$]*)\s*=\s*D\.([a-z]+)\s*[,;]", app):
    alias[m.group(1)] = m.group(2)
for m in re.finditer(r"const\s*\{([^}]*)\}\s*=\s*D\b", app):
    for part in m.group(1).split(","):
        n = part.strip().split(":")[-1].strip()
        if n:
            alias[n] = n

refs: set[tuple[str, str]] = set()
for proj, tail in re.findall(r"\bD\.([a-z]+)((?:\.[a-zA-Z_][\w]*)+)", app):
    refs.add((proj, tail))
for var, proj in alias.items():
    for tail in re.findall(rf"\b{re.escape(var)}((?:\.[a-zA-Z_][\w]*)+)", app):
        refs.add((proj, tail))

for proj, tail in sorted(refs):
    path = [x for x in tail.split(".") if x]
    if proj not in data:
        fails.append(f"D.{proj} 없음")
        continue
    # 배열 원소 속성 접근(x.factor 등)은 첫 단계만 확인한다
    if path[0] in JS_BUILTIN:
        continue
    val, ok = dig(data[proj], path)
    if ok:
        continue
    val, ok = dig(data[proj], path[:1])
    if not ok:
        fails.append(f"D.{proj}.{path[0]} 없음")
        continue
    if isinstance(val, list) and val and isinstance(val[0], dict):
        nxt = path[1] if len(path) > 1 else None
        if nxt and nxt not in JS_BUILTIN and nxt not in val[0]:
            fails.append(f"D.{proj}.{path[0]}[].{nxt} 없음")

# --- 2. 화면에 반드시 그려져야 하는 배열들
checks = [
    ("yieldlens.drivers", data["yieldlens"]["drivers"], 4),
    ("yieldlens.specificity", data["yieldlens"]["specificity"], 20),
    ("diceguard.sweep", data["diceguard"]["sweep"], 6),
    ("diceguard.series", list(data["diceguard"]["series"]), 4),
    ("diceguard.health", data["diceguard"]["health"], 6),
    ("cellhealth.queries", data["cellhealth"]["queries"], 7),
    ("relylab.per_test", data["relylab"]["per_test"], 8),
    ("relylab.attribution", data["relylab"]["attribution"], 6),
]
for name, arr, want in checks:
    if len(arr) < want:
        fails.append(f"{name} 길이 {len(arr)} < {want}")

for d in data["yieldlens"]["drivers"]:
    if not d["points"] or not d["profile"]["x"]:
        fails.append(f"driver {d['factor']} 산점도/프로파일 비어 있음")

for name, p in data["etchpilot"]["processes"].items():
    for k in ("best", "worst"):
        n = len(p["wafer_map"][k]["sites"])
        if n != data["etchpilot"]["sites_per_wafer"]:
            fails.append(f"etchpilot/{name} {k} 사이트 {n}개 (기대 {data['etchpilot']['sites_per_wafer']})")
    if len(p["ranking"]) != len(data["etchpilot"]["revisions"]):
        fails.append(f"etchpilot/{name} 랭킹 {len(p['ranking'])}행")
    if not p["ofat"]:
        fails.append(f"etchpilot/{name} OFAT 비어 있음")

# --- 3. 숫자 sanity
if not (0 < data["cellhealth"]["positive_rate"] < 0.05):
    fails.append("cellhealth positive_rate 이상")
if data["yieldlens"]["detection"]["precision"] > 1 or data["yieldlens"]["detection"]["recall"] > 1:
    fails.append("yieldlens detection 값 이상")

# --- 4. 정적 파일 존재
for f in ("index.html", "app.js", "chart.js", "style.css", "data.json"):
    if not (HERE / f).exists():
        fails.append(f"{f} 없음")

size = sum((HERE / f).stat().st_size for f in ("index.html", "app.js", "chart.js", "style.css", "data.json"))
print(f"참조 경로 {len(refs)}개 검사 · 정적 자산 {size / 1024:.0f} KB")
if fails:
    print(f"\n[FAIL] {len(fails)}건")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("[PASS] 모든 참조 경로와 데이터 구조 정상")
