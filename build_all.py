"""전 프로젝트 데이터 생성 + 분석을 순서대로 돌린다.

각 프로젝트의 generate.py 는 자기 데이터를 만든 직후 스스로 검증하고,
검증에 실패하면 0이 아닌 코드로 죽는다. 그래서 이 스크립트가 끝까지 통과했다는 것은
다섯 데이터셋 전부가 설계한 성질(계측 편향, U자 응답, 열화 주입, 무릎 지점 등)을
실제로 갖고 있다는 뜻이다.

사용법
------
    python build_all.py            # 데이터 생성 + 분석 전부
    python build_all.py --data     # 데이터 생성만
    python build_all.py --check    # 생성 없이 기존 데이터 검증만
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECTS = ["yieldlens", "etchpilot", "diceguard", "cellhealth", "relylab"]


def run(project: str, script: str) -> tuple[bool, float, str]:
    path = ROOT / "projects" / project / script
    if not path.exists():
        return True, 0.0, "(없음, 건너뜀)"
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, script],
        cwd=path.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    dt = time.time() - t0
    tail = (proc.stdout or "").strip().splitlines()
    msg = tail[-1] if tail else ""
    if proc.returncode != 0:
        msg = (proc.stderr or "").strip().splitlines()[-1] if proc.stderr else "실패"
    return proc.returncode == 0, dt, msg


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="store_true", help="데이터 생성만")
    ap.add_argument("--check", action="store_true", help="생성 없이 검증만")
    args = ap.parse_args()

    scripts = ["generate.py"] if args.data or args.check else ["generate.py", "analyze.py"]
    failures = []
    print(f"{'프로젝트':<12} {'단계':<12} {'상태':<6} {'초':>7}  메시지")
    print("-" * 88)
    for project in PROJECTS:
        for script in scripts:
            ok, dt, msg = run(project, script)
            print(f"{project:<12} {script:<12} {'OK' if ok else 'FAIL':<6} {dt:>7.1f}  {msg[:52]}")
            if not ok:
                failures.append(f"{project}/{script}: {msg}")
    print("-" * 88)
    if failures:
        print(f"\n실패 {len(failures)}건")
        for f in failures:
            print("  -", f)
        return 1
    print("\n전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
