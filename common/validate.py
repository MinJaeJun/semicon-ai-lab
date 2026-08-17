"""생성한 데이터셋이 설계 의도대로 나왔는지 검증한다.

데이터 생성기는 조용히 망가진다. 행 수는 맞는데 상관관계가 사라졌다거나,
계측 편향이 반대로 걸렸다거나 하는 식이다. 그래서 각 프로젝트는 생성 직후
여기 함수로 자기 데이터를 검사하고, 실패하면 커밋을 막는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class Report:
    dataset: str
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(CheckResult(name, bool(passed), detail))

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def render(self) -> str:
        lines = [f"[{'PASS' if self.ok else 'FAIL'}] {self.dataset}"]
        for c in self.checks:
            lines.append(f"   {'o' if c.passed else 'X'} {c.name}: {c.detail}")
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise AssertionError("\n" + self.render())


def check_shape(rep: Report, df: pd.DataFrame, rows: int, cols: int) -> None:
    rep.add("shape", df.shape == (rows, cols), f"{df.shape} (기대 {(rows, cols)})")


def check_no_duplicate_ids(rep: Report, df: pd.DataFrame, col: str) -> None:
    dup = int(df[col].duplicated().sum())
    rep.add(f"unique:{col}", dup == 0, f"중복 {dup}건")


def check_sum_identity(rep: Report, df: pd.DataFrame, cols: list[str], total: float, tol: float = 1e-6) -> None:
    s = df[cols].sum(axis=1)
    err = float((s - total).abs().max())
    rep.add("sum_identity", err <= tol, f"max|sum-{total}| = {err:.2e}")


def check_measure_rate(rep: Report, df: pd.DataFrame, cols: list[str], lo: float, hi: float) -> None:
    rate = float(df[cols].notna().to_numpy().mean() * 100)
    rep.add("measure_rate", lo <= rate <= hi, f"{rate:.2f}% (기대 {lo}~{hi}%)")


def check_mnar(rep: Report, df: pd.DataFrame, feature: str, target: str, min_gap: float) -> None:
    """계측된 행의 타깃 평균이 미계측 행보다 유의하게 나쁜지."""
    a = df.loc[df[feature].notna(), target]
    b = df.loc[df[feature].isna(), target]
    if len(a) < 30 or len(b) < 30:
        rep.add(f"mnar:{feature}", False, "표본 부족")
        return
    gap = float(b.mean() - a.mean())
    rep.add(f"mnar:{feature}", gap >= min_gap, f"미계측-계측 수율차 {gap:+.3f} (>= {min_gap})")


def check_relation(
    rep: Report,
    df: pd.DataFrame,
    feature: str,
    target: str,
    min_abs_rho: float,
    max_abs_rho: float = 0.80,
) -> None:
    """유효 인자가 검출 가능하되 너무 깨끗하지는 않은지 확인한다.

    상한을 두는 데 이유가 있다. 합성 데이터는 방심하면 상관계수가 0.9를 넘어가는데
    그런 데이터로 낸 모델 성능은 아무 의미가 없다. 상한을 넘으면 실패시켜서
    잡음 파라미터를 다시 잡게 만든다.
    """
    m = df[[feature, target]].dropna()
    if len(m) < 100:
        rep.add(f"rho:{feature}->{target}", False, f"표본 {len(m)}건")
        return
    rho, p = stats.spearmanr(m[feature], m[target])
    ok = bool(min_abs_rho <= abs(rho) <= max_abs_rho and p < 0.01)
    rep.add(
        f"rho:{feature}->{target}",
        ok,
        f"rho={rho:+.3f} p={p:.1e} n={len(m)} (허용 |rho| {min_abs_rho}~{max_abs_rho})",
    )


def check_u_shape(rep: Report, df: pd.DataFrame, feature: str, target: str, min_gain: float = 0.02) -> None:
    """2차 적합이 1차보다 확실히 나은지 = U자 형태인지."""
    m = df[[feature, target]].dropna()
    if len(m) < 200:
        rep.add(f"u_shape:{feature}", False, f"표본 {len(m)}건")
        return
    x, y = m[feature].to_numpy(float), m[target].to_numpy(float)
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = lambda c: 1 - float(((y - np.polyval(c, x)) ** 2).sum()) / ss  # noqa: E731
    c2, c1 = np.polyfit(x, y, 2), np.polyfit(x, y, 1)
    gain = r2(c2) - r2(c1)
    vertex = -c2[1] / (2 * c2[0]) if c2[0] != 0 else float("nan")
    rep.add(
        f"u_shape:{feature}",
        gain >= min_gain and c2[0] > 0,
        f"quad-lin R2 gain {gain:+.3f}, vertex {vertex:.2f}",
    )


def check_range(rep: Report, df: pd.DataFrame, col: str, lo: float, hi: float) -> None:
    s = df[col].dropna()
    ok = bool(s.min() >= lo and s.max() <= hi)
    rep.add(f"range:{col}", ok, f"{s.min():.3f} ~ {s.max():.3f} (허용 {lo}~{hi})")


def check_categories(rep: Report, df: pd.DataFrame, col: str, expected: set[str]) -> None:
    got = set(map(str, df[col].dropna().unique()))
    rep.add(f"cats:{col}", got <= expected and len(got) > 0, f"{sorted(got)[:8]}")
