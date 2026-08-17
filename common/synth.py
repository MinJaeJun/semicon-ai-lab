"""반도체 공정 합성 데이터 생성용 공통 프리미티브.

다섯 개 프로젝트가 공유하는 "현실적인 팹 데이터가 갖는 성질"을 함수로 모아둔다.

여기 담긴 성질은 실제 양산 데이터를 다뤄본 사람이면 아는 것들이다.

* 공정 파라미터는 최적점을 중심으로 U자(역세모) 형태로 불량률을 만든다.
  단조 증가가 아니다 — 너무 낮아도, 너무 높아도 불량이 난다.
* 계측은 전수로 하지 않는다. 그리고 무작위로 고르지도 않는다.
  이상 징후가 보이는 웨이퍼를 골라서 잰다 (MNAR).
* Lot 단위로 공통 바이어스가 걸리고, 그 위에 Wafer 개별 편차가 얹힌다.
* 스펙 경계에서 값이 잘린다(clipping). 스펙을 벗어난 값은 애초에 기록되지 않거나
  경계값으로 기록된다.
* 웨이퍼는 중심-엣지 방향으로 반경 의존 프로파일을 갖는다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SiteGrid",
    "clip_to_spec",
    "lot_bias",
    "mnar_mask",
    "radial_profile",
    "rng_for",
    "u_shape_response",
]


def rng_for(project: str, salt: str = "") -> np.random.Generator:
    """프로젝트 이름에서 결정적으로 시드를 만든다.

    파이썬 내장 hash() 를 쓰면 안 된다. 문자열 해시는 PYTHONHASHSEED 로 매 프로세스
    무작위화되기 때문에, 같은 코드를 두 번 돌려도 다른 데이터가 나온다.
    실제로 개발 중에 이걸로 한 번 당했다. blake2b 로 고정 해시를 만들어야
    실행 간 재현성이 보장된다.
    """
    digest = hashlib.blake2b(f"{project}|{salt}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big") % (2**32))


# --------------------------------------------------------------------------- 응답 형태
def u_shape_response(
    x: np.ndarray,
    center: float,
    baseline: float,
    curvature: float,
    asymmetry: float = 0.0,
) -> np.ndarray:
    """공정 파라미터 -> 불량 기여도. 최적점에서 최소, 양쪽으로 벌어질수록 증가.

    asymmetry > 0 이면 오른쪽(값이 큰 쪽) 이탈이 더 큰 벌점을 받는다.
    실제 공정에서 흔하다 — 예를 들어 에천트를 덜 넣는 것보다 더 넣는 쪽이 위험하다.
    """
    d = x - center
    penalty = curvature * d**2 + asymmetry * curvature * np.abs(d) * d
    return np.maximum(baseline + penalty, 0.0)


def radial_profile(radius_frac: np.ndarray, edge_delta: float, exponent: float = 2.0) -> np.ndarray:
    """웨이퍼 중심(0) -> 엣지(1) 방향 프로파일. 엣지에서 edge_delta 만큼 벌어진다."""
    return edge_delta * np.power(radius_frac, exponent)


# --------------------------------------------------------------------------- 구조적 노이즈
def lot_bias(n_lots: int, rng: np.random.Generator, heavy_tail: float = 0.18) -> np.ndarray:
    """Lot 단위 공통 바이어스. 대부분 정상이고 일부 Lot 이 크게 나쁘다.

    heavy_tail 비율만큼의 Lot 에 큰 바이어스를 주입한다. 균등 분포로 만들면
    "나쁜 Lot" 이라는 개념 자체가 안 생겨서 Lot 단위 조치 판단을 학습할 수 없다.
    """
    base = np.abs(rng.normal(0, 0.28, n_lots))
    n_bad = max(1, int(n_lots * heavy_tail))
    bad_idx = rng.choice(n_lots, size=n_bad, replace=False)
    base[bad_idx] += rng.uniform(0.7, 2.1, n_bad)
    return base


def clip_to_spec(
    values: np.ndarray,
    lsl: float,
    usl: float,
    rng: np.random.Generator,
    hard_ratio: float = 1.0,
) -> np.ndarray:
    """스펙 경계에서 값을 자른다.

    hard_ratio 는 경계 밖 값 중 실제로 경계값으로 기록되는 비율.
    1.0 이면 전부 경계로 눌리고(= 계측 장비 레인지 한계),
    0.0 이면 그대로 흘러나간다.
    """
    out = values.copy()
    over = out > usl
    under = out < lsl
    if hard_ratio >= 1.0:
        out[over] = usl
        out[under] = lsl
        return out
    pick_over = over & (rng.random(out.shape) < hard_ratio)
    pick_under = under & (rng.random(out.shape) < hard_ratio)
    out[pick_over] = usl
    out[pick_under] = lsl
    return out


def mnar_mask(
    n: int,
    base_rate: float,
    risk_score: np.ndarray,
    rng: np.random.Generator,
    risk_gain: float = 2.4,
) -> np.ndarray:
    """계측 여부 마스크. 위험도가 높은 행일수록 계측될 확률이 올라간다 (MNAR).

    risk_score 는 아무 스케일이나 받아서 내부에서 0~1 로 정규화한다.
    반환 True = 계측함.
    """
    r = np.asarray(risk_score, dtype=float)
    lo, hi = np.nanpercentile(r, [5, 95])
    z = np.clip((r - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    p = np.clip(base_rate * (1.0 + risk_gain * (z - z.mean())), 0.002, 0.98)
    return rng.random(n) < p


# --------------------------------------------------------------------------- 사이트 배치
@dataclass(frozen=True)
class SiteGrid:
    """웨이퍼 계측 사이트 배치.

    zones 는 (존 이름, 사이트 개수, 반경비율) 튜플의 나열.
    각 존 안에서는 각도를 균등 분할하고, 존마다 오프셋을 줘서
    같은 각도에 사이트가 겹쳐 보이지 않게 한다.
    """

    zones: tuple[tuple[str, int, float], ...]

    def build(self) -> list[tuple[str, str, float, float]]:
        """[(site_id, zone, radius_frac, angle_deg), ...]"""
        sites: list[tuple[str, str, float, float]] = []
        for zi, (zone, count, rfrac) in enumerate(self.zones):
            if count == 1:
                sites.append(("C1", zone, rfrac, 0.0))
                continue
            step = 360.0 / count
            offset = (step / 2.0) * (zi % 2)
            for i in range(count):
                sid = f"{zone[:3].upper()}-{i + 1:02d}"
                sites.append((sid, zone, rfrac, round(offset + i * step, 1)))
        return sites

    @property
    def size(self) -> int:
        return sum(c for _, c, _ in self.zones)
