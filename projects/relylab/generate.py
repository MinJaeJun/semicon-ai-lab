"""RelyLab — FinFET 소자 파라미터 + 신뢰성 시험 + 설비 이력 데이터셋 생성기.

Lot 48개 x Wafer 25장 = 1,200장에 대해
  (1) 14개 소자/공정 파라미터 (TEG 파라메트릭 계측)
  (2) 8종 신뢰성 시험 결과 (수명, 열화율, 합격 여부)
  (3) 각 공정 모듈이 어느 설비에서 처리됐는지
를 만든다.

이 데이터셋의 목적
-----------------
신뢰성 불합격이 났을 때 "설계 마진 문제인가, 특정 설비 문제인가" 를 가르는 것이다.
현업에서 이 둘을 섞어 보면 엉뚱한 대책이 나온다.

그래서 두 종류의 원인을 동시에 심어뒀다.
* 파라미터 원인: 시험마다 민감한 파라미터가 정해져 있다.
  TDDB 는 게이트 산화막 두께, EM 은 배선 폭과 Via CD 에 붙는다.
* 설비 원인: 특정 설비 두 대가 자기가 담당한 모듈의 파라미터를 조용히 틀어놓는다.
  파라미터 값만 보면 스펙 안이라 정상으로 보이지만, 그 설비를 거친 웨이퍼의
  신뢰성 합격률이 유의하게 낮다.

실행: python generate.py
출력: data/wafer_params.csv, data/reliability_tests.csv, data/tool_history.csv, data/schema.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.synth import lot_bias, rng_for  # noqa: E402
from common.validate import Report, check_categories, check_no_duplicate_ids, check_shape  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

N_LOTS = 48
WAFERS_PER_LOT = 25

# 5nm 급 노드 기준 스펙: (LSL, USL, target, bias 계수, sigma, 단위)
SPECS: dict[str, tuple[float, float, float, float, float, str]] = {
    "fin_width_nm": (4.6, 6.4, 5.5, +0.62, 0.24, "nm"),
    "fin_height_nm": (50.0, 58.0, 54.0, +1.05, 1.30, "nm"),
    "gate_length_nm": (14.5, 17.5, 16.0, +0.44, 0.34, "nm"),
    "gate_ox_thk_nm": (0.85, 1.15, 1.00, +0.115, 0.042, "nm"),
    "hk_thk_nm": (1.45, 1.85, 1.65, +0.082, 0.062, "nm"),
    "sd_epi_thk_nm": (11.5, 14.5, 13.0, -0.66, 0.50, "nm"),
    "contact_cd_nm": (14.0, 17.0, 15.5, +0.50, 0.42, "nm"),
    "m1_width_nm": (17.0, 21.0, 19.0, -0.28, 0.58, "nm"),
    "m1_space_nm": (17.0, 21.0, 19.0, -0.42, 0.52, "nm"),
    "via_cd_nm": (15.5, 19.5, 17.5, +0.26, 0.44, "nm"),
    "vt_n_V": (0.205, 0.275, 0.240, +0.033, 0.0105, "V"),
    "vt_p_V": (-0.285, -0.215, -0.250, -0.026, 0.0098, "V"),
    "idsat_uA_um": (820.0, 1020.0, 920.0, -52.0, 26.0, "uA/um"),
    "ss_mV_dec": (60.0, 72.0, 66.0, +3.6, 1.8, "mV/dec"),
}

# 공정 모듈 -> 처리 가능 설비
MODULE_TOOLS: dict[str, list[str]] = {
    "Fin Etch": ["ETCH-01", "ETCH-02", "ETCH-03"],
    "Gate Oxide": ["FURN-01", "FURN-02"],
    "HK/MG": ["ALD-01", "ALD-02"],
    "S/D Epi": ["EPI-01", "EPI-02"],
    "Contact": ["ETCH-04", "ETCH-05"],
    "M1 BEOL": ["PVD-01", "PVD-02"],
    "Via": ["ETCH-06", "ETCH-07"],
}
# 모듈이 실제로 결정하는 파라미터
MODULE_PARAMS: dict[str, list[str]] = {
    "Fin Etch": ["fin_width_nm", "fin_height_nm"],
    "Gate Oxide": ["gate_ox_thk_nm"],
    "HK/MG": ["hk_thk_nm", "vt_n_V", "vt_p_V"],
    "S/D Epi": ["sd_epi_thk_nm", "idsat_uA_um"],
    "Contact": ["contact_cd_nm"],
    "M1 BEOL": ["m1_width_nm", "m1_space_nm"],
    "Via": ["via_cd_nm"],
}
# 조용히 틀어져 있는 설비: (설비, 흔드는 세기 sigma 배수)
ROGUE_TOOLS = {"FURN-02": 1.05, "PVD-02": 1.00}

# 신뢰성 시험: (민감 파라미터, 수명 기준, 수명 중심, bias 계수, 열화 중심, 열화 bias, 열화 한계)
TESTS: dict[str, dict] = {
    "HTOL": {"params": ["gate_ox_thk_nm", "hk_thk_nm", "vt_n_V"], "life_c": 2100, "life_b": -380, "life_min": 1000,
             "deg_c": 5.2, "deg_b": 7.5, "deg_max": 20.0},
    "TDDB": {"params": ["gate_ox_thk_nm", "hk_thk_nm", "gate_length_nm"], "life_c": 9.5e6, "life_b": -4.2e6, "life_min": 1e6,
             "deg_c": 3.1, "deg_b": 11.0, "deg_max": 30.0},
    "NBTI": {"params": ["gate_ox_thk_nm", "vt_p_V", "fin_height_nm"], "life_c": 1550, "life_b": -300, "life_min": 800,
             "deg_c": 8.4, "deg_b": 14.0, "deg_max": 25.0},
    "HCI": {"params": ["gate_length_nm", "fin_width_nm", "ss_mV_dec"], "life_c": 1850, "life_b": -340, "life_min": 1000,
            "deg_c": 6.1, "deg_b": 9.5, "deg_max": 20.0},
    "EM": {"params": ["m1_width_nm", "m1_space_nm", "via_cd_nm"], "life_c": 5200, "life_b": -1450, "life_min": 2000,
           "deg_c": 2.2, "deg_b": 4.8, "deg_max": 12.0},
    "ESD": {"params": ["fin_width_nm", "sd_epi_thk_nm", "contact_cd_nm"], "life_c": 3600, "life_b": -780, "life_min": 2000,
            "deg_c": 1.3, "deg_b": 3.1, "deg_max": 10.0},
    "TC": {"params": ["contact_cd_nm", "m1_width_nm", "via_cd_nm"], "life_c": 1050, "life_b": -195, "life_min": 500,
           "deg_c": 4.1, "deg_b": 6.2, "deg_max": 15.0},
    "LTOL": {"params": ["gate_ox_thk_nm", "gate_length_nm", "fin_height_nm"], "life_c": 3100, "life_b": -610, "life_min": 1500,
             "deg_c": 3.2, "deg_b": 5.4, "deg_max": 14.0},
}
FAIL_MODES = ["Oxide BD", "Wear-out", "Metal Migration", "Junction Leak", "Contact Open", "Vt Shift", "Fin Collapse"]


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    rng = rng_for("relylab", "v1")
    n = N_LOTS * WAFERS_PER_LOT

    lot_idx = np.repeat(np.arange(N_LOTS), WAFERS_PER_LOT)
    slot = np.tile(np.arange(1, WAFERS_PER_LOT + 1), N_LOTS)
    lot = np.array([f"RL{i + 1:03d}" for i in lot_idx])
    wafer = np.array([f"{l}W{s:02d}" for l, s in zip(lot, slot)])

    lb = lot_bias(N_LOTS, rng, heavy_tail=0.20)[lot_idx]
    # 캐리어 가장자리 슬롯이 불리하다
    slot_pen = np.where(np.isin(slot, [1, 2, 24, 25]), 0.34, 0.0)
    bias = lb + slot_pen + rng.normal(0, 0.12, n).clip(-0.3, None)

    # 모듈별 설비 배정: Lot 단위로 고정된다 (실제 팹 디스패치와 같다)
    tool_hist = {"Wafer_ID": wafer, "Lot_ID": lot, "Slot": slot}
    rogue_hit = np.zeros(n)
    for module, tools in MODULE_TOOLS.items():
        lot_tool = rng.choice(tools, size=N_LOTS)
        assigned = lot_tool[lot_idx]
        tool_hist[module] = assigned
        for rogue, strength in ROGUE_TOOLS.items():
            if rogue in tools:
                rogue_hit += (assigned == rogue) * strength
    tools_df = pd.DataFrame(tool_hist)

    # 소자 파라미터
    params: dict[str, np.ndarray] = {"Wafer_ID": wafer, "Lot_ID": lot, "Slot": slot}
    for p, (lsl, usl, tgt, bcoef, sig, _unit) in SPECS.items():
        v = tgt + bcoef * bias + rng.normal(0, sig, n)
        # 문제 설비가 자기 모듈 파라미터를 조용히 민다 (스펙은 안 넘는 수준)
        owner = next((m for m, ps in MODULE_PARAMS.items() if p in ps), None)
        if owner:
            for rogue, strength in ROGUE_TOOLS.items():
                if rogue in MODULE_TOOLS[owner]:
                    v = v + (tools_df[owner].to_numpy() == rogue) * strength * sig * np.sign(bcoef or 1.0)
        params[p] = np.round(v, 5)
    pdf = pd.DataFrame(params)
    for p, (lsl, usl, *_rest) in SPECS.items():
        pdf[f"{p}__in_spec"] = pdf[p].between(lsl, usl)

    # 신뢰성 시험
    rows = []
    for test, cfg in TESTS.items():
        # 민감 파라미터가 타깃에서 벗어난 정도를 표준화해 스트레스로 쓴다
        stress = np.zeros(n)
        for p in cfg["params"]:
            lsl, usl, tgt, _b, sig, _u = SPECS[p]
            stress += np.abs(pdf[p].to_numpy() - tgt) / sig
        stress /= len(cfg["params"])
        # 설비 항의 계수를 크게 잡는다. 문제 설비는 두께 같은 계측값으로는 안 잡히는
        # 방식(계면 품질 등)으로도 신뢰성을 갉아먹기 때문이다. 그리고 Lot 이 48개뿐이라
        # 계수가 작으면 Lot 간 편차에 묻혀 검출 자체가 안 된다.
        eff = 0.50 * bias + 0.40 * stress + 0.95 * rogue_hit

        life = cfg["life_c"] + cfg["life_b"] * eff + rng.normal(0, abs(cfg["life_c"]) * 0.08, n)
        deg = cfg["deg_c"] + cfg["deg_b"] * eff * 0.55 + rng.normal(0, cfg["deg_c"] * 0.22, n)
        life = np.maximum(life, 0.0)
        deg = np.maximum(deg, 0.0)
        passed = (life >= cfg["life_min"]) & (deg <= cfg["deg_max"])
        rows.append(
            pd.DataFrame(
                {
                    "Wafer_ID": wafer,
                    "Lot_ID": lot,
                    "test": test,
                    "lifetime": np.round(life, 2),
                    "degradation_pct": np.round(deg, 3),
                    "life_criterion": cfg["life_min"],
                    "deg_criterion": cfg["deg_max"],
                    "passed": passed,
                }
            )
        )
    rdf = pd.concat(rows, ignore_index=True)

    fail_count = rdf.groupby("Wafer_ID")["passed"].apply(lambda s: int((~s).sum()))
    pdf["fail_count"] = pdf["Wafer_ID"].map(fail_count).astype(int)
    pdf["overall_pass"] = pdf["fail_count"] == 0
    pdf["fail_mode"] = np.where(
        pdf["fail_count"] > 0, rng.choice(FAIL_MODES, n), "None"
    )

    schema = {
        "dataset": "RelyLab FinFET reliability",
        "node": "5nm class (합성)",
        "lots": N_LOTS,
        "wafers": int(n),
        "specs": {k: {"lsl": v[0], "usl": v[1], "target": v[2], "unit": v[5]} for k, v in SPECS.items()},
        "module_tools": MODULE_TOOLS,
        "module_params": MODULE_PARAMS,
        "reliability_tests": {k: {"params": v["params"], "life_min": v["life_min"], "deg_max": v["deg_max"]} for k, v in TESTS.items()},
        "fail_modes": FAIL_MODES,
        "ground_truth_rogue_tools": ROGUE_TOOLS,
        "note": "문제 설비는 파라미터를 스펙 안에서만 흔든다. 파라미터 판정만으로는 못 잡고, 신뢰성 합격률과 설비를 엮어야 보인다.",
    }
    return pdf, rdf, tools_df, schema


def validate(pdf: pd.DataFrame, rdf: pd.DataFrame, tools: pd.DataFrame, schema: dict) -> Report:
    rep = Report("relylab")
    n = N_LOTS * WAFERS_PER_LOT
    check_shape(rep, pdf, n, 3 + len(SPECS) * 2 + 3)
    check_shape(rep, rdf, n * len(TESTS), 8)
    check_no_duplicate_ids(rep, pdf, "Wafer_ID")
    check_categories(rep, rdf, "test", set(TESTS))

    overall = float(pdf["overall_pass"].mean())
    rep.add("overall_pass_band", 0.25 <= overall <= 0.85, f"전 시험 합격 웨이퍼 비율 {overall:.3f}")

    per_test = rdf.groupby("test")["passed"].mean()
    rep.add("per_test_band", bool(per_test.min() > 0.45 and per_test.max() < 0.995), f"시험별 합격률 {per_test.min():.3f}~{per_test.max():.3f}")

    # 문제 설비 효과가 실제로 존재하는지
    merged = tools.merge(pdf[["Wafer_ID", "overall_pass"]], on="Wafer_ID")
    for rogue in ROGUE_TOOLS:
        module = next(m for m, ts in MODULE_TOOLS.items() if rogue in ts)
        hit = merged[merged[module] == rogue]["overall_pass"].mean()
        rest = merged[merged[module] != rogue]["overall_pass"].mean()
        rep.add(f"rogue:{rogue}", bool(hit < rest - 0.04), f"합격률 {hit:.3f} vs 정상 설비 {rest:.3f}")

    # 문제 설비가 파라미터를 스펙 밖으로 내보내면 안 된다 (그러면 너무 쉬운 문제가 된다)
    for rogue in ROGUE_TOOLS:
        module = next(m for m, ts in MODULE_TOOLS.items() if rogue in ts)
        p0 = MODULE_PARAMS[module][0]
        sub = pdf.merge(tools[["Wafer_ID", module]], on="Wafer_ID")
        oos = float((~sub[sub[module] == rogue][f"{p0}__in_spec"]).mean())
        rep.add(f"rogue_within_spec:{rogue}", oos < 0.30, f"{p0} 스펙이탈률 {oos:.3f}")
    return rep


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    DATA.mkdir(exist_ok=True)
    pdf, rdf, tools, schema = build()
    rep = validate(pdf, rdf, tools, schema)
    print(rep.render())
    rep.raise_if_failed()
    pdf.to_csv(DATA / "wafer_params.csv", index=False, encoding="utf-8")
    rdf.to_csv(DATA / "reliability_tests.csv", index=False, encoding="utf-8")
    tools.to_csv(DATA / "tool_history.csv", index=False, encoding="utf-8")
    schema["shapes"] = {"wafer_params": list(pdf.shape), "reliability_tests": list(rdf.shape), "tool_history": list(tools.shape)}
    (DATA / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nparams {pdf.shape} / tests {rdf.shape} / tools {tools.shape} -> {DATA}")


if __name__ == "__main__":
    main()
