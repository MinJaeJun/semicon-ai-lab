import { scatter, timeseries, barh, bars, wafer, el, fmt } from "./chart.js";

const D = await (await fetch("data.json")).json();
const $ = (s, r = document) => r.querySelector(s);
const h = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content; };
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const num = (v, d = 3) => (v === null || v === undefined ? "–" : (+v).toFixed(d));
const sci = (p) => (p < 1e-4 ? p.toExponential(1).replace("e", "e") : p.toFixed(4));

const PAGES = [
  ["home", "개요"],
  ["yieldlens", "YieldLens"],
  ["etchpilot", "EtchPilot"],
  ["diceguard", "DiceGuard"],
  ["cellhealth", "CellHealth"],
  ["relylab", "RelyLab"],
];

/* ------------------------------------------------------------------ chrome */
const nav = $("#nav");
PAGES.forEach(([id, label]) => {
  const b = document.createElement("button");
  b.textContent = label; b.dataset.id = id;
  b.onclick = () => go(id);
  nav.appendChild(b);
});
function go(id) {
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("on", p.id === "p-" + id));
  nav.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.id === id));
  if (location.hash !== "#" + id) history.replaceState(null, "", "#" + id);
  window.scrollTo({ top: 0 });
}
addEventListener("hashchange", () => go(location.hash.slice(1) || "home"));
$("#stamp").textContent = `데이터 생성 ${D.generated} · 이 페이지의 모든 수치는 저장소에서 python build_all.py 로 재현된다.`;

const card = (inner, cls = "") => `<div class="card ${cls}">${inner}</div>`;
const kpi = (v, k, s = "", cls = "") => card(`<div class="kpi"><div class="v ${cls}">${v}</div><div class="k">${k}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`);
function chart(title, sub, render) {
  const box = document.createElement("div");
  box.className = "card pad0";
  const inner = h(`<div class="chartbox"><div class="ct">${title}</div><div class="cs">${sub}</div></div>`).firstElementChild;
  box.appendChild(inner);
  render(inner);
  return box;
}
function table(cols, rows) {
  // 파라미터를 col/row 로 쓴다. c, r 로 두면 바깥의 D 별칭(c = D.cellhealth 등)을 가린다.
  const th = cols.map((col) => `<th class="${col.num ? "num" : ""}">${col.t}</th>`).join("");
  const tr = rows
    .map((row) => `<tr class="${row._hl ? "hl" : ""}">${cols.map((col) => `<td class="${col.num ? "num" : ""}">${row[col.k] ?? ""}</td>`).join("")}</tr>`)
    .join("");
  return `<div class="card pad0" style="padding:4px 6px"><table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>`;
}

/* ------------------------------------------------------------------ home */
{
  const y = D.yieldlens, e = D.etchpilot, g = D.diceguard, c = D.cellhealth, r = D.relylab;
  const totalRows = y.meta.rows + Object.values(e.processes).reduce((a, p) => a + p.meta.rows, 0) + g.meta.rows + c.meta.rows + r.meta.wafers;
  const cards = [
    ["01", "yieldlens", "YieldLens", "24단 공정에서 수율을 떨어뜨리는 진짜 원인 인자 찾기", `교란 제거 후 precision <b>${y.detection.precision}</b> / recall <b>${y.detection.recall}</b>, 오탐 ${y.detection.false_positives.length}`],
    ["02", "etchpilot", "EtchPilot", "Etch 레시피 품질 사전 예측과 파라미터 조정 추천", `미학습 레시피 Top CD <b>−38.7%</b> (기준선 대비)`],
    ["03", "diceguard", "DiceGuard", "스펙 안에서 진행되는 장비 열화 조기 감지", `주입 열화 재현율 <b>100%</b>, 판정 구간 하한 <b>21일</b>`],
    ["04", "cellhealth", "CellHealth", "스토리지 헬스 자연어 질의 + 마모 위험 예측", `위험 선별 lift <b>${c.model.logistic.lift}배</b>, 질의 <b>7/7</b>`],
    ["05", "relylab", "RelyLab", "신뢰성 불합격이 설계 마진 탓인가 특정 설비 탓인가", `설비 정보로 AUC <b>+0.104</b>, 문제 설비 recall <b>${r.detection.recall}</b>`],
  ].map(([n, id, t, d, m]) => `<a class="projcard" href="#${id}"><div class="n">${n}</div><h3>${t}</h3><div class="d">${d}</div><div class="m">${m}</div></a>`).join("");

  $("#p-home").innerHTML = `
    <h1>반도체 제조 데이터로 푸는 다섯 가지 문제</h1>
    <p class="lead">공개 공정 데이터는 구조가 너무 단순해서, 실제 팹 데이터가 갖는 어려움이 전부 빠져 있다.
    그 어려움이 사실 문제의 본질이다. 그래서 그 성질들을 의도적으로 심은 합성 데이터를 만들고,
    그걸 실제로 푸는 코드를 붙였다. <b>정답을 알고 있으니 검출 성능을 정직하게 잴 수 있다.</b></p>

    <div class="grid g4">
      ${kpi(totalRows.toLocaleString(), "총 데이터 행", "5개 데이터셋")}
      ${kpi("100%", "재현 가능", "python build_all.py")}
      ${kpi("0", "외부 데이터", "전부 코드가 생성")}
      ${kpi("4", "검증 실패 경험", "생성기가 스스로 잡음")}
    </div>

    <h2>프로젝트</h2>
    <div class="grid g3">${cards}</div>

    <h2>이 데이터가 일부러 갖고 있는 어려움</h2>
    <div class="grid g2">
      ${card(`<b>계측이 무작위가 아니다</b><p class="tiny" style="margin:6px 0 0">전수 계측이 아니고, 위험해 보이는 것만 골라 잰다(MNAR).
        YieldLens 에서 계측된 웨이퍼의 평균 수율이 미계측보다 <b>4.4%p 낮다</b>. 계측 데이터만 보면 라인이 실제보다 나빠 보인다.</p>`)}
      ${card(`<b>응답이 단조가 아니다</b><p class="tiny" style="margin:6px 0 0">공정 파라미터에는 최적 구간이 있고 양쪽으로 벗어나면 나빠진다.
        그래서 "값을 낮추세요" 같은 단방향 지시가 성립하지 않고 <b>구간으로 답해야</b> 한다.</p>`)}
      ${card(`<b>교란이 전부를 원인처럼 보이게 한다</b><p class="tiny" style="margin:6px 0 0">Lot 건강도가 나쁘면 계측값도 불량률도 같이 나빠진다.
        상관 스크리닝만 돌리면 <b>${y.meta.n_sig_factors}개 인자가 전부 유의</b>하게 나온다. 실제 원인은 5개다.</p>`)}
      ${card(`<b>열화는 스펙을 넘지 않는다</b><p class="tiny" style="margin:6px 0 0">DiceGuard 에 심은 열화 3건은 120일 내내 관리한계 안에서만 움직인다.
        <b>스펙 위반 알람으로는 하나도 안 잡힌다.</b></p>`)}
    </div>

    <h2>설계 원칙</h2>
    <div class="note ok"><b>데이터 생성기가 스스로를 검증한다.</b> 합성 데이터는 조용히 망가진다.
    행 수는 맞는데 상관이 사라졌다거나, 계측 편향이 반대로 걸렸다거나 하는 식이다.
    그래서 저장 직전에 자기 데이터를 검사하고 하나라도 어긋나면 파일을 안 쓰고 죽는다.
    실제로 개발 중 네 번 걸렸다 — 열화가 관리한계를 넘었고, U자가 잡음에 묻혔고, 설비 효과가 Lot 편차에 묻혔고, 시드가 프로세스마다 달라졌다.</div>
    <div class="note"><b>"너무 깨끗한" 데이터는 실패로 처리한다.</b> 검증기에 상관계수 <b>상한</b>이 있다.
    처음 돌렸을 때 rho 가 0.87~0.98 이 나왔는데 그런 데이터로 낸 모델 성능은 의미가 없어서, 상한 0.72 를 걸고 잡음을 다시 잡았다.</div>
    <div class="note bad"><b>안 되는 것은 안 된다고 적는다.</b> EtchPilot pad 공정은 Bottom CD·Depth 예측이 기준선보다 나쁘다.
    CellHealth 는 무릎이 있는데도 트리 모델이 로지스틱을 못 이겼다. RelyLab 은 p&lt;0.05 만으로 자르면 오탐이 절반이다.
    셋 다 이 페이지에 수치 그대로 있다.</div>`;
}

/* ------------------------------------------------------------------ yieldlens */
{
  const y = D.yieldlens, s = $("#p-yieldlens");
  s.innerHTML = `
    <h1>YieldLens</h1>
    <p class="lead">24단 공정을 거친 웨이퍼 ${y.meta.rows.toLocaleString()}장(Lot ${y.meta.lots}개)에서 수율을 떨어뜨리는 진짜 원인 인자를 찾는다.
    계측률은 Response ${y.measure_rate.response}%, Defect ${y.measure_rate.defect}% 뿐이고, 그 계측조차 무작위가 아니다.</p>
    <div class="grid g4">
      ${kpi(y.meta.tests, "전수 검정 조합", "46 인자 × 6 타깃")}
      ${kpi(y.meta.n_sig_factors, "상관상 유의 인자", "BH-FDR q ≤ 0.05", "warn")}
      ${kpi(y.detection.n_causal_candidates, "교란 제거 후 후보", "specificity > 0", "good")}
      ${kpi(`${y.detection.precision} / ${y.detection.recall}`, "precision / recall", `오탐 ${y.detection.false_positives.length}건`, "good")}
    </div>
    <h2>1. 교란 제거 — 상관계수만으로는 46개가 전부 원인처럼 보인다</h2>
    <p>Lot 건강도가 나쁘면 계측값도 불량률도 같이 나빠진다. 그래서 무관한 인자까지 전부 유의하게 나온다.
    진짜 원인 인자는 <b>특정 불량모드</b>에 붙어 있고, 교란으로만 엮인 인자는 <b>전체 수율</b> 쪽 상관이 더 강하다.
    그 차이의 부호로 가른다.</p>
    <div id="yl-spec" class="grid g2"></div>
    <h2>2. 검출된 원인 인자와 권장 운전구간</h2>
    <p>네 인자 모두 U자다. 최적 구간이 있고 양쪽으로 벗어나면 나빠지므로 구간으로 답해야 한다.
    아래 초록 띠가 실측 기준 권장 구간, 빨간 점선이 현재 공정 중앙값이다.</p>
    <div class="seg" id="yl-seg"></div>
    <div id="yl-driver" class="grid g2"></div>
    <h2>3. 수율 예측</h2>
    <div class="grid g3">
      ${kpi(num(y.model.mae_mean, 2) + " %p", "MAE", "GroupKFold(5) by Lot")}
      ${kpi(num(y.model.r2_mean, 3), "R²", "웨이퍼 단위 절대 예측")}
      ${kpi(pct(y.model.top30_worst_precision), "위험 Top30 정밀도", "실제 하위 10% 적중률", "good")}
    </div>
    <div class="note">R² 0.62 는 높은 수치가 아니고 <b>그게 정상이다</b>. 계측률이 22% 라 웨이퍼 한 장의 수율을
    정확히 맞추는 건 애초에 불가능하다. 이 모델의 쓰임새는 절대값 예측이 아니라 <b>검사 우선순위 정렬</b>이고,
    그 지표가 Top30 정밀도다. 결측 지시자를 피처로 넣은 효과는 R² +${num(y.model.r2_mean - y.ablation.r2_mean, 3)} 로
    기대만큼 크지 않았고, Top30 정밀도는 오히려 ${pct(y.ablation.top30_worst_precision - y.model.top30_worst_precision, 1)} 높았다. 그대로 적는다.</div>`;

  const sp = y.specificity.slice().sort((a, b) => b.spec - a.spec);
  const top = sp.slice(0, 9);
  $("#yl-spec").append(
    chart("specificity = |ρ(인자, 불량모드)| − |ρ(인자, 전체수율)|", "양수인 것만 인과 후보로 남긴다. 초록 = 실제 원인 인자",
      (n) => barh(n, {
        labels: top.map((d) => d.factor), values: top.map((d) => d.spec),
        colors: top.map((d) => (d.truth ? "#3ddc97" : "#5d6982")), fmtv: (v) => (v >= 0 ? "+" : "") + v.toFixed(3),
      })),
    h(table(
      [{ t: "인자", k: "f" }, { t: "불량모드 ρ", k: "a", num: 1 }, { t: "전체수율 ρ", k: "b", num: 1 }, { t: "specificity", k: "c", num: 1 }, { t: "판정", k: "d" }],
      sp.slice(0, 9).map((d) => ({
        f: `<code>${d.factor}</code>`, a: num(d.rho_mode), b: num(d.rho_yield),
        c: (d.spec >= 0 ? "+" : "") + num(d.spec),
        d: d.spec > 0 ? `<span class="tag ok">인과 후보</span>` : `<span class="tag dim">교란</span>`,
        _hl: d.truth,
      })),
    )).firstElementChild,
  );

  const seg = $("#yl-seg"), host = $("#yl-driver");
  y.drivers.forEach((d, i) => {
    const b = document.createElement("button");
    b.textContent = d.factor; b.onclick = () => draw(i);
    seg.appendChild(b);
  });
  function draw(i) {
    seg.querySelectorAll("button").forEach((b, j) => b.classList.toggle("on", j === i));
    const d = y.drivers[i];
    host.innerHTML = "";
    host.append(
      chart(`${d.factor} → ${d.target}`, `${d.shape === "u_shape" ? "U자 응답" : "단조 증가"} · 최적 중심 ${d.optimum} · 권장 ${d.recommended_low} ~ ${d.recommended_high}`,
        (n) => scatter(n, {
          points: d.points, line: d.profile, xlab: d.factor, ylab: d.target,
          band: [d.recommended_low, d.recommended_high],
          vlines: [{ x: d.current_median, color: "#ff6b6b", label: "현재 중앙값" }],
        })),
      h(card(`<b>${d.factor}</b> 조정 판단
        <table style="margin-top:8px">
          <tr><td>형태</td><td class="num">${d.shape === "u_shape" ? "U자 (최적 구간 존재)" : "단조 증가"}</td></tr>
          <tr><td>2차 적합 최적 중심</td><td class="num">${d.optimum}</td></tr>
          <tr><td>실측 권장 구간</td><td class="num">${d.recommended_low} ~ ${d.recommended_high}</td></tr>
          <tr><td>현재 공정 중앙값</td><td class="num">${d.current_median}</td></tr>
          <tr><td>조정 방향</td><td class="num">${d.current_median > d.recommended_high ? `<span class="tag no">낮춰야 함</span>` : d.current_median < d.recommended_low ? `<span class="tag mid">높여야 함</span>` : `<span class="tag ok">유지</span>`}</td></tr>
        </table>
        <p class="tiny" style="margin-top:10px">주황 선은 분위 구간별 실측 평균 불량률이다. 바닥이 최적점이고 양쪽으로 올라간다.
        현재 공정이 최적점보다 오른쪽에 있어서 <b>낮추는 방향</b>이 개선이다.</p>`)).firstElementChild,
    );
  }
  draw(0);
}

/* ------------------------------------------------------------------ etchpilot */
{
  const e = D.etchpilot, s = $("#p-etchpilot");
  const names = Object.keys(e.processes);
  s.innerHTML = `
    <h1>EtchPilot</h1>
    <p class="lead">4개 공정 × 12개 리비전 × 웨이퍼당 ${e.sites_per_wafer}개 사이트. 레시피를 실제로 돌리기 전에 결과를 예측하고,
    어느 파라미터를 어느 방향으로 조정할지 추천한다. 웨이퍼 평균이 아니라 <b>사이트 단위로 예측한 뒤 집계</b>한다 —
    같은 평균이라도 산포가 다르면 전혀 다른 레시피이기 때문이다.</p>
    <div class="seg" id="ep-seg"></div>
    <div id="ep-body"></div>`;

  const seg = $("#ep-seg"), body = $("#ep-body");
  names.forEach((n, i) => { const b = document.createElement("button"); b.textContent = n; b.onclick = () => draw(i); seg.appendChild(b); });

  function draw(i) {
    seg.querySelectorAll("button").forEach((b, j) => b.classList.toggle("on", j === i));
    const name = names[i], p = e.processes[name];
    body.innerHTML = `
      <div class="grid g4">
        ${kpi(p.meta.wafers, "웨이퍼", `${p.meta.rows.toLocaleString()} 사이트 계측`)}
        ${kpi(p.stages + " / " + p.meta.recipe_params, "Stage / 파라미터", `Depth 단위 ${p.meta.depth_unit}`)}
        ${kpi(p.ranking[0].rev, "최고 레시피", `품질점수 ${p.ranking[0].score}`, "good")}
        ${kpi(p.ranking.at(-1).rev, "최하위 레시피", `품질점수 ${p.ranking.at(-1).score}`, "bad")}
      </div>
      <h2>미학습 레시피에 대한 예측 성능</h2>
      <p class="tiny">리비전 단위 LeaveOneGroupOut. 모델은 평가 대상 레시피를 한 번도 본 적이 없다. 기준선은 "전체 중앙값으로 찍기".</p>
      <div id="ep-metric"></div>
      <h2>웨이퍼 맵 — 같은 공정, 최고 vs 최하위 레시피</h2>
      <p class="tiny">점 하나가 계측 사이트다. 색은 Top CD. 중심에서 엣지로 갈수록 CD 가 벌어지는 반경 프로파일이 보인다.</p>
      <div class="grid g2" id="ep-wafer"></div>
      <h2>레시피 순위 (실측)</h2>
      <p class="tiny">종합 점수 = Pass Rate 60% + Uniformity 25% + 목표 근접도 15%</p>
      <div id="ep-rank"></div>
      <h2>최하위 레시피에 대한 OFAT 조정 추천</h2>
      <p class="tiny">파라미터를 <b>하나씩만</b> ±6% 흔들어보고 예상 점수 변화를 낸다. 한 번에 여러 개를 바꾸면 원인을 못 가린다.</p>
      <div id="ep-ofat"></div>`;

    const m = p.metrics;
    const imp = m.map((x) => (1 - x.mae_mean / x.baseline_mae) * 100);
    $("#ep-metric").append(chart("기준선 대비 MAE 개선율", "음수면 기준선보다 나쁘다 = 예측 실패",
      (n) => bars(n, {
        labels: m.map((x) => x.target.replace(/_nm|_A/, "")), values: imp,
        colors: imp.map((v) => (v > 10 ? "#3ddc97" : v > 0 ? "#4da3ff" : "#ff6b6b")),
        yfmt: (v) => v.toFixed(0) + "%", ylab: "개선율", h: 230,
      })));
    if (Math.min(...imp) < 0) {
      $("#ep-metric").appendChild(h(`<div class="note bad" style="margin-top:10px"><b>${name} 공정의 일부 항목은 예측이 안 된다.</b>
        기준선보다 나쁘게 나왔다. 숨기지 않고 그대로 표시한다. 전반적으로 Top CD 는 잘 맞고
        <b>Bottom CD·Depth 로 갈수록 나빠진다</b> — 식각 깊이 방향으로 갈수록 레시피 파라미터로 설명되지 않는 요인이 늘어난다.</div>`));
    }

    const wf = $("#ep-wafer");
    const all = [...p.wafer_map.best.sites, ...p.wafer_map.worst.sites].map((x) => x.cd);
    const lo = Math.min(...all), hi = Math.max(...all);
    for (const [k, lab] of [["best", "최고"], ["worst", "최하위"]]) {
      const w = p.wafer_map[k];
      wf.appendChild(chart(`${lab} — ${w.rev}`, `Top CD (nm) · 사이트 ${w.sites.length}점`, (n) => wafer(n, { sites: w.sites, field: "cd", label: "Top CD", lo, hi })));
    }

    $("#ep-rank").innerHTML = table(
      [{ t: "순위", k: "r", num: 1 }, { t: "리비전", k: "v" }, { t: "품질점수", k: "s", num: 1 }, { t: "Pass Rate", k: "p", num: 1 }, { t: "Uniformity", k: "u", num: 1 }, { t: "목표근접", k: "x", num: 1 }],
      p.ranking.map((x) => ({ r: x.rank, v: `<code>${x.rev}</code>`, s: x.score, p: num(x.pass), u: num(x.unif), x: num(x.prox), _hl: x.rank === 1 })),
    );

    $("#ep-ofat").innerHTML = p.ofat.length ? table(
      [{ t: "파라미터", k: "p" }, { t: "방향", k: "d" }, { t: "현재", k: "c", num: 1 }, { t: "제안", k: "n", num: 1 }, { t: "예상 점수", k: "s", num: 1 }, { t: "개선", k: "g", num: 1 }],
      p.ofat.map((o) => ({
        p: `<code>${o.parameter}</code>`, d: o.direction, c: o.current, n: o.proposed, s: o.predicted_score,
        g: `<span class="tag ${o.gain > 3 ? "ok" : o.gain > 0 ? "mid" : "dim"}">${o.gain > 0 ? "+" : ""}${o.gain}</span>`,
        _hl: o === p.ofat[0],
      })),
    ) : "";
  }
  draw(0);
}

/* ------------------------------------------------------------------ diceguard */
{
  const g = D.diceguard, s = $("#p-diceguard");
  const best = g.sweep.filter((x) => x.recall >= 1 && x.window !== "full").map((x) => +x.window.replace("last", ""));
  const minw = best.length ? Math.min(...best) : null;
  s.innerHTML = `
    <h1>DiceGuard</h1>
    <p class="lead">레이저 다이싱 장비 ${g.meta.machines}대의 ${g.meta.days}일 가공 이력 ${g.meta.rows.toLocaleString()}건.
    <b>관리한계를 한 번도 넘지 않고 서서히 나빠지는 장비</b>를 찾아낸다. 심어둔 열화 3건은 전부 ±3σ 안에서만 움직인다 —
    스펙 위반 알람으로는 하나도 안 잡힌다.</p>
    <div class="grid g4">
      ${kpi(g.injected.length, "주입한 열화", "전부 관리한계 안")}
      ${kpi("100%", "재현율 / 정밀도", "전 구간 기준, 72회 검정", "good")}
      ${kpi(minw ? minw + "일" : "–", "판정 구간 하한", "재현율 100% 유지 최소", "warn")}
      ${kpi(num(g.kerf_confound.corr, 2), "레이저파워–커프폭 상관", "교란의 크기", "bad")}
    </div>

    <h2>1. 열화는 스펙 안에서 진행된다</h2>
    <p>아래는 주입한 열화 3건의 일 평균 추이다. 초록 영역이 관리한계(±3σ). <b>120일 내내 한 번도 벗어나지 않는다.</b>
    맨 아래는 대조군으로 열화가 없는 정상 변수다.</p>
    <div class="grid g2" id="dg-series"></div>

    <h2>2. 추세 판정 구간을 며칠로 잡을 것인가</h2>
    <p>"추세를 보자"는 말은 누구나 한다. 그런데 <b>며칠</b>을 봐야 하는지는 데이터로 답해야 한다.
    구간을 짧게 잡으면 표본이 줄어 검정력이 사라지고, 열화가 포화형이라 후반 기울기도 완만해진다. 두 효과가 겹친다.</p>
    <div class="grid g2" id="dg-sweep"></div>
    <div class="note ${minw ? "" : "bad"}">재현율 100% 를 유지하는 <b>최소 구간은 ${minw}일</b>이다.
    14일로 줄이면 재현율과 정밀도가 <b>동시에</b> 무너진다 — 열화 하나를 놓치면서 멀쩡한 변수를 열화로 지목한다.</div>

    <h2>3. 장비 Health Index</h2>
    <p>순위가 아니라 <b>관리한계까지 남은 여유</b>로 잰다.
    <code>HI = 10 + (level − 10) × (1 − 0.45 × max(성숙도, 긴급도))</code>, 장비 점수 = 변수 점수의 <b>최솟값</b>.
    최솟값을 쓰는 이유는 평균에 묻히지 않게 하려는 것이다.</p>
    <div class="grid g2" id="dg-health"></div>

    <h2>4. 원인과 감시지표를 분리한다</h2>
    <p>레이저 파워와 커프 폭은 <b>r = ${num(g.kerf_confound.corr, 2)}</b> 로 붙어 있다. 불량과의 상관만 보면 커프 폭이 상위 원인으로 뽑힌다.
    그런데 커프 폭은 <b>가공 결과</b>다. "커프 폭을 조치하세요"는 실행 불가능한 지시다.</p>
    <div class="grid g2" id="dg-confound"></div>`;

  const sh = $("#dg-series");
  Object.values(g.series).forEach((sr) => {
    const inj = g.injected.find((x) => x.machine === sr.machine && x.variable === sr.variable);
    sh.appendChild(chart(`${sr.machine} · ${sr.variable}`,
      inj ? `주입 열화 ${inj.in_sigma > 0 ? "+" : ""}${inj.in_sigma}σ · ${inj.onset_day}일차 시작` : "대조군 — 열화 없음",
      (n) => timeseries(n, { x: sr.day, y: sr.value, center: sr.center, ucl: sr.ucl, lcl: sr.lcl, xlab: "경과일", ylab: sr.variable })));
  });

  $("#dg-sweep").append(
    chart("판정 구간별 검출 성능", "주입 열화 3건 기준 재현율 / 정밀도",
      (n) => bars(n, {
        labels: g.sweep.map((x) => x.window.replace("last", "").replace("full", "전체")),
        values: g.sweep.map((x) => x.recall * 100),
        colors: g.sweep.map((x) => (x.recall >= 1 ? "#3ddc97" : x.recall > 0 ? "#ffb454" : "#ff6b6b")),
        yfmt: (v) => v.toFixed(0) + "%", ylab: "재현율", xlab: "판정 구간 (일)", h: 240,
      })),
    h(table(
      [{ t: "판정 구간", k: "w" }, { t: "일수", k: "d", num: 1 }, { t: "유의 검출", k: "n", num: 1 }, { t: "재현율", k: "r", num: 1 }, { t: "정밀도", k: "p", num: 1 }],
      g.sweep.map((x) => ({
        w: x.window === "full" ? "전 구간" : x.window.replace("last", "최근 ") + "일",
        d: x.days_used, n: x.n_significant,
        r: `<span class="tag ${x.recall >= 1 ? "ok" : x.recall > 0 ? "mid" : "no"}">${pct(x.recall, 0)}</span>`,
        p: pct(x.precision, 0), _hl: x.window === "last" + minw,
      })),
    )).firstElementChild,
  );

  $("#dg-health").append(
    chart("장비 Health Index", "낮을수록 급함 · 초록 = 열화 주입한 장비",
      (n) => barh(n, {
        labels: g.health.map((x) => `${x.machine}  ${x.variable.slice(0, 14)}`),
        values: g.health.map((x) => x.hi),
        colors: g.health.map((x) => (x.trend ? "#ff6b6b" : "#4da3ff")),
        fmtv: (v) => v.toFixed(1), max: 100,
      })),
    h(table(
      [{ t: "장비", k: "m" }, { t: "최악 변수", k: "v" }, { t: "HI", k: "h", num: 1 }, { t: "level", k: "l", num: 1 }, { t: "긴급도", k: "u", num: 1 }, { t: "추세", k: "t" }],
      g.health.map((x) => ({
        m: x.machine, v: `<code>${x.variable}</code>`, h: x.hi, l: x.level, u: x.urgency,
        t: x.trend ? `<span class="tag no">유의</span>` : `<span class="tag dim">정상</span>`, _hl: x.trend,
      })),
    )).firstElementChild,
  );

  const a = g.attribution;
  $("#dg-confound").append(
    chart("레이저 파워 vs 커프 폭", `설정값과 가공 결과가 r = ${num(g.kerf_confound.corr, 2)} 로 붙어 있다`,
      (n) => scatter(n, { points: g.kerf_confound.points, xlab: "Laser_Power_W", ylab: "Kerf_Width_um", color: "#b48ead" })),
    h(`<div class="card"><b>불량 선별 모델 비교</b>
      <table style="margin-top:8px">
        <thead><tr><th>모델</th><th class="num">피처</th><th class="num">Top 5% 정밀도</th><th class="num">Lift</th></tr></thead>
        <tbody>
          <tr><td>감시지표 포함</td><td class="num">${a.with_monitoring.n_features}</td><td class="num">${num(a.with_monitoring.precision_at_top5pct)}</td><td class="num">${a.with_monitoring.lift}</td></tr>
          <tr class="hl"><td>설비 설정값만</td><td class="num">${a.actionable_only.n_features}</td><td class="num">${num(a.actionable_only.precision_at_top5pct)}</td><td class="num">${a.actionable_only.lift}</td></tr>
        </tbody>
      </table>
      <p class="tiny" style="margin-top:10px">이 데이터에서는 감시지표를 빼도 성능이 거의 안 떨어졌고 1순위 인자도 양쪽 다
      <code>${a.actionable_only.top_features[0].feature}</code>(설정값)였다. 운이 좋았던 경우다.
      <b>그렇더라도 조치 추천은 설정값 모델에서만 낸다.</b> 실행 가능성이 정확도보다 앞선다.</p></div>`).firstElementChild,
  );
}

/* ------------------------------------------------------------------ cellhealth */
{
  const c = D.cellhealth, s = $("#p-cellhealth");
  const lr = c.model.logistic, gb = c.model.gradient_boosting;
  s.innerHTML = `
    <h1>CellHealth</h1>
    <p class="lead">출하된 모바일 스토리지 ${c.meta.devices.toLocaleString()}대를 2회차에 걸쳐 수집한 텔레메트리 ${c.meta.rows.toLocaleString()}행.
    자연어로 물으면 정확한 숫자로 답하는 엔진과, 마모 위험 예측 모델을 만든다.</p>
    <div class="grid g4">
      ${kpi("7 / 7", "자연어 질의 해석", "LLM 없이 규칙 파서로", "good")}
      ${kpi(pct(c.positive_rate, 2), "uecc 발생률", "극단적 불균형", "warn")}
      ${kpi(lr.lift + "배", "상위 2% 농축", "기저율 대비", "good")}
      ${kpi(num(lr.pr_auc, 3), "PR-AUC", `ROC-AUC ${num(lr.roc_auc, 3)}`)}
    </div>

    <h2>1. 자연어 질의 엔진 — LLM 을 부르지 않는다</h2>
    <p>데이터를 LLM 에 통째로 넘기면 <b>숫자가 틀린다.</b> 품질 데이터에서 숫자가 틀리면 도구로서 가치가 0 이다.
    컬럼 사전과 규칙 파서로 질문을 실행 계획으로 바꾸고, 집계는 pandas 가 한다.
    답변에는 항상 <b>사용한 컬럼 · 적용한 필터 · 매칭 행 수</b>를 붙인다. 사람이 검증할 수 있어야 하기 때문이다.</p>
    <div id="ch-q"></div>

    <h2>2. 마모 무릎 지점</h2>
    <p>정격 대비 소모율(pe_ratio)이 0.75 를 넘는 순간 uecc 발생률이 자릿수 단위로 뛴다.
    그 아래로는 완전히 평평하다. <b>잔여수명을 선형 외삽하면 안 되는 이유다.</b></p>
    <div class="grid g2" id="ch-knee"></div>

    <h2>3. 마모 위험 예측</h2>
    <div class="grid g2" id="ch-model"></div>
    <div class="note">양성 비율이 ${pct(c.positive_rate, 2)} 라 정확도는 무의미하다. PR-AUC 로 평가한다.
    <b>무릎이 있는데도 트리 모델이 로지스틱을 이기지 못했다</b> (${num(gb.pr_auc, 3)} vs ${num(lr.pr_auc, 3)}).
    핵심 인자 하나가 위험도를 단조적으로 결정하면 선형 모델도 충분히 순위를 매길 수 있기 때문이다.
    무릎이 있다고 자동으로 트리가 이기는 게 아니라는 뜻이고, 확인해 보기 전에는 단정하면 안 된다.</div>`;

  $("#ch-q").innerHTML = table(
    [{ t: "질문", k: "q" }, { t: "해석한 컬럼", k: "c" }, { t: "매칭 행", k: "m", num: 1 }, { t: "결과", k: "r" }],
    c.queries.map((x) => ({
      q: x.q,
      c: x.cols.map((v) => `<code>${v}</code>`).join(" "),
      m: (x.matched ?? 0).toLocaleString(),
      r: x.result.map((o) => { const v = Object.values(o); return v.length > 1 ? `${v[0]} = <b>${fmt(v.at(-1), 3)}</b>` : `<b>${fmt(v[0], 3)}</b>`; }).join(", "),
    })),
  );

  const kb = c.knee.bins, kr = c.knee.rate;
  $("#ch-knee").append(
    chart("pe_ratio 구간별 uecc 발생률", "0.75 를 넘는 순간 20배, 0.85 를 넘으면 100배 이상",
      (n) => bars(n, {
        labels: kb.map((b) => b.replace(/[()\[\]]/g, "").split(", ")[1]),
        values: kr.map((v) => v * 100),
        colors: kr.map((v) => (v > .3 ? "#ff6b6b" : v > .05 ? "#ffb454" : "#3ddc97")),
        yfmt: (v) => v.toFixed(0) + "%", ylab: "uecc 발생률", xlab: "pe_ratio 구간 상한", h: 250,
      })),
    h(table(
      [{ t: "pe_ratio 구간", k: "b" }, { t: "uecc 발생률", k: "r", num: 1 }, { t: "", k: "t" }],
      kb.map((b, i) => ({
        b: `<code>${b}</code>`, r: pct(kr[i], 2),
        t: kr[i] > .3 ? `<span class="tag no">위험</span>` : kr[i] > .05 ? `<span class="tag mid">주의</span>` : `<span class="tag ok">안정</span>`,
        _hl: kr[i] > .05,
      })),
    )).firstElementChild,
  );

  $("#ch-model").append(
    chart("모델 비교 (PR-AUC)", "양성 0.76% 의 불균형 문제 — ROC 만 보면 틀린다",
      (n) => barh(n, {
        labels: ["로지스틱 PR-AUC", "GBM PR-AUC", "로지스틱 ROC-AUC", "GBM ROC-AUC"],
        values: [lr.pr_auc, gb.pr_auc, lr.roc_auc, gb.roc_auc],
        colors: ["#3ddc97", "#4da3ff", "#5d6982", "#5d6982"], fmtv: (v) => v.toFixed(3), max: 1,
      })),
    chart("벤더별 마모 현황", "벤더마다 정격 P/E 수명이 달라 pe_cycle 원값 비교는 틀린다",
      (n) => bars(n, {
        labels: c.vendors.map((v) => v.vendor), values: c.vendors.map((v) => v.pe),
        colors: ["#4da3ff", "#b48ead", "#ffb454"], yfmt: (v) => v.toFixed(0), ylab: "평균 pe_cycle", h: 250,
      })),
  );
}

/* ------------------------------------------------------------------ relylab */
{
  const r = D.relylab, s = $("#p-relylab");
  const gain = r.per_test.reduce((a, x) => a + x.gain, 0) / r.per_test.length;
  s.innerHTML = `
    <h1>RelyLab</h1>
    <p class="lead">FinFET 웨이퍼 ${r.meta.wafers.toLocaleString()}장(Lot ${r.meta.lots}개)의 소자 파라미터 · 신뢰성 시험 ${r.meta.tests.toLocaleString()}건 · 설비 이력을 엮어
    <b>불합격의 원인이 파라미터 마진인지 특정 설비인지</b> 가른다. 문제 설비 2대를 심어뒀고, 둘 다 자기 담당 파라미터를 <b>스펙 안에서만</b> 민다.</p>
    <div class="grid g4">
      ${kpi(pct(r.meta.overall_pass, 1), "전 시험 합격", "웨이퍼 기준")}
      ${kpi("+" + num(gain, 3), "설비 추가 시 AUC 개선", "8개 시험 평균", "good")}
      ${kpi(r.detection.recall, "문제 설비 recall", `정답 ${r.rogue.length}대 전부 검출`, "good")}
      ${kpi(r.detection.precision, "precision", `${r.detection.n_flagged}대 플래그 중`, "warn")}
    </div>

    <h2>1. 파라미터만으로는 부족하다</h2>
    <p>소자 파라미터 14개로 시험별 불합격을 예측한 뒤, 같은 모델에 <b>설비 배정</b>을 추가하고 AUC 차이를 잰다.
    유의하게 좋아지면 파라미터 계측으로는 안 보이는 원인이 설비 쪽에 남아 있다는 뜻이다.</p>
    <div class="grid g2" id="rl-auc"></div>

    <h2>2. 어느 설비인가</h2>
    <p>웨이퍼를 독립 표본으로 보면 안 된다. 설비는 <b>Lot 단위로 배정</b>되므로 웨이퍼 1,200장을 그대로 검정에 넣으면
    유의성이 과대평가된다. Lot 평균으로 집계한 뒤 Mann-Whitney U 단측 검정을 돌린다.</p>
    <div id="rl-attr"></div>
    <div class="note">p &lt; 0.05 만으로 자르면 ${r.detection.n_flagged}대가 걸려 <b>오탐이 절반</b>이다.
    그런데 효과 크기가 확실히 갈린다 — 진짜 문제 설비 2대는 차이가 −0.40, −0.54 이고 p 가 1e-4, 1e-6 수준인데,
    오탐 2대는 차이가 −0.20 수준이고 p 가 겨우 유의선을 넘겼다.
    실무에서는 <b>효과 크기 하한(예: 차이 0.3 이상)</b>을 같이 걸어야 하고, 그러면 정확히 2대만 남는다.</div>

    <h2>3. 왜 파라미터 판정으로는 못 잡는가</h2>
    <div id="rl-hidden"></div>
    <div class="note bad"><b>PVD-02 는 파라미터 스펙내 비율이 98.3% 다.</b> 타 설비(99.8%)와 1.5%p 차이뿐이라
    인라인 계측 판정으로는 완벽한 정상 설비다. 그런데 신뢰성 합격률은 31% 로 타 설비 71% 의 절반 아래다.
    <b>설비 배정 이력과 신뢰성 결과를 엮어야만 드러난다.</b></div>`;

  $("#rl-auc").append(
    chart("설비 정보 추가에 따른 AUC 개선", "개선폭이 큰 시험이 문제 설비 담당 모듈에 민감한 항목들이다",
      (n) => barh(n, {
        labels: r.per_test.map((x) => x.test), values: r.per_test.map((x) => x.gain),
        colors: r.per_test.map((x) => (x.gain > .12 ? "#3ddc97" : x.gain > .08 ? "#4da3ff" : "#5d6982")),
        fmtv: (v) => "+" + v.toFixed(3),
      })),
    h(table(
      [{ t: "시험", k: "t" }, { t: "불합격률", k: "f", num: 1 }, { t: "파라미터", k: "a", num: 1 }, { t: "+설비", k: "b", num: 1 }, { t: "개선", k: "g", num: 1 }],
      r.per_test.map((x) => ({
        t: `<b>${x.test}</b>`, f: num(x.fail_rate), a: num(x.auc_params_only), b: num(x.auc_params_plus_tools),
        g: `<span class="tag ${x.gain > .12 ? "ok" : "dim"}">+${num(x.gain)}</span>`, _hl: x.gain > .12,
      })),
    )).firstElementChild,
  );

  $("#rl-attr").innerHTML = table(
    [{ t: "모듈", k: "m" }, { t: "설비", k: "t" }, { t: "Lot", k: "n", num: 1 }, { t: "합격률", k: "a", num: 1 }, { t: "타 설비", k: "b", num: 1 }, { t: "차이", k: "d", num: 1 }, { t: "p", k: "p", num: 1 }, { t: "플래그", k: "f" }, { t: "실제 문제", k: "r" }],
    r.attribution.slice(0, 8).map((x) => ({
      m: x.module, t: `<code>${x.tool}</code>`, n: x.lots, a: num(x.pass), b: num(x.other), d: num(x.delta),
      p: sci(x.p),
      f: x.flagged ? `<span class="tag ${x.truth ? "ok" : "mid"}">O</span>` : `<span class="tag dim">–</span>`,
      r: x.truth ? `<span class="tag no">문제 설비</span>` : "",
      _hl: x.truth,
    })),
  );

  $("#rl-hidden").innerHTML = table(
    [{ t: "설비", k: "t" }, { t: "모듈", k: "m" }, { t: "파라미터", k: "p" }, { t: "스펙내 (해당)", k: "a", num: 1 }, { t: "스펙내 (타)", k: "b", num: 1 }, { t: "신뢰성 합격 (해당)", k: "c", num: 1 }, { t: "신뢰성 합격 (타)", k: "d", num: 1 }],
    r.hidden.map((x) => ({
      t: `<code>${x.tool}</code>`, m: x.module, p: `<code>${x.param}</code>`,
      a: pct(x.in_spec_rate_on_tool, 1), b: pct(x.in_spec_rate_others, 1),
      c: `<span class="tag no">${pct(x.reliability_pass_on_tool, 1)}</span>`, d: pct(x.reliability_pass_others, 1),
      _hl: true,
    })),
  );
}

go(location.hash.slice(1) || "home");
