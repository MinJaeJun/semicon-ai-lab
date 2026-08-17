/* 의존성 없는 최소 SVG 차트. Chart.js 를 안 쓴 이유는 CDN 없이 즉시 뜨게 하려는 것. */

const NS = "http://www.w3.org/2000/svg";
const el = (t, a = {}, kids = []) => {
  const n = document.createElementNS(NS, t);
  for (const [k, v] of Object.entries(a)) if (v !== null && v !== undefined) n.setAttribute(k, v);
  for (const c of [].concat(kids)) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  return n;
};
const fmt = (v, d = 2) => (Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 }) : (+v).toFixed(d).replace(/\.?0+$/, ""));

function frame(w, h, m) {
  const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: "xMidYMid meet" });
  return { svg, iw: w - m.l - m.r, ih: h - m.t - m.b, m };
}

function axes(g, { xd, yd, xlab, ylab, xticks = 5, yticks = 4, xfmt = fmt, yfmt = fmt }) {
  const { svg, iw, ih, m } = g;
  const X = (v) => m.l + ((v - xd[0]) / (xd[1] - xd[0] || 1)) * iw;
  const Y = (v) => m.t + ih - ((v - yd[0]) / (yd[1] - yd[0] || 1)) * ih;
  for (let i = 0; i <= yticks; i++) {
    const v = yd[0] + ((yd[1] - yd[0]) * i) / yticks;
    svg.appendChild(el("line", { class: "grid", x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v) }));
    svg.appendChild(el("text", { x: m.l - 7, y: Y(v) + 3.5, "text-anchor": "end" }, yfmt(v)));
  }
  for (let i = 0; i <= xticks; i++) {
    const v = xd[0] + ((xd[1] - xd[0]) * i) / xticks;
    svg.appendChild(el("text", { x: X(v), y: m.t + ih + 15, "text-anchor": "middle" }, xfmt(v)));
  }
  svg.appendChild(el("line", { class: "axis", x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }));
  if (xlab) svg.appendChild(el("text", { x: m.l + iw / 2, y: m.t + ih + 32, "text-anchor": "middle" }, xlab));
  if (ylab) svg.appendChild(el("text", { x: 11, y: m.t + ih / 2, "text-anchor": "middle", transform: `rotate(-90 11 ${m.t + ih / 2})` }, ylab));
  return { X, Y };
}

const ext = (a, pad = 0.06) => {
  let lo = Math.min(...a), hi = Math.max(...a);
  if (lo === hi) { lo -= 1; hi += 1; }
  const p = (hi - lo) * pad;
  return [lo - p, hi + p];
};

/* 산점도 + 선택적 추세선(구간 평균) */
export function scatter(host, { points, line, xlab, ylab, color = "#4da3ff", lineColor = "#ffb454", vlines = [], band = null, h = 260 }) {
  const w = 560, m = { l: 46, r: 14, t: 10, b: 40 };
  const g = frame(w, h, m);
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const xd = ext(xs), yd = ext(ys);
  const { X, Y } = axes(g, { xd, yd, xlab, ylab });
  if (band) {
    g.svg.appendChild(el("rect", {
      x: X(band[0]), width: Math.max(1, X(band[1]) - X(band[0])), y: m.t, height: g.ih,
      fill: "#3ddc97", opacity: .09,
    }));
  }
  for (const v of vlines) {
    g.svg.appendChild(el("line", { x1: X(v.x), x2: X(v.x), y1: m.t, y2: m.t + g.ih, stroke: v.color || "#ff6b6b", "stroke-width": 1.4, "stroke-dasharray": "4 3" }));
    if (v.label) g.svg.appendChild(el("text", { x: X(v.x) + 4, y: m.t + 11, fill: v.color || "#ff6b6b" }, v.label));
  }
  const dots = el("g");
  for (const [x, y] of points) dots.appendChild(el("circle", { cx: X(x), cy: Y(y), r: 1.7, fill: color, opacity: .32 }));
  g.svg.appendChild(dots);
  if (line && line.x.length) {
    const d = line.x.map((x, i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(line.y[i]).toFixed(1)}`).join("");
    g.svg.appendChild(el("path", { d, fill: "none", stroke: lineColor, "stroke-width": 2.2, "stroke-linejoin": "round" }));
    line.x.forEach((x, i) => g.svg.appendChild(el("circle", { cx: X(x), cy: Y(line.y[i]), r: 2.8, fill: lineColor })));
  }
  host.appendChild(g.svg);
}

/* 시계열 + 관리한계 */
export function timeseries(host, { x, y, center, ucl, lcl, xlab, ylab, h = 230 }) {
  const w = 560, m = { l: 52, r: 14, t: 10, b: 38 };
  const g = frame(w, h, m);
  const yd = ext([...y, ucl, lcl], .04), xd = [Math.min(...x), Math.max(...x)];
  const { X, Y } = axes(g, { xd, yd, xlab, ylab, xticks: 6, yfmt: (v) => fmt(v, 2) });
  g.svg.appendChild(el("rect", { x: m.l, width: g.iw, y: Y(ucl), height: Math.max(1, Y(lcl) - Y(ucl)), fill: "#3ddc97", opacity: .05 }));
  for (const [v, c, t] of [[ucl, "#ff6b6b", "+3σ"], [center, "#5d6982", "중심"], [lcl, "#ff6b6b", "−3σ"]]) {
    g.svg.appendChild(el("line", { x1: m.l, x2: m.l + g.iw, y1: Y(v), y2: Y(v), stroke: c, "stroke-width": 1, "stroke-dasharray": "5 4", opacity: .75 }));
    g.svg.appendChild(el("text", { x: m.l + g.iw + 3, y: Y(v) + 3.5, fill: c }, t));
  }
  const d = x.map((v, i) => `${i ? "L" : "M"}${X(v).toFixed(1)},${Y(y[i]).toFixed(1)}`).join("");
  g.svg.appendChild(el("path", { d, fill: "none", stroke: "#4da3ff", "stroke-width": 1.6 }));
  host.appendChild(g.svg);
}

/* 가로 막대 */
export function barh(host, { labels, values, colors, fmtv = (v) => fmt(v, 2), h = null, max = null }) {
  const w = 560, rowH = 26, m = { l: 150, r: 54, t: 6, b: 6 };
  const H = h || m.t + m.b + labels.length * rowH;
  const g = frame(w, H, m);
  const hi = max ?? Math.max(...values.map(Math.abs)) * 1.08;
  labels.forEach((lab, i) => {
    const y = m.t + i * rowH + 5;
    const v = values[i];
    const bw = Math.max(2, (Math.abs(v) / hi) * g.iw);
    g.svg.appendChild(el("text", { x: m.l - 8, y: y + 11, "text-anchor": "end", fill: "#8b98ad" }, lab));
    g.svg.appendChild(el("rect", { x: m.l, y, width: bw, height: 15, rx: 3, fill: (colors && colors[i]) || "#4da3ff" }));
    g.svg.appendChild(el("text", { x: m.l + bw + 6, y: y + 11.5, fill: "#e6edf7" }, fmtv(v)));
  });
  host.appendChild(g.svg);
}

/* 세로 막대 (히스토그램/분포) */
export function bars(host, { labels, values, colors, xlab, ylab, yfmt = (v) => fmt(v, 1), h = 250, rotate = false }) {
  const w = 560, m = { l: 48, r: 12, t: 12, b: rotate ? 74 : 40 };
  const g = frame(w, h, m);
  const yd = [0, Math.max(...values) * 1.12 || 1];
  const { Y } = axes(g, { xd: [0, 1], yd, ylab, xticks: 0, yfmt });
  const bw = (g.iw / labels.length) * 0.72, gap = g.iw / labels.length;
  labels.forEach((lab, i) => {
    const x = m.l + i * gap + (gap - bw) / 2;
    const y = Y(values[i]);
    g.svg.appendChild(el("rect", { x, y, width: bw, height: Math.max(1, m.t + g.ih - y), rx: 3, fill: (colors && colors[i]) || "#4da3ff" }));
    const t = el("text", { x: x + bw / 2, y: m.t + g.ih + (rotate ? 12 : 15), "text-anchor": rotate ? "end" : "middle", fill: "#8b98ad" }, lab);
    if (rotate) t.setAttribute("transform", `rotate(-42 ${x + bw / 2} ${m.t + g.ih + 12})`);
    g.svg.appendChild(t);
  });
  if (xlab) g.svg.appendChild(el("text", { x: m.l + g.iw / 2, y: h - 4, "text-anchor": "middle" }, xlab));
  host.appendChild(g.svg);
}

/* 웨이퍼 맵: 극좌표 사이트를 원판에 찍는다 */
export function wafer(host, { sites, field, label, lo, hi, h = 300 }) {
  const size = 300, cx = size / 2, cy = size / 2, R = size / 2 - 16;
  const svg = el("svg", { viewBox: `0 0 ${size} ${size + 26}` });
  svg.appendChild(el("circle", { cx, cy, r: R, fill: "#0b0e14", stroke: "#232c3d", "stroke-width": 1.5 }));
  for (const rf of [0.52, 0.84, 0.96]) svg.appendChild(el("circle", { cx, cy, r: R * rf, fill: "none", stroke: "#1a2130", "stroke-dasharray": "2 4" }));
  svg.appendChild(el("line", { x1: cx - R, x2: cx + R, y1: cy, y2: cy, stroke: "#1a2130" }));
  svg.appendChild(el("line", { x1: cx, x2: cx, y1: cy - R, y2: cy + R, stroke: "#1a2130" }));
  const vals = sites.map((s) => s[field]);
  const min = lo ?? Math.min(...vals), max = hi ?? Math.max(...vals);
  const col = (v) => {
    const t = Math.max(0, Math.min(1, (v - min) / (max - min || 1)));
    const stops = [[61, 220, 151], [255, 214, 102], [255, 107, 107]];
    const i = t < .5 ? 0 : 1, k = t < .5 ? t * 2 : (t - .5) * 2;
    const a = stops[i], b = stops[i + 1];
    return `rgb(${a.map((c, j) => Math.round(c + (b[j] - c) * k)).join(",")})`;
  };
  for (const s of sites) {
    const a = (s.ang * Math.PI) / 180;
    const x = cx + Math.cos(a) * R * s.rf, y = cy - Math.sin(a) * R * s.rf;
    const c = el("circle", { cx: x, cy: y, r: 7.5, fill: col(s[field]), stroke: "#0b0e14", "stroke-width": 1.2 });
    c.appendChild(el("title", {}, `${s.id} (${s.zone})\n${label} ${fmt(s[field], 3)}`));
    svg.appendChild(c);
  }
  const gw = 150, gx = cx - gw / 2, gy = size + 10;
  const grad = el("linearGradient", { id: "wg" + field.replace(/\W/g, "") });
  [[0, "#3ddc97"], [.5, "#ffd666"], [1, "#ff6b6b"]].forEach(([o, c]) => grad.appendChild(el("stop", { offset: o, "stop-color": c })));
  svg.appendChild(el("defs", {}, [grad]));
  svg.appendChild(el("rect", { x: gx, y: gy, width: gw, height: 7, rx: 3, fill: `url(#wg${field.replace(/\W/g, "")})` }));
  svg.appendChild(el("text", { x: gx - 5, y: gy + 7, "text-anchor": "end" }, fmt(min, 2)));
  svg.appendChild(el("text", { x: gx + gw + 5, y: gy + 7 }, fmt(max, 2)));
  host.appendChild(svg);
}

export { el, fmt };
