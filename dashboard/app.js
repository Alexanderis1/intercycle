/* InterCycle observer map view. Static, no build step. Reads intercycle.stream/1 runs (see docs/DASHBOARD.md).
   Data: window.__IC_BUNDLE__ (single-file bundle) or fetch: layout.json, runs.json, <run>/index.json, <run>/snap_*.json.
   Live: ws://127.0.0.1:8765 (observer/stream.py StreamWriter(live=True)).
   URL params: run=<id> mode=live ws=<url> autoplay=1 speed=<x> u=<0..1> machine=<id> layers=l1,l2,l3,unk,dir,cycle
               base=<dir holding run folders, default ../out/stream>. Everything shown is SYNTHETIC: simulated plant. */
(() => {
"use strict";

const Q = new URLSearchParams(location.search);
const BUNDLE = window.__IC_BUNDLE__ || null;
const $ = (id) => document.getElementById(id);
const PROD_COLOR = { running: "#0ca30c", andon: "#d03b3b", break: "#fab219", shift_change: "#ec835a" };
const KIND_LABEL = { machine_converged: "envelope converged", change_cleared: "change cleared", change_detected: "change detected", machine_spawned: "machine spawned", phase_discovered: "phase discovered",
                     phase_named: "phase named (Gemma)", camera_offline: "camera offline", production_state: "production state", note: "note" };
const METRIC_PREF = ["l1_iou", "iou", "unsafe_free", "unsafe_free_rate", "false_occupied", "false_occupied_share", "l2_auroc", "l3_auroc",
                     "unsafe_free_pre_detection", "unsafe_free_transition", "unsafe_free_learned",
                     "brier", "auroc", "l2_brier", "l1_precision", "l1_recall",
                     "l3_hot_m2", "l2_hot_m2", "slice_ticks", "unknown_frac"];
const METRIC_NAME = { l1_iou: "L1 envelope IoU (held-out)", iou: "IoU", unsafe_free: "Unsafe-free rate (target 0)", unsafe_free_rate: "Unsafe-free rate",
                      false_occupied_share: "False-occupied share", l2_auroc: "L2 vehicles AUROC", l3_auroc: "L3 people AUROC",
                      l2_brier: "L2 Brier", l3_brier: "L3 Brier", l1_recall: "L1 recall", unknown_share: "UNKNOWN share",
                      env_unsafe_free: "L1 envelope unsafe-free", unsafe_free_centre: "Unsafe-free (cell centres)",
                      unsafe_free_pre_detection: "Unsafe-free before detection", unsafe_free_transition: "Unsafe-free in transition (target 0)",
                      unsafe_free_learned: "Unsafe-free after learning",
                      false_occupied: "False-occupied", brier: "Brier (L2/L3)", auroc: "AUROC", slice_ticks: "Slice observations (ticks)",
                      unknown_frac: "UNKNOWN share of site", l2_hot_m2: "L2 hot area p≥0.3 (m²)", l3_hot_m2: "L3 hot area p≥0.3 (m²)",
                      l2_mean_occ: "L2 mean occupancy", l3_mean_occ: "L3 mean occupancy" };
const nowS = () => (window.__IC_CLOCK__ != null ? window.__IC_CLOCK__ : performance.now() / 1000);   // recorder may pin the clock
const CYCLE_RATE = 4;            // cycle preview speed (x real time)
const SWEEP_S = 30;              // a full replay sweep at 1x lasts 30 s (video scene length)

const S = {
  layout: null, runs: [], run: null, index: null, snaps: new Map(), decoded: new Map(),
  axis: "t_sim", u: 0, playing: false, speed: 1, layers: { l1: true, l2: true, l3: true, unk: true, dir: false, cycle: true },
  machine: null, mode: "replay", ws: null, follow: true, cur: null, curFile: null, anchor: { wall: 0, phase: {} },
  view: null, rasters: null, lastCaptionSeq: -1,
};

// ------------------------------------------------------------------------------------------------ data
async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}
function runBase(run) { return run.path || `${Q.get("base") || "../out/stream"}/${run.id}`; }

async function loadSnap(file) {
  if (S.snaps.has(file)) return S.snaps.get(file);
  let snap;
  const b = S.run && S.run._bundle;
  if (b) snap = b.snaps[file];
  else snap = await getJSON(`${runBase(S.run)}/${file}`);
  S.snaps.set(file, snap);
  return snap;
}

function b64bytes(s) {
  const bin = atob(s); const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
async function inflate(bytes) {
  const ds = new DecompressionStream("deflate");
  const buf = await new Response(new Blob([bytes]).stream().pipeThrough(ds)).arrayBuffer();
  return new Uint8Array(buf);
}
async function decodeField(f, n) {
  if (!f) return null;
  if (f.enc === "rle") {
    const out = new Uint8Array(n); let k = 0, v = 0;
    for (const r of f.runs) { if (v) out.fill(1, k, k + r); k += r; v ^= 1; }
    return out;
  }
  let bytes = b64bytes(f.b64);
  if (f.enc === "u8z") bytes = await inflate(bytes);
  return bytes;
}
async function decodeSnap(snap) {
  const key = `${snap.run}/${snap.seq}/${snap.t_sim}`;
  if (S.decoded.has(key)) return S.decoded.get(key);
  const g = snap.grid, n = g.nx * g.nz, L = snap.layers || {};
  const d = { grid: g, l2: await decodeField(L.l2_occ, n), l3: await decodeField(L.l3_occ, n), h2: await decodeField(L.l2_head, n),
              h3: await decodeField(L.l3_head, n), unk: await decodeField(L.unknown, n) };
  d.unkCells = d.unk ? d.unk.reduce((a, v) => a + v, 0) : 0;
  S.decoded.set(key, d);
  if (S.decoded.size > 64) S.decoded.delete(S.decoded.keys().next().value);
  return d;
}

// ------------------------------------------------------------------------------------------------ axis + time
function snapsList() { return (S.index && S.index.snapshots) || []; }
function axisVal(s) { return S.axis === "t_obs" ? s.t_obs : s.t_sim; }
function pickAxis() {
  const L = snapsList();
  const metaAxis = S.index && S.index.meta && S.index.meta.axis;
  if (metaAxis === "t_sim" || metaAxis === "t_obs") { S.axis = metaAxis; return; }
  if (!L.length) { S.axis = "t_sim"; return; }
  const span = L[L.length - 1].t_obs - L[0].t_obs;
  S.axis = span > 0 && L[0].t_obs > 0 ? "t_obs" : "t_sim";
}
function axisRange() {
  const L = snapsList();
  if (!L.length) return [0, 1];
  let a = axisVal(L[0]), b = axisVal(L[L.length - 1]);
  if (S.axis === "t_obs") a = Math.max(a, 60);
  if (b <= a) b = a + 1;
  return [a, b];
}
const isLog = () => S.axis === "t_obs";
function uToVal(u) {
  const [a, b] = axisRange();
  return isLog() ? Math.exp(Math.log(a) + u * (Math.log(b) - Math.log(a))) : a + u * (b - a);
}
function valToU(v) {
  const [a, b] = axisRange();
  if (isLog()) return (Math.log(Math.max(v, a)) - Math.log(a)) / (Math.log(b) - Math.log(a));
  return (v - a) / (b - a);
}
function snapIndexAt(u) {
  const L = snapsList(); if (!L.length) return -1;
  const v = uToVal(u) + 1e-6;
  let lo = 0, hi = L.length - 1, ans = 0;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (axisVal(L[m]) <= v) { ans = m; lo = m + 1; } else hi = m - 1; }
  return ans;
}
// events carry t_sim; map to the axis via the snapshots (t_obs grows with t_sim)
function evAxis(ev) {
  if (S.axis === "t_sim") return ev.t_sim;
  if (ev.t_obs != null) return ev.t_obs;
  const L = snapsList(); if (!L.length) return 0;
  if (ev.t_sim <= L[0].t_sim) return L[0].t_obs - (L[0].t_sim - ev.t_sim);
  for (let i = 1; i < L.length; i++) {
    if (ev.t_sim <= L[i].t_sim) {
      const a = L[i - 1], b = L[i], f = (ev.t_sim - a.t_sim) / Math.max(b.t_sim - a.t_sim, 1e-9);
      return a.t_obs + f * (b.t_obs - a.t_obs);
    }
  }
  const z = L[L.length - 1]; return z.t_obs + (ev.t_sim - z.t_sim);
}
function fmtClock(t) {
  const d = Math.floor(t / 86400), r = t - d * 86400;
  const hh = Math.floor(r / 3600), mm = Math.floor((r % 3600) / 60);
  return `D${d + 1} ${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
}
function tInstall() { const m = S.index && S.index.meta; return m && m.t_install != null ? m.t_install : null; }
function fmtRel(t) {
  const d = t - tInstall(), a = Math.abs(d), sg = d < 0 ? "−" : "+";
  return a < 3600 ? `T${sg}${Math.round(a / 60)} min` : `T${sg}${(a / 3600).toFixed(1)} h`;
}
function fmtAxisVal(v) { return isLog() ? fmtDur(v) : tInstall() != null ? fmtRel(v) : fmtClock(v); }
function fmtDur(s) {
  if (s < 3600) return `${Math.round(s / 60)} min`;
  if (s < 86400) return `${(s / 3600).toFixed(s < 36000 ? 1 : 0)} h`;
  const d = Math.floor(s / 86400), h = Math.round((s - d * 86400) / 3600);
  return h ? `${d} d ${h} h` : `${d} d`;
}
function fmtNum(v) {
  if (v == null || !isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (Number.isInteger(v) || a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  if (a >= 0.01) return v.toFixed(3);
  return v.toPrecision(2);
}

// ------------------------------------------------------------------------------------------------ plant view
const cv = $("plant"), ctx = cv.getContext("2d");
let hatch = null;
function resizeCanvas(c) {
  const r = c.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  return { w, h, dpr };
}
function worldBounds() {
  const L = S.layout;
  const x0 = Math.min(L.hall.x[0], L.data_hall.x[0]) - 1.5, x1 = Math.max(L.hall.x[1], L.data_hall.x[1]) + 1.5;
  const z0 = Math.min(L.hall.z[0], L.data_hall.z[0]) - 1.5, z1 = Math.max(L.hall.z[1], L.data_hall.z[1]) + 1.5;
  return { x0, x1, z0, z1 };
}
function computeView(w, h) {
  const B = worldBounds(), pad = 14 * (window.devicePixelRatio || 1);
  const topPad = 64 * Math.min(window.devicePixelRatio || 1, 2);
  const sc = Math.min((w - 2 * pad) / (B.x1 - B.x0), (h - topPad - pad - 6) / (B.z1 - B.z0));
  const ox = (w - sc * (B.x1 - B.x0)) / 2 - B.x0 * sc;
  const oy = topPad + (h - topPad - pad - sc * (B.z1 - B.z0)) / 2 + B.z1 * sc;   // z up
  return { sc, X: (x) => ox + x * sc, Y: (z) => oy - z * sc, ox, oy };
}
function rectPath(c, r, V) {
  const cs = Math.cos(r.yaw || 0), sn = Math.sin(r.yaw || 0), hw = r.w / 2, hd = r.d / 2;
  // yaw = atan2(dx, dz): local x axis rotates with yaw (rect w along x at yaw 0)
  const pts = [[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]].map(([a, b]) => [r.cx + a * cs + b * sn, r.cz - a * sn + b * cs]);
  c.beginPath(); pts.forEach(([x, z], i) => (i ? c.lineTo(V.X(x), V.Y(z)) : c.moveTo(V.X(x), V.Y(z)))); c.closePath();
}
const L0_STYLE = {
  wall: ["#c9d2dc", null], conveyor: ["#2a3340", "#4a5666"], robot_base: ["#56606d", null], tip_dresser: ["#434c58", null],
  curtain_post: ["#fab219", null], lifter_post: ["#7d8794", null], fixture_table: ["#2d3642", "#56606d"], column: ["#3d4652", null],
  rack_market: ["#1f2731", "#3a4553"], rack_dh: ["#1f2731", "#3a4553"], crah: ["#26303b", "#45505e"],
};

function buildRasters(d) {
  const g = d.grid, mk = (occ, rgb, gain) => {
    if (!occ) return null;
    const c = document.createElement("canvas"); c.width = g.nx; c.height = g.nz;
    const im = c.getContext("2d").createImageData(g.nx, g.nz);
    for (let i = 0; i < g.nz; i++) for (let j = 0; j < g.nx; j++) {
      const v = occ[i * g.nx + j] / 255; const o = ((g.nz - 1 - i) * g.nx + j) * 4;   // flip: row 0 = z0 at bottom
      if (v <= 0) continue;
      const a = Math.min(1, Math.pow(v, 0.55) * gain);
      im.data[o] = rgb[0]; im.data[o + 1] = rgb[1]; im.data[o + 2] = rgb[2]; im.data[o + 3] = Math.round(a * 235);
    }
    c.getContext("2d").putImageData(im, 0, 0); return c;
  };
  let unk = null;
  if (d.unk && d.unkCells) {
    unk = document.createElement("canvas"); unk.width = g.nx; unk.height = g.nz;
    const im = unk.getContext("2d").createImageData(g.nx, g.nz);
    for (let i = 0; i < g.nz; i++) for (let j = 0; j < g.nx; j++) if (d.unk[i * g.nx + j]) {
      const o = ((g.nz - 1 - i) * g.nx + j) * 4; im.data[o] = im.data[o + 1] = im.data[o + 2] = 255; im.data[o + 3] = 255;
    }
    unk.getContext("2d").putImageData(im, 0, 0);
  }
  return { l2: mk(d.l2, [57, 135, 229], 1.15), l3: mk(d.l3, [31, 191, 134], 1.25), unk, d };
}
function getHatch() {
  if (hatch) return hatch;
  const p = document.createElement("canvas"); p.width = p.height = 8; const c = p.getContext("2d");
  c.strokeStyle = "rgba(199,204,212,.85)"; c.lineWidth = 1.6;
  c.beginPath(); c.moveTo(-2, 10); c.lineTo(10, -2); c.moveTo(6, 10); c.lineTo(10, 6); c.moveTo(-2, 2); c.lineTo(2, -2); c.stroke();
  return (hatch = ctx.createPattern(p, "repeat"));
}

function machinePhase(mid) {
  // phase fraction 0..1 or null (stopped / unknown); cycle preview advances it with the learned period
  const s = S.cur; if (!s) return null;
  const b = s.phase_now ? s.phase_now[mid] : null;
  const stop = (S.index.meta && S.index.meta.stop_bin) ?? 16, nb = (S.index.meta && S.index.meta.phase_bins) || 16;
  const per = periodOf(mid);
  if (b == null && S.layers.cycle && per) {            // phase not streamed: preview the learned cycle from its start
    const g = ((nowS() - S.anchor.wall) * CYCLE_RATE) / per; return g - Math.floor(g);
  }
  if (b == null || b < 0 || b >= stop) return null;
  let f = (b + 0.5) / nb;
  if (S.layers.cycle && per) f += ((nowS() - S.anchor.wall) * CYCLE_RATE) / per;
  return f - Math.floor(f);
}
function periodOf(mid) {
  const m = S.index && S.index.meta;
  if (m && m.period_s && m.period_s[mid]) return m.period_s[mid];
  if (S.cur && S.cur.metrics && S.cur.metrics[`period_${mid}`]) return S.cur.metrics[`period_${mid}`];
  if (S.phases && S.phases.machines[mid]) return S.phases.machines[mid].period_s;
  const lm = S.layout.machines.find((x) => x.id === mid);
  return lm ? lm.period_nominal : null;
}
function phaseBinNow(mid) {
  const f = machinePhase(mid), nb = (S.index.meta && S.index.meta.phase_bins) || 16, stop = (S.index.meta && S.index.meta.stop_bin) ?? 16;
  if (f == null) { const b = S.cur && S.cur.phase_now ? S.cur.phase_now[mid] : null; return b == null ? null : stop; }
  return Math.min(nb - 1, Math.floor(f * nb));
}

function drawPlant() {
  const { w, h, dpr } = resizeCanvas(cv);
  const V = (S.view = computeView(w, h));
  const L = S.layout;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#0c1015"; ctx.fillRect(0, 0, w, h);
  // floor
  ctx.fillStyle = "#131a22";
  for (const a of [L.hall, L.data_hall]) ctx.fillRect(V.X(a.x[0]), V.Y(a.z[1]), (a.x[1] - a.x[0]) * V.sc, (a.z[1] - a.z[0]) * V.sc);
  // 12 m column grid + scale bar
  ctx.strokeStyle = "rgba(120,140,160,.07)"; ctx.lineWidth = 1;
  for (let x = 0; x <= 120; x += 6) { ctx.beginPath(); ctx.moveTo(V.X(x), V.Y(L.hall.z[0])); ctx.lineTo(V.X(x), V.Y(L.hall.z[1])); ctx.stroke(); }
  for (let z = 0; z <= 48; z += 6) { ctx.beginPath(); ctx.moveTo(V.X(L.hall.x[0]), V.Y(z)); ctx.lineTo(V.X(L.data_hall.x[1]), V.Y(z)); ctx.stroke(); }
  // learned rasters
  const R = S.rasters;
  if (R) {
    const g = R.d.grid;
    const gx = V.X(g.x0), gy = V.Y(g.z0 + g.nz * g.res), gw = g.nx * g.res * V.sc, gh = g.nz * g.res * V.sc;
    ctx.imageSmoothingEnabled = false;
    if (S.layers.l2 && R.l2) ctx.drawImage(R.l2, gx, gy, gw, gh);
    if (S.layers.l3 && R.l3) { ctx.globalCompositeOperation = "lighter"; ctx.drawImage(R.l3, gx, gy, gw, gh); ctx.globalCompositeOperation = "source-over"; }
    if (S.layers.unk && R.unk) {
      // mask -> hatched overlay: draw the mask, then fill the hatch pattern only where the mask is
      const o = offscreen(w, h, 4), off = o.canvas; o.globalCompositeOperation = "source-over"; o.clearRect(0, 0, w, h);
      o.imageSmoothingEnabled = false; o.drawImage(R.unk, gx, gy, gw, gh);
      o.globalCompositeOperation = "source-in"; o.fillStyle = "rgba(60,64,72,.55)"; o.fillRect(0, 0, w, h);
      o.globalCompositeOperation = "source-atop"; o.fillStyle = getHatch(); o.fillRect(0, 0, w, h);
      ctx.drawImage(off, 0, 0);
    }
    if (S.layers.dir) drawArrows(V, R.d);
  }
  // L0 static bodies
  for (const r of L.rects) {
    const st = L0_STYLE[r.cls] || ["#3a4350", null];
    rectPath(ctx, r, V); ctx.fillStyle = st[0]; ctx.fill();
    if (st[1]) { ctx.strokeStyle = st[1]; ctx.lineWidth = 1; ctx.stroke(); }
  }
  // hazard zones
  ctx.setLineDash([5 * dpr, 4 * dpr]); ctx.lineWidth = 1 * dpr;
  for (const m of L.machines) {
    const [x0, z0, x1, z1] = m.rect;
    ctx.strokeStyle = m.id === S.machine ? "rgba(240,128,60,.9)" : "rgba(139,151,166,.55)";
    ctx.strokeRect(V.X(x0), V.Y(z1), (x1 - x0) * V.sc, (z1 - z0) * V.sc);
  }
  ctx.setLineDash([]);
  // L1 envelopes: boxes overlap, so each set is painted opaque off-screen and composited once with its alpha
  if (S.layers.l1 && S.cur && S.cur.l1) {
    const stop = (S.index.meta && S.index.meta.stop_bin) ?? 16;
    const all = offscreen(w, h, 0), now = offscreen(w, h, 1), nw = offscreen(w, h, 2), st = offscreen(w, h, 3);
    for (const c of [all, now, nw, st]) c.clearRect(0, 0, w, h);
    all.fillStyle = now.fillStyle = "#f0803c"; nw.fillStyle = "#fab219"; st.fillStyle = "#d03b3b";
    for (const [mid, rows] of Object.entries(S.cur.l1)) {
      const bnow = phaseBinNow(mid), isNew = !L.machines.some((m) => m.id === mid);
      const tgt = bnow === stop ? st : isNew ? nw : now;
      for (const [b, x0, z0, x1, z1] of rows) {
        const r = [V.X(x0), V.Y(z1), (x1 - x0) * V.sc, (z1 - z0) * V.sc];
        all.fillRect(...r);
        if (b === bnow) tgt.fillRect(...r);
      }
      if (isNew && rows.length) S._newLabels = (S._newLabels || []).concat([[mid, Math.min(...rows.map((r) => r[1])), Math.max(...rows.map((r) => r[4]))]]);
    }
    ctx.globalAlpha = 0.2; ctx.drawImage(all.canvas, 0, 0);
    ctx.globalAlpha = 0.9; ctx.drawImage(now.canvas, 0, 0); ctx.drawImage(nw.canvas, 0, 0);
    ctx.globalAlpha = 0.6; ctx.drawImage(st.canvas, 0, 0); ctx.globalAlpha = 1;
    for (const [mid, x, z] of S._newLabels || []) label(V.X(x), V.Y(z) - 6 * dpr, `${mid} · learned`, "#fab219", dpr);
    S._newLabels = null;
  }
  // cameras
  ctx.fillStyle = "rgba(169,180,194,.55)";
  for (const [x, z] of L.cameras) { ctx.beginPath(); ctx.arc(V.X(x), V.Y(z), 2.2 * dpr, 0, 7); ctx.fill(); }
  // machine labels
  ctx.font = `600 ${11 * dpr}px ${getComputedStyle(document.body).getPropertyValue("--mono")}`;
  for (const m of L.machines) {
    const [x0, , , z1] = m.rect;
    ctx.fillStyle = m.id === S.machine ? "#ffb27a" : "rgba(169,180,194,.85)";
    ctx.fillText(m.id, V.X(x0) + 3 * dpr, V.Y(z1) + 12 * dpr);
  }
  ctx.font = `${10 * dpr}px ${getComputedStyle(document.body).getPropertyValue("--sans")}`;
  ctx.fillStyle = "rgba(111,123,139,.9)";
  ctx.fillText("BODY SHOP", V.X(L.hall.x[0]) + 4 * dpr, V.Y(L.hall.z[1]) - 4 * dpr);
  ctx.fillText("DATA HALL", V.X(L.data_hall.x[0]) + 4 * dpr, V.Y(L.data_hall.z[1]) - 4 * dpr);
  // event pulses (events with a location near the cursor)
  drawEventMarks(V, dpr);
  // scale bar
  const sx = V.X(L.hall.x[0]), sy = V.Y(L.hall.z[0]) + 12 * dpr;
  ctx.strokeStyle = "#6f7b8b"; ctx.lineWidth = 1.5 * dpr; ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx + 10 * V.sc, sy); ctx.stroke();
  ctx.fillStyle = "#6f7b8b"; ctx.fillText("10 m", sx + 10 * V.sc + 5 * dpr, sy + 3 * dpr);
  ctx.fillText("x →", V.X(L.data_hall.x[1]) - 22 * dpr, sy + 3 * dpr);
}
const _off = [];
function offscreen(w, h, i) {
  if (!_off[i]) _off[i] = document.createElement("canvas");
  const c = _off[i]; if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  return c.getContext("2d");
}
function label(x, y, txt, col, dpr) {
  ctx.font = `600 ${11 * dpr}px ${getComputedStyle(document.body).getPropertyValue("--mono")}`;
  const w = ctx.measureText(txt).width + 8 * dpr;
  ctx.fillStyle = "rgba(10,13,17,.85)"; ctx.fillRect(x, y - 13 * dpr, w, 16 * dpr);
  ctx.fillStyle = col; ctx.fillText(txt, x + 4 * dpr, y);
}
function drawArrows(V, d) {
  const g = d.grid;
  for (const [occ, hd, col] of [[d.l2, d.h2, "rgba(150,195,255,.9)"], [d.l3, d.h3, "rgba(140,240,200,.9)"]]) {
    if (!occ || !hd) continue;
    ctx.strokeStyle = col; ctx.lineWidth = 1.2;
    for (let i = 0; i < g.nz; i++) for (let j = 0; j < g.nx; j++) {
      const k = i * g.nx + j, hb = hd[k];
      if (hb === 255 || occ[k] < 30) continue;
      const th = hb * Math.PI / 4, cx = g.x0 + (j + 0.5) * g.res, cz = g.z0 + (i + 0.5) * g.res, L = g.res * 0.38;
      const dx = Math.sin(th) * L, dz = Math.cos(th) * L;
      ctx.beginPath(); ctx.moveTo(V.X(cx - dx), V.Y(cz - dz)); ctx.lineTo(V.X(cx + dx), V.Y(cz + dz));
      const ax = V.X(cx + dx), ay = V.Y(cz + dz), ang = Math.atan2(-(dz), dx);
      ctx.moveTo(ax, ay); ctx.lineTo(ax - Math.cos(ang - 0.5) * 3.5, ay + Math.sin(ang - 0.5) * 3.5);
      ctx.moveTo(ax, ay); ctx.lineTo(ax - Math.cos(ang + 0.5) * 3.5, ay + Math.sin(ang + 0.5) * 3.5);
      ctx.stroke();
    }
  }
}
function evXZ(ev) {
  if (ev.x != null && ev.z != null) return [ev.x, ev.z];
  const bx = ev.box || ev.rect;
  if (bx) return [(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2];
  const m = ev.machine && S.layout.machines.find((q) => q.id === ev.machine);
  if (m) return [(m.rect[0] + m.rect[2]) / 2, (m.rect[1] + m.rect[3]) / 2];
  return null;
}
function drawEventMarks(V, dpr) {
  const evs = (S.index && S.index.events) || []; if (!evs.length) return;
  const v = uToVal(S.u), [a, b] = axisRange(), win = (isLog() ? v * 0.25 : (b - a) * 0.04);
  for (const ev of evs) {
    const p = evXZ(ev); if (!p) continue;
    const t = evAxis(ev); if (t > v || v - t > win) continue;
    const f = (v - t) / win, r = (8 + 22 * f) * dpr;
    ctx.strokeStyle = `rgba(250,178,25,${0.9 * (1 - f)})`; ctx.lineWidth = 2 * dpr;
    ctx.beginPath(); ctx.arc(V.X(p[0]), V.Y(p[1]), r, 0, 7); ctx.stroke();
  }
}

// ------------------------------------------------------------------------------------------------ side panels
function machineIds() {
  const ids = S.layout.machines.map((m) => m.id);
  if (S.cur && S.cur.l1) for (const k of Object.keys(S.cur.l1)) if (!ids.includes(k)) ids.push(k);
  return ids;
}
function renderMachines() {
  const box = $("machines"), ids = machineIds();
  if (box.dataset.ids !== ids.join(",")) {
    box.dataset.ids = ids.join(",");
    box.innerHTML = ids.map((id) => {
      const m = S.layout.machines.find((q) => q.id === id);
      const cells = Array.from({ length: 16 }, () => "<i></i>").join("") + '<i class="gap"></i><i class="stop" title="stopped"></i>';
      return `<div class="mrow" role="option" data-id="${id}" aria-selected="${id === S.machine}">
        <span class="id ${m ? "" : "new"}">${id}<small>${m ? m.kind.replace(/_/g, " ") : "new (not in L0)"}</small></span>
        <span class="pbar">${cells}</span><span class="per"></span></div>`;
    }).join("");
    box.querySelectorAll(".mrow").forEach((el) => el.addEventListener("click", () => selectMachine(el.dataset.id)));
  }
  const stop = (S.index && S.index.meta && S.index.meta.stop_bin) ?? 16;
  box.querySelectorAll(".mrow").forEach((el) => {
    const id = el.dataset.id, b = S.index ? phaseBinNow(id) : null, cells = el.querySelectorAll(".pbar i");
    cells.forEach((c, i) => c.classList.toggle("on", (i < 16 && i === b) || (i === 17 && b === stop)));
    const per = S.index ? periodOf(id) : null;
    el.querySelector(".per").textContent = per ? `${per.toFixed(1)} s` : "–";
    el.setAttribute("aria-selected", String(id === S.machine));
  });
}
function selectMachine(id) { S.machine = id; renderMachines(); }

function evText(ev) {
  if (ev.caption) return ev.caption;
  if (ev.kind === "change_detected") return `Candidate ${ev.candidate ?? ""}: persistent motion where L0 is static${ev.cells != null ? ` (${ev.cells} cells)` : ""} → cells UNKNOWN (= occupied)`;
  if (ev.kind === "change_cleared") return `Candidate ${ev.candidate ?? ""} cleared: the motion did not persist, cells released`;
  if (ev.kind === "machine_spawned") return `${ev.machine || "new machine"}: cycle learned${ev.period_s ? ` (period ${fmtNum(ev.period_s)} s)` : ""}${(ev.cycles ?? ev.n_cycles) ? ` after ${ev.cycles ?? ev.n_cycles} stable cycles` : ""} → L1 model spawned, cells stay UNKNOWN until it converges`;
  if (ev.kind === "machine_converged") return `${ev.machine || "new machine"}: period converged${ev.period_s ? ` (${fmtNum(ev.period_s)} s)` : ""} → envelope settles, UNKNOWN released`;
  if (ev.kind === "camera_offline") return `Camera ${ev.camera ?? ""} offline → its cells UNKNOWN (= occupied)`;
  const f = { ...ev }; delete f.schema; delete f.seq; delete f.kind; delete f.t_sim; delete f.t_obs;
  if (ev.kind === "phase_named" && ev.name) return `${ev.machine || ""} phase ${ev.bin ?? ev.phase ?? ""} named “${ev.name}”`;
  if (ev.label) return ev.label;
  return Object.entries(f).map(([k, v]) => `${k}=${typeof v === "number" ? fmtNum(v) : JSON.stringify(v)}`).join(" · ");
}
function renderEvents() {
  const evs = (S.index && S.index.events) || [], box = $("events");
  $("evCount").textContent = evs.length ? `${evs.length} events` : "";
  if (box.dataset.n !== String(evs.length) || box.dataset.run !== String(S.run && S.run.id)) {
    box.dataset.n = String(evs.length); box.dataset.run = String(S.run && S.run.id);
    box.innerHTML = evs.length ? evs.map((ev, i) => `<div class="ev" data-i="${i}"><span class="t">${fmtClock(ev.t_sim)}</span>
        <div><span class="kind ${ev.kind}">${KIND_LABEL[ev.kind] || ev.kind}</span><div class="txt">${escapeHtml(evText(ev))}</div></div></div>`).join("")
      : '<div class="empty">No events in this run yet. Change detection, new machines, discovered phases and Gemma labels appear here.</div>';
    box.querySelectorAll(".ev").forEach((el) => el.addEventListener("click", () => { const ev = evs[+el.dataset.i]; seekU(valToU(evAxis(ev)) + 1e-4); }));
  }
  const v = uToVal(S.u) + 1e-6; let cur = -1;
  box.querySelectorAll(".ev").forEach((el) => {
    const past = evAxis(evs[+el.dataset.i]) <= v; el.classList.toggle("past", past); if (past) cur = +el.dataset.i;
  });
  box.querySelectorAll(".ev").forEach((el) => el.classList.toggle("cur", +el.dataset.i === cur));
  // caption: the latest past event (Gemma labels and new-machine steps are the video captions)
  const capEl = $("caption");
  const ev = cur >= 0 ? evs[cur] : null;
  const [a, b] = axisRange(), win = isLog() ? v * 0.5 : (b - a) * 0.12;
  if (ev && v - evAxis(ev) <= win && ev.kind !== "production_state") {
    $("capKind").textContent = `${KIND_LABEL[ev.kind] || ev.kind} · ${fmtClock(ev.t_sim)}`; $("capTxt").textContent = evText(ev); capEl.style.opacity = 1;
    capEl.style.borderLeftColor = ev.kind === "change_detected" ? "#fab219" : ev.kind === "camera_offline" ? "#d03b3b" : ev.kind === "machine_spawned" ? "#f0803c" : "#5aa9ff";
  } else capEl.style.opacity = 0;
  if (cur >= 0 && cur !== S.lastCaptionSeq && S.playing) {
    const el = box.querySelector(`.ev[data-i="${cur}"]`); if (el) el.scrollIntoView({ block: "nearest" });
  }
  S.lastCaptionSeq = cur;
}
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]); }

// ------------------------------------------------------------------------------------------------ track + metrics
const tcv = $("trackCv"), tctx = tcv.getContext("2d");
function axisTicks() {
  const [a, b] = axisRange(), out = [];
  if (isLog()) {
    for (const [t, l] of [[3600, "1 h"], [6 * 3600, "6 h"], [86400, "1 day"], [3 * 86400, "3 days"], [7 * 86400, "1 week"]]) if (t >= a * 0.999 && t <= b * 1.001) out.push([t, l]);
  } else {
    const span = b - a, step = span > 4 * 86400 ? 86400 : span > 30 * 3600 ? 6 * 3600 : span > 6 * 3600 ? 3 * 3600 : 3600;
    const ti = tInstall();
    if (ti != null) {
      const st = span > 12 * 3600 ? 6 * 3600 : span > 4 * 3600 ? 3600 : span > 3600 ? 1800 : 600;
      for (let t = ti + Math.ceil((a - ti) / st) * st; t <= b; t += st) out.push([t, Math.abs(t - ti) < 1 ? "install" : fmtRel(t)]);
    } else for (let t = Math.ceil(a / step) * step; t <= b; t += step) out.push([t, span > 30 * 3600 ? fmtClock(t) : fmtClock(t).slice(3)]);
  }
  return out;
}
function drawTrack() {
  const { w, h, dpr } = resizeCanvas(tcv), c = tctx, pad = 8 * dpr, y = h * 0.56;
  c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, w, h);
  const X = (u) => pad + u * (w - 2 * pad);
  c.fillStyle = "#222b36"; c.fillRect(pad, y - 2 * dpr, w - 2 * pad, 4 * dpr);
  c.fillStyle = "#2d5a8a"; c.fillRect(pad, y - 2 * dpr, X(S.u) - pad, 4 * dpr);
  // snapshots as faint ticks (production-state colored)
  for (const s of snapsList()) {
    const u = valToU(axisVal(s)); c.fillStyle = PROD_COLOR[s.prod_state] || "#6f7b8b"; c.globalAlpha = 0.55;
    c.fillRect(X(u) - 0.5 * dpr, y + 5 * dpr, 1 * dpr, 4 * dpr); c.globalAlpha = 1;
  }
  // new-machine install (truth-side marker: the observer is not told)
  if (tInstall() != null && !isLog()) {
    const x = X(Math.min(1, Math.max(0, valToU(tInstall()))));
    c.fillStyle = "#fab219"; c.fillRect(x - 1 * dpr, y - 26 * dpr, 2 * dpr, 34 * dpr);
    c.font = `${10 * dpr}px ${getComputedStyle(document.body).getPropertyValue("--mono")}`;
    c.fillText(`${(S.index.meta && S.index.meta.machine) || "new machine"} installed (observer not told)`, x + 5 * dpr, y - 18 * dpr);
  }
  // events as diamonds
  for (const ev of (S.index && S.index.events) || []) {
    if (ev.kind === "production_state") continue;
    const x = X(Math.min(1, Math.max(0, valToU(evAxis(ev))))), s = 4 * dpr;
    c.fillStyle = ev.kind === "change_detected" ? "#fab219" : ev.kind === "camera_offline" ? "#d03b3b" : ev.kind === "machine_spawned" ? "#f0803c" : "#5aa9ff";
    c.beginPath(); c.moveTo(x, y - 8 * dpr - s); c.lineTo(x + s, y - 8 * dpr); c.lineTo(x, y - 8 * dpr + s); c.lineTo(x - s, y - 8 * dpr); c.fill();
  }
  c.font = `${10 * dpr}px ${getComputedStyle(document.body).getPropertyValue("--mono")}`; c.fillStyle = "#6f7b8b"; c.textAlign = "center";
  for (const [t, l] of axisTicks()) { const x = X(valToU(t)); c.fillRect(x - 0.5 * dpr, y + 3 * dpr, 1 * dpr, 7 * dpr); c.fillText(l, Math.min(Math.max(x, 20 * dpr), w - 24 * dpr), h - 3 * dpr); }
  c.textAlign = "start";
  // playhead
  const px = X(S.u); c.fillStyle = "#e6ebf1"; c.beginPath(); c.arc(px, y, 6 * dpr, 0, 7); c.fill();
  c.strokeStyle = "#0a0d11"; c.lineWidth = 2 * dpr; c.stroke();
  $("track").setAttribute("aria-valuenow", String(Math.round(S.u * 1000)));
}
function metricKeys() {
  const L = snapsList(), keys = new Set();
  for (const s of L) for (const [k, v] of Object.entries(s.metrics || {})) if (typeof v === "number" && !k.startsWith("period_") && k !== "hours") keys.add(k);
  const arr = [...keys];
  arr.sort((a, b) => { const ia = METRIC_PREF.indexOf(a), ib = METRIC_PREF.indexOf(b); return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b); });
  return arr.slice(0, 4);
}
function renderMetricCards() {
  const keys = metricKeys(), box = $("metrics");
  if (box.dataset.keys === keys.join(",")) return;
  box.dataset.keys = keys.join(",");
  box.innerHTML = keys.length ? keys.map((k) => `<div class="mcard" data-k="${k}"><div class="mt"><span>${METRIC_NAME[k] || k.replace(/_/g, " ")}</span><span class="mx"></span></div>
      <div class="mv">–</div><canvas></canvas></div>`).join("") : '<div class="mcard"><div class="empty">No metrics in this run.</div></div>';
  box.querySelectorAll(".mcard canvas").forEach((c) => {
    c.addEventListener("pointermove", (e) => { c._hover = e.offsetX / c.clientWidth; });
    c.addEventListener("pointerleave", () => { c._hover = null; });
    c.addEventListener("click", (e) => seekU(e.offsetX / c.clientWidth));
  });
}
function drawMetrics() {
  const L = snapsList(); const cur = snapIndexAt(S.u);
  $("metrics").querySelectorAll(".mcard[data-k]").forEach((card) => {
    const k = card.dataset.k, c = card.querySelector("canvas"), { w, h, dpr } = resizeCanvas(c), g = c.getContext("2d");
    g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, w, h);
    const pts = L.map((s) => [valToU(axisVal(s)), s.metrics ? s.metrics[k] : null]).filter((p) => typeof p[1] === "number");
    if (!pts.length) return;
    let lo = Math.min(...pts.map((p) => p[1])), hi = Math.max(...pts.map((p) => p[1]));
    if (/iou|auroc|precision|recall/.test(k) && lo >= 0 && hi <= 1) { lo = Math.min(/auroc/.test(k) ? 0.5 : 0, lo); hi = 1; }
    else if (lo >= 0) { lo = 0; hi = hi * 1.15; }
    if (hi - lo < 1e-12) { hi = lo + 1; }
    const pt = 4 * dpr, pb = 4 * dpr, Y = (v) => h - pb - ((v - lo) / (hi - lo)) * (h - pt - pb), X = (u) => u * w;
    g.strokeStyle = "rgba(120,140,160,.12)"; g.lineWidth = 1; g.beginPath(); g.moveTo(0, Y(lo)); g.lineTo(w, Y(lo)); g.stroke();
    g.strokeStyle = "#5aa9ff"; g.lineWidth = 2 * dpr; g.lineJoin = "round"; g.beginPath();
    pts.forEach(([u, v], i) => (i ? g.lineTo(X(u), Y(v)) : g.moveTo(X(u), Y(v)))); g.stroke();
    // cursor
    const cx = X(S.u); g.strokeStyle = "rgba(230,235,241,.7)"; g.lineWidth = 1 * dpr; g.beginPath(); g.moveTo(cx, 0); g.lineTo(cx, h); g.stroke();
    const vNow = cur >= 0 && L[cur].metrics ? L[cur].metrics[k] : null;
    if (typeof vNow === "number") { g.fillStyle = "#e6ebf1"; g.beginPath(); g.arc(X(valToU(axisVal(L[cur]))), Y(vNow), 4 * dpr, 0, 7); g.fill(); }
    card.querySelector(".mv").textContent = fmtNum(vNow);
    card.querySelector(".mx").textContent = `${fmtNum(lo)}–${fmtNum(hi)}`;
    // hover crosshair + tooltip
    let tip = card.querySelector(".mtip");
    if (c._hover != null) {
      const near = pts.reduce((best, p) => (Math.abs(p[0] - c._hover) < Math.abs(best[0] - c._hover) ? p : best), pts[0]);
      g.strokeStyle = "rgba(90,169,255,.6)"; g.setLineDash([3 * dpr, 3 * dpr]); g.beginPath(); g.moveTo(X(near[0]), 0); g.lineTo(X(near[0]), h); g.stroke(); g.setLineDash([]);
      if (!tip) { tip = document.createElement("div"); tip.className = "mtip"; card.appendChild(tip); }
      const val = uToVal(near[0]);
      tip.textContent = `${fmtAxisVal(val)} · ${fmtNum(near[1])}`;
      tip.style.left = `${Math.min(near[0] * c.clientWidth + 8, c.clientWidth - 90)}px`; tip.style.top = "6px";
    } else if (tip) tip.remove();
  });
}

// ------------------------------------------------------------------------------------------------ phase strip
const scv = $("strip"), sctx = scv.getContext("2d");
function binAreas(mid) {
  const meta = S.index && S.index.meta;
  if (meta && meta.l1_area_m2 && meta.l1_area_m2[mid]) return meta.l1_area_m2[mid];
  const rows = (S.cur && S.cur.l1 && S.cur.l1[mid]) || [], out = new Array(17).fill(0);
  for (const [b, x0, z0, x1, z1] of rows) if (b >= 0 && b < 17) out[b] += (x1 - x0) * (z1 - z0);
  return out;
}
function namedPhases(mid) {
  // discovered sub-phases: meta.phases[mid] = [{name, start, end}] (cycle fractions) and/or phase_named events up to the cursor
  const meta = S.index && S.index.meta, out = [];
  if (meta && meta.phases && meta.phases[mid]) for (const p of meta.phases[mid]) out.push({ ...p });
  else if (S.phases && S.phases.machines[mid]) for (const p of S.phases.machines[mid].segments) out.push({ ...p });
  const v = uToVal(S.u) + 1e-6;
  for (const ev of (S.index && S.index.events) || []) {
    if ((ev.kind !== "phase_discovered" && ev.kind !== "phase_named") || ev.machine !== mid || evAxis(ev) > v) continue;
    const byBin = ev.bin != null ? out.find((p) => p.bin === ev.bin) : null;
    if (byBin) { if (ev.name || ev.label) { byBin.name = ev.name || ev.label; byBin.named = true; } continue; }
    const s = ev.start ?? (ev.bin != null ? ev.bin / 16 : null), e = ev.end ?? (ev.bin != null ? (ev.bin + 1) / 16 : null);
    if (s == null) continue;
    const ex = out.find((p) => Math.abs(p.start - s) < 1e-6);
    if (ex) { if (ev.name) ex.name = ev.name; } else out.push({ name: ev.name || ev.label || "", start: s, end: e });
  }
  return out.sort((a, b) => a.start - b.start);
}
function drawStrip() {
  const { w, h, dpr } = resizeCanvas(scv), c = sctx; c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, w, h);
  const mid = S.machine; if (!mid || !S.index) { $("stripTitle").textContent = "select a machine"; return; }
  const per = periodOf(mid), A = binAreas(mid), nb = 16, stopW = Math.max(0.07 * w, 34 * dpr), gap = 6 * dpr, cw = w - stopW - gap;
  $("stripTitle").textContent = `${mid} · cycle ${per ? per.toFixed(1) + " s" : "unknown"}`;
  const maxA = Math.max(...A, 1e-9), top = 26 * dpr, base = h - 16 * dpr, H = base - top;
  const named = namedPhases(mid), f0 = machinePhase(mid);
  // named / discovered segments band
  c.font = `${10 * dpr}px ${getComputedStyle(document.body).getPropertyValue("--sans")}`;
  if (named.length) {
    named.forEach((p, i) => {
      const x0 = p.start * cw, x1 = (p.end ?? (named[i + 1] ? named[i + 1].start : 1)) * cw;
      const active = f0 != null && f0 >= p.start && f0 < (p.end ?? 1);
      c.fillStyle = active ? "#2d5a8a" : p.name === "idle" ? "#1a2028" : i % 2 ? "#1d2836" : "#22344a";
      c.fillRect(x0, 4 * dpr, Math.max(1, x1 - x0 - 2 * dpr), 16 * dpr);
      c.fillStyle = p.named ? "#ffffff" : "#cfe3ff";
      if (c.measureText(p.name || "?").width + 6 * dpr <= x1 - x0) c.fillText(p.name || "?", x0 + 4 * dpr, 16 * dpr);
    });
  } else { c.fillStyle = "#6f7b8b"; c.fillText("no discovered sub-phases for this machine yet", 0, 15 * dpr); }
  // envelope area per bin (bars, 2px gaps, baseline-anchored)
  const bw = cw / nb, bnow = phaseBinNow(mid);
  for (let b = 0; b < nb; b++) {
    const v = A[b] || 0, bh = Math.max(1, (v / maxA) * H);
    c.fillStyle = b === bnow ? "#f0803c" : "rgba(240,128,60,.32)"; c.fillRect(b * bw + 1 * dpr, base - bh, bw - 2 * dpr, bh);
  }
  const sv = A[16] || 0, sh = Math.max(1, (sv / maxA) * H);
  c.fillStyle = bnow === 16 ? "#d03b3b" : "rgba(208,59,59,.35)"; c.fillRect(cw + gap, base - sh, stopW, sh);
  c.fillStyle = "#6f7b8b"; c.fillText("0", 0, h - 3 * dpr); c.fillText("cycle phase →", cw / 2 - 30 * dpr, h - 3 * dpr); c.fillText("1", cw - 6 * dpr, h - 3 * dpr);
  c.fillText("STOP", cw + gap, h - 3 * dpr);
  // phase-now marker
  const f = machinePhase(mid);
  if (f != null) { const x = f * cw; c.strokeStyle = "#e6ebf1"; c.lineWidth = 2 * dpr; c.beginPath(); c.moveTo(x, top - 4 * dpr); c.lineTo(x, base); c.stroke();
    c.fillStyle = "#e6ebf1"; c.beginPath(); c.moveTo(x - 5 * dpr, top - 8 * dpr); c.lineTo(x + 5 * dpr, top - 8 * dpr); c.lineTo(x, top - 1 * dpr); c.fill(); }
  const streamed = S.cur && S.cur.phase_now && S.cur.phase_now[mid] != null;
  $("stripNote").textContent = `Top band: sub-phases discovered by the observer (workstream P; Gemma names them later). Bars: L1 envelope area per equal-slice bin (m², max ${fmtNum(maxA)}). ` +
    (S.layers.cycle ? (streamed ? `Marker: phase now, advanced at ${CYCLE_RATE}× real time.` : `Marker: cycle preview at ${CYCLE_RATE}× (phase not streamed in this run).`) : "Marker: phase now at the snapshot.");
}

// ------------------------------------------------------------------------------------------------ HUD + cursor
async function applyCursor() {
  const L = snapsList(); if (!L.length) { S.cur = null; S.rasters = null; return; }
  const i = snapIndexAt(S.u), s = L[i];
  if (S.curFile !== s.file) {
    S.curFile = s.file;
    const snap = await loadSnap(s.file);
    if (S.curFile !== s.file) return;                 // a newer seek won
    const d = await decodeSnap(snap);
    if (S.curFile !== s.file) return;
    S.cur = snap; S.rasters = buildRasters(d); S.anchor.wall = nowS();
    // prefetch neighbours
    for (const j of [i + 1, i + 2]) if (L[j] && !S.snaps.has(L[j].file)) loadSnap(L[j].file).catch(() => {});
  }
  const snap = S.cur; if (!snap) return;
  $("hudClock").textContent = fmtClock(snap.t_sim);
  $("hudObs").textContent = tInstall() != null ? fmtRel(snap.t_sim) : fmtDur(snap.t_obs);
  $("hudState").textContent = snap.prod_state.replace("_", " ");
  $("hudStateSw").style.background = PROD_COLOR[snap.prod_state] || "#6f7b8b";
  const d = S.rasters && S.rasters.d;
  $("hudUnk").textContent = d ? `${(d.unkCells * d.grid.res * d.grid.res).toFixed(0)} m²` : "–";
  const v = uToVal(S.u);
  $("readout").innerHTML = isLog() ? `<b>${fmtDur(v)}</b> observed` : tInstall() != null ? `<b>${fmtRel(v)}</b> vs install` : `<b>${fmtClock(v)}</b> sim`;
}
function seekU(u) { S.u = Math.min(1, Math.max(0, u)); S.follow = S.u >= 0.999; return applyCursor(); }

// ------------------------------------------------------------------------------------------------ run loading
async function openRun(run) {
  S.run = run; S.snaps.clear(); S.decoded.clear(); S.cur = null; S.curFile = null; S.rasters = null;
  if (run._bundle) S.index = run._bundle.index;
  else S.index = await getJSON(`${runBase(run)}/index.json`);
  pickAxis();
  const m = S.index.meta || {};
  $("runNote").textContent = `${S.index.label} · run “${S.index.run}”${m.title ? " · " + m.title : ""}${m.note ? " — " + m.note : ""}`;
  S.u = Q.has("u") ? +Q.get("u") : 0;
  if (!S.machine) S.machine = Q.get("machine") || (S.layout.machines.find((x) => x.id === "S30") || S.layout.machines[0]).id;
  renderMetricCards(); await applyCursor(); renderMachines(); renderEvents();
}
async function loadRuns() {
  if (BUNDLE) return BUNDLE.runs.map((r) => ({ ...r.meta, _bundle: r }));
  let runs = [];
  try { runs = await getJSON("runs.json"); } catch (e) { runs = []; }
  if (Q.get("runpath")) { const p = Q.get("runpath").replace(/\/$/, ""); runs.unshift({ id: p.split("/").pop(), title: p.split("/").pop(), path: p }); }
  return runs;
}
function fillRunSelect() {
  const sel = $("runSel");
  sel.innerHTML = S.runs.map((r) => `<option value="${r.id}">${escapeHtml(r.title || r.id)}</option>`).join("") + (S.mode === "live" ? '<option value="__live">live feed</option>' : "");
  if (S.run) sel.value = S.run.id;
}

// ------------------------------------------------------------------------------------------------ live mode
function setConn(state, txt) {
  const dot = $("conn").querySelector(".dot"); dot.className = "dot " + (state === "on" ? "on" : state === "wait" ? "wait" : "");
  $("connTxt").textContent = txt;
}
function startLive() {
  stopLive(); S.mode = "live"; $("modeLive").setAttribute("aria-pressed", "true"); $("modeReplay").setAttribute("aria-pressed", "false");
  const url = Q.get("ws") || "ws://127.0.0.1:8765";
  setConn("wait", `connecting ${url.replace(/^ws:\/\//, "")}`);
  let ws;
  try { ws = new WebSocket(url); } catch (e) { setConn("off", "websocket blocked"); return; }
  S.ws = ws;
  ws.onopen = () => setConn("on", "LIVE");
  ws.onclose = () => { if (S.ws === ws) { setConn("wait", "reconnecting…"); S._retry = setTimeout(() => S.mode === "live" && startLive(), 2000); } };
  ws.onerror = () => {};
  ws.onmessage = async (msg) => {
    const m = JSON.parse(msg.data);
    if (m.type === "hello") {
      const run = { id: "__live", title: `live: ${m.index.run}`, _bundle: { index: m.index, snaps: {} } };
      if (m.last) { const f = m.index.snapshots.find((s) => s.t_sim === m.last.t_sim); if (f) run._bundle.snaps[f.file] = m.last; }
      S.liveRun = run; S.follow = true;
      // snapshots older than the hello are fetched from disk when the page is served next to out/stream
      run.path = `${Q.get("base") || "../out/stream"}/${m.index.run}`;
      S.run = run; S.index = m.index; S.snaps.clear(); S.decoded.clear(); S.curFile = null; pickAxis(); renderMetricCards();
      Object.entries(run._bundle.snaps).forEach(([k, v]) => S.snaps.set(k, v));
      S.run._bundle = null;                         // fall back to fetch for files not received live
      S.u = 1; await applyCursor(); renderMachines(); renderEvents();
      $("runNote").textContent = `${m.index.label} · LIVE run “${m.index.run}”`;
      const sel = $("runSel"); if (![...sel.options].some((o) => o.value === "__live")) sel.insertAdjacentHTML("beforeend", '<option value="__live">live feed</option>');
      sel.value = "__live";
    } else if (m.type === "snapshot" && S.index) {
      const s = m.snap, file = `snap_${String(Math.round(s.t_sim)).padStart(9, "0")}.json`;
      S.snaps.set(file, s);
      S.index.snapshots = S.index.snapshots.filter((q) => q.file !== file);
      S.index.snapshots.push({ file, t_sim: s.t_sim, t_obs: s.t_obs, prod_state: s.prod_state, metrics: s.metrics });
      S.index.snapshots.sort((a, b) => a.t_obs - b.t_obs || a.t_sim - b.t_sim);
      if (S.index.snapshots.length === 2) pickAxis();
      renderMetricCards();
      if (S.follow) { S.u = 1; S.curFile = null; } await applyCursor();
    } else if (m.type === "event" && S.index) {
      S.index.events.push(m.event); renderEvents();
    } else if (m.type === "close") { setConn("off", "run complete"); }
  };
}
function stopLive() {
  if (S._retry) clearTimeout(S._retry);
  if (S.ws) { const w = S.ws; S.ws = null; try { w.close(); } catch (e) {} }
}
function startReplay() {
  stopLive(); S.mode = "replay"; $("modeLive").setAttribute("aria-pressed", "false"); $("modeReplay").setAttribute("aria-pressed", "true");
  setConn("off", BUNDLE ? "bundled replay" : "replay");
  const id = $("runSel").value; const run = S.runs.find((r) => r.id === id) || S.runs[0];
  if (run) openRun(run).catch(showErr);
}
function showErr(e) { console.error(e); $("runNote").textContent = `Could not load the run: ${e.message}. Serve the repo root (python -m http.server) and open /dashboard/.`; }

// ------------------------------------------------------------------------------------------------ main loop + controls
let lastT = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - lastT) / 1000); lastT = now;
  if (S.playing && S.mode === "replay" && snapsList().length > 1) {
    S.u += (dt * S.speed) / SWEEP_S;
    if (S.u >= 1) { S.u = 1; S.playing = false; $("play").textContent = "▶"; }
    applyCursor();
  }
  if (S.layout) {
    drawPlant(); drawTrack(); drawMetrics(); drawStrip();
    if (now - (S._panelT || 0) > 120) { S._panelT = now; renderEvents(); renderMachines(); }
  }
  requestAnimationFrame(frame);
}
function bindControls() {
  $("play").addEventListener("click", () => {
    if (S.u >= 1) S.u = 0;
    S.playing = !S.playing; $("play").textContent = S.playing ? "❚❚" : "▶"; $("play").setAttribute("aria-label", S.playing ? "Pause" : "Play");
  });
  $("speed").addEventListener("change", (e) => { S.speed = +e.target.value; });
  const tr = $("track");
  const uAt = (e) => { const r = tr.getBoundingClientRect(), pad = 8; return (e.clientX - r.left - pad) / (r.width - 2 * pad); };
  tr.addEventListener("pointerdown", (e) => { tr.setPointerCapture(e.pointerId); tr._drag = true; seekU(uAt(e)); });
  tr.addEventListener("pointermove", (e) => { if (tr._drag) seekU(uAt(e)); });
  tr.addEventListener("pointerup", () => { tr._drag = false; });
  tr.addEventListener("keydown", (e) => {
    const n = Math.max(1, snapsList().length - 1);
    if (e.key === "ArrowRight") { seekU(S.u + 1 / n); e.preventDefault(); }
    if (e.key === "ArrowLeft") { seekU(S.u - 1 / n); e.preventDefault(); }
    if (e.key === " ") { $("play").click(); e.preventDefault(); }
  });
  $("toggles").querySelectorAll(".tog").forEach((b) => b.addEventListener("click", () => {
    const k = b.dataset.layer; S.layers[k] = !S.layers[k]; b.setAttribute("aria-pressed", String(S.layers[k]));
    if (k === "cycle") S.anchor.wall = nowS();
  }));
  $("modeReplay").addEventListener("click", startReplay);
  $("modeLive").addEventListener("click", startLive);
  $("runSel").addEventListener("change", () => { if ($("runSel").value !== "__live") startReplay(); });
  $("stage").addEventListener("click", (e) => {
    if (!S.view) return;
    const r = cv.getBoundingClientRect(), dpr = cv.width / r.width, px = (e.clientX - r.left) * dpr, py = (e.clientY - r.top) * dpr;
    const x = (px - S.view.ox) / S.view.sc, z = (S.view.oy - py) / S.view.sc;
    const m = S.layout.machines.find((q) => x >= q.rect[0] && x <= q.rect[2] && z >= q.rect[1] && z <= q.rect[3]);
    if (m) selectMachine(m.id);
  });
}

async function init() {
  try {
    S.layout = BUNDLE ? BUNDLE.layout : await getJSON("layout.json");
    S.phases = BUNDLE ? BUNDLE.phases : await getJSON("phases.json").catch(() => null);
  } catch (e) { showErr(e); return; }
  if (Q.has("layers")) { const on = new Set(Q.get("layers").split(",")); for (const k of Object.keys(S.layers)) S.layers[k] = on.has(k); }
  document.querySelectorAll("#toggles .tog").forEach((b) => b.setAttribute("aria-pressed", String(S.layers[b.dataset.layer])));
  if (Q.has("speed")) { S.speed = +Q.get("speed"); $("speed").value = Q.get("speed"); }
  bindControls();
  S.runs = await loadRuns(); fillRunSelect();
  if (Q.get("run")) $("runSel").value = Q.get("run");
  else if (Q.get("runpath")) $("runSel").value = S.runs[0].id;
  S.machine = Q.get("machine") || (S.layout.machines.find((x) => x.id === "S30") || S.layout.machines[0]).id;
  if (Q.get("mode") === "live") startLive();
  else if (S.runs.length) { startReplay(); }
  else { $("runNote").textContent = "No recorded runs found (dashboard/runs.json). Switch to Live or run dashboard/export_static.py."; setConn("off", "no runs"); }
  if (Q.get("autoplay") === "1") setTimeout(() => { S.playing = true; $("play").textContent = "❚❚"; }, 600);
  requestAnimationFrame(frame);
}
init();
window.__IC_STATE__ = S;   // for headless recording / debugging
window.__IC_SEEK__ = (u) => { S.playing = false; return seekU(u); };
})();
