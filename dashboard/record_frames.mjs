// Headless capture of the dashboard (screenshot or a deterministic frame sequence for the video). Node >= 22, Chrome.
//   node dashboard/record_frames.mjs --url "file:///C:/AMD/intercycle/out/stream/dashboard_bundle.html?run=build_map_72h" \
//        --shot out/stream/shot.png [--u 0.4] [--wait 2500]
//   node dashboard/record_frames.mjs --url ... --frames out/stream/frames --seconds 30 --fps 15 [--u0 0 --u1 1]
// Frames step the replay cursor (window.__IC_STATE__.u) from u0 to u1, so the result does not depend on capture speed;
// the cycle preview runs on a pinned clock (k / fps), so it is smooth at any capture speed. Assemble with ffmpeg:  ffmpeg -framerate 15 -i frames/f_%04d.png -pix_fmt yuv420p map.mp4
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";

const A = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => (a.startsWith("--") ? acc.concat([[a.slice(2), arr[i + 1] && !arr[i + 1].startsWith("--") ? arr[i + 1] : "1"]]) : acc), []));
const CHROME = A.chrome || "C:/Program Files/Google/Chrome/Application/chrome.exe";
const W = +(A.width || 1920), H = +(A.height || 1080), PORT = +(A.port || 9333);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const prof = mkdtempSync(join(tmpdir(), "ic-dash-"));
const chrome = spawn(CHROME, ["--headless=new", "--hide-scrollbars", "--mute-audio", `--remote-debugging-port=${PORT}`, `--user-data-dir=${prof}`,
  `--window-size=${W},${H}`, "--allow-file-access-from-files", "about:blank"], { stdio: "ignore" });
let ws, id = 0;
const pending = new Map();
function send(method, params = {}) {
  return new Promise((res, rej) => { const i = ++id; pending.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params })); });
}
async function main() {
  let target;
  for (let k = 0; k < 50 && !target; k++) {
    try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find((t) => t.type === "page"); } catch { await sleep(200); }
  }
  if (!target) throw new Error("chrome did not start");
  ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { const p = pending.get(d.id); pending.delete(d.id); d.error ? p.rej(new Error(d.error.message)) : p.res(d.result); } };
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: A.url });
  await sleep(+(A.wait || 2500));
  const ev = (expr) => send("Runtime.evaluate", { expression: expr, awaitPromise: true });
  const shot = async (file) => {
    const r = await send("Page.captureScreenshot", { format: "png" });
    mkdirSync(dirname(file), { recursive: true }); writeFileSync(file, Buffer.from(r.data, "base64"));
  };
  if (A.shot) {
    if (A.u) { await ev(`window.__IC_SEEK__(${+A.u})`); await sleep(800); }
    await shot(A.shot); console.log(`wrote ${A.shot}`);
  }
  if (A.frames) {
    const fps = +(A.fps || 15), n = Math.round(+(A.seconds || 30) * fps), u0 = +(A.u0 || 0), u1 = +(A.u1 || 1);
    mkdirSync(A.frames, { recursive: true });
    await ev("window.__IC_STATE__.playing=false");
    const t0 = Date.now();
    for (let k = 0; k < n; k++) {
      const u = u0 + (u1 - u0) * (k / Math.max(1, n - 1));
      await ev(`window.__IC_CLOCK__ = ${(k / fps).toFixed(4)}; window.__IC_SEEK__(${u})`);   // pinned clock: smooth cycle preview                    // fetch/decode the snapshot under the cursor
      await ev(`new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(()=>r(1))))`);   // let it paint
      await shot(join(A.frames, `f_${String(k).padStart(4, "0")}.png`));
      if (k % 50 === 0) console.log(`frame ${k}/${n} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
    }
    console.log(`wrote ${n} frames to ${A.frames} at ${fps} fps`);
  }
}
main().catch((e) => { console.error(e); process.exitCode = 1; }).finally(() => { try { ws && ws.close(); } catch {} chrome.kill(); });
