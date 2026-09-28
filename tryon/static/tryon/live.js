// Live lip try-on: MediaPipe FaceLandmarker (VIDEO mode) + LipFX finishes, all in the browser.
import { FaceLandmarker, FilesetResolver } from "./vendor/tasks-vision/vision_bundle.mjs";

const cfg = document.getElementById("app").dataset;
const $ = id => document.getElementById(id);
const ALPHA = 0.5;                                   // same rgba alpha as the photo mode
const OUTER = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146];
const INNER = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95];

const video = $("video"), view = $("view"), vctx = view.getContext("2d");
const roiC = document.createElement("canvas"), roiX = roiC.getContext("2d", { willReadFrequently: true });
const mskC = document.createElement("canvas"), mskX = mskC.getContext("2d", { willReadFrequently: true });
const noise = LipFX.makeNoise(42), fxState = {};
let landmarker = null, stream = null, running = false, lastT = -1, smooth = null, lastTs = 0;
let fps = 0, tick0 = performance.now(), frames = 0, missing = 0;

const status = t => { $("status").textContent = t; };
const hexToRgb = h => { h = h.replace("#", ""); if (h.length === 3) h = [...h].map(c => c + c).join(""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); };
const validHex = v => /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.test(v);
const finish = () => document.querySelector("input[name=finish]:checked").value;
let color = hexToRgb($("hex").value);

async function loadModel() {
  if (landmarker) return;
  status("Loading face model…");
  const files = await FilesetResolver.forVisionTasks(cfg.wasm);
  const make = delegate => FaceLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetPath: cfg.model, delegate },
    runningMode: "VIDEO", numFaces: 1,
  });
  try { landmarker = await make("GPU"); } catch (e) { console.warn("GPU delegate failed, using CPU", e); landmarker = await make("CPU"); }
}

async function start() {
  $("start").disabled = true; $("err").textContent = "";
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera needs HTTPS (or localhost).");
    await loadModel();
    status("Starting camera…");
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    video.srcObject = stream; await video.play();
    view.width = video.videoWidth; view.height = video.videoHeight;
    running = true; smooth = null; lastTs = 0; $("stage").hidden = false; $("empty").hidden = true;
    $("start").hidden = true; $("stop").hidden = false; $("snap").hidden = false;
    requestAnimationFrame(loop);
  } catch (e) {
    $("err").textContent = e.name === "NotAllowedError" ? "Camera permission was denied." : e.message;
    $("start").disabled = false; status("");
  }
}
function stop() {
  running = false; stream?.getTracks().forEach(t => t.stop()); stream = null;
  $("stage").hidden = true; $("empty").hidden = false;
  $("start").hidden = false; $("start").disabled = false; $("stop").hidden = true; $("snap").hidden = true; status("");
}

function loop() {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastT) {
    lastT = video.currentTime;
    const ts = Math.max(performance.now(), lastTs + 1); lastTs = ts;         // timestamps must increase
    const res = landmarker.detectForVideo(video, ts);
    vctx.drawImage(video, 0, 0);
    if (res.faceLandmarks.length) { missing = 0; processLips(res.faceLandmarks[0]); status(`${fps.toFixed(0)} fps`); }
    else if (++missing > 5) { smooth = null; status("No face detected"); }
    frames++; const now = performance.now();
    if (now - tick0 > 500) { fps = frames * 1000 / (now - tick0); frames = 0; tick0 = now; }
  }
  requestAnimationFrame(loop);
}

function processLips(lm) {
  const W = view.width, H = view.height, idx = OUTER.concat(INNER);
  // light temporal smoothing of the landmarks -> less jitter (faster when the lips really move)
  if (!smooth) smooth = idx.map(i => [lm[i].x * W, lm[i].y * H]);
  else idx.forEach((i, k) => {
    const nx = lm[i].x * W, ny = lm[i].y * H, d = Math.hypot(nx - smooth[k][0], ny - smooth[k][1]);
    const a = Math.min(1, 0.45 + d / (0.02 * W));
    smooth[k][0] += a * (nx - smooth[k][0]); smooth[k][1] += a * (ny - smooth[k][1]);
  });
  const outer = smooth.slice(0, OUTER.length), inner = smooth.slice(OUTER.length);
  const xs = outer.map(p => p[0]), ys = outer.map(p => p[1]);
  const lipW = Math.max(...xs) - Math.min(...xs), mg = Math.max(8, Math.round(lipW * 0.14));
  const x0 = Math.max(0, Math.floor(Math.min(...xs)) - mg), y0 = Math.max(0, Math.floor(Math.min(...ys)) - mg);
  const x1 = Math.min(W, Math.ceil(Math.max(...xs)) + mg), y1 = Math.min(H, Math.ceil(Math.max(...ys)) + mg);
  const w = x1 - x0, h = y1 - y0; if (w < 8 || h < 8) return;

  // mask: outer lip polygon minus mouth opening, anti-aliased by the canvas, then feathered
  if (mskC.width !== w || mskC.height !== h) { mskC.width = w; mskC.height = h; roiC.width = w; roiC.height = h; }
  mskX.clearRect(0, 0, w, h); mskX.globalCompositeOperation = "source-over"; mskX.fillStyle = "#fff";
  const path = pts => { mskX.beginPath(); pts.forEach(([x, y], i) => i ? mskX.lineTo(x - x0, y - y0) : mskX.moveTo(x - x0, y - y0)); mskX.closePath(); mskX.fill(); };
  path(outer); mskX.globalCompositeOperation = "destination-out"; path(inner);
  const md = mskX.getImageData(0, 0, w, h).data, m0 = new Float32Array(w * h);
  for (let i = 0; i < m0.length; i++) m0[i] = md[4 * i + 3] / 255;
  const mask = LipFX.gauss(m0, w, h, Math.max(1, lipW * 0.02));

  roiX.drawImage(video, x0, y0, w, h, 0, 0, w, h);                        // same frame as the full view
  const img = roiX.getImageData(0, 0, w, h);
  LipFX.render(img.data, w, h, mask, { color, alpha: ALPHA, finish: finish(), lipW, noise, ox: x0, oy: y0, state: fxState });
  roiX.putImageData(img, 0, 0);
  vctx.drawImage(roiC, x0, y0);
}

function snapshot() {                                                     // saves what you see (mirrored)
  const c = document.createElement("canvas"); c.width = view.width; c.height = view.height;
  const x = c.getContext("2d"); x.translate(c.width, 0); x.scale(-1, 1); x.drawImage(view, 0, 0);
  const a = document.createElement("a"); a.download = "lip-live.png"; a.href = c.toDataURL("image/png"); a.click();
}

$("start").onclick = start; $("stop").onclick = stop; $("snap").onclick = snapshot;
$("pick").addEventListener("input", () => { $("hex").value = $("pick").value.toUpperCase(); color = hexToRgb($("hex").value); });
$("hex").addEventListener("input", () => {
  const v = $("hex").value; if (!validHex(v)) return;
  const h = v.startsWith("#") ? v : "#" + v; color = hexToRgb(h); if (/^#[0-9a-f]{6}$/i.test(h)) $("pick").value = h;
});
document.querySelectorAll("input[name=finish]").forEach(r => r.addEventListener("change", () => { delete fxState.lo; delete fxState.hi; }));
window.addEventListener("pagehide", stop);
