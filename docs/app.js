// Show random image/mask pairs from the corpus, one at a time; send each vote to the sheet.
// Tap the image to hide the outlines; swipe or use the arrow keys to vote.
import { CORPUS_URL, VOTE_URL } from "./config.js";

const $ = (s) => document.querySelector(s);
const store = {
  get: (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set: (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};
let user = store.get("mv-user", null);       // {name, pw}
// randomUUID only exists on HTTPS or localhost pages; plain-HTTP LAN tests need the fallback
const newId = () => crypto.randomUUID?.() ?? [...crypto.getRandomValues(new Uint8Array(16))].map((b) => b.toString(16).padStart(2, "0")).join("");
const device = store.get("mv-device", null) || newId();
store.set("mv-device", device);
const seen = new Set(store.get("mv-seen", []));  // item ids rated or skipped on this device
let shards = [], total = 0, current = null, next = null, outlines = true;

const PHI = 0.6180339887498949;
function hue(l) {  // a distinct saturated colour per label, as in the training-data explorer
  const h = (l * PHI) % 1, i = Math.floor(h * 6), f = h * 6 - i, q = 1 - f;
  return [[1, f, 0], [q, 1, 0], [0, 1, f], [0, q, 1], [f, 0, 1], [1, 0, q]][i].map((c) => Math.round(c * 255));
}

// batches.json lists item lists ("shards"); each entry is a path, or {path, n} with its
// item count, so a shard's list is only fetched once a draw lands on it.
async function loadCorpus() {
  const { batches } = await (await fetch(CORPUS_URL + "batches.json", { cache: "no-cache" })).json();
  shards = batches.map((b) => (typeof b === "string" ? { path: b, n: null } : { ...b }));
  await Promise.all(shards.filter((s) => s.n == null).map(loadShard));  // old entries carry no count
  total = shards.reduce((t, s) => t + s.n, 0);
}

async function loadShard(s) {
  if (!s.items) {
    const l = await (await fetch(CORPUS_URL + s.path)).json();
    s.items = l.items.map((it) => ({ ...it, batch: l.batch }));
    s.n = s.items.length;
  }
  return s.items;
}

const taken = (it) => seen.has(it.id) || it.id === current?.item.id || it.id === next?.item.id;

async function pick() {  // every unseen item equally likely: draw a shard by its unseen count
  for (let tries = 0; tries < 20; tries++) {
    const w = shards.map((s) => (s.items ? s.items.filter((it) => !taken(it)).length : s.n));
    let r = Math.random() * w.reduce((a, b) => a + b, 0);
    if (!(r > 0)) return null;
    let i = 0;
    while (r >= w[i]) r -= w[i++];
    const left = (await loadShard(shards[i])).filter((it) => !taken(it));
    if (left.length) return left[Math.floor(Math.random() * left.length)];
  }
  return null;
}

async function bitmap(url) {  // exact pixel values: masks encode labels in their colour
  const blob = await (await fetch(url)).blob();
  return createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
}

async function prepare(item) {
  if (!item) return null;
  const [img, mask] = await Promise.all([bitmap(CORPUS_URL + item.image), bitmap(CORPUS_URL + item.mask)]);
  const c = new OffscreenCanvas(mask.width, mask.height), x = c.getContext("2d", { willReadFrequently: true });
  x.drawImage(mask, 0, 0);
  const px = x.getImageData(0, 0, mask.width, mask.height).data, labels = new Uint32Array(mask.width * mask.height);
  for (let i = 0; i < labels.length; i++) labels[i] = (px[4 * i] << 16) | (px[4 * i + 1] << 8) | px[4 * i + 2];
  return { item, img, labels, w: mask.width, h: mask.height };
}

function draw() {
  const v = $("#view"), s = $("#stage"), d = current;
  if (!d) return;
  const dpr = window.devicePixelRatio || 1;
  const scale = Math.min((s.clientWidth - 32) / d.w, s.clientHeight / d.h);  // CSS px per image px
  const W = Math.max(1, Math.round(d.w * scale * dpr)), H = Math.max(1, Math.round(d.h * scale * dpr));
  v.width = W; v.height = H; v.style.width = W / dpr + "px"; v.style.height = H / dpr + "px";
  const x = v.getContext("2d");
  x.imageSmoothingEnabled = false;
  x.drawImage(d.img, 0, 0, W, H);
  if (!outlines) return;
  // outlines at screen resolution: a screen pixel is an edge where its label differs from a neighbour's
  const xs = new Uint32Array(W), ys = new Uint32Array(H);
  for (let i = 0; i < W; i++) xs[i] = Math.min(d.w - 1, Math.floor(i / W * d.w));
  for (let j = 0; j < H; j++) ys[j] = Math.min(d.h - 1, Math.floor(j / H * d.h));
  const lab = (i, j) => d.labels[ys[j] * d.w + xs[i]];
  const out = x.getImageData(0, 0, W, H), o = out.data, t = Math.max(1, Math.round(dpr));
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const l = lab(i, j);
    if (!l) continue;
    let edge = false;
    for (let k = 1; k <= t && !edge; k++)
      edge = i < k || j < k || i >= W - k || j >= H - k || lab(i - k, j) !== l || lab(i + k, j) !== l || lab(i, j - k) !== l || lab(i, j + k) !== l;
    if (edge) { const [r, g, b] = hue(l), p = 4 * (j * W + i); o[p] = r; o[p + 1] = g; o[p + 2] = b; }
  }
  x.putImageData(out, 0, 0);
}

async function show() {
  current = next ? await next.ready : await prepare(await pick());
  if (!current) { $("#info").textContent = "All rated — thanks!"; $("#view").width = 0; return; }
  $("#info").textContent = `${current.item.w}×${current.item.h} · ${current.item.cells} cells`;
  $("#count").textContent = `${seen.size} seen · ${Math.max(0, total - seen.size)} left`;
  draw();
  const n = await pick(); next = n ? { item: n, ready: prepare(n) } : null;  // fetch the next one now
}

function vote(v) {
  if (!current) return;
  const item = current.item;
  seen.add(item.id); store.set("mv-seen", [...seen]);
  if (v !== "skip") {
    fetch(VOTE_URL, {  // text/plain keeps it a simple request, which Apps Script accepts
      method: "POST", mode: "no-cors", headers: { "Content-Type": "text/plain" },
      body: JSON.stringify({ pw: user.pw, voter: user.name, device, item: item.id, batch: item.batch, vote: v }),
    }).catch(() => {});
    const f = $("#flash"); f.style.background = v === "up" ? "rgba(63,179,127,.25)" : "rgba(224,86,91,.25)";
    f.style.opacity = 1; setTimeout(() => (f.style.opacity = 0), 150);
  }
  show();
}

$("#good").onclick = () => vote("up");
$("#bad").onclick = () => vote("down");
$("#skip").onclick = () => vote("skip");
$("#view").onclick = () => { outlines = !outlines; draw(); };
addEventListener("keydown", (e) => {
  if ($("#login").classList.contains("on")) return;
  if (e.key === "ArrowRight") vote("up"); else if (e.key === "ArrowLeft") vote("down");
  else if (e.key === "ArrowDown" || e.key === " ") vote("skip"); else if (e.key === "o") $("#view").onclick();
});
let touch = null;
$("#stage").addEventListener("touchstart", (e) => (touch = e.touches[0]), { passive: true });
$("#stage").addEventListener("touchend", (e) => {
  const dx = e.changedTouches[0].clientX - touch?.clientX;
  if (Math.abs(dx) > 70) vote(dx > 0 ? "up" : "down");
});
addEventListener("resize", () => draw());

async function check(pw) {
  try { return (await (await fetch(`${VOTE_URL}?pw=${encodeURIComponent(pw)}`)).json()).ok; } catch { return false; }
}
$("#login").onsubmit = async (e) => {
  e.preventDefault();
  $("#msg").textContent = "";
  const u = { name: $("#name").value.trim(), pw: $("#pw").value };
  if (!(await check(u.pw))) { $("#msg").textContent = "Wrong password."; return; }
  user = u; store.set("mv-user", u); $("#login").classList.remove("on"); start();
};
async function start() {
  $("#info").textContent = "Loading…";
  await loadCorpus();
  show();
}
if (user) start(); else $("#login").classList.add("on");
