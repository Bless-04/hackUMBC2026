/* GuideSense: local UI, no external scripts or network assets. */
"use strict";
const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="session-token"]').content;
let settings = { mode: "demo", camera: 0, voice: false, gemini: false, backboard: false, logging: false };
let snapshot = null;
let connected = false;
let pending = false;
let imageUrl = null;
let latestEventId = -1;
let toastTimer;
let view = "overview";
const emptyObjects = $("objects-body").innerHTML;
try {
  const saved = JSON.parse(localStorage.getItem("guidesense-settings"));
  if (saved && ["demo", "camera"].includes(saved.mode)) settings = { ...settings, ...saved };
} catch { /* Local storage is optional, including in private browsing. */ }

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}
function setHtml(element, content) { if (element.innerHTML !== content) element.innerHTML = content; }
function active() { return snapshot && ["starting", "running", "stopping"].includes(snapshot.status); }
function elapsed(seconds) {
  const value = Math.max(0, Math.floor(seconds || 0));
  return `${String(Math.floor(value / 60)).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`;
}
function toast(text) {
  $("toast").textContent = text;
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500);
}
async function api(path, body) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json", "X-GuideSense-Token": token }, body: JSON.stringify(body || {}), signal: AbortSignal.timeout(6000) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Could not update the session.");
  return data;
}
async function toggleSession() {
  if (pending || !connected) return;
  pending = true;
  updateButtons();
  try {
    const stopping = active();
    snapshot = await api(stopping ? "/api/stop" : "/api/start", settings);
    latestEventId = -1;
    render(snapshot);
  } catch (error) {
    toast(error.message || "The dashboard could not reach the server.");
  } finally { pending = false; updateButtons(); }
}
function updateButtons() {
  const status = snapshot?.status;
  const running = active();
  $("session-button").disabled = !connected || pending || status === "stopping";
  $("session-button").classList.toggle("running", !!running);
  $("session-button").querySelector("use").setAttribute("href", running ? "#i-stop" : "#i-play");
  $("session-button").querySelector("span").textContent = status === "stopping" ? "Stopping…" : running ? "End session" : "Start session";
  $("stage-start").disabled = !connected || pending;
}
function changeView(next) {
  view = next;
  $("overview-view").hidden = next !== "overview";
  $("activity-view").hidden = next !== "activity";
  document.querySelectorAll(".nav-item[data-view]").forEach((button) => {
    const selected = button.dataset.view === next;
    button.classList.toggle("active", selected);
    if (selected) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
  });
  $("breadcrumb-page").textContent = next === "overview" ? "Overview" : "Session activity";
  $("page-title").innerHTML = next === "overview" ? "Your world, in focus<span>.</span>" : "Every moment, understood<span>.</span>";
  $("page-description").textContent = next === "overview" ? "A real-time view of what’s ahead. A little confidence for every step." : "Follow your session through detections, guidance, and changes in awareness.";
}
function openSettings() {
  const form = $("settings-form");
  form.elements.mode.value = settings.mode;
  form.elements.camera.value = settings.camera;
  ["voice", "gemini", "backboard", "logging"].forEach((key) => { form.elements[key].checked = !!settings[key]; });
  $("settings-fields").disabled = !!active();
  form.querySelector('[type="submit"]').disabled = !!active();
  updateModeOptions();
  $("settings-dialog").showModal();
}
function updateModeOptions() {
  const form = $("settings-form");
  const demo = form.elements.mode.value === "demo";
  $("camera-settings").hidden = demo;
  ["voice", "gemini", "backboard"].forEach((key) => { form.elements[key].disabled = demo; });
  $("settings-note").textContent = active() ? "End the current session to change its configuration." : demo ? "Demo runs locally with simulated data. Voice and cloud services stay off." : "Cloud features use your configured API keys. “Configured” means credentials are present; it does not verify a cloud connection.";
}
$("settings-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (active()) return;
  const form = event.currentTarget;
  settings = { mode: form.elements.mode.value, camera: Number(form.elements.camera.value) };
  ["voice", "gemini", "backboard", "logging"].forEach((key) => { settings[key] = form.elements[key].checked && (settings.mode === "camera" || key === "logging"); });
  try { localStorage.setItem("guidesense-settings", JSON.stringify(settings)); } catch { /* Optional. */ }
  $("settings-dialog").close();
  toast("Configuration saved. Ready for your next session.");
  if (snapshot) render(snapshot);
});
document.querySelectorAll('[name="mode"]').forEach((radio) => radio.addEventListener("change", updateModeOptions));
["configure-button", "sidebar-settings", "manage-button"].forEach((id) => $(id).addEventListener("click", openSettings));
$("help-button").addEventListener("click", () => $("help-dialog").showModal());
document.querySelectorAll(".close-dialog").forEach((button) => button.addEventListener("click", () => button.closest("dialog").close()));
document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => changeView(button.dataset.view)));
$("session-button").addEventListener("click", toggleSession);
$("stage-start").addEventListener("click", toggleSession);

async function fullscreen() {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await $("camera-stage").requestFullscreen();
  } catch { toast("Full screen is unavailable in this browser."); }
}
$("fullscreen-button").addEventListener("click", fullscreen);
document.addEventListener("keydown", (event) => {
  if (event.key.toLowerCase() === "f" && view === "overview" && !event.ctrlKey && !event.metaKey && !event.altKey && !document.querySelector("dialog[open]") && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) {
    event.preventDefault(); fullscreen();
  }
});

function render(data) {
  const running = data.status === "running";
  const mode = active() ? data.mode : settings.mode;
  const demo = mode === "demo";
  const valid = running && data.distance !== null;
  const distance = data.distance;
  const detections = running ? data.detections : [];
  $("mode-label").textContent = demo ? "Demo environment" : `Live camera ${settings.camera}`;
  $("processing-label").textContent = demo ? "Simulated demo · no model inference" : "Object detection runs locally · MobileNet-SSD";
  $("summary-status").textContent = ({ idle: "Ready to explore", starting: "Preparing session", running: demo ? "Demo in progress" : "Session in progress", stopping: "Ending session", stopped: "Session complete", error: "Needs attention" })[data.status] || "Standby";
  $("metric-fps").textContent = running ? data.fps.toFixed(1) : "—";
  $("metric-objects").textContent = running ? detections.length : "—";
  $("metric-time").textContent = elapsed(data.elapsed);
  $("notice").hidden = !data.error;
  $("notice").textContent = data.error || "";
  const stage = $("camera-stage");
  stage.classList.toggle("idle", !running);
  stage.classList.toggle("is-live", !demo || (running && data.has_frame));
  stage.style.aspectRatio = running ? `${data.frame_width} / ${data.frame_height}` : "";
  $("demo-scene").hidden = running && data.has_frame ? true : !demo;
  $("camera-image").hidden = !running || !data.has_frame;
  $("feed-source").textContent = demo ? "DEMO HUD STREAM" : `LIVE CAMERA · ${settings.camera}`;
  $("feed-badge").className = `pill ${running ? demo ? "demo" : "live" : "neutral"}`;
  $("feed-badge").innerHTML = `<i></i>${running ? demo ? "HUD DEMO" : data.has_frame ? "LIVE" : "NO FRAME" : "STANDBY"}`;
  $("frame-info").textContent = running ? data.has_frame ? `${data.frame_width} × ${data.frame_height} · GuideSense OpenCV HUD` : "Waiting for a fresh camera frame" : demo ? "Illustrated preview · no camera access" : "Camera is inactive";
  $("stage-empty").hidden = running && (demo || data.has_frame);
  $("focus-corners").hidden = running;
  $("stage-empty").querySelector("h3").textContent = data.status === "starting" ? "Bringing the world into view…" : running ? "Waiting for your camera." : data.status === "stopping" ? "See you in a moment." : "Ready when you are.";
  $("stage-empty").querySelector("p").textContent = active() ? "Your session status will update here." : demo ? "Start a session to bring your surroundings into focus." : "Start a session to connect your camera.";
  $("stage-start").hidden = !!active();
  $("stage-start").innerHTML = `${demo ? "Explore demo" : "Connect camera"} <svg><use href="#i-arrow"/></svg>`;
  $("awareness-card").dataset.state = valid ? data.state : "SILENT";
  $("state-tag").textContent = !valid ? running ? "AWAITING CAMERA DATA" : "AWAITING SESSION" : data.state === "URGENT" ? "NEAR OBJECT · URGENT" : data.state === "INFORMATIVE" ? "OBJECT IN RANGE" : "MONITORING SURROUNDINGS";
  $("distance-value").textContent = valid ? distance.toFixed(2) : "—";
  $("distance-caption").textContent = !valid ? "Waiting for distance data" : demo ? "Simulated distance · demonstration" : detections.length ? "Nearest object · camera estimate" : "No detection · fallback estimate";
  $("zone-marker").hidden = !valid;
  $("zone-marker").style.left = `${Math.min(97, Math.max(1, distance < .6 ? distance / .6 * 33 : distance <= 2 ? 33 + (distance - .6) / 1.4 * 33 : 66 + (distance - 2) / 2 * 33))}%`;
  $("radar-target").hidden = !detections.length;
  if (detections.length) {
    $("radar-target").style.left = `${({ LEFT: 25, CENTER: 50, RIGHT: 75 })[detections[0].direction] || 50}%`;
    $("radar-target").style.top = `${Math.min(76, Math.max(8, 84 - distance * 22))}px`;
  }
  $("object-count").textContent = detections.length;
  setHtml($("objects-body"), detections.length ? detections.map((d) => `<tr><td>${escapeHtml(d.label.charAt(0).toUpperCase() + d.label.slice(1))}</td><td><span class="direction-pill">${({ LEFT: "↖ Left", CENTER: "↑ Ahead", RIGHT: "↗ Right" })[d.direction] || "Ahead"}</span></td><td><div class="confidence-cell"><span class="confidence-meter"><i style="width:${Math.max(0, Math.min(100, d.confidence * 100))}%"></i></span>${Math.round(d.confidence * 100)}%</div></td><td><span class="signal-bars" aria-label="Object detected"><i></i><i></i><i></i><i></i></span></td></tr>`).join("") : emptyObjects);
  ["gemini", "voice", "backboard"].forEach((name) => {
    const text = active() ? data.services[name] : settings[name] && !demo ? "Next session" : "Off";
    $(name + "-status").textContent = text;
    $(name + "-status").classList.toggle("enabled", !["Off", "Key missing"].includes(text));
  });
  if (data.announcement) {
    $("guidance-text").textContent = `“${data.announcement.text}”`;
    $("guidance-source").textContent = `${data.announcement.source.toUpperCase()}${demo ? " · DEMONSTRATION" : " · LATEST ANNOUNCEMENT"}`;
  } else {
    $("guidance-text").textContent = "“A little awareness makes a world of difference.”";
    $("guidance-source").textContent = running ? "LISTENING FOR A CONFIRMED DETECTION" : "GUIDANCE WILL APPEAR HERE";
  }
  drawDetections(detections, data, demo);
  $("activity-count").textContent = data.events.length;
  if ((data.events[0]?.id || 0) !== latestEventId) {
    latestEventId = data.events[0]?.id || 0;
    $("latest-event").textContent = data.events[0]?.message || "Your next session starts a new story.";
    $("activity-list").innerHTML = data.events.length ? data.events.map((event) => `<div class="activity-row"><time>${escapeHtml(event.time)}</time><span>${event.category === "State" ? "↗" : "·"}</span><div><strong>${escapeHtml(event.category)}</strong><p>${escapeHtml(event.message)}</p></div></div>`).join("") : '<p class="activity-empty">Start a session to see its story unfold.</p>';
  }
  updateButtons();
}
function drawDetections(detections, data, demo) {
  const overlay = $("detection-overlay");
  if (data.has_frame) {
    setHtml(overlay, "");
    return;
  }
  overlay.setAttribute("viewBox", `0 0 ${data.frame_width} ${data.frame_height}`);
  const parts = detections.map((d) => {
    const [x1, y1, x2, y2] = d.bbox;
    const width = x2 - x1, height = y2 - y1;
    const cx = (x1 + x2) / 2;
    const person = demo ? `<g class="demo-person"><circle cx="${cx}" cy="${y1 + height * .1}" r="${height * .08}"/><path d="M${cx - width * .19},${y1 + height * .18} Q${cx},${y1 + height * .18} ${cx + width * .19},${y1 + height * .22} L${cx + width * .33},${y1 + height * .62} L${cx + width * .18},${y1 + height * .66} L${cx + width * .13},${y2} L${cx},${y2} L${cx - width * .04},${y1 + height * .7} L${cx - width * .13},${y2} L${cx - width * .27},${y2} L${cx - width * .18},${y1 + height * .6} L${cx - width * .33},${y1 + height * .6}Z"/></g>` : "";
    return `${person}<rect class="detection-box" x="${x1}" y="${y1}" width="${width}" height="${height}"/><rect class="detection-label-bg" x="${x1}" y="${Math.max(0, y1 - 20)}" width="${Math.max(125, d.label.length * 7 + 65)}" height="20" rx="3"/><text class="detection-label" x="${x1 + 7}" y="${Math.max(14, y1 - 6)}">${escapeHtml(d.label)} · ${Math.round(d.confidence * 100)}% · ${escapeHtml(d.direction.toLowerCase())}</text>`;
  });
  setHtml(overlay, parts.join(""));
}

async function poll() {
  try {
    const response = await fetch("/api/state", { signal: AbortSignal.timeout(3000) });
    if (!response.ok) throw new Error("Disconnected");
    snapshot = await response.json();
    connected = true;
    $("connection-label").textContent = "Local connection";
    $("connection-dot").classList.remove("offline");
    render(snapshot);
  } catch {
    connected = false;
    $("connection-label").textContent = "Disconnected";
    $("connection-dot").classList.add("offline");
    $("notice").hidden = false;
    $("notice").textContent = "Dashboard disconnected. The session may still be running on the server. Reconnecting…";
    $("feed-badge").innerHTML = "<i></i>OFFLINE";
    $("camera-image").hidden = true;
    $("detection-overlay").innerHTML = "";
    updateButtons();
  } finally { setTimeout(poll, 350); }
}
async function pollFrame() {
  try {
    if (connected && snapshot?.status === "running" && snapshot.has_frame) {
      const response = await fetch("/api/frame", { signal: AbortSignal.timeout(3000) });
      if (response.ok && response.status !== 204) {
        const url = URL.createObjectURL(await response.blob());
        $("camera-image").src = url;
        if (imageUrl) URL.revokeObjectURL(imageUrl);
        imageUrl = url;
      }
    }
  } catch { /* State polling communicates connection problems. */ }
  finally { setTimeout(pollFrame, 150); }
}
setInterval(() => { $("feed-clock").textContent = new Date().toLocaleTimeString("en-GB"); }, 1000);
poll();
pollFrame();
