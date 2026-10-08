// Grid state: flat arrays indexed row-major as i * ny + j, matching the
// backend's SolveRequestDto.permittivity_re/im convention exactly.
let grid = {
  nx: 0,
  ny: 0,
  xMeters: 1,
  yMeters: 1,
  re: new Float64Array(0),
  im: new Float64Array(0),
};

// Result of the most recently completed solve, keyed by the same domain as
// `grid` at the time it was submitted.
let lastSolve = null;

const canvas = document.getElementById("grid-canvas");
const ctx = canvas.getContext("2d");

const el = (id) => document.getElementById(id);

function backgroundValues() {
  return { re: parseFloat(el("bg-re").value), im: parseFloat(el("bg-im").value) };
}

function resetGrid() {
  const nx = parseInt(el("nx").value, 10);
  const ny = parseInt(el("ny").value, 10);
  const xMeters = parseFloat(el("x-meters").value);
  const yMeters = parseFloat(el("y-meters").value);
  const bg = backgroundValues();

  grid.nx = nx;
  grid.ny = ny;
  grid.xMeters = xMeters;
  grid.yMeters = yMeters;
  grid.re = new Float64Array(nx * ny).fill(bg.re);
  grid.im = new Float64Array(nx * ny).fill(bg.im);
  lastSolve = null;

  syncAnimation();
  scheduleSolve(150);
}

// --- Color scales -----------------------------------------------------

function permittivityColor(value, bgValue) {
  const span = 8; // permittivity units mapped across the full color range
  const t = Math.max(0, Math.min(1, (value - bgValue) / span));
  const r = Math.round(70 + t * (210 - 70));
  const g = Math.round(120 + t * (60 - 120));
  const b = Math.round(220 + t * (60 - 220));
  return `rgb(${r},${g},${b})`;
}

function interpolateStops(t, stops) {
  t = Math.max(0, Math.min(1, t));
  for (let k = 0; k < stops.length - 1; k++) {
    const [t0, r0, g0, b0] = stops[k];
    const [t1, r1, g1, b1] = stops[k + 1];
    if (t >= t0 && t <= t1) {
      const f = (t - t0) / (t1 - t0);
      const r = Math.round(r0 + f * (r1 - r0));
      const g = Math.round(g0 + f * (g1 - g0));
      const b = Math.round(b0 + f * (b1 - b0));
      return `rgb(${r},${g},${b})`;
    }
  }
  const [, r, g, b] = stops[stops.length - 1];
  return `rgb(${r},${g},${b})`;
}

// A handful of viridis-ish control points, for magnitude (0..1) plots.
const VIRIDIS_STOPS = [
  [0.0, 68, 1, 84],
  [0.25, 59, 82, 139],
  [0.5, 33, 145, 140],
  [0.75, 94, 201, 98],
  [1.0, 253, 231, 37],
];

function fieldColor(t) {
  return interpolateStops(t, VIRIDIS_STOPS);
}

// A blue-white-red diverging scale for signed, time-harmonic snapshots.
const DIVERGING_STOPS = [
  [0.0, 33, 102, 172],
  [0.25, 103, 169, 207],
  [0.5, 247, 247, 247],
  [0.75, 239, 138, 98],
  [1.0, 178, 24, 43],
];

function divergingColor(t) {
  return interpolateStops(t, DIVERGING_STOPS);
}

// --- Rendering ----------------------------------------------------------
// The canvas is the single figure: it's painted directly by the user as the
// contrast profile, and once a solve completes the same cells are recolored
// by field magnitude, with an outline traced around the painted region so
// you can still see what you drew underneath the field.

function isTargetCell(i, j, bgRe, threshold) {
  return Math.abs(grid.re[i * grid.ny + j] - bgRe) > threshold;
}

function drawContrastOutline(cellW, cellH) {
  const { nx, ny } = grid;
  const bg = backgroundValues();
  const threshold = 0.05;
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.lineWidth = 1;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      if (!isTargetCell(i, j, bg.re, threshold)) continue;
      const neighbors = [
        [i - 1, j],
        [i + 1, j],
        [i, j - 1],
        [i, j + 1],
      ];
      const onBoundary = neighbors.some(
        ([ni, nj]) => ni < 0 || ni >= nx || nj < 0 || nj >= ny || !isTargetCell(ni, nj, bg.re, threshold)
      );
      if (onBoundary) {
        ctx.strokeRect(i * cellW, j * cellH, cellW, cellH);
      }
    }
  }
}

// "-anim" modes reinterpret the same complex field as a time-harmonic
// snapshot Re(E * e^{-i*omega*t}) = |E| cos(phase(E) - omega*t), animated by
// sweeping omega*t, instead of collapsing it to a static |E| magnitude.
function isAnimatedMode(mode) {
  return mode === "total-anim" || mode === "scattered-anim";
}

function fieldForMode(mode) {
  if (mode === "total" || mode === "total-anim") return "total_field";
  if (mode === "scattered" || mode === "scattered-anim") return "scattered_field";
  return null;
}

// The color scale for the time-harmonic view: normalizing by the true max
// magnitude lets a single hot cell (common right next to a painted
// high-contrast blob) wash out the color range everywhere else, so the rest
// of the canvas barely shifts and the animation looks static. Using a high
// percentile instead lets rare outliers saturate to solid red/blue rather
// than flattening everyone else's contrast. Computed once per solve, not
// per animation frame, since it doesn't depend on phase.
function animationScale(field) {
  const n = field.re.length;
  const mag = new Float64Array(n);
  for (let k = 0; k < n; k++) mag[k] = Math.hypot(field.re[k], field.im[k]);
  mag.sort();
  const scale = mag[Math.min(n - 1, Math.floor(0.98 * n))];
  return scale > 1e-12 ? scale : 1;
}

function redraw(animPhase = 0) {
  const { nx, ny } = grid;
  if (nx === 0 || ny === 0) return;
  const cellW = canvas.width / nx;
  const cellH = canvas.height / ny;
  const bg = backgroundValues();

  const mode = el("display-mode").value;
  const fieldKey = fieldForMode(mode);
  let fieldGrid = null;
  if (fieldKey && lastSolve && lastSolve.nx === nx && lastSolve.ny === ny) {
    fieldGrid = lastSolve.result[fieldKey];
  }

  // Only actually animate once a matching solve has landed; otherwise fall
  // through to the static/contrast rendering below instead of crashing on a
  // null fieldGrid (which would silently kill the requestAnimationFrame loop).
  const animated = isAnimatedMode(mode) && !!fieldGrid;

  if (animated) {
    const scale = mode === "total-anim" ? lastSolve.totalScale : lastSolve.scatteredScale;
    const cosPhase = Math.cos(animPhase);
    const sinPhase = Math.sin(animPhase);
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const idx = i * ny + j;
        const instantaneous = fieldGrid.re[idx] * cosPhase + fieldGrid.im[idx] * sinPhase;
        ctx.fillStyle = divergingColor(instantaneous / scale / 2 + 0.5);
        ctx.fillRect(i * cellW, j * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }
    drawContrastOutline(cellW, cellH);
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.font = "14px monospace";
    ctx.fillText(`ωt = ${Math.round((animPhase * 180) / Math.PI)}°`, 8, 18);
    return;
  }

  let mag = null;
  let fMin = 0;
  let fMax = 1;
  if (fieldGrid) {
    mag = new Float64Array(nx * ny);
    for (let k = 0; k < mag.length; k++) {
      mag[k] = Math.hypot(fieldGrid.re[k], fieldGrid.im[k]);
    }
    fMin = Math.min(...mag);
    fMax = Math.max(...mag);
    if (fMax - fMin < 1e-12) fMax = fMin + 1;
  }

  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const idx = i * ny + j;
      ctx.fillStyle = mag ? fieldColor((mag[idx] - fMin) / (fMax - fMin)) : permittivityColor(grid.re[idx], bg.re);
      ctx.fillRect(i * cellW, j * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }
  }

  if (fieldGrid) drawContrastOutline(cellW, cellH);
}

// --- Time-harmonic animation loop ---------------------------------------

let animFrameHandle = null;
const ANIMATION_PERIOD_MS = 2500; // wall-clock time for one full oscillation

function animationStep(startTime) {
  return (now) => {
    // Always reschedule, even if redraw() throws - a single bad frame
    // shouldn't permanently kill the loop (and would otherwise leave
    // animFrameHandle stuck non-null, blocking any future restart).
    try {
      const phase = (((now - startTime) / ANIMATION_PERIOD_MS) * 2 * Math.PI) % (2 * Math.PI);
      redraw(phase);
    } finally {
      animFrameHandle = requestAnimationFrame(animationStep(startTime));
    }
  };
}

function startAnimation() {
  if (animFrameHandle !== null) return;
  animFrameHandle = requestAnimationFrame(animationStep(performance.now()));
}

function stopAnimation() {
  if (animFrameHandle !== null) {
    cancelAnimationFrame(animFrameHandle);
    animFrameHandle = null;
  }
}

function syncAnimation() {
  if (isAnimatedMode(el("display-mode").value)) {
    startAnimation();
  } else {
    stopAnimation();
    redraw();
  }
}

function paintAt(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * canvas.width;
  const y = ((clientY - rect.top) / rect.height) * canvas.height;
  const { nx, ny } = grid;
  const cellW = canvas.width / nx;
  const cellH = canvas.height / ny;
  const ci = Math.floor(x / cellW);
  const cj = Math.floor(y / cellH);

  const radius = parseInt(el("brush-radius").value, 10);
  const erase = el("brush-erase").checked;
  const bg = backgroundValues();
  const brush = erase ? bg : { re: parseFloat(el("brush-re").value), im: parseFloat(el("brush-im").value) };

  for (let di = -radius; di <= radius; di++) {
    for (let dj = -radius; dj <= radius; dj++) {
      if (di * di + dj * dj > radius * radius) continue;
      const i = ci + di;
      const j = cj + dj;
      if (i < 0 || i >= nx || j < 0 || j >= ny) continue;
      grid.re[i * ny + j] = brush.re;
      grid.im[i * ny + j] = brush.im;
    }
  }
  redraw();
  scheduleSolve();
}

let painting = false;
canvas.addEventListener("pointerdown", (e) => {
  painting = true;
  paintAt(e.clientX, e.clientY);
});
canvas.addEventListener("pointermove", (e) => {
  if (painting) paintAt(e.clientX, e.clientY);
});
window.addEventListener("pointerup", () => {
  painting = false;
});

el("apply-domain").addEventListener("click", resetGrid);
el("clear-grid").addEventListener("click", resetGrid);
el("display-mode").addEventListener("change", syncAnimation);

for (const id of ["bg-re", "bg-im", "frequency", "incident-angle", "solver-restart", "solver-max-iter", "solver-tol"]) {
  el(id).addEventListener("change", () => scheduleSolve(150));
}

function setStatus(message, kind) {
  const statusEl = el("status");
  statusEl.textContent = message;
  statusEl.className = "status" + (kind ? " " + kind : "");
}

// --- Auto-solve -----------------------------------------------------------
// Painting fires constantly, so solves are debounced: each call resets the
// timer, and a solve only actually starts once things go quiet. If a solve
// is already in flight when the timer fires, one more run is queued for
// right after it finishes rather than piling up requests.

let debounceTimer = null;
let solveInFlight = false;
let solveQueued = false;

function scheduleSolve(delay = 500) {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(triggerSolve, delay);
}

async function triggerSolve() {
  if (solveInFlight) {
    solveQueued = true;
    return;
  }
  solveInFlight = true;
  try {
    await runSolve();
  } finally {
    solveInFlight = false;
    if (solveQueued) {
      solveQueued = false;
      triggerSolve();
    }
  }
}

async function submitJob(payload) {
  const createRes = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!createRes.ok) {
    throw new Error(await createRes.text());
  }
  const { job_id } = await createRes.json();

  const started = performance.now();
  while (true) {
    const pollRes = await fetch(`/api/jobs/${job_id}`);
    if (!pollRes.ok) {
      throw new Error(await pollRes.text());
    }
    const body = await pollRes.json();
    const elapsedS = ((performance.now() - started) / 1000).toFixed(1);

    if (body.status === "done") {
      return body;
    }
    if (body.status === "failed") {
      throw new Error(body.error || "solve failed");
    }
    setStatus(`${body.status}... (${elapsedS}s)`);
    await new Promise((r) => setTimeout(r, 300));
  }
}

async function runSolve() {
  setStatus("Solving...");
  const nx = grid.nx;
  const ny = grid.ny;
  try {
    const payload = {
      domain: { nx, ny, x_meters: grid.xMeters, y_meters: grid.yMeters },
      frequency_hz: parseFloat(el("frequency").value),
      background_permittivity: backgroundValues(),
      permittivity_re: Array.from(grid.re),
      permittivity_im: Array.from(grid.im),
      incident_angle_deg: parseFloat(el("incident-angle").value),
      receivers: [],
      solver: {
        restart: parseInt(el("solver-restart").value, 10),
        max_iter: parseInt(el("solver-max-iter").value, 10),
        tol: parseFloat(el("solver-tol").value),
      },
    };

    const result = await submitJob(payload);
    lastSolve = {
      nx,
      ny,
      result,
      totalScale: animationScale(result.total_field),
      scatteredScale: animationScale(result.scattered_field),
    };
    setStatus(`Done in ${result.solve_time_ms} ms`, "ok");
    syncAnimation();
  } catch (err) {
    setStatus(`Error: ${err.message}`, "error");
  }
}

resetGrid();
