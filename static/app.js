"use strict";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function setStatus(text, state /* idle | running | done | error */) {
  const chip = document.getElementById("statusChip");
  const label = document.getElementById("statusText");
  chip.className = `status-chip status-${state}`;
  label.textContent = text;
}

function activateSection(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add("active");
}

function revokeBlobUrls(container) {
  if (!container) return;
  const imgs = container.querySelectorAll ? container.querySelectorAll("img") : [];
  for (const img of imgs) {
    if (img.src && img.src.startsWith("blob:")) {
      URL.revokeObjectURL(img.src);
    }
  }
}

function resetViewPlaceholders() {
  const slots = [
    { id: "viewLeft",  icon: "L",  label: "Left Side Fullbody" },
    { id: "viewFront", icon: "F",  label: "Front Face" },
    { id: "viewRight", icon: "R",  label: "Right Side Fullbody" },
    { id: "viewFace",  icon: "FC", label: "Face Close-up" },
    { id: "viewBack",  icon: "B",  label: "Back Fullbody" },
  ];
  for (const slot of slots) {
    const el = document.getElementById(slot.id);
    if (!el) continue;
    revokeBlobUrls(el);
    el.classList.remove("loaded");
    el.innerHTML = `<div class="placeholder-inner"><div class="placeholder-icon">${slot.icon}</div><span>${slot.label}</span></div>`;
  }
}

// ---------------------------------------------------------------------------
// Smart defaults engine
// ---------------------------------------------------------------------------

function updateRecommendations(numImages) {
  let lr, steps, rank, batch;
  if (numImages <= 25) {
    lr = "1e-4"; steps = 1500; rank = 32; batch = 1;
  } else if (numImages <= 50) {
    lr = "8e-5"; steps = 2000; rank = 32; batch = 1;
  } else if (numImages <= 100) {
    lr = "5e-5"; steps = 2500; rank = 48; batch = 2;
  } else if (numImages <= 200) {
    lr = "4e-5"; steps = 3000; rank = 48; batch = 2;
  } else {
    lr = "3e-5"; steps = 3500; rank = 64; batch = 4;
  }

  const hintEl = document.getElementById("batchSizeHint");
  if (hintEl) {
    hintEl.textContent = `Recommended: ${batch} for ${numImages} images`;
  }

  return { lr, steps, rank, batch };
}

// ---------------------------------------------------------------------------
// Hero card — single large preview of the latest processed image
// ---------------------------------------------------------------------------

function updateHeroCard(imageUrl, stage, index, total) {
  const heroCard = document.getElementById("heroCard");
  const heroImg = document.getElementById("heroImg");
  const heroTitle = document.getElementById("heroTitle");
  const heroStep = document.getElementById("heroStep");
  if (!heroCard) return;

  const stageConfig = {
    gen:  { title: "Generating...",  cls: "hero-stage-gen" },
    up:   { title: "Upscaling...",   cls: "hero-stage-up" },
    enh:  { title: "Enhancing...",   cls: "hero-stage-enh" },
  };
  const cfg = stageConfig[stage] || stageConfig.gen;

  heroCard.classList.add("active", "pulse");
  heroTitle.textContent = cfg.title;
  heroTitle.className = "hero-card-title " + cfg.cls;
  heroStep.textContent = (index + 1) + " / " + total;

  // Show the image — reuse existing <img> if present, else create one
  let img = heroImg.querySelector("img");
  if (!img) {
    // Remove "Waiting..." text node
    heroImg.textContent = "";
    img = document.createElement("img");
    img.alt = "latest image";
    heroImg.appendChild(img);
  }
  img.src = imageUrl;
}

function resetHeroCard() {
  const heroCard = document.getElementById("heroCard");
  const heroImg = document.getElementById("heroImg");
  const heroTitle = document.getElementById("heroTitle");
  const heroStep = document.getElementById("heroStep");
  if (!heroCard) return;

  heroCard.classList.remove("active", "pulse");
  heroImg.textContent = "Waiting...";
  heroTitle.textContent = "Latest Image";
  heroTitle.className = "hero-card-title hero-stage-gen";
  heroStep.textContent = "--";
}

// ---------------------------------------------------------------------------
// Synthetic grid initialisation
// ---------------------------------------------------------------------------

function initSyntheticGrid(count) {
  const gridLeft  = document.getElementById("gridLeft");
  const gridRight = document.getElementById("gridRight");
  if (gridLeft)  { revokeBlobUrls(gridLeft);  gridLeft.innerHTML  = ""; }
  if (gridRight) { revokeBlobUrls(gridRight); gridRight.innerHTML = ""; }

  // Reset hero card whenever the grid is re-initialised
  resetHeroCard();

  const half = Math.ceil(count / 2);
  for (let i = 0; i < count; i++) {
    const cell = document.createElement("div");
    cell.className = "synthetic-cell";
    cell.dataset.index = i;
    cell.dataset.idx = String(i + 1).padStart(2, "0");
    cell.style.setProperty("--i", i);
    if (i < half) {
      if (gridLeft)  gridLeft.appendChild(cell);
    } else {
      if (gridRight) gridRight.appendChild(cell);
    }
  }
}

// ---------------------------------------------------------------------------
// Checkpoint rows
// ---------------------------------------------------------------------------

function addCheckpointRow(step, imageUrls, downloadUrl) {
  const container = document.getElementById("checkpointContainer");

  // Deduplicate: skip if a row for this step already exists
  if (container.querySelector(`.checkpoint-row[data-step="${step}"]`)) return;

  // Extract jobId from downloadUrl if present (format: /api/download-checkpoint/{jobId}/{step})
  let jobId = null;
  if (downloadUrl) {
    const m = downloadUrl.match(/\/api\/download-checkpoint\/([^/]+)\/\d+/);
    if (m) jobId = m[1];
  }

  const row = document.createElement("div");
  row.className = "checkpoint-row";
  row.dataset.step = step;
  row.style.animation = "slide-in 0.4s ease";

  const label = document.createElement("div");
  label.className = "checkpoint-row-label";
  label.textContent = `Step ${step.toLocaleString()}`;
  if (downloadUrl) {
    const dlBtn = document.createElement("a");
    dlBtn.className = "checkpoint-dl-btn";
    dlBtn.href = downloadUrl;
    dlBtn.download = "";
    dlBtn.title = `Download LoRA at step ${step.toLocaleString()}`;
    dlBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Download LoRA`;
    label.appendChild(dlBtn);
  }
  row.appendChild(label);

  const grid = document.createElement("div");
  grid.className = "checkpoint-images";

  imageUrls.forEach(url => {
    const wrap = document.createElement("div");
    wrap.className = "checkpoint-img-wrap";
    const img = document.createElement("img");
    img.src = url;
    img.alt = `checkpoint step ${step}`;
    img.loading = "lazy";
    img.dataset.type = "checkpoint";
    img.dataset.step = step;
    wrap.appendChild(img);

    // Click → fullscreen overlay with download button
    wrap.addEventListener("click", () => {
      openFullscreenOverlay(url, "checkpoint", { step, jobId });
    });

    grid.appendChild(wrap);
  });

  row.appendChild(grid);
  container.appendChild(row);

  activateSection("sectionCheckpoints");

  // Smooth scroll to reveal new row
  row.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---------------------------------------------------------------------------
// Upload handling
// ---------------------------------------------------------------------------

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    alert("Please upload an image file (PNG, JPG, or WEBP).");
    return;
  }

  uploadedFile = file;

  const reader = new FileReader();
  reader.onload = e => {
    const preview = document.getElementById("previewImage");
    const inner = document.getElementById("uploadInner");
    preview.src = e.target.result;
    preview.hidden = false;
    inner.style.display = "none";
    document.getElementById("uploadArea").classList.add("has-image");
  };
  reader.readAsDataURL(file);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Model Download UI (Ember-to-Forge)
// ---------------------------------------------------------------------------

function getForgeColor(pct) {
  // Ember-to-Forge: dim ember → orange → golden → bright gold → green
  const stops = [
    { pct: 0,   h: 15,  s: 80,  l: 30 },
    { pct: 30,  h: 25,  s: 90,  l: 48 },
    { pct: 60,  h: 38,  s: 95,  l: 52 },
    { pct: 85,  h: 48,  s: 100, l: 58 },
    { pct: 100, h: 150, s: 70,  l: 45 },
  ];
  let lower = stops[0], upper = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (pct >= stops[i].pct && pct <= stops[i + 1].pct) {
      lower = stops[i];
      upper = stops[i + 1];
      break;
    }
  }
  const range = upper.pct - lower.pct || 1;
  const t = (pct - lower.pct) / range;
  const h = Math.round(lower.h + (upper.h - lower.h) * t);
  const s = Math.round(lower.s + (upper.s - lower.s) * t);
  const l = Math.round(lower.l + (upper.l - lower.l) * t);
  return `hsl(${h}, ${s}%, ${l}%)`;
}

function getForgeGlow(pct) {
  const intensity = Math.min(pct / 100, 1);
  const spread = Math.round(4 + intensity * 12);
  const alpha = (0.1 + intensity * 0.4).toFixed(2);
  const color = getForgeColor(Math.min(pct, 85));
  const hsla = color.replace(/hsl\((.+)\)/, "hsla($1, " + alpha + ")");
  return `0 0 ${spread}px ${Math.round(spread * 0.6)}px ${hsla}`;
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function createModelDlBar(key, name, sizeHint, status) {
  const bar = document.createElement("div");
  bar.className = "model-dl-bar" + (status === "ready" ? " ready" : "");
  bar.id = `model-dl-${key}`;
  const isReady = status === "ready";
  bar.innerHTML =
    '<div class="model-dl-track">' +
      '<div class="model-dl-fill" style="width: ' + (isReady ? '100' : '0') + '%"></div>' +
      '<div class="model-dl-label">' +
        '<span class="model-dl-name">' + escapeHtml(name) + '</span>' +
        '<div class="model-dl-meta">' +
          '<span class="model-dl-size">' + escapeHtml(sizeHint) + '</span>' +
          '<span class="model-dl-pct">' + (isReady ? '\u2713' : '') + '</span>' +
        '</div>' +
      '</div>' +
    '</div>';
  return bar;
}

function updateModelDlCount() {
  const stack = document.getElementById("modelDlStack");
  const countEl = document.getElementById("modelDlCount");
  if (stack && countEl) {
    countEl.textContent = (stack.dataset.done || "0") + " / " + (stack.dataset.total || "0");
  }
}

function onModelDownload(data) {
  const section = document.getElementById("sectionModelDownload");
  const stack = document.getElementById("modelDlStack");
  if (!section || !stack) return;

  section.style.display = "";
  section.classList.add("active");

  if (data.action === "init") {
    stack.innerHTML = "";
    let readyCount = 0;
    for (const model of data.models) {
      stack.appendChild(createModelDlBar(model.key, model.name, model.size_hint, model.status));
      if (model.status === "ready") readyCount++;
    }
    stack.dataset.total = data.models.length;
    stack.dataset.done = readyCount;
    updateModelDlCount();
    return;
  }

  // Find or create bar
  let bar = document.getElementById("model-dl-" + data.key);
  if (!bar) {
    bar = createModelDlBar(data.key, data.name, data.size_hint || "", "queued");
    stack.appendChild(bar);
    stack.dataset.total = parseInt(stack.dataset.total || "0") + 1;
    updateModelDlCount();
  }

  const fill = bar.querySelector(".model-dl-fill");
  const pctEl = bar.querySelector(".model-dl-pct");

  if (data.action === "progress") {
    const pct = Math.round(data.percent);
    if (fill) {
      fill.style.width = pct + "%";
      fill.style.background = getForgeColor(pct);
      fill.style.boxShadow = getForgeGlow(pct);
    }
    if (pctEl) pctEl.textContent = pct + "%";
  } else if (data.action === "complete") {
    // Flash then settle to green
    bar.classList.add("forge-flash");
    if (fill) {
      fill.style.width = "100%";
    }
    if (pctEl) pctEl.textContent = "\u2713";
    setTimeout(function() {
      bar.classList.add("complete");
      bar.classList.remove("forge-flash");
      // Clear inline styles after class is applied so CSS takes over cleanly
      requestAnimationFrame(function() {
        if (fill) {
          fill.style.background = "";
          fill.style.boxShadow = "";
        }
      });
    }, 600);
    stack.dataset.done = parseInt(stack.dataset.done || "0") + 1;
    updateModelDlCount();
  } else if (data.action === "error") {
    bar.classList.remove("forge-flash");
    bar.classList.add("dl-error");
    if (fill) {
      fill.style.background = "";
      fill.style.boxShadow = "";
    }
    if (pctEl) pctEl.textContent = "\u2717";
  }
}

// SSE event handlers
// ---------------------------------------------------------------------------

function onStageEvent(data) {
  setStatus(data.status, "running");
  console.log("[stage]", data);
}

// Map view names to UI element IDs.
// v0.2 enhanced mode uses 10 dedicated slots; legacy uses 5.
const VIEW_NAME_TO_SLOT = {
  // v0.2 enhanced mode — 10 slots
  "front_face_closeup":         "viewFrontFaceCloseup",
  "front_midbody":              "viewFrontMidbody",
  "front_fullbody":             "viewFrontFullbody",
  "left_34_midbody":            "viewLeft34Midbody",
  "right_profile_closeup":      "viewRightProfileCloseup",
  "right_34_midbody":           "viewRight34Midbody",
  "left_fullbody_walking":      "viewLeftFullbodyWalking",
  "front_midbody_laughing":     "viewFrontMidbodyLaughing",
  "rear_34_midbody":            "viewRear34Midbody",
  "left_34_closeup_dramatic":   "viewLeft34CloseupDramatic",
  // Legacy 5 slots
  "left":  "viewLeft",
  "front": "viewFront",
  "right": "viewRight",
  "face":  "viewFace",
  "back":  "viewBack",
};

function onViewEvent(data, jobId) {
  const position = data.position;
  const slotId = (position in VIEW_NAME_TO_SLOT)
    ? VIEW_NAME_TO_SLOT[position]
    : `view${capitalize(position)}`;  // fallback for any unknown legacy name

  activateSection("sectionViews");

  if (slotId) {
    const el = document.getElementById(slotId);
    if (el && !el.classList.contains("loaded")) {
      const imgUrl = data.url + "?t=" + Date.now();
      const img = document.createElement("img");
      img.src = imgUrl;
      img.alt = position;
      img.loading = "lazy";
      img.dataset.type = "view";

      el.innerHTML = "";
      el.appendChild(img);
      el.classList.add("loaded");
      el.style.cursor = "pointer";

      el.onclick = () => openFullscreenOverlay(imgUrl, "view", {});
    }
  }

  // Show download button once enough slots are filled (5 legacy or 10 enhanced)
  const vgEnh = document.getElementById("viewsGridEnhanced");
  const enhancedOn = vgEnh && vgEnh.style.display !== "none";
  const activeGrid = enhancedOn ? "#viewsGridEnhanced" : "#viewsGrid";
  const loaded = document.querySelectorAll(`${activeGrid} .view-placeholder.loaded`).length;
  const threshold = enhancedOn ? 10 : 5;
  if (loaded >= threshold && jobId) {
    const dlBtn = document.getElementById("downloadViewsBtn");
    dlBtn.href = `/api/download-views/${jobId}`;
    dlBtn.hidden = false;
  }
}

function onSyntheticEvent(data, jobId) {
  const cell = document.querySelector(`.synthetic-cell[data-index="${data.index}"]`);
  if (!cell) return;

  const imgUrl = data.url + "?t=" + Date.now();
  const img = document.createElement("img");
  img.src = imgUrl;
  img.alt = `synthetic ${data.index + 1}`;
  img.loading = "lazy";
  img.dataset.type = "synthetic";

  // Extract image filename from URL for caption fetch
  const imageName = data.url.split("/").pop();
  cell.dataset.type = "synthetic";
  cell.dataset.imageName = imageName;
  cell.dataset.jobId = jobId || "";

  cell.innerHTML = "";
  cell.appendChild(img);
  cell.classList.add("loaded");

  // Update count
  const loaded = document.querySelectorAll(".synthetic-cell.loaded").length;
  const total = document.querySelectorAll(".synthetic-cell").length;
  document.getElementById("syntheticCount").textContent = `${loaded} / ${total}`;

  // Update hero card with latest generated image
  updateHeroCard(imgUrl, "gen", data.index, total);

  // Show download button once all synthetic images are generated
  if (loaded >= total && jobId) {
    const dlBtn = document.getElementById("downloadDatasetBtn");
    dlBtn.href = `/api/download-dataset/${jobId}`;
    dlBtn.hidden = false;
  }

  activateSection("sectionSynthetic");
}

function onProgressEvent(data) {
  const pct = Math.min((data.step / data.total) * 100, 100);
  document.getElementById("progressFill").style.width = `${pct}%`;
  document.getElementById("progressGlow").style.left = `${pct}%`;
  document.getElementById("progressText").textContent =
    `${data.step.toLocaleString()} / ${data.total.toLocaleString()}`;
  activateSection("sectionTraining");
}

function onCheckpointEvent(data) {
  addCheckpointRow(data.step, data.images, data.download_url || null);

  // Update live preview with the first sample image
  if (data.images && data.images.length > 0) {
    const livePreview = document.getElementById("livePreviewImg");
    if (livePreview) {
      let img = livePreview.querySelector("img");
      if (!img) {
        livePreview.textContent = "";
        img = document.createElement("img");
        img.alt = "checkpoint sample";
        livePreview.appendChild(img);
      }
      img.src = data.images[0];
      livePreview.title = `Step ${data.step.toLocaleString()}`;
    }
  }
}

function onDiffusionPreview(data) {
  const cell = document.querySelector(`.synthetic-cell[data-index="${data.index}"]`);
  if (!cell || cell.classList.contains("loaded")) return;

  // Create or update preview overlay
  let overlay = cell.querySelector(".diffusion-preview");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.className = "diffusion-preview";

    const img = document.createElement("img");
    img.className = "diffusion-preview-img";
    overlay.appendChild(img);

    const label = document.createElement("div");
    label.className = "diffusion-preview-label";
    overlay.appendChild(label);

    cell.appendChild(overlay);
  }

  overlay.querySelector(".diffusion-preview-img").src = data.preview;
  overlay.querySelector(".diffusion-preview-label").textContent =
    `${data.step}/${data.total_steps}`;

  activateSection("sectionSynthetic");
}

function onUpscaledEvent(data) {
  const cell = document.querySelector(`.synthetic-cell[data-index="${data.index}"]`);
  if (!cell) return;

  // Store both URLs on the cell for the comparison slider
  cell.dataset.originalUrl = data.original_url;
  cell.dataset.upscaledUrl = data.upscaled_url;
  cell.classList.add("has-comparison");

  // Update the displayed image to the upscaled version
  const upscaledUrl = data.upscaled_url + "?t=" + Date.now();
  const img = cell.querySelector("img:not(.diffusion-preview-img)");
  if (img) img.src = upscaledUrl;

  // Add compare badge
  if (!cell.querySelector(".compare-badge")) {
    const badge = document.createElement("div");
    badge.className = "compare-badge";
    badge.textContent = "2048px";
    cell.appendChild(badge);
  }

  // Update hero card with latest upscaled image
  const total = document.querySelectorAll(".synthetic-cell").length;
  updateHeroCard(upscaledUrl, "up", data.index, total);
}

// ---------------------------------------------------------------------------
// First-pass samples gallery
// ---------------------------------------------------------------------------

let firstPassSamples = [];

function onFirstPassCheckpointEvent(data) {
  if (!data.images || data.images.length === 0) return;

  // Store each image with its step — deduplicate on reconnect replay
  for (const url of data.images) {
    const isDuplicate = firstPassSamples.some(s => s.step === data.step && s.url === url);
    if (!isDuplicate) {
      firstPassSamples.push({ step: data.step, url: url });
    }
  }

  // Update live preview thumbnail with the latest sample
  const livePreview = document.getElementById("livePreviewImg");
  if (livePreview) {
    let img = livePreview.querySelector("img");
    if (!img) {
      livePreview.textContent = "";
      img = document.createElement("img");
      img.alt = "first pass sample";
      livePreview.appendChild(img);
    }
    img.src = data.images[data.images.length - 1];
    livePreview.title = "Click to view all samples";

    // Add clickable state
    livePreview.classList.add("has-samples");

    // Update or create badge
    let badge = livePreview.querySelector(".live-preview-badge");
    if (!badge) {
      badge = document.createElement("div");
      badge.className = "live-preview-badge";
      livePreview.appendChild(badge);
    }
    badge.textContent = firstPassSamples.length;
  }

  // Update meta text
  const meta = document.getElementById("livePreviewMeta");
  if (meta) {
    meta.textContent = `Step ${data.step.toLocaleString()} \u2022 ${firstPassSamples.length} sample(s) generated`;
  }
}

function openFirstPassModal() {
  if (firstPassSamples.length === 0) return;

  // Remove any existing modal
  closeFirstPassModal();

  const overlay = document.createElement("div");
  overlay.className = "fp-modal-overlay";
  overlay.id = "fpModalOverlay";

  const modal = document.createElement("div");
  modal.className = "fp-modal";

  // Header
  const header = document.createElement("div");
  header.className = "fp-modal-header";
  header.innerHTML = `
    <div>
      <span class="fp-modal-title">First Pass Samples</span>
      <span class="fp-modal-count">${firstPassSamples.length} image(s)</span>
    </div>
    <button class="fp-modal-close" id="fpModalClose">&times;</button>
  `;
  modal.appendChild(header);

  // Body with grid
  const body = document.createElement("div");
  body.className = "fp-modal-body";

  const grid = document.createElement("div");
  grid.className = "fp-samples-grid";
  grid.id = "fpSamplesGrid";

  for (let i = 0; i < firstPassSamples.length; i++) {
    grid.appendChild(_createFPSampleCard(firstPassSamples[i], i));
  }

  body.appendChild(grid);
  modal.appendChild(body);
  overlay.appendChild(modal);
  document.body.appendChild(overlay);

  // Close handlers
  document.getElementById("fpModalClose").addEventListener("click", closeFirstPassModal);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closeFirstPassModal();
  });

  // ESC key
  overlay._escHandler = (e) => {
    if (e.key === "Escape") closeFirstPassModal();
  };
  document.addEventListener("keydown", overlay._escHandler);
}

function _createFPSampleCard(sample, index) {
  const card = document.createElement("div");
  card.className = "fp-sample-card";
  card.dataset.index = index;

  const imgWrap = document.createElement("div");
  imgWrap.className = "fp-sample-img";
  const img = document.createElement("img");
  img.src = sample.url;
  img.alt = `Step ${sample.step}`;
  img.loading = "lazy";
  imgWrap.appendChild(img);
  card.appendChild(imgWrap);

  const label = document.createElement("div");
  label.className = "fp-sample-label";
  label.innerHTML = `<div class="fp-sample-step">Step ${sample.step.toLocaleString()}</div>`;
  card.appendChild(label);

  // Click to fullscreen
  card.addEventListener("click", () => {
    openFullscreenViewer(sample.url, `Step ${sample.step.toLocaleString()}`);
  });

  return card;
}

function closeFirstPassModal() {
  const overlay = document.getElementById("fpModalOverlay");
  if (overlay) {
    if (overlay._escHandler) {
      document.removeEventListener("keydown", overlay._escHandler);
    }
    overlay.remove();
  }
}

function updateFirstPassModalIfOpen() {
  const grid = document.getElementById("fpSamplesGrid");
  if (!grid) return; // modal not open

  // Update count
  const count = document.querySelector(".fp-modal-count");
  if (count) count.textContent = `${firstPassSamples.length} image(s)`;

  // Add any new cards
  const existing = grid.querySelectorAll(".fp-sample-card").length;
  for (let i = existing; i < firstPassSamples.length; i++) {
    grid.appendChild(_createFPSampleCard(firstPassSamples[i], i));
  }
}

// ---------------------------------------------------------------------------
// Fullscreen image viewer (legacy — used by first-pass modal)
// ---------------------------------------------------------------------------

function openFullscreenViewer(imageUrl, label) {
  closeFullscreenViewer();

  const viewer = document.createElement("div");
  viewer.className = "fullscreen-viewer";
  viewer.id = "fullscreenViewer";

  const closeBtn = document.createElement("div");
  closeBtn.className = "fullscreen-close";
  closeBtn.textContent = "\u00d7";
  viewer.appendChild(closeBtn);

  const img = document.createElement("img");
  img.src = imageUrl;
  img.alt = label;
  viewer.appendChild(img);

  const stepLabel = document.createElement("div");
  stepLabel.className = "fullscreen-step-label";
  stepLabel.textContent = label;
  viewer.appendChild(stepLabel);

  document.body.appendChild(viewer);

  // Close on click anywhere
  viewer.addEventListener("click", closeFullscreenViewer);

  // ESC key
  viewer._escHandler = (e) => {
    if (e.key === "Escape") closeFullscreenViewer();
  };
  document.addEventListener("keydown", viewer._escHandler);
}

function closeFullscreenViewer() {
  const viewer = document.getElementById("fullscreenViewer");
  if (viewer) {
    if (viewer._escHandler) {
      document.removeEventListener("keydown", viewer._escHandler);
    }
    viewer.remove();
  }
}

// ---------------------------------------------------------------------------
// Fullscreen overlay — unified modal for checkpoint, synthetic, and view images
// ---------------------------------------------------------------------------

let _fullscreenEscHandler = null;

/**
 * Open the fullscreen overlay for any image type.
 * @param {string} imageUrl  - URL of the image to display
 * @param {'checkpoint'|'synthetic'|'view'} type - image type
 * @param {object} opts      - type-specific options:
 *   checkpoint: { step, jobId }
 *   synthetic:  { jobId, imageName }
 *   view:       {}
 */
async function openFullscreenOverlay(imageUrl, type, opts = {}) {
  const overlay   = document.getElementById("fullscreenOverlay");
  const img       = document.getElementById("fullscreenOverlayImg");
  const caption   = document.getElementById("fullscreenCaption");
  const captionTx = document.getElementById("fullscreenCaptionText");
  const actions   = document.getElementById("fullscreenActions");
  const dlBtn     = document.getElementById("downloadCheckpointBtn");
  const stepSpan  = document.getElementById("downloadCheckpointStep");

  if (!overlay || !img) return;

  // Reset state
  img.src = imageUrl;
  caption.style.display = "none";
  captionTx.textContent = "";
  actions.style.display = "none";
  dlBtn.href = "#";

  if (type === "checkpoint") {
    const { step, jobId } = opts;
    actions.style.display = "";
    stepSpan.textContent = step ? step.toLocaleString() : "0";
    if (jobId && step != null) {
      dlBtn.href = `/api/download-checkpoint/${jobId}/${step}`;
      dlBtn.download = "";
    }
  } else if (type === "synthetic") {
    const { jobId, imageName } = opts;
    if (jobId && imageName) {
      caption.style.display = "";
      captionTx.textContent = "Loading…";
      try {
        const resp = await fetch(`/api/caption/${jobId}/${imageName}`);
        if (resp.ok) {
          const data = await resp.json();
          captionTx.textContent = data.caption || "(no caption)";
        } else {
          captionTx.textContent = "(caption unavailable)";
        }
      } catch (_) {
        captionTx.textContent = "(caption unavailable)";
      }
    }
  }
  // type === "view": no extras — just show the image

  // Show overlay, lock body scroll
  overlay.style.display = "flex";
  document.body.style.overflow = "hidden";

  // ESC key handler
  if (_fullscreenEscHandler) {
    document.removeEventListener("keydown", _fullscreenEscHandler);
  }
  _fullscreenEscHandler = (e) => {
    if (e.key === "Escape") closeFullscreenOverlay();
  };
  document.addEventListener("keydown", _fullscreenEscHandler);
}

function closeFullscreenOverlay() {
  const overlay = document.getElementById("fullscreenOverlay");
  if (overlay) overlay.style.display = "none";
  document.body.style.overflow = "";
  if (_fullscreenEscHandler) {
    document.removeEventListener("keydown", _fullscreenEscHandler);
    _fullscreenEscHandler = null;
  }
}

function handleFullscreenOverlayClick(e) {
  // Close only when clicking the backdrop (not the content)
  if (e.target === document.getElementById("fullscreenOverlay")) {
    closeFullscreenOverlay();
  }
}

function onFirstPassProgress(data) {
  const pct = Math.min((data.step / data.total) * 100, 100);
  document.getElementById("firstPassFill").style.width = pct + "%";
  document.getElementById("firstPassGlow").style.left = pct + "%";
  document.getElementById("firstPassText").textContent =
    data.step.toLocaleString() + " / " + data.total.toLocaleString();
  activateSection("sectionFirstPass");
}

// ---------------------------------------------------------------------------
// Identity LoRA training handlers
// ---------------------------------------------------------------------------

function onIdentityProgress(data) {
  const pct = Math.min((data.step / data.total_steps) * 100, 100);
  const fill = document.getElementById("identityProgressFill");
  const glow = document.getElementById("identityProgressGlow");
  const text = document.getElementById("identityProgressText");
  if (fill) fill.style.width = pct + "%";
  if (glow) glow.style.left = pct + "%";
  if (text) text.textContent = data.step.toLocaleString() + " / " + data.total_steps.toLocaleString();
  activateSection("sectionIdentityTraining");
}

function onIdentityCheckpoint(data, jobId) {
  // data: { step, checkpoint_path, sample_paths }
  const grid = document.getElementById("identityCheckpointGrid");
  if (!grid) return;

  activateSection("sectionIdentityTraining");

  for (const url of (data.sample_paths || [])) {
    // Deduplicate on replay
    if (grid.querySelector(`[data-url="${CSS.escape(url)}"]`)) continue;

    const wrap = document.createElement("div");
    wrap.className = "identity-checkpoint-img-wrap";
    wrap.dataset.url = url;
    wrap.dataset.step = data.step;

    const img = document.createElement("img");
    img.src = url + "?t=" + Date.now();
    img.alt = `Identity step ${data.step}`;
    img.loading = "lazy";
    wrap.appendChild(img);

    // Step label badge
    const badge = document.createElement("div");
    badge.className = "identity-checkpoint-badge";
    badge.textContent = `Step ${data.step.toLocaleString()}`;
    wrap.appendChild(badge);

    // Click → fullscreen with download button
    wrap.addEventListener("click", () => {
      openFullscreenOverlay(img.src, "checkpoint", { step: data.step, jobId });
    });

    grid.appendChild(wrap);
  }

  // Update live preview thumbnail
  if (data.sample_paths && data.sample_paths.length > 0) {
    const livePreview = document.getElementById("livePreviewImg");
    if (livePreview) {
      let previewImg = livePreview.querySelector("img");
      if (!previewImg) {
        livePreview.textContent = "";
        previewImg = document.createElement("img");
        previewImg.alt = "identity checkpoint sample";
        livePreview.appendChild(previewImg);
      }
      previewImg.src = data.sample_paths[data.sample_paths.length - 1];
      livePreview.title = `Identity Step ${data.step.toLocaleString()}`;
    }
    const meta = document.getElementById("livePreviewMeta");
    if (meta) {
      meta.textContent = `Identity LoRA \u2022 Step ${data.step.toLocaleString()}`;
    }
  }
}

function onIdentityComplete(data) {
  const fill = document.getElementById("identityProgressFill");
  const glow = document.getElementById("identityProgressGlow");
  if (fill) fill.style.width = "100%";
  if (glow) glow.style.left = "100%";
  activateSection("sectionIdentityTraining");
}

function onEnhanceProgress(data) {
  document.getElementById("enhancementCount").textContent =
    data.current + " / " + data.total;
  activateSection("sectionEnhancement");
}

function onEnhancedEvent(data) {
  const grid = document.getElementById("enhancementGrid");
  activateSection("sectionEnhancement");

  const cell = document.createElement("div");
  cell.className = "synthetic-cell loaded";
  cell.dataset.index = data.index;
  cell.style.setProperty("--i", data.index);

  // Store URLs for triple comparison
  cell.dataset.preEnhanceUrl = data.pre_enhance_url;
  cell.dataset.enhancedUrl = data.enhanced_url;
  cell.classList.add("has-comparison");

  const enhancedUrl = data.enhanced_url + "?t=" + Date.now();
  const img = document.createElement("img");
  img.src = enhancedUrl;
  img.alt = "enhanced " + (data.index + 1);
  img.loading = "lazy";
  cell.appendChild(img);

  // Add "Enhanced" badge
  const badge = document.createElement("div");
  badge.className = "compare-badge";
  badge.textContent = "Enhanced";
  cell.appendChild(badge);

  grid.appendChild(cell);

  // Update hero card with latest enhanced image (scope count to enhancement grid only)
  const enhTotal = document.querySelectorAll("#enhancementGrid .synthetic-cell").length;
  updateHeroCard(enhancedUrl, "enh", data.index, enhTotal);
}

// ---------------------------------------------------------------------------
// Before/After comparison slider (opens on click)
// ---------------------------------------------------------------------------

function openComparison(originalUrl, upscaledUrl) {
  // Remove existing overlay
  const existing = document.getElementById("comparisonOverlay");
  if (existing) existing.remove();

  const overlay = document.createElement("div");
  overlay.id = "comparisonOverlay";
  overlay.className = "comparison-overlay";

  const container = document.createElement("div");
  container.className = "comparison-container";

  const closeBtn = document.createElement("div");
  closeBtn.className = "comparison-close";
  closeBtn.textContent = "\u00d7";
  container.appendChild(closeBtn);

  const labels = document.createElement("div");
  labels.className = "comparison-labels";
  const labelLeft = document.createElement("span");
  labelLeft.className = "comparison-label-left";
  labelLeft.textContent = "Original 1024px";
  const labelRight = document.createElement("span");
  labelRight.className = "comparison-label-right";
  labelRight.textContent = "SeedVR2 2048px";
  labels.appendChild(labelLeft);
  labels.appendChild(labelRight);
  container.appendChild(labels);

  const wrapper = document.createElement("div");
  wrapper.className = "comparison-wrapper";

  const imgUpscaled = document.createElement("img");
  imgUpscaled.className = "comparison-img comparison-img-upscaled";
  imgUpscaled.src = upscaledUrl;
  imgUpscaled.alt = "upscaled";
  wrapper.appendChild(imgUpscaled);

  const clip = document.createElement("div");
  clip.className = "comparison-clip";
  const imgOriginal = document.createElement("img");
  imgOriginal.className = "comparison-img comparison-img-original";
  imgOriginal.src = originalUrl;
  imgOriginal.alt = "original";
  clip.appendChild(imgOriginal);
  wrapper.appendChild(clip);

  const slider = document.createElement("div");
  slider.className = "comparison-slider";
  const handle = document.createElement("div");
  handle.className = "comparison-handle";
  slider.appendChild(handle);
  wrapper.appendChild(slider);

  container.appendChild(wrapper);
  overlay.appendChild(container);
  document.body.appendChild(overlay);

  function updateSlider(x) {
    const rect = wrapper.getBoundingClientRect();
    const pct = Math.max(0, Math.min(100, ((x - rect.left) / rect.width) * 100));
    clip.style.width = pct + "%";
    slider.style.left = pct + "%";
  }

  // Start at 50%
  setTimeout(() => {
    const rect = wrapper.getBoundingClientRect();
    updateSlider(rect.left + rect.width * 0.5);
  }, 50);

  let dragging = false;

  // Use named handlers so we can remove window listeners on cleanup
  function onMouseUp() { dragging = false; }
  function cleanup() {
    window.removeEventListener("mouseup", onMouseUp);
    overlay.remove();
  }

  wrapper.addEventListener("mousedown", () => { dragging = true; });
  window.addEventListener("mouseup", onMouseUp);
  wrapper.addEventListener("mousemove", e => { if (dragging) updateSlider(e.clientX); });
  wrapper.addEventListener("click", e => updateSlider(e.clientX));

  // Touch support
  wrapper.addEventListener("touchstart", () => { dragging = true; });
  wrapper.addEventListener("touchend", () => { dragging = false; });
  wrapper.addEventListener("touchmove", e => {
    if (dragging) updateSlider(e.touches[0].clientX);
  });

  // Close — cleanup window listeners to prevent memory leak
  closeBtn.addEventListener("click", cleanup);
  overlay.addEventListener("click", e => { if (e.target === overlay) cleanup(); });
}

// ---------------------------------------------------------------------------
// Triple comparison viewer (tabbed: Original | Upscaled | Enhanced)
// ---------------------------------------------------------------------------

function openTripleComparison(originalUrl, upscaledUrl, enhancedUrl) {
  // Remove existing overlay
  const existing = document.getElementById("comparisonOverlay");
  if (existing) existing.remove();

  const overlay = document.createElement("div");
  overlay.id = "comparisonOverlay";
  overlay.className = "comparison-overlay";

  const container = document.createElement("div");
  container.className = "comparison-container triple-comparison-container";

  const closeBtn = document.createElement("div");
  closeBtn.className = "comparison-close";
  closeBtn.textContent = "\u00d7";
  container.appendChild(closeBtn);

  // Tab bar
  const tabs = document.createElement("div");
  tabs.className = "triple-tabs";
  const tabDefs = [
    { key: "original",  label: "Original (1024px)" },
    { key: "upscaled",  label: "Upscaled (2048px)" },
    { key: "enhanced",  label: "Enhanced (2048px)" },
  ];
  const tabEls = {};
  tabDefs.forEach(t => {
    const btn = document.createElement("button");
    btn.className = "triple-tab";
    btn.textContent = t.label;
    btn.dataset.key = t.key;
    tabs.appendChild(btn);
    tabEls[t.key] = btn;
  });
  container.appendChild(tabs);

  // Slider label row
  const labels = document.createElement("div");
  labels.className = "comparison-labels triple-slider-labels";
  const labelLeft = document.createElement("span");
  labelLeft.className = "comparison-label-left";
  const labelRight = document.createElement("span");
  labelRight.className = "comparison-label-right";
  labels.appendChild(labelLeft);
  labels.appendChild(labelRight);
  container.appendChild(labels);

  // Comparison wrapper (reuse existing slider mechanism)
  const wrapper = document.createElement("div");
  wrapper.className = "comparison-wrapper";

  const imgBack = document.createElement("img");
  imgBack.className = "comparison-img comparison-img-upscaled";
  imgBack.alt = "right panel";
  wrapper.appendChild(imgBack);

  const clip = document.createElement("div");
  clip.className = "comparison-clip";
  const imgFront = document.createElement("img");
  imgFront.className = "comparison-img comparison-img-original";
  imgFront.alt = "left panel";
  clip.appendChild(imgFront);
  wrapper.appendChild(clip);

  const sliderEl = document.createElement("div");
  sliderEl.className = "comparison-slider";
  const handle = document.createElement("div");
  handle.className = "comparison-handle";
  sliderEl.appendChild(handle);
  wrapper.appendChild(sliderEl);

  container.appendChild(wrapper);
  overlay.appendChild(container);
  document.body.appendChild(overlay);

  // URL map
  const urlMap = {
    original: originalUrl,
    upscaled: upscaledUrl,
    enhanced: enhancedUrl,
  };

  // State: which two panels are being compared
  // Default: upscaled (left) vs enhanced (right)
  let leftKey = "upscaled";
  let rightKey = "enhanced";

  function applyTab(activeKey) {
    // Determine pair: clicking a tab sets it as left, right becomes the "next" in sequence
    const order = ["original", "upscaled", "enhanced"];
    const idx = order.indexOf(activeKey);
    leftKey = activeKey;
    rightKey = order[(idx + 1) % order.length];

    // Update tab styles
    Object.entries(tabEls).forEach(([k, el]) => {
      el.classList.toggle("active", k === activeKey);
    });

    // Update images
    imgFront.src = urlMap[leftKey];
    imgBack.src  = urlMap[rightKey];

    // Update labels
    const labelNames = {
      original: "Original 1024px",
      upscaled: "SeedVR2 2048px",
      enhanced: "Enhanced 2048px",
    };
    labelLeft.textContent  = labelNames[leftKey];
    labelRight.textContent = labelNames[rightKey];

    // Reset slider to 50%
    setTimeout(() => {
      const rect = wrapper.getBoundingClientRect();
      updateSlider(rect.left + rect.width * 0.5);
    }, 30);
  }

  function updateSlider(x) {
    const rect = wrapper.getBoundingClientRect();
    const pct = Math.max(0, Math.min(100, ((x - rect.left) / rect.width) * 100));
    clip.style.width = pct + "%";
    sliderEl.style.left = pct + "%";
  }

  // Tab click handlers
  Object.entries(tabEls).forEach(([key, el]) => {
    el.addEventListener("click", () => applyTab(key));
  });

  // Default view: upscaled vs enhanced
  applyTab("upscaled");

  let dragging = false;
  function onMouseUp() { dragging = false; }
  function cleanup() {
    window.removeEventListener("mouseup", onMouseUp);
    overlay.remove();
  }

  wrapper.addEventListener("mousedown", () => { dragging = true; });
  window.addEventListener("mouseup", onMouseUp);
  wrapper.addEventListener("mousemove", e => { if (dragging) updateSlider(e.clientX); });
  wrapper.addEventListener("click", e => updateSlider(e.clientX));

  wrapper.addEventListener("touchstart", () => { dragging = true; });
  wrapper.addEventListener("touchend", () => { dragging = false; });
  wrapper.addEventListener("touchmove", e => {
    if (dragging) updateSlider(e.touches[0].clientX);
  });

  closeBtn.addEventListener("click", cleanup);
  overlay.addEventListener("click", e => { if (e.target === overlay) cleanup(); });
}

function onCompleteEvent(data, evtSource, startBtn) {
  setStatus("Complete!", "done");

  const section = document.getElementById("outputSection");
  section.hidden = false;
  section.style.animation = "fade-in 0.6s ease";

  document.getElementById("outputPath").textContent = data.lora_path;
  const dlLink = document.getElementById("downloadLink");
  dlLink.href = data.download_url;

  // Final progress = 100%
  document.getElementById("progressFill").style.width = "100%";
  document.getElementById("progressGlow").style.left = "100%";

  // Stop hero card pulse animation on completion
  const heroCard = document.getElementById("heroCard");
  if (heroCard) heroCard.classList.remove("pulse");
  const heroTitle = document.getElementById("heroTitle");
  if (heroTitle) heroTitle.textContent = "Complete";
  const heroStep = document.getElementById("heroStep");
  if (heroStep) heroStep.textContent = "Done";

  startBtn.disabled = false;
  evtSource.close();

  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

function onErrorEvent(data, evtSource, startBtn) {
  setStatus(`Error: ${data.message}`, "error");
  console.error("[pipeline error]", data.message);

  // Stop hero card animation on error
  const heroCard = document.getElementById("heroCard");
  if (heroCard) heroCard.classList.remove("pulse");
  const heroTitle = document.getElementById("heroTitle");
  if (heroTitle) heroTitle.textContent = "Error";

  startBtn.disabled = false;
  evtSource.close();
}

// ---------------------------------------------------------------------------
// SSE connection (reusable — used by start button and auto-reconnect)
// ---------------------------------------------------------------------------

function connectToJob(jobId, startBtn) {
  startBtn.disabled = true;
  setStatus(`Running… job ${jobId}`, "running");

  // Reset first-pass gallery for fresh replay
  firstPassSamples = [];
  closeFirstPassModal();
  closeFullscreenViewer();

  // Show correct views grid based on enhanced mode toggle
  const enhToggle = document.getElementById("enhancedMode");
  const vgLegacy = document.getElementById("viewsGrid");
  const vgEnhanced = document.getElementById("viewsGridEnhanced");
  if (enhToggle && vgLegacy && vgEnhanced) {
    const enhanced = enhToggle.checked;
    vgLegacy.style.display = enhanced ? "none" : "";
    vgEnhanced.style.display = enhanced ? "" : "none";
  }

  const evtSource = new EventSource(`/api/stream/${jobId}`);

  evtSource.addEventListener("stage", e => {
    onStageEvent(JSON.parse(e.data));
  });

  evtSource.addEventListener("model_download", e => {
    onModelDownload(JSON.parse(e.data));
  });

  evtSource.addEventListener("view", e => {
    onViewEvent(JSON.parse(e.data), jobId);
  });

  evtSource.addEventListener("synthetic", e => {
    onSyntheticEvent(JSON.parse(e.data), jobId);
  });

  evtSource.addEventListener("diffusion_preview", e => {
    onDiffusionPreview(JSON.parse(e.data));
  });

  evtSource.addEventListener("upscaled", e => {
    onUpscaledEvent(JSON.parse(e.data));
  });

  evtSource.addEventListener("progress", e => {
    onProgressEvent(JSON.parse(e.data));
  });

  evtSource.addEventListener("checkpoint", e => {
    onCheckpointEvent(JSON.parse(e.data));
  });

  evtSource.addEventListener("identity_progress", e => {
    onIdentityProgress(JSON.parse(e.data));
  });

  evtSource.addEventListener("identity_checkpoint", e => {
    onIdentityCheckpoint(JSON.parse(e.data), jobId);
  });

  evtSource.addEventListener("identity_complete", e => {
    onIdentityComplete(JSON.parse(e.data));
  });

  evtSource.addEventListener("reg_image", e => {
    const data = JSON.parse(e.data);
    lastActivity = Date.now();
    const section = document.getElementById("sectionRegularization");
    if (section) section.style.display = "";
    const grid = document.getElementById("regGrid");
    if (grid) {
      const img = document.createElement("img");
      const imgUrl = data.url + "?t=" + Date.now();
      img.src = imgUrl;
      img.alt = `reg ${data.index}`;
      img.loading = "lazy";
      img.onclick = () => openFullscreenOverlay(imgUrl, "view", {});
      grid.appendChild(img);
    }
    const regText = document.getElementById("regProgressText");
    if (regText) regText.textContent = `${data.index + 1} / 25`;
  });

  evtSource.addEventListener("reg_progress", e => {
    const data = JSON.parse(e.data);
    lastActivity = Date.now();
    const section = document.getElementById("sectionRegularization");
    if (section) section.style.display = "";
    const regText = document.getElementById("regProgressText");
    if (regText) regText.textContent = `${data.current} / ${data.total}`;
    const stageEl = document.getElementById("stageMessage");
    if (stageEl) {
      stageEl.textContent = `Generating regularization image ${data.current}/${data.total}...`;
    }
  });

  evtSource.addEventListener("enhanced", e => {
    onEnhancedEvent(JSON.parse(e.data));
  });

  evtSource.addEventListener("enhance_progress", e => {
    onEnhanceProgress(JSON.parse(e.data));
  });

  evtSource.addEventListener("complete", e => {
    onCompleteEvent(JSON.parse(e.data), evtSource, startBtn);
  });

  evtSource.addEventListener("error", e => {
    if (e.data) {
      try {
        onErrorEvent(JSON.parse(e.data), evtSource, startBtn);
      } catch (_) {
        // connection-level error (no data) — ignore heartbeat drops
      }
    }
  });

  evtSource.addEventListener("heartbeat", () => {
    // Keep-alive — do nothing
  });

  // Fallback timeout: if no event arrives for 5 minutes, surface an error
  let lastActivity = Date.now();
  const activityGuard = setInterval(() => {
    if (Date.now() - lastActivity > 5 * 60 * 1000) {
      clearInterval(activityGuard);
      onErrorEvent(
        { message: "No activity for 5 minutes. Check server logs." },
        evtSource,
        startBtn,
      );
    }
  }, 30_000);

  [
    "stage", "view", "synthetic", "diffusion_preview", "upscaled",
    "progress", "checkpoint", "enhanced",
    "enhance_progress", "identity_progress", "identity_checkpoint", "identity_complete",
    "reg_progress", "reg_image", "complete", "heartbeat",
  ].forEach(
    name => evtSource.addEventListener(name, () => { lastActivity = Date.now(); })
  );

  evtSource.addEventListener("complete", () => clearInterval(activityGuard));
  evtSource.addEventListener("error",    () => clearInterval(activityGuard));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

let uploadedFile = null;
let viewsZipFile = null;
let datasetZipFile = null;

document.addEventListener("DOMContentLoaded", () => {
  const uploadArea   = document.getElementById("uploadArea");
  const imageInput   = document.getElementById("imageInput");
  const startBtn     = document.getElementById("startBtn");
  const loraStepsEl  = document.getElementById("loraSteps");
  const viewsZipInput = document.getElementById("viewsZipInput");
  const viewsZipName  = document.getElementById("viewsZipName");
  const viewsZipClear = document.getElementById("viewsZipClear");
  const datasetZipInput = document.getElementById("datasetZipInput");
  const datasetZipName  = document.getElementById("datasetZipName");
  const datasetZipClear = document.getElementById("datasetZipClear");

  // Views zip upload — extract and preview images
  viewsZipInput.addEventListener("change", async e => {
    const file = e.target.files[0];
    if (!file || !file.name.endsWith(".zip")) return;

    // Revoke previous blob URLs before loading new zip
    resetViewPlaceholders();
    viewsZipFile = file;
    viewsZipName.textContent = file.name;
    viewsZipClear.hidden = false;

    // Extract images from zip and populate view placeholders
    try {
      const zip = await JSZip.loadAsync(file);
      const imageFiles = Object.keys(zip.files)
        .filter(name => !zip.files[name].dir && /\.(png|jpg|jpeg|webp)$/i.test(name))
        .sort();

      const viewSlots = ["viewLeft", "viewFront", "viewRight", "viewFace", "viewBack"];
      const viewNames = ["left", "front", "right", "face", "back"];

      // Try matching by name first (left.png, front.png, right.png)
      const matched = [];
      for (const vn of viewNames) {
        const found = imageFiles.find(f => f.toLowerCase().replace(/.*\//, "").startsWith(vn));
        matched.push(found || null);
      }

      // Fill remaining slots with unmatched images in order
      const used = new Set(matched.filter(Boolean));
      const remaining = imageFiles.filter(f => !used.has(f));
      for (let i = 0; i < matched.length; i++) {
        if (!matched[i] && remaining.length > 0) {
          matched[i] = remaining.shift();
        }
      }

      for (let i = 0; i < viewSlots.length; i++) {
        if (!matched[i]) continue;
        const blob = await zip.files[matched[i]].async("blob");
        const url = URL.createObjectURL(blob);
        const el = document.getElementById(viewSlots[i]);
        const img = document.createElement("img");
        img.src = url;
        img.alt = viewNames[i];
        el.innerHTML = "";
        el.appendChild(img);
        el.classList.add("loaded");
      }

      activateSection("sectionViews");
    } catch (err) {
      console.error("Failed to preview zip contents:", err);
    }
  });

  viewsZipClear.addEventListener("click", () => {
    viewsZipFile = null;
    viewsZipName.textContent = "Choose .zip file...";
    viewsZipClear.hidden = true;
    viewsZipInput.value = "";

    // Reset view placeholders
    resetViewPlaceholders();
    document.getElementById("sectionViews").classList.remove("active");
  });

  // Dataset zip upload
  datasetZipInput.addEventListener("change", e => {
    const file = e.target.files[0];
    if (!file || !file.name.endsWith(".zip")) return;
    datasetZipFile = file;
    datasetZipName.textContent = file.name;
    datasetZipClear.hidden = false;
    // Clear server dataset selection (mutually exclusive)
    document.getElementById("existingDataset").value = "";
  });

  datasetZipClear.addEventListener("click", () => {
    datasetZipFile = null;
    datasetZipName.textContent = "Choose .zip file...";
    datasetZipClear.hidden = true;
    datasetZipInput.value = "";
  });

  // Existing dataset selector
  const existingDataset = document.getElementById("existingDataset");
  const refreshDatasets = document.getElementById("refreshDatasets");

  async function fetchDatasets() {
    try {
      existingDataset.disabled = true;
      refreshDatasets.disabled = true;
      const resp = await fetch("/api/datasets");
      const data = await resp.json();
      existingDataset.innerHTML = '<option value="">Select from server...</option>';
      if (data.datasets && data.datasets.length > 0) {
        for (const ds of data.datasets) {
          const opt = document.createElement("option");
          opt.value = ds.job_id;
          opt.dataset.count = ds.image_count;
          const date = new Date(ds.created).toLocaleDateString();
          const lora = ds.has_lora ? " \u2713" : "";
          opt.textContent = `${ds.job_id} \u2014 ${ds.image_count} imgs (${date})${lora}`;
          existingDataset.appendChild(opt);
        }
      } else {
        const opt = document.createElement("option");
        opt.value = "";
        opt.disabled = true;
        opt.textContent = "No datasets found on server";
        existingDataset.appendChild(opt);
      }
    } catch (err) {
      console.log("[datasets] Could not fetch:", err.message);
    } finally {
      existingDataset.disabled = false;
      refreshDatasets.disabled = false;
    }
  }

  fetchDatasets();
  refreshDatasets.addEventListener("click", () => fetchDatasets());

  existingDataset.addEventListener("change", () => {
    if (existingDataset.value) {
      // Clear zip selection (mutually exclusive)
      datasetZipFile = null;
      datasetZipName.textContent = "Choose .zip file...";
      datasetZipClear.hidden = true;
      datasetZipInput.value = "";
    }
  });

  // ---- Settings panel ----
  const gearBtn        = document.getElementById("gearBtn");
  const settingsOverlay = document.getElementById("settingsOverlay");
  const settingsPanel  = document.getElementById("settingsPanel");
  const panelClose     = document.getElementById("panelClose");

  function openPanel()  {
    settingsOverlay.classList.add("open");
    settingsPanel.classList.add("open");
  }
  function closePanel() {
    settingsOverlay.classList.remove("open");
    settingsPanel.classList.remove("open");
  }
  if (gearBtn)         gearBtn.addEventListener("click", openPanel);
  if (panelClose)      panelClose.addEventListener("click", closePanel);
  if (settingsOverlay) settingsOverlay.addEventListener("click", closePanel);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && settingsPanel.classList.contains("open")) closePanel();
  });

  // ---- Trigger word + sample prompt live preview ----
  const triggerInput      = document.getElementById("triggerWord");
  const previewTrigger    = document.getElementById("previewTrigger");
  const livePreviewPrompt = document.getElementById("livePreviewPrompt");
  const promptList        = document.getElementById("promptList");
  const addPromptBtn      = document.getElementById("addPromptBtn");

  function getPromptInputs() {
    return promptList ? Array.from(promptList.querySelectorAll(".prompt-input")) : [];
  }

  function getSamplePromptsValue() {
    return getPromptInputs().map(el => el.value.trim()).filter(Boolean).join("\n");
  }

  function createPromptEntry(placeholder) {
    const entry = document.createElement("div");
    entry.className = "prompt-entry";
    const input = document.createElement("input");
    input.type = "text";
    input.className = "prompt-input";
    input.spellcheck = false;
    input.placeholder = placeholder || "Describe a scene with TRIGGER...";
    input.addEventListener("input", updatePreviewPrompt);
    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "prompt-remove-btn";
    removeBtn.title = "Remove";
    removeBtn.innerHTML = "&times;";
    removeBtn.addEventListener("click", () => {
      entry.remove();
      updatePreviewPrompt();
    });
    entry.appendChild(input);
    entry.appendChild(removeBtn);
    return entry;
  }

  if (addPromptBtn && promptList) {
    addPromptBtn.addEventListener("click", () => {
      promptList.appendChild(createPromptEntry());
      const inputs = getPromptInputs();
      inputs[inputs.length - 1].focus();
    });
  }

  // Wire up remove buttons on the initial HTML entries
  if (promptList) {
    promptList.querySelectorAll(".prompt-remove-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        btn.closest(".prompt-entry").remove();
        updatePreviewPrompt();
      });
    });
    promptList.querySelectorAll(".prompt-input").forEach(input => {
      input.addEventListener("input", updatePreviewPrompt);
    });
  }

  function updatePreviewPrompt() {
    if (!livePreviewPrompt) return;
    const tw = (triggerInput && triggerInput.value) || "chrx";
    const firstPrompt = getPromptInputs().map(el => el.value.trim()).find(Boolean) || "";
    if (firstPrompt) {
      const prompt = firstPrompt.replace(/TRIGGER/g, `<span class="trigger-highlight">${tw}</span>`);
      livePreviewPrompt.innerHTML = prompt;
    } else {
      livePreviewPrompt.innerHTML = `a portrait of <span class="trigger-highlight">${tw}</span>, looking at the camera, studio lighting`;
    }
  }

  if (triggerInput) triggerInput.addEventListener("input", updatePreviewPrompt);
  updatePreviewPrompt();  // set initial state

  // ---- Live preview click → open first-pass samples modal ----
  const livePreviewImgEl = document.getElementById("livePreviewImg");
  if (livePreviewImgEl) {
    livePreviewImgEl.addEventListener("click", () => {
      if (firstPassSamples.length > 0) {
        openFirstPassModal();
      }
    });
  }

  // Init grid with default 25 placeholders
  initSyntheticGrid(25);

  // Click handler for before/after (or triple) comparison on synthetic cells
  // Cells live in gridLeft and gridRight — use event delegation on a common ancestor
  function handleSyntheticClick(e) {
    const cell = e.target.closest(".synthetic-cell");
    if (!cell || !cell.classList.contains("loaded")) return;
    const hasEnhanced = cell.dataset.enhancedUrl && cell.dataset.originalUrl && cell.dataset.upscaledUrl;
    if (hasEnhanced) {
      openTripleComparison(cell.dataset.originalUrl, cell.dataset.upscaledUrl, cell.dataset.enhancedUrl);
    } else if (cell.classList.contains("has-comparison")) {
      openComparison(cell.dataset.originalUrl, cell.dataset.upscaledUrl);
    } else if (cell.dataset.type === "synthetic") {
      // Plain synthetic image — show caption overlay
      const img = cell.querySelector("img");
      const imgUrl = img ? img.src : "";
      openFullscreenOverlay(imgUrl, "synthetic", {
        jobId: cell.dataset.jobId || null,
        imageName: cell.dataset.imageName || null,
      });
    }
  }

  const gridLeftEl  = document.getElementById("gridLeft");
  const gridRightEl = document.getElementById("gridRight");
  if (gridLeftEl)  gridLeftEl.addEventListener("click",  handleSyntheticClick);
  if (gridRightEl) gridRightEl.addEventListener("click", handleSyntheticClick);

  // Click handler for enhanced image comparison
  document.getElementById("enhancementGrid").addEventListener("click", e => {
    const cell = e.target.closest(".synthetic-cell.has-comparison");
    if (!cell) return;
    openComparison(cell.dataset.preEnhanceUrl, cell.dataset.enhancedUrl);
  });

  // When synthesizer changes, lock inference steps for Klein
  document.getElementById("synthesizer").addEventListener("change", e => {
    const stepsEl = document.getElementById("inferenceSteps");
    if (e.target.value === "klein_kv") {
      stepsEl.dataset.prevValue = stepsEl.value;
      stepsEl.value = 4;
      stepsEl.disabled = true;
    } else {
      stepsEl.value = stepsEl.dataset.prevValue || 50;
      stepsEl.disabled = false;
    }
  });

  // Keep grid in sync with numImages setting; update recommendations
  document.getElementById("numImages").addEventListener("input", e => {
    const n = Math.max(10, Math.min(300, parseInt(e.target.value, 10) || 25));
    initSyntheticGrid(n);
    updateRecommendations(n);
  });

  // Enhanced mode toggle
  const enhancedModeToggle = document.getElementById("enhancedMode");
  const enhancementSettings = document.getElementById("enhancementSettings");

  if (enhancedModeToggle && enhancementSettings) {
    const viewsGrid = document.getElementById("viewsGrid");
    const viewsGridEnhanced = document.getElementById("viewsGridEnhanced");
    enhancedModeToggle.addEventListener("change", () => {
      const enhanced = enhancedModeToggle.checked;
      enhancementSettings.style.display = enhanced ? "" : "none";
      if (viewsGrid) viewsGrid.style.display = enhanced ? "none" : "";
      if (viewsGridEnhanced) viewsGridEnhanced.style.display = enhanced ? "" : "none";
    });
  }

  // Initial recommendations for default 25 images
  updateRecommendations(25);

  // ---------------------------------------------------------------------------
  // Upload interactions
  // ---------------------------------------------------------------------------

  uploadArea.addEventListener("click", () => imageInput.click());

  uploadArea.addEventListener("dragover", e => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add("dragover");
  });

  uploadArea.addEventListener("dragleave", e => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove("dragover");
  });

  uploadArea.addEventListener("drop", e => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  imageInput.addEventListener("change", e => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  });

  // ---------------------------------------------------------------------------
  // Start pipeline
  // ---------------------------------------------------------------------------

  startBtn.addEventListener("click", async () => {
    const selectedDatasetId = document.getElementById("existingDataset").value;
    if (!uploadedFile && !viewsZipFile && !datasetZipFile && !selectedDatasetId) {
      alert("Please upload a character image, views zip, dataset zip, or select an existing dataset.");
      return;
    }

    const geminiKey = document.getElementById("geminiKey").value.trim();
    if (!geminiKey && !viewsZipFile && !datasetZipFile && !selectedDatasetId) {
      alert("Please enter your Gemini API key, or upload a views/dataset zip to skip generation.");
      document.getElementById("geminiKey").focus();
      return;
    }

    // Reset UI state — use existing dataset image count if selected
    let numImages = parseInt(document.getElementById("numImages").value, 10) || 25;
    if (selectedDatasetId) {
      const opt = document.getElementById("existingDataset").selectedOptions[0];
      if (opt && opt.dataset.count) numImages = parseInt(opt.dataset.count, 10);
    }
    initSyntheticGrid(numImages);
    resetHeroCard();
    document.getElementById("syntheticCount").textContent = `0 / ${numImages}`;

    // Reset view placeholders
    resetViewPlaceholders();

    // Reset progress
    document.getElementById("progressFill").style.width = "0%";
    document.getElementById("progressGlow").style.left = "0%";
    const totalSteps = parseInt(loraStepsEl.value, 10) || 1500;
    document.getElementById("progressText").textContent = `0 / ${totalSteps.toLocaleString()}`;

    // Reset first-pass progress
    document.getElementById("firstPassFill").style.width = "0%";
    document.getElementById("firstPassGlow").style.left = "0%";
    const firstPassStepsVal = parseInt(document.getElementById("firstPassSteps").value, 10) || 1000;
    document.getElementById("firstPassText").textContent = `0 / ${firstPassStepsVal.toLocaleString()}`;

    // Reset enhancement count
    document.getElementById("enhancementCount").textContent = "0 / 0";
    document.getElementById("enhancementGrid").innerHTML = "";

    // Reset checkpoint container
    document.getElementById("checkpointContainer").innerHTML = "";

    // Reset live preview and first-pass samples gallery
    firstPassSamples = [];
    closeFirstPassModal();
    closeFullscreenViewer();
    const livePreviewReset = document.getElementById("livePreviewImg");
    if (livePreviewReset) {
      livePreviewReset.textContent = "Step 0";
      livePreviewReset.title = "";
      livePreviewReset.classList.remove("has-samples");
      const oldBadge = livePreviewReset.querySelector(".live-preview-badge");
      if (oldBadge) oldBadge.remove();
    }
    const metaReset = document.getElementById("livePreviewMeta");
    if (metaReset) metaReset.textContent = "Checkpoint sample \u2022 First pass: 100 steps \u2022 Final: 250 steps";

    // Hide output section and download buttons
    document.getElementById("outputSection").hidden = true;
    const dlBtn = document.getElementById("downloadViewsBtn");
    dlBtn.hidden = true;
    dlBtn.href = "#";
    const dlDatasetBtn = document.getElementById("downloadDatasetBtn");
    dlDatasetBtn.hidden = true;
    dlDatasetBtn.href = "#";

    // Reset identity training section
    const identityFill = document.getElementById("identityProgressFill");
    const identityGlow = document.getElementById("identityProgressGlow");
    const identityText = document.getElementById("identityProgressText");
    const identityGrid = document.getElementById("identityCheckpointGrid");
    if (identityFill) identityFill.style.width = "0%";
    if (identityGlow) identityGlow.style.left = "0%";
    if (identityText) identityText.textContent = "0 / 750";
    if (identityGrid) identityGrid.innerHTML = "";

    // Deactivate pipeline sections
    [
      "sectionViews", "sectionSynthetic", "sectionFirstPass",
      "sectionEnhancement", "sectionTraining", "sectionCheckpoints",
      "sectionIdentityTraining",
    ].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.classList.remove("active");
    });

    // Update synthesizer tag in UI
    const synthChoice = document.getElementById("synthesizer").value;
    const synthTag = document.getElementById("synthesizerTag");
    if (synthTag) {
      synthTag.textContent = synthChoice === "klein_kv" ? "Klein 9B" : "Flux 2";
    }

    // Klein uses fixed 4 steps — restore inference steps value after submit
    if (synthChoice === "klein_kv") {
      document.getElementById("inferenceSteps").value = 4;
    }

    startBtn.disabled = true;
    setStatus("Starting pipeline...", "running");

    // Build form data
    const formData = new FormData();
    if (uploadedFile) {
      formData.append("image", uploadedFile);
    }
    formData.append("trigger_word", document.getElementById("triggerWord").value.trim() || "chrx");
    formData.append("gemini_key", geminiKey);
    formData.append("hf_token", document.getElementById("hfToken").value.trim());
    formData.append("num_images", numImages);
    formData.append("synthesizer", document.getElementById("synthesizer").value);
    formData.append("base_model", document.getElementById("baseModel").value);
    formData.append("lora_rank", document.getElementById("loraRank").value);
    formData.append("lora_steps", totalSteps);
    formData.append("learning_rate", document.getElementById("learningRate").value);
    formData.append("inference_steps", document.getElementById("inferenceSteps").value);
    formData.append("batch_size", document.getElementById("batchSize").value);
    formData.append("enhanced_mode", document.getElementById("enhancedMode").checked ? "true" : "false");
    if (document.getElementById("enhancedMode").checked) {
      formData.append("first_pass_rank", document.getElementById("firstPassRank").value);
      formData.append("first_pass_steps", document.getElementById("firstPassSteps").value);
      formData.append("enhance_denoise", document.getElementById("enhanceDenoise").value);
      formData.append("enhance_steps", document.getElementById("enhanceSteps").value);
      formData.append("enhance_lora_weight", document.getElementById("enhanceLoraWeight").value);
      if (document.getElementById("recaptionAfterEnhance").checked) {
        formData.append("recaption_after_enhance", "true");
      }
    }

    const samplePromptsRaw = getSamplePromptsValue();
    if (samplePromptsRaw) {
      formData.append("sample_prompts", samplePromptsRaw);
    }
    if (viewsZipFile) {
      formData.append("views_zip", viewsZipFile);
    }
    if (datasetZipFile) {
      formData.append("dataset_zip", datasetZipFile);
    }
    if (selectedDatasetId) {
      formData.append("existing_dataset", selectedDatasetId);
    }

    let jobId;
    try {
      const json = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/start");

        xhr.upload.addEventListener("progress", (e) => {
          if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100);
            const mb = (e.loaded / 1024 / 1024).toFixed(1);
            const totalMb = (e.total / 1024 / 1024).toFixed(1);
            setStatus(`Uploading... ${pct}% (${mb} / ${totalMb} MB)`, "running");
          }
        });

        xhr.addEventListener("load", () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try { resolve(JSON.parse(xhr.responseText)); }
            catch { reject(new Error("Invalid server response")); }
          } else {
            try {
              const err = JSON.parse(xhr.responseText);
              reject(new Error(err.error || xhr.statusText));
            } catch { reject(new Error(xhr.statusText)); }
          }
        });

        xhr.addEventListener("error", () => reject(new Error("Network error — upload failed")));
        xhr.addEventListener("timeout", () => reject(new Error("Upload timed out")));
        xhr.timeout = 600000; // 10 minute timeout for large files

        setStatus("Uploading...", "running");
        xhr.send(formData);
      });
      jobId = json.job_id;
    } catch (err) {
      setStatus(`Failed to start: ${err.message}`, "error");
      startBtn.disabled = false;
      return;
    }

    connectToJob(jobId, startBtn);
  });

  // ---------------------------------------------------------------------------
  // Auto-reconnect: check for active job on page load
  // ---------------------------------------------------------------------------

  (async () => {
    try {
      const resp = await fetch("/api/jobs/active");
      const data = await resp.json();
      if (data.job_id) {
        console.log("[reconnect] Found active job:", data.job_id);

        // Prepare the grid with the right count
        const numImages = (data.params && data.params.num_images) || 25;
        initSyntheticGrid(numImages);
        document.getElementById("syntheticCount").textContent = `0 / ${numImages}`;

        // Activate enhanced mode sections and views grid if relevant
        if (data.params && data.params.enhanced_mode) {
          const fpSection = document.getElementById("sectionFirstPass");
          if (fpSection) fpSection.classList.add("active");
          const enhSection = document.getElementById("sectionEnhancement");
          if (enhSection) enhSection.classList.add("active");
          // Toggle to enhanced 10-slot views grid
          const vgLegacy = document.getElementById("viewsGrid");
          const vgEnhanced = document.getElementById("viewsGridEnhanced");
          if (vgLegacy) vgLegacy.style.display = "none";
          if (vgEnhanced) vgEnhanced.style.display = "";
          // Sync the checkbox so connectToJob picks it up
          const enhToggle = document.getElementById("enhancedMode");
          if (enhToggle) enhToggle.checked = true;
        }

        // Reconnect — the server replays all past events
        connectToJob(data.job_id, startBtn);
      }
    } catch (err) {
      console.log("[reconnect] No active job or server unreachable:", err.message);
    }
  })();
});

// ---------------------------------------------------------------------------
// Update step numbers on pipeline sections based on enhanced mode
// ---------------------------------------------------------------------------

