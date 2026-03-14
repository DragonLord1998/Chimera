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
    el.classList.remove("loaded");
    el.innerHTML = `<div class="placeholder-inner"><div class="placeholder-icon">${slot.icon}</div><span>${slot.label}</span></div>`;
  }
}

// ---------------------------------------------------------------------------
// Synthetic grid initialisation
// ---------------------------------------------------------------------------

function initSyntheticGrid(count) {
  const grid = document.getElementById("syntheticGrid");
  grid.innerHTML = "";
  for (let i = 0; i < count; i++) {
    const cell = document.createElement("div");
    cell.className = "synthetic-cell";
    cell.dataset.index = i;
    cell.dataset.idx = String(i + 1).padStart(2, "0");
    cell.style.setProperty("--i", i);
    grid.appendChild(cell);
  }
}

// ---------------------------------------------------------------------------
// Checkpoint rows
// ---------------------------------------------------------------------------

function addCheckpointRow(step, imageUrls) {
  const container = document.getElementById("checkpointContainer");

  const row = document.createElement("div");
  row.className = "checkpoint-row";
  row.style.animation = "slide-in 0.4s ease";

  const label = document.createElement("div");
  label.className = "checkpoint-row-label";
  label.textContent = `Step ${step.toLocaleString()}`;
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
    wrap.appendChild(img);
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
// SSE event handlers
// ---------------------------------------------------------------------------

function onStageEvent(data) {
  setStatus(data.status, "running");
  console.log("[stage]", data);
}

function onViewEvent(data, jobId) {
  const id = `view${capitalize(data.position)}`;
  const el = document.getElementById(id);
  if (!el) return;

  const img = document.createElement("img");
  img.src = data.url + "?t=" + Date.now();
  img.alt = data.position;
  img.loading = "lazy";

  el.innerHTML = "";
  el.appendChild(img);
  el.classList.add("loaded");
  activateSection("sectionViews");

  // Show download button once all 5 views are loaded
  const loaded = document.querySelectorAll(".view-placeholder.loaded").length;
  if (loaded >= 5 && jobId) {
    const dlBtn = document.getElementById("downloadViewsBtn");
    dlBtn.href = `/api/download-views/${jobId}`;
    dlBtn.hidden = false;
  }
}

function onSyntheticEvent(data, jobId) {
  const cell = document.querySelector(`.synthetic-cell[data-index="${data.index}"]`);
  if (!cell) return;

  const img = document.createElement("img");
  img.src = data.url + "?t=" + Date.now();
  img.alt = `synthetic ${data.index + 1}`;
  img.loading = "lazy";

  cell.innerHTML = "";
  cell.appendChild(img);
  cell.classList.add("loaded");

  // Update count
  const loaded = document.querySelectorAll(".synthetic-cell.loaded").length;
  const total = document.querySelectorAll(".synthetic-cell").length;
  document.getElementById("syntheticCount").textContent = `${loaded} / ${total}`;

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
  addCheckpointRow(data.step, data.images);
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
  const img = cell.querySelector("img:not(.diffusion-preview-img)");
  if (img) img.src = data.upscaled_url + "?t=" + Date.now();

  // Add compare badge
  if (!cell.querySelector(".compare-badge")) {
    const badge = document.createElement("div");
    badge.className = "compare-badge";
    badge.textContent = "2048px";
    cell.appendChild(badge);
  }
}

// Before/After comparison slider (opens on click)
function openComparison(originalUrl, upscaledUrl) {
  // Remove existing overlay
  const existing = document.getElementById("comparisonOverlay");
  if (existing) existing.remove();

  const overlay = document.createElement("div");
  overlay.id = "comparisonOverlay";
  overlay.className = "comparison-overlay";
  overlay.innerHTML = `
    <div class="comparison-container">
      <div class="comparison-close">&times;</div>
      <div class="comparison-labels">
        <span class="comparison-label-left">Original 1024px</span>
        <span class="comparison-label-right">SeedVR2 2048px</span>
      </div>
      <div class="comparison-wrapper">
        <img class="comparison-img comparison-img-upscaled" src="${upscaledUrl}" alt="upscaled" />
        <div class="comparison-clip">
          <img class="comparison-img comparison-img-original" src="${originalUrl}" alt="original" />
        </div>
        <div class="comparison-slider">
          <div class="comparison-handle"></div>
        </div>
      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  const wrapper = overlay.querySelector(".comparison-wrapper");
  const clip = overlay.querySelector(".comparison-clip");
  const slider = overlay.querySelector(".comparison-slider");

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
  wrapper.addEventListener("mousedown", () => { dragging = true; });
  window.addEventListener("mouseup", () => { dragging = false; });
  wrapper.addEventListener("mousemove", e => { if (dragging) updateSlider(e.clientX); });
  wrapper.addEventListener("click", e => updateSlider(e.clientX));

  // Touch support
  wrapper.addEventListener("touchstart", () => { dragging = true; });
  wrapper.addEventListener("touchend", () => { dragging = false; });
  wrapper.addEventListener("touchmove", e => {
    if (dragging) updateSlider(e.touches[0].clientX);
  });

  // Close
  overlay.querySelector(".comparison-close").addEventListener("click", () => overlay.remove());
  overlay.addEventListener("click", e => { if (e.target === overlay) overlay.remove(); });
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

  startBtn.disabled = false;
  evtSource.close();

  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

function onErrorEvent(data, evtSource, startBtn) {
  setStatus(`Error: ${data.message}`, "error");
  console.error("[pipeline error]", data.message);
  startBtn.disabled = false;
  evtSource.close();
}

// ---------------------------------------------------------------------------
// SSE connection (reusable — used by start button and auto-reconnect)
// ---------------------------------------------------------------------------

function connectToJob(jobId, startBtn) {
  startBtn.disabled = true;
  setStatus(`Running… job ${jobId}`, "running");

  const evtSource = new EventSource(`/api/stream/${jobId}`);

  evtSource.addEventListener("stage", e => {
    onStageEvent(JSON.parse(e.data));
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

  ["stage", "view", "synthetic", "diffusion_preview", "upscaled", "progress", "checkpoint", "complete", "heartbeat"].forEach(
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
    viewsZipName.textContent = "";
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
  });

  datasetZipClear.addEventListener("click", () => {
    datasetZipFile = null;
    datasetZipName.textContent = "";
    datasetZipClear.hidden = true;
    datasetZipInput.value = "";
  });

  // Init grid with default 25 placeholders
  initSyntheticGrid(25);

  // Click handler for before/after comparison on synthetic cells
  document.getElementById("syntheticGrid").addEventListener("click", e => {
    const cell = e.target.closest(".synthetic-cell.has-comparison");
    if (!cell) return;
    openComparison(cell.dataset.originalUrl, cell.dataset.upscaledUrl);
  });

  // Keep grid in sync with numImages setting
  document.getElementById("numImages").addEventListener("input", e => {
    const n = Math.max(10, Math.min(50, parseInt(e.target.value, 10) || 25));
    initSyntheticGrid(n);
  });

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
    if (!uploadedFile && !viewsZipFile && !datasetZipFile) {
      alert("Please upload a character image, views zip, or dataset zip.");
      return;
    }

    const geminiKey = document.getElementById("geminiKey").value.trim();
    if (!geminiKey && !viewsZipFile && !datasetZipFile) {
      alert("Please enter your Gemini API key, or upload a views/dataset zip to skip generation.");
      document.getElementById("geminiKey").focus();
      return;
    }

    // Reset UI state
    const numImages = parseInt(document.getElementById("numImages").value, 10) || 25;
    initSyntheticGrid(numImages);
    document.getElementById("syntheticCount").textContent = `0 / ${numImages}`;

    // Reset view placeholders
    resetViewPlaceholders();

    // Reset progress
    document.getElementById("progressFill").style.width = "0%";
    document.getElementById("progressGlow").style.left = "0%";
    const totalSteps = parseInt(loraStepsEl.value, 10) || 1500;
    document.getElementById("progressText").textContent = `0 / ${totalSteps.toLocaleString()}`;

    // Reset checkpoint container
    document.getElementById("checkpointContainer").innerHTML = "";

    // Hide output section and download buttons
    document.getElementById("outputSection").hidden = true;
    const dlBtn = document.getElementById("downloadViewsBtn");
    dlBtn.hidden = true;
    dlBtn.href = "#";
    const dlDatasetBtn = document.getElementById("downloadDatasetBtn");
    dlDatasetBtn.hidden = true;
    dlDatasetBtn.href = "#";

    // Deactivate pipeline sections
    ["sectionViews", "sectionSynthetic", "sectionTraining", "sectionCheckpoints"].forEach(id => {
      document.getElementById(id).classList.remove("active");
    });

    // Update synthesizer tag in UI
    const synthChoice = document.getElementById("synthesizer").value;
    const synthTag = document.getElementById("synthesizerTag");
    if (synthTag) {
      synthTag.textContent = synthChoice === "klein_kv" ? "Klein 9B" : "Flux 2";
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
    const samplePromptsRaw = document.getElementById("samplePrompts").value.trim();
    if (samplePromptsRaw) {
      formData.append("sample_prompts", samplePromptsRaw);
    }
    if (viewsZipFile) {
      formData.append("views_zip", viewsZipFile);
    }
    if (datasetZipFile) {
      formData.append("dataset_zip", datasetZipFile);
    }

    let jobId;
    try {
      const resp = await fetch("/api/start", { method: "POST", body: formData });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: resp.statusText }));
        throw new Error(err.error || "Failed to start pipeline");
      }
      const json = await resp.json();
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

        // Reconnect — the server replays all past events
        connectToJob(data.job_id, startBtn);
      }
    } catch (err) {
      console.log("[reconnect] No active job or server unreachable:", err.message);
    }
  })();
});
