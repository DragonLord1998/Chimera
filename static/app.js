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

function onViewEvent(data) {
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
}

function onSyntheticEvent(data) {
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
// Main
// ---------------------------------------------------------------------------

let uploadedFile = null;

document.addEventListener("DOMContentLoaded", () => {
  const uploadArea   = document.getElementById("uploadArea");
  const imageInput   = document.getElementById("imageInput");
  const startBtn     = document.getElementById("startBtn");
  const loraStepsEl  = document.getElementById("loraSteps");

  // Init grid with default 25 placeholders
  initSyntheticGrid(25);

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
    if (!uploadedFile) {
      alert("Please upload a character image first.");
      return;
    }

    const geminiKey = document.getElementById("geminiKey").value.trim();
    if (!geminiKey) {
      alert("Please enter your Gemini API key.");
      document.getElementById("geminiKey").focus();
      return;
    }

    // Reset UI state
    const numImages = parseInt(document.getElementById("numImages").value, 10) || 25;
    initSyntheticGrid(numImages);
    document.getElementById("syntheticCount").textContent = `0 / ${numImages}`;

    // Reset view placeholders
    ["viewLeft", "viewFront", "viewRight"].forEach(id => {
      const el = document.getElementById(id);
      const viewName = id.replace("view", "");
      const shortLabel = viewName === "Left" ? "L" : viewName === "Front" ? "F" : "R";
      const longLabel = viewName === "Left" ? "Left Side Fullbody"
                      : viewName === "Front" ? "Front Face"
                      : "Right Side Fullbody";
      el.classList.remove("loaded");
      el.innerHTML = `<div class="placeholder-inner">
        <div class="placeholder-icon">${shortLabel}</div>
        <span>${longLabel}</span>
      </div>`;
    });

    // Reset progress
    document.getElementById("progressFill").style.width = "0%";
    document.getElementById("progressGlow").style.left = "0%";
    const totalSteps = parseInt(loraStepsEl.value, 10) || 1500;
    document.getElementById("progressText").textContent = `0 / ${totalSteps.toLocaleString()}`;

    // Reset checkpoint container
    document.getElementById("checkpointContainer").innerHTML = "";

    // Hide output section
    document.getElementById("outputSection").hidden = true;

    // Deactivate pipeline sections
    ["sectionViews", "sectionSynthetic", "sectionTraining", "sectionCheckpoints"].forEach(id => {
      document.getElementById(id).classList.remove("active");
    });

    startBtn.disabled = true;
    setStatus("Starting pipeline...", "running");

    // Build form data
    const formData = new FormData();
    formData.append("image", uploadedFile);
    formData.append("trigger_word", document.getElementById("triggerWord").value.trim() || "chrx");
    formData.append("gemini_key", geminiKey);
    formData.append("hf_token", document.getElementById("hfToken").value.trim());
    formData.append("num_images", numImages);
    formData.append("lora_rank", document.getElementById("loraRank").value);
    formData.append("lora_steps", totalSteps);
    formData.append("learning_rate", document.getElementById("learningRate").value);
    const samplePromptsRaw = document.getElementById("samplePrompts").value.trim();
    if (samplePromptsRaw) {
      formData.append("sample_prompts", samplePromptsRaw);
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

    setStatus(`Running… job ${jobId}`, "running");

    // ---------------------------------------------------------------------------
    // SSE connection
    // ---------------------------------------------------------------------------

    const evtSource = new EventSource(`/api/stream/${jobId}`);

    evtSource.addEventListener("stage", e => {
      onStageEvent(JSON.parse(e.data));
    });

    evtSource.addEventListener("view", e => {
      onViewEvent(JSON.parse(e.data));
    });

    evtSource.addEventListener("synthetic", e => {
      onSyntheticEvent(JSON.parse(e.data));
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
      // The native EventSource 'error' event fires on connection drop too
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

    // Reset timer on any message
    ["stage", "view", "synthetic", "progress", "checkpoint", "complete", "heartbeat"].forEach(
      name => evtSource.addEventListener(name, () => { lastActivity = Date.now(); })
    );

    // Clear guard when done
    evtSource.addEventListener("complete", () => clearInterval(activityGuard));
    evtSource.addEventListener("error",    () => clearInterval(activityGuard));
  });
});
