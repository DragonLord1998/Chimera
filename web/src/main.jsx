import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Check,
  CheckCircle2,
  Cpu,
  Database,
  FileText,
  FolderInput,
  Gauge,
  ImagePlus,
  Layers,
  Library,
  Loader2,
  Play,
  RefreshCcw,
  Save,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  SquareTerminal,
  UploadCloud,
  Wand2,
  Zap,
} from "lucide-react";
import "./styles.css";

const API = "";

const phases = [
  { id: "ingest", label: "Ingest", eyebrow: "Phase 1" },
  { id: "pulid-dataset", label: "Flux2-PuLID Dataset", eyebrow: "Phase 2" },
  { id: "strict-qc", label: "Strict QC", eyebrow: "Phase 3" },
  { id: "identity-lora", label: "Identity LoRA", eyebrow: "Phase 4" },
  { id: "zimage-expansion", label: "Z-Image Expansion", eyebrow: "Phase 5" },
  { id: "final-qc", label: "Very Strict QC", eyebrow: "Phase 6" },
  { id: "factory", label: "Other Model LoRAs", eyebrow: "Phase 7" },
];

const defaultGenerationCommand = `# The backend sets REF_DIR, CANDIDATE_DIR, CASE_DIR, COUNT, and TRIGGER.
python - <<'PYGEN'
import glob, os, shutil
src = "/content/drive/MyDrive/GenAI/ComfyUI/output"
out = os.environ["CANDIDATE_DIR"]
os.makedirs(out, exist_ok=True)
for i, path in enumerate(glob.glob(src + "/*.png")[: int(os.environ.get("COUNT", "200"))]):
    shutil.copy2(path, os.path.join(out, f"comfy_{i+1:04d}.png"))
print("copied ComfyUI outputs")
PYGEN`;

const defaultSamplePrompts = `[trigger] person, studio portrait, plain white background
[trigger] person, side profile photo, natural lighting
[trigger] person, full body photo, wearing a black jacket
[trigger] person, cinematic close-up, night street
[trigger] person, casual selfie, indoor lighting
[trigger] person, half body photo, outdoor background
[trigger] person, fashion editorial portrait
[trigger] person, close-up portrait, soft daylight`;

const fixedIdentityThreshold = 0.92;

const modelPresets = [
  {
    label: "FLUX",
    status: "Actionable now",
    rank: 32,
    steps: 1200,
    lr: "1e-4",
    caption: "Trigger first, class noun second, then framing, pose, lighting, clothing, and background.",
  },
  {
    label: "Flux2-PuLID",
    status: "Generator path",
    rank: 32,
    steps: 900,
    lr: "8e-5",
    caption: "Synthetic prompt buckets provide caption tags; identity remains enforced by QC, not facial descriptions.",
  },
  {
    label: "Wan",
    status: "Preset only",
    rank: 32,
    steps: 1600,
    lr: "8e-5",
    caption: "Caption subject, shot type, motion/pose, lighting, and camera distance; keep trigger token stable across frames.",
  },
  {
    label: "LTX",
    status: "Preset only",
    rank: 16,
    steps: 1400,
    lr: "8e-5",
    caption: "Compact trigger-led captions for shot type, motion, and scene; avoid over-describing facial identity.",
  },
];

const defaultTraining = {
  trigger: "zphchar",
  base_caption: "",
  model_name: "black-forest-labs/FLUX.2-klein-base-9B",
  rank: 32,
  steps: 1200,
  lr: "1e-4",
  sample_every: 200,
  save_every: 250,
};

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    headers: options.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...options,
  });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    throw new Error(payload?.detail || payload || `Request failed: ${response.status}`);
  }
  return payload;
}

function cx(...parts) {
  return parts.filter(Boolean).join(" ");
}

function fmt(value, fallback = "n/a") {
  return value === undefined || value === null || value === "" ? fallback : value;
}

function clampPercent(value) {
  const parsed = Number(value || 0);
  if (!Number.isFinite(parsed)) return 0;
  return Math.max(0, Math.min(100, parsed));
}

function scoreToPercent(score) {
  const parsed = Number(score);
  if (!Number.isFinite(parsed)) return 0;
  return clampPercent(parsed <= 1 ? parsed * 100 : parsed);
}

function normalizeBool(value) {
  return value === true || value === "True" || value === "true" || value === 1 || value === "1";
}

function preflightItemOk(preflight, key) {
  return Boolean((preflight?.full_items || preflight?.items || []).find((item) => item.key === key && item.ok));
}

function hashUtilityView() {
  const hash = window.location.hash.replace("#", "");
  return ["hardware", "datasets", "logs", "terminal"].includes(hash) ? hash : "";
}

function AppShell({
  activePhase,
  setActivePhase,
  utilityView,
  setUtilityView,
  cases,
  activeCase,
  setActiveCase,
  caseLabel,
  setCaseLabel,
  onNewCase,
  workRoot,
  busy,
  status,
  children,
}) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand-mark" onClick={() => setActivePhase("ingest")}>
          Chimera
        </button>
        <PhaseNav activePhase={activePhase} setActivePhase={setActivePhase} />
        <div className="top-actions">
          <button className="search-pill" type="button" onClick={() => setUtilityView("datasets")}>
            <Database size={16} />
            <span>Dataset snapshot</span>
          </button>
          <button className="icon-button" aria-label="Open logs" onClick={() => setUtilityView("logs")}>
            <FileText size={20} />
          </button>
          <button className="icon-button" aria-label="Runtime access" onClick={() => setUtilityView("terminal")}>
            <SquareTerminal size={20} />
          </button>
        </div>
      </header>

      <aside className="side-rail">
        <div className="session-card">
          <span className="session-icon">
            <Cpu size={20} />
          </span>
          <div>
            <strong>Project Chimera</strong>
            <span>Active Session</span>
          </div>
        </div>
        <button className="primary rail-action" onClick={onNewCase} disabled={busy}>
          <ImagePlus size={16} /> New Training Run
        </button>
        <nav className="rail-nav">
          <button
            type="button"
            className={cx("rail-link", utilityView === "hardware" && "active")}
            onClick={() => setUtilityView("hardware")}
          >
            <Cpu size={18} /> Hardware
          </button>
          <button
            type="button"
            className={cx("rail-link", utilityView === "datasets" && "active")}
            onClick={() => setUtilityView("datasets")}
          >
            <Database size={18} /> Datasets
          </button>
          <button
            type="button"
            className={cx("rail-link", utilityView === "logs" && "active")}
            onClick={() => setUtilityView("logs")}
          >
            <FileText size={18} /> Logs
          </button>
          <button
            type="button"
            className={cx("rail-link", utilityView === "terminal" && "active")}
            onClick={() => setUtilityView("terminal")}
          >
            <SquareTerminal size={18} /> Runtime Access
          </button>
        </nav>
      </aside>

      <main className="workspace">
        <div className="case-bar">
          <div>
            <span className="micro-label">Workspace</span>
            <strong>{workRoot || "Preparing workspace"}</strong>
          </div>
          <div className="case-controls">
            <label>
              Case
              <select
                aria-label="Case"
                value={activeCase}
                onChange={(event) => setActiveCase(event.target.value)}
              >
                {cases.map((name) => <option key={name}>{name}</option>)}
              </select>
            </label>
            <label>
              New label
              <input value={caseLabel} onChange={(event) => setCaseLabel(event.target.value)} />
            </label>
            <button onClick={onNewCase} disabled={busy}>
              <ImagePlus size={16} /> New Case
            </button>
          </div>
        </div>

        {status && (
          <div className="status-banner">
            {busy && <Loader2 className="spin" size={16} />}
            <span>{status}</span>
          </div>
        )}

        {children}
      </main>
    </div>
  );
}

function PhaseNav({ activePhase, setActivePhase }) {
  return (
    <nav className="phase-nav" aria-label="Project phase navigation">
      {phases.map((phase) => (
        <button
          key={phase.id}
          className={activePhase === phase.id ? "active" : ""}
          onClick={() => setActivePhase(phase.id)}
        >
          {phase.label}
        </button>
      ))}
    </nav>
  );
}

function PageHeader({ eyebrow, title, description, actions }) {
  return (
    <header className="phase-header">
      <div>
        <span className="phase-chip">{eyebrow}</span>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions && <div className="phase-actions">{actions}</div>}
    </header>
  );
}

function SurfaceCard({ title, icon: Icon, children, className, action }) {
  return (
    <section className={cx("surface-card", className)}>
      {(title || Icon || action) && (
        <div className="card-heading">
          <div>
            {Icon && <Icon size={20} />}
            {title && <h2>{title}</h2>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

function StatPill({ label, value, tone = "neutral" }) {
  return (
    <div className={cx("stat-pill", tone)}>
      <span>{label}</span>
      <strong>{fmt(value, "0")}</strong>
    </div>
  );
}

function MetricTile({ icon: Icon, label, value }) {
  return (
    <div className="metric-tile">
      <Icon size={18} />
      <span>{label}</span>
      <strong>{fmt(value)}</strong>
    </div>
  );
}

function ReadinessList({ preflight }) {
  const items = preflight?.items || [];
  return (
    <div className="readiness-list">
      {items.map((item) => (
        <div key={item.key} className={cx("readiness-item", item.ok ? "ok" : "blocked")}>
          {item.ok ? <CheckCircle2 size={16} /> : <Gauge size={16} />}
          <span>
            <strong>{item.label}</strong>
            <small>{item.detail || "n/a"}</small>
          </span>
          <b>{item.ok ? "Ready" : "Needs setup"}</b>
        </div>
      ))}
      {items.length === 0 && <div className="empty-state">Runtime checks have not loaded yet</div>}
    </div>
  );
}

function ImageGallery({ title, items = [], qcByPath, selected = false, empty = "No images yet", compact = false }) {
  return (
    <SurfaceCard title={title} icon={Library} className={compact ? "compact-card" : ""}>
      <div className={cx("gallery-grid", compact && "compact")}>
        {items.map((item) => {
          const qc = qcByPath?.get(item.path);
          const passed = normalizeBool(qc?.passed);
          const score = qc?.identity_score;
          const hasScore = score !== undefined && score !== null && score !== "";
          return (
            <article key={item.path} className={cx("image-card", (selected || item.selected || passed) && "accepted", hasScore && !passed && "rejected")}>
              <img src={`${item.url}&t=${Date.now()}`} alt={item.name} loading="lazy" />
              <div className="image-card-meta">
                <span>{item.name}</span>
                {hasScore && (
                  <b className={passed ? "score-pass" : "score-fail"}>
                    {Number(score).toFixed(2)}
                  </b>
                )}
              </div>
              {hasScore && <div className="mini-bar"><i style={{ width: `${scoreToPercent(score)}%` }} /></div>}
            </article>
          );
        })}
        {items.length === 0 && <div className="empty-state">{empty}</div>}
      </div>
    </SurfaceCard>
  );
}

function PromptCards({ promptsText, dashboard, steps, sampleEvery }) {
  const prompts = promptsText
    .split("\n")
    .map((line) => line.trim().replaceAll("[trigger]", "zphchar"))
    .filter(Boolean)
    .slice(0, 8);
  const expected = dashboard?.expected_sample_steps?.length
    ? dashboard.expected_sample_steps
    : Array.from({ length: Math.floor(Number(steps || 0) / Number(sampleEvery || 1)) }, (_, idx) => (idx + 1) * Number(sampleEvery || 1));
  const actual = new Set(dashboard?.artifacts?.sample_steps || []);

  return (
    <div className="prompt-grid">
      {prompts.map((prompt) => (
        <div className="prompt-card" key={prompt}>
          <span>{prompt}</span>
          <div className="sample-dots">
            {expected.slice(0, 12).map((step) => (
              <i key={step} className={actual.has(step) ? "done" : ""} title={`sample step ${step}`} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function TrainingOverview({ dashboard, training, samplePrompts, configPath, onPrepare, onStart, onRefresh, busy, title = "Training Progress", startLabel = "Start Training", startDisabled = false, notice = "" }) {
  const current = dashboard?.current_step;
  const total = dashboard?.total_steps || training.steps;
  const percent = clampPercent(dashboard?.percent || 0);
  const stats = dashboard?.stats || {};
  const trainingState = dashboard?.training_state || {};
  const stepText = current == null ? "waiting for ai-toolkit step report" : `${current}/${total} steps`;

  return (
    <SurfaceCard title={title} icon={Activity} className="training-card">
      {notice && <div className="readiness-banner blocked">{notice}</div>}
      <div className="progress-summary">
        <div>
          <span>{stepText}</span>
          <strong>{percent}%</strong>
        </div>
        <span className={dashboard?.running ? "state-chip live" : "state-chip"}>{dashboard?.running ? "Running" : "Idle"}</span>
      </div>
      <div className="progress-track"><i style={{ width: `${percent}%` }} /></div>
      <code className="step-line">{trainingState.last_step_line || "No step line reported yet"}</code>

      <div className="telemetry-grid">
        <MetricTile icon={Cpu} label="CPU" value={stats.pid_cpu} />
        <MetricTile icon={Database} label="RAM" value={stats.pid_rss} />
        <MetricTile icon={Activity} label="GPU" value={stats.gpu} />
        <MetricTile icon={Gauge} label="VRAM" value={stats.vram_used} />
      </div>

      <div className="artifact-strip">
        <span>Samples every {training.sample_every} steps</span>
        <span>{dashboard?.artifacts?.sample_files?.length || 0} sample files</span>
        <span>Checkpoints every {training.save_every} steps</span>
        <span>{dashboard?.artifacts?.checkpoint_files?.length || 0} checkpoint files</span>
      </div>

      <div className="sample-preview-card">
        {dashboard?.latest_sample ? (
          <img src={`${dashboard.latest_sample}&t=${Date.now()}`} alt="Latest training sample" />
        ) : (
          <div>
            <Sparkles size={24} />
            <span>Latest sample will appear here</span>
          </div>
        )}
      </div>

      <PromptCards promptsText={samplePrompts} dashboard={dashboard} steps={training.steps} sampleEvery={training.sample_every} />

      <div className="action-row">
        <button onClick={onPrepare} disabled={busy}>
          <Save size={16} /> Caption + Config
        </button>
        <button onClick={onStart} disabled={busy || !configPath || startDisabled} className="primary">
          <Play size={16} /> {startLabel}
        </button>
        <button onClick={onRefresh} disabled={busy}>
          <RefreshCcw size={16} /> Refresh
        </button>
      </div>
    </SurfaceCard>
  );
}

function IngestPhase({ state, preflight, files, setFiles, consent, setConsent, busy, onRunPipeline, onRefreshPreflight, caseLabel, setCaseLabel, count, setCount, qc, setQc, training, setTraining }) {
  return (
    <>
      <PageHeader
        eyebrow="Phase 1: Ingest"
        title="Upload One Photo"
        description="Review the run settings, upload one clear reference, then Chimera uses the configured Flux2-PuLID generator to build the seed dataset before strict QC."
      />
      <section className="ingest-layout">
        <div className="main-stack">
          <SurfaceCard className="upload-hero">
            <label className="drop-zone">
              <input type="file" multiple accept="image/*" onChange={(event) => setFiles([...event.target.files])} />
              <span className="upload-orb"><UploadCloud size={36} /></span>
              <strong>Drag and drop photos here</strong>
              <small>or click to browse local files</small>
              <em>JPG</em><em>PNG</em><em>WEBP</em>
            </label>
            <div className="upload-footer">
              <label className="check-line">
                <input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} />
                I have consent and rights to train this identity
              </label>
              <button className="primary" disabled={busy || !files.length || !consent || preflight?.ready !== true} onClick={onRunPipeline}>
                <UploadCloud size={16} /> Run Seed Pipeline
              </button>
            </div>
          </SurfaceCard>

          {files.length > 0 && (
            <SurfaceCard title="Pending Input" icon={ImagePlus}>
              <div className="pending-file-list">
                {files.map((file) => (
                  <div key={`${file.name}-${file.size}`}>
                    <span>{file.name}</span>
                    <b>{(file.size / 1024 / 1024).toFixed(2)} MB</b>
                  </div>
                ))}
              </div>
            </SurfaceCard>
          )}

          <ImageGallery title="Reference Anchors" items={state?.refs || []} empty="Saved references will appear here" />

          <SurfaceCard title="Character LoRA Captioning" icon={FileText}>
            <label>Constant suffix <input placeholder="Optional, usually empty" value={training.base_caption} onChange={(event) => setTraining({ ...training, base_caption: event.target.value })} /></label>
            <div className="preset-list">
              {modelPresets.map((preset) => (
                <article key={preset.label}>
                  <strong>{preset.label}</strong>
                  <span>{preset.status}</span>
                  <p>{preset.caption}</p>
                </article>
              ))}
            </div>
          </SurfaceCard>
        </div>

        <div className="side-stack">
          <SurfaceCard
            title="Runtime Readiness"
            icon={ShieldCheck}
            action={<button className="ghost-button" type="button" onClick={onRefreshPreflight} disabled={busy}><RefreshCcw size={14} /> Check</button>}
          >
            <div className={cx("readiness-banner", preflight?.ready ? "ready" : "blocked")}>
              {preflight?.ready ? "Ready for one-click run" : "Setup required before one-click run"}
            </div>
            <ReadinessList preflight={preflight} />
          </SurfaceCard>
          <SurfaceCard title="Run Settings" icon={SlidersHorizontal}>
            <div className="grid-form compact-form">
              <label className="wide">Generator <input value="Flux2-PuLID" readOnly /></label>
              <label>Candidate count <input type="number" min="1" value={count} onChange={(event) => setCount(Number(event.target.value))} /></label>
              <label>Keep after QC <input type="number" min="1" value={qc.top_n} onChange={(event) => setQc({ ...qc, top_n: Number(event.target.value) })} /></label>
              <label>Trigger <input value={training.trigger} onChange={(event) => setTraining({ ...training, trigger: event.target.value })} /></label>
              <label>Rank <input type="number" min="1" value={training.rank} onChange={(event) => setTraining({ ...training, rank: Number(event.target.value) })} /></label>
              <label>Steps <input type="number" min="1" value={training.steps} onChange={(event) => setTraining({ ...training, steps: Number(event.target.value) })} /></label>
              <label>Learning rate <input value={training.lr} onChange={(event) => setTraining({ ...training, lr: event.target.value })} /></label>
              <label>Sample every <input type="number" min="1" value={training.sample_every} onChange={(event) => setTraining({ ...training, sample_every: Number(event.target.value) })} /></label>
              <label>Checkpoint every <input type="number" min="1" value={training.save_every} onChange={(event) => setTraining({ ...training, save_every: Number(event.target.value) })} /></label>
            </div>
            <div className="fixed-qc-note">
              <ShieldCheck size={16} />
              Identity QC is fixed at {Math.round(fixedIdentityThreshold * 100)}% similarity.
            </div>
          </SurfaceCard>
          <SurfaceCard title="Quality Requirements" icon={ShieldCheck}>
            <ul className="quality-list">
              <li><CheckCircle2 size={16} /> Clear face, no heavy occlusion</li>
              <li><CheckCircle2 size={16} /> One primary subject per image</li>
              <li><CheckCircle2 size={16} /> Mix of angles if uploading more than one</li>
              <li><CheckCircle2 size={16} /> Original refs stay immutable for QC</li>
            </ul>
          </SurfaceCard>
          <SurfaceCard title="Case Setup" icon={Database}>
            <label>
              Case label
              <input value={caseLabel} onChange={(event) => setCaseLabel(event.target.value)} />
            </label>
            <div className="stat-row">
              <StatPill label="Refs" value={state?.refs?.length || 0} />
              <StatPill label="Candidates" value={state?.candidates?.length || 0} />
            </div>
          </SurfaceCard>
        </div>
      </section>
    </>
  );
}

function SeedPhase({ state, dashboard, preflight, training, setTraining, samplePrompts, setSamplePrompts, configPath, busy, onPrepare, onStart, onRefresh }) {
  const zimageReady = preflightItemOk(preflight, "zimage_identity_train_command");
  const canStartIdentity = Boolean(configPath && state?.train?.length && zimageReady);
  return (
    <>
      <PageHeader
        eyebrow="Phase 4: Identity LoRA"
        title="Train Identity LoRA For Z-Image Turbo"
        description="This stage consumes the strict-QC curated Flux2-PuLID seed dataset. Chimera exposes a command bridge here; Colab must provide the real Z-Image Turbo trainer command."
        actions={<span className="state-chip">{configPath ? "Config ready" : "Config pending"}</span>}
      />
      <section className="bento two-col">
        <TrainingOverview
          dashboard={dashboard}
          training={training}
          samplePrompts={samplePrompts}
          configPath={configPath}
          busy={busy}
          onPrepare={onPrepare}
          onStart={onStart}
          onRefresh={onRefresh}
          title="Identity LoRA Progress"
          startLabel="Start Z-Image Turbo LoRA"
          startDisabled={!canStartIdentity}
          notice={zimageReady ? "This command-backed stage writes the identity LoRA artifact into the case identity_lora folder." : "Set ZIMAGE_IDENTITY_TRAIN_COMMAND in Colab before starting this stage."}
        />
        <div className="side-stack">
          <SurfaceCard title="Current Config Parameters" icon={SlidersHorizontal}>
            <div className="grid-form">
              <label>Trigger <input value={training.trigger} onChange={(event) => setTraining({ ...training, trigger: event.target.value })} /></label>
              <label>Current backend model <input value={training.model_name} onChange={(event) => setTraining({ ...training, model_name: event.target.value })} /></label>
              <label>Rank <input type="number" value={training.rank} onChange={(event) => setTraining({ ...training, rank: Number(event.target.value) })} /></label>
              <label>Steps <input type="number" value={training.steps} onChange={(event) => setTraining({ ...training, steps: Number(event.target.value) })} /></label>
              <label>Learning rate <input value={training.lr} onChange={(event) => setTraining({ ...training, lr: event.target.value })} /></label>
              <label>Sample every <input type="number" value={training.sample_every} onChange={(event) => setTraining({ ...training, sample_every: Number(event.target.value) })} /></label>
              <label>Checkpoint every <input type="number" value={training.save_every} onChange={(event) => setTraining({ ...training, save_every: Number(event.target.value) })} /></label>
            </div>
          </SurfaceCard>
          <SurfaceCard title="Sample Prompts" icon={Sparkles}>
            <textarea value={samplePrompts} onChange={(event) => setSamplePrompts(event.target.value)} rows={9} />
          </SurfaceCard>
          <SurfaceCard title="Seed Inputs" icon={ImagePlus}>
            <div className="stat-row">
              <StatPill label="Refs" value={state?.refs?.length || 0} />
              <StatPill label="Curated" value={state?.train?.length || 0} />
              <StatPill label="Identity LoRA" value={state?.identity_lora_artifacts?.length || 0} />
            </div>
          </SurfaceCard>
        </div>
      </section>
    </>
  );
}

function ExpansionPhase({ state, count, setCount, importFolder, setImportFolder, generationCommand, setGenerationCommand, busy, onSmoke, onImport, onStartCommand, dashboard }) {
  const stats = dashboard?.stats || {};
  return (
    <>
      <PageHeader
        eyebrow="Phase 2: Flux2-PuLID Dataset"
        title="Generate The Seed Candidate Dataset"
        description="Production generation uses Flux2-PuLID from the original references. This creates the first identity-preserving candidate pool before any LoRA is trained."
      />
      <section className="bento two-col">
        <div className="main-stack">
          <SurfaceCard title="Expansion Controls" icon={Wand2}>
            <div className="control-grid">
              <label>Target candidates <input type="number" value={count} onChange={(event) => setCount(Number(event.target.value))} /></label>
              <button disabled={busy} onClick={onSmoke}><Sparkles size={16} /> Dev Smoke Generate</button>
              <label className="wide">Import folder <input value={importFolder} onChange={(event) => setImportFolder(event.target.value)} /></label>
              <button disabled={busy} onClick={onImport}><FolderInput size={16} /> Import Images</button>
            </div>
          </SurfaceCard>
          <SurfaceCard title="Generation Command" icon={SquareTerminal}>
            <textarea value={generationCommand} onChange={(event) => setGenerationCommand(event.target.value)} rows={12} />
            <div className="action-row">
              <button className="primary" disabled={busy} onClick={onStartCommand}><Play size={16} /> Start Command</button>
            </div>
          </SurfaceCard>
          <ImageGallery title="Synthetic Candidates" items={state?.candidates || []} empty="Generated or imported candidates will appear here" />
        </div>
        <div className="side-stack">
          <SurfaceCard title="Seed Dataset Summary" icon={Database}>
            <div className="stat-row vertical">
              <StatPill label="Candidates" value={state?.candidates?.length || 0} />
              <StatPill label="Accepted" value={state?.candidates?.filter((item) => item.selected).length || 0} tone="good" />
              <StatPill label="Target" value={count || 0} />
            </div>
          </SurfaceCard>
          <SurfaceCard title="Node Telemetry" icon={Activity}>
            <div className="telemetry-grid single">
              <MetricTile icon={Activity} label="GPU" value={stats.gpu} />
              <MetricTile icon={Gauge} label="VRAM" value={stats.vram_used} />
              <MetricTile icon={Zap} label="Power" value={stats.power} />
            </div>
          </SurfaceCard>
          <SurfaceCard title="Generation Log" icon={FileText}>
            <pre>{state?.logs?.generate || "No generation log yet"}</pre>
          </SurfaceCard>
        </div>
      </section>
    </>
  );
}

function QcPhase({ state, qc, setQc, qcSummary, qcByPath, busy, onScoreSelect, dashboard }) {
  const stats = dashboard?.stats || {};
  return (
    <>
      <PageHeader
        eyebrow="Phase 3: Strict QC"
        title="Strict QC Against Original References"
        description="Filter the Flux2-PuLID seed dataset against the original reference anchors before training the identity LoRA."
        actions={<button disabled={busy} onClick={onScoreSelect}><SlidersHorizontal size={16} /> Score + Select</button>}
      />
      <section className="qc-layout">
        <div className="main-stack">
          <ImageGallery title="Audit Grid" items={state?.candidates || []} qcByPath={qcByPath} empty="Run QC after candidates are available" />
        </div>
        <div className="side-stack">
          <SurfaceCard title="QC Summary" icon={ShieldCheck}>
            <div className="summary-list">
              <span>Total Evaluated <b>{qcSummary.total}</b></span>
              <span><i className="dot good" /> Passed <b>{qcSummary.passed}</b></span>
              <span><i className="dot bad" /> Failed <b>{qcSummary.failed}</b></span>
              <span><i className="dot pending" /> Pending Review <b>{qcSummary.pending}</b></span>
            </div>
          </SurfaceCard>
          <SurfaceCard title="ArcFace Similarity" icon={Gauge}>
            <div className="fixed-qc-note">
              <ShieldCheck size={16} />
              Fixed at {Math.round(fixedIdentityThreshold * 100)}% similarity.
            </div>
            <label>Min face <input type="number" step="0.001" value={qc.min_face_area} onChange={(event) => setQc({ ...qc, min_face_area: Number(event.target.value) })} /></label>
            <label>Keep <input type="number" value={qc.top_n} onChange={(event) => setQc({ ...qc, top_n: Number(event.target.value) })} /></label>
            <button className="primary full" disabled={busy} onClick={onScoreSelect}><Check size={16} /> Finalize Dataset</button>
          </SurfaceCard>
          <SurfaceCard title="Node Telemetry" icon={Cpu}>
            <div className="telemetry-grid single">
              <MetricTile icon={Activity} label="GPU" value={stats.gpu} />
              <MetricTile icon={Gauge} label="VRAM" value={stats.vram_used} />
            </div>
          </SurfaceCard>
        </div>
      </section>
    </>
  );
}

function ZImageExpansionPhase({ state, dashboard, preflight, zimageCount, setZimageCount, busy, onRun }) {
  const stats = dashboard?.stats || {};
  const zimageReady = preflightItemOk(preflight, "zimage_expansion_command");
  const hasIdentityLora = Boolean(state?.identity_lora_artifacts?.length);
  return (
    <>
      <PageHeader
        eyebrow="Phase 5: Z-Image Expansion"
        title="Massive Dataset From The Identity LoRA"
        description="After the Z-Image Turbo identity LoRA exists, this stage expands from the small curated seed set into the large production candidate dataset."
        actions={<button disabled={busy || !zimageReady || !hasIdentityLora} onClick={onRun}><Wand2 size={16} /> Run Expansion</button>}
      />
      <section className="bento two-col">
        <div className="main-stack">
          <SurfaceCard title="Expansion Source" icon={Layers}>
            <div className="summary-list">
              <span>Original references <b>{state?.refs?.length || 0}</b></span>
              <span>Seed candidates <b>{state?.candidates?.length || 0}</b></span>
              <span>Strict-QC curated seed images <b>{state?.train?.length || 0}</b></span>
              <span>Required identity LoRA <b>{state?.identity_lora_artifacts?.length ? "artifact available" : "n/a"}</b></span>
            </div>
          </SurfaceCard>
          <ImageGallery title="Z-Image Production Candidates" items={state?.production_candidates || []} empty="Z-Image production candidates will appear here after expansion" compact />
        </div>
        <div className="side-stack">
          <SurfaceCard title="Readiness" icon={ShieldCheck}>
            <div className="summary-list">
              <span>Identity LoRA config <b>{state?.config_path ? "ready" : "n/a"}</b></span>
              <span>Training state <b>{dashboard?.training_state?.status || "n/a"}</b></span>
              <span>Z-Image command <b>{zimageReady ? "ready" : "n/a"}</b></span>
              <span>Identity LoRA artifact <b>{hasIdentityLora ? "ready" : "n/a"}</b></span>
              <span>Actual generated images <b>{state?.production_candidates?.length || 0}</b></span>
            </div>
            <label>Target raw images <input type="number" value={zimageCount} onChange={(event) => setZimageCount(Number(event.target.value))} /></label>
          </SurfaceCard>
          <SurfaceCard title="Node Telemetry" icon={Activity}>
            <div className="telemetry-grid single">
              <MetricTile icon={Activity} label="GPU" value={stats.gpu} />
              <MetricTile icon={Gauge} label="VRAM" value={stats.vram_used} />
              <MetricTile icon={Zap} label="Power" value={stats.power} />
            </div>
          </SurfaceCard>
        </div>
      </section>
    </>
  );
}

function FinalQcPhase({ state, qcSummary, dashboard, busy, onScoreSelect }) {
  const stats = dashboard?.stats || {};
  return (
    <>
      <PageHeader
        eyebrow="Phase 6: Very Strict QC"
        title="Audit The Z-Image Production Dataset"
        description="This second-pass QC will run after Z-Image Turbo expands the identity LoRA into the large production pool. It is separate from the first strict QC gate."
        actions={<button disabled={busy || !(state?.production_candidates?.length)} onClick={onScoreSelect}><ShieldCheck size={16} /> Score Final Dataset</button>}
      />
      <section className="qc-layout">
        <div className="main-stack">
          <ImageGallery title="Z-Image Production Candidates" items={state?.production_candidates || []} empty="No Z-Image production candidates exist yet" />
          <ImageGallery title="Final Curated Dataset" items={state?.final || []} empty="Final selected images will appear after very strict QC" compact />
        </div>
        <div className="side-stack">
          <SurfaceCard title="Existing Seed QC" icon={Database}>
            <div className="summary-list">
              <span>Seed evaluated <b>{qcSummary.total}</b></span>
              <span>Seed passed <b>{qcSummary.passed}</b></span>
              <span>Seed failed <b>{qcSummary.failed}</b></span>
              <span>Curated seed images <b>{state?.train?.length || 0}</b></span>
            </div>
          </SurfaceCard>
          <SurfaceCard title="Final QC Inputs" icon={Gauge}>
            <div className="summary-list">
              <span>Z-Image raw images <b>{state?.production_candidates?.length || 0}</b></span>
              <span>Duplicate pass <b>n/a</b></span>
              <span>Manual review <b>n/a</b></span>
              <span>Final dataset <b>{state?.final?.length || 0}</b></span>
            </div>
          </SurfaceCard>
          <SurfaceCard title="Node Telemetry" icon={Cpu}>
            <div className="telemetry-grid single">
              <MetricTile icon={Activity} label="GPU" value={stats.gpu} />
              <MetricTile icon={Gauge} label="VRAM" value={stats.vram_used} />
            </div>
          </SurfaceCard>
        </div>
      </section>
    </>
  );
}

function LibraryPhase({ state, training, setTraining, samplePrompts, setSamplePrompts, busy, onPrepare }) {
  return (
    <>
      <PageHeader
        eyebrow="Dataset Utility"
        title="Curated Seed Character Dataset"
        description="The accepted Flux2-PuLID seed set is the input for the Z-Image Turbo identity LoRA stage."
        actions={<button className="primary" disabled={busy} onClick={onPrepare}><Save size={16} /> Caption + Config</button>}
      />
      <section className="bento two-col">
        <div className="main-stack">
          <ImageGallery title="Curated Training Set" items={state?.train || []} empty="Accepted training images will appear after QC selection" />
        </div>
        <div className="side-stack">
          <SurfaceCard title="Dataset Stats" icon={Database}>
            <div className="stat-row vertical">
              <StatPill label="Reference anchors" value={state?.refs?.length || 0} />
              <StatPill label="Candidate pool" value={state?.candidates?.length || 0} />
              <StatPill label="Curated images" value={state?.train?.length || 0} tone="good" />
            </div>
          </SurfaceCard>
          <SurfaceCard title="Caption Template" icon={FileText}>
            <label>Trigger <input value={training.trigger} onChange={(event) => setTraining({ ...training, trigger: event.target.value })} /></label>
            <label>Constant suffix <input placeholder="Optional, usually empty" value={training.base_caption} onChange={(event) => setTraining({ ...training, base_caption: event.target.value })} /></label>
          </SurfaceCard>
          <SurfaceCard title="Validation Prompts" icon={Sparkles}>
            <textarea value={samplePrompts} onChange={(event) => setSamplePrompts(event.target.value)} rows={8} />
          </SurfaceCard>
        </div>
      </section>
    </>
  );
}

function FactoryPhase({ state, dashboard, training, setTraining, configPath, samplePrompts, busy, onPrepare, onStart, onRefresh }) {
  const architectures = [
    { label: "FLUX", detail: "Current ai-toolkit config", active: true },
    { label: "Z-Image", detail: "Not wired yet" },
    { label: "Wan", detail: "Not wired yet" },
    { label: "LTX", detail: "Not wired yet" },
  ];
  return (
    <>
      <PageHeader
        eyebrow="Phase 7: Other Model LoRAs"
        title="LoRA Generation For Other Models"
        description="After the very-strict final dataset exists, this stage trains model-specific LoRAs. Only the current selected ai-toolkit config is actionable in this frontend pass."
      />
      <section className="bento two-col">
        <div className="main-stack">
          <SurfaceCard title="Target Architecture" icon={Layers}>
            <div className="architecture-grid">
              {architectures.map((item) => (
                <button key={item.label} className={item.active ? "architecture active" : "architecture"} disabled={!item.active}>
                  <strong>{item.label}</strong>
                  <span>{item.detail}</span>
                  {item.active && <CheckCircle2 size={16} />}
                </button>
              ))}
            </div>
          </SurfaceCard>
          <SurfaceCard title="Adapter Parameters" icon={SlidersHorizontal}>
            <div className="grid-form">
              <label>Network Rank <input type="number" value={training.rank} onChange={(event) => setTraining({ ...training, rank: Number(event.target.value) })} /></label>
              <label>Learning Rate <input value={training.lr} onChange={(event) => setTraining({ ...training, lr: event.target.value })} /></label>
              <label>Steps <input type="number" value={training.steps} onChange={(event) => setTraining({ ...training, steps: Number(event.target.value) })} /></label>
              <label>Base model <input value={training.model_name} onChange={(event) => setTraining({ ...training, model_name: event.target.value })} /></label>
            </div>
            <div className="action-row">
              <button onClick={onPrepare} disabled={busy}><Save size={16} /> Build Config</button>
              <button className="primary" onClick={onStart} disabled={busy || !configPath}><Play size={16} /> Start Current Target</button>
            </div>
          </SurfaceCard>
          <TrainingOverview
            dashboard={dashboard}
            training={training}
            samplePrompts={samplePrompts}
            configPath={configPath}
            busy={busy}
            onPrepare={onPrepare}
            onStart={onStart}
            onRefresh={onRefresh}
          />
        </div>
        <div className="side-stack">
          <SurfaceCard title="Telemetry" icon={Activity}>
            <div className="summary-list">
              <span>Run state <b>{dashboard?.running ? "Running" : "Idle"}</b></span>
              <span>Current step <b>{fmt(dashboard?.current_step, "0")}</b></span>
              <span>Final dataset images <b>{state?.final?.length || 0}</b></span>
            </div>
          </SurfaceCard>
          <SurfaceCard title="Training Log" icon={FileText}>
            <pre>{state?.logs?.train || "No training log yet"}</pre>
          </SurfaceCard>
        </div>
      </section>
    </>
  );
}

function UtilityPanel({ view, state, dashboard, qcSummary, configPath, training, onClose, onRefresh, onOpenLibrary }) {
  const stats = dashboard?.stats || {};
  const trainingState = dashboard?.training_state || {};
  const titles = {
    hardware: {
      eyebrow: "Session Utility",
      title: "Hardware Telemetry",
      description: "Live runtime stats from the current FastAPI dashboard payload. Missing Colab or trainer values stay as n/a.",
    },
    datasets: {
      eyebrow: "Session Utility",
      title: "Dataset Snapshot",
      description: "Current case inventory from refs, candidates, QC rows, and curated training files.",
    },
    logs: {
      eyebrow: "Session Utility",
      title: "Runtime Logs",
      description: "Generation and training logs saved by the existing backend actions.",
    },
    terminal: {
      eyebrow: "Session Utility",
      title: "Runtime Access",
      description: "Current runtime access paths and the proxy-only browser boundary.",
    },
  };
  const meta = titles[view];
  if (!meta) return null;

  return (
    <section className="utility-panel" id="utility-panel">
      <PageHeader
        eyebrow={meta.eyebrow}
        title={meta.title}
        description={meta.description}
        actions={(
          <>
            <button onClick={onRefresh}><RefreshCcw size={16} /> Refresh</button>
            <button onClick={onClose}>Close</button>
          </>
        )}
      />

      {view === "hardware" && (
        <section className="bento utility-grid">
          <SurfaceCard title="Runtime" icon={Cpu}>
            <div className="telemetry-grid hardware-grid">
              <MetricTile icon={Cpu} label="CPU" value={stats.cpu} />
              <MetricTile icon={Database} label="RAM" value={stats.ram} />
              <MetricTile icon={Activity} label="GPU" value={stats.gpu} />
              <MetricTile icon={Gauge} label="VRAM" value={stats.vram_used} />
              <MetricTile icon={Zap} label="Power" value={stats.power} />
              <MetricTile icon={Activity} label="Temperature" value={stats.temp} />
            </div>
          </SurfaceCard>
          <SurfaceCard title="Tracked Training Process" icon={Activity}>
            <div className="summary-list">
              <span>Status <b>{dashboard?.running ? "Running" : "Idle"}</b></span>
              <span>PID <b>{fmt(trainingState.pid)}</b></span>
              <span>Process CPU <b>{fmt(stats.pid_cpu)}</b></span>
              <span>Process RAM <b>{fmt(stats.pid_rss)}</b></span>
              <span>Current step <b>{fmt(dashboard?.current_step, "0")}</b></span>
              <span>Total steps <b>{fmt(dashboard?.total_steps || training.steps, "0")}</b></span>
            </div>
          </SurfaceCard>
          <SurfaceCard title="Artifacts" icon={Library} className="wide-card">
            <div className="stat-row">
              <StatPill label="Sample files" value={dashboard?.artifacts?.sample_files?.length || 0} />
              <StatPill label="Checkpoint files" value={dashboard?.artifacts?.checkpoint_files?.length || 0} />
              <StatPill label="Dashboard progress" value={`${clampPercent(dashboard?.percent || 0)}%`} />
              <StatPill label="Latest sample" value={dashboard?.latest_sample ? "available" : "n/a"} />
            </div>
          </SurfaceCard>
        </section>
      )}

      {view === "datasets" && (
        <section className="bento utility-grid">
          <SurfaceCard title="Case Inventory" icon={Database}>
            <div className="stat-row">
              <StatPill label="References" value={state?.refs?.length || 0} />
              <StatPill label="Candidates" value={state?.candidates?.length || 0} />
              <StatPill label="QC rows" value={state?.qc?.length || 0} />
              <StatPill label="Curated train" value={state?.train?.length || 0} tone="good" />
              <StatPill label="Z-Image raw" value={state?.production_candidates?.length || 0} />
              <StatPill label="Final dataset" value={state?.final?.length || 0} tone="good" />
            </div>
            <div className="action-row utility-actions">
              <button onClick={onOpenLibrary}><Library size={16} /> Open Dataset Library</button>
            </div>
          </SurfaceCard>
          <SurfaceCard title="QC Result" icon={ShieldCheck}>
            <div className="summary-list">
              <span>Total evaluated <b>{qcSummary.total}</b></span>
              <span>Passed <b>{qcSummary.passed}</b></span>
              <span>Failed <b>{qcSummary.failed}</b></span>
              <span>Pending <b>{qcSummary.pending}</b></span>
            </div>
          </SurfaceCard>
          <SurfaceCard title="Storage" icon={FolderInput} className="wide-card">
            <div className="path-list">
              <span>Work root <b>{state?.work_root || "n/a"}</b></span>
              <span>Training config <b>{configPath || "n/a"}</b></span>
            </div>
          </SurfaceCard>
        </section>
      )}

      {view === "logs" && (
        <section className="bento utility-grid">
          <SurfaceCard title="Generation Log" icon={FileText}>
            <pre>{state?.logs?.generate || "No generation log yet"}</pre>
          </SurfaceCard>
          <SurfaceCard title="Training Log" icon={FileText}>
            <pre>{state?.logs?.train || "No training log yet"}</pre>
          </SurfaceCard>
          <SurfaceCard title="Z-Image Logs" icon={FileText}>
            <pre>{[state?.logs?.zimage_identity_lora, state?.logs?.zimage_expansion].filter(Boolean).join("\n\n") || "No Z-Image logs yet"}</pre>
          </SurfaceCard>
        </section>
      )}

      {view === "terminal" && (
        <section className="bento utility-grid">
          <SurfaceCard title="Current Boundary" icon={SquareTerminal}>
            <div className="terminal-note">
              <p>Project Chimera is proxy-only from the browser. The web app does not expose an interactive shell endpoint.</p>
              <p>Use the Colab notebook cell or temporary SSH/tmate session for terminal operations.</p>
            </div>
          </SurfaceCard>
          <SurfaceCard title="Useful Paths" icon={FolderInput}>
            <div className="path-list">
              <span>Runtime app <b>/content/project-chimera</b></span>
              <span>Persistent repo <b>/content/drive/MyDrive/GenAI/Chimera</b></span>
              <span>Work root <b>{state?.work_root || "n/a"}</b></span>
            </div>
          </SurfaceCard>
        </section>
      )}
    </section>
  );
}

function App() {
  const [cases, setCases] = useState([]);
  const [activeCase, setActiveCase] = useState("");
  const [caseLabel, setCaseLabel] = useState("character");
  const [state, setState] = useState(null);
  const [status, setStatus] = useState("");
  const [preflight, setPreflight] = useState(null);
  const [busy, setBusy] = useState(false);
  const [files, setFiles] = useState([]);
  const [consent, setConsent] = useState(false);
  const [count, setCount] = useState(10);
  const [zimageCount, setZimageCount] = useState(5000);
  const [importFolder, setImportFolder] = useState("/content/drive/MyDrive/GenAI/ComfyUI/output");
  const [generationCommand, setGenerationCommand] = useState(defaultGenerationCommand);
  const [qc, setQc] = useState({ identity_threshold: fixedIdentityThreshold, min_face_area: 0.01, top_n: 10 });
  const [training, setTraining] = useState(defaultTraining);
  const [samplePrompts, setSamplePrompts] = useState(defaultSamplePrompts);
  const [configPath, setConfigPath] = useState("");
  const [activePhase, setActivePhase] = useState("ingest");
  const [utilityView, setUtilityViewState] = useState(hashUtilityView);
  const pollRef = useRef(null);

  const dashboard = state?.dashboard;
  const qcByPath = useMemo(() => {
    const map = new Map();
    (state?.qc || []).forEach((row) => {
      if (row.file) map.set(row.file, row);
    });
    return map;
  }, [state?.qc]);
  const qcSummary = useMemo(() => {
    const rows = state?.qc || [];
    const passed = rows.filter((row) => normalizeBool(row.passed)).length;
    const failed = rows.length - passed;
    const pending = Math.max((state?.candidates?.length || 0) - rows.length, 0);
    return { total: rows.length, passed, failed, pending };
  }, [state]);

  function changePhase(nextPhase) {
    setUtilityViewState("");
    window.history.replaceState(null, "", window.location.pathname + window.location.search);
    setActivePhase(nextPhase);
    window.requestAnimationFrame(() => window.scrollTo({ top: 0, behavior: "smooth" }));
  }

  function changeUtilityView(nextView) {
    setUtilityViewState(nextView);
    window.history.replaceState(null, "", `#${nextView}`);
    window.requestAnimationFrame(() => {
      document.getElementById("utility-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  function closeUtilityView() {
    setUtilityViewState("");
    window.history.replaceState(null, "", window.location.pathname + window.location.search);
  }

  async function loadCases() {
    const payload = await request("/api/cases");
    setCases(payload.cases);
    setActiveCase((existing) => existing || payload.active);
    if (payload.active) await loadState(payload.active);
  }

  async function loadState(caseName = activeCase) {
    if (!caseName) return;
    const payload = await request(`/api/cases/${encodeURIComponent(caseName)}/state?top_n=${qc.top_n}`);
    setConfigPath(payload.config_path || "");
    setState({ ...payload, configPath: payload.config_path || "" });
  }

  async function loadPreflight() {
    const payload = await request("/api/preflight");
    setPreflight(payload);
    return payload;
  }

  async function runAction(action) {
    setBusy(true);
    try {
      const payload = await action();
      if (payload?.status) setStatus(payload.status);
      if (payload?.config_path) {
        setConfigPath(payload.config_path);
        setState((previous) => ({ ...(payload.state || previous), configPath: payload.config_path }));
      } else if (payload?.state) {
        setConfigPath(payload.state.config_path || "");
        setState({ ...payload.state, configPath: payload.state.config_path || "" });
      }
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    loadCases().catch((error) => setStatus(error.message));
    loadPreflight().catch((error) => setStatus(error.message));
  }, []);

  useEffect(() => {
    const syncUtilityHash = () => setUtilityViewState(hashUtilityView());
    window.addEventListener("hashchange", syncUtilityHash);
    return () => window.removeEventListener("hashchange", syncUtilityHash);
  }, []);

  useEffect(() => {
    if (!activeCase) return;
    setConfigPath("");
    loadState(activeCase).catch((error) => setStatus(error.message));
  }, [activeCase]);

  useEffect(() => {
    if (!activeCase) return;
    clearInterval(pollRef.current);
    pollRef.current = setInterval(() => {
      loadState(activeCase).catch(() => {});
    }, 5000);
    return () => clearInterval(pollRef.current);
  }, [activeCase, qc.top_n]);

  const actions = {
    newCase: () => runAction(async () => {
      const payload = await request("/api/cases", { method: "POST", body: JSON.stringify({ label: caseLabel }) });
      setCases(payload.cases);
      setActiveCase(payload.case);
      return { ...payload, status: `Created case: ${payload.case}` };
    }),
    saveReferences: () => runAction(async () => {
      const data = new FormData();
      data.append("consent", String(consent));
      files.forEach((file) => data.append("files", file));
      return request(`/api/cases/${encodeURIComponent(activeCase)}/references`, { method: "POST", body: data });
    }),
    runFullPipeline: () => runAction(async () => {
      const data = new FormData();
      data.append("consent", String(consent));
      files.forEach((file) => data.append("files", file));
      await request(`/api/cases/${encodeURIComponent(activeCase)}/references`, { method: "POST", body: data });
      return request(`/api/cases/${encodeURIComponent(activeCase)}/pipeline/run`, {
        method: "POST",
        body: JSON.stringify({
          ...training,
          count,
          top_n: qc.top_n,
          min_face_area: qc.min_face_area,
          sample_prompts: samplePrompts,
          start_training: false,
          generation_backend: "flux2_pulid",
          allow_smoke_fallback: false,
        }),
      });
    }),
    smoke: () => runAction(() => request(`/api/cases/${encodeURIComponent(activeCase)}/smoke`, { method: "POST", body: JSON.stringify({ count }) })),
    importImages: () => runAction(() => request(`/api/cases/${encodeURIComponent(activeCase)}/import`, { method: "POST", body: JSON.stringify({ source_folder: importFolder, copy_limit: count }) })),
    startCommand: () => runAction(() => request(`/api/cases/${encodeURIComponent(activeCase)}/generation/start`, {
      method: "POST",
      body: JSON.stringify({ command: generationCommand, trigger: training.trigger, count }),
    })),
    scoreSelect: () => runAction(() => request(`/api/cases/${encodeURIComponent(activeCase)}/qc/score-select`, { method: "POST", body: JSON.stringify(qc) })),
    prepare: () => runAction(() => request(`/api/cases/${encodeURIComponent(activeCase)}/training/prepare`, {
      method: "POST",
      body: JSON.stringify({ ...training, sample_prompts: samplePrompts }),
    })),
    startTraining: () => runAction(() => request(`/api/cases/${encodeURIComponent(activeCase)}/training/start`, {
      method: "POST",
      body: JSON.stringify({ config_path: configPath, trigger: training.trigger, steps: training.steps, sample_prompts: samplePrompts, sample_every: training.sample_every, save_every: training.save_every }),
    })),
    runIdentityLora: () => runAction(() => request(`/api/cases/${encodeURIComponent(activeCase)}/identity-lora/run`, {
      method: "POST",
      body: JSON.stringify({ ...training, sample_prompts: samplePrompts, count: state?.train?.length || 1 }),
    })),
    runZImageExpansion: () => runAction(() => request(`/api/cases/${encodeURIComponent(activeCase)}/zimage-expansion/run`, {
      method: "POST",
      body: JSON.stringify({ trigger: training.trigger, count: zimageCount }),
    })),
    runFinalQc: () => runAction(() => request(`/api/cases/${encodeURIComponent(activeCase)}/final-qc/score-select`, {
      method: "POST",
      body: JSON.stringify({ min_face_area: qc.min_face_area, top_n: qc.top_n }),
    })),
    refresh: () => loadState(),
    refreshPreflight: () => runAction(async () => {
      const payload = await loadPreflight();
      return { status: payload.ready ? "Runtime readiness check passed." : "Runtime readiness check found setup issues." };
    }),
  };

  const phaseProps = { state, dashboard, busy, training, setTraining, samplePrompts, setSamplePrompts, configPath };
  const phase = {
    ingest: (
      <IngestPhase
        state={state}
        preflight={preflight}
        files={files}
        setFiles={setFiles}
        consent={consent}
        setConsent={setConsent}
        busy={busy}
        onRunPipeline={actions.runFullPipeline}
        onRefreshPreflight={actions.refreshPreflight}
        caseLabel={caseLabel}
        setCaseLabel={setCaseLabel}
        count={count}
        setCount={setCount}
        qc={qc}
        setQc={setQc}
        training={training}
        setTraining={setTraining}
      />
    ),
    "pulid-dataset": (
      <ExpansionPhase
        state={state}
        count={count}
        setCount={setCount}
        importFolder={importFolder}
        setImportFolder={setImportFolder}
        generationCommand={generationCommand}
        setGenerationCommand={setGenerationCommand}
        busy={busy}
        onSmoke={actions.smoke}
        onImport={actions.importImages}
        onStartCommand={actions.startCommand}
        dashboard={dashboard}
      />
    ),
    "strict-qc": (
      <QcPhase
        state={state}
        qc={qc}
        setQc={setQc}
        qcSummary={qcSummary}
        qcByPath={qcByPath}
        busy={busy}
        onScoreSelect={actions.scoreSelect}
        dashboard={dashboard}
      />
    ),
    "identity-lora": (
      <SeedPhase
        {...phaseProps}
        preflight={preflight}
        onPrepare={actions.prepare}
        onStart={actions.runIdentityLora}
        onRefresh={actions.refresh}
      />
    ),
    "zimage-expansion": (
      <ZImageExpansionPhase
        state={state}
        dashboard={dashboard}
        preflight={preflight}
        zimageCount={zimageCount}
        setZimageCount={setZimageCount}
        busy={busy}
        onRun={actions.runZImageExpansion}
      />
    ),
    "final-qc": (
      <FinalQcPhase
        state={state}
        qcSummary={qcSummary}
        dashboard={dashboard}
        busy={busy}
        onScoreSelect={actions.runFinalQc}
      />
    ),
    library: (
      <LibraryPhase
        state={state}
        training={training}
        setTraining={setTraining}
        samplePrompts={samplePrompts}
        setSamplePrompts={setSamplePrompts}
        busy={busy}
        onPrepare={actions.prepare}
      />
    ),
    factory: (
      <FactoryPhase
        {...phaseProps}
        onPrepare={actions.prepare}
        onStart={actions.startTraining}
        onRefresh={actions.refresh}
      />
    ),
  }[activePhase];

  return (
    <AppShell
      activePhase={activePhase}
      setActivePhase={changePhase}
      utilityView={utilityView}
      setUtilityView={changeUtilityView}
      cases={cases}
      activeCase={activeCase}
      setActiveCase={setActiveCase}
      caseLabel={caseLabel}
      setCaseLabel={setCaseLabel}
      onNewCase={actions.newCase}
      workRoot={state?.work_root}
      busy={busy}
      status={status}
    >
      <UtilityPanel
        view={utilityView}
        state={state}
        dashboard={dashboard}
        qcSummary={qcSummary}
        configPath={configPath}
        training={training}
        onClose={closeUtilityView}
        onRefresh={actions.refresh}
        onOpenLibrary={() => changeUtilityView("datasets")}
      />
      {phase}
    </AppShell>
  );
}

createRoot(document.getElementById("root")).render(<App />);
