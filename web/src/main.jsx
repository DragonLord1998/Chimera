import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Check,
  Cpu,
  Database,
  FolderInput,
  Gauge,
  ImagePlus,
  Loader2,
  Play,
  RefreshCcw,
  Save,
  Sparkles,
  Upload,
  Wand2,
} from "lucide-react";
import "./styles.css";

const API = "";

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

function Metric({ icon: Icon, label, value }) {
  return (
    <div className="metric">
      <Icon size={18} />
      <span>{label}</span>
      <strong>{value || "n/a"}</strong>
    </div>
  );
}

function ImageGrid({ title, items, selected = false, empty = "No images" }) {
  return (
    <section className="panel media-panel">
      <div className="panel-title">
        <span>{title}</span>
        <span className="count">{items.length}</span>
      </div>
      <div className="image-grid">
        {items.map((item) => (
          <img
            key={item.path}
            src={`${item.url}&t=${Date.now()}`}
            alt={item.name}
            className={cx("tile", (selected || item.selected) && "selected")}
            loading="lazy"
          />
        ))}
        {items.length === 0 && <div className="empty">{empty}</div>}
      </div>
    </section>
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

function TrainingDashboard({ state, dashboard, samplePrompts, training, onPrepare, onStart, onRefresh, busy }) {
  const current = dashboard?.current_step;
  const total = dashboard?.total_steps || training.steps;
  const percent = dashboard?.percent || 0;
  const stats = dashboard?.stats || {};
  const trainingState = dashboard?.training_state || {};
  const stepText = current == null ? "waiting for ai-toolkit step report" : `${current}/${total} steps`;

  return (
    <section className="dashboard">
      <div className="progress-block">
        <div className="progress-labels">
          <strong>steps</strong>
          <span>{stepText}</span>
          <span className={dashboard?.running ? "live" : "idle"}>{dashboard?.running ? "RUNNING" : "IDLE"}</span>
          <span>PID {trainingState.pid || "n/a"}</span>
        </div>
        <div className="progress-track">
          <div style={{ width: `${percent}%` }} />
        </div>
        <code className="step-line">{trainingState.last_step_line || "No step line reported yet"}</code>
      </div>

      <div className="dashboard-main">
        <div className="sample-preview">
          {dashboard?.latest_sample ? (
            <img src={`${dashboard.latest_sample}&t=${Date.now()}`} alt="Latest training sample" />
          ) : (
            <span>Character front face portrait sample</span>
          )}
        </div>
        <aside className="stats-panel">
          <Metric icon={Cpu} label="CPU" value={stats.pid_cpu} />
          <Metric icon={Database} label="RAM" value={stats.pid_rss} />
          <Metric icon={Activity} label="GPU" value={stats.gpu} />
          <Metric icon={Gauge} label="VRAM" value={stats.vram_used} />
          <Metric icon={Sparkles} label="POWER" value={`${stats.power || "n/a"} / ${stats.temp || "n/a"}`} />
        </aside>
      </div>

      <div className="artifact-strip">
        <span>Samples every {training.sample_every} steps</span>
        <span>{dashboard?.artifacts?.sample_files?.length || 0} sample files</span>
        <span>Checkpoints every {training.save_every} steps</span>
        <span>{dashboard?.artifacts?.checkpoint_files?.length || 0} checkpoint files</span>
      </div>

      <PromptCards promptsText={samplePrompts} dashboard={dashboard} steps={training.steps} sampleEvery={training.sample_every} />

      <div className="action-row">
        <button onClick={onPrepare} disabled={busy}>
          <Save size={16} /> Caption + Config
        </button>
        <button onClick={onStart} disabled={busy || !state?.configPath} className="primary">
          <Play size={16} /> Start Training
        </button>
        <button onClick={onRefresh} disabled={busy}>
          <RefreshCcw size={16} /> Refresh
        </button>
      </div>
    </section>
  );
}

function App() {
  const [cases, setCases] = useState([]);
  const [activeCase, setActiveCase] = useState("");
  const [caseLabel, setCaseLabel] = useState("character");
  const [state, setState] = useState(null);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [files, setFiles] = useState([]);
  const [consent, setConsent] = useState(false);
  const [count, setCount] = useState(200);
  const [importFolder, setImportFolder] = useState("/content/drive/MyDrive/GenAI/ComfyUI/output");
  const [generationCommand, setGenerationCommand] = useState(defaultGenerationCommand);
  const [qc, setQc] = useState({ identity_threshold: 0.32, min_face_area: 0.01, top_n: 100 });
  const [training, setTraining] = useState({
    trigger: "zphchar",
    base_caption: "natural skin texture, realistic photo, high detail",
    model_name: "black-forest-labs/FLUX.2-klein-base-9B",
    rank: 64,
    steps: 2000,
    lr: "1e-4",
    sample_every: 250,
    save_every: 250,
  });
  const [samplePrompts, setSamplePrompts] = useState(defaultSamplePrompts);
  const [configPath, setConfigPath] = useState("");
  const [view, setView] = useState("pipeline");
  const pollRef = useRef(null);

  const dashboard = state?.dashboard;

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
  }, []);

  useEffect(() => {
    if (!activeCase) return;
    clearInterval(pollRef.current);
    pollRef.current = setInterval(() => {
      loadState(activeCase).catch(() => {});
    }, 5000);
    return () => clearInterval(pollRef.current);
  }, [activeCase, qc.top_n, configPath]);

  const selectedCount = useMemo(() => state?.candidates?.filter((item) => item.selected).length || 0, [state]);

  return (
    <main>
      <header className="app-header">
        <div>
          <h1>Project Chimera</h1>
          <p>{state?.work_root || "Preparing workspace"}</p>
        </div>
        <div className="header-actions">
          <select value={activeCase} onChange={(event) => { setActiveCase(event.target.value); setConfigPath(""); loadState(event.target.value); }}>
            {cases.map((name) => <option key={name}>{name}</option>)}
          </select>
          <input value={caseLabel} onChange={(event) => setCaseLabel(event.target.value)} />
          <button onClick={() => runAction(async () => {
            const payload = await request("/api/cases", { method: "POST", body: JSON.stringify({ label: caseLabel }) });
            setCases(payload.cases);
            setActiveCase(payload.case);
            return payload;
          })}>
            <ImagePlus size={16} /> New Case
          </button>
        </div>
      </header>

      <nav className="tabs">
        {["pipeline", "advanced", "logs"].map((name) => (
          <button key={name} className={view === name ? "active" : ""} onClick={() => setView(name)}>{name}</button>
        ))}
      </nav>

      {status && <div className="status">{busy && <Loader2 className="spin" size={16} />} {status}</div>}

      {view === "pipeline" && (
        <>
          <section className="pipeline">
            <div className="panel step-panel">
              <div className="panel-title"><span>1. Upload</span><Upload size={18} /></div>
              <label className="check-line">
                <input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} />
                I have consent/rights
              </label>
              <input type="file" multiple accept="image/*" onChange={(event) => setFiles([...event.target.files])} />
              <button className="primary" disabled={busy || !files.length || !consent} onClick={() => runAction(async () => {
                const data = new FormData();
                data.append("consent", String(consent));
                files.forEach((file) => data.append("files", file));
                return request(`/api/cases/${encodeURIComponent(activeCase)}/references`, { method: "POST", body: data });
              })}>
                <Upload size={16} /> Save References
              </button>
              <ImageGrid title="References" items={state?.refs || []} />
            </div>

            <div className="panel step-panel">
              <div className="panel-title"><span>2. Generate</span><Wand2 size={18} /></div>
              <label>Count <input type="number" value={count} onChange={(event) => setCount(Number(event.target.value))} /></label>
              <button disabled={busy} onClick={() => runAction(() => request(`/api/cases/${encodeURIComponent(activeCase)}/smoke`, { method: "POST", body: JSON.stringify({ count }) }))}>
                <Sparkles size={16} /> Smoke Test
              </button>
              <label>Import folder <input value={importFolder} onChange={(event) => setImportFolder(event.target.value)} /></label>
              <button disabled={busy} onClick={() => runAction(() => request(`/api/cases/${encodeURIComponent(activeCase)}/import`, { method: "POST", body: JSON.stringify({ source_folder: importFolder, copy_limit: count }) }))}>
                <FolderInput size={16} /> Import Images
              </button>
              <ImageGrid title="Synthetic Candidates" items={state?.candidates || []} />
            </div>

            <div className="panel step-panel">
              <div className="panel-title"><span>3. QC Select</span><Check size={18} /></div>
              <label>Identity <input type="number" step="0.01" value={qc.identity_threshold} onChange={(event) => setQc({ ...qc, identity_threshold: Number(event.target.value) })} /></label>
              <label>Min face <input type="number" step="0.001" value={qc.min_face_area} onChange={(event) => setQc({ ...qc, min_face_area: Number(event.target.value) })} /></label>
              <label>Keep <input type="number" value={qc.top_n} onChange={(event) => setQc({ ...qc, top_n: Number(event.target.value) })} /></label>
              <button className="primary" disabled={busy} onClick={() => runAction(() => request(`/api/cases/${encodeURIComponent(activeCase)}/qc/score-select`, { method: "POST", body: JSON.stringify(qc) }))}>
                <Check size={16} /> Score + Select
              </button>
              <ImageGrid title={`QC Highlights (${selectedCount})`} items={state?.candidates || []} />
            </div>
          </section>

          <ImageGrid title="Curated Training Set" items={state?.train || []} />

          <TrainingDashboard
            state={{ configPath }}
            dashboard={dashboard}
            samplePrompts={samplePrompts}
            training={training}
            busy={busy}
            onPrepare={() => runAction(() => request(`/api/cases/${encodeURIComponent(activeCase)}/training/prepare`, {
              method: "POST",
              body: JSON.stringify({ ...training, sample_prompts: samplePrompts }),
            }))}
            onStart={() => runAction(() => request(`/api/cases/${encodeURIComponent(activeCase)}/training/start`, {
              method: "POST",
              body: JSON.stringify({ config_path: configPath, trigger: training.trigger, steps: training.steps, sample_prompts: samplePrompts, sample_every: training.sample_every, save_every: training.save_every }),
            }))}
            onRefresh={() => loadState()}
          />
        </>
      )}

      {view === "advanced" && (
        <section className="advanced">
          <div className="panel">
            <div className="panel-title"><span>Generation Command</span><Play size={18} /></div>
            <textarea value={generationCommand} onChange={(event) => setGenerationCommand(event.target.value)} rows={12} />
            <button disabled={busy} onClick={() => runAction(() => request(`/api/cases/${encodeURIComponent(activeCase)}/generation/start`, {
              method: "POST",
              body: JSON.stringify({ command: generationCommand, trigger: training.trigger, count }),
            }))}>
              <Play size={16} /> Start Command
            </button>
          </div>
          <div className="panel">
            <div className="panel-title"><span>Training Config</span><Save size={18} /></div>
            <label>Trigger <input value={training.trigger} onChange={(event) => setTraining({ ...training, trigger: event.target.value })} /></label>
            <label>Base model <input value={training.model_name} onChange={(event) => setTraining({ ...training, model_name: event.target.value })} /></label>
            <label>Caption suffix <input value={training.base_caption} onChange={(event) => setTraining({ ...training, base_caption: event.target.value })} /></label>
            <div className="grid-form">
              <label>Rank <input type="number" value={training.rank} onChange={(event) => setTraining({ ...training, rank: Number(event.target.value) })} /></label>
              <label>Steps <input type="number" value={training.steps} onChange={(event) => setTraining({ ...training, steps: Number(event.target.value) })} /></label>
              <label>LR <input value={training.lr} onChange={(event) => setTraining({ ...training, lr: event.target.value })} /></label>
              <label>Sample every <input type="number" value={training.sample_every} onChange={(event) => setTraining({ ...training, sample_every: Number(event.target.value) })} /></label>
              <label>Checkpoint every <input type="number" value={training.save_every} onChange={(event) => setTraining({ ...training, save_every: Number(event.target.value) })} /></label>
            </div>
            <textarea value={samplePrompts} onChange={(event) => setSamplePrompts(event.target.value)} rows={8} />
          </div>
        </section>
      )}

      {view === "logs" && (
        <section className="logs">
          <div className="panel">
            <div className="panel-title"><span>Generation Log</span></div>
            <pre>{state?.logs?.generate || ""}</pre>
          </div>
          <div className="panel">
            <div className="panel-title"><span>Training Log</span></div>
            <pre>{state?.logs?.train || ""}</pre>
          </div>
        </section>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
