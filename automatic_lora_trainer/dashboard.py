from .media import dashboard_sample, training_artifact_steps
from .state import read_log, read_training_state
from .system import system_stats


def expected_interval_steps(total_steps, interval):
    try:
        total = int(float(total_steps))
        every = int(float(interval))
    except Exception:
        return []
    if total <= 0 or every <= 0:
        return []
    return list(range(every, total + 1, every))


def training_progress(case_name, target_steps):
    state = read_training_state(case_name)
    target = state.get("total_steps") or max(int(float(target_steps or 1)), 1)
    current = state.get("current_step")
    running = state.get("status") == "running"
    pct = 0 if current is None else max(0, min(100, int(round((int(current) / max(int(target), 1)) * 100))))
    return current, target, pct, running, state


def prompt_card_html(sample_prompts, case_name=None, total_steps=2000, sample_every=250):
    prompts = [p.replace("[trigger]", "zphchar").strip() for p in sample_prompts.splitlines() if p.strip()]
    prompts = prompts[:8] or [
        "zphchar person, studio portrait",
        "zphchar person, side profile",
        "zphchar person, full body",
        "zphchar person, cinematic close-up",
    ]
    expected = expected_interval_steps(total_steps, sample_every)
    actual_sample_steps = set(training_artifact_steps(case_name)["sample_steps"]) if case_name else set()
    cards = []
    for prompt in prompts:
        safe = prompt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        dots = "".join(
            f"<span class='{'done' if step in actual_sample_steps else 'pending'}' title='sample step {step}'></span>"
            for step in expected[:12]
        )
        cards.append(f"<div class='lf-card'><div class='lf-card-title'>{safe}</div><div class='lf-dots'>{dots}</div></div>")
    return "<div class='lf-card-grid'>" + "".join(cards) + "</div>"


def dashboard_html(case_name, target_steps, sample_prompts, sample_every=250, save_every=250):
    current, target, pct, running, _ = training_progress(case_name, target_steps)
    stats = system_stats(case_name)
    artifacts = training_artifact_steps(case_name)
    state = "RUNNING" if running else "IDLE"
    status_color = "#43b66b" if running else "#777"
    step_text = "waiting for ai-toolkit step report" if current is None else f"{current}/{target} steps"
    train_state = read_training_state(case_name)
    last_line = train_state.get("last_step_line") or ""
    safe_line = last_line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    sample_summary = f"samples every {int(float(sample_every or 0))} steps; {len(artifacts['sample_files'])} sample files found"
    checkpoint_summary = f"checkpoints every {int(float(save_every or 0))} steps; {len(artifacts['checkpoint_files'])} checkpoint files found"
    return f"""
<style>
.lf-shell {{border:2px solid #1f2933;border-radius:14px;padding:16px;background:#ffffff;}}
.lf-steps {{height:26px;border:2px solid #1f2933;border-radius:7px;overflow:hidden;background:#9ed8fb;}}
.lf-steps-fill {{height:100%;width:{pct}%;background:#a9efb7;}}
.lf-top {{display:grid;grid-template-columns:2.2fr .95fr;gap:16px;margin-top:14px;}}
.lf-sample {{min-height:142px;border:2px solid #1f2933;border-radius:18px;background:#b3efc1;display:flex;align-items:center;justify-content:center;text-align:center;font-weight:700;}}
.lf-stats {{border:2px solid #1f2933;border-radius:18px;background:#ffe99b;padding:12px;display:grid;grid-template-columns:1fr 1fr;gap:8px;}}
.lf-stat {{border:2px solid #1f2933;border-radius:9px;background:#ffe99b;padding:8px;text-align:center;font-weight:700;}}
.lf-power {{grid-column:1 / span 2;}}
.lf-meta {{display:flex;gap:12px;margin-top:8px;font-size:13px;color:#394150;flex-wrap:wrap;}}
.lf-state {{color:{status_color};font-weight:800;}}
.lf-line {{margin-top:8px;font-family:monospace;font-size:12px;color:#52606d;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.lf-artifacts {{margin-top:8px;font-size:12px;color:#394150;display:flex;gap:14px;flex-wrap:wrap;}}
.lf-card-grid {{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:12px;}}
.lf-card {{min-height:112px;border:2px solid #1f2933;border-radius:14px;background:#a8d8f7;padding:12px;display:flex;flex-direction:column;justify-content:space-between;}}
.lf-card-title {{font-weight:700;font-size:13px;line-height:1.25;}}
.lf-dots span {{display:inline-block;width:13px;height:13px;border:2px solid #1f2933;border-radius:4px;margin-right:4px;}}
.lf-dots span.done {{background:#ffe99b;}}
.lf-dots span.pending {{background:#f8fafc;}}
</style>
<div class="lf-shell">
  <div><strong>steps</strong></div>
  <div class="lf-steps"><div class="lf-steps-fill"></div></div>
  <div class="lf-meta"><span>{step_text}</span><span>{pct}%</span><span class="lf-state">{state}</span><span>PID {train_state.get("pid") or "n/a"}</span></div>
  <div class="lf-line">{safe_line or "No step line reported yet"}</div>
  <div class="lf-artifacts"><span>{sample_summary}</span><span>{checkpoint_summary}</span></div>
  <div class="lf-top">
    <div class="lf-sample">Character front face<br/>portrait sample</div>
    <div class="lf-stats">
      <div class="lf-stat">CPU<br/>{stats["pid_cpu"]}</div>
      <div class="lf-stat">RAM<br/>{stats["pid_rss"]}</div>
      <div class="lf-stat">GPU<br/>{stats["gpu"]}</div>
      <div class="lf-stat">VRAM<br/>{stats["vram_used"]}</div>
      <div class="lf-stat lf-power">POWER {stats["power"]} / TEMP {stats["temp"]}</div>
    </div>
  </div>
</div>
"""


def refresh_dashboard(case_name, target_steps, sample_prompts, sample_every=250, save_every=250):
    return (
        dashboard_html(case_name, target_steps, sample_prompts, sample_every, save_every),
        dashboard_sample(case_name),
        prompt_card_html(sample_prompts, case_name, target_steps, sample_every),
        read_log(case_name, "train"),
    )
