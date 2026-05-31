import subprocess

try:
    import psutil
except Exception:
    psutil = None

from .state import read_training_state


def process_tree_stats(pid):
    if psutil is None or not pid:
        return {"pid_cpu": "n/a", "pid_ram": "n/a", "pid_rss": "n/a"}
    try:
        root = psutil.Process(int(pid))
        procs = [root] + root.children(recursive=True)
        cpu = 0.0
        rss = 0
        for proc in procs:
            try:
                cpu += proc.cpu_percent(interval=0.0)
                rss += proc.memory_info().rss
            except Exception:
                pass
        return {
            "pid_cpu": f"{cpu:.0f}%",
            "pid_ram": f"{rss / max(psutil.virtual_memory().total, 1):.0%}",
            "pid_rss": f"{rss / (1024 ** 3):.1f}GB",
        }
    except Exception:
        return {"pid_cpu": "n/a", "pid_ram": "n/a", "pid_rss": "n/a"}


def system_stats(case_name=None):
    stats = {
        "cpu": "n/a",
        "ram": "n/a",
        "gpu": "n/a",
        "vram": "n/a",
        "vram_used": "n/a",
        "power": "n/a",
        "temp": "n/a",
        "pid_cpu": "n/a",
        "pid_ram": "n/a",
        "pid_rss": "n/a",
    }
    if psutil is not None:
        try:
            stats["cpu"] = f"{psutil.cpu_percent(interval=0.05):.0f}%"
            stats["ram"] = f"{psutil.virtual_memory().percent:.0f}%"
        except Exception:
            pass
    try:
        query = "utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
        if out:
            gpu, mem_used, mem_total, power, temp = [part.strip() for part in out.splitlines()[0].split(",")[:5]]
            stats["gpu"] = f"{gpu}%"
            stats["vram"] = f"{float(mem_used) / max(float(mem_total), 1):.0%}"
            stats["vram_used"] = f"{float(mem_used) / 1024:.1f}/{float(mem_total) / 1024:.1f}GB"
            stats["power"] = f"{float(power):.0f}W"
            stats["temp"] = f"{temp}C"
    except Exception:
        pass
    if case_name:
        stats.update(process_tree_stats(read_training_state(case_name).get("pid")))
    return stats
