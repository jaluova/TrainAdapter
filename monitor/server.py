import base64
import json
import os
import shlex
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "index.html"
REFRESH_SECONDS = float(os.environ.get("TRAIN_MONITOR_REFRESH_SECONDS", "8"))


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _remote_status_script() -> str:
    log_path = os.environ.get(
        "TRAIN_MONITOR_LOG_PATH",
        "/root/autodl-tmp/Data/train_outputs/fast1000_grid_6epoch_20260331/nohup.log",
    )
    save_dir = os.environ.get(
        "TRAIN_MONITOR_SAVE_DIR",
        "/root/autodl-tmp/Data/train_outputs/fast1000_grid_6epoch_20260331",
    )
    process_pattern = os.environ.get(
        "TRAIN_MONITOR_PROCESS_PATTERN",
        "python train.py",
    )
    return f"""
import json
import os
import re
import subprocess
from pathlib import Path

log_path = Path({log_path!r})
save_dir = Path({save_dir!r})
process_pattern = {process_pattern!r}

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

raw = b""
if log_path.exists():
    raw = log_path.read_bytes().replace(b"\\x00", b"")
text = raw.decode("utf-8", errors="replace")
lines = [line for line in text.splitlines() if line.strip()]

step_matches = re.findall(
    r"Step\\s+(\\d+),\\s+Loss:\\s+([0-9.]+),\\s+Avg Loss:\\s+([0-9.]+)(?:,\\s+Relation Loss:\\s+([0-9.]+))?",
    text,
)
val_matches = re.findall(
    r"Validation - Loss:\\s+([0-9.]+),\\s+Mean Min Grid Distance:\\s+([0-9.]+),\\s+Acc@1Grid:\\s+([0-9.%]+),\\s+Acc@Top4:\\s+([0-9.%]+),\\s+Relation Acc@Top4:\\s+([0-9.%]+)",
    text,
)
epoch_matches = re.findall(r"Epoch\\s+(\\d+)/(\\d+)", text)

latest_step = None
if step_matches:
    step, loss, avg_loss, relation_loss = step_matches[-1]
    latest_step = {{
        "step": int(step),
        "loss": float(loss),
        "avg_loss": float(avg_loss),
        "relation_loss": float(relation_loss) if relation_loss else None,
    }}

latest_val = None
if val_matches:
    val_loss, min_dist, acc1, acc4, rel_acc4 = val_matches[-1]
    latest_val = {{
        "loss": float(val_loss),
        "mean_min_grid_distance": float(min_dist),
        "acc_1grid": acc1,
        "acc_top4": acc4,
        "relation_acc_top4": rel_acc4,
    }}

epoch_info = None
if epoch_matches:
    current_epoch, total_epochs = epoch_matches[-1]
    epoch_info = {{
        "current_epoch": int(current_epoch),
        "total_epochs": int(total_epochs),
    }}

gpu_csv = run([
    "nvidia-smi",
    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
    "--format=csv,noheader,nounits",
])
processes = run(["bash", "-lc", f"pgrep -af {{process_pattern!r}} || true"])

checkpoints = []
checkpoint_dir = save_dir / "checkpoints"
if checkpoint_dir.exists():
    checkpoints = sorted(
        [path.name for path in checkpoint_dir.glob("checkpoint_step_*.pth")]
    )[-5:]

payload = {{
    "timestamp": int(float(Path("/proc/uptime").read_text().split()[0])) if Path("/proc/uptime").exists() else None,
    "log_path": str(log_path),
    "save_dir": str(save_dir),
    "latest_step": latest_step,
    "latest_validation": latest_val,
    "epoch_info": epoch_info,
    "gpu": gpu_csv.splitlines(),
    "processes": processes.splitlines(),
    "checkpoints": checkpoints,
    "recent_lines": lines[-80:],
}}
print(json.dumps(payload, ensure_ascii=False))
"""


def _run_remote_command() -> dict[str, Any]:
    host = _required_env("TRAIN_MONITOR_HOST")
    port = _required_env("TRAIN_MONITOR_PORT")
    user = _required_env("TRAIN_MONITOR_USER")
    password = _required_env("TRAIN_MONITOR_PASSWORD")

    remote_script = _remote_status_script()
    encoded = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
    remote_cmd = (
        "python -c "
        "\"import base64; exec(base64.b64decode('{}').decode('utf-8'))\"".format(encoded)
    )
    remote_shell_cmd = f"bash -lc {shlex.quote(remote_cmd)}"
    expect_script = "\n".join(
        [
            "set timeout 40",
            f"spawn ssh -o StrictHostKeyChecking=no -p {port} {user}@{host} {{{remote_shell_cmd}}}",
            'expect "password:" {send "' + password + '\\r"}',
            "expect eof",
        ]
    )
    completed = subprocess.run(
        ["/usr/bin/expect", "-c", expect_script],
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or output).strip() or "remote command failed")

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    json_line = None
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            json_line = line
            break
    if json_line is None:
        raise RuntimeError("No JSON payload returned from remote status command")
    return json.loads(json_line)


class StatusCache:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.payload: dict[str, Any] = {
            "ok": False,
            "error": "Waiting for first refresh",
            "last_refresh_epoch_ms": None,
        }

    def update(self) -> None:
        try:
            remote_payload = _run_remote_command()
            payload = {
                "ok": True,
                "last_refresh_epoch_ms": int(time.time() * 1000),
                "data": remote_payload,
            }
        except Exception as exc:
            payload = {
                "ok": False,
                "last_refresh_epoch_ms": int(time.time() * 1000),
                "error": str(exc),
            }
        with self.lock:
            self.payload = payload

    def get(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.payload)


STATUS_CACHE = StatusCache()


def _refresh_loop() -> None:
    while True:
        STATUS_CACHE.update()
        time.sleep(REFRESH_SECONDS)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_html(INDEX_PATH.read_bytes())
            return
        if self.path == "/api/status":
            self._send_json(STATUS_CACHE.get())
            return
        self._send_json({"ok": False, "error": "Not found"}, status=404)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    port = int(os.environ.get("TRAIN_MONITOR_PORT_LOCAL", "4173"))
    threading.Thread(target=_refresh_loop, daemon=True).start()
    STATUS_CACHE.update()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Train monitor listening on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
