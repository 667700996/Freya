"""
Freya Load Test Harness
=======================

Freya is a parallel load testing utility inspired by the Hermes rate tester.
It can generate sustained load against one or more HTTP endpoints using a
weighted scenario definition and produces detailed summaries and optional logs.

Two primary modes are supported:

1. Run mode (`freya.py run ...`) executes a load test against real services.
   Scenarios can be provided via CLI flags or a JSON file.
2. Simulate mode (`freya.py simulate ...`) spins up a lightweight HTTP service
   with configurable latency and error profiles for local experiments.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import random
import signal
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    import aiohttp
    from aiohttp import web
except ImportError as exc:  # noqa: F401
    raise SystemExit(
        "필수 라이브러리 'aiohttp'가 설치되어 있지 않습니다.\n"
        "다음 명령으로 설치 후 다시 실행하세요:\n"
        "    pip install aiohttp>=3.9"
    ) from exc

try:  # GUI 의존성 (옵션)
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception:  # noqa: BLE001
    tk = None
    filedialog = None
    messagebox = None
    ttk = None


# ------------------------------------------------------------------------------
# Shared utility classes
# ------------------------------------------------------------------------------


@dataclass
class LatencyStats:
    count: int = 0
    total_ms: float = 0.0
    min_ms: Optional[float] = None
    max_ms: Optional[float] = None

    def add(self, latency_ms: float) -> None:
        self.count += 1
        self.total_ms += latency_ms
        if self.min_ms is None or latency_ms < self.min_ms:
            self.min_ms = latency_ms
        if self.max_ms is None or latency_ms > self.max_ms:
            self.max_ms = latency_ms

    def mean(self) -> Optional[float]:
        if not self.count:
            return None
        return self.total_ms / self.count


@dataclass
class RequestOutcome:
    target: str
    status: Optional[int]
    latency_ms: float
    error: Optional[str] = None


@dataclass
class TargetConfig:
    name: str
    url: str
    method: str = "GET"
    weight: float = 1.0
    timeout_s: float = 10.0
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetConfig":
        try:
            name = str(data["name"]).strip()
            url = str(data["url"]).strip()
        except KeyError as exc:  # noqa: F401
            raise ValueError("Target requires 'name' and 'url' fields") from exc
        if not name or not url:
            raise ValueError("Target 'name' and 'url' must be non-empty")
        method = str(data.get("method", "GET")).strip().upper() or "GET"
        weight_raw = float(data.get("weight", 1.0))
        if weight_raw <= 0:
            raise ValueError(f"Target '{name}' weight must be positive")
        timeout = float(data.get("timeout_s", 10.0))
        if timeout <= 0:
            raise ValueError(f"Target '{name}' timeout must be positive")
        headers_raw = data.get("headers") or {}
        if not isinstance(headers_raw, dict):
            raise ValueError(f"Target '{name}' headers must be a JSON object")
        body = data.get("body")
        return cls(
            name=name,
            url=url,
            method=method,
            weight=weight_raw,
            timeout_s=timeout,
            headers={str(k): str(v) for k, v in headers_raw.items()},
            body=None if body is None else str(body),
        )


@dataclass
class LoadScenario:
    duration_s: float
    global_rps: float
    concurrency: int
    targets: List[TargetConfig]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoadScenario":
        duration = float(data.get("duration_s", 30))
        if duration <= 0:
            raise ValueError("duration_s must be positive")
        rps = float(data.get("global_rps", 10))
        if rps <= 0:
            raise ValueError("global_rps must be positive")
        concurrency = int(data.get("concurrency", max(1, int(rps))))
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        raw_targets = data.get("targets")
        if not raw_targets or not isinstance(raw_targets, list):
            raise ValueError("scenario requires a list of targets")
        targets = [TargetConfig.from_dict(entry) for entry in raw_targets]
        return cls(duration_s=duration, global_rps=rps, concurrency=concurrency, targets=targets)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "LoadScenario":
        if args.scenario:
            payload = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
            return cls.from_dict(payload)

        if not args.url:
            raise ValueError("단일 타겟 실행 시 --url 을 지정하세요 (또는 --scenario 사용)")
        target = TargetConfig(
            name=args.name or "default",
            url=args.url,
            method=(args.method or "GET").upper(),
            weight=1.0,
            timeout_s=args.timeout,
            headers=parse_header_lines(args.headers or ""),
            body=args.body,
        )
        scenario_dict = {
            "duration_s": args.duration,
            "global_rps": args.rps,
            "concurrency": args.concurrency or max(1, int(args.rps)),
            "targets": [target.__dict__],
        }
        return cls.from_dict(scenario_dict)


# ------------------------------------------------------------------------------
# Load test runner
# ------------------------------------------------------------------------------


def parse_header_lines(raw: str) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for line in raw.splitlines():
        striped = line.strip()
        if not striped:
            continue
        if ":" not in striped:
            raise ValueError(f"헤더 형식 오류: {line!r} (Name: Value)")
        name, value = striped.split(":", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise ValueError(f"헤더 이름이 비어 있습니다: {line!r}")
        headers[name] = value
    return headers


async def execute_request(
    session: aiohttp.ClientSession,
    target: TargetConfig,
) -> RequestOutcome:
    started = time.perf_counter()
    try:
        async with session.request(
            target.method,
            target.url,
            timeout=target.timeout_s,
            headers=target.headers or None,
            data=target.body.encode("utf-8") if target.body else None,
        ) as resp:
            await resp.read()
            latency = (time.perf_counter() - started) * 1000.0
            return RequestOutcome(target=target.name, status=resp.status, latency_ms=latency)
    except Exception as exc:  # noqa: BLE001
        latency = (time.perf_counter() - started) * 1000.0
        return RequestOutcome(target=target.name, status=None, latency_ms=latency, error=str(exc))


def weighted_choice(targets: List[TargetConfig]) -> TargetConfig:
    total_weight = sum(t.weight for t in targets)
    pick = random.random() * total_weight
    cumulative = 0.0
    for target in targets:
        cumulative += target.weight
        if pick <= cumulative:
            return target
    return targets[-1]


class MetricsRecorder:
    def __init__(self) -> None:
        self.started_at = datetime.now(timezone.utc)
        self.total_sent = 0
        self.by_target: Dict[str, Counter] = defaultdict(Counter)
        self.latencies: Dict[str, LatencyStats] = defaultdict(LatencyStats)
        self.errors: Counter = Counter()

    def record(self, outcome: RequestOutcome) -> None:
        self.total_sent += 1
        status_key = "ERR" if outcome.status is None else str(outcome.status)
        self.by_target[outcome.target][status_key] += 1
        if outcome.status is not None:
            self.latencies[outcome.target].add(outcome.latency_ms)
        if outcome.error:
            self.errors[outcome.error.split(":")[0]] += 1

    def to_summary(self, elapsed_s: float) -> Dict[str, Any]:
        targets_summary = {}
        for target, status_counts in self.by_target.items():
            summary: Dict[str, Any] = {
                "status_counts": dict(status_counts),
            }
            lat = self.latencies.get(target)
            if lat and lat.count:
                summary["latency"] = {
                    "count": lat.count,
                    "average_ms": round(lat.mean() or 0.0, 3),
                    "min_ms": round(lat.min_ms or 0.0, 3),
                    "max_ms": round(lat.max_ms or 0.0, 3),
                }
            else:
                summary["latency"] = {"count": 0}
            targets_summary[target] = summary

        return {
            "started_at": self.started_at.isoformat().replace("+00:00", "Z"),
            "elapsed_s": round(elapsed_s, 3),
            "total_sent": self.total_sent,
            "targets": targets_summary,
            "errors": dict(self.errors),
        }


async def run_load_test(
    scenario: LoadScenario,
    print_log: bool = False,
    log_file: Optional[Path] = None,
    summary_file: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    stop_flag: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    metrics = MetricsRecorder()
    log_lines: List[str] = []
    shared_stop = stop_flag or threading.Event()

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    connector = aiohttp.TCPConnector(ssl=False, limit=None)
    timeout = aiohttp.ClientTimeout(total=None)

    queue_tasks: asyncio.Queue[Optional[TargetConfig]] = asyncio.Queue()

    def emit(msg: str) -> None:
        if log_file or not log_callback:
            log_lines.append(msg)
        if log_callback:
            log_callback(msg)
        elif print_log:
            print(msg)

    async def producer() -> None:
        interval = 1.0 / scenario.global_rps
        loop = asyncio.get_running_loop()
        end_time = loop.time() + scenario.duration_s
        next_tick = loop.time()
        while loop.time() < end_time and not shared_stop.is_set():
            target = weighted_choice(scenario.targets)
            await queue_tasks.put(target)
            next_tick += interval
            delay = max(0.0, next_tick - loop.time())
            if delay:
                await asyncio.sleep(delay)
        for _ in range(scenario.concurrency):
            await queue_tasks.put(None)

    async def worker(index: int, session: aiohttp.ClientSession) -> None:
        while True:
            target = await queue_tasks.get()
            if target is None:
                queue_tasks.task_done()
                break
            if shared_stop.is_set():
                queue_tasks.task_done()
                continue
            outcome = await execute_request(session, target)
            metrics.record(outcome)
            stamp = datetime.now().isoformat(timespec="milliseconds")
            status = outcome.status if outcome.status is not None else "ERR"
            msg = f"{stamp} worker={index:02d} target={target.name} status={status} latency_ms={outcome.latency_ms:.2f}"
            if outcome.error:
                msg += f" error={outcome.error}"
            emit(msg)
            queue_tasks.task_done()

    started = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        workers = [asyncio.create_task(worker(i, session)) for i in range(1, scenario.concurrency + 1)]
        producer_task = asyncio.create_task(producer())
        try:
            await asyncio.gather(producer_task, queue_tasks.join())
        finally:
            shared_stop.set()
            for _ in range(scenario.concurrency):
                await queue_tasks.put(None)
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    elapsed = time.perf_counter() - started
    summary = metrics.to_summary(elapsed)

    if log_file:
        log_file.write_text("\n".join(log_lines), encoding="utf-8")
    if summary_file:
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


# ------------------------------------------------------------------------------
# GUI
# ------------------------------------------------------------------------------


class TargetDialog:
    def __init__(self, parent: tk.Tk, target: Optional[TargetConfig] = None):
        if tk is None or ttk is None:
            raise RuntimeError("tkinter를 사용할 수 없습니다.")
        self.result: Optional[TargetConfig] = None
        self.window = tk.Toplevel(parent)
        self.window.title("타겟 설정")
        self.window.transient(parent)
        self.window.grab_set()
        self.window.resizable(False, False)

        frame = ttk.Frame(self.window, padding=16)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="이름").grid(row=0, column=0, sticky="w")
        self.ent_name = ttk.Entry(frame, width=32)
        self.ent_name.grid(row=0, column=1, sticky="we", pady=4)

        ttk.Label(frame, text="URL").grid(row=1, column=0, sticky="w")
        self.ent_url = ttk.Entry(frame, width=48)
        self.ent_url.grid(row=1, column=1, sticky="we", pady=4)

        ttk.Label(frame, text="메서드").grid(row=2, column=0, sticky="w")
        self.cmb_method = ttk.Combobox(
            frame,
            values=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
            width=8,
        )
        self.cmb_method.grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(frame, text="가중치").grid(row=3, column=0, sticky="w")
        self.ent_weight = ttk.Entry(frame, width=12)
        self.ent_weight.grid(row=3, column=1, sticky="w", pady=4)

        ttk.Label(frame, text="타임아웃(초)").grid(row=4, column=0, sticky="w")
        self.ent_timeout = ttk.Entry(frame, width=12)
        self.ent_timeout.grid(row=4, column=1, sticky="w", pady=4)

        headers_box = ttk.Labelframe(frame, text="헤더 (Name: Value)")
        headers_box.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(8, 4))
        self.txt_headers = tk.Text(headers_box, height=4, width=48, wrap="none")
        self.txt_headers.pack(fill="both", expand=True)

        body_box = ttk.Labelframe(frame, text="본문 (선택)")
        body_box.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=(4, 8))
        self.txt_body = tk.Text(body_box, height=6, width=48, wrap="none")
        self.txt_body.pack(fill="both", expand=True)

        btns = ttk.Frame(frame)
        btns.grid(row=7, column=0, columnspan=2, sticky="e")
        ttk.Button(btns, text="취소", command=self.window.destroy).pack(side="right", padx=4)
        ttk.Button(btns, text="저장", command=self.on_save).pack(side="right", padx=4)

        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(5, weight=1)
        frame.grid_rowconfigure(6, weight=1)

        if target:
            self.ent_name.insert(0, target.name)
            self.ent_url.insert(0, target.url)
            self.cmb_method.set(target.method)
            self.ent_weight.insert(0, str(target.weight))
            self.ent_timeout.insert(0, str(target.timeout_s))
            headers_txt = "\n".join(f"{k}: {v}" for k, v in target.headers.items())
            self.txt_headers.insert("1.0", headers_txt)
            if target.body:
                self.txt_body.insert("1.0", target.body)
        else:
            self.cmb_method.set("GET")
            self.ent_weight.insert(0, "1.0")
            self.ent_timeout.insert(0, "10.0")

        self.window.bind("<Return>", lambda _: self.on_save())
        self.window.bind("<Escape>", lambda _: self.window.destroy())
        self.ent_name.focus_set()

    def on_save(self) -> None:
        try:
            target_dict = {
                "name": self.ent_name.get().strip(),
                "url": self.ent_url.get().strip(),
                "method": self.cmb_method.get().strip().upper() or "GET",
                "weight": float(self.ent_weight.get().strip() or "1.0"),
                "timeout_s": float(self.ent_timeout.get().strip() or "10.0"),
                "headers": parse_header_lines(self.txt_headers.get("1.0", "end-1c")),
                "body": self.txt_body.get("1.0", "end-1c") or None,
            }
            self.result = TargetConfig.from_dict(target_dict)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("오류", str(exc))
            return
        self.window.destroy()

    def show(self) -> Optional[TargetConfig]:
        self.window.wait_window()
        return self.result


class FreyaGUI:
    def __init__(self, root: tk.Tk):
        if tk is None or ttk is None:
            raise RuntimeError("tkinter를 사용할 수 없습니다.")
        self.root = root
        self.root.title("Freya")
        self.root.minsize(1024, 720)

        self.targets: List[TargetConfig] = []
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.summary_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.log_buffer: List[str] = []
        self.last_summary: Optional[Dict[str, Any]] = None

        outer = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)

        scenario_frame = ttk.Labelframe(outer, text="시나리오 설정")
        scenario_frame.pack(fill="x", pady=(0, 12))

        self.var_duration = tk.StringVar(value="60")
        self.var_rps = tk.StringVar(value="20")
        self.var_concurrency = tk.StringVar(value="20")

        ttk.Label(scenario_frame, text="실행 시간(초)").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(scenario_frame, textvariable=self.var_duration, width=10).grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(scenario_frame, text="총 RPS").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(scenario_frame, textvariable=self.var_rps, width=10).grid(row=0, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(scenario_frame, text="동시 작업자").grid(row=0, column=4, sticky="w", padx=4, pady=4)
        ttk.Entry(scenario_frame, textvariable=self.var_concurrency, width=10).grid(row=0, column=5, sticky="w", padx=4, pady=4)

        scenario_frame.grid_columnconfigure(6, weight=1)

        targets_frame = ttk.Labelframe(outer, text="타겟 구성")
        targets_frame.pack(fill="both", expand=True, pady=(0, 12))

        columns = ("name", "method", "weight", "url")
        self.tree_targets = ttk.Treeview(targets_frame, columns=columns, show="headings", height=6)
        self.tree_targets.heading("name", text="이름")
        self.tree_targets.heading("method", text="메서드")
        self.tree_targets.heading("weight", text="가중치")
        self.tree_targets.heading("url", text="URL")
        self.tree_targets.column("name", width=140, anchor="center")
        self.tree_targets.column("method", width=80, anchor="center")
        self.tree_targets.column("weight", width=80, anchor="center")
        self.tree_targets.column("url", width=480, anchor="w")
        self.tree_targets.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=8)

        scrollbar = ttk.Scrollbar(targets_frame, orient="vertical", command=self.tree_targets.yview)
        self.tree_targets.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y", pady=8)

        btns = ttk.Frame(outer)
        btns.pack(fill="x", pady=(0, 12))
        ttk.Button(btns, text="추가", command=self.add_target).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="수정", command=self.edit_target).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="삭제", command=self.remove_target).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="시나리오 저장", command=self.save_scenario).pack(side="right", padx=(8, 0))
        ttk.Button(btns, text="시나리오 불러오기", command=self.load_scenario).pack(side="right", padx=(8, 0))

        log_frame = ttk.Labelframe(outer, text="로그")
        log_frame.pack(fill="both", expand=True)

        self.txt_log = tk.Text(log_frame, height=18, wrap="none", state="disabled", background="#1e1e1e", foreground="#d0d0d0")
        self.txt_log.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=8)
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y", pady=8)

        status_frame = ttk.Frame(outer)
        status_frame.pack(fill="x", pady=(12, 0))

        self.lbl_status = ttk.Label(status_frame, text="대기")
        self.lbl_status.pack(side="left")

        controls = ttk.Frame(status_frame)
        controls.pack(side="right")
        self.btn_start = ttk.Button(controls, text="시작", width=14, command=self.start_test)
        self.btn_start.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(controls, text="중지", width=14, command=self.stop_test, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        self.btn_show_summary = ttk.Button(controls, text="요약 보기", width=14, command=self.show_summary_window, state="disabled")
        self.btn_show_summary.pack(side="left", padx=4)
        self.btn_export_summary = ttk.Button(controls, text="요약 저장", width=14, command=self.export_summary, state="disabled")
        self.btn_export_summary.pack(side="left", padx=4)
        self.btn_export_log = ttk.Button(controls, text="로그 저장", width=14, command=self.export_log, state="disabled")
        self.btn_export_log.pack(side="left", padx=4)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._poll_queues)

    def add_target(self) -> None:
        dialog = TargetDialog(self.root)
        result = dialog.show()
        if result:
            self.targets.append(result)
            self.refresh_targets()

    def edit_target(self) -> None:
        selection = self.tree_targets.selection()
        if not selection:
            messagebox.showinfo("알림", "수정할 타겟을 선택하세요.")
            return
        index = int(selection[0])
        dialog = TargetDialog(self.root, self.targets[index])
        result = dialog.show()
        if result:
            self.targets[index] = result
            self.refresh_targets()

    def remove_target(self) -> None:
        selection = self.tree_targets.selection()
        if not selection:
            messagebox.showinfo("알림", "삭제할 타겟을 선택하세요.")
            return
        index = int(selection[0])
        del self.targets[index]
        self.refresh_targets()

    def refresh_targets(self) -> None:
        self.tree_targets.delete(*self.tree_targets.get_children())
        for idx, target in enumerate(self.targets):
            self.tree_targets.insert("", "end", iid=str(idx), values=(target.name, target.method, target.weight, target.url))

    def load_scenario(self) -> None:
        if filedialog is None:
            messagebox.showerror("오류", "파일 다이얼로그를 사용할 수 없습니다.")
            return
        path = filedialog.askopenfilename(title="시나리오 불러오기", filetypes=[("JSON", "*.json"), ("모든 파일", "*.*")])
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            scenario = LoadScenario.from_dict(payload)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("오류", f"시나리오를 읽을 수 없습니다: {exc}")
            return
        self.var_duration.set(str(scenario.duration_s))
        self.var_rps.set(str(scenario.global_rps))
        self.var_concurrency.set(str(scenario.concurrency))
        self.targets = list(scenario.targets)
        self.refresh_targets()
        self.lbl_status.config(text=f"시나리오 로드 완료: {Path(path).name}")

    def save_scenario(self) -> None:
        if filedialog is None:
            messagebox.showerror("오류", "파일 다이얼로그를 사용할 수 없습니다.")
            return
        scenario = self.collect_scenario(validate_targets=False)
        if scenario is None:
            return
        path = filedialog.asksaveasfilename(
            title="시나리오 저장",
            defaultextension=".json",
            initialfile="freya-scenario.json",
        )
        if not path:
            return
        data = {
            "duration_s": scenario.duration_s,
            "global_rps": scenario.global_rps,
            "concurrency": scenario.concurrency,
            "targets": [
                {
                    "name": t.name,
                    "url": t.url,
                    "method": t.method,
                    "weight": t.weight,
                    "timeout_s": t.timeout_s,
                    "headers": t.headers,
                    "body": t.body,
                }
                for t in scenario.targets
            ],
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.lbl_status.config(text=f"시나리오 저장: {path}")

    def collect_scenario(self, validate_targets: bool = True) -> Optional[LoadScenario]:
        try:
            duration = float(self.var_duration.get().strip())
            rps = float(self.var_rps.get().strip())
            concurrency = int(self.var_concurrency.get().strip())
            if duration <= 0 or rps <= 0 or concurrency <= 0:
                raise ValueError
        except Exception:  # noqa: BLE001
            messagebox.showerror("오류", "시나리오 설정 값을 확인하세요.")
            return None
        if validate_targets and not self.targets:
            messagebox.showerror("오류", "타겟을 최소 1개 이상 추가하세요.")
            return None
        try:
            scenario_dict = {
                "duration_s": duration,
                "global_rps": rps,
                "concurrency": concurrency,
                "targets": [
                    {
                        "name": t.name,
                        "url": t.url,
                        "method": t.method,
                        "weight": t.weight,
                        "timeout_s": t.timeout_s,
                        "headers": t.headers,
                        "body": t.body,
                    }
                    for t in (self.targets if validate_targets else (self.targets or []))
                ],
            }
            return LoadScenario.from_dict(scenario_dict)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("오류", f"시나리오 구성이 유효하지 않습니다: {exc}")
            return None

    def start_test(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("안내", "테스트가 이미 진행 중입니다.")
            return
        scenario = self.collect_scenario()
        if scenario is None:
            return

        self.stop_flag = threading.Event()
        self.log_buffer = []
        self.last_summary = None
        self.log_queue = queue.Queue()
        self.summary_queue = queue.Queue()

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_show_summary.config(state="disabled")
        self.btn_export_summary.config(state="disabled")
        self.btn_export_log.config(state="disabled")
        self.lbl_status.config(text="실행 중")
        self.txt_log.configure(state="normal")
        self.txt_log.delete("1.0", "end")
        self.txt_log.configure(state="disabled")

        def log_handler(msg: str) -> None:
            self.log_buffer.append(msg)
            self.log_queue.put(msg)

        def runner() -> None:
            try:
                summary = asyncio.run(
                    run_load_test(
                        scenario,
                        print_log=False,
                        log_file=None,
                        summary_file=None,
                        log_callback=log_handler,
                        stop_flag=self.stop_flag,
                    )
                )
                self.summary_queue.put(summary)
            except Exception as exc:  # noqa: BLE001
                self.summary_queue.put({"__error__": str(exc)})

        self.worker = threading.Thread(target=runner, daemon=True)
        self.worker.start()

    def stop_test(self) -> None:
        if self.worker and self.worker.is_alive():
            self.stop_flag.set()
            self.lbl_status.config(text="중지 요청")

    def _poll_queues(self) -> None:
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.txt_log.configure(state="normal")
            self.txt_log.insert("end", msg + "\n")
            self.txt_log.see("end")
            self.txt_log.configure(state="disabled")

        summary = None
        try:
            summary = self.summary_queue.get_nowait()
        except queue.Empty:
            pass
        if summary is not None:
            if "__error__" in summary:
                messagebox.showerror("오류", summary["__error__"])
                self.lbl_status.config(text="오류 발생")
            else:
                self.last_summary = summary
                self.lbl_status.config(text="완료")
                self.btn_show_summary.config(state="normal")
                self.btn_export_summary.config(state="normal")
                self.btn_export_log.config(state="normal" if self.log_buffer else "disabled")
                self.show_summary_window()
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.worker = None

        self.root.after(100, self._poll_queues)

    def show_summary_window(self) -> None:
        if not self.last_summary:
            messagebox.showinfo("안내", "표시할 요약 정보가 없습니다.")
            return
        summary = self.last_summary
        win = tk.Toplevel(self.root)
        win.title("요약")
        win.resizable(True, True)
        frame = ttk.Frame(win, padding=16)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text=f"총 요청 수: {summary.get('total_sent', 0)}").pack(anchor="w")
        ttk.Label(frame, text=f"경과 시간: {summary.get('elapsed_s', 0)}s").pack(anchor="w")

        tree = ttk.Treeview(frame, columns=("target", "status", "count"), show="headings", height=8)
        tree.heading("target", text="타겟")
        tree.heading("status", text="상태")
        tree.heading("count", text="건수")
        tree.column("target", width=160, anchor="center")
        tree.column("status", width=120, anchor="center")
        tree.column("count", width=100, anchor="center")

        for target, data in summary.get("targets", {}).items():
            status_counts = data.get("status_counts", {})
            for status, count in status_counts.items():
                tree.insert("", "end", values=(target, status, count))
        tree.pack(fill="both", expand=True, pady=8)

        latency_box = ttk.Labelframe(frame, text="지연 통계")
        latency_box.pack(fill="both", expand=True)

        for target, data in summary.get("targets", {}).items():
            latency = data.get("latency", {})
            if latency.get("count", 0):
                ttk.Label(
                    latency_box,
                    text=(
                        f"{target}: 평균 {latency.get('average_ms', 0):.2f} ms "
                        f"(최소 {latency.get('min_ms', 0):.2f} / 최대 {latency.get('max_ms', 0):.2f})"
                    ),
                ).pack(anchor="w", padx=8, pady=2)
            else:
                ttk.Label(latency_box, text=f"{target}: 데이터 없음").pack(anchor="w", padx=8, pady=2)

        ttk.Button(frame, text="닫기", command=win.destroy).pack(anchor="e", pady=(12, 0))

    def export_summary(self) -> None:
        if not self.last_summary:
            messagebox.showinfo("안내", "저장할 요약이 없습니다.")
            return
        if filedialog is None:
            messagebox.showerror("오류", "파일 다이얼로그를 사용할 수 없습니다.")
            return
        path = filedialog.asksaveasfilename(
            title="요약 저장",
            defaultextension=".json",
            initialfile="freya-summary.json",
        )
        if not path:
            return
        Path(path).write_text(json.dumps(self.last_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self.lbl_status.config(text=f"요약 저장: {path}")

    def export_log(self) -> None:
        if not self.log_buffer:
            messagebox.showinfo("안내", "로그 데이터가 없습니다.")
            return
        if filedialog is None:
            messagebox.showerror("오류", "파일 다이얼로그를 사용할 수 없습니다.")
            return
        path = filedialog.asksaveasfilename(
            title="로그 저장",
            defaultextension=".log",
            initialfile="freya-log.log",
        )
        if not path:
            return
        Path(path).write_text("\n".join(self.log_buffer), encoding="utf-8")
        self.lbl_status.config(text=f"로그 저장: {path}")

    def on_close(self) -> None:
        if self.worker and self.worker.is_alive():
            if messagebox.askyesno("종료 확인", "실행 중인 테스트가 있습니다. 종료하시겠습니까?"):
                self.stop_flag.set()
            else:
                return
        self.root.destroy()


def launch_gui() -> None:
    if tk is None or ttk is None:
        raise SystemExit("GUI 실행을 위해 tkinter가 필요합니다. (macOS의 경우 `brew install python-tk`)")
    root = tk.Tk()
    FreyaGUI(root)
    root.mainloop()


# ------------------------------------------------------------------------------
# Simulator
# ------------------------------------------------------------------------------


@dataclass
class SimRoute:
    path: str
    method: str
    status: int
    payload: Any
    min_latency_ms: float = 10.0
    max_latency_ms: float = 50.0
    error_rate: float = 0.0
    error_status: int = 500
    error_payload: Any = field(default_factory=lambda: {"error": "simulated failure"})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimRoute":
        path = str(data.get("path", "/"))
        method = str(data.get("method", "GET")).upper()
        status = int(data.get("status", 200))
        payload = data.get("payload", {"ok": True})
        min_latency = float(data.get("min_latency_ms", 10.0))
        max_latency = float(data.get("max_latency_ms", 50.0))
        if max_latency < min_latency:
            raise ValueError(f"simulate route '{path}' latency 범위가 올바르지 않습니다")
        error_rate = float(data.get("error_rate", 0.0))
        error_status = int(data.get("error_status", 500))
        error_payload = data.get("error_payload", {"error": "simulated failure"})
        return cls(
            path=path,
            method=method,
            status=status,
            payload=payload,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            error_rate=error_rate,
            error_status=error_status,
            error_payload=error_payload,
        )


@dataclass
class SimulationConfig:
    host: str
    port: int
    routes: List[SimRoute]

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SimulationConfig":
        if args.routes:
            data = json.loads(Path(args.routes).read_text(encoding="utf-8"))
            routes = [SimRoute.from_dict(entry) for entry in data.get("routes", [])]
        else:
            routes = [
                SimRoute(
                    path="/health",
                    method="GET",
                    status=200,
                    payload={"status": "ok"},
                    min_latency_ms=10,
                    max_latency_ms=40,
                ),
                SimRoute(
                    path="/orders",
                    method="POST",
                    status=201,
                    payload={"orderId": 123},
                    min_latency_ms=50,
                    max_latency_ms=250,
                    error_rate=0.1,
                ),
            ]
        return cls(host=args.host, port=args.port, routes=routes)


async def start_simulation_server(config: SimulationConfig) -> None:
    app = web.Application()

    for route in config.routes:
        async def handler(request: web.Request, cfg: SimRoute = route) -> web.Response:
            latency_ms = random.uniform(cfg.min_latency_ms, cfg.max_latency_ms)
            await asyncio.sleep(latency_ms / 1000.0)
            if cfg.error_rate > 0 and random.random() < cfg.error_rate:
                return web.json_response(cfg.error_payload, status=cfg.error_status)
            return web.json_response(cfg.payload, status=cfg.status)

        app.router.add_route(route.method, route.path, handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=config.host, port=config.port)
    await site.start()

    print(f"Freya 시뮬레이터 실행 중: http://{config.host}:{config.port}")
    for entry in config.routes:
        print(
            f"  {entry.method} {entry.path} -> status {entry.status} "
            f"latency {entry.min_latency_ms}-{entry.max_latency_ms}ms error_rate {entry.error_rate}"
        )

    stop_event = asyncio.Event()

    def _handle_signal(*_: Any) -> None:
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, sig_name):
            loop.add_signal_handler(getattr(signal, sig_name), _handle_signal)

    try:
        await stop_event.wait()
    finally:
        await runner.cleanup()


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freya parallel load tester")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="실제 엔드포인트에 부하 테스트 수행")
    run_parser.add_argument("--scenario", help="시나리오 JSON 파일 경로")
    run_parser.add_argument("--url", help="단일 타겟 URL (시나리오 없이)")
    run_parser.add_argument("--name", help="단일 타겟 이름", default="default")
    run_parser.add_argument("--method", help="HTTP 메서드", default="GET")
    run_parser.add_argument("--headers", help="헤더 텍스트 (Name: Value 줄 구분)")
    run_parser.add_argument("--body", help="요청 본문 텍스트")
    run_parser.add_argument("--timeout", type=float, default=10.0, help="요청 타임아웃 (초)")
    run_parser.add_argument("--duration", type=float, default=60.0, help="테스트 실행 시간 (초)")
    run_parser.add_argument("--rps", type=float, default=20.0, help="초당 총 요청 수")
    run_parser.add_argument("--concurrency", type=int, help="동시 연결 수 (기본: rps)")
    run_parser.add_argument("--print-log", action="store_true", help="실시간 로그 출력")
    run_parser.add_argument("--log-file", help="로그 저장 경로")
    run_parser.add_argument("--summary-json", help="요약 JSON 저장 경로")

    sim_parser = subparsers.add_parser("simulate", help="간단한 가상 서비스 실행")
    sim_parser.add_argument("--host", default="127.0.0.1", help="바인딩 호스트")
    sim_parser.add_argument("--port", type=int, default=8080, help="바인딩 포트")
    sim_parser.add_argument("--routes", help="라우트 설정 JSON 파일 경로")

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    if argv is None and len(sys.argv) == 1:
        launch_gui()
        return

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "run":
        try:
            scenario = LoadScenario.from_args(args)
        except Exception as exc:  # noqa: BLE001
            parser.error(str(exc))
            return

        log_path = Path(args.log_file).expanduser() if args.log_file else None
        summary_path = Path(args.summary_json).expanduser() if args.summary_json else None

        try:
            summary = asyncio.run(
                run_load_test(
                    scenario,
                    print_log=args.print_log,
                    log_file=log_path,
                    summary_file=summary_path,
                    log_callback=print if args.print_log else None,
                )
            )
        except KeyboardInterrupt:
            print("사용자 중단")
            sys.exit(130)

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "simulate":
        config = SimulationConfig.from_args(args)
        try:
            asyncio.run(start_simulation_server(config))
        except KeyboardInterrupt:
            print("시뮬레이터 종료")
        return


if __name__ == "__main__":
    main()
