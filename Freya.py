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
import random
import signal
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import aiohttp
    from aiohttp import web
except ImportError as exc:  # noqa: F401
    raise SystemExit(
        "필수 라이브러리 'aiohttp'가 설치되어 있지 않습니다.\n"
        "다음 명령으로 설치 후 다시 실행하세요:\n"
        "    pip install aiohttp>=3.9"
    ) from exc


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
) -> Dict[str, Any]:
    metrics = MetricsRecorder()
    log_lines: List[str] = []
    stop_event = asyncio.Event()

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    connector = aiohttp.TCPConnector(ssl=False, limit=None)
    timeout = aiohttp.ClientTimeout(total=None)

    queue: asyncio.Queue[Optional[TargetConfig]] = asyncio.Queue()

    async def producer() -> None:
        interval = 1.0 / scenario.global_rps
        loop = asyncio.get_running_loop()
        end_time = loop.time() + scenario.duration_s
        next_tick = loop.time()
        while loop.time() < end_time and not stop_event.is_set():
            target = weighted_choice(scenario.targets)
            await queue.put(target)
            next_tick += interval
            delay = max(0.0, next_tick - loop.time())
            await asyncio.sleep(delay)
        for _ in range(scenario.concurrency):
            await queue.put(None)

    async def worker(index: int, session: aiohttp.ClientSession) -> None:
        while True:
            target = await queue.get()
            if target is None:
                queue.task_done()
                break
            outcome = await execute_request(session, target)
            metrics.record(outcome)
            stamp = datetime.now().isoformat(timespec="milliseconds")
            status = outcome.status if outcome.status is not None else "ERR"
            msg = f"{stamp} worker={index:02d} target={target.name} status={status} latency_ms={outcome.latency_ms:.2f}"
            if outcome.error:
                msg += f" error={outcome.error}"
            if print_log:
                print(msg)
            log_lines.append(msg)
            queue.task_done()

    started = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        workers = [asyncio.create_task(worker(i, session)) for i in range(1, scenario.concurrency + 1)]
        producer_task = asyncio.create_task(producer())
        try:
            await asyncio.gather(producer_task, queue.join())
        finally:
            stop_event.set()
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
