#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_python.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr const char* kNexusBridgePythonCode = R"NEXUSPY(
import asyncio
import math
import os
import queue
import threading
import time
from urllib.parse import urlparse, urlunparse

METADATA_QUERY = "queries/observatory:getMetadata"
INSTANCE_METRICS_MUTATION = "mutations/metrics:publishInstanceMetrics"
DEFAULT_CONVEX_URL = (
    <<<NEXUS_URL>>>
)
METADATA_RETRY_SECONDS = float(os.getenv("NEXUS_OTL_WATCH_RETRY_SECONDS", "5"))
INSTANCE_ID = os.getenv("NEXUS_INSTANCE_ID", "").strip()
METRICS_INTERVAL_SECONDS = 5.0
METRICS_FORMAT_PREFIX = "private-stelline-metrics-"
STATUS_ENV_KEY = "nexus.bridge"

def _convex_url():
    raw = DEFAULT_CONVEX_URL
    parsed = urlparse(raw.strip())
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", "")).rstrip("/")


def _drain(source):
    while True:
        try:
            yield source.get_nowait()
        except queue.Empty:
            return


def _clean(value, convex_int64):
    if isinstance(value, convex_int64):
        return value.value
    if isinstance(value, dict):
        return {key: _clean(item, convex_int64) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item, convex_int64) for item in value]
    return value


def _environment_value(entry, convex_int64):
    value = _clean(entry.get("value"), convex_int64)
    value_type = entry.get("type")
    if value is None:
        return None
    if value_type in {"i32", "i64", "u32", "u64"}:
        return str(int(value))
    if value_type in {"vec<i32>", "vec<i64>", "vec<u32>", "vec<u64>"}:
        return [str(int(item)) for item in value]
    return value


def _metadata_entries_by_key(snapshot, convex_int64):
    return {
        entry["key"]: _clean({
            "value": _environment_value(entry, convex_int64),
            "type": entry.get("type"),
            "valid": entry.get("valid"),
        }, convex_int64)
        for entry in snapshot.get("data", [])
        if isinstance(entry, dict) and entry.get("key")
    }


def _normalize_metric(entry):
    if not isinstance(entry, dict):
        return None

    value = entry.get("value")
    metric_format = entry.get("format")
    if metric_format == f"{METRICS_FORMAT_PREFIX}number":
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number):
            return None
        return {"type": "number", "value": number}
    if metric_format == f"{METRICS_FORMAT_PREFIX}string" and isinstance(value, str):
        return {"type": "text", "value": value}
    return None


def _build_metrics_snapshot(ctx):
    metrics = {}
    for block_name, block_values in ctx.metrics.items():
        if not isinstance(block_values, dict):
            continue
        values = {}
        for name, entry in block_values.items():
            normalized = _normalize_metric(entry)
            if normalized is not None:
                values[name] = normalized
        if values:
            metrics[block_name] = values
    return metrics


class _NexusBridge:
    def __init__(self):
        self._pending_metadata_updates = queue.SimpleQueue()
        self._pending_metrics_snapshots = queue.Queue(maxsize=1)
        self._pending_statuses = queue.SimpleQueue()
        self._stop_event = threading.Event()
        self._status_lock = threading.Lock()
        self._connected = False
        self._variable_count = 0
        self._metrics_monitored = 0
        self._last_error = ""
        self._last_recorded_status = None
        self._last_applied_status = None
        self._last_metrics_sample_at = 0.0
        self._metrics_subscription_started = False
        self._metadata_thread = None
        self._metrics_thread = None

    def start(self):
        self._record_status(connected=False, variable_count=0, error="")
        self._metadata_thread = threading.Thread(
            target=self._metadata_watcher_loop,
            name="nexus-env-watcher",
            daemon=True,
        )
        self._metadata_thread.start()
        if INSTANCE_ID:
            self._metrics_thread = threading.Thread(
                target=self._metrics_publisher_loop,
                name="nexus-metrics-publisher",
                daemon=True,
            )
            self._metrics_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._metadata_thread is not None:
            self._metadata_thread.join(timeout=1.0)
        if self._metrics_thread is not None:
            self._metrics_thread.join(timeout=1.0)

    def compute(self, ctx):
        self._ensure_metrics_subscription(ctx)
        self._apply_metadata_updates(ctx)
        self._queue_metrics_snapshot(ctx)
        self._apply_status_update(ctx)

    # Metadata synchronization

    def _metadata_watcher_loop(self):
        url = _convex_url()
        known = {}

        while not self._stop_event.is_set():
            try:
                from convex import ConvexClient
                from convex.values import ConvexInt64
            except Exception as exc:
                self._record_metadata_error(exc, len(known))
                self._stop_event.wait(METADATA_RETRY_SECONDS)
                continue

            try:
                asyncio.run(self._stream_metadata(ConvexClient, ConvexInt64, url, known))
            except Exception as exc:
                self._record_metadata_error(exc, len(known))
            self._stop_event.wait(METADATA_RETRY_SECONDS)

    async def _stream_metadata(self, convex_client, convex_int64, url, known):
        client = convex_client(url)
        query_args = {"instanceId": INSTANCE_ID} if INSTANCE_ID else {}
        subscription = client.subscribe(METADATA_QUERY, query_args)
        print(f"subscribed to {METADATA_QUERY} at {url}")
        async for snapshot in subscription:
            if self._stop_event.is_set():
                return
            entries = _metadata_entries_by_key(snapshot, convex_int64)
            changed = {
                key: entry
                for key, entry in entries.items()
                if known.get(key) != entry
            }
            removed = [key for key in known if key not in entries]
            if changed or removed:
                self._pending_metadata_updates.put((changed, removed))
            known.clear()
            known.update(entries)
            self._record_status(connected=True, variable_count=len(known), error="")

    def _record_metadata_error(self, exc, variable_count):
        error = f"{type(exc).__name__}: {exc}"
        print(f"watcher error: {error}")
        self._record_status(
            connected=False,
            variable_count=variable_count,
            error=error,
        )

    def _apply_metadata_updates(self, ctx):
        changed = {}
        removed = set()
        for delta_changed, delta_removed in _drain(self._pending_metadata_updates):
            for key, entry in delta_changed.items():
                changed[key] = entry
                removed.discard(key)
            for key in delta_removed:
                changed.pop(key, None)
                removed.add(key)

        if changed:
            ctx.env.update(changed)
        for key in removed:
            ctx.env.pop(key, None)
        if changed or removed:
            print(f"applied {len(changed)} changed, {len(removed)} removed")

    # Bridge status

    def _status_snapshot(self):
        return {
            "connected": self._connected,
            "variables_loaded": self._variable_count,
            "metrics_monitored": self._metrics_monitored,
            "url": _convex_url(),
            "last_error": self._last_error,
        }

    def _record_status(
        self,
        connected=None,
        variable_count=None,
        metrics_monitored=None,
        error=None,
    ):
        with self._status_lock:
            if connected is not None:
                self._connected = bool(connected)
            if variable_count is not None:
                self._variable_count = int(variable_count)
            if metrics_monitored is not None:
                self._metrics_monitored = int(metrics_monitored)
            if error is not None:
                self._last_error = str(error)
            snapshot = self._status_snapshot()
            if snapshot == self._last_recorded_status:
                return
            self._last_recorded_status = dict(snapshot)

        self._pending_statuses.put(snapshot)

    def _apply_status_update(self, ctx):
        status = None
        for candidate in _drain(self._pending_statuses):
            status = candidate
        if status is not None and status != self._last_applied_status:
            ctx.env[STATUS_ENV_KEY] = status
            self._last_applied_status = dict(status)

    # Metrics reporting

    def _ensure_metrics_subscription(self, ctx):
        if not self._metrics_subscription_started:
            ctx.metrics.subscribe_all()
            self._metrics_subscription_started = True

    def _metrics_publisher_loop(self):
        client = None
        while not self._stop_event.is_set():
            try:
                snapshot = self._pending_metrics_snapshots.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                from convex import ConvexClient

                if client is None:
                    client = ConvexClient(_convex_url())
                client.mutation(INSTANCE_METRICS_MUTATION, {
                    "instanceId": INSTANCE_ID,
                    "timestamp": snapshot["timestamp"],
                    "metrics": snapshot["metrics"],
                })
            except Exception as exc:
                client = None
                print(f"metrics publisher error: {type(exc).__name__}: {exc}")

    def _queue_metrics_snapshot(self, ctx):
        now = time.monotonic()
        if self._metrics_thread is None:
            return
        if now - self._last_metrics_sample_at < METRICS_INTERVAL_SECONDS:
            return

        self._last_metrics_sample_at = now
        metrics = _build_metrics_snapshot(ctx)
        self._record_status(
            metrics_monitored=sum(len(values) for values in metrics.values()),
        )
        if not metrics:
            return

        metric_snapshot = {
            "timestamp": int(time.time() * 1000),
            "metrics": metrics,
        }
        for _ in _drain(self._pending_metrics_snapshots):
            pass
        try:
            self._pending_metrics_snapshots.put_nowait(metric_snapshot)
        except queue.Full:
            pass


_bridge = _NexusBridge()


def compute(ctx):
    _bridge.compute(ctx)


def cleanup():
    _bridge.stop()


_bridge.start()
)NEXUSPY";

}  // namespace

struct NexusBridgeImplPython : public NexusBridgeImpl,
                               public PythonRuntimeContext,
                               public Scheduler::Context {
    Result create() final;
    Result destroy() final;
    Result reconfigure() final;
    Result computeSubmit() final;
};

Result NexusBridgeImplPython::create() {
    JST_CHECK(NexusBridgeImpl::create());
    JST_CHECK(createCompute(kNexusBridgePythonCode,
                            {{"NEXUS_URL", jst::fmt::format("\"{}\"", url)}},
                            {},
                            inputs(),
                            {},
                            outputs(),
                            environment(),
                            view()));

    return Result::SUCCESS;
}

Result NexusBridgeImplPython::destroy() {
    JST_CHECK(destroyCompute());
    JST_CHECK(NexusBridgeImpl::destroy());

    return Result::SUCCESS;
}

Result NexusBridgeImplPython::reconfigure() {
    auto config = *candidate();
    if (config.url != url) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result NexusBridgeImplPython::computeSubmit() {
    const auto result = PythonRuntimeContext::computeSubmit();
    refreshStatus();

    return result;
}

JST_REGISTER_MODULE(NexusBridgeImplPython, DeviceType::CPU, RuntimeType::PYTHON, "generic");

}  // namespace Jetstream::Modules
