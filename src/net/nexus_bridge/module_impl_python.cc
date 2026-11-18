// TODO: Rewrite in C++.

#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_python.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr const char* kNexusBridgePythonCode = R"NEXUSPY(
import collections
import json
import math
import os
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from urllib.parse import quote, urlparse, urlunparse

DEFAULT_NEXUS_URL = (
    <<<NEXUS_URL>>>
)
METADATA_POLL_SECONDS = 5.0
METADATA_RETRY_SECONDS = float(os.getenv("NEXUS_OTL_WATCH_RETRY_SECONDS", "5"))
INSTANCE_ID = os.getenv("NEXUS_INSTANCE_ID", "").strip()
INSTANCE_CREDENTIALS_FILE = os.getenv("NEXUS_INSTANCE_CREDENTIALS_FILE", "").strip()
INSTANCE_CREDENTIALS = os.getenv("NEXUS_INSTANCE_CREDENTIALS", "").strip()
TELEMETRY_URL = os.getenv("NEXUS_TELEMETRY_URL", "").strip()
REQUEST_TIMEOUT_SECONDS = 10.0
METRICS_INTERVAL_SECONDS = 5.0
METRICS_RETRY_SECONDS = 5.0
MAX_PENDING_METRIC_SAMPLES = 256
METRICS_BATCH_SAMPLES = 16
MAX_METRICS_BATCH_BYTES = 768 * 1024
MAX_SAMPLE_METRICS_BYTES = 512 * 1024
JAVASCRIPT_NUMBER_MAX_LENGTH = 24
METRICS_FORMAT_PREFIX = "stelline-metrics-"
STATUS_ENV_KEY = "nexus.bridge"


def _telemetry_url():
    if TELEMETRY_URL:
        return TELEMETRY_URL.rstrip("/")
    parsed = urlparse(DEFAULT_NEXUS_URL.strip())
    origin = urlunparse((parsed.scheme, parsed.netloc, "", "", "", "")).rstrip("/")
    return f"{origin}/telemetry"


def _instance_id():
    if not INSTANCE_ID:
        raise ValueError("NEXUS_INSTANCE_ID is required")
    return INSTANCE_ID


def _instance_credentials():
    if INSTANCE_CREDENTIALS_FILE:
        try:
            with open(INSTANCE_CREDENTIALS_FILE, encoding="utf-8") as source:
                token = source.read().strip()
        except OSError as exc:
            raise ValueError(f"NEXUS_INSTANCE_CREDENTIALS_FILE is unreadable: {exc.strerror}") from None
        if token:
            return token
    if INSTANCE_CREDENTIALS:
        return INSTANCE_CREDENTIALS
    raise ValueError("NEXUS_INSTANCE_CREDENTIALS_FILE or NEXUS_INSTANCE_CREDENTIALS is required")


class TelemetryError(Exception):
    def __init__(self, status, message):
        super().__init__(f"HTTP {status}: {message}" if status else message)
        self.status = status

    @property
    def retryable(self):
        return self.status is None or self.status >= 500 or self.status == 429


def _error_message(body):
    try:
        payload = json.loads(body)
    except ValueError:
        return body.decode("utf-8", "replace").strip()[:200]
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"]
    return str(payload)[:200]


def _encode_json(value):
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _javascript_json_length(value):
    # Nexus measures JSON.stringify(), which can print a number longer than Python does
    # (1e20 is 100000000000000000000), so allow the longest JavaScript form for each one.
    numbers = 0
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            numbers += 1
    return len(_encode_json(value)) + numbers * JAVASCRIPT_NUMBER_MAX_LENGTH


def _post_json(url, token, body):
    request = urllib.request.Request(
        url,
        data=_encode_json(body),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raise TelemetryError(exc.code, _error_message(exc.read())) from None
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise TelemetryError(None, f"{type(exc).__name__}: {reason}") from None


def _drain(source):
    while True:
        try:
            yield source.get_nowait()
        except queue.Empty:
            return


def _environment_value(entry):
    value = entry.get("value")
    value_type = entry.get("type")
    if value is None:
        return None
    if value_type in {"i32", "i64", "u32", "u64"}:
        return str(int(value))
    if value_type in {"vec<i32>", "vec<i64>", "vec<u32>", "vec<u64>"}:
        return [str(int(item)) for item in value]
    return value


def _metadata_entries_by_key(snapshot):
    return {
        entry["key"]: {
            "value": _environment_value(entry),
            "type": entry.get("type"),
            "valid": entry.get("valid"),
        }
        for entry in snapshot.get("data", [])
        if isinstance(entry, dict) and entry.get("key")
    }


def _normalize_metric(entry):
    if not isinstance(entry, dict):
        return None

    value = entry.get("value")
    metric_format = entry.get("format")
    if not isinstance(metric_format, dict):
        return None
    metric_type = metric_format.get("type")
    if metric_type == f"{METRICS_FORMAT_PREFIX}number":
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number):
            return None
        return {"type": "number", "value": number}
    if metric_type == f"{METRICS_FORMAT_PREFIX}string" and isinstance(value, str):
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
    def __init__(self, post=_post_json):
        self._post = post
        self._pending_metadata_updates = queue.SimpleQueue()
        self._metric_samples = collections.deque(maxlen=MAX_PENDING_METRIC_SAMPLES)
        self._metric_samples_lock = threading.Lock()
        self._metric_samples_ready = threading.Event()
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
            target=self._run_worker,
            args=(self._metadata_watcher_loop,),
            name="nexus-env-watcher",
            daemon=True,
        )
        self._metadata_thread.start()
        if INSTANCE_ID:
            self._metrics_thread = threading.Thread(
                target=self._run_worker,
                args=(self._metrics_publisher_loop,),
                name="nexus-metrics-publisher",
                daemon=True,
            )
            self._metrics_thread.start()

    def _run_worker(self, loop):
        # A request can outlive cleanup(), after which the runtime clears this module's globals.
        try:
            loop()
        except BaseException:
            if not self._stop_event.is_set():
                raise

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
        known = {}
        while not self._stop_event.is_set():
            try:
                self._poll_metadata(known)
                delay = METADATA_POLL_SECONDS
            except Exception as exc:
                if self._stop_event.is_set():
                    return
                self._record_metadata_error(exc, len(known))
                delay = METADATA_RETRY_SECONDS
            self._stop_event.wait(delay)

    def _poll_metadata(self, known):
        instance_id = _instance_id()
        snapshot = self._post(
            f"{_telemetry_url()}/v1/otl/metadata",
            _instance_credentials(),
            {"instanceId": instance_id},
        )
        if self._stop_event.is_set():
            return
        if not isinstance(snapshot, dict):
            raise ValueError("Metadata response is not an object")
        if snapshot.get("unchanged"):
            self._record_status(connected=True, variable_count=len(known), error="")
            return
        if not isinstance(snapshot.get("data"), list):
            raise ValueError("Metadata response has no data")
        entries = _metadata_entries_by_key(snapshot)
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
        print(f"Watcher error: {error}")
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
            print(f"Applied {len(changed)} changed, {len(removed)} removed.")

    # Bridge status

    def _status_snapshot(self):
        return {
            "connected": self._connected,
            "variables_loaded": self._variable_count,
            "metrics_monitored": self._metrics_monitored,
            "url": _telemetry_url(),
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

    def _enqueue_metric_sample(self, sample):
        with self._metric_samples_lock:
            self._metric_samples.append(sample)
        self._metric_samples_ready.set()

    def _next_metric_batch(self):
        batch = []
        size = len(_encode_json({"samples": []}))
        with self._metric_samples_lock:
            for sample in self._metric_samples:
                sample_size = len(_encode_json(sample)) + 1
                if batch and (len(batch) == METRICS_BATCH_SAMPLES or size + sample_size > MAX_METRICS_BATCH_BYTES):
                    break
                batch.append(sample)
                size += sample_size
        return batch

    def _discard_metric_batch(self, batch):
        sent = {sample["eventId"] for sample in batch}
        with self._metric_samples_lock:
            while self._metric_samples and self._metric_samples[0]["eventId"] in sent:
                self._metric_samples.popleft()

    def _metrics_publisher_loop(self):
        while not self._stop_event.is_set():
            batch = self._next_metric_batch()
            if not batch:
                self._metric_samples_ready.wait(0.5)
                self._metric_samples_ready.clear()
                continue

            try:
                self._publish_metrics(batch)
            except TelemetryError as exc:
                if self._stop_event.is_set():
                    return
                print(f"Metrics publisher error: {exc}")
                if exc.retryable:
                    self._stop_event.wait(METRICS_RETRY_SECONDS)
                    continue
            except Exception as exc:
                if self._stop_event.is_set():
                    return
                print(f"Metrics publisher error: {type(exc).__name__}: {exc}")
                self._stop_event.wait(METRICS_RETRY_SECONDS)
                continue
            self._discard_metric_batch(batch)

    def _publish_metrics(self, samples):
        instance_id = _instance_id()
        self._post(
            f"{_telemetry_url()}/v1/instances/{quote(instance_id, safe='')}/metrics",
            _instance_credentials(),
            {"samples": samples},
        )

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
        if _javascript_json_length(metrics) > MAX_SAMPLE_METRICS_BYTES:
            print("Metrics snapshot exceeds the Nexus sample limit; skipped.")
            return

        self._enqueue_metric_sample({
            "eventId": uuid.uuid4().hex,
            "timestamp": int(time.time() * 1000),
            "metrics": metrics,
        })


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
    Result loadCompute() final;
    Result computeSubmit() final;
};

Result NexusBridgeImplPython::create() {
    JST_CHECK(NexusBridgeImpl::create());
    JST_CHECK(loadCompute());

    return Result::SUCCESS;
}

Result NexusBridgeImplPython::loadCompute() {
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
