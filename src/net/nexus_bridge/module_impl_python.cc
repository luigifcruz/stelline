#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_python.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr const char* kNexusBridgePythonCode = R"NEXUSPY(
import asyncio
import os
import queue
import threading
from urllib.parse import urlparse, urlunparse

OTL_METADATA_QUERY = "queries/observatory:getMetadata"
DEFAULT_CONVEX_URL = (
    <<<NEXUS_URL>>>
)
RETRY_SECONDS = float(os.getenv("NEXUS_OTL_WATCH_RETRY_SECONDS", "5"))
STATUS_ENV_KEY = "nexus.bridge"

_deltas = queue.SimpleQueue()
_status_events = queue.SimpleQueue()
_stop = threading.Event()
_state_lock = threading.Lock()
_connected = False
_variable_count = 0
_last_error = ""
_last_status_snapshot = None
_last_published_status = None


def _convex_url():
    raw = DEFAULT_CONVEX_URL
    parsed = urlparse(raw.strip())
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", "")).rstrip("/")


def _snapshot_status():
    return {
        "connected": _connected,
        "variables_loaded": _variable_count,
        "url": _convex_url(),
        "last_error": _last_error,
    }


def _set_status(connected=None, variable_count=None, error=None):
    global _connected, _variable_count, _last_error, _last_status_snapshot

    with _state_lock:
        if connected is not None:
            _connected = bool(connected)
        if variable_count is not None:
            _variable_count = int(variable_count)
        if error is not None:
            _last_error = str(error)
        snapshot = _snapshot_status()
        if snapshot == _last_status_snapshot:
            return
        _last_status_snapshot = dict(snapshot)

    _status_events.put(snapshot)


def _drain(source):
    while True:
        try:
            yield source.get_nowait()
        except queue.Empty:
            return


def _watch():
    stop = _stop
    deltas = _deltas
    retry = RETRY_SECONDS
    query = OTL_METADATA_QUERY
    run = asyncio.run
    url = _convex_url()
    known = {}

    def set_disconnected(error):
        _set_status(connected=False, variable_count=len(known), error=error)

    while not stop.is_set():
        try:
            from convex import ConvexClient
            from convex.values import ConvexInt64
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"watcher error: {error}")
            set_disconnected(error)
            stop.wait(retry)
            continue

        def clean(value):
            if isinstance(value, ConvexInt64):
                return value.value
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(item) for item in value]
            return value

        def entries_by_key(snapshot):
            return {
                entry["key"]: clean({
                    "value": entry.get("value"),
                    "type": entry.get("type"),
                    "valid": entry.get("valid"),
                })
                for entry in snapshot.get("data", [])
                if isinstance(entry, dict) and entry.get("key")
            }

        async def stream():
            client = ConvexClient(url)
            subscription = client.subscribe(query, {})
            print(f"subscribed to {query} at {url}")
            async for snapshot in subscription:
                if stop.is_set():
                    return
                entries = entries_by_key(snapshot)
                changed = {key: entry for key, entry in entries.items() if known.get(key) != entry}
                removed = [key for key in known if key not in entries]
                if changed or removed:
                    deltas.put((changed, removed))
                known.clear()
                known.update(entries)
                _set_status(connected=True, variable_count=len(known), error="")

        try:
            run(stream())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"watcher error: {error}")
            set_disconnected(error)
        stop.wait(retry)


_set_status(connected=False, variable_count=0, error="")
_watcher = threading.Thread(target=_watch, name="nexus-env-watcher", daemon=True)
_watcher.start()


def compute(ctx):
    global _last_published_status

    changed = {}
    removed = set()
    for delta_changed, delta_removed in _drain(_deltas):
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

    status = None
    for candidate in _drain(_status_events):
        status = candidate
    if status is not None and status != _last_published_status:
        ctx.env[STATUS_ENV_KEY] = status
        _last_published_status = dict(status)


def cleanup():
    _stop.set()
    _watcher.join(timeout=1.0)
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
