"""
Stelline Nexus client for metrics push and metadata pull.
"""

import json
import os
import threading
import urllib.request
from typing import Any

from stelline.utils import logger

try:
    from websocket import WebSocketApp
except ImportError:
    WebSocketApp = None


class NexusClient:
    def __init__(self, manifest_provider=None):
        self._server_url = os.environ.get("NEXUS_SERVER_URL", "").rstrip("/")
        self._instance_id = os.environ.get("NEXUS_INSTANCE_ID", "")
        self._metadata_url = os.environ.get("NEXUS_METADATA_URL", "")
        self._manifest_provider = manifest_provider

        self._ws: Any = None
        self._ws_thread: threading.Thread | None = None
        self._ws_ready = threading.Event()
        self._last_ws_error: str | None = None
        self._on_signal: dict = {}

        if self.available:
            self.connect()
            self.sync_metadata()

    @property
    def available(self) -> bool:
        return bool(self._server_url and self._instance_id)

    def connect(self):
        if not self.available:
            logger.warning("Nexus env vars not set, skipping connection.")
            return

        if WebSocketApp is None:
            logger.warning("websocket-client not installed, skipping Nexus WS.")
            return

        ws_scheme = "wss" if self._server_url.startswith("https") else "ws"
        host = self._server_url.split("://", 1)[-1]
        ws_url = f"{ws_scheme}://{host}/api/v1/instances/{self._instance_id}/ws"

        self._ws = WebSocketApp(
            ws_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_open=self._on_open,
            on_close=self._on_close,
        )

        self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._last_ws_error = None
        self._ws_ready.clear()
        self._ws_thread.start()
        logger.info(f"Nexus WS connecting to {ws_url}")
        self._ws_ready.wait(timeout=5)
        if not self._ws_ready.is_set():
            error = self._last_ws_error or "timeout while opening websocket"
            self.close()
            raise RuntimeError(f"Nexus WS connection failed: {error}")

    def on(self, signal: str, callback):
        self._on_signal[signal] = callback

    _METADATA_CASTERS = {
        "f64": float,
        "f32": float,
        "u64": int,
        "i64": int,
        "i32": int,
        "string": str,
    }

    @classmethod
    def _cast_metadata_value(cls, raw, dtype: str):
        caster = cls._METADATA_CASTERS.get(dtype)
        return caster(raw) if caster else raw

    def sync_metadata(self) -> None:
        if not self._manifest_provider:
            return
        url = self._metadata_url
        if not url:
            url = f"{self._server_url}/api/v1/instances/{self._instance_id}/metadata"
        try:
            body = json.dumps({"keys": []}).encode()
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req) as resp:
                payload = json.loads(resp.read())
            if not payload:
                return
            for entry in payload.get("data", []):
                key = entry["key"]
                dtype = entry.get("type", "string")
                value = self._cast_metadata_value(entry["value"], dtype)
                valid = entry.get("valid") or {}
                start = int(valid.get("start_timestamp", 0))
                end = int(valid.get("stop_timestamp", 0)) or (2**64 - 1)
                self._manifest_provider.store(key, value, dtype, start, end)
        except Exception as exc:
            logger.error(f"Failed to sync metadata from {url}: {exc}")
            raise RuntimeError(f"Failed to sync metadata from {url}") from exc

    def sync_metrics(self, metrics: dict):
        if self._ws:
            try:
                self._ws.send(json.dumps({"type": "metrics", "metrics": metrics}))
            except Exception as exc:
                logger.error(f"Failed to sync metrics: {exc}")

    def sync_status(self, status: str, log: dict | None = None):
        if self._ws:
            payload: dict[str, object] = {"type": "status", "status": status}
            if log:
                payload["log"] = log
            try:
                self._ws.send(json.dumps(payload))
            except Exception as exc:
                logger.error(f"Failed to sync status: {exc}")

    def close(self):
        if self._ws:
            self._ws.close()
        if self._ws_thread:
            self._ws_thread.join(timeout=5)
            self._ws_thread = None

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        msg_type = data.get("type")
        if msg_type == "signal":
            signal_name = data.get("signal")
            cb = self._on_signal.get(signal_name)
            if cb:
                cb(data)

    def _on_error(self, ws, error):
        self._last_ws_error = str(error)
        logger.error(f"Nexus WS error: {error}")

    def _on_open(self, ws):
        self._ws_ready.set()
        logger.info("Nexus WS connected.")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info("Nexus WS closed.")
