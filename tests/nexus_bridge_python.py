import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch


def load_bridge(**environment):
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src/net/nexus_bridge/module_impl_python.cc"
    )
    source = source_path.read_text().split('R"NEXUSPY(', 1)[1].split(')NEXUSPY"', 1)[0]
    source = source.replace("<<<NEXUS_URL>>>", repr("https://nexus.invalid"))
    bridge = {}
    with (
        patch.dict(os.environ, {
            "NEXUS_INSTANCE_ID": "",
            "NEXUS_INSTANCE_CREDENTIALS": "",
            "NEXUS_INSTANCE_CREDENTIALS_FILE": "",
            "NEXUS_TELEMETRY_URL": "",
            **environment,
        }),
        patch("threading.Thread.start"),
        patch("threading.Thread.join"),
    ):
        exec(compile(source, str(source_path), "exec"), bridge)
        bridge["cleanup"]()
    return bridge


class NexusMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = load_bridge()

    @staticmethod
    def metric(value, metric_type="stelline-metrics-number"):
        return {
            "value": value,
            "format": {"type": metric_type, "visibility": "internal"},
            "label": "Metric",
            "help": "Telemetry test metric.",
        }

    def test_descriptor_values_are_converted_for_nexus(self):
        normalize = self.bridge["_normalize_metric"]
        for value in (42, 42.5, "42.5"):
            with self.subTest(value=value):
                self.assertEqual(
                    normalize(self.metric(value)),
                    {"type": "number", "value": float(value)},
                )
        self.assertEqual(
            normalize(self.metric("[1, 2]", "stelline-metrics-string")),
            {"type": "text", "value": "[1, 2]"},
        )

    def test_unsupported_and_invalid_metrics_are_omitted(self):
        entries = [
            None,
            {},
            {"value": "42", "format": "private-stelline-metrics-number"},
            self.metric("42 MB/s", "label"),
            self.metric(42, "timing"),
            self.metric(42, "stelline-metrics-string"),
        ]
        entries += [
            self.metric(value)
            for value in (True, None, "invalid", "nan", "inf", float("-inf"))
        ]
        for entry in entries:
            with self.subTest(entry=entry):
                self.assertIsNone(self.bridge["_normalize_metric"](entry))

    def test_snapshot_selects_only_stelline_telemetry(self):
        ctx = SimpleNamespace(metrics={
            "ata": {
                "packetsReceived": self.metric("123"),
                "allAntennas": self.metric("[1, 2]", "stelline-metrics-string"),
                "throughput": self.metric("12.3 Gbps", "label"),
            },
            "display": {"status": self.metric("Running", "label")},
            "unavailable": None,
        })
        self.assertEqual(self.bridge["_build_metrics_snapshot"](ctx), {
            "ata": {
                "packetsReceived": {"type": "number", "value": 123.0},
                "allAntennas": {"type": "text", "value": "[1, 2]"},
            },
        })


class NexusTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.module = load_bridge(
            NEXUS_INSTANCE_ID="instance-1", NEXUS_INSTANCE_CREDENTIALS="bearer-token",
        )

    def test_telemetry_url_prefers_the_replicant_value(self):
        self.assertEqual(self.module["_telemetry_url"](), "https://nexus.invalid/telemetry")
        self.module["TELEMETRY_URL"] = "https://telemetry.invalid/telemetry/"
        self.assertEqual(self.module["_telemetry_url"](), "https://telemetry.invalid/telemetry")

    def test_credentials_file_takes_precedence_over_the_environment(self):
        with tempfile.NamedTemporaryFile("w") as credentials:
            credentials.write("file-token\n")
            credentials.flush()
            self.module["INSTANCE_CREDENTIALS_FILE"] = credentials.name
            self.assertEqual(self.module["_instance_credentials"](), "file-token")
        self.module["INSTANCE_CREDENTIALS_FILE"] = ""
        self.assertEqual(self.module["_instance_credentials"](), "bearer-token")

    def test_metadata_is_read_from_the_telemetry_service(self):
        post = Mock(return_value={
            "snapshotId": "1",
            "data": [
                {"key": "observation.sync_timestamp", "value": 123, "type": "u64"},
                {"key": "observatory.name", "value": "ATA", "type": "string"},
            ],
        })
        bridge = self.module["_NexusBridge"](post)
        known = {}
        bridge._poll_metadata(known)
        post.assert_called_once_with(
            "https://nexus.invalid/telemetry/v1/otl/metadata",
            "bearer-token",
            {"instanceId": "instance-1"},
        )
        self.assertTrue(bridge._connected)
        self.assertEqual(known["observation.sync_timestamp"]["value"], "123")

        ctx = SimpleNamespace(env={})
        bridge._apply_metadata_updates(ctx)
        self.assertEqual(set(ctx.env), {"observation.sync_timestamp", "observatory.name"})
        self.assertNotIn("bearer-token", repr(ctx.env))

        post.return_value = {"snapshotId": "2", "data": [
            {"key": "observatory.name", "value": "ATA", "type": "string"},
        ]}
        bridge._poll_metadata(known)
        bridge._apply_metadata_updates(ctx)
        self.assertEqual(set(ctx.env), {"observatory.name"})

    def test_missing_credentials_fail_before_any_request(self):
        self.module["INSTANCE_CREDENTIALS"] = ""
        post = Mock()
        bridge = self.module["_NexusBridge"](post)
        with self.assertRaisesRegex(ValueError, "NEXUS_INSTANCE_CREDENTIALS"):
            bridge._poll_metadata({})
        post.assert_not_called()

    @staticmethod
    def sample(event_id):
        return {
            "eventId": event_id,
            "timestamp": 1000,
            "metrics": {"block": {"count": {"type": "number", "value": 1}}},
        }

    def run_publisher(self, bridge, post, outcomes):
        def respond(*args):
            outcome = outcomes.pop(0)
            if not outcomes:
                bridge._stop_event.set()
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        post.side_effect = respond
        with patch.object(bridge._stop_event, "wait"):
            bridge._metrics_publisher_loop()

    def test_metrics_are_posted_as_samples_and_retried_with_the_same_event_id(self):
        post = Mock()
        bridge = self.module["_NexusBridge"](post)
        bridge._enqueue_metric_sample(self.sample("event-1"))
        error = self.module["TelemetryError"]
        self.run_publisher(bridge, post, [
            error(503, "unavailable"), error(None, "offline"), error(429, "rate limited"), {"accepted": 1},
        ])

        url = "https://nexus.invalid/telemetry/v1/instances/instance-1/metrics"
        body = {"samples": [self.sample("event-1")]}
        self.assertEqual(post.mock_calls, [call(url, "bearer-token", body)] * 4)

    def test_rejected_metrics_are_not_retried(self):
        post = Mock()
        bridge = self.module["_NexusBridge"](post)
        bridge._enqueue_metric_sample(self.sample("event-1"))
        error = self.module["TelemetryError"]
        sent = []

        def respond(url, token, body):
            sent.append([sample["eventId"] for sample in body["samples"]])
            if len(sent) == 1:
                bridge._enqueue_metric_sample(self.sample("event-2"))
                raise error(401, "Invalid credential")
            bridge._stop_event.set()
            return {"accepted": 1}

        post.side_effect = respond
        bridge._metrics_publisher_loop()
        self.assertEqual(sent, [["event-1"], ["event-2"]])

    def test_pending_samples_are_bounded_while_the_publisher_is_stalled(self):
        bridge = self.module["_NexusBridge"](Mock())
        for index in range(300):
            bridge._enqueue_metric_sample(self.sample(f"event-{index}"))
        self.assertEqual(len(bridge._metric_samples), 256)
        self.assertEqual(bridge._metric_samples[0]["eventId"], "event-44")

    def test_batches_stay_below_the_request_size_limit(self):
        post = Mock()
        bridge = self.module["_NexusBridge"](post)
        padding = "x" * (80 * 1024)
        for index in range(16):
            sample = self.sample(f"event-{index}")
            sample["metrics"]["block"]["padding"] = {"type": "text", "value": padding}
            bridge._enqueue_metric_sample(sample)
        sizes = []
        sent = []

        def respond(url, token, body):
            sizes.append(len(json.dumps(body, separators=(",", ":"))))
            sent.extend(sample["eventId"] for sample in body["samples"])
            if len(sent) >= 16 or len(sizes) > 16:
                bridge._stop_event.set()
            return {"accepted": len(body["samples"])}

        post.side_effect = respond
        bridge._metrics_publisher_loop()
        self.assertGreater(len(sizes), 1)
        self.assertTrue(all(size <= 768 * 1024 for size in sizes))
        self.assertEqual(sent, [f"event-{index}" for index in range(16)])
        self.assertEqual(len(bridge._metric_samples), 0)

    def test_oversized_snapshots_are_skipped(self):
        bridge = self.module["_NexusBridge"](Mock())
        bridge._metrics_thread = object()
        ctx = SimpleNamespace(metrics={"block": {"blob": {
            "value": "x" * (600 * 1024),
            "format": {"type": "stelline-metrics-string"},
        }}})
        bridge._queue_metrics_snapshot(ctx)
        self.assertEqual(len(bridge._metric_samples), 0)

    def test_number_expansion_in_javascript_is_counted(self):
        bridge = self.module["_NexusBridge"](Mock())
        bridge._metrics_thread = object()
        ctx = SimpleNamespace(metrics={"block": {
            f"m{index}": {"value": 1e20, "format": {"type": "stelline-metrics-number"}}
            for index in range(10500)
        }})
        self.assertLess(len(json.dumps(self.module["_build_metrics_snapshot"](ctx), separators=(",", ":"))), 512 * 1024)
        bridge._queue_metrics_snapshot(ctx)
        self.assertEqual(len(bridge._metric_samples), 0)

    def test_a_request_finishing_after_cleanup_is_ignored(self):
        bridge = self.module["_NexusBridge"]()

        def post(*args):
            bridge._stop_event.set()
            for name in ("_metadata_entries_by_key", "_telemetry_url", "METADATA_RETRY_SECONDS"):
                self.module[name] = None
            return {"data": [{"key": "observatory.name", "value": "ATA", "type": "string"}]}

        bridge._post = post
        bridge._run_worker(bridge._metadata_watcher_loop)
        self.assertTrue(bridge._pending_metadata_updates.empty())


if __name__ == "__main__":
    unittest.main()
