import asyncio
import os
from pathlib import Path
import sys
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
            "NEXUS_INSTANCE_ID": "", "NEXUS_INSTANCE_CREDENTIALS": "", **environment,
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


class NexusAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.module = load_bridge(
            NEXUS_INSTANCE_ID="instance-1", NEXUS_INSTANCE_CREDENTIALS="bearer-token",
        )

    def test_metadata_authenticates_with_the_bearer_before_subscribing(self):
        bridge = self.module["_NexusBridge"]()
        factory = Mock()

        async def snapshots():
            yield {"data": [{"key": "observation.sync_timestamp", "value": 123, "type": "u64"}]}

        factory.return_value.subscribe.return_value = snapshots()
        known = {}
        asyncio.run(bridge._stream_metadata(factory, type("ConvexInt64", (), {}), "https://nexus.invalid", known))
        self.assertEqual(factory.return_value.mock_calls, [
            call.set_auth("bearer-token"),
            call.subscribe("queries/observatory:getMetadata", {"instanceId": "instance-1"}),
        ])
        self.assertTrue(bridge._connected)
        self.assertIn("observation.sync_timestamp", known)
        ctx = SimpleNamespace(env={})
        bridge._apply_metadata_updates(ctx)
        self.assertNotIn("bearer-token", repr(ctx.env))

    def test_missing_bearer_fails_before_creating_a_client(self):
        self.module["INSTANCE_CREDENTIALS"] = ""
        factory = Mock()
        bridge = self.module["_NexusBridge"]()
        with self.assertRaisesRegex(ValueError, "NEXUS_INSTANCE_CREDENTIALS is required"):
            asyncio.run(bridge._stream_metadata(factory, object, "https://nexus.invalid", {}))
        factory.assert_not_called()

    def test_metrics_authenticates_with_the_same_bearer_before_publishing(self):
        bridge = self.module["_NexusBridge"]()
        factory = Mock()
        factory.return_value.mutation.side_effect = lambda *_: bridge._stop_event.set()
        snapshot = {"timestamp": 1000, "metrics": {"block": {"count": {"type": "number", "value": 1}}}}
        bridge._pending_metrics_snapshots.put(snapshot)

        with patch.dict(sys.modules, {"convex": SimpleNamespace(ConvexClient=factory)}):
            bridge._metrics_publisher_loop()
        self.assertEqual(factory.return_value.mock_calls, [
            call.set_auth("bearer-token"),
            call.mutation("mutations/metrics:publishInstanceMetrics", {"instanceId": "instance-1", **snapshot}),
        ])


if __name__ == "__main__":
    unittest.main()
