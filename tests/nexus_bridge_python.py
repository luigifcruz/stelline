import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class NexusMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "src/net/nexus_bridge/module_impl_python.cc"
        )
        source = source_path.read_text().split('R"NEXUSPY(', 1)[1].split(')NEXUSPY"', 1)[0]
        source = source.replace("<<<NEXUS_URL>>>", repr("https://nexus.invalid"))
        cls.bridge = {}
        with (
            patch.dict(os.environ, {"NEXUS_INSTANCE_ID": ""}),
            patch("threading.Thread.start"),
            patch("threading.Thread.join"),
        ):
            exec(compile(source, str(source_path), "exec"), cls.bridge)
            cls.bridge["cleanup"]()

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


if __name__ == "__main__":
    unittest.main()
