"""
Stelline Application class for building and running pipelines.
"""

import time
from pathlib import Path
from typing import Iterable, List

from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from holoscan.core import Application, MetadataPolicy, Operator
from holoscan.resources import UnboundedAllocator
from holoscan.schedulers import (
    EventBasedScheduler,
    GreedyScheduler,
    MultiThreadScheduler,
)

from stelline.manifest import ManifestProvider, MetricsProvider
from stelline.nexus import NexusClient
from stelline.registry import create_bit
from stelline.types import SystemInfo
from stelline.utils import logger


def _detect_system_info() -> None:
    """Detect system properties and populate the SystemInfo singleton."""
    # System name.
    try:
        name = Path("/sys/devices/virtual/dmi/id/product_name").read_text().strip()
    except OSError:
        name = "Unknown"

    # Unified memory and discrete GPU detection via CUDA runtime.
    unified = False
    discrete = True
    try:
        import cuda.cudart as cudart

        err, count = cudart.cudaGetDeviceCount()
        if err == cudart.cudaError_t.cudaSuccess and count > 0:
            err, props = cudart.cudaGetDeviceProperties(0)
            if err == cudart.cudaError_t.cudaSuccess:
                integrated = props.integrated
                unified = bool(integrated)
                discrete = not unified
    except (ImportError, Exception):
        pass

    SystemInfo.instance().configure(name, unified, discrete)

    logger.info(f"System: {name}, Unified Memory: {unified}, Discrete GPU: {discrete}")


class App(Application):
    def __init__(
        self,
        config_path: str,
        metrics: bool = False,
        metrics_interval: float = 1.0,
    ):
        super().__init__()
        # Configure the application
        self.config(config_path)

        # Detect and set system info globally.
        _detect_system_info()

        # Configure metadata
        self.enable_metadata(True)
        self.metadata_policy = MetadataPolicy.INPLACE_UPDATE

        # Configure metrics
        self._metrics_enabled = metrics
        self._metrics_interval = max(metrics_interval, 0.1)
        self._operators: List[Operator] = []
        self._metrics_providers: dict = {}
        self._metrics_thread = None
        self._metrics_stop_event = None

        # Manifest provider
        self._manifest_provider = ManifestProvider()

        # Configure Nexus
        self._nexus = NexusClient(self._manifest_provider)

        # Setup scheduler based on config
        self._setup_scheduler()

    @property
    def manifest(self):
        return self._manifest_provider

    def _setup_scheduler(self):
        """Setup the scheduler based on configuration."""

        scheduler_type = self.kwargs("scheduler_type")["scheduler_type"] or "greedy"

        if scheduler_type == "greedy":
            self.scheduler(GreedyScheduler(self, name="greedy-scheduler"))
        elif scheduler_type == "multi_thread":
            self.scheduler(MultiThreadScheduler(self, name="multithread-scheduler"))
        elif scheduler_type == "event_based":
            self.scheduler(EventBasedScheduler(self, name="event-scheduler"))
        else:
            raise ValueError(f"Invalid scheduler type: {scheduler_type}")

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        stelline_config = self.kwargs("stelline")
        graph_dict = stelline_config["graph"]

        bit_map = {}
        flows = []

        for idx, (node_id, node_config) in enumerate(graph_dict.items()):
            bit_type = node_config["bit"]
            bit_config_key = node_config["configuration"]
            input_map = node_config.get("input") or {}

            bit_map[node_id] = create_bit(bit_type, self, pool, idx, bit_config_key)

            for dst_port, src_id in input_map.items():
                flows.append((node_id, src_id))

        for dst_id, src_id in flows:
            if src_id not in bit_map:
                raise ValueError(f"Unknown source node: {src_id}")
            if dst_id not in bit_map:
                raise ValueError(f"Unknown destination node: {dst_id}")

            _, src_out = bit_map[src_id]
            dst_in, _ = bit_map[dst_id]
            self.add_flow(src_out, dst_in)

        # Determine operator ordering based on graph topology (Kahn's algorithm).
        indegree = {node: 0 for node in bit_map.keys()}
        adjacency = {node: [] for node in bit_map.keys()}
        for dst_id, src_id in flows:
            indegree[dst_id] = indegree.get(dst_id, 0) + 1
            adjacency.setdefault(src_id, []).append(dst_id)

        topo_order = []
        queue = [node for node, deg in indegree.items() if deg == 0]
        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            for neighbor in adjacency.get(node, []):
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        # Fallback to insertion order if a cycle is detected.
        if len(topo_order) != len(bit_map):
            topo_order = list(bit_map.keys())

        ordered_ops = []
        seen = set()
        for node in topo_order:
            ops = bit_map[node]
            for op in ops:
                if op and op not in seen:
                    ordered_ops.append(op)
                    seen.add(op)

        self._operators = ordered_ops

        # Wire manifest and metrics providers to operators.
        for op in self._operators:
            if hasattr(op, "set_manifest_provider") and self._manifest_provider:
                op.set_manifest_provider(self._manifest_provider)
            if hasattr(op, "set_metrics_provider"):
                provider = MetricsProvider()
                self._metrics_providers[id(op)] = provider
                op.set_metrics_provider(provider)

        if self._metrics_enabled:
            self._start_metrics_thread()

    def get_operators(self) -> Iterable[Operator]:
        return list(self._operators)

    def stop_metrics_display(self) -> None:
        if self._metrics_thread and self._metrics_stop_event:
            self._metrics_stop_event.set()
            self._metrics_thread.join()
            self._metrics_thread = None
            self._metrics_stop_event = None

    def _tick_operators(self):
        for op in self._operators:
            if hasattr(op, "tick"):
                op.tick()

    def _print_metrics(self):
        console = Console()
        timestamp = time.strftime("%H:%M:%S")
        parts = []
        for op in self._operators:
            provider = self._metrics_providers.get(id(op))
            if not provider:
                continue
            try:
                data = provider.snapshot()
                if not data:
                    continue
                name = getattr(op, "name", None) or op.__class__.__name__
                text = op.format_metrics(data)
                if parts:
                    parts.append(Rule(style="dim"))
                parts.append(Text(name, style="bold"))
                parts.append(Text(text))
            except Exception as exc:
                if parts:
                    parts.append(Rule(style="dim"))
                parts.append(Text(f"ERROR: {exc}", style="bold red"))
        if parts:
            console.print(
                Panel(Group(*parts), title=f"Metrics {timestamp}", border_style="cyan")
            )

    def _send_metrics(self):
        if not self._nexus.available:
            return
        global_metrics = {}
        for op in self._operators:
            provider = self._metrics_providers.get(id(op))
            if not provider:
                continue
            data = provider.snapshot(True)
            if not data:
                continue
            name = getattr(op, "name", None) or op.__class__.__name__
            group = {}
            for key, metric in data.items():
                value = metric.value
                if metric.type == "number":
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                group[key] = {"type": metric.type, "value": value}
            global_metrics[name] = group
        self._nexus.sync_metrics(global_metrics)

    def _start_metrics_thread(self) -> None:
        import threading

        self._metrics_stop_event = threading.Event()

        def _loop():
            while not self._metrics_stop_event.is_set():
                self._tick_operators()
                self._print_metrics()
                self._send_metrics()
                time.sleep(self._metrics_interval)

        self._metrics_thread = threading.Thread(target=_loop, daemon=True)
        self._metrics_thread.start()

    def run(self) -> None:
        if self._nexus.available:
            self._nexus.sync_status("RUNNING")
        try:
            super().run()
        finally:
            if self._nexus.available:
                self._nexus.sync_status("STOPPED")
                self._nexus.close()
            self.stop_metrics_display()


def run(
    config_path: str,
    *,
    metrics: bool = False,
    metrics_interval: float = 1.0,
) -> None:
    """
    Run Stelline with the specified configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
    metrics : bool
        Enable periodic metrics printing.
    metrics_interval : float
        Seconds between metrics refreshes.

    Example
    -------
    import stelline as st
    st.run("../recipes/loopback_test.yaml", metrics=True, metrics_interval=0.5)
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    print(f"Using configuration file: {config_path}")

    app = App(
        config_path,
        metrics=metrics,
        metrics_interval=metrics_interval,
    )
    try:
        app.run()
    finally:
        app.stop_metrics_display()
