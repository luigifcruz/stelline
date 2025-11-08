"""
Stelline Application class for building and running pipelines.
"""

from pathlib import Path

import yaml
from holoscan.core import Application
from holoscan.resources import UnboundedAllocator
from holoscan.schedulers import (
    EventBasedScheduler,
    GreedyScheduler,
    MultiThreadScheduler,
)

from stelline.registry import create_bit


class App(Application):
    def __init__(self, config_path):
        super().__init__()
        # Configure the application
        self.config(config_path)

        # Setup scheduler based on config
        self._setup_scheduler()

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


def run(config_path: str) -> None:
    """
    Run Stelline with the specified configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file

    Example
    -------
    import stelline as st
    st.run("../recipes/loopback_test.yaml")
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    print(f"Using configuration file: {config_path}")

    # Create and run the application
    app = App(config_path)
    app.run()
