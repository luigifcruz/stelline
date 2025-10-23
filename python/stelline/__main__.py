import sys
import argparse
import yaml
from pathlib import Path
from holoscan.core import Application
from holoscan.schedulers import (
    GreedyScheduler,
    MultiThreadScheduler,
    EventBasedScheduler,
)
from holoscan.resources import UnboundedAllocator

from stelline.bits import (
    BladeBit,
    FilesystemBit,
    TransportBit,
    SourceBit,
    SocketBit,
    FrbnnInferenceBit,
    FrbnnDetectionBit,
)
from stelline import __version__


class StellineApp(Application):
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path

    def compose(self):
        pool = UnboundedAllocator(self, name="pool")

        stelline_config = self.kwargs("stelline")
        graph_dict = stelline_config["graph"]

        bit_map = {}
        flows = []

        for idx, (node_id, node_config) in enumerate(graph_dict.items()):
            bit_type = node_config["bit"]
            bit_config_key = node_config["configuration"]
            input_map = node_config.get("input", {})

            if bit_type == "socket_bit":
                bit_map[node_id] = SocketBit(self, pool, idx, bit_config_key)
            elif bit_type == "transport_bit":
                bit_map[node_id] = TransportBit(self, pool, idx, bit_config_key)
            elif bit_type == "source_bit":
                bit_map[node_id] = SourceBit(self, pool, idx, bit_config_key)
            elif bit_type == "blade_bit":
                bit_map[node_id] = BladeBit(self, pool, idx, bit_config_key)
            elif bit_type == "filesystem_bit":
                bit_map[node_id] = FilesystemBit(self, pool, idx, bit_config_key)
            elif bit_type == "frbnn_inference_bit":
                bit_map[node_id] = FrbnnInferenceBit(self, pool, idx, bit_config_key)
            elif bit_type == "frbnn_detection_bit":
                bit_map[node_id] = FrbnnDetectionBit(self, pool, idx, bit_config_key)
            else:
                raise ValueError(f"Unknown bit: {bit_type}")

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


def main():
    parser = argparse.ArgumentParser(
        description="Stelline - Software Defined Telescope Pipeline"
    )
    parser.add_argument("config", help="Path to configuration YAML file")
    parser.add_argument(
        "--version", "-v", action="version", version=f"Stelline v{__version__}"
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    print(f"Using configuration file: {args.config}")

    app = StellineApp(args.config)
    app.config(args.config)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    scheduler_type = config.get("scheduler_type", "greedy")

    if scheduler_type == "greedy":
        app.scheduler(GreedyScheduler(app, name="greedy-scheduler"))
    elif scheduler_type == "multi_thread":
        app.scheduler(MultiThreadScheduler(app, name="multithread-scheduler"))
    elif scheduler_type == "event_based":
        app.scheduler(EventBasedScheduler(app, name="event-scheduler"))
    else:
        raise ValueError(f"Invalid scheduler type: {scheduler_type}")

    app.run()


if __name__ == "__main__":
    main()
