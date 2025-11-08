#!/bin/python3

import argparse
import random
import socket
import struct
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Configuration Profiles
MACHINE_PROFILES = {
    "spark": {
        "iface": "enp1s0f0np0",
        "dst_mac": "4cbb472d05e0",
        "src_mac": "4cbb472d05df",
        "src_ip": "10.0.1.1",
        "dst_ip": "10.0.2.1",
    },
}

OBSERVATORY_PROFILES = {
    "ata_96mhz": {
        "observatory": "ata",
        "ata_total_antenna": 28,
        "ata_total_channels": 192,
        "ata_total_samples": 8192,
        "ata_total_polarizations": 2,
        "ata_partial_antenna": 1,
        "ata_partial_channels": 96,
        "ata_partial_samples": 16,
        "ata_partial_polarizations": 2,
        "ata_sample_time": 0.002,
    },
}


@dataclass
class GlobalConfig:
    iface: str
    dst_mac: bytes
    src_mac: bytes
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int

    packets_per_batch: int = 1000

    ip_proto_udp: int = 17
    eth_type_ip: int = 0x0800


# Precompiled struct packers
_IP_HDR = struct.Struct("!BBHHHBBH4s4s")
_UDP_HDR = struct.Struct("!HHHH")


# Base class for different observatory packet formats
class Observatory(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_payload_header(self, time_index: int) -> bytes:
        pass

    @abstractmethod
    def generate_payload_data(self) -> bytes:
        pass

    @abstractmethod
    def add_config_rows(self, table) -> None:
        pass

    @staticmethod
    @abstractmethod
    def add_args(parser: argparse.ArgumentParser):
        pass

    @staticmethod
    @abstractmethod
    def from_args(args: argparse.Namespace) -> "Observatory":
        pass

    @abstractmethod
    def get_block_time_ms(self) -> float:
        pass

    def generate_packet(self, time_index: int) -> bytes:
        return self.generate_payload_header(time_index) + self.generate_payload_data()


# Allen Telescope Array packet format
class ATAObservatory(Observatory):
    _ATA_HDR = struct.Struct("!BBHHHQ")

    def __init__(
        self,
        partial_antenna: int = 1,
        partial_channels: int = 1,
        partial_samples: int = 1,
        partial_polarizations: int = 1,
        total_antenna: int = 42,
        total_channels: int = 96,
        total_samples: int = 8192,
        total_polarizations: int = 2,
        sample_time: float = 0.002,
    ):
        super().__init__("ATA")
        self.partial_antenna = partial_antenna
        self.partial_channels = partial_channels
        self.partial_samples = partial_samples
        self.partial_polarizations = partial_polarizations
        self.total_antenna = total_antenna
        self.total_channels = total_channels
        self.total_samples = total_samples
        self.total_polarizations = total_polarizations
        self.sample_time = sample_time

        self.timestamp_increment = partial_samples

        if total_channels % partial_channels != 0:
            raise ValueError(
                f"total_channels ({total_channels}) must be divisible by partial_channels ({partial_channels})"
            )
        self.packets_per_antenna = total_channels // partial_channels

        self.payload_size = (
            partial_channels * partial_samples * partial_polarizations * 2
        )
        self._payload_data = b"X" * self.payload_size

        self.total_packets = total_antenna * self.packets_per_antenna

        self.current_timestamp = 0.0

    def get_block_time_ms(self) -> float:
        return self.partial_samples * self.sample_time

    def generate_payload_header(
        self, time_index: int, antenna_id: int, channel_start: int
    ) -> bytes:
        timestamp_uint64 = int(self.current_timestamp)

        return self._ATA_HDR.pack(
            0,
            0,
            self.total_channels,
            channel_start,
            antenna_id,
            timestamp_uint64,
        )

    def generate_payload_data(self) -> bytes:
        return self._payload_data

    def generate_packet(
        self, time_index: int, antenna_id: int = 0, channel_start: int = 0
    ) -> bytes:
        return (
            self.generate_payload_header(time_index, antenna_id, channel_start)
            + self.generate_payload_data()
        )

    def add_config_rows(self, table) -> None:
        total_block = f"[{self.total_antenna}, {self.total_channels}, {self.total_samples}, {self.total_polarizations}]"
        partial_block = f"[{self.partial_antenna}, {self.partial_channels}, {self.partial_samples}, {self.partial_polarizations}]"
        timestamp_inc = f"{self.timestamp_increment} ticks/packet"
        partial_block_time = f"{self.get_block_time_ms():.6f} ms"
        total_block_time = f"{self.total_samples * self.sample_time:.6f} ms"
        sample_time = f"{self.sample_time * 1000:.3f} μs/sample"

        table.add_row("Total Block", total_block)
        table.add_row("Partial Block", partial_block)
        table.add_row("Timestamp Increment", timestamp_inc)
        table.add_row("Sample Time", sample_time)
        table.add_row("Partial Block Time", partial_block_time)
        table.add_row("Total Block Time", total_block_time)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--ata-partial-antenna",
            type=int,
            default=1,
            help="Partial antenna index (default: 1)",
        )
        parser.add_argument(
            "--ata-partial-channels",
            type=int,
            default=1,
            help="Number of channels per packet (default: 1)",
        )
        parser.add_argument(
            "--ata-partial-samples",
            type=int,
            default=1,
            help="Number of samples in this packet (default: 1)",
        )
        parser.add_argument(
            "--ata-partial-polarizations",
            type=int,
            default=1,
            help="Number of polarizations in this packet (default: 1)",
        )
        parser.add_argument(
            "--ata-total-antenna",
            type=int,
            default=42,
            help="Total number of antennas to simulate (default: 42)",
        )
        parser.add_argument(
            "--ata-total-channels",
            type=int,
            default=96,
            help="Total number of channels (default: 96)",
        )
        parser.add_argument(
            "--ata-total-samples",
            type=int,
            default=8192,
            help="Total number of samples (default: 8192)",
        )
        parser.add_argument(
            "--ata-total-polarizations",
            type=int,
            default=2,
            help="Total number of polarizations (default: 2)",
        )
        parser.add_argument(
            "--ata-sample-time",
            type=float,
            default=0.002,
            help="Time per sample in milliseconds (default: 0.002 ms, i.e., 500 kHz sample rate)",
        )

    @staticmethod
    def from_args(args: argparse.Namespace) -> "ATAObservatory":
        return ATAObservatory(
            partial_antenna=args.ata_partial_antenna,
            partial_channels=args.ata_partial_channels,
            partial_samples=args.ata_partial_samples,
            partial_polarizations=args.ata_partial_polarizations,
            total_antenna=args.ata_total_antenna,
            total_channels=args.ata_total_channels,
            total_samples=args.ata_total_samples,
            total_polarizations=args.ata_total_polarizations,
            sample_time=args.ata_sample_time,
        )


# Registry of available observatories
OBSERVATORIES = {
    "ata": ATAObservatory,
}


def checksum(data: bytes) -> int:
    if len(data) & 1:
        data += b"\x00"
    s = sum(struct.unpack("!%dH" % (len(data) >> 1), data))
    s = (s & 0xFFFF) + (s >> 16)
    s = (s & 0xFFFF) + (s >> 16)
    return (~s) & 0xFFFF


def generate_header_mac(dst_mac: bytes, src_mac: bytes, eth_type: int) -> bytes:
    return dst_mac + src_mac + struct.pack("!H", eth_type)


def generate_header_ip(
    src_ip_p: bytes,
    dst_ip_p: bytes,
    udp_payload_len: int,
    ident: int,
    proto: int,
) -> bytes:
    ver_ihl = (4 << 4) | 5
    tos = 0
    udp_len = 8 + udp_payload_len
    total_len = 20 + udp_len
    flags_frag = 0

    ip_wo_csum = _IP_HDR.pack(
        ver_ihl, tos, total_len, ident, flags_frag, 64, proto, 0, src_ip_p, dst_ip_p
    )
    ip_csum = checksum(ip_wo_csum)
    return _IP_HDR.pack(
        ver_ihl,
        tos,
        total_len,
        ident,
        flags_frag,
        64,
        proto,
        ip_csum,
        src_ip_p,
        dst_ip_p,
    )


def generate_header_udp(
    src_ip_p: bytes,
    dst_ip_p: bytes,
    src_port: int,
    dst_port: int,
    payload: bytes,
    proto: int,
) -> bytes:
    udp_len = 8 + len(payload)
    udp_wo_csum = _UDP_HDR.pack(src_port, dst_port, udp_len, 0)

    pseudo = struct.pack("!4s4sBBH", src_ip_p, dst_ip_p, 0, proto, udp_len)
    udp_csum = checksum(pseudo + udp_wo_csum + payload)
    if udp_csum == 0:
        udp_csum = 0xFFFF
    return _UDP_HDR.pack(src_port, dst_port, udp_len, udp_csum)


# Statistics tracker
class Stats:
    def __init__(self):
        self.packets_sent = 0
        self.bytes_sent = 0
        self.blocks_sent = 0
        self.start_time = time.time()
        self.last_print_time = self.start_time
        self.last_packets = 0
        self.last_bytes = 0
        self.last_blocks = 0

    def should_print(self, interval: float) -> bool:
        return (time.time() - self.last_print_time) >= interval

    def print_stats(self, console: Console = None, live_mode: bool = True):
        if console is None:
            console = Console()

        now = time.time()
        elapsed_total = now - self.start_time
        elapsed_interval = now - self.last_print_time

        interval_bytes = self.bytes_sent - self.last_bytes
        interval_blocks = self.blocks_sent - self.last_blocks

        bps_interval = interval_bytes / elapsed_interval if elapsed_interval > 0 else 0
        gbps_interval = (bps_interval * 8) / 1e9
        blocks_per_sec = (
            interval_blocks / elapsed_interval if elapsed_interval > 0 else 0
        )

        tb_total = self.bytes_sent / 1e12

        hours = int(elapsed_total // 3600)
        minutes = int((elapsed_total % 3600) // 60)
        seconds = elapsed_total % 60

        if live_mode:
            rate_panel = Panel(
                Align.center(
                    f"[bold green]{gbps_interval:.3f}[/bold green]\n[dim]Gbps[/dim]"
                ),
                title="Rate",
                border_style="green",
                width=15,
            )

            packets_panel = Panel(
                Align.center(
                    f"[bold cyan]{self.packets_sent:,}[/bold cyan]\n[dim]Packets[/dim]"
                ),
                title="Count",
                border_style="cyan",
                width=20,
            )

            blocks_panel = Panel(
                Align.center(
                    f"[bold blue]{blocks_per_sec:.2f}[/bold blue]\n[dim]Blocks/s[/dim]"
                ),
                title="Block Rate",
                border_style="blue",
                width=18,
            )

            data_panel = Panel(
                Align.center(
                    f"[bold yellow]{tb_total:.6f}[/bold yellow]\n[dim]TB[/dim]"
                ),
                title="Data",
                border_style="yellow",
                width=15,
            )

            time_panel = Panel(
                Align.center(
                    f"[bold magenta]{hours:02d}:{minutes:02d}:{seconds:06.3f}[/bold magenta]\n[dim]Elapsed[/dim]"
                ),
                title="Time",
                border_style="magenta",
                width=18,
            )

            return Align.center(
                Columns(
                    [rate_panel, packets_panel, blocks_panel, data_panel, time_panel],
                    expand=False,
                )
            )
        else:
            rate_panel = Panel(
                Align.center(
                    f"[bold green]{gbps_interval:.3f}[/bold green]\n[dim]Gbps[/dim]"
                ),
                title="Average Rate",
                border_style="green",
                width=18,
            )

            packets_panel = Panel(
                Align.center(
                    f"[bold cyan]{self.packets_sent:,}[/bold cyan]\n[dim]Packets[/dim]"
                ),
                title="Total Count",
                border_style="cyan",
                width=20,
            )

            blocks_panel = Panel(
                Align.center(
                    f"[bold blue]{blocks_per_sec:.2f}[/bold blue]\n[dim]Blocks/s[/dim]"
                ),
                title="Block Rate",
                border_style="blue",
                width=18,
            )

            data_panel = Panel(
                Align.center(
                    f"[bold yellow]{tb_total:.6f}[/bold yellow]\n[dim]TB[/dim]"
                ),
                title="Total Data",
                border_style="yellow",
                width=15,
            )

            time_panel = Panel(
                Align.center(
                    f"[bold magenta]{hours:02d}:{minutes:02d}:{seconds:06.3f}[/bold magenta]\n[dim]Total Time[/dim]"
                ),
                title="Duration",
                border_style="magenta",
                width=18,
            )

            final_stats_table = Table(
                title="Final Statistics",
                box=box.ROUNDED,
                show_header=False,
                expand=True,
                border_style="yellow",
                title_style="yellow",
            )
            final_stats_table.add_column("Stats", justify="center")
            final_stats_table.add_row(
                Align.center(
                    Columns(
                        [
                            rate_panel,
                            packets_panel,
                            blocks_panel,
                            data_panel,
                            time_panel,
                        ],
                        expand=False,
                    )
                )
            )
            console.print(final_stats_table)

        self.last_print_time = now
        self.last_packets = self.packets_sent
        self.last_bytes = self.bytes_sent
        self.last_blocks = self.blocks_sent


def rap_parser(subparsers):
    """
    Create and configure the rap subcommand parser.
    """
    parser = subparsers.add_parser(
        "rap",
        help="Modular packet transmission tool for different observatories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use machine and profile presets
  stelline rap --machine spark --profile ata_96mhz

  # Use profile with custom machine settings
  stelline rap --profile ata_96mhz --src-ip 10.0.5.1

  # Fully custom configuration
  stelline rap --observatory ata --ata-total-antenna 42
        """,
    )

    # Profile arguments (processed first)
    parser.add_argument(
        "--machine",
        choices=MACHINE_PROFILES.keys(),
        help="Machine profile for network configuration",
    )
    parser.add_argument(
        "--profile",
        choices=OBSERVATORY_PROFILES.keys(),
        help="Observatory profile for packet configuration",
    )

    # Global arguments
    parser.add_argument(
        "--iface",
        default="enp1s0f0np0",
        help="Network interface (default: enp1s0f0np0)",
    )
    parser.add_argument(
        "--dst-mac",
        default="4cbb472d05e0",
        help="Destination MAC address (hex) (default: 4cbb472d05e0)",
    )
    parser.add_argument(
        "--src-mac",
        default="4cbb472d05df",
        help="Source MAC address (hex) (default: 4cbb472d05df)",
    )
    parser.add_argument(
        "--src-ip", default="10.0.1.1", help="Source IP address (default: 10.0.1.1)"
    )
    parser.add_argument(
        "--dst-ip",
        default="10.0.2.1",
        help="Destination IP address (default: 10.0.2.1)",
    )
    parser.add_argument(
        "--src-port", type=int, default=50100, help="Source UDP port (default: 50100)"
    )
    parser.add_argument(
        "--dst-port",
        type=int,
        default=50000,
        help="Destination UDP port (default: 50000)",
    )
    parser.add_argument(
        "--packets-per-batch",
        type=int,
        default=1000,
        help="Packets to send before checking stats (default: 1000)",
    )

    # Observatory selection
    parser.add_argument(
        "--observatory", choices=OBSERVATORIES.keys(), help="Observatory type"
    )

    # Add observatory-specific arguments
    for obs_class in OBSERVATORIES.values():
        obs_class.add_args(parser)

    return parser


def rap_command(args):
    """
    Execute the rap command with the given arguments.
    """
    # Apply machine profile if specified
    if args.machine:
        machine_config = MACHINE_PROFILES[args.machine]
        for key, value in machine_config.items():
            arg_name = key.replace("_", "-")
            if f"--{arg_name}" not in sys.argv:
                setattr(args, key, value)

    if args.profile:
        profile_config = OBSERVATORY_PROFILES[args.profile]
        for key, value in profile_config.items():
            # Only override if not explicitly set by user
            arg_name = key.replace("_", "-")
            if f"--{arg_name}" not in sys.argv:
                setattr(args, key, value)

    # Validate that observatory is set
    if not args.observatory:
        raise ValueError(
            "--observatory is required (or use --profile to set it automatically)"
        )

    # Create global config
    config = GlobalConfig(
        iface=args.iface,
        dst_mac=bytes.fromhex(args.dst_mac),
        src_mac=bytes.fromhex(args.src_mac),
        src_ip=args.src_ip,
        dst_ip=args.dst_ip,
        src_port=args.src_port,
        dst_port=args.dst_port,
        packets_per_batch=args.packets_per_batch,
    )

    # Create observatory
    observatory = OBSERVATORIES[args.observatory].from_args(args)

    # Pre-compute addresses
    src_ip_p = socket.inet_aton(config.src_ip)
    dst_ip_p = socket.inet_aton(config.dst_ip)

    console = Console()
    console.print("[bold blue]Building packet templates...[/bold blue]")

    # Ethernet header (same for all packets)
    eth_header = generate_header_mac(config.dst_mac, config.src_mac, config.eth_type_ip)

    # Pre-build all unique packet structures
    packet_templates = []
    for antenna_id in range(observatory.total_antenna):
        for packet_idx in range(observatory.packets_per_antenna):
            channel_start = packet_idx * observatory.partial_channels

            packet_templates.append(
                {
                    "antenna_id": antenna_id,
                    "channel_start": channel_start,
                    "payload_data": observatory.generate_payload_data(),
                }
            )

    total_frames = len(packet_templates)

    # Calculate frame size using a sample packet
    sample_payload = observatory.generate_packet(
        time_index=0, antenna_id=0, channel_start=0
    )
    sample_ip_header = generate_header_ip(
        src_ip_p,
        dst_ip_p,
        len(sample_payload),
        ident=random.randrange(0, 65535),
        proto=config.ip_proto_udp,
    )
    sample_udp_header = generate_header_udp(
        src_ip_p,
        dst_ip_p,
        config.src_port,
        config.dst_port,
        sample_payload,
        config.ip_proto_udp,
    )
    frame_size = (
        len(eth_header)
        + len(sample_ip_header)
        + len(sample_udp_header)
        + len(sample_payload)
    )

    # Create reusable buffer for building packets
    header_struct = struct.Struct("!BBHHHQ")

    console = Console()

    # Configuration table
    config_table = Table(
        title="Configuration", box=box.ROUNDED, show_header=False, expand=True
    )
    config_table.add_column("Setting", style="bold cyan")
    config_table.add_column("Value")

    def format_mac(mac_bytes):
        return ":".join(f"{b:02x}" for b in mac_bytes)

    if args.machine:
        config_table.add_row("Machine Profile", args.machine)
    if args.profile:
        config_table.add_row("Observatory Profile", args.profile)

    if args.machine or args.profile:
        config_table.add_row("", "")

    config_table.add_row("Observatory", observatory.name)
    observatory.add_config_rows(config_table)
    config_table.add_row("", "")

    config_table.add_row("Interface", config.iface)
    config_table.add_row(
        "Source Address",
        f"{config.src_ip}:{config.src_port} ({format_mac(config.src_mac)})",
    )
    config_table.add_row(
        "Destination Address",
        f"{config.dst_ip}:{config.dst_port} ({format_mac(config.dst_mac)})",
    )
    config_table.add_row("", "")

    config_table.add_row("Packets per Batch", str(config.packets_per_batch))
    config_table.add_row(
        "Packets per Timestep",
        f"{total_frames} ({observatory.total_antenna} antennas × {observatory.packets_per_antenna} packets)",
    )

    console.print()
    console.print(config_table)

    console.print()

    eth_size = len(eth_header)
    ip_size = len(sample_ip_header)
    udp_size = len(sample_udp_header)
    payload_header_size = len(observatory.generate_payload_header(0, 0, 0))
    payload_data_size = len(observatory.generate_payload_data())

    header_total = eth_size + ip_size + udp_size + payload_header_size
    payload_total = payload_data_size

    # Packet structure visualization
    packet_content = Text()

    packet_content.append("\nHeaders: ", style="bold white")
    eth_ratio = eth_size / header_total
    ip_ratio = ip_size / header_total
    udp_ratio = udp_size / header_total

    bar_width = console.size.width - 30
    eth_chars = max(1, int(eth_ratio * bar_width))
    ip_chars = max(1, int(ip_ratio * bar_width))
    udp_chars = max(1, int(udp_ratio * bar_width))
    ph_chars = bar_width - eth_chars - ip_chars - udp_chars

    packet_content.append("█" * eth_chars, style="red")
    packet_content.append("█" * ip_chars, style="green")
    packet_content.append("█" * udp_chars, style="blue")
    packet_content.append("█" * ph_chars, style="yellow")
    packet_content.append(f" ({header_total} bytes)\n", style="dim")

    packet_content.append("Payload: ", style="bold white")
    packet_content.append("█" * bar_width, style="magenta")
    packet_content.append(f" ({payload_total} bytes)\n\n", style="dim")

    packet_content.append("██  ", style="red")
    packet_content.append(f"Ethernet ({eth_size}B)  ", style="white")
    packet_content.append("██  ", style="green")
    packet_content.append(f"IP ({ip_size}B)  ", style="white")
    packet_content.append("██  ", style="blue")
    packet_content.append(f"UDP ({udp_size}B)  ", style="white")
    packet_content.append("██  ", style="yellow")
    packet_content.append(f"Payload Header ({payload_header_size}B)  ", style="white")
    packet_content.append("██  ", style="magenta")
    packet_content.append(f"Payload Data ({payload_data_size}B)\n\n", style="white")
    packet_content.append(f"Total Frame Size: {frame_size} bytes", style="bold yellow")

    packet_table = Table(
        title="Packet Structure",
        box=box.ROUNDED,
        show_header=False,
        expand=True,
        border_style="cyan",
        title_style="cyan",
    )
    packet_table.add_column("Structure", justify="left")
    packet_table.add_row(packet_content)
    console.print(packet_table)

    console.print()

    console.print()
    console.print(
        "[bold green]Press Enter to start transmission...[/bold green]", end=""
    )
    input()

    # Create socket
    s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
    s.bind((config.iface, 0))

    # Initialize stats
    stats = Stats()

    console.print()
    console.print("[dim]Press Ctrl+C to stop streaming[/dim]")
    console.print()

    # Get block time in seconds for throttling
    block_time_sec = observatory.get_block_time_ms() / 1000.0

    # Create Live display for stats
    with Live(refresh_per_second=2, transient=True) as live:
        try:
            # Pre-build all frames with timestamp=0, then update just the timestamp field
            frame_templates = []
            timestamp_offsets = []

            for template in packet_templates:
                ata_header = header_struct.pack(
                    0,
                    0,
                    observatory.total_channels,
                    template["channel_start"],
                    template["antenna_id"],
                    0,
                )

                udp_payload = ata_header + template["payload_data"]

                ip_header = generate_header_ip(
                    src_ip_p,
                    dst_ip_p,
                    len(udp_payload),
                    random.randrange(0, 65535),
                    config.ip_proto_udp,
                )

                udp_header = generate_header_udp(
                    src_ip_p,
                    dst_ip_p,
                    config.src_port,
                    config.dst_port,
                    udp_payload,
                    config.ip_proto_udp,
                )

                frame = bytearray(eth_header + ip_header + udp_header + udp_payload)

                # Calculate offset to timestamp in the frame
                timestamp_offset = (
                    len(eth_header) + len(ip_header) + len(udp_header) + 8
                )

                frame_templates.append(frame)
                timestamp_offsets.append(timestamp_offset)

            current_timestamp = 0
            num_frames = len(frame_templates)

            # Get socket file descriptor for faster sending
            import os

            sock_fd = s.fileno()

            # Pre-compute struct format for batch timestamp updates
            timestamp_struct = struct.Struct("!Q")

            next_block_time = time.perf_counter()

            while True:
                next_block_time += block_time_sec

                for i in range(num_frames):
                    timestamp_struct.pack_into(
                        frame_templates[i], timestamp_offsets[i], current_timestamp
                    )

                for frame in frame_templates:
                    os.write(sock_fd, frame)

                stats.packets_sent += num_frames
                stats.bytes_sent += frame_size * num_frames
                stats.blocks_sent += 1

                current_timestamp += observatory.timestamp_increment

                if stats.should_print(1.0):
                    live_display = stats.print_stats(console, live_mode=True)
                    if live_display:
                        live.update(live_display)

                current_time = time.perf_counter()
                sleep_time = next_block_time - current_time

                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            pass

    s.close()

    console.print("[bold yellow]Transmission stopped by user[/bold yellow]")
    console.print()
    stats.print_stats(console, live_mode=False)
    console.print()
