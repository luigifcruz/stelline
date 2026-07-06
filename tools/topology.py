#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text


GPU_DRIVER_PATH = Path("/sys/bus/pci/drivers/nvidia")
INFINIBAND_PATH = Path("/sys/class/infiniband")

CPU_STYLES = {
    "isolated-irq-free": "bold green",
    "isolated-only": "bold red",
    "irq-free-only": "bold blue",
}


@dataclass(frozen=True)
class StyledCpu:
    value: str
    class_name: str


@dataclass(frozen=True)
class GpuInfo:
    idx: int
    pcie: str
    numa: str
    cpus: str
    name: str


@dataclass(frozen=True)
class NicInfo:
    name: str
    pcie: str
    numa: str
    iface: str


@dataclass(frozen=True)
class TopologyRow:
    gpu_id: str
    gpu_name: str
    gpu_pcie: str
    numa: str
    cpus: list[StyledCpu]
    nic: str
    nic_pcie: str
    iface: str
    section_start: bool


@dataclass(frozen=True)
class TopologyModel:
    rows: list[TopologyRow]
    isolated_detected: bool
    warnings: list[str]


def read_text(path: Path, default: str = "N/A") -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return default


def run_command(command: str, args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            [command, *args],
            check=False,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return None

    if proc.returncode != 0:
        return None

    return proc.stdout.strip()


def parse_cpu_list(text: str) -> set[int]:
    cpus: set[int] = set()
    value = text.strip()
    if not value:
        return cpus

    for part in value.split(","):
        token = part.strip()
        if not token:
            continue

        if "-" in token:
            lo_text, hi_text = token.split("-", 1)
            try:
                lo = int(lo_text, 10)
                hi = int(hi_text, 10)
            except ValueError:
                continue

            cpus.update(range(lo, hi + 1))
            continue

        try:
            cpus.add(int(token, 10))
        except ValueError:
            continue

    return cpus


def parse_cpu_mask(text: str) -> set[int]:
    normalized = text.strip().replace(",", "")
    if not normalized:
        return set()

    try:
        mask = int(normalized, 16)
    except ValueError:
        return set()

    cpus: set[int] = set()
    bit = 0
    while mask > 0:
        if mask & 1:
            cpus.add(bit)
        mask >>= 1
        bit += 1

    return cpus


def classify_cpu(cpu: int, isolated: set[int], irq_cpus: set[int]) -> str:
    is_isolated = cpu in isolated
    is_irq_free = cpu not in irq_cpus

    if is_isolated and is_irq_free:
        return "isolated-irq-free"
    if is_isolated:
        return "isolated-only"
    if is_irq_free:
        return "irq-free-only"
    return "default"


def style_cpu_list(cpu_text: str, isolated: set[int], irq_cpus: set[int]) -> list[StyledCpu]:
    if cpu_text == "N/A":
        return [StyledCpu("N/A", "default")]

    cpus: list[int] = []
    for token in cpu_text.split():
        try:
            cpus.append(int(token, 10))
        except ValueError:
            continue

    if not cpus:
        return [StyledCpu(cpu_text, "default")]

    return [
        StyledCpu(str(cpu), classify_cpu(cpu, isolated, irq_cpus))
        for cpu in cpus
    ]


@lru_cache(maxsize=1)
def numactl_hardware() -> str | None:
    return run_command("numactl", ["-H"])


def get_numa_cpus(numa_node: str) -> str:
    output = numactl_hardware()
    if not output:
        return "N/A"

    prefix = f"node {numa_node} cpus:"
    for line in output.splitlines():
        if line.startswith(prefix):
            _, cpu_text = line.split(":", 1)
            return cpu_text.strip() or "N/A"

    return "N/A"


def query_gpu_names() -> dict[str, str]:
    output = run_command(
        "nvidia-smi",
        ["--query-gpu=gpu_bus_id,gpu_name", "--format=csv,noheader,nounits"],
    )
    names: dict[str, str] = {}
    if not output:
        return names

    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            continue

        bus_id = parts[0].lower()
        segments = bus_id.split(":")
        if len(segments) >= 3 and len(segments[0]) == 8:
            bus_id = f"{segments[0][4:]}:{':'.join(segments[1:])}"

        names[bus_id] = parts[1]

    return names


def discover_gpus() -> list[GpuInfo]:
    if not GPU_DRIVER_PATH.exists():
        return []

    gpu_names = query_gpu_names()
    devices = sorted(
        entry.name
        for entry in GPU_DRIVER_PATH.iterdir()
        if re.fullmatch(r"\w{4}:\w{2}:\w{2}\.\w", entry.name)
    )

    gpus: list[GpuInfo] = []
    for idx, pcie in enumerate(devices):
        base_path = GPU_DRIVER_PATH / pcie
        numa = read_text(base_path / "numa_node", "N/A")
        cpus = get_numa_cpus(numa) if numa not in {"N/A", "-1"} else "N/A"
        gpus.append(
            GpuInfo(
                idx=idx,
                pcie=pcie,
                numa=numa,
                cpus=cpus,
                name=gpu_names.get(pcie.lower(), "N/A"),
            )
        )

    return gpus


def discover_nics() -> list[NicInfo]:
    if not INFINIBAND_PATH.exists():
        return []

    nic_names = sorted(
        entry.name for entry in INFINIBAND_PATH.iterdir() if entry.name.startswith("mlx5_")
    )
    nics: list[NicInfo] = []

    for name in nic_names:
        device_path = INFINIBAND_PATH / name / "device"
        try:
            pcie = os.readlink(device_path).split("/")[-1]
        except OSError:
            pcie = "N/A"

        numa = read_text(device_path / "numa_node", "N/A")
        try:
            iface = sorted(entry.name for entry in (device_path / "net").iterdir())[0]
        except (OSError, IndexError):
            iface = "N/A"

        nics.append(NicInfo(name=name, pcie=pcie, numa=numa, iface=iface))

    return nics


def build_topology_model() -> TopologyModel:
    gpus = discover_gpus()
    nics = discover_nics()
    warnings: list[str] = []

    if not gpus:
        warnings.append("No NVIDIA GPUs found in /sys/bus/pci/drivers/nvidia.")
        return TopologyModel(rows=[], isolated_detected=False, warnings=warnings)

    isolated = parse_cpu_list(read_text(Path("/sys/devices/system/cpu/isolated"), ""))
    irq_affinity_text = read_text(Path("/proc/irq/default_smp_affinity"), "")
    irq_cpus = parse_cpu_mask(irq_affinity_text) if irq_affinity_text else set()

    if not isolated:
        warnings.append(
            "No isolated cores detected. Transport usually expects isolcpus, "
            "nohz_full, rcu_nocbs, and irqaffinity kernel parameters."
        )

    rows: list[TopologyRow] = []
    for gpu_index, gpu in enumerate(gpus):
        matched_nics = [nic for nic in nics if nic.numa == gpu.numa]
        cpus = style_cpu_list(gpu.cpus, isolated, irq_cpus)

        if not matched_nics:
            rows.append(
                TopologyRow(
                    gpu_id=f"GPU{gpu.idx}",
                    gpu_name=gpu.name,
                    gpu_pcie=gpu.pcie,
                    numa=gpu.numa,
                    cpus=cpus,
                    nic="none",
                    nic_pcie="-",
                    iface="-",
                    section_start=gpu_index > 0,
                )
            )
            continue

        for nic_index, nic in enumerate(matched_nics):
            rows.append(
                TopologyRow(
                    gpu_id=f"GPU{gpu.idx}" if nic_index == 0 else "",
                    gpu_name=gpu.name if nic_index == 0 else "",
                    gpu_pcie=gpu.pcie if nic_index == 0 else "",
                    numa=gpu.numa if nic_index == 0 else "",
                    cpus=cpus if nic_index == 0 else [],
                    nic=nic.name,
                    nic_pcie=nic.pcie,
                    iface=nic.iface,
                    section_start=gpu_index > 0 and nic_index == 0,
                )
            )

    return TopologyModel(rows=rows, isolated_detected=bool(isolated), warnings=warnings)


def cpu_cell(cpus: list[StyledCpu]) -> Text:
    text = Text()
    for index, cpu in enumerate(cpus):
        if index:
            text.append(" ")
        text.append(cpu.value, style=CPU_STYLES.get(cpu.class_name))
    return text


def legend() -> Text:
    return Text.assemble(
        ("●", "green"),
        " isolated + IRQ-free   ",
        ("●", "red"),
        " isolated   ",
        ("●", "blue"),
        " IRQ-free",
    )


def build_table(model: TopologyModel) -> Table:
    table = Table(
        title="Stelline Topology",
        title_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        caption=legend(),
        caption_style="dim",
    )
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("GPU Name", overflow="fold", min_width=12)
    table.add_column("GPU PCIe", no_wrap=True)
    table.add_column("NUMA", justify="center", no_wrap=True)
    table.add_column("CPUs", overflow="fold", min_width=12, max_width=48)
    table.add_column("NIC", no_wrap=True)
    table.add_column("NIC PCIe", no_wrap=True)
    table.add_column("Interface", no_wrap=True)

    for row in model.rows:
        if row.section_start:
            table.add_section()
        nic_style = "magenta" if row.nic and row.nic != "none" else "dim"
        table.add_row(
            row.gpu_id,
            row.gpu_name,
            row.gpu_pcie,
            row.numa,
            cpu_cell(row.cpus),
            Text(row.nic, style=nic_style),
            row.nic_pcie,
            row.iface,
        )

    return table


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display Stelline system topology.")
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    console = Console(no_color=args.no_color or None, highlight=False)

    try:
        model = build_topology_model()
    except Exception as error:  # noqa: BLE001 - diagnostics should report unexpected host issues.
        Console(stderr=True).print(f"[bold red]Failed to build topology:[/] {error}")
        return 1

    for warning in model.warnings:
        console.print(f"[bold yellow]WARN:[/] {warning}")

    if not model.rows:
        console.print("No topology rows available.")
        return 1

    console.print(build_table(model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
