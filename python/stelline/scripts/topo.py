import argparse
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text


def _read_sysfs(path: str, default: str = "N/A") -> str:
    try:
        return Path(path).read_text().strip()
    except (OSError, IOError):
        return default


def _run_cmd(cmd: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return None


def _parse_cpu_list(text: str) -> set:
    """Parse kernel CPU list format (e.g., '0-2,5-7') into a set of integers."""
    cpus = set()
    text = text.strip()
    if not text:
        return cpus
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _parse_cpu_mask(text: str) -> set:
    """Parse hex CPU bitmask (e.g., '00000000,000000f8') into a set of integers."""
    text = text.strip().replace(",", "")
    if not text:
        return set()
    cpus = set()
    mask = int(text, 16)
    bit = 0
    while mask:
        if mask & 1:
            cpus.add(bit)
        mask >>= 1
        bit += 1
    return cpus


def _color_cpus(cpu_str: str, isolated: set, irq_cpus: set) -> Text:
    """Color-code CPU list. Green=isolated+IRQ-free, Yellow=one of the two, White=normal."""
    if cpu_str == "N/A":
        return Text("N/A")
    try:
        cpus = sorted(int(x) for x in cpu_str.split())
    except ValueError:
        return Text(cpu_str)
    if not cpus:
        return Text("N/A")

    def _style(cpu):
        is_iso = cpu in isolated
        is_irq_free = cpu not in irq_cpus
        if is_iso and is_irq_free:
            return "green"
        if is_iso:
            return "yellow"
        if is_irq_free:
            return "cyan"
        return "white"

    text = Text()
    for i, cpu in enumerate(cpus):
        if i > 0:
            text.append(" ")
        text.append(str(cpu), style=_style(cpu))
    return text


def _get_numa_cpus(numa_node: str) -> str:
    output = _run_cmd("numactl -H")
    if not output:
        return "N/A"
    for line in output.splitlines():
        if line.startswith(f"node {numa_node} cpus:"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return "N/A"


def _query_gpu_names() -> dict:
    """Query GPU names from nvidia-smi, keyed by PCI bus ID."""
    output = _run_cmd(
        "nvidia-smi --query-gpu=gpu_bus_id,gpu_name --format=csv,noheader,nounits"
    )
    if not output:
        return {}
    names = {}
    for line in output.splitlines():
        parts = line.split(", ", 1)
        if len(parts) == 2:
            bus_id = parts[0].strip().lower()
            name = parts[1].strip()
            # nvidia-smi may use 8-char domain (00000000:), normalize to 4-char (0000:)
            segments = bus_id.split(":")
            if len(segments) >= 3 and len(segments[0]) == 8:
                bus_id = segments[0][4:] + ":" + ":".join(segments[1:])
            names[bus_id] = name
    return names


def _discover_gpus() -> List[dict]:
    gpu_driver_path = "/sys/bus/pci/drivers/nvidia"
    if not Path(gpu_driver_path).is_dir():
        return []

    gpu_names = _query_gpu_names()

    gpus = []
    gpu_idx = 0
    for entry in sorted(Path(gpu_driver_path).iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        # Match PCI BDF format: XXXX:XX:XX.X
        if len(name) < 12 or name[4] != ":" or name[7] != ":" or name[10] != ".":
            continue
        gpu_pcie = name
        gpu_numa = _read_sysfs(str(entry / "numa_node"), "N/A")
        cpus = "N/A"
        if gpu_numa != "N/A" and gpu_numa != "-1":
            cpus = _get_numa_cpus(gpu_numa)
        gpu_name = gpu_names.get(gpu_pcie.lower(), "N/A")
        gpus.append(
            {
                "idx": gpu_idx,
                "pcie": gpu_pcie,
                "numa": gpu_numa,
                "cpus": cpus,
                "name": gpu_name,
            }
        )
        gpu_idx += 1
    return gpus


def _discover_nics() -> List[dict]:
    ib_path = "/sys/class/infiniband"
    if not Path(ib_path).is_dir():
        return []

    nics = []
    for entry in sorted(Path(ib_path).iterdir()):
        if not entry.is_dir() or not entry.name.startswith("mlx5_"):
            continue
        nic_name = entry.name
        device_link = entry / "device"
        nic_pcie = "N/A"
        try:
            nic_pcie = os.path.basename(os.readlink(str(device_link)))
        except OSError:
            pass
        nic_numa = _read_sysfs(str(device_link / "numa_node"), "N/A")
        iface = "N/A"
        net_path = device_link / "net"
        if net_path.is_dir():
            ifaces = sorted(os.listdir(str(net_path)))
            if ifaces:
                iface = ifaces[0]
        nics.append(
            {
                "name": nic_name,
                "pcie": nic_pcie,
                "numa": nic_numa,
                "iface": iface,
            }
        )
    return nics


def topo_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "topo",
        help="Display GPU-to-NIC topology mapping.",
        description="Map NVIDIA GPUs to Mellanox NICs showing PCIe, NUMA, and CPU affinity.",
    )
    parser.set_defaults(func=topo_command)
    return parser


def topo_command(args) -> int:
    console = Console()

    gpus = _discover_gpus()
    nics = _discover_nics()

    if not gpus:
        console.print()
        console.print("[yellow][⚠] No NVIDIA GPUs found in /sys/bus/pci/drivers/nvidia.[/yellow]")
        return 1

    # Detect isolated and IRQ-free cores for color coding.
    isolated = _parse_cpu_list(_read_sysfs("/sys/devices/system/cpu/isolated", ""))
    irq_aff_text = _read_sysfs("/proc/irq/default_smp_affinity", "")
    irq_cpus = _parse_cpu_mask(irq_aff_text) if irq_aff_text else set()

    table = Table(
        title="System Topology",
        box=box.ROUNDED,
        expand=True,
    )
    table.add_column("ID", style="bold cyan", no_wrap=True)
    table.add_column("GPU Name", no_wrap=True)
    table.add_column("GPU PCIe", no_wrap=True)
    table.add_column("NUMA", justify="center", no_wrap=True)
    table.add_column("CPUs", max_width=30)
    table.add_column("NIC", style="bold magenta", no_wrap=True)
    table.add_column("NIC PCIe", no_wrap=True)
    table.add_column("Interface", no_wrap=True)

    for idx, gpu in enumerate(gpus):
        if idx > 0:
            table.add_section()

        matched_nics = [n for n in nics if n["numa"] == gpu["numa"]]
        colored_cpus = _color_cpus(gpu["cpus"], isolated, irq_cpus)

        if not matched_nics:
            table.add_row(
                f"GPU{gpu['idx']}",
                gpu["name"],
                gpu["pcie"],
                gpu["numa"],
                colored_cpus,
                "[dim]none[/dim]",
                "[dim]-[/dim]",
                "[dim]-[/dim]",
            )
        else:
            for i, nic in enumerate(matched_nics):
                if i == 0:
                    table.add_row(
                        f"GPU{gpu['idx']}",
                        gpu["name"],
                        gpu["pcie"],
                        gpu["numa"],
                        colored_cpus,
                        nic["name"],
                        nic["pcie"],
                        nic["iface"],
                    )
                else:
                    table.add_row(
                        "",
                        "",
                        "",
                        "",
                        "",
                        nic["name"],
                        nic["pcie"],
                        nic["iface"],
                    )

    console.print()
    console.print(table)

    if isolated:
        console.print(
            "  [green]\u25a0[/green] isolated + irq-free"
            "  [yellow]\u25a0[/yellow] isolated only"
            "  [cyan]\u25a0[/cyan] irq-free only"
        )
    else:
        console.print(
            "[yellow][\u26a0] No isolated cores detected.[/yellow]\n"
            "[dim]    Transport requires: isolcpus, nohz_full, rcu_nocbs,"
            " irqaffinity kernel parameters.[/dim]"
        )

    console.print()

    return 0
