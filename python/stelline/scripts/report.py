import argparse
import shutil
import subprocess
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import List, Tuple

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

OK = "OK"
WARN = "WARN"
ERROR = "ERROR"
SKIP = "SKIP"

STATUS_ORDER = {
    OK: 0,
    SKIP: 0,
    WARN: 1,
    ERROR: 2,
}

ICONS = {
    OK: "[✔]",
    WARN: "[⚠]",
    ERROR: "[✖]",
    SKIP: "[ ]",
}

COLORS = {
    OK: "green",
    WARN: "yellow",
    ERROR: "red",
    SKIP: "dim",
}

LOGS: List[Tuple[str, str]] = []


class CmdResult:
    def __init__(
        self, cmd: str, status: str, stdout: str = "", stderr: str = "", note: str = ""
    ):
        self.cmd = cmd
        self.status = status
        self.stdout = stdout or ""
        self.stderr = stderr or ""
        self.note = note

    @property
    def text(self) -> str:
        out = self.stdout
        if self.stderr.strip():
            if out.strip():
                out += "\n# stderr\n" + self.stderr
            else:
                out = "# stderr\n" + self.stderr
        if self.note:
            if out.strip():
                out += f"\n# note\n{self.note}"
            else:
                out = f"# note\n{self.note}"
        return out.rstrip() + ("\n" if out else "")


def add_log(status: str, message: str) -> None:
    if status in (WARN, ERROR):
        LOGS.append((status, message))


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run_cmd(cmd: str, required: bool = False, allow_empty: bool = True) -> CmdResult:
    bin_name = cmd.split()[0]
    if (
        "/" not in bin_name
        and "|" not in cmd
        and ">" not in cmd
        and "<" not in cmd
        and " " not in bin_name
        and not have(bin_name)
    ):
        status = ERROR if required else WARN
        note = f"Command '{bin_name}' not found in PATH."
        add_log(status, f"{status} running `{cmd}`: {note}")
        return CmdResult(cmd, status, note=note)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        status = ERROR if required else WARN
        note = f"Exception while running: {e}"
        add_log(status, f"{status} running `{cmd}`: {note}")
        return CmdResult(cmd, status, note=note)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        status = ERROR if required else WARN
        note = f"Exit code {proc.returncode}."
    else:
        if not allow_empty and not stdout.strip() and not stderr.strip():
            status = WARN
            note = "No output captured."
        else:
            status = OK
            note = ""
    if status in (WARN, ERROR):
        add_log(status, f"{status} running `{cmd}`: {note or 'see output'}")
    return CmdResult(cmd, status, stdout=stdout, stderr=stderr, note=note)


def worst_status(statuses: List[str]) -> str:
    if not statuses:
        return SKIP
    return max(statuses, key=lambda s: STATUS_ORDER.get(s, 0))


def write_heading(f, level: int, text: str):
    f.write(f"{'#' * level} {text}\n\n")


def write_block(f, label: str, result: CmdResult, lang: str = "bash"):
    write_heading(f, 3, label)
    body = result.text.strip()
    if not body:
        body = f"# note\nNo data collected for `{result.cmd}` (status={result.status})."
        if result.status in (WARN, ERROR):
            add_log(result.status, body)
    f.write(f"```{lang}\n{body}\n```\n\n")


def checklist_line(section: str, status: str, detail: str = "") -> Text:
    icon = ICONS.get(status, "[ ]")
    color = COLORS.get(status, "white")

    text = Text()
    text.append(f"{icon} ", style=color)
    text.append(section, style="bold")
    text.append(f" - {status}", style=color)

    if detail:
        text.append(f" ({detail})", style="dim")

    return text


def detect_pci_functions(patterns: List[str]) -> List[Tuple[str, str]]:
    if not have("lspci"):
        return []
    res = run_cmd("lspci -D", required=False)
    if not res.stdout.strip():
        return []
    devs: List[Tuple[str, str]] = []
    for line in res.stdout.splitlines():
        if all(p.lower() in line.lower() for p in patterns):
            parts = line.split()
            if parts:
                bdf = parts[0]
                devs.append((bdf, line.strip()))
    return devs


def section_metadata(f) -> str:
    write_heading(f, 1, "System Report")
    host = run_cmd("hostname", required=False, allow_empty=False)
    kernel = run_cmd("uname -r", required=False, allow_empty=False)
    os_info = run_cmd(
        "grep PRETTY_NAME= /etc/os-release || cat /etc/os-release", required=False
    )
    os_text = os_info.stdout.strip()
    if os_text.startswith("PRETTY_NAME="):
        os_text = os_text.split("=", 1)[1].strip().strip('"')
    f.write(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`\n")
    f.write(f"- Hostname: `{(host.stdout.strip() or 'N/A')}`\n")
    f.write(f"- Kernel: `{(kernel.stdout.strip() or 'N/A')}`\n")
    f.write(f"- OS: `{(os_text or 'N/A')}`\n\n")
    return worst_status([host.status, kernel.status, os_info.status])


def section_pcie_topology(f) -> str:
    write_heading(f, 2, "PCIe Topology")
    if not have("lspci"):
        msg = "`lspci` not found; cannot capture PCIe topology."
        f.write(f"_ERROR: {msg}_\n\n")
        add_log(ERROR, msg)
        return ERROR
    topo = run_cmd("lspci -tv", required=True, allow_empty=False)
    write_block(f, "lspci -tv", topo)
    return topo.status


def section_numa(f) -> str:
    write_heading(f, 2, "NUMA Topology")
    statuses: List[str] = []
    if have("numactl"):
        numa = run_cmd("numactl --hardware", required=False)
        write_block(f, "numactl --hardware", numa)
        statuses.append(numa.status)
    else:
        msg = "`numactl` not found; skipping `numactl --hardware`."
        f.write(f"_WARN: {msg}_\n\n")
        add_log(WARN, msg)
        statuses.append(WARN)
    lscpu = run_cmd("lscpu", required=False)
    write_block(f, "lscpu", lscpu)
    statuses.append(lscpu.status)
    return worst_status(statuses)


def section_pci_devices(f) -> str:
    write_heading(f, 2, "PCIe Devices Details")
    if not have("lspci"):
        msg = "`lspci` not found; skipping device details."
        f.write(f"_ERROR: {msg}_\n\n")
        add_log(ERROR, msg)
        return ERROR
    statuses: List[str] = []
    gpus = detect_pci_functions(["VGA compatible controller", "NVIDIA"])
    if gpus:
        for idx, (bdf, line) in enumerate(gpus):
            res = run_cmd(f"lspci -vvv -s {bdf}", required=False)
            write_block(f, f"GPU #{idx} ({bdf}) — {line}", res)
            statuses.append(res.status)
    else:
        msg = "No NVIDIA VGA GPUs detected via `lspci`."
        f.write(f"_WARN: {msg}_\n\n")
        add_log(WARN, msg)
        statuses.append(WARN)
    nics = detect_pci_functions(["Mellanox", "Ethernet controller"])
    if nics:
        for idx, (bdf, line) in enumerate(nics):
            res = run_cmd(f"lspci -vvv -s {bdf}", required=False)
            write_block(f, f"ConnectX / Mellanox #{idx} ({bdf}) — {line}", res)
            statuses.append(res.status)
    else:
        msg = "No Mellanox / ConnectX devices detected via `lspci`."
        f.write(f"_WARN: {msg}_\n\n")
        add_log(WARN, msg)
        statuses.append(WARN)
    highpoint = detect_pci_functions(["HighPoint", "RAID bus controller"])
    if highpoint:
        for idx, (bdf, line) in enumerate(highpoint):
            res = run_cmd(f"lspci -vvv -s {bdf}", required=False)
            write_block(f, f"HighPoint Carrier Board #{idx} ({bdf}) — {line}", res)
            statuses.append(res.status)
    return worst_status(statuses)


def section_nvidia_smi(f) -> str:
    write_heading(f, 2, "NVIDIA SMI")
    if not have("nvidia-smi"):
        msg = "`nvidia-smi` not found; skipping GPU runtime info."
        f.write(f"_WARN: {msg}_\n\n")
        add_log(WARN, msg)
        return WARN
    statuses: List[str] = []
    basic = run_cmd("nvidia-smi", required=False)
    write_block(f, "nvidia-smi", basic)
    statuses.append(basic.status)
    topo = run_cmd("nvidia-smi topo -m", required=False)
    write_block(f, "nvidia-smi topo -m", topo)
    statuses.append(topo.status)
    return worst_status(statuses)


def section_cuda_bandwidth(f) -> str:
    write_heading(f, 2, "CUDA Bandwidth / P2P Tests")
    if not have("nvidia-smi"):
        msg = "No NVIDIA GPUs detected; skipping CUDA tests."
        f.write(f"_WARN: {msg}_\n\n")
        add_log(WARN, msg)
        return WARN
    statuses: List[str] = []
    p2p = shutil.which("p2pBandwidthLatencyTest")
    if p2p:
        res = run_cmd("p2pBandwidthLatencyTest", required=False)
        write_block(f, "p2pBandwidthLatencyTest", res)
        statuses.append(res.status)
    else:
        msg = "`p2pBandwidthLatencyTest` not found in PATH; install CUDA samples."
        f.write(f"_WARN: {msg}_\n\n")
        add_log(WARN, msg)
        statuses.append(WARN)
    bw = shutil.which("bandwidthTest")
    if bw:
        smi_out = run_cmd(
            "nvidia-smi --query-gpu=index --format=csv,noheader", required=False
        )
        if smi_out.stdout.strip():
            for line in smi_out.stdout.splitlines():
                idx = line.strip()
                if idx.isdigit():
                    res = run_cmd(f"bandwidthTest --device={idx}", required=False)
                    write_block(f, f"bandwidthTest --device={idx}", res)
                    statuses.append(res.status)
        else:
            msg = "Could not determine GPU indices from `nvidia-smi`; skipping bandwidthTest runs."
            f.write(f"_WARN: {msg}_\n\n")
            add_log(WARN, msg)
            statuses.append(WARN)
    else:
        msg = "`bandwidthTest` not found in PATH; install CUDA samples."
        f.write(f"_WARN: {msg}_\n\n")
        add_log(WARN, msg)
        statuses.append(WARN)
    return worst_status(statuses)


def section_storage(f) -> str:
    write_heading(f, 2, "Persistent Storage")
    if have("lsblk"):
        res = run_cmd("lsblk", required=False)
        write_block(f, "lsblk", res)
        return res.status
    msg = "`lsblk` not found; skipping block device inventory."
    f.write(f"_WARN: {msg}_\n\n")
    add_log(WARN, msg)
    return WARN


def section_connectx(f) -> str:
    write_heading(f, 2, "ConnectX / RDMA Configuration")
    if have("ibv_devinfo"):
        res = run_cmd("ibv_devinfo", required=False)
        write_block(f, "ibv_devinfo", res)
        return res.status
    msg = "`ibv_devinfo` not found; skipping RDMA HCA details."
    f.write(f"_WARN: {msg}_\n\n")
    add_log(WARN, msg)
    return WARN


def section_grub(f) -> str:
    write_heading(f, 2, "GRUB Configuration")
    grub_path = Path("/etc/default/grub")
    if grub_path.exists():
        try:
            content = grub_path.read_text()
            res = CmdResult("cat /etc/default/grub", OK, stdout=content)
        except Exception as e:
            note = f"Failed to read grub file: {e}"
            res = CmdResult("cat /etc/default/grub", ERROR, note=note)
            add_log(ERROR, note)
        write_block(f, "/etc/default/grub", res, lang="")
        return res.status
    msg = "`/etc/default/grub` not found."
    f.write(f"_WARN: {msg}_\n\n")
    add_log(WARN, msg)
    return WARN


def section_acs_state(f) -> str:
    write_heading(f, 2, "PCIe ACS State")
    if not have("lspci"):
        msg = "`lspci` not found; cannot query ACSCtl."
        f.write(f"_WARN: {msg}_\n\n")
        add_log(WARN, msg)
        return WARN
    res = run_cmd("lspci -vvv | grep -i ACSCtl || true", required=False)
    if res.stdout.strip():
        write_block(f, "lspci -vvv | grep -i ACSCtl", res)
    else:
        note = "No ACSCtl capabilities reported by lspci."
        res = CmdResult("lspci -vvv | grep -i ACSCtl || true", OK, note=note)
        write_block(f, "lspci -vvv | grep -i ACSCtl", res)
    return res.status


def section_gds(f) -> str:
    write_heading(f, 2, "GPUDirect Storage (GDS)")
    candidates = [
        "/usr/local/cuda/gds/tools/gdscheck",
        "gdscheck",
    ]
    gds_cmd = next((c for c in candidates if have(c)), None)
    if not gds_cmd:
        msg = "`gdscheck` not found; skipping GDS diagnostics."
        f.write(f"_WARN: {msg}_\n\n")
        add_log(WARN, msg)
        return WARN
    res = run_cmd(f"{gds_cmd} -p", required=False)
    write_block(f, f"{gds_cmd} -p", res)
    return res.status


# Mapping of filter names to section functions and display names
SECTION_MAP = {
    "metadata": ("Metadata", section_metadata),
    "pcie_topology": ("PCIe Topology", section_pcie_topology),
    "numa": ("NUMA Topology", section_numa),
    "pcie_devices": ("PCIe Devices Details", section_pci_devices),
    "nvidia_smi": ("NVIDIA SMI", section_nvidia_smi),
    "cuda_bandwidth": ("CUDA Bandwidth / P2P Tests", section_cuda_bandwidth),
    "storage": ("Persistent Storage", section_storage),
    "connectx": ("ConnectX / RDMA Configuration", section_connectx),
    "grub": ("GRUB Configuration", section_grub),
    "acs_state": ("PCIe ACS State", section_acs_state),
    "gds": ("GPUDirect Storage (GDS)", section_gds),
}


def generate_report_markdown(
    filter_sections: str = "all",
) -> Tuple[str, List[Tuple[str, str]]]:
    global LOGS
    LOGS = []
    buf = StringIO()
    section_statuses: List[Tuple[str, str]] = []

    if filter_sections == "all":
        sections_to_run = list(SECTION_MAP.keys())
    else:
        sections_to_run = [
            s.strip() for s in filter_sections.split(",") if s.strip() in SECTION_MAP
        ]
        if not sections_to_run:
            sections_to_run = list(SECTION_MAP.keys())

    for section_key in sections_to_run:
        display_name, section_func = SECTION_MAP[section_key]
        section_statuses.append((display_name, section_func(buf)))

    return buf.getvalue(), section_statuses


def report_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "report",
        help="Generate a hardware/PCIe/GPU/GDS markdown system report.",
        description="Generate a markdown report with PCIe, NUMA, GPU, NIC, storage, and GDS info.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="report.md",
        help="Output markdown file path (default: report.md).",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        dest="print_only",
        help="Print report markdown to stdout instead of writing a file.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="all",
        help="Filter sections to run: all, metadata, pcie_topology, numa, pcie_devices, nvidia_smi, cuda_bandwidth, storage, connectx, grub, acs_state, gds (default: all)",
    )
    parser.set_defaults(func=report_command)
    return parser


def print_report_sections(console: Console, md: str) -> None:
    lines = md.split("\n")
    current_section = ""
    current_content = []

    for line in lines:
        if line.startswith("## "):
            if current_section and current_content:
                content_text = "\n".join(current_content).strip()
                if content_text:
                    console.print(
                        Panel(
                            content_text,
                            title=current_section,
                            border_style="blue",
                            expand=True,
                        )
                    )
                    console.print()

            current_section = line[3:].strip()
            current_content = []
        elif line.startswith("# "):
            continue
        else:
            current_content.append(line)

    if current_section and current_content:
        content_text = "\n".join(current_content).strip()
        if content_text:
            console.print(
                Panel(
                    content_text,
                    title=current_section,
                    border_style="blue",
                    expand=True,
                )
            )


def report_command(args) -> int:
    console = Console()
    md, section_statuses = generate_report_markdown(args.filter)
    overall = worst_status([s for _, s in section_statuses])

    if args.print_only:
        print_report_sections(console, md)
    else:
        out_path = Path(args.file)
        try:
            out_path.write_text(md)
        except Exception as e:
            error_msg = f"Failed to write report to {out_path}: {e}"
            console.print(f"[✖] {error_msg}", style="red")
            add_log(ERROR, error_msg)
            overall = ERROR if overall == ERROR else WARN

    console.print()

    summary_table = Table(
        title="System Report Summary", box=box.ROUNDED, show_header=False, expand=True
    )
    summary_table.add_column("Status", style="bold")

    for name, status in section_statuses:
        status_text = Text()
        status_text.append(
            f"{ICONS.get(status, '[ ]')} ", style=COLORS.get(status, "white")
        )
        status_text.append(name, style="bold")
        status_text.append(f" - {status}", style=COLORS.get(status, "white"))

        summary_table.add_row(status_text)

    console.print(summary_table)

    if LOGS:
        console.print()
        log_table = Table(
            title="Warnings and Errors",
            box=box.ROUNDED,
            show_header=False,
            expand=True,
            border_style="yellow",
            title_style="yellow",
        )
        log_table.add_column("Messages", style="bold")

        for status, message in LOGS:
            log_text = Text()
            log_text.append(
                f"{ICONS.get(status, '[ ]')} {status}: ",
                style=COLORS.get(status, "white"),
            )
            log_text.append(message, style="dim")
            log_table.add_row(log_text)

        console.print(log_table)

    if not args.print_only:
        console.print()
        console.print(
            f"Report written to: [bold green]{Path(args.file).resolve()}[/bold green]"
        )

    return 1 if overall == ERROR else 0
