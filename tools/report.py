#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text


OK = "OK"
WARN = "WARN"
ERROR = "ERROR"
SKIP = "SKIP"
STATUSES = {OK, WARN, ERROR, SKIP}
STATUS_ORDER = {OK: 0, SKIP: 0, WARN: 1, ERROR: 2}
STATUS_STYLES = {OK: "bold green", WARN: "bold yellow", ERROR: "bold red", SKIP: "dim"}


@dataclass(frozen=True)
class LogEntry:
    section: str
    status: str
    message: str


@dataclass(frozen=True)
class CmdResult:
    cmd: str
    status: str
    stdout: str
    stderr: str
    note: str
    text: str


@dataclass
class ReportContext:
    current_section: str
    logs: list[LogEntry]


@dataclass(frozen=True)
class ReportData:
    markdown: str
    section_statuses: list[tuple[str, str]]
    logs: list[LogEntry]
    overall_status: str


@dataclass(frozen=True)
class ReportWriteResult:
    markdown: str
    section_statuses: list[tuple[str, str]]
    logs: list[LogEntry]
    overall_status: str
    output_path: str
    output_message: str
    write_succeeded: bool


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def expand_tabs(text: str, tab_width: int = 4) -> str:
    return "\n".join(line.expandtabs(tab_width) for line in text.split("\n"))


def normalize_command_text(text: str) -> str:
    return expand_tabs(strip_ansi(text).replace("\r\n", "\n").replace("\r", "\n"))


def have(cmd: str) -> bool:
    if "/" in cmd:
        return os.access(cmd, os.X_OK)
    return shutil.which(cmd) is not None


def add_log(ctx: ReportContext, status: str, message: str) -> None:
    if status in {WARN, ERROR}:
        ctx.logs.append(LogEntry(ctx.current_section, status, message))


def build_cmd_result(
    cmd: str,
    status: str,
    stdout: str = "",
    stderr: str = "",
    note: str = "",
) -> CmdResult:
    normalized_stdout = normalize_command_text(stdout)
    normalized_stderr = normalize_command_text(stderr)
    normalized_note = normalize_command_text(note)
    parts: list[str] = []

    if normalized_stdout.strip():
        parts.append(normalized_stdout.rstrip())
    if normalized_stderr.strip():
        parts.append(f"# stderr\n{normalized_stderr.rstrip()}")
    if normalized_note:
        parts.append(f"# note\n{normalized_note}")

    text = "\n".join(parts) + "\n" if parts else ""

    return CmdResult(
        cmd=cmd,
        status=status,
        stdout=normalized_stdout,
        stderr=normalized_stderr,
        note=normalized_note,
        text=text,
    )


def run_cmd(
    ctx: ReportContext,
    cmd: str,
    required: bool = False,
    allow_empty: bool = True,
) -> CmdResult:
    bin_name = cmd.strip().split(maxsplit=1)[0] if cmd.strip() else ""
    has_shell_syntax = any(token in cmd for token in ("|", ">", "<"))

    if bin_name and "/" not in bin_name and not has_shell_syntax and not have(bin_name):
        status = ERROR if required else WARN
        note = f"Command '{bin_name}' not found in PATH."
        add_log(ctx, status, f"{status} running `{cmd}`: {note}")
        return build_cmd_result(cmd, status, note=note)

    try:
        proc = subprocess.run(
            ["bash", "-lc", cmd],
            check=False,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        status = ERROR if required else WARN
        note = f"Exception while running: {error}"
        add_log(ctx, status, f"{status} running `{cmd}`: {note}")
        return build_cmd_result(cmd, status, note=note)

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    note = ""

    if proc.returncode != 0:
        status = ERROR if required else WARN
        note = f"Exit code {proc.returncode}."
    elif not allow_empty and not stdout.strip() and not stderr.strip():
        status = WARN
        note = "No output captured."
    else:
        status = OK

    if status in {WARN, ERROR}:
        add_log(ctx, status, f"{status} running `{cmd}`: {note or 'see output'}")

    return build_cmd_result(cmd, status, stdout, stderr, note)


def worst_status(statuses: list[str]) -> str:
    if not statuses:
        return SKIP
    return max(statuses, key=lambda status: STATUS_ORDER[status])


def write_heading(chunks: list[str], level: int, text: str) -> None:
    chunks.append(f"{'#' * level} {text}\n\n")


def write_notice(chunks: list[str], severity: str, message: str) -> None:
    chunks.append(f"_{severity}: {message}_\n\n")


def write_block(
    ctx: ReportContext,
    chunks: list[str],
    label: str,
    result: CmdResult,
    lang: str = "bash",
) -> None:
    write_heading(chunks, 3, label)

    body = result.text.strip()
    if not body:
        body = f"# note\nNo data collected for `{result.cmd}` (status={result.status})."
        if result.status in {WARN, ERROR}:
            add_log(ctx, result.status, body)

    chunks.append(f"```{lang}\n{body}\n```\n\n")


def detect_pci_functions(ctx: ReportContext, patterns: list[str]) -> list[tuple[str, str]]:
    if not have("lspci"):
        return []

    result = run_cmd(ctx, "lspci -D")
    if not result.stdout.strip():
        return []

    devices: list[tuple[str, str]] = []
    lowered_patterns = [pattern.lower() for pattern in patterns]
    for line in result.stdout.splitlines():
        if not line:
            continue
        lowered_line = line.lower()
        if all(pattern in lowered_line for pattern in lowered_patterns):
            bdf = line.strip().split(maxsplit=1)[0]
            if bdf:
                devices.append((bdf, line.strip()))

    return devices


def now_iso_seconds() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def section_metadata(ctx: ReportContext, chunks: list[str]) -> str:
    write_heading(chunks, 1, "System Report")

    host = run_cmd(ctx, "hostname", allow_empty=False)
    kernel = run_cmd(ctx, "uname -r", allow_empty=False)
    os_info = run_cmd(ctx, "grep PRETTY_NAME= /etc/os-release || cat /etc/os-release")

    os_text = os_info.stdout.strip()
    if os_text.startswith("PRETTY_NAME="):
        os_text = os_text.split("=", 1)[1].strip().strip('"')

    chunks.append(f"- Generated: `{now_iso_seconds()}`\n")
    chunks.append(f"- Hostname: `{host.stdout.strip() or 'N/A'}`\n")
    chunks.append(f"- Kernel: `{kernel.stdout.strip() or 'N/A'}`\n")
    chunks.append(f"- OS: `{os_text or 'N/A'}`\n\n")

    return worst_status([host.status, kernel.status, os_info.status])


def section_pcie_topology(ctx: ReportContext, chunks: list[str]) -> str:
    write_heading(chunks, 2, "PCIe Topology")

    if not have("lspci"):
        message = "`lspci` not found; cannot capture PCIe topology."
        write_notice(chunks, ERROR, message)
        add_log(ctx, ERROR, message)
        return ERROR

    topology = run_cmd(ctx, "lspci -tv", required=True, allow_empty=False)
    write_block(ctx, chunks, "lspci -tv", topology)
    return topology.status


def section_numa(ctx: ReportContext, chunks: list[str]) -> str:
    statuses: list[str] = []
    write_heading(chunks, 2, "NUMA Topology")

    if have("numactl"):
        numa = run_cmd(ctx, "numactl --hardware")
        write_block(ctx, chunks, "numactl --hardware", numa)
        statuses.append(numa.status)
    else:
        message = "`numactl` not found; skipping `numactl --hardware`."
        write_notice(chunks, WARN, message)
        add_log(ctx, WARN, message)
        statuses.append(WARN)

    lscpu = run_cmd(ctx, "lscpu")
    write_block(ctx, chunks, "lscpu", lscpu)
    statuses.append(lscpu.status)

    return worst_status(statuses)


def section_pci_devices(ctx: ReportContext, chunks: list[str]) -> str:
    statuses: list[str] = []
    write_heading(chunks, 2, "PCIe Devices Details")

    if not have("lspci"):
        message = "`lspci` not found; skipping device details."
        write_notice(chunks, ERROR, message)
        add_log(ctx, ERROR, message)
        return ERROR

    gpus = detect_pci_functions(ctx, ["VGA compatible controller", "NVIDIA"])
    if gpus:
        for index, (bdf, line) in enumerate(gpus):
            result = run_cmd(ctx, f"lspci -vvv -s {bdf}")
            write_block(ctx, chunks, f"GPU #{index} ({bdf}) - {line}", result)
            statuses.append(result.status)
    else:
        message = "No NVIDIA VGA GPUs detected via `lspci`."
        write_notice(chunks, WARN, message)
        add_log(ctx, WARN, message)
        statuses.append(WARN)

    nics = detect_pci_functions(ctx, ["Mellanox", "Ethernet controller"])
    if nics:
        for index, (bdf, line) in enumerate(nics):
            result = run_cmd(ctx, f"lspci -vvv -s {bdf}")
            write_block(
                ctx,
                chunks,
                f"ConnectX / Mellanox #{index} ({bdf}) - {line}",
                result,
            )
            statuses.append(result.status)
    else:
        message = "No Mellanox / ConnectX devices detected via `lspci`."
        write_notice(chunks, WARN, message)
        add_log(ctx, WARN, message)
        statuses.append(WARN)

    highpoint = detect_pci_functions(ctx, ["HighPoint", "RAID bus controller"])
    for index, (bdf, line) in enumerate(highpoint):
        result = run_cmd(ctx, f"lspci -vvv -s {bdf}")
        write_block(
            ctx,
            chunks,
            f"HighPoint Carrier Board #{index} ({bdf}) - {line}",
            result,
        )
        statuses.append(result.status)

    return worst_status(statuses)


def section_nvidia_smi(ctx: ReportContext, chunks: list[str]) -> str:
    statuses: list[str] = []
    write_heading(chunks, 2, "NVIDIA SMI")

    if not have("nvidia-smi"):
        message = "`nvidia-smi` not found; skipping GPU runtime info."
        write_notice(chunks, WARN, message)
        add_log(ctx, WARN, message)
        return WARN

    basic = run_cmd(ctx, "nvidia-smi")
    write_block(ctx, chunks, "nvidia-smi", basic)
    statuses.append(basic.status)

    topo = run_cmd(ctx, "nvidia-smi topo -m")
    write_block(ctx, chunks, "nvidia-smi topo -m", topo)
    statuses.append(topo.status)

    return worst_status(statuses)


def section_cuda_bandwidth(ctx: ReportContext, chunks: list[str]) -> str:
    statuses: list[str] = []
    write_heading(chunks, 2, "CUDA Bandwidth / P2P Tests")

    if not have("nvidia-smi"):
        message = "No NVIDIA GPUs detected; skipping CUDA tests."
        write_notice(chunks, WARN, message)
        add_log(ctx, WARN, message)
        return WARN

    if have("p2pBandwidthLatencyTest"):
        result = run_cmd(ctx, "p2pBandwidthLatencyTest")
        write_block(ctx, chunks, "p2pBandwidthLatencyTest", result)
        statuses.append(result.status)
    else:
        message = "`p2pBandwidthLatencyTest` not found in PATH; install CUDA samples."
        write_notice(chunks, WARN, message)
        add_log(ctx, WARN, message)
        statuses.append(WARN)

    if have("bandwidthTest"):
        smi_out = run_cmd(ctx, "nvidia-smi --query-gpu=index --format=csv,noheader")
        if smi_out.stdout.strip():
            for line in smi_out.stdout.splitlines():
                index = line.strip()
                if not re.fullmatch(r"\d+", index):
                    continue
                result = run_cmd(ctx, f"bandwidthTest --device={index}")
                write_block(ctx, chunks, f"bandwidthTest --device={index}", result)
                statuses.append(result.status)
        else:
            message = (
                "Could not determine GPU indices from `nvidia-smi`; "
                "skipping bandwidthTest runs."
            )
            write_notice(chunks, WARN, message)
            add_log(ctx, WARN, message)
            statuses.append(WARN)
    else:
        message = "`bandwidthTest` not found in PATH; install CUDA samples."
        write_notice(chunks, WARN, message)
        add_log(ctx, WARN, message)
        statuses.append(WARN)

    return worst_status(statuses)


def section_storage(ctx: ReportContext, chunks: list[str]) -> str:
    write_heading(chunks, 2, "Persistent Storage")

    if have("lsblk"):
        result = run_cmd(ctx, "lsblk")
        write_block(ctx, chunks, "lsblk", result)
        return result.status

    message = "`lsblk` not found; skipping block device inventory."
    write_notice(chunks, WARN, message)
    add_log(ctx, WARN, message)
    return WARN


def section_connectx(ctx: ReportContext, chunks: list[str]) -> str:
    write_heading(chunks, 2, "ConnectX / RDMA Configuration")

    if have("ibv_devinfo"):
        result = run_cmd(ctx, "ibv_devinfo")
        write_block(ctx, chunks, "ibv_devinfo", result)
        return result.status

    message = "`ibv_devinfo` not found; skipping RDMA HCA details."
    write_notice(chunks, WARN, message)
    add_log(ctx, WARN, message)
    return WARN


def section_grub(ctx: ReportContext, chunks: list[str]) -> str:
    write_heading(chunks, 2, "GRUB Configuration")

    try:
        content = Path("/etc/default/grub").read_text(encoding="utf-8")
        write_block(ctx, chunks, "/etc/default/grub", build_cmd_result("cat /etc/default/grub", OK, content), "")
        return OK
    except OSError as error:
        if isinstance(error, FileNotFoundError):
            message = "`/etc/default/grub` not found."
            severity = WARN
        else:
            message = f"Failed to read grub file: {error}"
            severity = ERROR
        write_notice(chunks, severity, message)
        add_log(ctx, severity, message)
        return severity


def section_acs_state(ctx: ReportContext, chunks: list[str]) -> str:
    write_heading(chunks, 2, "PCIe ACS State")

    if not have("lspci"):
        message = "`lspci` not found; cannot query ACSCtl."
        write_notice(chunks, WARN, message)
        add_log(ctx, WARN, message)
        return WARN

    result = run_cmd(ctx, "lspci -vvv | grep -i ACSCtl || true")
    if result.stdout.strip():
        write_block(ctx, chunks, "lspci -vvv | grep -i ACSCtl", result)
        return result.status

    write_block(
        ctx,
        chunks,
        "lspci -vvv | grep -i ACSCtl",
        build_cmd_result(
            "lspci -vvv | grep -i ACSCtl || true",
            OK,
            note="No ACSCtl capabilities reported by lspci.",
        ),
    )
    return OK


def section_gds(ctx: ReportContext, chunks: list[str]) -> str:
    write_heading(chunks, 2, "GPUDirect Storage (GDS)")

    command = next(
        (candidate for candidate in ["/usr/local/cuda/gds/tools/gdscheck", "gdscheck"] if have(candidate)),
        None,
    )
    if not command:
        message = "`gdscheck` not found; skipping GDS diagnostics."
        write_notice(chunks, WARN, message)
        add_log(ctx, WARN, message)
        return WARN

    result = run_cmd(ctx, f"{command} -p")
    write_block(ctx, chunks, f"{command} -p", result)
    return result.status


SECTIONS = [
    ("metadata", "Metadata", section_metadata),
    ("pcie_topology", "PCIe Topology", section_pcie_topology),
    ("numa", "NUMA Topology", section_numa),
    ("pcie_devices", "PCIe Devices Details", section_pci_devices),
    ("nvidia_smi", "NVIDIA SMI", section_nvidia_smi),
    ("cuda_bandwidth", "CUDA Bandwidth / P2P Tests", section_cuda_bandwidth),
    ("storage", "Persistent Storage", section_storage),
    ("connectx", "ConnectX / RDMA Configuration", section_connectx),
    ("grub", "GRUB Configuration", section_grub),
    ("acs_state", "PCIe ACS State", section_acs_state),
    ("gds", "GPUDirect Storage (GDS)", section_gds),
]


def generate_report_data(on_section: Callable[[str], None] | None = None) -> ReportData:
    ctx = ReportContext(current_section="Report", logs=[])
    chunks: list[str] = []
    section_statuses: list[tuple[str, str]] = []

    for _key, display_name, section_fn in SECTIONS:
        if on_section:
            on_section(display_name)
        ctx.current_section = display_name
        section_statuses.append((display_name, section_fn(ctx, chunks)))

    return ReportData(
        markdown="".join(chunks),
        section_statuses=section_statuses,
        logs=ctx.logs,
        overall_status=worst_status([status for _name, status in section_statuses]),
    )


def write_report_file(
    output_path: str = "report.md",
    on_section: Callable[[str], None] | None = None,
) -> ReportWriteResult:
    report = generate_report_data(on_section)
    output_message = f"Report written to: {output_path}"
    overall_status = report.overall_status
    write_succeeded = False

    try:
        Path(output_path).write_text(report.markdown, encoding="utf-8")
        write_succeeded = True
    except OSError as error:
        output_message = f"Failed to write report to {output_path}: {error}"
        overall_status = ERROR

    return ReportWriteResult(
        markdown=report.markdown,
        section_statuses=report.section_statuses,
        logs=list(report.logs),
        overall_status=overall_status,
        output_path=output_path,
        output_message=output_message,
        write_succeeded=write_succeeded,
    )


def report_exit_code(status: str) -> int:
    return 1 if status == ERROR else 0


def status_text(status: str) -> Text:
    return Text(status, style=STATUS_STYLES[status])


def print_summary(report: ReportWriteResult, console: Console) -> None:
    table = Table(
        title="Stelline System Report",
        title_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
    )
    table.add_column("Section")
    table.add_column("Status", justify="center")

    for name, status in report.section_statuses:
        table.add_row(name, status_text(status))

    table.add_section()
    table.add_row(Text("Overall", style="bold"), status_text(report.overall_status))
    console.print(table)

    if report.logs:
        console.print(Text("Issues", style="bold"))
        for log in report.logs:
            console.print(
                Text.assemble(
                    ("  • ", "dim"),
                    (f"[{log.status}] ", STATUS_STYLES[log.status]),
                    (f"{log.section}: ", "bold"),
                    log.message,
                )
            )
        console.print()

    if report.write_succeeded:
        console.print(
            Text.assemble(
                ("✔ ", "bold green"),
                "Report written to: ",
                (str(Path(report.output_path).resolve()), "cyan"),
            )
        )
    else:
        Console(stderr=True).print(
            Text.assemble(("✘ ", "bold red"), report.output_message)
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Stelline markdown system report.")
    parser.add_argument(
        "--file",
        nargs="?",
        const="report.md",
        default="report.md",
        metavar="PATH",
        help="Write markdown report to PATH. Defaults to report.md.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    console = Console(highlight=False)

    with console.status("[cyan]Collecting system information...[/]") as status:
        report = write_report_file(
            args.file,
            on_section=lambda name: status.update(f"[cyan]Collecting:[/] {name}"),
        )

    print_summary(report, console)
    return report_exit_code(report.overall_status)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
