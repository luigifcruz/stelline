import { spawnSync } from "node:child_process";
import { accessSync, constants, readFileSync, writeFileSync } from "node:fs";

export const OK = "OK";
export const WARN = "WARN";
export const ERROR = "ERROR";
export const SKIP = "SKIP";

export type Status = typeof OK | typeof WARN | typeof ERROR | typeof SKIP;
export interface LogEntry {
    section: string;
    status: Status;
    message: string;
}
export type SectionStatus = [string, Status];

interface CmdResult {
    cmd: string;
    status: Status;
    stdout: string;
    stderr: string;
    note: string;
    text: string;
}

interface ReportContext {
    currentSection: string;
    logs: LogEntry[];
}

type SectionFn = (ctx: ReportContext, chunks: string[]) => Status;

export interface ReportData {
    markdown: string;
    sectionStatuses: SectionStatus[];
    logs: LogEntry[];
    overallStatus: Status;
}

export interface ReportWriteResult extends ReportData {
    outputPath: string;
    outputMessage: string;
    writeSucceeded: boolean;
}

const STATUS_ORDER: Record<Status, number> = {
    OK: 0,
    SKIP: 0,
    WARN: 1,
    ERROR: 2,
};

function shellQuote(text: string): string {
    return `'${text.replace(/'/g, "'\"'\"'")}'`;
}

function stripAnsi(text: string): string {
    return text.replace(/\u001b\[[0-?]*[ -/]*[@-~]/g, "");
}

function expandTabs(text: string, tabWidth = 4): string {
    return text
        .split("\n")
        .map((line) => {
            let column = 0;
            let output = "";

            for (const char of line) {
                if (char === "\t") {
                    const spaces = tabWidth - (column % tabWidth || 0);
                    output += " ".repeat(spaces);
                    column += spaces;
                    continue;
                }

                output += char;
                column += 1;
            }

            return output;
        })
        .join("\n");
}

function normalizeCommandText(text: string): string {
    return expandTabs(stripAnsi(text).replace(/\r\n/g, "\n").replace(/\r/g, "\n"));
}

function have(cmd: string): boolean {
    if (cmd.includes("/")) {
        try {
            accessSync(cmd, constants.X_OK);
            return true;
        } catch {
            return false;
        }
    }

    const proc = spawnSync(
        "bash",
        ["-lc", `command -v ${shellQuote(cmd)} >/dev/null 2>&1`],
        { stdio: "ignore" },
    );

    return proc.status === 0;
}

function buildCmdResult(
    cmd: string,
    status: Status,
    stdout = "",
    stderr = "",
    note = "",
): CmdResult {
    const normalizedStdout = normalizeCommandText(stdout);
    const normalizedStderr = normalizeCommandText(stderr);
    const normalizedNote = normalizeCommandText(note);
    const parts: string[] = [];

    if (normalizedStdout.trim()) {
        parts.push(normalizedStdout.trimEnd());
    }

    if (normalizedStderr.trim()) {
        parts.push(`# stderr\n${normalizedStderr.trimEnd()}`);
    }

    if (normalizedNote) {
        parts.push(`# note\n${normalizedNote}`);
    }

    return {
        cmd,
        status,
        stdout: normalizedStdout,
        stderr: normalizedStderr,
        note: normalizedNote,
        text: parts.length > 0 ? `${parts.join("\n")}\n` : "",
    };
}

function addLog(ctx: ReportContext, status: Status, message: string): void {
    if (status === WARN || status === ERROR) {
        ctx.logs.push({
            section: ctx.currentSection,
            status,
            message,
        });
    }
}

function runCmd(
    ctx: ReportContext,
    cmd: string,
    required = false,
    allowEmpty = true,
): CmdResult {
    const binName = cmd.trim().split(/\s+/, 1)[0] ?? "";

    if (
        !binName.includes("/") &&
        !cmd.includes("|") &&
        !cmd.includes(">") &&
        !cmd.includes("<") &&
        binName.length > 0 &&
        !have(binName)
    ) {
        const status: Status = required ? ERROR : WARN;
        const note = `Command '${binName}' not found in PATH.`;
        addLog(ctx, status, `${status} running \`${cmd}\`: ${note}`);
        return buildCmdResult(cmd, status, "", "", note);
    }

    let proc;

    try {
        proc = spawnSync("bash", ["-lc", cmd], {
            encoding: "utf8",
            maxBuffer: 10 * 1024 * 1024,
        });
    } catch (error) {
        const status: Status = required ? ERROR : WARN;
        const note = `Exception while running: ${String(error)}`;
        addLog(ctx, status, `${status} running \`${cmd}\`: ${note}`);
        return buildCmdResult(cmd, status, "", "", note);
    }

    if (proc.error) {
        const status: Status = required ? ERROR : WARN;
        const note = `Exception while running: ${proc.error.message}`;
        addLog(ctx, status, `${status} running \`${cmd}\`: ${note}`);
        return buildCmdResult(cmd, status, "", "", note);
    }

    const stdout = proc.stdout ?? "";
    const stderr = proc.stderr ?? "";
    let status: Status;
    let note = "";

    if (proc.status !== 0) {
        status = required ? ERROR : WARN;
        note = `Exit code ${proc.status ?? 1}.`;
    } else if (!allowEmpty && !stdout.trim() && !stderr.trim()) {
        status = WARN;
        note = "No output captured.";
    } else {
        status = OK;
    }

    if (status === WARN || status === ERROR) {
        addLog(ctx, status, `${status} running \`${cmd}\`: ${note || "see output"}`);
    }

    return buildCmdResult(cmd, status, stdout, stderr, note);
}

function worstStatus(statuses: Status[]): Status {
    if (statuses.length === 0) {
        return SKIP;
    }

    return statuses.slice(1).reduce((worst, status) => {
        return STATUS_ORDER[status] > STATUS_ORDER[worst] ? status : worst;
    }, statuses[0]);
}

function writeHeading(chunks: string[], level: number, text: string): void {
    chunks.push(`${"#".repeat(level)} ${text}\n\n`);
}

function writeNotice(
    chunks: string[],
    severity: "WARN" | "ERROR",
    message: string,
): void {
    chunks.push(`_${severity}: ${message}_\n\n`);
}

function writeBlock(
    ctx: ReportContext,
    chunks: string[],
    label: string,
    result: CmdResult,
    lang = "bash",
): void {
    writeHeading(chunks, 3, label);

    let body = result.text.trim();
    if (!body) {
        body = `# note\nNo data collected for \`${result.cmd}\` (status=${result.status}).`;

        if (result.status === WARN || result.status === ERROR) {
            addLog(ctx, result.status, body);
        }
    }

    const fence = "```";
    chunks.push(`${fence}${lang}\n${body}\n${fence}\n\n`);
}

function detectPciFunctions(
    ctx: ReportContext,
    patterns: string[],
): Array<[string, string]> {
    if (!have("lspci")) {
        return [];
    }

    const result = runCmd(ctx, "lspci -D", false);
    if (!result.stdout.trim()) {
        return [];
    }

    const devices: Array<[string, string]> = [];

    for (const line of result.stdout.split(/\r?\n/)) {
        if (!line) {
            continue;
        }

        if (patterns.every((pattern) => line.toLowerCase().includes(pattern.toLowerCase()))) {
            const [bdf] = line.trim().split(/\s+/, 1);
            if (bdf) {
                devices.push([bdf, line.trim()]);
            }
        }
    }

    return devices;
}

function nowIsoSeconds(): string {
    const date = new Date();
    const pad = (value: number) => String(value).padStart(2, "0");

    return [
        `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`,
        `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`,
    ].join("T");
}

function sectionMetadata(ctx: ReportContext, chunks: string[]): Status {
    writeHeading(chunks, 1, "System Report");

    const host = runCmd(ctx, "hostname", false, false);
    const kernel = runCmd(ctx, "uname -r", false, false);
    const osInfo = runCmd(
        ctx,
        "grep PRETTY_NAME= /etc/os-release || cat /etc/os-release",
        false,
    );

    let osText = osInfo.stdout.trim();
    if (osText.startsWith("PRETTY_NAME=")) {
        osText = osText.split("=", 2)[1]?.trim().replace(/^"|"$/g, "") ?? osText;
    }

    chunks.push(`- Generated: \`${nowIsoSeconds()}\`\n`);
    chunks.push(`- Hostname: \`${host.stdout.trim() || "N/A"}\`\n`);
    chunks.push(`- Kernel: \`${kernel.stdout.trim() || "N/A"}\`\n`);
    chunks.push(`- OS: \`${osText || "N/A"}\`\n\n`);

    return worstStatus([host.status, kernel.status, osInfo.status]);
}

function sectionPcieTopology(ctx: ReportContext, chunks: string[]): Status {
    writeHeading(chunks, 2, "PCIe Topology");

    if (!have("lspci")) {
        const message = "`lspci` not found; cannot capture PCIe topology.";
        writeNotice(chunks, "ERROR", message);
        addLog(ctx, ERROR, message);
        return ERROR;
    }

    const topology = runCmd(ctx, "lspci -tv", true, false);
    writeBlock(ctx, chunks, "lspci -tv", topology);
    return topology.status;
}

function sectionNuma(ctx: ReportContext, chunks: string[]): Status {
    const statuses: Status[] = [];
    writeHeading(chunks, 2, "NUMA Topology");

    if (have("numactl")) {
        const numa = runCmd(ctx, "numactl --hardware", false);
        writeBlock(ctx, chunks, "numactl --hardware", numa);
        statuses.push(numa.status);
    } else {
        const message = "`numactl` not found; skipping `numactl --hardware`.";
        writeNotice(chunks, "WARN", message);
        addLog(ctx, WARN, message);
        statuses.push(WARN);
    }

    const lscpu = runCmd(ctx, "lscpu", false);
    writeBlock(ctx, chunks, "lscpu", lscpu);
    statuses.push(lscpu.status);

    return worstStatus(statuses);
}

function sectionPciDevices(ctx: ReportContext, chunks: string[]): Status {
    const statuses: Status[] = [];
    writeHeading(chunks, 2, "PCIe Devices Details");

    if (!have("lspci")) {
        const message = "`lspci` not found; skipping device details.";
        writeNotice(chunks, "ERROR", message);
        addLog(ctx, ERROR, message);
        return ERROR;
    }

    const gpus = detectPciFunctions(ctx, ["VGA compatible controller", "NVIDIA"]);

    if (gpus.length > 0) {
        gpus.forEach(([bdf, line], index) => {
            const result = runCmd(ctx, `lspci -vvv -s ${bdf}`, false);
            writeBlock(ctx, chunks, `GPU #${index} (${bdf}) — ${line}`, result);
            statuses.push(result.status);
        });
    } else {
        const message = "No NVIDIA VGA GPUs detected via `lspci`.";
        writeNotice(chunks, "WARN", message);
        addLog(ctx, WARN, message);
        statuses.push(WARN);
    }

    const nics = detectPciFunctions(ctx, ["Mellanox", "Ethernet controller"]);
    if (nics.length > 0) {
        nics.forEach(([bdf, line], index) => {
            const result = runCmd(ctx, `lspci -vvv -s ${bdf}`, false);
            writeBlock(
                ctx,
                chunks,
                `ConnectX / Mellanox #${index} (${bdf}) — ${line}`,
                result,
            );
            statuses.push(result.status);
        });
    } else {
        const message = "No Mellanox / ConnectX devices detected via `lspci`.";
        writeNotice(chunks, "WARN", message);
        addLog(ctx, WARN, message);
        statuses.push(WARN);
    }

    const highpoint = detectPciFunctions(ctx, ["HighPoint", "RAID bus controller"]);
    highpoint.forEach(([bdf, line], index) => {
        const result = runCmd(ctx, `lspci -vvv -s ${bdf}`, false);
        writeBlock(
            ctx,
            chunks,
            `HighPoint Carrier Board #${index} (${bdf}) — ${line}`,
            result,
        );
        statuses.push(result.status);
    });

    return worstStatus(statuses);
}

function sectionNvidiaSmi(ctx: ReportContext, chunks: string[]): Status {
    const statuses: Status[] = [];
    writeHeading(chunks, 2, "NVIDIA SMI");

    if (!have("nvidia-smi")) {
        const message = "`nvidia-smi` not found; skipping GPU runtime info.";
        writeNotice(chunks, "WARN", message);
        addLog(ctx, WARN, message);
        return WARN;
    }

    const basic = runCmd(ctx, "nvidia-smi", false);
    writeBlock(ctx, chunks, "nvidia-smi", basic);
    statuses.push(basic.status);

    const topo = runCmd(ctx, "nvidia-smi topo -m", false);
    writeBlock(ctx, chunks, "nvidia-smi topo -m", topo);
    statuses.push(topo.status);

    return worstStatus(statuses);
}

function sectionCudaBandwidth(ctx: ReportContext, chunks: string[]): Status {
    const statuses: Status[] = [];
    writeHeading(chunks, 2, "CUDA Bandwidth / P2P Tests");

    if (!have("nvidia-smi")) {
        const message = "No NVIDIA GPUs detected; skipping CUDA tests.";
        writeNotice(chunks, "WARN", message);
        addLog(ctx, WARN, message);
        return WARN;
    }

    if (have("p2pBandwidthLatencyTest")) {
        const result = runCmd(ctx, "p2pBandwidthLatencyTest", false);
        writeBlock(ctx, chunks, "p2pBandwidthLatencyTest", result);
        statuses.push(result.status);
    } else {
        const message = "`p2pBandwidthLatencyTest` not found in PATH; install CUDA samples.";
        writeNotice(chunks, "WARN", message);
        addLog(ctx, WARN, message);
        statuses.push(WARN);
    }

    if (have("bandwidthTest")) {
        const smiOut = runCmd(
            ctx,
            "nvidia-smi --query-gpu=index --format=csv,noheader",
            false,
        );

        if (smiOut.stdout.trim()) {
            for (const line of smiOut.stdout.split(/\r?\n/)) {
                const index = line.trim();
                if (!/^\d+$/.test(index)) {
                    continue;
                }

                const result = runCmd(ctx, `bandwidthTest --device=${index}`, false);
                writeBlock(ctx, chunks, `bandwidthTest --device=${index}`, result);
                statuses.push(result.status);
            }
        } else {
            const message =
                "Could not determine GPU indices from `nvidia-smi`; skipping bandwidthTest runs.";
            writeNotice(chunks, "WARN", message);
            addLog(ctx, WARN, message);
            statuses.push(WARN);
        }
    } else {
        const message = "`bandwidthTest` not found in PATH; install CUDA samples.";
        writeNotice(chunks, "WARN", message);
        addLog(ctx, WARN, message);
        statuses.push(WARN);
    }

    return worstStatus(statuses);
}

function sectionStorage(ctx: ReportContext, chunks: string[]): Status {
    writeHeading(chunks, 2, "Persistent Storage");

    if (have("lsblk")) {
        const result = runCmd(ctx, "lsblk", false);
        writeBlock(ctx, chunks, "lsblk", result);
        return result.status;
    }

    const message = "`lsblk` not found; skipping block device inventory.";
    writeNotice(chunks, "WARN", message);
    addLog(ctx, WARN, message);
    return WARN;
}

function sectionConnectx(ctx: ReportContext, chunks: string[]): Status {
    writeHeading(chunks, 2, "ConnectX / RDMA Configuration");

    if (have("ibv_devinfo")) {
        const result = runCmd(ctx, "ibv_devinfo", false);
        writeBlock(ctx, chunks, "ibv_devinfo", result);
        return result.status;
    }

    const message = "`ibv_devinfo` not found; skipping RDMA HCA details.";
    writeNotice(chunks, "WARN", message);
    addLog(ctx, WARN, message);
    return WARN;
}

function sectionGrub(ctx: ReportContext, chunks: string[]): Status {
    writeHeading(chunks, 2, "GRUB Configuration");

    try {
        const content = readFileSync("/etc/default/grub", "utf8");
        writeBlock(
            ctx,
            chunks,
            "/etc/default/grub",
            buildCmdResult("cat /etc/default/grub", OK, content),
            "",
        );
        return OK;
    } catch (error) {
        const message =
            error instanceof Error && /ENOENT/.test(error.message)
                ? "`/etc/default/grub` not found."
                : `Failed to read grub file: ${error instanceof Error ? error.message : String(error)}`;

        const severity: Status = message.includes("not found") ? WARN : ERROR;
        writeNotice(chunks, severity === ERROR ? "ERROR" : "WARN", message);
        addLog(ctx, severity, message);
        return severity;
    }
}

function sectionAcsState(ctx: ReportContext, chunks: string[]): Status {
    writeHeading(chunks, 2, "PCIe ACS State");

    if (!have("lspci")) {
        const message = "`lspci` not found; cannot query ACSCtl.";
        writeNotice(chunks, "WARN", message);
        addLog(ctx, WARN, message);
        return WARN;
    }

    const result = runCmd(ctx, "lspci -vvv | grep -i ACSCtl || true", false);
    if (result.stdout.trim()) {
        writeBlock(ctx, chunks, "lspci -vvv | grep -i ACSCtl", result);
        return result.status;
    }

    writeBlock(
        ctx,
        chunks,
        "lspci -vvv | grep -i ACSCtl",
        buildCmdResult(
            "lspci -vvv | grep -i ACSCtl || true",
            OK,
            "",
            "",
            "No ACSCtl capabilities reported by lspci.",
        ),
    );

    return OK;
}

function sectionGds(ctx: ReportContext, chunks: string[]): Status {
    writeHeading(chunks, 2, "GPUDirect Storage (GDS)");

    const candidates = ["/usr/local/cuda/gds/tools/gdscheck", "gdscheck"];
    const command = candidates.find((candidate) => have(candidate));

    if (!command) {
        const message = "`gdscheck` not found; skipping GDS diagnostics.";
        writeNotice(chunks, "WARN", message);
        addLog(ctx, WARN, message);
        return WARN;
    }

    const result = runCmd(ctx, `${command} -p`, false);
    writeBlock(ctx, chunks, `${command} -p`, result);
    return result.status;
}

const SECTION_MAP: Record<string, [string, SectionFn]> = {
    metadata: ["Metadata", sectionMetadata],
    pcie_topology: ["PCIe Topology", sectionPcieTopology],
    numa: ["NUMA Topology", sectionNuma],
    pcie_devices: ["PCIe Devices Details", sectionPciDevices],
    nvidia_smi: ["NVIDIA SMI", sectionNvidiaSmi],
    cuda_bandwidth: ["CUDA Bandwidth / P2P Tests", sectionCudaBandwidth],
    storage: ["Persistent Storage", sectionStorage],
    connectx: ["ConnectX / RDMA Configuration", sectionConnectx],
    grub: ["GRUB Configuration", sectionGrub],
    acs_state: ["PCIe ACS State", sectionAcsState],
    gds: ["GPUDirect Storage (GDS)", sectionGds],
};

export function generateReportData(): ReportData {
    const ctx: ReportContext = { currentSection: "Report", logs: [] };
    const chunks: string[] = [];
    const sectionStatuses: SectionStatus[] = [];

    for (const section of Object.keys(SECTION_MAP)) {
        const [displayName, sectionFn] = SECTION_MAP[section]!;
        ctx.currentSection = displayName;
        sectionStatuses.push([displayName, sectionFn(ctx, chunks)]);
    }

    return {
        markdown: chunks.join(""),
        sectionStatuses,
        logs: ctx.logs,
        overallStatus: worstStatus(sectionStatuses.map(([, status]) => status)),
    };
}

export function writeReportFile(outputPath = "report.md"): ReportWriteResult {
    const report = generateReportData();
    const logs = [...report.logs];
    let overallStatus = report.overallStatus;
    let outputMessage = `Report written to: ${outputPath}`;
    let writeSucceeded = false;

    try {
        writeFileSync(outputPath, report.markdown);
        writeSucceeded = true;
    } catch (error) {
        outputMessage = `Failed to write report to ${outputPath}: ${error instanceof Error ? error.message : String(error)}`;
        overallStatus = ERROR;
    }

    return {
        ...report,
        logs,
        overallStatus,
        outputPath,
        outputMessage,
        writeSucceeded,
    };
}

export function reportExitCode(status: Status): number {
    return status === ERROR ? 1 : 0;
}
