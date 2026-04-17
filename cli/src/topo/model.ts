import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { promises as fs } from "node:fs";

const execFileAsync = promisify(execFile);

const GPU_DRIVER_PATH = "/sys/bus/pci/drivers/nvidia";
const INFINIBAND_PATH = "/sys/class/infiniband";

export type CpuClass =
    | "isolated-irq-free"
    | "isolated-only"
    | "irq-free-only"
    | "default";

export interface StyledCpu {
    value: string;
    className: CpuClass;
}

export interface GpuInfo {
    idx: number;
    pcie: string;
    numa: string;
    cpus: string;
    name: string;
}

export interface NicInfo {
    name: string;
    pcie: string;
    numa: string;
    iface: string;
}

export interface TopologyRow {
    gpuId: string;
    gpuName: string;
    gpuPcie: string;
    numa: string;
    cpus: StyledCpu[];
    nic: string;
    nicPcie: string;
    iface: string;
    sectionStart: boolean;
}

export interface TopologyModel {
    rows: TopologyRow[];
    isolatedDetected: boolean;
    warnings: string[];
}

async function pathExists(path: string): Promise<boolean> {
    try {
        await fs.access(path);
        return true;
    } catch {
        return false;
    }
}

async function readText(path: string, defaultValue = "N/A"): Promise<string> {
    try {
        return (await fs.readFile(path, "utf8")).trim();
    } catch {
        return defaultValue;
    }
}

async function runCommand(
    command: string,
    args: string[],
): Promise<string | null> {
    try {
        const { stdout } = await execFileAsync(command, args, {
            encoding: "utf8",
        });
        return stdout.trim();
    } catch {
        return null;
    }
}

function parseCpuList(text: string): Set<number> {
    const cpus = new Set<number>();
    const value = text.trim();

    if (!value) {
        return cpus;
    }

    for (const part of value.split(",")) {
        const token = part.trim();
        if (!token) {
            continue;
        }

        if (token.includes("-")) {
            const [loText, hiText] = token.split("-", 2);
            const lo = Number.parseInt(loText, 10);
            const hi = Number.parseInt(hiText, 10);

            if (Number.isNaN(lo) || Number.isNaN(hi)) {
                continue;
            }

            for (let cpu = lo; cpu <= hi; cpu += 1) {
                cpus.add(cpu);
            }
            continue;
        }

        const cpu = Number.parseInt(token, 10);
        if (!Number.isNaN(cpu)) {
            cpus.add(cpu);
        }
    }

    return cpus;
}

function parseCpuMask(text: string): Set<number> {
    const normalized = text.trim().replaceAll(",", "");
    if (!normalized) {
        return new Set<number>();
    }

    let mask = BigInt(`0x${normalized}`);
    let bit = 0;
    const cpus = new Set<number>();

    while (mask > 0n) {
        if ((mask & 1n) === 1n) {
            cpus.add(bit);
        }
        mask >>= 1n;
        bit += 1;
    }

    return cpus;
}

function classifyCpu(
    cpu: number,
    isolated: Set<number>,
    irqCpus: Set<number>,
): CpuClass {
    const isIsolated = isolated.has(cpu);
    const isIrqFree = !irqCpus.has(cpu);

    if (isIsolated && isIrqFree) {
        return "isolated-irq-free";
    }
    if (isIsolated) {
        return "isolated-only";
    }
    if (isIrqFree) {
        return "irq-free-only";
    }
    return "default";
}

function styleCpuList(
    cpuText: string,
    isolated: Set<number>,
    irqCpus: Set<number>,
): StyledCpu[] {
    if (cpuText === "N/A") {
        return [{ value: "N/A", className: "default" }];
    }

    const values = cpuText
        .split(/\s+/)
        .map((token) => Number.parseInt(token, 10))
        .filter((value) => !Number.isNaN(value));

    if (values.length === 0) {
        return [{ value: cpuText, className: "default" }];
    }

    return values.map((cpu) => ({
        value: String(cpu),
        className: classifyCpu(cpu, isolated, irqCpus),
    }));
}

async function getNumaCpus(numaNode: string): Promise<string> {
    const output = await runCommand("numactl", ["-H"]);
    if (!output) {
        return "N/A";
    }

    for (const line of output.split("\n")) {
        if (line.startsWith(`node ${numaNode} cpus:`)) {
            const [, cpuText = "N/A"] = line.split(":", 2);
            return cpuText.trim() || "N/A";
        }
    }

    return "N/A";
}

async function queryGpuNames(): Promise<Map<string, string>> {
    const output = await runCommand("nvidia-smi", [
        "--query-gpu=gpu_bus_id,gpu_name",
        "--format=csv,noheader,nounits",
    ]);

    const names = new Map<string, string>();
    if (!output) {
        return names;
    }

    for (const line of output.split("\n")) {
        const [busIdRaw, nameRaw] = line.split(", ", 2);
        if (!busIdRaw || !nameRaw) {
            continue;
        }

        let busId = busIdRaw.trim().toLowerCase();
        const segments = busId.split(":");
        if (segments.length >= 3 && segments[0]?.length === 8) {
            busId = `${segments[0].slice(4)}:${segments.slice(1).join(":")}`;
        }

        names.set(busId, nameRaw.trim());
    }

    return names;
}

export async function discoverGpus(): Promise<GpuInfo[]> {
    if (!(await pathExists(GPU_DRIVER_PATH))) {
        return [];
    }

    const gpuNames = await queryGpuNames();
    const entries = await fs.readdir(GPU_DRIVER_PATH, { withFileTypes: true });
    const devices = entries
        .filter((entry) => /^\w{4}:\w{2}:\w{2}\.\w$/.test(entry.name))
        .map((entry) => entry.name)
        .sort();

    const gpus: GpuInfo[] = [];

    for (const [idx, pcie] of devices.entries()) {
        const basePath = `${GPU_DRIVER_PATH}/${pcie}`;
        const numa = await readText(`${basePath}/numa_node`, "N/A");
        const cpus =
            numa !== "N/A" && numa !== "-1" ? await getNumaCpus(numa) : "N/A";

        gpus.push({
            idx,
            pcie,
            numa,
            cpus,
            name: gpuNames.get(pcie.toLowerCase()) ?? "N/A",
        });
    }

    return gpus;
}

export async function discoverNics(): Promise<NicInfo[]> {
    if (!(await pathExists(INFINIBAND_PATH))) {
        return [];
    }

    const entries = await fs.readdir(INFINIBAND_PATH, { withFileTypes: true });
    const nicNames = entries
        .filter((entry) => entry.name.startsWith("mlx5_"))
        .map((entry) => entry.name)
        .sort();

    const nics: NicInfo[] = [];

    for (const name of nicNames) {
        const devicePath = `${INFINIBAND_PATH}/${name}/device`;
        let pcie = "N/A";

        try {
            pcie = (await fs.readlink(devicePath)).split("/").at(-1) ?? "N/A";
        } catch {
            pcie = "N/A";
        }

        const numa = await readText(`${devicePath}/numa_node`, "N/A");
        let iface = "N/A";

        try {
            const netEntries = await fs.readdir(`${devicePath}/net`);
            if (netEntries.length > 0) {
                iface = [...netEntries].sort()[0] ?? "N/A";
            }
        } catch {
            iface = "N/A";
        }

        nics.push({ name, pcie, numa, iface });
    }

    return nics;
}

export async function buildTopologyModel(): Promise<TopologyModel> {
    const gpus = await discoverGpus();
    const nics = await discoverNics();
    const warnings: string[] = [];

    if (gpus.length === 0) {
        warnings.push("No NVIDIA GPUs found in /sys/bus/pci/drivers/nvidia.");
        return {
            rows: [],
            isolatedDetected: false,
            warnings,
        };
    }

    const isolated = parseCpuList(
        await readText("/sys/devices/system/cpu/isolated", ""),
    );
    const irqAffinityText = await readText(
        "/proc/irq/default_smp_affinity",
        "",
    );
    const irqCpus = irqAffinityText
        ? parseCpuMask(irqAffinityText)
        : new Set<number>();

    if (isolated.size === 0) {
        warnings.push(
            "No isolated cores detected. Transport usually expects isolcpus, nohz_full, rcu_nocbs, and irqaffinity kernel parameters.",
        );
    }

    const rows: TopologyRow[] = [];

    for (const [gpuIndex, gpu] of gpus.entries()) {
        const matchedNics = nics.filter((nic) => nic.numa === gpu.numa);
        const cpus = styleCpuList(gpu.cpus, isolated, irqCpus);

        if (matchedNics.length === 0) {
            rows.push({
                gpuId: `GPU${gpu.idx}`,
                gpuName: gpu.name,
                gpuPcie: gpu.pcie,
                numa: gpu.numa,
                cpus,
                nic: "none",
                nicPcie: "-",
                iface: "-",
                sectionStart: gpuIndex > 0,
            });
            continue;
        }

        matchedNics.forEach((nic, nicIndex) => {
            rows.push({
                gpuId: nicIndex === 0 ? `GPU${gpu.idx}` : "",
                gpuName: nicIndex === 0 ? gpu.name : "",
                gpuPcie: nicIndex === 0 ? gpu.pcie : "",
                numa: nicIndex === 0 ? gpu.numa : "",
                cpus: nicIndex === 0 ? cpus : [],
                nic: nic.name,
                nicPcie: nic.pcie,
                iface: nic.iface,
                sectionStart: gpuIndex > 0 && nicIndex === 0,
            });
        });
    }

    return {
        rows,
        isolatedDetected: isolated.size > 0,
        warnings,
    };
}
