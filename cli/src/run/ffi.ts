import { CString, dlopen, ptr } from "bun:ffi";
import { allocStruct, defineStruct } from "bun-ffi-structs";

import type { FFIFunction, Pointer } from "bun:ffi";

import type { RunResolution } from "./launch.js";
import type { ManifestEntry, MetricsEntry } from "./nexus.js";

export interface StellineStatusEntry {
    status: string;
    log?: string;
}

interface ControlSymbols {
    StellineLogPipeOpen(): number;
    StellineLogPipeClose(): void;
    StellineNexusReset(): number;
    StellineNexusManifestBegin(): number;
    StellineNexusManifestPush(view: Pointer): number;
    StellineNexusManifestCommit(connected: number): number;
    StellineNexusMetricRead(view: Pointer): number;
    StellineNexusStatusRead(view: Pointer): number;
}

interface RunnerSymbols {
    StellineRunApp(argc: number, argv: number): number;
}

interface LibraryHandle<TSymbols> {
    symbols: TSymbols;
    close(): void;
}

interface LibraryBundle<TSymbols> {
    handle: LibraryHandle<TSymbols>;
    close(): void;
}

function openLibrary<TSymbols>(
    resolution: RunResolution,
    symbols: Record<string, FFIFunction>,
): LibraryBundle<TSymbols> {
    try {
        const handle = dlopen(resolution.stellineLibraryPath, symbols) as LibraryHandle<TSymbols>;
        return {
            handle,
            close() {
                handle.close();
            },
        };
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Failed to load '${resolution.stellineLibraryPath}': ${message}`);
    }
}

function encodeCString(value: string): Uint8Array {
    return Buffer.from(`${value}\0`, "utf8");
}

function buildNativeArgv(args: string[]): { argc: number; argv: number; keepAlive: Array<Uint8Array | BigUint64Array> } {
    const argvEntries = ["cyberether", ...args].map((entry) => encodeCString(entry));
    const argvPointers = new BigUint64Array(argvEntries.length);

    argvEntries.forEach((entry, index) => {
        argvPointers[index] = BigInt(ptr(entry));
    });

    return {
        argc: argvEntries.length,
        argv: ptr(argvPointers),
        keepAlive: [argvPointers, ...argvEntries],
    };
}

function decodeCStringPointer(raw: bigint): string {
    if (raw === 0n) {
        return "";
    }

    return String(new CString(Number(raw) as Pointer));
}

const metricViewStruct = defineStruct([
    ["block", "pointer", { unpackTransform: decodeCStringPointer }],
    ["key", "pointer", { unpackTransform: decodeCStringPointer }],
    ["label", "pointer", { unpackTransform: decodeCStringPointer }],
    ["format", "pointer", { unpackTransform: decodeCStringPointer }],
    ["value", "pointer", { unpackTransform: decodeCStringPointer }],
] as const);

const statusViewStruct = defineStruct([
    ["status", "pointer", { unpackTransform: decodeCStringPointer }],
    ["log", "pointer", { unpackTransform: decodeCStringPointer }],
] as const);

const manifestViewStruct = defineStruct([
    ["key", "pointer"],
    ["valueType", "pointer"],
    ["value", "pointer"],
    ["start", "pointer"],
    ["end", "pointer"],
] as const);

function packManifestView(entry: ManifestEntry): { buffer: ArrayBuffer; keepAlive: Uint8Array[] } {
    const keepAlive = [
        encodeCString(entry.key),
        encodeCString(entry.valueType),
        encodeCString(entry.value),
        encodeCString(entry.start),
        encodeCString(entry.end),
    ];

    return {
        buffer: manifestViewStruct.pack({
            key: ptr(keepAlive[0]),
            valueType: ptr(keepAlive[1]),
            value: ptr(keepAlive[2]),
            start: ptr(keepAlive[3]),
            end: ptr(keepAlive[4]),
        }),
        keepAlive,
    };
}

export class StellineFfi {
    private readonly library: LibraryBundle<ControlSymbols>;

    constructor(resolution: RunResolution) {
        this.library = openLibrary<ControlSymbols>(resolution, {
            StellineLogPipeOpen: { args: [], returns: "i32" },
            StellineLogPipeClose: { args: [], returns: "void" },
            StellineNexusReset: { args: [], returns: "i32" },
            StellineNexusManifestBegin: { args: [], returns: "i32" },
            StellineNexusManifestPush: { args: ["ptr"], returns: "i32" },
            StellineNexusManifestCommit: { args: ["i32"], returns: "i32" },
            StellineNexusMetricRead: { args: ["ptr"], returns: "i32" },
            StellineNexusStatusRead: { args: ["ptr"], returns: "i32" },
        });
    }

    reset(): void {
        this.library.handle.symbols.StellineNexusReset();
    }

    openLogPipe(): number {
        return this.library.handle.symbols.StellineLogPipeOpen();
    }

    closeLogPipe(): void {
        this.library.handle.symbols.StellineLogPipeClose();
    }

    syncManifest(entries: ManifestEntry[], connected: boolean | null): void {
        this.library.handle.symbols.StellineNexusManifestBegin();

        for (const entry of entries) {
            const manifestView = packManifestView(entry);
            const ok = this.library.handle.symbols.StellineNexusManifestPush(ptr(manifestView.buffer));
            if (ok === 0) {
                throw new Error(`Failed to push manifest entry '${entry.key}' into libstelline.so.`);
            }
        }

        this.library.handle.symbols.StellineNexusManifestCommit(connected === true ? 1 : 0);
    }

    drainMetrics(): MetricsEntry[] {
        const entries: MetricsEntry[] = [];
        const metricView = allocStruct(metricViewStruct);

        while (this.library.handle.symbols.StellineNexusMetricRead(ptr(metricView.buffer)) !== 0) {
            entries.push(metricViewStruct.unpack(metricView.buffer));
        }

        return entries;
    }

    drainStatuses(): StellineStatusEntry[] {
        const entries: StellineStatusEntry[] = [];
        const statusView = allocStruct(statusViewStruct);

        while (this.library.handle.symbols.StellineNexusStatusRead(ptr(statusView.buffer)) !== 0) {
            const status = statusViewStruct.unpack(statusView.buffer);
            entries.push({
                status: status.status,
                ...(status.log.length > 0 ? { log: status.log } : {}),
            });
        }

        return entries;
    }

    close(): void {
        this.library.close();
    }
}

export function runStellineApp(resolution: RunResolution, args: string[]): number {
    const library = openLibrary<RunnerSymbols>(resolution, {
        StellineRunApp: { args: ["i32", "ptr"], returns: "i32" },
    });

    try {
        const nativeArgv = buildNativeArgv(args);
        return library.handle.symbols.StellineRunApp(nativeArgv.argc, nativeArgv.argv);
    } finally {
        library.close();
    }
}
