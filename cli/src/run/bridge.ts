import { createReadStream, type ReadStream } from "node:fs";

import type { RoomInfo } from "./footer.js";
import { StellineFfi, type StellineStatusEntry } from "./ffi.js";
import type { RunResolution } from "./launch.js";
import type { ManifestEntry, MetricsEntry } from "./nexus.js";
import { LineBuffer, parseRemoteInfo } from "./remote.js";

export interface RunBridgeCallbacks {
    onMetrics?(entries: MetricsEntry[]): void;
    onOutput?(chunk: string): void;
    onRoom?(room: RoomInfo): void;
    onStatus?(entry: StellineStatusEntry): void;
}

export class RunBridge {
    private readonly ffi: StellineFfi;
    private readonly callbacks: RunBridgeCallbacks;

    private logReader?: ReadStream;
    private logBuffer?: LineBuffer;
    private pollTimer?: ReturnType<typeof setInterval>;
    private started = false;

    constructor(resolution: RunResolution, callbacks: RunBridgeCallbacks = {}) {
        this.ffi = new StellineFfi(resolution);
        this.callbacks = callbacks;
    }

    reset(): void {
        this.ffi.reset();
    }

    syncManifest(entries: ManifestEntry[], connected: boolean | null): void {
        this.ffi.syncManifest(entries, connected);
    }

    start(captureOutput: boolean): void {
        if (this.started) {
            return;
        }

        this.started = true;

        if (captureOutput) {
            const logReadFd = this.ffi.openLogPipe();
            if (logReadFd < 0) {
                throw new Error("Failed to open the native Stelline log pipe.");
            }

            this.logBuffer = new LineBuffer((line) => {
                const remoteInfo = parseRemoteInfo(line);
                if (remoteInfo) {
                    this.callbacks.onRoom?.(remoteInfo);
                }
            });

            this.logReader = createReadStream("/dev/null", {
                fd: logReadFd,
                autoClose: false,
            });
            this.logReader.setEncoding("utf8");
            this.logReader.on("data", (chunk) => {
                const text = typeof chunk === "string" ? chunk : chunk.toString("utf8");
                this.callbacks.onOutput?.(text);
                this.logBuffer?.push(text);
            });
            this.logReader.on("error", () => undefined);
        }

        this.pollTimer = setInterval(() => {
            this.drain();
        }, 200);
    }

    close(): void {
        this.drain();
        this.logBuffer?.flush();

        if (this.pollTimer) {
            clearInterval(this.pollTimer);
            this.pollTimer = undefined;
        }

        this.logReader?.destroy();
        this.logReader = undefined;
        this.logBuffer = undefined;
        this.started = false;

        this.ffi.closeLogPipe();
        this.ffi.reset();
        this.ffi.close();
    }

    private drain(): void {
        const metrics = this.ffi.drainMetrics();
        if (metrics.length > 0) {
            this.callbacks.onMetrics?.(metrics);
        }

        const statuses = this.ffi.drainStatuses();
        for (const status of statuses) {
            this.callbacks.onStatus?.(status);
        }
    }
}
