import { capture } from "@opentui/core";

import { RunBridge } from "./bridge.js";
import { createRunFooter } from "./footer.js";
import { resolveRunLaunch, type RunResolution } from "./launch.js";
import { NexusClient, type ManifestEntry } from "./nexus.js";

interface RunCommandExecutionOptions {
    plain?: boolean;
}

type RunWorkerResponse =
    | { type: "exit"; code: number }
    | { type: "error"; message: string };

function errorOutputStream(footerActive: boolean): NodeJS.WriteStream {
    if (footerActive && process.stderr.isTTY === true) {
        return process.stdout;
    }

    return process.stderr;
}

function writeRunMessage(text: string, footerActive: boolean): void {
    errorOutputStream(footerActive).write(`${text}\n`);
}

function createRunWorker(): Worker {
    return new Worker(new URL("./worker.js", import.meta.url).href);
}

export async function runCyberetherCommand(
    args: string[],
    options: RunCommandExecutionOptions = {},
): Promise<number> {
    let resolution: RunResolution;

    try {
        resolution = resolveRunLaunch();
    } catch (error) {
        process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
        return 1;
    }

    const footer = await createRunFooter({
        enabled: options.plain !== true,
    });
    const nexus = new NexusClient({ env: process.env });
    const bridge = new RunBridge(resolution, {
        onMetrics: (entries) => {
            footer.addMetrics(entries);
            void nexus.pushMetrics(entries);
        },
        onOutput: (chunk) => {
            capture.write("stdout", chunk);
        },
        onRoom: (room) => {
            footer.setRoom(room);
        },
        onStatus: (status) => {
            void nexus.syncStatus(status.status, status.log);
        },
    });

    let currentManifest: ManifestEntry[] = [];
    let currentConnectionState: boolean | null = null;
    const syncManifest = () => {
        bridge.syncManifest(currentManifest, currentConnectionState);
    };

    bridge.reset();

    nexus.onManifest((entries) => {
        currentManifest = entries.map((entry) => ({ ...entry }));
        footer.setManifest(entries);
        syncManifest();
    });
    nexus.onConnectionState((connected) => {
        currentConnectionState = connected;
        footer.setNexusConnected(connected);
        syncManifest();
    });

    try {
        await nexus.start();
        bridge.start(footer.active);
    } catch (error) {
        await footer.close().catch(() => undefined);
        await nexus.close().catch(() => undefined);
        bridge.close();
        writeRunMessage(error instanceof Error ? error.message : String(error), footer.active);
        return 1;
    }

    return await new Promise<number>((resolve) => {
        let settled = false;

        const worker = createRunWorker();

        const finalize = async (code: number): Promise<void> => {
            if (settled) {
                return;
            }

            settled = true;
            process.off("SIGINT", handleSignal);
            process.off("SIGTERM", handleSignal);

            worker.terminate();
            await footer.close().catch(() => undefined);
            bridge.close();
            await nexus.close().catch(() => undefined);
            resolve(code);
        };

        const handleSignal = (signal: NodeJS.Signals) => {
            void finalize(signal === "SIGINT" ? 130 : 143);
        };

        process.on("SIGINT", handleSignal);
        process.on("SIGTERM", handleSignal);

        worker.addEventListener("message", (event: MessageEvent<RunWorkerResponse>) => {
            if (event.data.type === "error") {
                writeRunMessage(event.data.message, footer.active);
                void finalize(1);
                return;
            }

            void finalize(event.data.code);
        });

        worker.addEventListener("error", (event) => {
            const message = event.error instanceof Error ? event.error.message : "libstelline worker failed.";
            writeRunMessage(message, footer.active);
            void finalize(1);
        });

        worker.addEventListener("close", () => {
            if (settled) {
                return;
            }

            void finalize(1);
        });

        worker.postMessage({
            type: "run",
            args,
            resolution,
        });
    });
}
