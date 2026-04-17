declare var self: Worker;

import { runStellineApp } from "./ffi.js";

import type { RunResolution } from "./launch.js";

type RunWorkerRequest = {
    type: "run";
    args: string[];
    resolution: RunResolution;
};

self.onmessage = (event: MessageEvent<RunWorkerRequest>) => {
    if (event.data.type !== "run") {
        return;
    }

    try {
        const code = runStellineApp(event.data.resolution, event.data.args);
        postMessage({ type: "exit", code });
        process.exit(0);
    } catch (error) {
        postMessage({
            type: "error",
            message: error instanceof Error ? error.message : String(error),
        });
        process.exit(1);
    }
}
