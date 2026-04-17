const MAX_U64 = "18446744073709551615";
const METADATA_CONTENT_TYPE = "application/json";

export interface ManifestEntry {
    key: string;
    value: string;
    valueType: string;
    start: string;
    end: string;
}

export interface MetricsEntry {
    block: string;
    key: string;
    label: string;
    format: string;
    value: string;
}

export interface NexusClientOptions {
    env?: NodeJS.ProcessEnv;
    backend?: NexusBackend;
}

export interface NexusBackend {
    start(
        onManifest: (entries: ManifestEntry[]) => void,
        onConnectionState: (connected: boolean | null) => void,
    ): Promise<void>;
    close(): Promise<void>;
    fetchManifest(): Promise<ManifestEntry[]>;
    pushMetrics(entries: MetricsEntry[]): Promise<void>;
    syncStatus(status: string, log?: string): Promise<void>;
}

function metricTypeForFormat(format: string): "number" | "text" | null {
    if (format === "stelline-metrics-global-number") {
        return "number";
    }

    if (format === "stelline-metrics-global-string") {
        return "text";
    }

    return null;
}

interface HttpWsNexusBackendOptions {
    env: NodeJS.ProcessEnv;
}

type ManifestListener = (entries: ManifestEntry[]) => void;
type ConnectionListener = (connected: boolean | null) => void;

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null;
}

function normalizeBound(value: unknown, fallback: string): string {
    if (typeof value === "string" && value.length > 0) {
        return value;
    }

    if (typeof value === "number" && Number.isFinite(value)) {
        return `${Math.trunc(value)}`;
    }

    if (typeof value === "bigint") {
        return value.toString();
    }

    return fallback;
}

function encodeManifestValue(value: unknown, valueType: string): string {
    if (value === undefined || value === null) {
        return "";
    }

    if (typeof value === "string") {
        return value;
    }

    if (typeof value === "number" || typeof value === "boolean" || typeof value === "bigint") {
        return `${value}`;
    }

    const encoded = JSON.stringify(value);
    return encoded ?? `${value}`;
}

function cloneManifestEntries(entries: ManifestEntry[]): ManifestEntry[] {
    return entries.map((entry) => ({ ...entry }));
}

export class NexusClient {
    private readonly backend: NexusBackend;
    private readonly manifestListeners = new Set<ManifestListener>();
    private readonly connectionListeners = new Set<ConnectionListener>();
    private manifestEntries: ManifestEntry[] | null = null;
    private upstreamConnected: boolean | null = null;

    constructor(options: NexusClientOptions = {}) {
        this.backend =
            options.backend ??
            new HttpWsNexusBackend({
                env: options.env ?? process.env,
            });
    }

    async start(): Promise<void> {
        await this.backend.start(
            (entries) => {
                this.emitManifest(entries);
            },
            (connected) => {
                this.emitConnectionState(connected);
            },
        );

        if (this.manifestEntries === null) {
            this.emitManifest([]);
        }
    }

    async close(): Promise<void> {
        await this.backend.close();
    }

    onManifest(callback: ManifestListener): () => void {
        this.manifestListeners.add(callback);

        if (this.manifestEntries !== null) {
            callback(cloneManifestEntries(this.manifestEntries));
        }

        return () => {
            this.manifestListeners.delete(callback);
        };
    }

    onConnectionState(callback: ConnectionListener): () => void {
        this.connectionListeners.add(callback);
        callback(this.upstreamConnected);

        return () => {
            this.connectionListeners.delete(callback);
        };
    }

    async pushMetrics(entries: MetricsEntry[]): Promise<void> {
        await this.backend.pushMetrics(entries);
    }

    async syncStatus(status: string, log?: string): Promise<void> {
        await this.backend.syncStatus(status, log);
    }

    private emitManifest(entries: ManifestEntry[]): void {
        const snapshot = cloneManifestEntries(entries);
        this.manifestEntries = snapshot;

        for (const listener of this.manifestListeners) {
            listener(cloneManifestEntries(snapshot));
        }
    }

    private emitConnectionState(connected: boolean | null): void {
        if (this.upstreamConnected === connected) {
            return;
        }

        this.upstreamConnected = connected;
        for (const listener of this.connectionListeners) {
            listener(connected);
        }
    }
}

export class HttpWsNexusBackend implements NexusBackend {
    private readonly env: NodeJS.ProcessEnv;
    private readonly serverUrl: string;
    private readonly instanceId: string;
    private readonly metadataUrl: string;
    private websocket?: WebSocket;
    private outboundQueue: Array<Record<string, unknown>> = [];
    private upstreamConnected: boolean | null = null;
    private onConnectionState?: ConnectionListener;

    constructor(options: HttpWsNexusBackendOptions) {
        this.env = options.env;
        this.serverUrl = (this.env.NEXUS_SERVER_URL ?? "").trim();
        this.instanceId = (this.env.NEXUS_INSTANCE_ID ?? "").trim();
        this.metadataUrl = (this.env.NEXUS_METADATA_URL ?? "").trim();
    }

    async start(
        onManifest: (entries: ManifestEntry[]) => void,
        onConnectionState: (connected: boolean | null) => void,
    ): Promise<void> {
        this.onConnectionState = onConnectionState;
        const entries = await this.fetchManifest();
        onManifest(entries);

        if (!this.available()) {
            this.reportUpstreamConnection(null);
            return;
        }

        this.connectWebSocket(onManifest);
    }

    async close(): Promise<void> {
        this.outboundQueue = [];

        const socket = this.websocket;
        this.websocket = undefined;
        if (!socket) {
            return;
        }

        try {
            socket.close();
        } catch {
            // Ignore websocket shutdown errors during CLI cleanup.
        }

        this.reportUpstreamConnection(null);
    }

    async fetchManifest(): Promise<ManifestEntry[]> {
        if (!this.available()) {
            this.reportUpstreamConnection(null);
            return [];
        }

        try {
            const response = await fetch(this.buildMetadataUrl(), {
                method: "POST",
                headers: {
                    "Content-Type": METADATA_CONTENT_TYPE,
                },
                body: JSON.stringify({ keys: [] }),
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const payload = (await response.json()) as unknown;
            if (!isRecord(payload) || !Array.isArray(payload.data)) {
                return [];
            }

            const entries: ManifestEntry[] = [];
            for (const rawEntry of payload.data) {
                const entry = this.normalizeManifestEntry(rawEntry);
                if (entry) {
                    entries.push(entry);
                }
            }

            this.reportUpstreamConnection(true);
            return entries;
        } catch (error) {
            this.reportUpstreamConnection(false);
            return [];
        }
    }

    async pushMetrics(entries: MetricsEntry[]): Promise<void> {
        if (!this.available() || entries.length === 0) {
            return;
        }

        const metrics: Record<string, Record<string, { type: "number" | "text"; value: string }>> = {};
        for (const entry of entries) {
            const type = metricTypeForFormat(entry.format);
            if (!type) {
                continue;
            }

            const blockMetrics = metrics[entry.block] ?? (metrics[entry.block] = {});
            blockMetrics[entry.key] = {
                type,
                value: entry.value,
            };
        }

        if (Object.keys(metrics).length === 0) {
            return;
        }

        this.sendOrQueue({
            type: "metrics",
            metrics,
        });
    }

    async syncStatus(status: string, log?: string): Promise<void> {
        if (!this.available()) {
            return;
        }

        this.sendOrQueue({
            type: "status",
            status,
            ...(log ? { log } : {}),
        });
    }

    private available(): boolean {
        return this.serverUrl.length > 0 && this.instanceId.length > 0;
    }

    private connectWebSocket(onManifest: (entries: ManifestEntry[]) => void): void {
        let socket: WebSocket;
        try {
            socket = new WebSocket(this.buildWebSocketUrl());
        } catch (error) {
            this.reportUpstreamConnection(false);
            return;
        }

        this.websocket = socket;

        socket.addEventListener("open", () => {
            if (this.websocket !== socket) {
                return;
            }

            this.reportUpstreamConnection(true);

            const pendingMessages = this.outboundQueue;
            this.outboundQueue = [];
            for (const message of pendingMessages) {
                this.sendOrQueue(message);
            }
        });

        socket.addEventListener("message", (event) => {
            void this.handleMessage(event.data, onManifest);
        });

        socket.addEventListener("close", () => {
            if (this.websocket === socket) {
                this.websocket = undefined;
            }

            this.reportUpstreamConnection(false);
        });

        socket.addEventListener("error", () => {
            if (this.websocket === socket && socket.readyState === WebSocket.CLOSED) {
                this.websocket = undefined;
            }

            if (socket.readyState !== WebSocket.OPEN) {
                this.reportUpstreamConnection(false);
            }
        });
    }

    private async handleMessage(
        rawMessage: unknown,
        onManifest: (entries: ManifestEntry[]) => void,
    ): Promise<void> {
        const text =
            typeof rawMessage === "string"
                ? rawMessage
                : rawMessage instanceof ArrayBuffer
                  ? Buffer.from(rawMessage).toString("utf8")
                  : `${rawMessage}`;

        let payload: unknown;
        try {
            payload = JSON.parse(text);
        } catch {
            return;
        }

        if (!isRecord(payload)) {
            return;
        }

        if (payload.type === "manifest" && Array.isArray(payload.entries)) {
            const entries: ManifestEntry[] = [];
            for (const rawEntry of payload.entries) {
                const entry = this.normalizeManifestEntry(rawEntry);
                if (entry) {
                    entries.push(entry);
                }
            }

            onManifest(entries);
            return;
        }

        if (payload.type === "signal" && typeof payload.signal === "string") {
            const signal = payload.signal.toLowerCase();
            if (signal.includes("manifest") || signal.includes("metadata")) {
                onManifest(await this.fetchManifest());
            }
        }
    }

    private buildMetadataUrl(): string {
        if (this.metadataUrl.length > 0) {
            return this.metadataUrl;
        }

        const url = new URL(this.serverUrl);
        url.pathname = `/api/v1/instances/${this.instanceId}/metadata`;
        url.search = "";
        url.hash = "";
        return url.toString();
    }

    private buildWebSocketUrl(): string {
        const url = new URL(this.serverUrl);
        url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
        url.pathname = `/api/v1/instances/${this.instanceId}/ws`;
        url.search = "";
        url.hash = "";
        return url.toString();
    }

    private normalizeManifestEntry(raw: unknown): ManifestEntry | null {
        if (!isRecord(raw) || typeof raw.key !== "string") {
            return null;
        }

        const valid = isRecord(raw.valid) ? raw.valid : {};
        const valueType =
            typeof raw.type === "string"
                ? raw.type
                : typeof raw.valueType === "string"
                  ? raw.valueType
                  : "string";
        const stopTimestamp = normalizeBound(
            valid.stop_timestamp ?? (typeof raw.end === "string" || typeof raw.end === "number" ? raw.end : undefined),
            MAX_U64,
        );

        return {
            key: raw.key,
            value: encodeManifestValue(raw.value, valueType),
            valueType,
            start: normalizeBound(
                valid.start_timestamp ?? (typeof raw.start === "string" || typeof raw.start === "number" ? raw.start : undefined),
                "0",
            ),
            end: stopTimestamp === "0" ? MAX_U64 : stopTimestamp,
        };
    }

    private sendOrQueue(message: Record<string, unknown>): void {
        const socket = this.websocket;
        if (!socket || socket.readyState !== WebSocket.OPEN) {
            this.outboundQueue.push(message);
            if (this.outboundQueue.length > 64) {
                this.outboundQueue.shift();
            }
            return;
        }

        socket.send(JSON.stringify(message));
    }

    private reportUpstreamConnection(connected: boolean | null): void {
        if (this.upstreamConnected === connected) {
            return;
        }

        this.upstreamConnected = connected;
        this.onConnectionState?.(connected);
    }
}
