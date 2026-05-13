import { PassThrough } from "node:stream";
import { useSyncExternalStore, type ReactNode } from "react";

import { createCliRenderer } from "@opentui/core";
import { createRoot, flushSync } from "@opentui/react";

import type { ManifestEntry, MetricsEntry } from "./nexus.js";

const COLORS = {
    border: "gray",
    accent: "cyan",
    success: "green",
    error: "red",
    text: "white",
    muted: "gray",
    dim: "gray",
    disabled: "gray",
} as const;

const HEADER_ONLY_HEIGHT = 3;
const REMOTE_CARD_HEIGHT = 6;

export interface RoomInfo {
    roomId?: string;
    joinUrl?: string;
    accessToken?: string;
}

interface FooterState {
    metricsEntries: MetricsEntry[];
    manifestCount: number;
    nexusConnected: boolean | null;
    room?: RoomInfo;
    elapsedSeconds: number;
}

type MetricRow =
    | { type: "header"; text: string }
    | { type: "metric"; label: string; labelSuffix: string; value: string }
    | { type: "spacer" };

interface FooterViewState {
    state: FooterState;
    metricRows: MetricRow[];
    metricColumnWidth: number;
    metricsVisible: boolean;
    remoteVisible: boolean;
}

interface MetricColumns {
    left: MetricRow[];
    right: MetricRow[];
}

interface MetricDisplayEntry {
    label: string;
    labelSuffix: string;
    value: string;
}

interface MetricGroup {
    block: string;
    entries: MetricDisplayEntry[];
}

class FooterViewStore {
    private snapshot: FooterViewState;
    private readonly listeners = new Set<() => void>();

    constructor(snapshot: FooterViewState) {
        this.snapshot = snapshot;
    }

    getSnapshot = (): FooterViewState => this.snapshot;

    subscribe = (listener: () => void): (() => void) => {
        this.listeners.add(listener);
        return () => {
            this.listeners.delete(listener);
        };
    };

    setSnapshot(snapshot: FooterViewState): void {
        this.snapshot = snapshot;
        for (const listener of this.listeners) {
            listener();
        }
    }
}

function formatElapsedTime(totalSeconds: number): string {
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    const pad = (n: number) => n.toString().padStart(2, "0");
    return `T+${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
}

function truncateMiddle(text: string, limit: number): string {
    if (text.length <= limit) {
        return text;
    }

    const prefixLength = Math.max(1, Math.floor((limit - 1) / 2));
    const suffixLength = Math.max(1, limit - prefixLength - 1);
    return `${text.slice(0, prefixLength)}…${text.slice(-suffixLength)}`;
}

function hasRemoteInfo(state: FooterState): boolean {
    return Boolean(state.room?.roomId || state.room?.accessToken || state.room?.joinUrl);
}

function nexusIndicatorColor(connected: boolean | null): string {
    if (connected === true) {
        return COLORS.success;
    }

    if (connected === false) {
        return COLORS.error;
    }

    return COLORS.muted;
}

function Badge({ children }: { children: ReactNode }) {
    return (
        <box
            border
            borderStyle="rounded"
            borderColor={COLORS.border}
            height={3}
            paddingX={1}
            alignItems="center"
            justifyContent="center"
        >
            {children}
        </box>
    );
}

function HeaderBox({ state }: { state: FooterState }) {
    const nexusColor = nexusIndicatorColor(state.nexusConnected);

    return (
        <box width="100%" height={3} flexDirection="row" alignItems="center">
            <box
                border
                borderStyle="rounded"
                borderColor={COLORS.border}
                height={3}
                paddingX={1}
                flexGrow={1}
                flexDirection="row"
                alignItems="center"
            >
                <text>
                    <strong fg={COLORS.accent}>Stelline Runner</strong>
                </text>
            </box>
            <box width={1} />
            <Badge>
                <text>
                    <span fg={nexusColor}>●</span>
                    <span fg={COLORS.text}> Manifest</span>
                    <span fg={COLORS.dim}>{` (${state.manifestCount})`}</span>
                </text>
            </Badge>
            <box width={1} />
            <Badge>
                <text>
                    <span fg={nexusColor}>●</span>
                    <span fg={COLORS.text}> Metrics</span>
                    <span fg={COLORS.dim}>{` (${state.metricsEntries.length})`}</span>
                </text>
            </Badge>
            <box width={1} />
            <Badge>
                <text>
                    <span fg={COLORS.text}>{formatElapsedTime(state.elapsedSeconds)}</span>
                </text>
            </Badge>
        </box>
    );
}

function RemoteCard({ state }: { state: FooterState }) {
    const labelWidth = 13;
    const pad = (text: string, width: number) => text.length >= width ? text : text + " ".repeat(width - text.length);
    const visible = hasRemoteInfo(state);
    const roomId = state.room?.roomId ? truncateMiddle(state.room.roomId, 64) : "";
    const accessToken = state.room?.accessToken ? truncateMiddle(state.room.accessToken, 64) : "";
    const joinUrl = state.room?.joinUrl ? truncateMiddle(state.room.joinUrl, 80) : "";

    return (
        <box
            border={visible}
            borderStyle="rounded"
            borderColor={COLORS.border}
            width="100%"
            height={visible ? REMOTE_CARD_HEIGHT : 0}
            visible={visible}
            paddingX={1}
            flexDirection="column"
        >
            <box width="100%" alignItems="center" justifyContent="center">
                <text>
                    <strong fg={COLORS.accent}>CyberEther Remote</strong>
                </text>
            </box>
            <text>
                <span fg={COLORS.muted}>{pad("Room ID", labelWidth)}</span>
                <span fg={COLORS.text}>{roomId}</span>
            </text>
            <text>
                <span fg={COLORS.muted}>{pad("Access Token", labelWidth)}</span>
                <span fg={COLORS.text}>{accessToken}</span>
            </text>
            <text>
                <span fg={COLORS.muted}>{pad("Join URL", labelWidth)}</span>
                <span fg={COLORS.accent}>{joinUrl}</span>
            </text>
        </box>
    );
}

function isGlobalMetric(entry: MetricsEntry): boolean {
    return entry.format === "private-stelline-metrics-global-number" || entry.format === "private-stelline-metrics-global-string";
}

function formatBlockName(name: string): string {
    return name
        .split(/[_-]+/)
        .filter((part) => part.length > 0)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ")
        .toUpperCase();
}

function formatMetricKey(key: string): string {
    return key
        .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
        .split(/[_-]+|\s+/)
        .filter((part) => part.length > 0)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
}

function stripBlockPrefix(label: string, block: string, blockTitle: string): string {
    const trimmedLabel = label.trim();
    if (trimmedLabel.length === 0) {
        return "";
    }

    const blockPrefixes = [
        block,
        blockTitle,
        block.replace(/[_-]+/g, " "),
    ].map((prefix) => prefix.trim()).filter((prefix, index, values) => prefix.length > 0 && values.indexOf(prefix) === index);

    for (const prefix of blockPrefixes) {
        if (trimmedLabel.localeCompare(prefix, undefined, { sensitivity: "accent" }) === 0) {
            return "";
        }

        const escapedPrefix = prefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        const stripped = trimmedLabel.match(new RegExp(`^${escapedPrefix}(?:\\s*[:._-]\\s*|\\s+)(.+)$`, "i"));
        if (stripped) {
            return stripped[1].trim();
        }
    }

    return trimmedLabel;
}

function formatMetricLabel(entry: MetricsEntry, blockTitle: string): string {
    const explicitLabel = stripBlockPrefix(entry.label, entry.block, blockTitle);
    if (explicitLabel.length > 0) {
        return explicitLabel;
    }

    return formatMetricKey(entry.key);
}

function buildMetricGroups(entries: MetricsEntry[]): MetricGroup[] {
    const grouped = new Map<string, MetricGroup>();
    for (const entry of entries) {
        const blockTitle = formatBlockName(entry.block);
        const group = grouped.get(blockTitle);
        if (group) {
            group.entries.push({
                label: formatMetricLabel(entry, blockTitle),
                labelSuffix: isGlobalMetric(entry) ? " (exported)" : "",
                value: entry.value,
            });
            continue;
        }

        grouped.set(blockTitle, {
            block: blockTitle,
            entries: [{
                label: formatMetricLabel(entry, blockTitle),
                labelSuffix: isGlobalMetric(entry) ? " (exported)" : "",
                value: entry.value,
            }],
        });
    }

    return [...grouped.values()].sort((leftGroup, rightGroup) =>
        leftGroup.block.localeCompare(rightGroup.block, undefined, { numeric: true }),
    );
}

function buildMetricRows(entries: MetricsEntry[]): MetricRow[] {
    const rows: MetricRow[] = [];
    for (const group of buildMetricGroups(entries)) {
        rows.push({
            type: "header",
            text: group.block,
        });

        for (const entry of group.entries) {
            rows.push({
                type: "metric",
                label: entry.label,
                labelSuffix: entry.labelSuffix,
                value: entry.value,
            });
        }

        rows.push({ type: "spacer" });
    }

    return rows;
}

function metricSections(rows: MetricRow[]): MetricRow[][] {
    const sections: MetricRow[][] = [];
    let current: MetricRow[] = [];

    for (const row of rows) {
        current.push(row);
        if (row.type === "spacer") {
            sections.push(current);
            current = [];
        }
    }

    if (current.length > 0) {
        sections.push(current);
    }

    return sections;
}

function splitMetricRows(rows: MetricRow[], columnWidth: number): MetricColumns {
    const sections = metricSections(rows).map((section, index) => ({
        index,
        rows: section,
        height: section.reduce((sum, row) => sum + metricRowHeight(row, columnWidth), 0),
    }));

    const left: typeof sections = [];
    const right: typeof sections = [];
    let leftHeight = 0;
    let rightHeight = 0;

    for (const section of [...sections].sort((leftSection, rightSection) => {
        if (rightSection.height !== leftSection.height) {
            return rightSection.height - leftSection.height;
        }

        return leftSection.index - rightSection.index;
    })) {
        if (leftHeight <= rightHeight) {
            left.push(section);
            leftHeight += section.height;
            continue;
        }

        right.push(section);
        rightHeight += section.height;
    }

    return {
        left: left.sort((a, b) => a.index - b.index).flatMap((section) => section.rows),
        right: right.sort((a, b) => a.index - b.index).flatMap((section) => section.rows),
    };
}

function wrappedLineCount(text: string, lineWidth: number): number {
    if (text.length === 0) {
        return 1;
    }

    let count = 0;
    for (const line of text.split("\n")) {
        count += Math.max(1, Math.ceil(Math.max(1, line.length) / lineWidth));
    }

    return count;
}

function wrapRightAlignedLines(text: string, lineWidth: number): Array<{ pad: string; text: string }> {
    if (lineWidth <= 0) {
        return [{ pad: "", text }];
    }

    const lines: Array<{ pad: string; text: string }> = [];
    for (const sourceLine of text.split("\n")) {
        const line = sourceLine.length === 0 ? "" : sourceLine;
        let firstChunk = true;
        for (let offset = 0; offset < Math.max(1, line.length); offset += lineWidth) {
            const chunk = line.slice(offset, offset + lineWidth);
            const padWidth = Math.max(0, lineWidth - chunk.length);
            lines.push({
                pad: (firstChunk ? "." : " ").repeat(padWidth),
                text: chunk,
            });
            firstChunk = false;
        }
    }

    return lines;
}

function metricLeaderText(text: string, lineWidth: number): string {
    if (lineWidth <= 0) {
        return "";
    }

    const remainder = text.length % lineWidth;
    const dots = remainder === 0 ? 0 : lineWidth - remainder;
    return ".".repeat(dots);
}

function metricKeyWidth(columnWidth: number): number {
    const targetValueWidth = Math.max(4, Math.floor(columnWidth * 0.4));
    const keyWidth = Math.max(3, columnWidth - targetValueWidth);
    return keyWidth;
}

function metricValueWidth(columnWidth: number): number {
    return Math.max(1, columnWidth - metricKeyWidth(columnWidth));
}

function metricRowHeight(row: MetricRow, columnWidth: number): number {
    if (row.type === "header" || row.type === "spacer") {
        return 1;
    }

    return Math.max(
        wrappedLineCount(`${row.label}${row.labelSuffix}`, metricKeyWidth(columnWidth)),
        wrappedLineCount(row.value, metricValueWidth(columnWidth)),
    );
}

function metricColumnsHeight(rows: MetricRow[], columnWidth: number): number {
    const columns = splitMetricRows(rows, columnWidth);
    const leftHeight = columns.left.reduce((sum, row) => sum + metricRowHeight(row, columnWidth), 0);
    const rightHeight = columns.right.reduce((sum, row) => sum + metricRowHeight(row, columnWidth), 0);
    return Math.max(leftHeight, rightHeight);
}

function MetricsCard({
    rows,
    visible,
    columnWidth,
}: {
    rows: MetricRow[];
    visible: boolean;
    columnWidth: number;
}) {
    const columns = splitMetricRows(rows, columnWidth);
    const keyWidth = metricKeyWidth(columnWidth);
    const valueWidth = metricValueWidth(columnWidth);

    return (
        <box
            border={visible}
            borderStyle="rounded"
            borderColor={COLORS.border}
            width="100%"
            height={visible ? metricColumnsHeight(rows, columnWidth) + 2 : 0}
            visible={visible}
            paddingX={1}
            flexDirection="column"
        >
            <box width="100%" flexDirection="row">
                <box flexGrow={1} flexDirection="column">
                    {columns.left.map((row, index) => {
                        if (row.type === "header") {
                            return (
                                <text key={`left:${index}`} width="100%" wrapMode="char">
                                    <strong fg={COLORS.accent}>{row.text}</strong>
                                </text>
                            );
                        }

                        if (row.type === "spacer") {
                            return <text key={`left:${index}`}> </text>;
                        }

                        return (
                            <box key={`left:${index}`} width="100%" flexDirection="row" alignItems="flex-start">
                                <text width={keyWidth} wrapMode="char">
                                    <span fg={COLORS.text}>{row.label}</span>
                                    <span fg={COLORS.accent}>{row.labelSuffix}</span>
                                    <span fg={COLORS.disabled}>{metricLeaderText(`${row.label}${row.labelSuffix}`, keyWidth)}</span>
                                </text>
                                <box width={valueWidth} flexDirection="column">
                                    {wrapRightAlignedLines(row.value, valueWidth).map((line, lineIndex) => (
                                        <text key={`left:${index}:value:${lineIndex}`} width="100%" wrapMode="none">
                                            <span fg={COLORS.disabled}>{line.pad}</span>
                                            <span fg={COLORS.text}>{line.text}</span>
                                        </text>
                                    ))}
                                </box>
                            </box>
                        );
                    })}
                </box>
                <box width={2} />
                <box flexGrow={1} flexDirection="column">
                    {columns.right.map((row, index) => {
                        if (row.type === "header") {
                            return (
                                <text key={`right:${index}`} width="100%" wrapMode="char">
                                    <strong fg={COLORS.accent}>{row.text}</strong>
                                </text>
                            );
                        }

                        if (row.type === "spacer") {
                            return <text key={`right:${index}`}> </text>;
                        }

                        return (
                            <box key={`right:${index}`} width="100%" flexDirection="row" alignItems="flex-start">
                                <text width={keyWidth} wrapMode="char">
                                    <span fg={COLORS.text}>{row.label}</span>
                                    <span fg={COLORS.accent}>{row.labelSuffix}</span>
                                    <span fg={COLORS.disabled}>{metricLeaderText(`${row.label}${row.labelSuffix}`, keyWidth)}</span>
                                </text>
                                <box width={valueWidth} flexDirection="column">
                                    {wrapRightAlignedLines(row.value, valueWidth).map((line, lineIndex) => (
                                        <text key={`right:${index}:value:${lineIndex}`} width="100%" wrapMode="none">
                                            <span fg={COLORS.disabled}>{line.pad}</span>
                                            <span fg={COLORS.text}>{line.text}</span>
                                        </text>
                                    ))}
                                </box>
                            </box>
                        );
                    })}
                </box>
            </box>
        </box>
    );
}

function RunFooterView({
    state,
    metricRows,
    metricColumnWidth,
    metricsVisible,
    remoteVisible,
}: FooterViewState) {
    return (
        <box
            style={{
                width: "100%",
                height: "100%",
                flexDirection: "column",
                justifyContent: "flex-end",
            }}
        >
            <MetricsCard rows={metricRows} visible={metricsVisible} columnWidth={metricColumnWidth} />
            <RemoteCard state={remoteVisible ? state : { ...state, room: undefined }} />
            <HeaderBox state={state} />
        </box>
    );
}

function FooterRoot({ store }: { store: FooterViewStore }) {
    const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot);
    return <RunFooterView {...snapshot} />;
}

function createFooterInput(): NodeJS.ReadStream {
    const stream = new PassThrough() as unknown as NodeJS.ReadStream;
    (stream as NodeJS.ReadStream & { isTTY?: boolean }).isTTY = false;
    return stream;
}

export interface RunFooterHandle {
    readonly active: boolean;
    addMetrics(entries: MetricsEntry[]): void;
    setManifest(entries: ManifestEntry[]): void;
    setNexusConnected(connected: boolean | null): void;
    setRoom(room: RoomInfo): void;
    close(): Promise<void>;
}

class ActiveRunFooter implements RunFooterHandle {
    readonly active = true;

    private readonly root: ReturnType<typeof createRoot>;
    private readonly renderer: Awaited<ReturnType<typeof createCliRenderer>>;
    private readonly store: FooterViewStore;

    private readonly handleResize = () => {
        this.schedulePublish();
    };

    private state: FooterState;
    private closed = false;
    private publishScheduled = false;
    private timerInterval?: ReturnType<typeof setInterval>;

    constructor(renderer: Awaited<ReturnType<typeof createCliRenderer>>, state: FooterState) {
        this.renderer = renderer;
        this.root = createRoot(renderer);
        this.state = state;
        this.store = new FooterViewStore(this.buildViewState());
        this.renderer.footerHeight = HEADER_ONLY_HEIGHT;
        this.renderer.on("resize", this.handleResize);

        flushSync(() => {
            this.root.render(<FooterRoot store={this.store} />);
        });

        this.startTimer();
        this.publish();
    }

    private startTimer(): void {
        this.timerInterval = setInterval(() => {
            this.state = {
                ...this.state,
                elapsedSeconds: this.state.elapsedSeconds + 1,
            };
            this.schedulePublish();
        }, 1000);
    }

    addMetrics(entries: MetricsEntry[]): void {
        if (entries.length === 0) {
            return;
        }

        const existing = new Map<string, MetricsEntry>();
        for (const entry of this.state.metricsEntries) {
            existing.set(`${entry.block}\0${entry.key}`, entry);
        }

        for (const entry of entries) {
            existing.set(`${entry.block}\0${entry.key}`, { ...entry });
        }

        this.state = {
            ...this.state,
            metricsEntries: [...existing.values()],
        };
        this.schedulePublish();
    }

    setManifest(entries: ManifestEntry[]): void {
        this.state = {
            ...this.state,
            manifestCount: entries.length,
        };
        this.schedulePublish();
    }

    setNexusConnected(connected: boolean | null): void {
        this.state = {
            ...this.state,
            nexusConnected: connected,
        };
        this.schedulePublish();
    }

    setRoom(room: RoomInfo): void {
        this.state = {
            ...this.state,
            room: {
                ...this.state.room,
                ...room,
            },
        };
        this.schedulePublish();
    }

    async close(): Promise<void> {
        if (this.closed) {
            return;
        }

        this.closed = true;
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
        }

        this.publishScheduled = false;
        this.renderer.off("resize", this.handleResize);
        this.renderer.destroy();
    }

    private buildViewState(): FooterViewState {
        const remoteVisible = hasRemoteInfo(this.state);
        const remoteCardHeight = remoteVisible ? REMOTE_CARD_HEIGHT : 0;
        const metricColumnWidth = Math.max(8, Math.floor((this.renderer.width - 6) / 2));
        const metricRows = buildMetricRows(this.state.metricsEntries);
        const metricsVisible = metricRows.length > 0;
        const metricsCardHeight = metricsVisible ? metricColumnsHeight(metricRows, metricColumnWidth) + 2 : 0;

        this.renderer.footerHeight = HEADER_ONLY_HEIGHT + remoteCardHeight + metricsCardHeight;

        return {
            state: this.state,
            metricRows,
            metricColumnWidth,
            metricsVisible,
            remoteVisible,
        };
    }

    private schedulePublish(): void {
        if (this.closed || this.publishScheduled) {
            return;
        }

        this.publishScheduled = true;
        setImmediate(() => {
            this.publishScheduled = false;
            this.publish();
        });
    }

    private publish(): void {
        if (this.closed) {
            return;
        }

        this.store.setSnapshot(this.buildViewState());
    }
}

class InactiveRunFooter implements RunFooterHandle {
    readonly active = false;
    addMetrics(): void {}
    setManifest(): void {}
    setNexusConnected(): void {}
    setRoom(): void {}
    async close(): Promise<void> {}
}

interface CreateRunFooterOptions {
    enabled?: boolean;
    stdout?: NodeJS.WriteStream;
}

export async function createRunFooter(options: CreateRunFooterOptions = {}): Promise<RunFooterHandle> {
    const output = options.stdout ?? process.stdout;
    if (options.enabled === false || output.isTTY !== true) {
        return new InactiveRunFooter();
    }

    try {
        const renderer = await createCliRenderer({
            stdin: createFooterInput(),
            stdout: output,
            exitOnCtrlC: false,
            autoFocus: false,
            useMouse: false,
            screenMode: "split-footer",
            footerHeight: HEADER_ONLY_HEIGHT,
            externalOutputMode: "capture-stdout",
            consoleMode: "disabled",
            targetFps: 8,
            gatherStats: false,
            openConsoleOnError: false,
        });

        return new ActiveRunFooter(renderer, {
            metricsEntries: [],
            manifestCount: 0,
            nexusConnected: null,
            elapsedSeconds: 0,
        });
    } catch {
        return new InactiveRunFooter();
    }
}
