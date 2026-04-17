import { TextAttributes } from "@opentui/core";

import {
    buildTopologyModel,
    type CpuClass,
    type StyledCpu,
    type TopologyModel,
    type TopologyRow,
} from "./model.js";
import { renderStaticView } from "../shared/tui.js";

const COLUMNS = [
    {
        key: "gpuId",
        label: "ID",
        preferredWidth: 6,
        minWidth: 4,
        grow: 0,
        shrink: 0,
    },
    {
        key: "gpuName",
        label: "GPU Name",
        preferredWidth: 24,
        minWidth: 14,
        grow: 3,
        shrink: 3,
    },
    {
        key: "gpuPcie",
        label: "GPU PCIe",
        preferredWidth: 14,
        minWidth: 12,
        grow: 1,
        shrink: 2,
    },
    {
        key: "numa",
        label: "NUMA",
        preferredWidth: 6,
        minWidth: 4,
        grow: 0,
        shrink: 0,
    },
    {
        key: "cpus",
        label: "CPUs",
        preferredWidth: 28,
        minWidth: 18,
        grow: 4,
        shrink: 4,
    },
    {
        key: "nic",
        label: "NIC",
        preferredWidth: 10,
        minWidth: 7,
        grow: 1,
        shrink: 1,
    },
    {
        key: "nicPcie",
        label: "NIC PCIe",
        preferredWidth: 14,
        minWidth: 12,
        grow: 1,
        shrink: 2,
    },
    {
        key: "iface",
        label: "Interface",
        preferredWidth: 10,
        minWidth: 8,
        grow: 2,
        shrink: 2,
    },
] as const;

type ColumnKey = (typeof COLUMNS)[number]["key"];
type ColumnWidths = Record<ColumnKey, number>;
type InkColor = "green" | "red" | "blue" | undefined;
type CpuSegment = { text: string; color?: InkColor };
type TableLine =
    | { kind: "separator"; key: string }
    | {
          kind: "row";
          key: string;
          gpuId: string;
          gpuName: string;
          gpuPcie: string;
          numa: string;
          cpus: CpuSegment[];
          nic: string;
          nicColor?: string;
          nicDimColor?: boolean;
          nicPcie: string;
          nicPcieDimColor?: boolean;
          iface: string;
          ifaceDimColor?: boolean;
      };

const PREFERRED_TABLE_WIDTH =
    COLUMNS.reduce((sum, column) => sum + column.preferredWidth, 0) +
    (COLUMNS.length - 1);
const MIN_TABLE_WIDTH =
    COLUMNS.reduce((sum, column) => sum + column.minWidth, 0) +
    (COLUMNS.length - 1);
const CARD_HORIZONTAL_CHROME = 4;

function getTableWidth(columnWidths: ColumnWidths): number {
    return (
        COLUMNS.reduce((sum, column) => sum + columnWidths[column.key], 0) +
        (COLUMNS.length - 1)
    );
}

function distributeExtraWidth(columnWidths: ColumnWidths, extraWidth: number): void {
    const growableColumns = COLUMNS.filter((column) => column.grow > 0);
    const totalGrow = growableColumns.reduce((sum, column) => sum + column.grow, 0);

    if (extraWidth <= 0 || totalGrow === 0) {
        return;
    }

    let allocated = 0;

    for (const column of growableColumns) {
        const extra = Math.floor((extraWidth * column.grow) / totalGrow);
        columnWidths[column.key] += extra;
        allocated += extra;
    }

    let remaining = extraWidth - allocated;

    for (const column of growableColumns) {
        if (remaining === 0) {
            break;
        }

        columnWidths[column.key] += 1;
        remaining -= 1;
    }
}

function shrinkWidths(columnWidths: ColumnWidths, deficit: number): void {
    let remaining = deficit;

    while (remaining > 0) {
        const shrinkableColumns = COLUMNS.filter((column) => {
            return column.shrink > 0 && columnWidths[column.key] > column.minWidth;
        });

        if (shrinkableColumns.length === 0) {
            break;
        }

        const totalShrink = shrinkableColumns.reduce(
            (sum, column) => sum + column.shrink,
            0,
        );

        let shrunkThisRound = 0;

        for (const column of shrinkableColumns) {
            const capacity = columnWidths[column.key] - column.minWidth;
            const requested = Math.max(
                1,
                Math.floor((remaining * column.shrink) / totalShrink),
            );
            const applied = Math.min(capacity, requested);

            columnWidths[column.key] -= applied;
            shrunkThisRound += applied;
        }

        remaining -= shrunkThisRound;

        if (shrunkThisRound === 0) {
            break;
        }
    }
}

function computeColumnWidths(availableWidth: number): ColumnWidths {
    const columnWidths = Object.fromEntries(
        COLUMNS.map((column) => [column.key, column.preferredWidth]),
    ) as ColumnWidths;

    if (availableWidth >= PREFERRED_TABLE_WIDTH) {
        distributeExtraWidth(columnWidths, availableWidth - PREFERRED_TABLE_WIDTH);
        return columnWidths;
    }

    if (availableWidth >= MIN_TABLE_WIDTH) {
        shrinkWidths(columnWidths, PREFERRED_TABLE_WIDTH - availableWidth);
        return columnWidths;
    }

    return Object.fromEntries(
        COLUMNS.map((column) => [column.key, column.minWidth]),
    ) as ColumnWidths;
}

function pad(value: string, width: number): string {
    if (value.length > width) {
        return value.slice(0, Math.max(width - 1, 1)) + (width > 1 ? "…" : "");
    }

    return value.padEnd(width, " ");
}

function wrapText(value: string, width: number): string[] {
    if (!value) {
        return [" ".repeat(width)];
    }

    const words = value.split(/\s+/).filter(Boolean);

    if (words.length === 0) {
        return [" ".repeat(width)];
    }

    const lines: string[] = [];
    let currentLine = "";

    const pushLine = () => {
        lines.push(currentLine.padEnd(width, " "));
        currentLine = "";
    };

    for (const word of words) {
        if (word.length > width) {
            if (currentLine) {
                pushLine();
            }

            let remaining = word;
            while (remaining.length > width) {
                lines.push(remaining.slice(0, Math.max(width - 1, 1)) + (width > 1 ? "…" : ""));
                remaining = remaining.slice(Math.max(width - 1, 1));
            }

            currentLine = remaining;
            continue;
        }

        if (!currentLine) {
            currentLine = word;
            continue;
        }

        if (currentLine.length + 1 + word.length <= width) {
            currentLine = `${currentLine} ${word}`;
            continue;
        }

        pushLine();
        currentLine = word;
    }

    if (currentLine) {
        pushLine();
    }

    return lines.length > 0 ? lines : [" ".repeat(width)];
}

function cpuColor(className: CpuClass): InkColor {
    if (className === "isolated-irq-free") {
        return "green";
    }
    if (className === "isolated-only") {
        return "red";
    }
    if (className === "irq-free-only") {
        return "blue";
    }
    return undefined;
}

function finalizeCpuLine(segments: CpuSegment[], width: number): CpuSegment[] {
    const used = segments.reduce((sum, segment) => sum + segment.text.length, 0);

    if (segments.length === 0) {
        return [{ text: " ".repeat(width) }];
    }

    if (used < width) {
        return [...segments, { text: " ".repeat(width - used) }];
    }

    return segments;
}

function wrapCpuSegments(cpus: StyledCpu[], width: number): CpuSegment[][] {
    if (cpus.length === 0) {
        return [[{ text: " ".repeat(width) }]];
    }

    const lines: CpuSegment[][] = [];
    let currentLine: CpuSegment[] = [];
    let used = 0;

    const pushLine = () => {
        lines.push(finalizeCpuLine(currentLine, width));
        currentLine = [];
        used = 0;
    };

    for (const cpu of cpus) {
        const token = cpu.value;
        const tokenLength = token.length;
        const color = cpuColor(cpu.className);

        if (currentLine.length > 0 && used + 1 + tokenLength > width) {
            pushLine();
        }

        if (currentLine.length > 0) {
            currentLine.push({ text: " " });
            used += 1;
        }

        currentLine.push({ text: token, color });
        used += tokenLength;
    }

    pushLine();

    return lines;
}

function makeTableLines(rows: TopologyRow[], columnWidths: ColumnWidths): TableLine[] {
    const lines: TableLine[] = [];

    for (const [rowIndex, row] of rows.entries()) {
        if (row.sectionStart) {
            lines.push({ kind: "separator", key: `separator-${rowIndex}` });
        }

        const cpuLines = wrapCpuSegments(row.cpus, columnWidths.cpus);
        const gpuNameLines = wrapText(row.gpuName, columnWidths.gpuName);
        const lineCount = Math.max(cpuLines.length, gpuNameLines.length);

        for (let lineIndex = 0; lineIndex < lineCount; lineIndex += 1) {
            const isFirstLine = lineIndex === 0;
            const cpuLine = cpuLines[lineIndex] ?? [{ text: " ".repeat(columnWidths.cpus) }];
            const gpuNameLine = gpuNameLines[lineIndex] ?? " ".repeat(columnWidths.gpuName);

            lines.push({
                kind: "row",
                key: `row-${rowIndex}-${lineIndex}-${row.gpuId}-${row.nic}`,
                gpuId: isFirstLine ? row.gpuId : "",
                gpuName: gpuNameLine,
                gpuPcie: isFirstLine ? row.gpuPcie : "",
                numa: isFirstLine ? row.numa : "",
                cpus: cpuLine,
                nic: isFirstLine ? row.nic : "",
                nicColor: isFirstLine && row.nic !== "none" ? "magenta" : undefined,
                nicDimColor: isFirstLine ? row.nic === "none" : false,
                nicPcie: isFirstLine ? row.nicPcie : "",
                nicPcieDimColor: isFirstLine ? row.nicPcie === "-" : false,
                iface: isFirstLine ? row.iface : "",
                ifaceDimColor: isFirstLine ? row.iface === "-" : false,
            });
        }
    }

    return lines;
}

function Cell({
    width,
    value,
    color,
    dimColor,
}: {
    width: number;
    value: string;
    color?: string;
    dimColor?: boolean;
}) {
    return (
        <box width={width} flexShrink={0}>
            <text
                content={pad(value, width)}
                fg={color}
                attributes={dimColor ? TextAttributes.DIM : 0}
            />
        </box>
    );
}

function Gap() {
    return (
        <box width={1} flexShrink={0}>
            <text content=" " />
        </box>
    );
}

function CpuCell({ cpus, width }: { cpus: CpuSegment[]; width: number }) {
    return (
        <box width={width} flexShrink={0}>
            <text>
                {cpus.map((segment, index) => (
                    <span key={index} fg={segment.color}>
                        {segment.text}
                    </span>
                ))}
            </text>
        </box>
    );
}

function TableHeader({ columnWidths }: { columnWidths: ColumnWidths }) {
    return (
        <box flexDirection="row" width={getTableWidth(columnWidths)}>
            <Cell width={columnWidths.gpuId} value="ID" color="cyan" />
            <Gap />
            <Cell width={columnWidths.gpuName} value="GPU Name" color="cyan" />
            <Gap />
            <Cell width={columnWidths.gpuPcie} value="GPU PCIe" color="cyan" />
            <Gap />
            <Cell width={columnWidths.numa} value="NUMA" color="cyan" />
            <Gap />
            <CpuHeaderCell width={columnWidths.cpus} />
            <Gap />
            <Cell width={columnWidths.nic} value="NIC" color="cyan" />
            <Gap />
            <Cell width={columnWidths.nicPcie} value="NIC PCIe" color="cyan" />
            <Gap />
            <Cell width={columnWidths.iface} value="Interface" color="cyan" />
        </box>
    );
}

function CpuHeaderCell({ width }: { width: number }) {
    return (
        <box width={width} flexShrink={0}>
            <text>
                <span fg="cyan">CPUs (</span>
                <span fg="green">■</span>
                <span fg="cyan"> iso+irq </span>
                <span fg="red">■</span>
                <span fg="cyan"> iso </span>
                <span fg="blue">■</span>
                <span fg="cyan"> irq)</span>
            </text>
        </box>
    );
}

function DataRow({
    line,
    columnWidths,
}: {
    line: Extract<TableLine, { kind: "row" }>;
    columnWidths: ColumnWidths;
}) {
    return (
        <box flexDirection="row" width={getTableWidth(columnWidths)}>
            <Cell width={columnWidths.gpuId} value={line.gpuId} color="cyan" />
            <Gap />
            <Cell width={columnWidths.gpuName} value={line.gpuName} />
            <Gap />
            <Cell width={columnWidths.gpuPcie} value={line.gpuPcie} />
            <Gap />
            <Cell width={columnWidths.numa} value={line.numa} />
            <Gap />
            <CpuCell cpus={line.cpus} width={columnWidths.cpus} />
            <Gap />
            <Cell
                width={columnWidths.nic}
                value={line.nic}
                color={line.nicColor}
                dimColor={line.nicDimColor}
            />
            <Gap />
            <Cell
                width={columnWidths.nicPcie}
                value={line.nicPcie}
                dimColor={line.nicPcieDimColor}
            />
            <Gap />
            <Cell
                width={columnWidths.iface}
                value={line.iface}
                dimColor={line.ifaceDimColor}
            />
        </box>
    );
}

function TableBody({
    lines,
    columnWidths,
}: {
    lines: TableLine[];
    columnWidths: ColumnWidths;
}) {
    const separator = "".padEnd(getTableWidth(columnWidths), "─");

    if (lines.length === 0) {
        return <text fg="gray">No topology rows available.</text>;
    }

    return (
        <box flexDirection="column">
            {lines.map((line) => {
                if (line.kind === "separator") {
                    return <text key={line.key} fg="gray" content={separator} />;
                }

                return (
                    <DataRow key={line.key} line={line} columnWidths={columnWidths} />
                );
            })}
        </box>
    );
}

function HeaderCard({ width }: { width: number }) {
    return (
        <box
            flexDirection="column"
            border
            borderStyle="rounded"
            borderColor="cyan"
            paddingX={1}
            width={width + CARD_HORIZONTAL_CHROME}
        >
            <box width={width} alignItems="center" justifyContent="center">
                <text>
                    <strong fg="cyan">Stelline CLI - Topology</strong>
                </text>
            </box>
        </box>
    );
}

function WidthNote({
    tableWidth,
    renderWidth,
}: {
    tableWidth: number;
    renderWidth: number;
}) {
    if (tableWidth <= renderWidth) {
        return null;
    }

    return (
        <text fg="gray" attributes={TextAttributes.DIM}>
            {`Note: the table needs ${tableWidth} columns for a fully unwrapped layout.`}
        </text>
    );
}

function TopologyReport({
    model,
    renderWidth,
}: {
    model: TopologyModel;
    renderWidth: number;
}) {
    const availableContentWidth = Math.max(
        renderWidth - CARD_HORIZONTAL_CHROME,
        MIN_TABLE_WIDTH,
    );
    const columnWidths = computeColumnWidths(availableContentWidth);
    const tableWidth = getTableWidth(columnWidths);
    const lines = makeTableLines(model.rows, columnWidths);

    return (
        <box flexDirection="column" width={tableWidth + CARD_HORIZONTAL_CHROME}>
            <HeaderCard width={tableWidth} />
            {model.warnings.map((warning) => (
                <text key={warning} fg="yellow">
                    {warning}
                </text>
            ))}
            <box
                flexDirection="column"
                border
                borderStyle="rounded"
                borderColor="gray"
                paddingX={1}
            >
                <TableHeader columnWidths={columnWidths} />
                <text fg="gray" content={"".padEnd(tableWidth, "─")} />
                <TableBody lines={lines} columnWidths={columnWidths} />
            </box>
            <WidthNote
                tableWidth={tableWidth + CARD_HORIZONTAL_CHROME}
                renderWidth={renderWidth}
            />
        </box>
    );
}

function getRenderWidth(): number {
    const stdoutWidth = process.stdout.columns;
    if (typeof stdoutWidth === "number" && stdoutWidth > 0) {
        return stdoutWidth;
    }

    const envWidth = Number.parseInt(process.env.COLUMNS ?? "", 10);
    if (Number.isFinite(envWidth) && envWidth > 0) {
        return envWidth;
    }

    return PREFERRED_TABLE_WIDTH;
}

export async function runTopoCommand(): Promise<number> {
    try {
        const model = await buildTopologyModel();
        const renderWidth = getRenderWidth();
        const output = await renderStaticView(
            <TopologyReport model={model} renderWidth={renderWidth} />,
            {
                width: Math.max(renderWidth, MIN_TABLE_WIDTH + CARD_HORIZONTAL_CHROME),
                height: Math.max(model.rows.length * 4 + model.warnings.length + 16, 32),
            },
        );

        process.stdout.write(`${output}\n`);
        return model.rows.length > 0 ? 0 : 1;
    } catch (error) {
        const message =
            error instanceof Error ? error.stack ?? error.message : String(error);
        process.stderr.write(`Failed to start topo view:\n${message}\n`);
        return 1;
    }
}
