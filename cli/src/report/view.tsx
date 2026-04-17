import { resolve } from "node:path";
import type { ReactNode } from "react";

import {
    reportExitCode,
    type LogEntry,
    type SectionStatus,
    type Status,
    writeReportFile,
} from "./model.js";
import { renderStaticView } from "../shared/tui.js";

type TuiColor = "green" | "yellow" | "red" | "gray" | "cyan";

interface ReportCommandOptions {
    file?: string;
}

const CARD_HORIZONTAL_CHROME = 4;
const DEFAULT_RENDER_WIDTH = 100;

function statusColor(status: Status): TuiColor {
    if (status === "OK") {
        return "green";
    }
    if (status === "WARN") {
        return "yellow";
    }
    if (status === "ERROR") {
        return "red";
    }
    return "gray";
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

    return DEFAULT_RENDER_WIDTH;
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
                    <strong fg="cyan">Stelline CLI - Report</strong>
                </text>
            </box>
        </box>
    );
}

function Card({
    width,
    title,
    borderColor = "gray",
    children,
}: {
    width: number;
    title?: string;
    borderColor?: TuiColor;
    children: ReactNode;
}) {
    return (
        <box
            flexDirection="column"
            border
            borderStyle="rounded"
            borderColor={borderColor}
            paddingX={1}
            width={width + CARD_HORIZONTAL_CHROME}
        >
            {title ? (
                <text>
                    <strong fg="cyan">{title}</strong>
                </text>
            ) : null}
            {children}
        </box>
    );
}

function SummaryCard({
    width,
    sectionStatuses,
    logs,
}: {
    width: number;
    sectionStatuses: SectionStatus[];
    logs: LogEntry[];
}) {
    const trailingLogs = logs.filter(
        (log) =>
            log.section !== "Output" &&
            !sectionStatuses.some(([name]) => name === log.section),
    );

    return (
        <Card width={width}>
            <box flexDirection="column">
                {sectionStatuses.map(([name, status]) => {
                    const sectionLogs = logs.filter(
                        (log) => log.section === name,
                    );

                    return (
                        <box key={`${name}-${status}`} flexDirection="column">
                            <text>
                                <span fg={statusColor(status)}>
                                    {status === "OK"
                                        ? "[✓]"
                                        : status === "WARN"
                                          ? "[!]"
                                          : status === "ERROR"
                                            ? "[x]"
                                            : "[ ]"}
                                </span>{" "}
                                <strong>{name}</strong>
                                <span
                                    fg={statusColor(status)}
                                >{` - ${status}`}</span>
                            </text>
                            {sectionLogs.map((log, index) => (
                                <text key={`${name}-${log.status}-${index}`}>
                                    {"   "}
                                    <span fg={statusColor(log.status)}>-</span>
                                    <span fg="gray">{` ${log.message}`}</span>
                                </text>
                            ))}
                        </box>
                    );
                })}
                {trailingLogs.length > 0 ? <text> </text> : null}
                {trailingLogs.map((log, index) => (
                    <text key={`${log.section}-${log.status}-${index}`}>
                        {"   "}
                        <span fg={statusColor(log.status)}>-</span>
                        <span fg="gray">{` ${log.message}`}</span>
                    </text>
                ))}
            </box>
        </Card>
    );
}

function OutputCard({
    width,
    message,
    writeSucceeded,
}: {
    width: number;
    message: string;
    writeSucceeded: boolean;
}) {
    return (
        <Card width={width} borderColor={writeSucceeded ? "green" : "red"}>
            <text>
                <span fg={writeSucceeded ? "green" : "red"}>{message}</span>
            </text>
        </Card>
    );
}

function ReportWriteView({
    width,
    sectionStatuses,
    logs,
    outputMessage,
    writeSucceeded,
}: {
    width: number;
    sectionStatuses: SectionStatus[];
    logs: LogEntry[];
    outputMessage: string;
    writeSucceeded: boolean;
}) {
    return (
        <box flexDirection="column" width={width + CARD_HORIZONTAL_CHROME}>
            <HeaderCard width={width} />
            <SummaryCard
                width={width}
                sectionStatuses={sectionStatuses}
                logs={logs}
            />
            <OutputCard
                width={width}
                message={outputMessage}
                writeSucceeded={writeSucceeded}
            />
        </box>
    );
}

export async function runReportCommand(
    options: ReportCommandOptions,
): Promise<number> {
    const report = writeReportFile(options.file ?? "report.md");
    const renderWidth = getRenderWidth();
    const contentWidth = Math.max(renderWidth - CARD_HORIZONTAL_CHROME, 60);
    const output = await renderStaticView(
        <ReportWriteView
            width={contentWidth}
            sectionStatuses={report.sectionStatuses}
            logs={report.logs}
            outputMessage={
                report.writeSucceeded
                    ? `Report written to: ${resolve(report.outputPath)}`
                    : report.outputMessage
            }
            writeSucceeded={report.writeSucceeded}
        />,
        {
            width: Math.max(renderWidth, contentWidth + CARD_HORIZONTAL_CHROME),
            height: Math.max(
                report.sectionStatuses.length + report.logs.length + 12,
                32,
            ),
        },
    );

    process.stdout.write(`${output}\n`);
    return reportExitCode(report.overallStatus);
}
