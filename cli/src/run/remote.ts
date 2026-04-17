import type { RoomInfo } from "./footer.js";

const ANSI_ESCAPE_RE = /\x1b\[[0-?]*[ -/]*[@-~]/g;

function stripAnsi(text: string): string {
    return text.replace(ANSI_ESCAPE_RE, "").replace(/\r/g, "");
}

function parseJoinUrl(text: string): string | undefined {
    const match = text.match(/https?:\/\/\S*\/remote#\S+/i);
    return match?.[0];
}

export class LineBuffer {
    private buffer = "";

    constructor(private readonly onLine: (line: string) => void) {}

    push(chunk: string): void {
        let next = `${this.buffer}${chunk.replace(/\r\n/g, "\n").replace(/\r/g, "\n")}`;
        let newlineIndex = next.indexOf("\n");

        while (newlineIndex >= 0) {
            const line = next.slice(0, newlineIndex);
            next = next.slice(newlineIndex + 1);
            this.onLine(line);
            newlineIndex = next.indexOf("\n");
        }

        this.buffer = next;
    }

    flush(): void {
        if (this.buffer.length > 0) {
            this.onLine(this.buffer);
            this.buffer = "";
        }
    }
}

export function parseRemoteInfo(line: string): RoomInfo | undefined {
    const cleanLine = stripAnsi(line).trim();
    if (cleanLine.length === 0) {
        return undefined;
    }

    const joinUrl = parseJoinUrl(cleanLine);
    const info: RoomInfo = {};

    const roomIdMatch = cleanLine.match(/\broom\s+id\b\s*[:=]?\s+(.+)$/i);
    if (roomIdMatch) {
        info.roomId = roomIdMatch[1].trim();
    }

    const accessTokenMatch = cleanLine.match(/\baccess\s+token\b\s*[:=]?\s+(.+)$/i);
    if (accessTokenMatch) {
        info.accessToken = accessTokenMatch[1].trim();
    }

    const joinUrlMatch = cleanLine.match(/\bjoin\s+url\b\s*[:=]?\s+(https?:\/\/\S+)$/i);
    if (joinUrlMatch) {
        info.joinUrl = joinUrlMatch[1].trim();
    } else if (joinUrl) {
        info.joinUrl = joinUrl;
    }

    if (!info.accessToken && info.joinUrl) {
        try {
            const url = new URL(info.joinUrl);
            if (url.hash.length > 1) {
                info.accessToken = url.hash.slice(1);
            }
        } catch {
            // Ignore malformed URLs in process output.
        }
    }

    if (!info.roomId && !info.accessToken && !info.joinUrl) {
        return undefined;
    }

    return info;
}
