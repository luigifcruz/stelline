import {
    TextAttributes,
    getBaseAttributes,
    type CapturedFrame,
    type CapturedSpan,
} from "@opentui/core";
import { createTestRenderer } from "@opentui/core/testing";
import { createRoot, flushSync } from "@opentui/react";
import type { ReactNode } from "react";

const ANSI_RESET = "\u001b[0m";
const DEFAULT_RENDER_HEIGHT = 512;

interface StaticRenderOptions {
    width: number;
    height?: number;
}

function spanColorCode(span: CapturedSpan, kind: "fg" | "bg"): string | undefined {
    const color = kind === "fg" ? span.fg : span.bg;
    const [red, green, blue, alpha] = color.toInts();

    if (alpha === 0) {
        return undefined;
    }

    if (kind === "fg" && red === 255 && green === 255 && blue === 255) {
        return undefined;
    }

    return kind === "fg"
        ? `38;2;${red};${green};${blue}`
        : `48;2;${red};${green};${blue}`;
}

function spanAttributeCodes(span: CapturedSpan): string[] {
    const attributes = getBaseAttributes(span.attributes);
    const codes: string[] = [];

    if ((attributes & TextAttributes.BOLD) !== 0) {
        codes.push("1");
    }
    if ((attributes & TextAttributes.DIM) !== 0) {
        codes.push("2");
    }
    if ((attributes & TextAttributes.ITALIC) !== 0) {
        codes.push("3");
    }
    if ((attributes & TextAttributes.UNDERLINE) !== 0) {
        codes.push("4");
    }
    if ((attributes & TextAttributes.BLINK) !== 0) {
        codes.push("5");
    }
    if ((attributes & TextAttributes.INVERSE) !== 0) {
        codes.push("7");
    }
    if ((attributes & TextAttributes.HIDDEN) !== 0) {
        codes.push("8");
    }
    if ((attributes & TextAttributes.STRIKETHROUGH) !== 0) {
        codes.push("9");
    }

    const fgCode = spanColorCode(span, "fg");
    if (fgCode) {
        codes.push(fgCode);
    }

    const bgCode = spanColorCode(span, "bg");
    if (bgCode) {
        codes.push(bgCode);
    }

    return codes;
}

function renderSpan(span: CapturedSpan): string {
    const codes = spanAttributeCodes(span);
    if (codes.length === 0) {
        return span.text;
    }

    return `\u001b[${codes.join(";")}m${span.text}${ANSI_RESET}`;
}

function frameToAnsi(frame: CapturedFrame): string {
    const lines = frame.lines.map((line) => {
        return line.spans.map(renderSpan).join("").replace(/[ \t]+$/g, "");
    });

    while (lines.length > 0 && lines.at(-1)?.trim().length === 0) {
        lines.pop();
    }

    return lines.join("\n");
}

export async function renderStaticView(
    view: ReactNode,
    options: StaticRenderOptions,
): Promise<string> {
    const { renderer, renderOnce, captureSpans } = await createTestRenderer({
        width: options.width,
        height: options.height ?? DEFAULT_RENDER_HEIGHT,
        screenMode: "main-screen",
        consoleMode: "disabled",
        externalOutputMode: "passthrough",
    });
    const root = createRoot(renderer);

    try {
        flushSync(() => {
            root.render(view);
        });
        await renderOnce();

        return frameToAnsi(captureSpans());
    } finally {
        root.unmount();
        renderer.destroy();
    }
}
