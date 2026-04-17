#!/usr/bin/env bun

import { Command } from "commander";

function buildProgram(setExitCode: (code: number) => void): Command {
    const program = new Command()
        .name("stelline")
        .description("Stelline command line interface")
        .usage("<command> [options]")
        .enablePositionalOptions()
        .showHelpAfterError()
        .showSuggestionAfterError()
        .addHelpText(
            "after",
            ["", "Examples:", "  $ stelline --help"].join("\n"),
        );

    program
        .command("help")
        .description("Display help information")
        .action(() => {
            program.outputHelp();
        });

    program
        .command("topo")
        .description("Display system topology.")
        .action(async () => {
            const { runTopoCommand } = await import("./commands/topo.js");
            setExitCode(await runTopoCommand());
        });

    program
        .command("report")
        .description("Generate and write a markdown system report.")
        .option("--file [path]", "Write markdown report to a custom path.")
        .action(async (options) => {
            const { runReportCommand } = await import("./commands/report.js");
            setExitCode(
                await runReportCommand({
                    file:
                        options.file === true
                            ? "report.md"
                            : typeof options.file === "string"
                              ? options.file
                              : "report.md",
                }),
            );
        });

    program
        .command("run [args...]")
        .description("Run CyberEther with Stelline and BLADE modules.")
        .allowUnknownOption(true)
        .passThroughOptions()
        .action(async (args: string[]) => {
            const { runRunCommand } = await import("./commands/run.js");
            setExitCode(await runRunCommand(args));
        });

    return program;
}

async function main(argv: string[]): Promise<number> {
    let exitCode = 0;
    const program = buildProgram((code) => {
        exitCode = code;
    });

    if (argv.length === 0) {
        program.outputHelp();
        return 0;
    }

    await program.parseAsync(argv, { from: "user" });
    return exitCode;
}

process.exitCode = await main(process.argv.slice(2));
