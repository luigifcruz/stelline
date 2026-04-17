import { runCyberetherCommand } from "../run/run.js";

interface RunCommandOptions {
    plain?: boolean;
}

function normalizeRunInvocation(
    rawArgs: string[],
    requestedPlain: boolean,
): { args: string[]; plain: boolean } {
    const args: string[] = [];
    let plain = requestedPlain;
    let parsingWrapperOptions = true;

    for (const arg of rawArgs) {
        if (parsingWrapperOptions && arg === "--") {
            parsingWrapperOptions = false;
            continue;
        }

        if (parsingWrapperOptions && arg === "--plain") {
            plain = true;
            continue;
        }

        args.push(arg);

        if (parsingWrapperOptions && !arg.startsWith("-")) {
            parsingWrapperOptions = false;
        }
    }

    return { args, plain };
}

export async function runRunCommand(
    rawArgs: string[],
    options: RunCommandOptions = {},
): Promise<number> {
    const { args, plain } = normalizeRunInvocation(
        rawArgs,
        options.plain ?? false,
    );

    return await runCyberetherCommand(args, { plain });
}
