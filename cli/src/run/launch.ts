import { spawnSync } from "node:child_process";
import { accessSync, constants } from "node:fs";
import { join } from "node:path";

export interface RunResolution {
    stellineLibraryPath: string;
}

function uniquePaths(entries: Array<string | undefined>): string[] {
    const seen = new Set<string>();
    const values: string[] = [];

    for (const entry of entries) {
        if (!entry || seen.has(entry)) {
            continue;
        }

        seen.add(entry);
        values.push(entry);
    }

    return values;
}

function isReadableFile(path: string): boolean {
    try {
        accessSync(path, constants.R_OK);
        return true;
    } catch {
        return false;
    }
}

function runPkgConfig(args: string[]): { status: number | null; stdout: string; error?: Error } {
    const proc = spawnSync("pkg-config", args, {
        encoding: "utf8",
        env: process.env,
    });

    return {
        status: proc.status,
        stdout: proc.stdout ?? "",
        error: proc.error,
    };
}

function pkgConfigVariable(packageName: string, variableName: string): string | undefined {
    const proc = runPkgConfig([`--variable=${variableName}`, packageName]);
    if (proc.error) {
        throw new Error(`pkg-config is required to resolve '${packageName}': ${proc.error.message}`);
    }

    if (proc.status !== 0) {
        return undefined;
    }

    const value = proc.stdout.trim();
    return value.length > 0 ? value : undefined;
}

function pkgConfigLibraryDirs(packageName: string): string[] {
    const proc = runPkgConfig(["--libs-only-L", packageName]);
    if (proc.error) {
        throw new Error(`pkg-config is required to resolve '${packageName}': ${proc.error.message}`);
    }

    if (proc.status !== 0) {
        return [];
    }

    return proc.stdout
        .trim()
        .split(/\s+/)
        .map((token) => (token.startsWith("-L") ? token.slice(2) : ""))
        .filter((token) => token.length > 0);
}

function resolveLibraryPath(packageName: string, soname: string): string {
    const directories = uniquePaths([
        pkgConfigVariable(packageName, "libdir"),
        ...pkgConfigLibraryDirs(packageName),
    ]);

    if (directories.length === 0) {
        throw new Error(
            `Could not resolve '${packageName}' with pkg-config. Make sure ${packageName} is installed and visible in PKG_CONFIG_PATH.`,
        );
    }

    for (const directory of directories) {
        const candidate = join(directory, soname);
        if (isReadableFile(candidate)) {
            return candidate;
        }
    }

    throw new Error(
        `pkg-config resolved '${packageName}', but '${soname}' was not found in: ${directories.join(", ")}`,
    );
}

export function resolveRunLaunch(): RunResolution {
    return {
        stellineLibraryPath: resolveLibraryPath("stelline", "libstelline.so"),
    };
}
