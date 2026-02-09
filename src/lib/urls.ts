import { readFile } from "node:fs/promises";

function isHttpUrl(s: string): boolean {
  return s.startsWith("http://") || s.startsWith("https://");
}

function uniq(arr: string[]): string[] {
  return Array.from(new Set(arr));
}

export async function loadUrlsFromArgsEnvOrFile(args: string[], opts: {
  envUrls?: string;
  defaultFile: string;
}): Promise<{ urls: string[]; source: string }> {
  const fileIdx = args.findIndex((a) => a === "--file" || a === "-f");
  const fileFromArgs = fileIdx >= 0 ? args[fileIdx + 1] : undefined;

  const argUrls = args.filter((a) => isHttpUrl(a));
  if (argUrls.length > 0) {
    return { urls: uniq(argUrls), source: "cli args" };
  }

  const envUrls = opts.envUrls?.trim();
  if (envUrls) {
    const urls = envUrls
      .split(/[\s,]+/g)
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
      .filter(isHttpUrl);
    return { urls: uniq(urls), source: "URLS env var" };
  }

  const filePath = (fileFromArgs ?? opts.defaultFile).trim();
  const contents = await readFile(filePath, "utf8");
  const urls = contents
    .split(/\r?\n/g)
    .map((l) => l.trim())
    .filter((l) => l.length > 0)
    .filter((l) => !l.startsWith("#"))
    .filter(isHttpUrl);

  return { urls: uniq(urls), source: `file:${filePath}` };
}


