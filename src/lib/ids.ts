import { createHash } from "node:crypto";

export function stableChunkId(source: string, content: string, index?: number): string {
  const hash = createHash("sha256")
    .update(source)
    .update("\n");

  if (index !== undefined) {
    hash.update(String(index)).update("\n");
  }

  const hex = hash.update(content).digest("hex");
  // Keep IDs short-ish but still collision-resistant for small projects.
  return hex.slice(0, 32);
}


