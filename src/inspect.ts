import { ChromaClient } from "chromadb";
import { CONFIG } from "./lib/config.js";

function chromaClientParams(): { tenant?: string; database?: string } {
    return {
      ...(CONFIG.chromaTenant ? { tenant: CONFIG.chromaTenant } : {}),
      ...(CONFIG.chromaDatabase ? { database: CONFIG.chromaDatabase } : {}),
    };
}

function printChunkCountByDateTs(metadatas: Array<Record<string, unknown> | null | undefined>): void {
    const counts = new Map<number, number>();

    for (const metadata of metadatas) {
        const raw = metadata?.date_ts;
        const dateTs = typeof raw === "number" && Number.isFinite(raw) ? raw : 0;
        counts.set(dateTs, (counts.get(dateTs) ?? 0) + 1);
    }

    if (counts.size === 0) {
        console.log("  - No metadata rows found");
        return;
    }

    console.log("  Chunk count grouped by metadata.date_ts:");
    for (const [dateTs, count] of [...counts.entries()].sort((a, b) => a[0] - b[0])) {
        const dateLabel = dateTs > 0 ? new Date(dateTs).toISOString().split("T")[0] : "unknown";
        console.log(`    - ${dateTs} (${dateLabel}): ${count} chunk(s)`);
    }
}

function printDocumentCountByDateTs(metadatas: Array<Record<string, unknown> | null | undefined>): void {
    const docsByDate = new Map<number, Set<string>>();

    for (let i = 0; i < metadatas.length; i++) {
        const metadata = metadatas[i];
        const rawDateTs = metadata?.date_ts;
        const dateTs = typeof rawDateTs === "number" && Number.isFinite(rawDateTs) ? rawDateTs : 0;

        const rawPageId = metadata?.page_id;
        const pageId =
            rawPageId !== undefined && rawPageId !== null && String(rawPageId).trim() !== ""
                ? String(rawPageId)
                : `missing-page-id-row-${i}`;

        if (!docsByDate.has(dateTs)) {
            docsByDate.set(dateTs, new Set<string>());
        }
        docsByDate.get(dateTs)!.add(pageId);
    }

    if (docsByDate.size === 0) {
        console.log("  - No metadata rows found");
        return;
    }

    console.log("  Document count grouped by metadata.date_ts (unique by page_id):");
    for (const [dateTs, pageIds] of [...docsByDate.entries()].sort((a, b) => a[0] - b[0])) {
        const dateLabel = dateTs > 0 ? new Date(dateTs).toISOString().split("T")[0] : "unknown";
        console.log(`    - ${dateTs} (${dateLabel}): ${pageIds.size} document(s)`);
    }
}

async function main()
{
    const client = new ChromaClient(
        {
            ...chromaClientParams(),
        });
    const collections = await client.listCollections();
    if (collections === undefined || collections === null)
    {
        console.log("Failed in listing collections");
    }
    if (collections.length == 0)
    {
        console.log("No collections found");
        return;
    }
    for (const col of collections)
    {
        console.log(`Collection Name ${col.name} with ${await col.count()} records`);
        const rows = await col.get();
        printChunkCountByDateTs(rows.metadatas ?? []);
        printDocumentCountByDateTs(rows.metadatas ?? []);
    }
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});