import { ChromaClient } from "chromadb";
import { CONFIG } from "./lib/config.js";

function chromaClientParams(): { tenant?: string; database?: string } {
    return {
      ...(CONFIG.chromaTenant ? { tenant: CONFIG.chromaTenant } : {}),
      ...(CONFIG.chromaDatabase ? { database: CONFIG.chromaDatabase } : {}),
    };
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
        // console.log(await col.get());
    }
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});