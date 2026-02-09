import { Chroma } from "@langchain/community/vectorstores/chroma";
import { Document } from "@langchain/core/documents";
import { OllamaEmbeddings } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { ChromaClient } from "chromadb";

import { CONFIG } from "./lib/config.js";
import { stableChunkId } from "./lib/ids.js";

function chromaClientParams(): { tenant?: string; database?: string } {
  return {
    ...(CONFIG.chromaTenant ? { tenant: CONFIG.chromaTenant } : {}),
    ...(CONFIG.chromaDatabase ? { database: CONFIG.chromaDatabase } : {}),
  };
}

async function maybeResetCollection(): Promise<void> {
  if (!CONFIG.resetCollection) return;

  const client = new ChromaClient({
    // This library uses `path` for full base URL. (Deprecated in type defs, but used by LangChain's Chroma integration too.)
    path: CONFIG.chromaUrl,
    ...chromaClientParams(),
  });

  try {
    await client.deleteCollection({ name: CONFIG.chromaCollection });
    console.log(`Deleted existing collection "${CONFIG.chromaCollection}"`);
  } catch (err: unknown) {
    // If it doesn't exist, that's fine.
    console.warn(
      `RESET_COLLECTION was set, but deleteCollection failed (often means it didn't exist yet): ${String(err)}`
    );
  }
}

async function loadDocsFromPageIds(page_ids: number[], targetDate?: Date): Promise<Document[]> {
  const all: Document[] = [];
  for (const page_id of page_ids) {
    const url = `https://en.wikinews.org/w/api.php?action=query&pageids=${page_id}&prop=extracts&explaintext=true&format=json`;
    const response = await fetch(url);
    const data = await response.json();
    const page = data.query.pages[page_id];
    console.log(page);
    if (page.extract) {
      all.push(new Document({
        pageContent: page.extract,
        metadata: {
          source: "Wikinews",
          page_id: page_id,
          title: page.title || '',
          // Store date as Unix timestamp (milliseconds) for ChromaDB numeric filtering
          date_ts: targetDate ? targetDate.getTime() : 0,
        },
      }));
    }
  }
  return all;
}

async function loadLatestNews(): Promise<Document[]> {
  // Use UTC to avoid timezone issues with filtering
  const targetDate = new Date(Date.UTC(2005, 5, 3));
  const targetDateString = targetDate.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric', timeZone: 'UTC' }).replace(/ /g, '_');
  const url = `https://en.wikinews.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:${targetDateString}&cmnamespace=0&cmlimit=50&format=json`;
  const response = await fetch(url);
  const data = await response.json();
  const page_ids = data.query.categorymembers.map((page: any) => page?.pageid).filter((id: number) => id !== undefined);
  const pages = await loadDocsFromPageIds(page_ids, targetDate);
  return pages;
}

async function main(): Promise<void> {
  const pages = await loadLatestNews();
  console.log(pages);

  if (pages.length === 0) {
    console.error("No documents loaded. Nothing to ingest.");
    process.exit(1);
  }

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CONFIG.chunkSize,
    chunkOverlap: CONFIG.chunkOverlap,
  });

  const chunks = await splitter.splitDocuments(pages);
  for (let i = 0; i < chunks.length; i++) {
    chunks[i].metadata = {
      ...chunks[i].metadata,
      chunkIndex: i,
    };
  }

  console.log(
    `Chunked into ${chunks.length} chunk(s) (chunkSize=${CONFIG.chunkSize}, overlap=${CONFIG.chunkOverlap})`
  );

  const embeddings = new OllamaEmbeddings({
    model: CONFIG.ollamaEmbedModel,
    baseUrl: CONFIG.ollamaBaseUrl,
  });

  const vectorStore = new Chroma(embeddings, {
    collectionName: CONFIG.chromaCollection,
    url: CONFIG.chromaUrl,
    clientParams: chromaClientParams(),
  });

  const ids = chunks.map((d, index) =>
    stableChunkId(String(d.metadata?.source ?? "unknown"), d.pageContent, index)
  );

  await maybeResetCollection();
  const inserted = await vectorStore.addDocuments(chunks, { ids });

  console.log(
    `Upserted ${inserted.length} chunk(s) into Chroma collection "${CONFIG.chromaCollection}" at ${CONFIG.chromaUrl}`
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});


