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

const WIKINEWS_API_URL = "https://en.wikinews.org/w/api.php";

async function fetchJsonFromWikinews(params: Record<string, string>): Promise<any> {
  const url = new URL(WIKINEWS_API_URL);
  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, value);
  }

  const response = await fetch(url.toString(), {
    headers: {
      Accept: "application/json",
      "User-Agent": "RAG_Wikinews/1.0 (local development script)",
    },
  });

  const contentType = response.headers.get("content-type") ?? "";
  const rawBody = await response.text();

  if (!response.ok) {
    const snippet = rawBody.slice(0, 200).replace(/\s+/g, " ");
    throw new Error(
      `Wikinews API request failed (${response.status} ${response.statusText}) for ${url.toString()}. Response preview: ${snippet}`
    );
  }

  if (!contentType.toLowerCase().includes("application/json")) {
    const snippet = rawBody.slice(0, 200).replace(/\s+/g, " ");
    throw new Error(
      `Wikinews API returned non-JSON content-type "${contentType}" for ${url.toString()}. Response preview: ${snippet}`
    );
  }

  try {
    return JSON.parse(rawBody);
  } catch {
    const snippet = rawBody.slice(0, 200).replace(/\s+/g, " ");
    throw new Error(
      `Failed to parse Wikinews API JSON for ${url.toString()}. Response preview: ${snippet}`
    );
  }
}

async function maybeResetCollection(): Promise<void> {
  console.log(`Resetting collection: ${CONFIG.resetCollection}`);
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
    const data = await fetchJsonFromWikinews({
      action: "query",
      pageids: String(page_id),
      prop: "extracts",
      explaintext: "true",
      format: "json",
    });
    const page = data?.query?.pages?.[page_id];
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

async function loadNewsForDate(date: Date): Promise<Document[]> {
  // Use UTC to avoid timezone issues with filtering
  const targetDateString = date.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric', timeZone: 'UTC' }).replace(/ /g, '_');
  const data = await fetchJsonFromWikinews({
    action: "query",
    list: "categorymembers",
    cmtitle: `Category:${targetDateString}`,
    cmnamespace: "0",
    cmlimit: "50",
    format: "json",
  });
  const page_ids = (data?.query?.categorymembers ?? [])
    .map((page: any) => page?.pageid)
    .filter((id: number) => id !== undefined);
  return await loadDocsFromPageIds(page_ids, date);
}

const isValidDate = (dateString: string): boolean => {
  // YYYY-MM-DD
  const regex = /^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$/;
  return regex.test(dateString);
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const dIdx = argv.indexOf("--date");
  const dateString =
    (dIdx !== -1 && dIdx + 1 < argv.length ? argv[dIdx + 1] : "")?.trim() ||
    process.env.DATE?.trim() ||
    "";

  if (!dateString) {
    console.log(`No date provided. Using today's date in UTC timezone: ${new Date().toISOString().split('T')[0]}.`);
  }
  else if (!isValidDate(dateString)) {
    console.error("Invalid date format. Please use YYYY-MM-DD in UTC timezone. Example: 2026-02-10");
    process.exit(1);
  } else {
    console.log(`Using date: ${dateString} in UTC timezone.`);
  }

  const date = dateString ? new Date(dateString) : new Date();
  const pages = await loadNewsForDate(date);

  if (pages.length === 0) {
    console.error(`No documents loaded for the given date: ${dateString}. Nothing to ingest.`);
    process.exit(1);
  }
  else {
    console.log(`Loaded ${pages.length} pages for the given date: ${dateString}.`);
  }

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CONFIG.chunkSize,
    chunkOverlap: CONFIG.chunkOverlap,
    separators: ["\n\n", "\n", ".", "!", "?", " ", ""]
  });

  const chunks = await splitter.splitDocuments(pages);
  for (let i = 0; i < chunks.length; i++) {
    chunks[i].metadata = {
      ...chunks[i].metadata,
      chunkIndex: i,
    };
    console.log(`Chunk ${i}: ${chunks[i].pageContent}`);
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


