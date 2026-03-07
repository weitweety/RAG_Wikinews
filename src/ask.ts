import { Chroma } from "@langchain/community/vectorstores/chroma";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";

import { CONFIG } from "./lib/config.js";
import { BroadTemporalRetriever } from "./lib/retrieval/BroadTemporalRetriever.js";
import type { AnalyzedQuery } from "./lib/models/QueryTypes.js";
import { PromptTemplates } from "./lib/prompt/PromptTemplates.js";

interface ParsedQuery {
  clean_query: string;
  date: string | null;
  date_range: { start: string; end: string } | null;
}

const QUERY_PARSER_PROMPT = `You are a query parser.

From the user query:
- Extract any referenced date or date range.
- Remove the date information from the query.
- Do not answer the question.

Output JSON with this exact schema:
{{
  "clean_query": string,
  "date": string | null,
  "date_range": {{ "start": string, "end": string }} | null
}}
Dates must be in ISO format (YYYY-MM-DD).

User query: {query}

Output only valid JSON, no additional text.`;

async function parseQuery(query: string, llm: ChatOllama): Promise<ParsedQuery> {
  const prompt = ChatPromptTemplate.fromTemplate(QUERY_PARSER_PROMPT);
  const chain = prompt.pipe(llm);
  const result = await chain.invoke({ query });

  const content = typeof result.content === "string" ? result.content : String(result.content);

  // Extract JSON from the response (handle potential markdown code blocks)
  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    console.warn("Query parser did not return valid JSON, using original query");
    return { clean_query: query, date: null, date_range: null };
  }

  try {
    const parsed = JSON.parse(jsonMatch[0]) as ParsedQuery;
    return {
      clean_query: parsed.clean_query || query,
      date: parsed.date || null,
      date_range: parsed.date_range || null,
    };
  } catch {
    console.warn("Failed to parse query parser JSON, using original query");
    return { clean_query: query, date: null, date_range: null };
  }
}

function buildDateFilter(parsedQuery: ParsedQuery): Record<string, any> | undefined {
  if (parsedQuery.date) {
    // Convert date string to timestamp range for the day
    const startOfDay = new Date(`${parsedQuery.date}T00:00:00Z`).getTime();
    const endOfDay = new Date(`${parsedQuery.date}T23:59:59Z`).getTime();
    // Filter for exact date match using $and for range on same field
    return {
      $and: [
        { date_ts: { $gte: startOfDay } },
        { date_ts: { $lte: endOfDay } },
      ],
    };
  }

  if (parsedQuery.date_range) {
    // Convert date range to timestamps
    const startTs = new Date(`${parsedQuery.date_range.start}T00:00:00Z`).getTime();
    const endTs = new Date(`${parsedQuery.date_range.end}T23:59:59Z`).getTime();
    // Filter for date range using $and
    return {
      $and: [
        { date_ts: { $gte: startTs } },
        { date_ts: { $lte: endTs } },
      ],
    };
  }

  return undefined;
}

function chromaClientParams(): { tenant?: string; database?: string } {
  return {
    ...(CONFIG.chromaTenant ? { tenant: CONFIG.chromaTenant } : {}),
    ...(CONFIG.chromaDatabase ? { database: CONFIG.chromaDatabase } : {}),
  };
}

function mergeChunksByPageId(docs: Document[]): Document[] {
  type ChunkEntry = {
    text: string;
    chunkIndex: number | null;
    originalOrder: number;
  };

  type Group = {
    metadata: Record<string, unknown>;
    chunks: ChunkEntry[];
  };

  const groups = new Map<string, Group>();
  let fallbackCounter = 0;

  for (let i = 0; i < docs.length; i++) {
    const doc = docs[i];
    const pageId = doc.metadata?.page_id;
    const groupKey = pageId !== undefined && pageId !== null && String(pageId).trim() !== ""
      ? String(pageId)
      : `unknown-${fallbackCounter++}`;

    if (!groups.has(groupKey)) {
      groups.set(groupKey, {
        metadata: { ...doc.metadata },
        chunks: [],
      });
    }

    const group = groups.get(groupKey)!;
    const rawChunkIndex = doc.metadata?.chunkIndex;
    const chunkIndex =
      typeof rawChunkIndex === "number" && Number.isFinite(rawChunkIndex)
        ? rawChunkIndex
        : null;

    group.chunks.push({
      text: doc.pageContent,
      chunkIndex,
      originalOrder: i,
    });
  }

  const mergedDocs: Document[] = [];
  for (const [, group] of groups) {
    group.chunks.sort((a, b) => {
      if (a.chunkIndex !== null && b.chunkIndex !== null) {
        return a.chunkIndex - b.chunkIndex;
      }
      if (a.chunkIndex !== null) return -1;
      if (b.chunkIndex !== null) return 1;
      return a.originalOrder - b.originalOrder;
    });

    const mergedContent = group.chunks
      .map((chunk) => chunk.text.trim())
      .filter((chunk) => chunk.length > 0)
      .join("\n\n");

    mergedDocs.push(
      new Document({
        pageContent: mergedContent,
        metadata: group.metadata,
      })
    );
  }

  return mergedDocs;
}

// DONE: 1. Improve chunking to respect the sentence structures.
// DONE: 2. Try to implement more retrieval logics. E.g. MMR which should consider fetching chunks from different documents.
// TODO: 3. Combine chunks sharing same document. Structure the prompt to ensure LLM process all documents.
// https://chatgpt.com/c/698c43a4-51ec-832e-ba0a-5bf1a9206024

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const qIdx = argv.indexOf("--question");
  const question =
    (qIdx !== -1 && qIdx + 1 < argv.length ? argv[qIdx + 1] : "")?.trim() ||
    process.env.QUESTION?.trim() ||
    "";

  if (!question) {
    console.error(
      [
        "No question provided.",
        "",
        'Usage: npm run ask -- --question "Your question here"',
        'Or: QUESTION="Your question" npm run ask',
      ].join("\n")
    );
    process.exit(1);
  }

  const embeddings = new OllamaEmbeddings({
    model: CONFIG.ollamaEmbedModel,
    baseUrl: CONFIG.ollamaBaseUrl,
  });

  const vectorStore = new Chroma(embeddings, {
    collectionName: CONFIG.chromaCollection,
    url: CONFIG.chromaUrl,
    clientParams: chromaClientParams(),
  });

  // Query parser LLM to extract date information
  const parserLlm = new ChatOllama({
    model: CONFIG.ollamaChatModel,
    baseUrl: CONFIG.ollamaBaseUrl,
    temperature: 0,
  });

  // Parse the query to extract date info and clean query
  console.log("Parsing query...");
  const parsedQuery = await parseQuery(question, parserLlm);
  console.log("Parsed query:", JSON.stringify(parsedQuery, null, 2));

  // Build date filter if date was extracted
  const dateFilter = buildDateFilter(parsedQuery);

  // Log filter details for debugging
  if (dateFilter) {
    console.log("Date filter applied:", JSON.stringify(dateFilter, null, 2));
    // Also log human-readable dates
    if (parsedQuery.date) {
      const startOfDay = new Date(`${parsedQuery.date}T00:00:00Z`);
      const endOfDay = new Date(`${parsedQuery.date}T23:59:59Z`);
      console.log(`Filtering for date: ${parsedQuery.date} (${startOfDay.getTime()} to ${endOfDay.getTime()})`);
    }
  } else {
    console.log("No date filter applied");
  }

  // TODO: the retriever and prompt template should be determined by the real query_type.
  // Build analyzed query for the retriever
  const analyzedQuery: AnalyzedQuery = {
    clean_query: parsedQuery.clean_query,
    query_embedding: await embeddings.embedQuery(parsedQuery.clean_query),
    query_type: "broad_temporal",
    filter: dateFilter,
  };

  // Retrieve documents using BroadTemporalRetriever (MMR)
  const broadRetriever = new BroadTemporalRetriever(vectorStore);
  console.log(`\nRetrieving with BroadTemporalRetriever (MMR): "${analyzedQuery.clean_query}"`);
  const retrievedDocs = await broadRetriever.retrieve(analyzedQuery);
  console.log(`Retrieved ${retrievedDocs.length} chunk(s)`);
  if (retrievedDocs.length > 0) {
    console.log("Retrieved documents metadata:");
    for (const doc of retrievedDocs) {
      const meta = doc.metadata;
      const dateTs = meta.date_ts;
      const dateStr = dateTs ? new Date(dateTs).toISOString() : "N/A";
      console.log(`  - title: ${meta.title || "N/A"}, date_ts: ${dateTs} (${dateStr}), source: ${meta.source || "N/A"}`);
    }
  }

  const mergedDocs = mergeChunksByPageId(retrievedDocs);
  console.log(`Merged into ${mergedDocs.length} document(s) by page_id`);

  // Format retrieved documents as context
  const context = mergedDocs
    .map((doc) => doc.pageContent)
    .join("\n\n---\n\n");

  // Main chat LLM for answering questions
  const llm = new ChatOllama({
    model: CONFIG.ollamaChatModel,
    baseUrl: CONFIG.ollamaBaseUrl,
    temperature: 0,
  });

  const prompt = ChatPromptTemplate.fromTemplate(PromptTemplates.getPromptTemplate(analyzedQuery.query_type, "context", "input"));

  const chain = prompt.pipe(llm).pipe(new StringOutputParser());

  const answer = await chain.invoke({
    context,
    input: parsedQuery.clean_query,
  });

  console.log(answer.trim());

  if (retrievedDocs.length > 0) {
    console.log("\nSources:");
    const seen = new Set<string>();
    for (const doc of retrievedDocs) {
      const source = String(doc.metadata?.source ?? "").trim();
      if (!source || seen.has(source)) continue;
      seen.add(source);
      console.log(`- ${source}`);
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});


