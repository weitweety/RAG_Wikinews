import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

import { CONFIG } from "./lib/config.js";

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

async function main(): Promise<void> {
  const question =
    process.argv.slice(2).join(" ").trim() ??
    process.env.QUESTION?.trim() ??
    "";

  if (!question) {
    console.error(
      [
        "No question provided.",
        "",
        'Usage: npm run ask -- "Your question here"',
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

  // Create retriever with optional date filter
  const retriever = vectorStore.asRetriever({
    k: CONFIG.topK,
    filter: dateFilter,
  });

  // Test the retrieval directly to see what's returned
  console.log(`\nTesting retrieval with query: "${parsedQuery.clean_query}"`);
  const retrievedDocs = await retriever.invoke(parsedQuery.clean_query);
  console.log(`Retrieved ${retrievedDocs.length} document(s)`);
  if (retrievedDocs.length > 0) {
    console.log("Retrieved documents metadata:");
    for (const doc of retrievedDocs) {
      const meta = doc.metadata;
      const dateTs = meta.date_ts;
      const dateStr = dateTs ? new Date(dateTs).toISOString() : "N/A";
      console.log(`  - title: ${meta.title || "N/A"}, date_ts: ${dateTs} (${dateStr}), source: ${meta.source || "N/A"}`);
    }
  }

  // Main chat LLM for answering questions
  const llm = new ChatOllama({
    model: CONFIG.ollamaChatModel,
    baseUrl: CONFIG.ollamaBaseUrl,
    temperature: 0,
  });

  const prompt = ChatPromptTemplate.fromTemplate(
    [
      "You are a helpful assistant. Answer the question using ONLY the provided context.",
      "If the context does not contain the answer, say you don't know.",
      "",
      "<context>",
      "{context}",
      "</context>",
      "",
      "Question: {input}",
      "Answer:",
    ].join("\n")
  );

  const combineDocsChain = await createStuffDocumentsChain({
    llm,
    prompt,
  });

  const chain = await createRetrievalChain({
    retriever,
    combineDocsChain,
  });

  // Use the clean query (with date info removed) for retrieval
  const result = await chain.invoke({ input: parsedQuery.clean_query });

  // result.answer should be string (default output parser)
  console.log(String(result.answer).trim());

  const ctx = Array.isArray(result.context) ? result.context : [];
  if (ctx.length > 0) {
    console.log("\nSources:");
    const seen = new Set<string>();
    for (const doc of ctx) {
      const source = String((doc as any)?.metadata?.source ?? "").trim();
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


