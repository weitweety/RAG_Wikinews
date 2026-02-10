import dotenv from "dotenv";

dotenv.config();

function envString(key: string, defaultValue: string): string {
  const v = process.env[key];
  return v === undefined || v.trim() === "" ? defaultValue : v;
}

function envOptionalString(key: string): string | undefined {
  const v = process.env[key];
  return v === undefined || v.trim() === "" ? undefined : v;
}

function envInt(key: string, defaultValue: number): number {
  const raw = process.env[key];
  if (raw === undefined || raw.trim() === "") return defaultValue;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) ? n : defaultValue;
}

function envBool(key: string, defaultValue = false): boolean {
  const raw = process.env[key];
  if (raw === undefined || raw.trim() === "") return defaultValue;
  return ["1", "true", "yes", "y", "on"].includes(raw.trim().toLowerCase());
}

export const CONFIG = {
  // Inputs
  urlsFile: envString("URLS_FILE", "urls.txt"),
  urlsEnv: envOptionalString("URLS"),

  // Chunking
  chunkSize: envInt("CHUNK_SIZE", 1000),
  chunkOverlap: envInt("CHUNK_OVERLAP", 200),

  // Ollama
  ollamaBaseUrl: envString("OLLAMA_BASE_URL", "http://localhost:11434"),
  ollamaEmbedModel: envString("OLLAMA_EMBED_MODEL", "all-minilm"),
  ollamaChatModel: envString("OLLAMA_CHAT_MODEL", "phi3:mini"),

  // Chroma (HTTP server)
  chromaUrl: envString("CHROMA_URL", "http://localhost:8000"),
  chromaCollection: envString("CHROMA_COLLECTION", "rag_urls"),
  chromaTenant: envOptionalString("CHROMA_TENANT"),
  chromaDatabase: envOptionalString("CHROMA_DATABASE"),
  resetCollection: envBool("RESET_COLLECTION", false),

  // Retrieval
  topK: envInt("TOP_K", 4)
};


