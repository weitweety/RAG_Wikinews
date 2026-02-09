# Simple RAG (TypeScript) — LangChain + Ollama + Chroma

This repo contains a minimal RAG pipeline in TypeScript. Cursor is used to assist in development.

- **Load news from specifed date from Wikinews** (URL list)
- **Chunk with LangChain**
- **Embed with Ollama** using **`all-minilm`**
- **Store/retrieve with Chroma** (vector DB)
- **Answer with Ollama** using **`phi3:mini`**

## Data Sources

This project uses text content from **Wikinews**
(https://www.wikinews.org), which is licensed under the
**Creative Commons Attribution License (CC BY 2.5 / CC BY 4.0)**.

The content has been processed, chunked, and embedded for
research and demonstration purposes. Original authors retain
their copyrights.

License text:
- https://creativecommons.org/licenses/by/2.5/
- https://creativecommons.org/licenses/by/4.0/

## Prereqs

- Node.js (you have it)
- Ollama running locally
- A Chroma **HTTP server** running (LangChain connects via `CHROMA_URL`)

### Ollama

Make sure the models exist and Ollama is running:

```bash
ollama pull all-minilm
ollama pull phi3:mini
ollama serve
```

### Chroma server

The `chromadb` npm package is installed, but the bundled CLI (`npx chroma ...`) may fail on some Linux distros due to a `GLIBCXX` runtime mismatch.

The simplest reliable option is Docker (recommended: `docker compose`):

```bash
docker compose up -d
```

Chroma will be available at `http://localhost:8000` and persist data under `./chroma_data/`.

If you prefer running a single container directly:

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/chroma_data:/chroma/chroma" chromadb/chroma
```

## Setup

Install deps:

```bash
npm install
```

Optional: create env file:

```bash
cp env.example .env
```

To wipe and rebuild the collection:

```bash
RESET_COLLECTION=true npm run ingest
```

## Ask questions

```bash
npm run ask -- "What happened on June 3, 2005?"
```

The script prints an answer plus a short **Sources** list.

## Configuration

All config can be set via env vars:

- `OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `OLLAMA_EMBED_MODEL` (default `all-minilm`)
- `OLLAMA_CHAT_MODEL` (default `opencoder`)
- `CHROMA_URL` (default `http://localhost:8000`)
- `CHROMA_COLLECTION` (default `rag_urls`)
- `CHUNK_SIZE` / `CHUNK_OVERLAP`
- `TOP_K`
- `RESET_COLLECTION`


