# Simple RAG (TypeScript) - LangChain + Ollama + Chroma

This repo is a small RAG pipeline over Wikinews data:

- Ingest articles from a specific UTC date
- Split into chunks and embed with Ollama
- Store vectors in Chroma
- Retrieve with either:
  - `BroadTemporalRetriever` (MMR reranking), or
  - built-in Chroma similarity search
- Answer with an Ollama chat model

## Data Source and License

This project uses content from [Wikinews](https://www.wikinews.org), licensed under Creative Commons Attribution:

- [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/)
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Content is processed for research/demonstration. Original authors retain their rights.

## Prerequisites

- Node.js
- Ollama running locally
- Chroma HTTP server reachable from this project

### Ollama setup

```bash
ollama pull all-minilm
ollama pull phi3:mini
ollama serve
```

### Chroma setup

Recommended:

```bash
docker compose up -d
```

This serves Chroma at `http://localhost:8000` and persists local data under `./chroma_data/`.

Alternative single-container command:

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/chroma_data:/chroma/chroma" chromadb/chroma
```

## Setup

Install dependencies:

```bash
npm install
```

Optional env file:

```bash
cp env.example .env
```

## Ingest data

`src/ingest.ts` fetches Wikinews pages for one UTC date, chunks them, and upserts to Chroma.

### Ingest for an explicit date

```bash
npm run ingest -- --date 2026-02-10
```

### Ingest using environment variable

```bash
DATE=2026-02-10 npm run ingest
```

### Ingest using "today" (UTC)

If no `--date` and no `DATE` are provided, ingest defaults to today in UTC.

```bash
npm run ingest
```

### Rebuild collection from scratch

```bash
RESET_COLLECTION=true npm run ingest -- --date 2026-02-10
```

Notes:

- Date format must be `YYYY-MM-DD`.
- If no pages are found for the date, ingest exits with an error.
- Chunks include metadata such as `source`, `title`, `page_id`, `date_ts`, and `chunkIndex`.

## Ask questions

`src/ask.ts`:

1. Parses question text to extract possible date/date-range.
2. Optionally builds a pre-retrieval date filter (`date_ts` range).
3. Retrieves chunks (MMR retriever or built-in Chroma similarity).
4. Merges chunks by `page_id` for context assembly.
5. Generates an answer and prints sources.

### Basic usage

```bash
npm run ask -- --question "What happened on June 3, 2005?"
```

You can also use environment variable input:

```bash
QUESTION="What happened on June 3, 2005?" npm run ask
```

### Retrieval toggles

Both toggles default to `on`.

#### Date pre-filter toggle

- Flag: `--date-filtering on|off`
- When `off`: no pre-date filtering is applied (even if parser finds dates).

```bash
npm run ask -- --question "What happened in early 2005?" --date-filtering off
```

#### Broad temporal retriever toggle

- Flag: `--broad-temporal-retriever on|off`
- When `on`: uses `BroadTemporalRetriever` (MMR reranking).
- When `off`: uses built-in Chroma `similaritySearch`.

```bash
npm run ask -- --question "What happened in early 2005?" --broad-temporal-retriever off
```

### Toggle combinations

```bash
# default behavior (both on)
npm run ask -- --question "What happened in 2005?"

# disable only date pre-filtering
npm run ask -- --question "What happened in 2005?" --date-filtering off

# disable only BroadTemporalRetriever
npm run ask -- --question "What happened in 2005?" --broad-temporal-retriever off

# disable both
npm run ask -- --question "What happened in 2005?" --date-filtering off --broad-temporal-retriever off
```

## Inspect Chroma collections

Use the helper script:

```bash
npm run inspect
```

It lists available collections and record counts.

## Configuration

All values are loaded from environment variables in `src/lib/config.ts`.

### Input and chunking

- `URLS_FILE` (default `urls.txt`)
- `URLS` (optional)
- `CHUNK_SIZE` (default `1000`)
- `CHUNK_OVERLAP` (default `200`)

### Ollama

- `OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `OLLAMA_EMBED_MODEL` (default `all-minilm`)
- `OLLAMA_CHAT_MODEL` (default `phi3:mini`)

### Chroma

- `CHROMA_URL` (default `http://localhost:8000`)
- `CHROMA_COLLECTION` (default `rag_urls`)
- `CHROMA_TENANT` (optional)
- `CHROMA_DATABASE` (optional)
- `RESET_COLLECTION` (default `false`)

### Retrieval

- `TOP_K` (default `4`)


