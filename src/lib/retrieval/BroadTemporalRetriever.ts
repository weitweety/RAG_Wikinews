import { Chroma } from "@langchain/community/vectorstores/chroma";
import { Document, DocumentInterface } from "@langchain/core/documents";
import { AnalyzedQuery } from "../models/QueryTypes.js";
import { BaseRetriever } from "./BaseRetriever.js";
import { CONFIG } from "../config.js";

/** Cosine similarity between two vectors of equal length. */
function cosineSimilarity(a: number[], b: number[]): number {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
}

/**
 * MMR selection: picks k items from candidates that balance
 * relevance to the query with diversity among selected items.
 *
 * @param queryEmbedding  - the query vector
 * @param candidateEmbeddings - embeddings of each candidate
 * @param candidateScores - similarity scores of each candidate to the query
 * @param k       - number of results to return
 * @param lambda  - trade-off: 1 = pure relevance, 0 = pure diversity
 * @returns indices into the candidates array, in selection order
 */
function mmrSelect(
    queryEmbedding: number[],
    candidateEmbeddings: number[][],
    candidateScores: number[],
    k: number,
    lambda: number,
): number[] {
    const selected: number[] = [];
    const remaining = new Set(candidateEmbeddings.map((_, i) => i));

    while (selected.length < k && remaining.size > 0) {
        let bestIdx = -1;
        let bestScore = -Infinity;

        for (const idx of remaining) {
            // Relevance: similarity to query (use pre-computed score)
            const relevance = candidateScores[idx];

            // Diversity: max similarity to any already-selected candidate
            let maxSimToSelected = 0;
            for (const selIdx of selected) {
                const sim = cosineSimilarity(candidateEmbeddings[idx], candidateEmbeddings[selIdx]);
                if (sim > maxSimToSelected) {
                    maxSimToSelected = sim;
                }
            }

            const mmrScore = lambda * relevance - (1 - lambda) * maxSimToSelected;

            if (mmrScore > bestScore) {
                bestScore = mmrScore;
                bestIdx = idx;
            }
        }

        if (bestIdx === -1) break;
        selected.push(bestIdx);
        remaining.delete(bestIdx);
    }

    return selected;
}

export class BroadTemporalRetriever extends BaseRetriever {
    /** fetchK: how many candidates to retrieve before MMR re-ranking */
    private readonly fetchK: number;
    /** lambda: MMR trade-off (1 = pure relevance, 0 = pure diversity) */
    private readonly lambda: number;

    constructor(
        private readonly vectorStore: Chroma,
        options?: { fetchK?: number; lambda?: number },
    ) {
        super();
        this.fetchK = options?.fetchK ?? CONFIG.topK * 4;
        this.lambda = options?.lambda ?? 0.5;
    }

    async retrieve(query: AnalyzedQuery): Promise<DocumentInterface[]> {
        const queryEmbedding = query.query_embedding;
        const k = CONFIG.topK;

        // 1. Query Chroma for candidates with embeddings, documents, metadatas, and distances
        const collection = await this.vectorStore.ensureCollection();
        const queryResult = await collection.query({
            queryEmbeddings: [queryEmbedding],
            nResults: this.fetchK,
            where: query.filter,
            include: ["embeddings", "documents", "metadatas", "distances"],
        });

        const docs = queryResult.documents?.[0] ?? [];
        const metas = queryResult.metadatas?.[0] ?? [];
        const embeddings = (queryResult.embeddings?.[0] ?? []).filter((e): e is number[] => e !== null);
        const distances = queryResult.distances?.[0] ?? [];

        if (embeddings.length === 0) return [];
        if (embeddings.length <= k) {
            return docs.map((content, i) => new Document({
                pageContent: content ?? "",
                metadata: (metas[i] as Record<string, any>) ?? {},
            }));
        }

        // 2. Convert distances to similarity scores (smaller distance = more similar)
        const maxDist = Math.max(...distances.filter((d): d is number => d !== null)) || 1;
        const scores = distances.map((d) => d !== null ? 1 - d / maxDist : 0);

        // 3. Apply MMR selection
        const selectedIndices = mmrSelect(queryEmbedding, embeddings, scores, k, this.lambda);

        return selectedIndices.map((idx) => new Document({
            pageContent: docs[idx] ?? "",
            metadata: (metas[idx] as Record<string, any>) ?? {},
        }));
    }
}