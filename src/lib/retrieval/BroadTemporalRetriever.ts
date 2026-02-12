import { VectorStore } from "@langchain/core/vectorstores";
import { DocumentInterface } from "@langchain/core/documents";
import { AnalyzedQuery } from "../models/QueryTypes.js";
import { BaseRetriever } from "./BaseRetriever.js";
import { CONFIG } from "../config.js";

export class BroadTemporalRetriever extends BaseRetriever {
    constructor(private readonly vectorStore: VectorStore) {
        super();
    }

    async retrieve(query: AnalyzedQuery): Promise<DocumentInterface[]> {
        if (!this.vectorStore.maxMarginalRelevanceSearch) {
            throw new Error("Vector store does not support maxMarginalRelevanceSearch");
        }
        const results = await this.vectorStore.maxMarginalRelevanceSearch(
            query.clean_query,
            {
                k: CONFIG.topK,
                filter: query.filter,
            },
            undefined,
        );
        return results
    }
}