// We actually only have one query type now, but we keep the interface for future expansion.
export type QueryType = "broad_temporal" | "specific_fact";
export interface AnalyzedQuery {
    clean_query: string;
    query_embedding: number[];
    query_type: QueryType;
    filter: Record<string, any> | undefined;
}