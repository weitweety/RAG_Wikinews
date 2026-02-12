export type QueryType = "broad_temporal" | "specific_fact";
export interface AnalyzedQuery {
    clean_query: string;
    query_type: QueryType;
    filter: Record<string, any> | undefined;
}