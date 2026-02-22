import { QueryType } from "../models/QueryTypes.js";

export class PromptTemplates {
    static broadTemporalRetriever(context_variable_name: string, query_variable_name: string): string {
        return [
            "You are a helpful assistant. Answer the question using ONLY the provided context.",
            "If the context does not contain the answer, say you don't know.",
            "Do not use prior knowledge. Always Quote the sentence(s) you used.",
            "Quote the exact sentence from the context that answers the question. Do not paraphrase unnecessarily.",
            "Do not make up information. If you don't know the answer, say you don't know.",
            "Do not use any other information than the context provided.",
            "You are given multiple news articles.",
            "Each article represents ONE distinct news event.",
            "Your task:",
            "- For EACH article, extract exactly ONE main event.",
            "- Do NOT skip any article.",
            "- Do NOT extract multiple events from the same article.",
            "- Ignore background or explanatory details.",
            "Output:",
            "- One bullet per article",
            "- Each bullet must include the article title",
            "",
            "<context>",
            `{${context_variable_name}}`,
            "</context>",
            "",
            `Question: {${query_variable_name}}`,
            "Answer:",
        ].join("\n");
    }
    static getPromptTemplate(query_type: QueryType, context_variable_name: string, query_variable_name: string): string {
        switch (query_type) {
            case "broad_temporal":
                return PromptTemplates.broadTemporalRetriever(context_variable_name, query_variable_name);
            default:
                throw new Error(`Unknown query type: ${query_type}`);
        }
    }
}