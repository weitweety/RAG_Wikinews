import { QueryType } from "../models/QueryTypes.js";

export class PromptTemplates {
    static broadTemporalRetriever(
        context_variable_name: string,
        query_variable_name: string,
        options?: { includePerArticleInstructions?: boolean }
    ): string {
        const includePerArticleInstructions = options?.includePerArticleInstructions ?? true;
        let lines: string[] = [];
        if (includePerArticleInstructions) {
            lines = [
                "You are a helpful assistant. Answer the question using ONLY the provided context.",
                "You are given multiple news articles. Articles are separated by '----- Article Separator -----'.",
                "Each article represents ONE distinct news event.",
                "Your task:",
                "- For EACH article, extract exactly ONE main event.",
                "- Do NOT skip any article.",
                "- Do NOT extract multiple events from the same article.",
                "- Ignore background or explanatory details.",
                "Output:",
                "- One bullet per article for the extracted main event.",
                "- Only one bullet per article is allowed. Input articles are separated by '----- Article Separator -----'.",
                "- Output exactly {article_count} bullets, no more and no less.",
                "",
                "Do not make up information. If you don't know the answer, say you don't know.",
            ];
        } else {
            lines = [
                "You are a helpful assistant. Answer the question using ONLY the provided context.",
                "If the context does not contain the answer, say you don't know.",
                "Do not use prior knowledge. Always Quote the sentence(s) you used.",
                "Quote the exact sentence from the context that answers the question. Do not paraphrase unnecessarily.",
                "Do not make up information. If you don't know the answer, say you don't know.",
                "Do not use any other information than the context provided.",
            ];
        }
        lines.push(
            "",
            "<context>",
            `{${context_variable_name}}`,
            "</context>",
            "",
            `Question: {${query_variable_name}}`,
            "Answer:",
        );

        return lines.join("\n");
    }
    static getPromptTemplate(
        query_type: QueryType,
        context_variable_name: string,
        query_variable_name: string,
        options?: { includePerArticleInstructions?: boolean }
    ): string {
        switch (query_type) {
            case "broad_temporal":
                return PromptTemplates.broadTemporalRetriever(
                    context_variable_name,
                    query_variable_name,
                    options
                );
            default:
                throw new Error(`Unknown query type: ${query_type}`);
        }
    }
}