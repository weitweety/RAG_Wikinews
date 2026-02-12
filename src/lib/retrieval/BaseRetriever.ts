import { DocumentInterface } from "@langchain/core/documents";
import { AnalyzedQuery } from "../models/QueryTypes.js";

export abstract class BaseRetriever {
    abstract retrieve(query: AnalyzedQuery): Promise<DocumentInterface[]>;
}