import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

// --- モデル設定 ---

export const llm = new ChatOllama({ model: "llama3.2:3b", temperature: 0 });

export const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" });

// --- RAG プロンプト ---

export const ragPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "以下のコンテキスト情報のみを使って質問に回答してください。\n\nコンテキスト:\n{context}",
  ],
  ["human", "{question}"],
]);

// --- RAG チェーン ---

export const ragChain = ragPrompt.pipe(llm).pipe(new StringOutputParser());
