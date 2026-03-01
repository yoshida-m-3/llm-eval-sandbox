import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

// --- 1. 準備: ドキュメントをベクトル化して保存 ---

// サンプルドキュメント（LLMが学習していない独自情報を想定）
const documents = `
llm-eval-sandbox は LangChain エコシステムを学習するためのプロジェクトです。
技術スタックは TypeScript と pnpm を使用しています。
LLMプロバイダーとして Ollama（ローカル）を使用し、将来的に Amazon Bedrock に移行予定です。
マイルストーンは4つあります。
Milestone 1 は LangChain と LangSmith の学習です。
Milestone 2 は LangGraph によるマルチエージェントワークフローの構築です。
Milestone 3 は Hono を使った API サーバーの構築です。
Milestone 4 は LangFlow によるビジュアルフロー構築です。
`;

// テキストを小さなチャンクに分割（大きなドキュメントの場合に必要）
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 100,
  chunkOverlap: 20,
});
const chunks = await splitter.createDocuments([documents]);

console.log(`=== ドキュメント分割: ${chunks.length} チャンク ===`);
chunks.forEach((chunk, i) => {
  console.log(`  [${i}] ${chunk.pageContent.trim().substring(0, 50)}...`);
});

// Embedding モデルでベクトル化し、インメモリのベクトルDBに保存
const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" });
const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);

// --- 2. 実行: 質問に関連するドキュメントを検索 → LLMに渡して回答 ---

const question = "このプロジェクトのMilestone 3は何をしますか？";

// ベクトル類似検索で関連チャンクを取得
const relevantDocs = await vectorStore.similaritySearch(question, 2);

console.log(`\n=== 検索結果: ${relevantDocs.length} 件 ===`);
relevantDocs.forEach((doc, i) => {
  console.log(`  [${i}] ${doc.pageContent.trim()}`);
});

// 検索結果をコンテキストとしてプロンプトに注入
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "以下のコンテキスト情報のみを使って質問に回答してください。\n\nコンテキスト:\n{context}",
  ],
  ["human", "{question}"],
]);

const model = new ChatOllama({ model: "llama3.2:3b", temperature: 0 });
const chain = prompt.pipe(model).pipe(new StringOutputParser());

const context = relevantDocs.map((doc) => doc.pageContent).join("\n");
const answer = await chain.invoke({ context, question });

console.log(`\n=== 質問: ${question} ===`);
console.log(`回答: ${answer}`);
