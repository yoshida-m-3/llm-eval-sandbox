import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { embeddings, ragChain } from "./shared.js";
import { expensePolicy } from "./documents/expense-policy.js";

// --- チャンクサイズ・オーバーラップ調整の比較 ---

const configs = [
  { label: "小 (50/10)", chunkSize: 50, chunkOverlap: 10 },
  { label: "中 (100/20)", chunkSize: 100, chunkOverlap: 20 },
  { label: "大 (200/40)", chunkSize: 200, chunkOverlap: 40 },
];

// Q1: 原則NGだが例外条件がある質問 → 小チャンクだと例外が欠落しやすい
// Q2: 複数セクションの情報を組み合わせる質問 → 小チャンクだと片方しか取得できない
const questions = [
  "アルバイトスタッフが宿泊を伴う出張をすることはできますか？",
  "管理職が1泊2日で出張した場合、日当と宿泊費の合計上限はいくらですか？",
];

for (const config of configs) {
  console.log(`\n${"=".repeat(60)}`);
  console.log(`チャンク設定: ${config.label}`);
  console.log(
    `  chunkSize=${config.chunkSize}, chunkOverlap=${config.chunkOverlap}`,
  );
  console.log("=".repeat(60));

  // 1. チャンク分割
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: config.chunkSize,
    chunkOverlap: config.chunkOverlap,
  });
  const chunks = await splitter.createDocuments([expensePolicy]);

  console.log(`\n--- 分割結果: ${chunks.length} チャンク ---`);
  chunks.forEach((chunk, i) => {
    console.log(
      `  [${i}] (${chunk.pageContent.length}文字) ${chunk.pageContent.trim()}`,
    );
  });

  for (const question of questions) {
    // 2. ベクトルストアに格納 & 検索
    const vectorStore = await MemoryVectorStore.fromDocuments(
      chunks,
      embeddings,
    );
    const relevantDocs = await vectorStore.similaritySearch(question, 2);

    console.log(`\n--- 質問: ${question} ---`);
    console.log(`検索結果 (k=2):`);
    relevantDocs.forEach((doc, i) => {
      console.log(`  [${i}] ${doc.pageContent.trim()}`);
    });

    // 3. RAG で回答生成
    const context = relevantDocs.map((doc) => doc.pageContent).join("\n");
    const answer = await ragChain.invoke({ context, question });

    console.log(`回答: ${answer}`);
  }
}
