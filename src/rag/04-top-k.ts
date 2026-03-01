import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { embeddings, ragChain } from "./shared.js";
import { expensePolicy } from "./documents/expense-policy.js";

// --- チャンク分割（中サイズで固定）---

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 100,
  chunkOverlap: 20,
});
const chunks = await splitter.createDocuments([expensePolicy]);

console.log("=".repeat(60));
console.log("取得件数(k)の調整 — Top-K 比較検証");
console.log("=".repeat(60));

console.log(`\nチャンク設定: chunkSize=100, chunkOverlap=20`);
console.log(`チャンク数: ${chunks.length}`);
chunks.forEach((chunk, i) => {
  console.log(
    `  [${i}] (${chunk.pageContent.length}文字) ${chunk.pageContent.replace(/\n/g, " ").trim()}`,
  );
});

// --- ベクトルストア構築 ---

const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);

// --- k の値 ---

const kValues = [1, 2, 5, 10];

// --- 質問 ---
// Q1: 単一セクションの情報で回答可能 → k が小さくても正確に答えられるはず
// Q2: 複数セクションの情報を組み合わせる必要がある → k を増やすと改善が期待できる
// Q3: 特定の情報だけが必要 → k を増やしすぎると無関係な情報が混入し悪化する可能性
const questions = [
  "アルバイトスタッフが宿泊を伴う出張をすることはできますか？",
  "管理職が1泊2日で出張した場合、日当と宿泊費の合計上限はいくらですか？",
  "経費精算の締め日はいつですか？",
];

// --- 比較実行 ---

for (const question of questions) {
  console.log(`\n${"=".repeat(60)}`);
  console.log(`質問: ${question}`);
  console.log("=".repeat(60));

  for (const k of kValues) {
    const searchK = Math.min(k, chunks.length);
    const relevantDocs = await vectorStore.similaritySearch(question, searchK);

    console.log(`\n--- k=${k} (取得: ${relevantDocs.length}件) ---`);
    console.log("検索結果:");
    relevantDocs.forEach((doc, i) => {
      console.log(
        `  [${i}] (${doc.pageContent.length}文字) ${doc.pageContent.replace(/\n/g, " ").trim()}`,
      );
    });

    const context = relevantDocs.map((doc) => doc.pageContent).join("\n");
    const answer = await ragChain.invoke({ context, question });
    console.log(`回答: ${answer}`);
  }
}
