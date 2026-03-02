/**
 * Parent-Child Chunks RAG
 *
 * 小チャンクで検索精度を確保しつつ、ヒットしたチャンクの親（大チャンク）を
 * LLM に渡すことで文脈の欠落を防ぐ手法。
 *
 * 検証結果:
 * - 親子チャンクの最大の強みは「原則＋例外」パターンの完全な取得。
 *   Q1（アルバイトの宿泊出張可否）では、親チャンク（132文字）に原則NGと
 *   例外条件（部長承認・上限8,000円）が両方含まれ、正確な回答を生成できた。
 * - 小チャンクのみ（50文字）では、例外条件の途中でチャンクが分断され、
 *   「原則として認められていません」のみの不完全な回答となった。
 *   2件目のヒットも「【宿泊費】」（5文字）という見出しだけで情報量がなかった。
 * - Q2（管理職の日当＋宿泊費）では、親子チャンクが【宿泊費】と【出張日当】の
 *   2つの親チャンクを取得し、複数セクションにまたがる情報を統合して回答できた。
 * - 一方、小チャンクのみではアルバイト関連の無関係チャンクと見出しだけがヒットし、
 *   管理職の金額情報が一切取得できず、LLM が「労働者保護法」等の架空情報で
 *   完全なハルシネーションを起こした。
 * - 親子チャンクは検索の粒度（小チャンク）とコンテキストの広さ（親チャンク）を
 *   分離する設計のため、チャンクサイズのジレンマを解消できる実用的な手法と言える。
 */
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { ParentDocumentRetriever } from "@langchain/classic/retrievers/parent_document";
import { InMemoryStore } from "@langchain/core/stores";
import { Document } from "@langchain/core/documents";
import { embeddings, ragChain } from "./shared.js";
import { expensePolicy } from "./documents/expense-policy.js";

// --- 質問 ---
// Q1: 原則NGだが例外条件がある質問 → 小チャンクだと例外が欠落しやすい
// Q2: 複数セクションの情報を組み合わせる質問
const questions = [
  "アルバイトスタッフが宿泊を伴う出張をすることはできますか？",
  "管理職が1泊2日で出張した場合、日当と宿泊費の合計上限はいくらですか？",
];

// --- スプリッター設定 ---
// parentSplitter: 大きいチャンク（セクション単位に近い）
// childSplitter: 小さいチャンク（検索精度を高める）

const parentSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 0,
});
const childSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 50,
  chunkOverlap: 0,
});

console.log("=".repeat(60));
console.log("親子チャンク（Parent-Child Chunks）比較検証");
console.log("=".repeat(60));

// =====================================================================
// A) 親子チャンク（ParentDocumentRetriever）
//    小チャンクで検索 → ヒットしたチャンクの親（大チャンク）を返す
// =====================================================================

console.log("\n▶ A) 親子チャンク（ParentDocumentRetriever）");
console.log("  parentSplitter: 200文字, childSplitter: 50文字");

const vectorStore = new MemoryVectorStore(embeddings);
const byteStore = new InMemoryStore<Uint8Array>();

const retriever = new ParentDocumentRetriever({
  vectorstore: vectorStore,
  byteStore,
  parentSplitter,
  childSplitter,
  childK: 5,
  parentK: 2,
});

const docs = [new Document({ pageContent: expensePolicy })];
await retriever.addDocuments(docs);

// 子チャンク（ベクトルストアに格納されたもの）を確認
const allChildDocs = await vectorStore.similaritySearch("", 100);
console.log(`\n  子チャンク数: ${allChildDocs.length}`);
allChildDocs.forEach((doc, i) => {
  console.log(
    `    [${i}] (${doc.pageContent.length}文字) ${doc.pageContent.replace(/\n/g, " ").trim()}`,
  );
});

// 親チャンクの確認
const parentChunks = await parentSplitter.createDocuments([expensePolicy]);
console.log(`\n  親チャンク数: ${parentChunks.length}`);
parentChunks.forEach((doc, i) => {
  const preview = doc.pageContent.replace(/\n/g, " ").trim().substring(0, 80);
  console.log(`    [${i}] (${doc.pageContent.length}文字) ${preview}…`);
});

// =====================================================================
// B) 小チャンクのみ（ベースライン）
//    50文字チャンクで検索 → そのまま小チャンクを返す
// =====================================================================

console.log("\n▶ B) 小チャンクのみ（ベースライン: 50文字）");

const smallChunks = await childSplitter.createDocuments([expensePolicy]);
const smallVectorStore = await MemoryVectorStore.fromDocuments(
  smallChunks,
  embeddings,
);
console.log(`  チャンク数: ${smallChunks.length}`);

// =====================================================================
// 比較実行
// =====================================================================

for (const question of questions) {
  console.log(`\n${"=".repeat(60)}`);
  console.log(`質問: ${question}`);
  console.log("=".repeat(60));

  // --- A) 親子チャンク ---
  console.log("\n--- A) 親子チャンク ---");
  const parentResults = await retriever.invoke(question);

  console.log(`検索結果 (${parentResults.length}件の親チャンク):`);
  parentResults.forEach((doc, i) => {
    console.log(
      `  [${i}] (${doc.pageContent.length}文字) ${doc.pageContent.replace(/\n/g, " ").trim()}`,
    );
  });

  const contextA = parentResults.map((doc) => doc.pageContent).join("\n");
  const answerA = await ragChain.invoke({ context: contextA, question });
  console.log(`回答: ${answerA}`);

  // --- B) 小チャンクのみ ---
  console.log("\n--- B) 小チャンクのみ（50文字）---");
  const smallResults = await smallVectorStore.similaritySearch(question, 2);

  console.log(`検索結果 (k=2):`);
  smallResults.forEach((doc, i) => {
    console.log(
      `  [${i}] (${doc.pageContent.length}文字) ${doc.pageContent.replace(/\n/g, " ").trim()}`,
    );
  });

  const contextB = smallResults.map((doc) => doc.pageContent).join("\n");
  const answerB = await ragChain.invoke({ context: contextB, question });
  console.log(`回答: ${answerB}`);
}
