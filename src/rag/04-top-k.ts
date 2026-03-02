/**
 * Top-K 調整 RAG
 *
 * 取得件数(k)を 1, 2, 5, 10 で変化させ、3種類の質問に対する回答品質を比較検証。
 *
 * 検証結果:
 * - Q1（単一セクションで回答可能な質問）: k=1 でも正解チャンクが取得でき正確に回答。
 *   k を増やしても回答品質は変わらず、ノイズによる悪影響も見られなかった。
 * - Q2（複数セクションの情報を組み合わせる質問）: k=1,2 では管理職の宿泊費チャンクが
 *   取得されず、アルバイト向けの宿泊費(8,000円)を管理職に誤適用して不正解。
 *   k=5 で初めて管理職宿泊費(15,000円)のチャンクが取得され正しい情報が揃った。
 *   一方 k=10（全8チャンク）では無関係な情報が増え、LLM が再びアルバイトの
 *   宿泊費を誤適用する結果となり、コンテキスト過多による精度低下が確認された。
 * - Q3（特定情報のピンポイント質問）: k=1 では締め日を含むチャンクが取得されず
 *   「情報がありません」と回答。k=2 以上で正解チャンクが含まれ正確に回答できた。
 *   k=5,10 でも回答品質は維持され、単純なファクト質問にはノイズの影響が小さい。
 * - 総合: k が小さすぎると必要な情報を取りこぼし、大きすぎると無関係な情報が
 *   LLM の判断を混乱させる。本検証では k=2〜5 が最適範囲であり、特に複数セクション
 *   にまたがる質問では k=5 程度が必要。ただし k=10 のような過剰取得は逆効果になりうる。
 */
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
