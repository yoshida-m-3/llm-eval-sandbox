/**
 * チャンクサイズ・オーバーラップ調整の比較
 *
 * 小 (50/10)・中 (100/20)・大 (200/40) の 3 設定で、経費精算規定に対する
 * 2 つの質問（例外条件を含む質問 / 複数セクションの情報を組み合わせる質問）を検証。
 *
 * 検証結果:
 * - 小チャンクでは文が途中で切れ、例外条件が欠落する。
 *   例: アルバイト宿泊出張の質問で「ただし、事前に部長の承認を得た場」で切断され、
 *   例外の上限額（8,000 円）が失われ「原則認められていません」のみの回答になった。
 * - 小チャンクでは検索精度も低下し、管理職の日当・宿泊費の質問にアルバイト関連
 *   チャンクがヒット。コンテキスト不足で LLM が「労働者保護法」等の架空情報を生成
 *   する深刻なハルシネーションが発生した。
 * - 中チャンクでは例外条件を含む文が 1 チャンクに収まり Q1 は正確に回答できた。
 *   ただし Q2 では管理職の宿泊費チャンクが検索にヒットせず、アルバイトの上限
 *   8,000 円を誤って適用する取り違えが発生した。
 * - 大チャンクでは宿泊費セクション全体が 1 チャンクに収まり、管理職の宿泊費
 *   15,000 円と日当 5,000 円の両方を正しく取得できた。ただし LLM が「1 泊 2 日」
 *   を 2 泊と誤解釈し合計を 40,000 円と算出（正解は 25,000 円）する計算ミスが残った。
 * - 総じて、チャンクサイズが大きいほど検索で必要な文脈を取得しやすく、
 *   ハルシネーションや情報欠落のリスクが低減する。一方、チャンクサイズだけでは
 *   解決できない LLM の推論・計算ミスも存在するため、プロンプト設計との併用が重要。
 */
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
