/**
 * Map-Reduce RAG
 *
 * 検証結果:
 * - Map フェーズで各チャンクを個別に LLM へ渡すため、チャンク単体では文脈が不足し
 *   ドキュメントにない情報を補完するハルシネーションが発生しやすい。
 *   例: 退職手続きの質問で、ドキュメントにない「説明会」「退職証明書」等を生成。
 * - 一方、通常 RAG はコンテキストを一括で渡すため事実に忠実な回答になりやすい。
 * - Map-Reduce が有効なのは、長文にまたがる情報の要約・統合が必要なケース。
 *   コンテキストウィンドウに収まらないほど大量のチャンクを扱う場合に真価を発揮する。
 * - 今回の検証ドキュメント（社内ガイド）程度の規模では通常 RAG で十分であり、
 *   Map-Reduce のメリットよりハルシネーションのリスクが目立つ結果となった。
 */
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import type { Document } from "@langchain/core/documents";
import { llm, embeddings, ragChain } from "./shared.js";
import { employeeGuide } from "./documents/employee-guide.js";

// --- チャンク分割 ---

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 100,
  chunkOverlap: 20,
});
const chunks = await splitter.createDocuments([employeeGuide]);

console.log("=".repeat(60));
console.log("Map-Reduce — 各チャンクを個別に要約 → まとめて最終回答を生成");
console.log("=".repeat(60));

console.log(`\nチャンク設定: chunkSize=100, chunkOverlap=20`);
console.log(`チャンク数: ${chunks.length}`);

// --- ベクトルストア構築 ---

const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);

// --- Map-Reduce チェーン ---

// Map: 各チャンクから質問に関連する情報を抽出・要約
const mapPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `以下の文書から、質問に関連する情報を簡潔に抽出してください。
関連する情報がない場合は「関連情報なし」と回答してください。

文書:
{chunk}`,
  ],
  ["human", "{question}"],
]);
const mapChain = mapPrompt.pipe(llm).pipe(new StringOutputParser());

// Reduce: Map の要約をまとめて最終回答を生成
const reducePrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `以下は複数の文書から抽出された要約です。これらの要約を統合して、質問に対する最終的な回答を生成してください。
矛盾する情報がある場合は、その旨を明記してください。

要約一覧:
{summaries}`,
  ],
  ["human", "{question}"],
]);
const reduceChain = reducePrompt.pipe(llm).pipe(new StringOutputParser());

/**
 * Map-Reduce で回答を生成する。
 * 1. 各チャンクを個別に要約（Map）
 * 2. 要約をまとめて最終回答を生成（Reduce）
 */
async function mapReduceAnswer(
  question: string,
  docs: Document[],
): Promise<{ answer: string; summaries: string[] }> {
  // Map: 各チャンクから並列に要約を抽出
  const summaries = await Promise.all(
    docs.map((doc) =>
      mapChain.invoke({ chunk: doc.pageContent, question }),
    ),
  );

  // 「関連情報なし」を除外
  const relevantSummaries = summaries.filter(
    (s) => !s.includes("関連情報なし"),
  );

  // Reduce: 要約を統合して最終回答を生成
  const answer = await reduceChain.invoke({
    summaries:
      relevantSummaries.length > 0
        ? relevantSummaries.map((s, i) => `[${i + 1}] ${s}`).join("\n")
        : "関連する情報は見つかりませんでした。",
    question,
  });

  return { answer, summaries };
}

// --- 質問 ---
// 長文にまたがる情報を統合する必要がある質問
const questions = [
  "パートやアルバイトが受けられる待遇をすべて教えてください",
  "退職するときに必要な手続きを教えてください",
  "リモートワークに関するルールを教えてください",
];

const k = 10; // Map-Reduce では多めに取得して網羅性を高める

// --- 比較実行 ---

for (const question of questions) {
  console.log(`\n${"=".repeat(60)}`);
  console.log(`質問: ${question}`);
  console.log("=".repeat(60));

  const docs = await vectorStore.similaritySearch(question, k);
  console.log(`\n取得チャンク数: ${docs.length}`);
  docs.forEach((doc, i) => {
    console.log(
      `  [${i}] (${doc.pageContent.length}文字) ${doc.pageContent.replace(/\n/g, " ").trim()}`,
    );
  });

  // --- 通常 RAG ---
  console.log(`\n【通常 RAG】(k=${k})`);
  const baselineContext = docs.map((doc) => doc.pageContent).join("\n");
  const baselineAnswer = await ragChain.invoke({
    context: baselineContext,
    question,
  });
  console.log(`回答: ${baselineAnswer}`);

  // --- Map-Reduce ---
  console.log(`\n【Map-Reduce】(k=${k})`);
  console.log(`Map フェーズ: ${docs.length} チャンクを個別に要約中...`);
  const { answer, summaries } = await mapReduceAnswer(question, docs);

  summaries.forEach((s, i) => {
    const label = s.includes("関連情報なし") ? "除外" : "採用";
    console.log(`  Map[${i}] [${label}] ${s.replace(/\n/g, " ").trim()}`);
  });

  const relevantCount = summaries.filter(
    (s) => !s.includes("関連情報なし"),
  ).length;
  console.log(
    `\nReduce フェーズ: ${relevantCount}/${summaries.length} 件の要約を統合`,
  );
  console.log(`回答: ${answer}`);
}
