/**
 * Refine RAG
 *
 * チャンクを順に読みながら回答を段階的に改善していく手法。
 * Map-Reduce が並列処理で要約→統合するのに対し、
 * Refine は逐次処理で「既存の回答＋新しいチャンク」→「改善された回答」を繰り返す。
 *
 * Map-Reduce との違い:
 * - Map-Reduce: 各チャンクを独立に処理（並列可能だが文脈が断片的）
 * - Refine: 前の回答を引き継ぐため文脈が蓄積される（逐次処理で遅いが一貫性が高い）
 *
 * 検証結果:
 * - 初回チャンクの情報が少ないと LLM がハルシネーションで補完し、
 *   それが以降のステップにも引き継がれてしまう（雪だるま式に悪化）。
 *   例: リモートワークの質問で「業務時間外の業務禁止」等の架空ルールを生成。
 * - 「関連情報がなければ既存回答をそのまま返す」指示が守られず、
 *   無関係なチャンクの情報まで回答に蓄積される傾向がある。
 * - Map-Reduce はチャンクごとに独立処理するため個別のハルシネーションを除外できるが、
 *   Refine は初期のハルシネーションが最終回答まで残り続ける点が大きな弱点。
 * - 退職手続きのように最初のチャンクに十分な情報がある場合は安定して動作する。
 * - 今回の検証規模では通常 RAG が最も簡潔・正確であり、
 *   Refine は大量チャンクから段階的に情報を統合する場面で真価を発揮する手法と言える。
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
console.log("Refine — チャンクを順に読みながら回答を段階的に改善");
console.log("=".repeat(60));

console.log(`\nチャンク設定: chunkSize=100, chunkOverlap=20`);
console.log(`チャンク数: ${chunks.length}`);

// --- ベクトルストア構築 ---

const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);

// --- Refine チェーン ---

// 初回: 最初のチャンクから初期回答を生成
const initialPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `以下の文書の情報をもとに、質問に回答してください。
文書に関連情報がない場合は「この文書には関連情報がありません」と回答してください。

文書:
{chunk}`,
  ],
  ["human", "{question}"],
]);
const initialChain = initialPrompt.pipe(llm).pipe(new StringOutputParser());

// 改善: 既存の回答を新しいチャンクの情報で改善
const refinePrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `あなたには既存の回答と、新しい文書が与えられます。
新しい文書に質問に関連する情報があれば、既存の回答を改善してください。
新しい文書に関連情報がなければ、既存の回答をそのまま返してください。

既存の回答:
{existing_answer}

新しい文書:
{chunk}`,
  ],
  ["human", "{question}"],
]);
const refineChain = refinePrompt.pipe(llm).pipe(new StringOutputParser());

/**
 * Refine で回答を生成する。
 * 1. 最初のチャンクで初期回答を生成
 * 2. 残りのチャンクを順に読み、回答を段階的に改善
 */
async function refineAnswer(
  question: string,
  docs: Document[],
): Promise<{ answer: string; steps: string[] }> {
  const steps: string[] = [];

  // 初回: 最初のチャンクから初期回答を生成
  let currentAnswer = await initialChain.invoke({
    chunk: docs[0].pageContent,
    question,
  });
  steps.push(currentAnswer);

  // 2 番目以降のチャンクで段階的に改善
  for (let i = 1; i < docs.length; i++) {
    currentAnswer = await refineChain.invoke({
      existing_answer: currentAnswer,
      chunk: docs[i].pageContent,
      question,
    });
    steps.push(currentAnswer);
  }

  return { answer: currentAnswer, steps };
}

// --- 質問 ---
// Map-Reduce と同じ質問で比較
const questions = [
  "パートやアルバイトが受けられる待遇をすべて教えてください",
  "退職するときに必要な手続きを教えてください",
  "リモートワークに関するルールを教えてください",
];

const k = 10; // Map-Reduce と同条件で比較

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

  // --- Refine ---
  console.log(`\n【Refine】(k=${k})`);
  console.log(`${docs.length} チャンクを順に読みながら回答を改善中...`);
  const { answer, steps } = await refineAnswer(question, docs);

  steps.forEach((s, i) => {
    const label = i === 0 ? "初期回答" : `改善 ${i}`;
    console.log(`  [${label}] ${s.replace(/\n/g, " ").trim()}`);
  });

  console.log(`\n最終回答: ${answer}`);
}
