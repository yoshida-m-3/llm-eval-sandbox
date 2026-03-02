import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
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
console.log("リランキング — 多めに取得 → LLM で関連度再スコアリング → 上位選択");
console.log("=".repeat(60));

console.log(`\nチャンク設定: chunkSize=100, chunkOverlap=20`);
console.log(`チャンク数: ${chunks.length}`);

// --- ベクトルストア構築 ---

const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);

// --- LLM リランキング ---

/**
 * LLM に各チャンクと質問の関連度スコア（1-10）を出力させ、
 * スコア順に並べ替えて上位 k 件を返す。
 */
async function rerankWithLLM(
  question: string,
  docs: Document[],
  topK: number,
): Promise<{ doc: Document; score: number }[]> {
  const scored: { doc: Document; score: number }[] = [];

  for (const doc of docs) {
    const response = await llm.invoke([
      {
        role: "system",
        content: `あなたは検索結果の関連度を評価する専門家です。
質問と文書の関連度を 1〜10 の整数で評価してください。
- 10: 質問に対する直接的な回答を含む
- 7-9: 質問に強く関連する情報を含む
- 4-6: 質問にある程度関連する
- 1-3: 質問にほとんど関連しない

数字のみを出力してください。`,
      },
      {
        role: "human",
        content: `質問: ${question}\n\n文書: ${doc.pageContent}`,
      },
    ]);

    const scoreText = response.content?.toString().trim() ?? "";
    const score = parseInt(scoreText, 10);
    scored.push({ doc, score: Number.isNaN(score) ? 0 : score });
  }

  return scored.sort((a, b) => b.score - a.score).slice(0, topK);
}

// --- 質問 ---

const questions = [
  "VPN接続は必要ですか？",
  "退職届はいつまでに出せばいいですか？",
  "バイトでも有給はもらえますか？",
];

const initialK = 10; // 初回取得件数（多め）
const finalK = 3; // リランキング後の最終件数

// --- 比較実行 ---

for (const question of questions) {
  console.log(`\n${"=".repeat(60)}`);
  console.log(`質問: ${question}`);
  console.log("=".repeat(60));

  // --- 通常のベクトル検索 (k=3) ---
  console.log(`\n【ベクトル検索のみ】(k=${finalK})`);
  const baselineDocs = await vectorStore.similaritySearch(question, finalK);
  console.log(`取得チャンク数: ${baselineDocs.length}`);
  baselineDocs.forEach((doc, i) => {
    console.log(
      `  [${i}] (${doc.pageContent.length}文字) ${doc.pageContent.replace(/\n/g, " ").trim()}`,
    );
  });

  const baselineContext = baselineDocs.map((doc) => doc.pageContent).join("\n");
  const baselineAnswer = await ragChain.invoke({
    context: baselineContext,
    question,
  });
  console.log(`回答: ${baselineAnswer}`);

  // --- リランキング: 多めに取得 → LLM でスコアリング → 上位選択 ---
  console.log(
    `\n【リランキング】(初回 k=${initialK} → LLM スコアリング → 上位 ${finalK} 件)`,
  );
  const candidateDocs = await vectorStore.similaritySearch(question, initialK);
  console.log(`初回取得チャンク数: ${candidateDocs.length}`);
  candidateDocs.forEach((doc, i) => {
    console.log(
      `  [${i}] (${doc.pageContent.length}文字) ${doc.pageContent.replace(/\n/g, " ").trim()}`,
    );
  });

  console.log(`\nLLM によるスコアリング中...`);
  const reranked = await rerankWithLLM(question, candidateDocs, finalK);

  console.log(`\nリランキング結果 (上位 ${finalK} 件):`);
  reranked.forEach((entry, i) => {
    console.log(
      `  [${i}] スコア: ${entry.score}/10 (${entry.doc.pageContent.length}文字) ${entry.doc.pageContent.replace(/\n/g, " ").trim()}`,
    );
  });

  const rerankedContext = reranked
    .map((entry) => entry.doc.pageContent)
    .join("\n");
  const rerankedAnswer = await ragChain.invoke({
    context: rerankedContext,
    question,
  });
  console.log(`回答: ${rerankedAnswer}`);

  // --- 比較サマリー ---
  const baselineSet = new Set(baselineDocs.map((d) => d.pageContent));
  const rerankedSet = new Set(reranked.map((e) => e.doc.pageContent));
  const overlap = baselineDocs.filter((d) => rerankedSet.has(d.pageContent));
  const onlyBaseline = baselineDocs.filter(
    (d) => !rerankedSet.has(d.pageContent),
  );
  const onlyReranked = reranked.filter(
    (e) => !baselineSet.has(e.doc.pageContent),
  );

  console.log(`\n【比較】`);
  console.log(`  ベクトル検索 (k=${finalK}):   ${baselineDocs.length}件`);
  console.log(`  リランキング後:              ${reranked.length}件`);
  console.log(`  共通:                        ${overlap.length}件`);
  console.log(`  ベクトル検索のみ:            ${onlyBaseline.length}件`);
  console.log(`  リランキングで追加:          ${onlyReranked.length}件`);
}
