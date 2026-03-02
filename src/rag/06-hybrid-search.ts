import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import type { Document } from "@langchain/core/documents";
import { embeddings, ragChain } from "./shared.js";
import { employeeGuide } from "./documents/employee-guide.js";
import { JapaneseBM25Retriever } from "./lib/japanese-bm25-retriever.js";
import { tokenizeJapanese } from "./lib/japanese-tokenizer.js";

// --- チャンク分割 ---

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 100,
  chunkOverlap: 20,
});
const chunks = await splitter.createDocuments([employeeGuide]);

console.log("=".repeat(60));
console.log("ハイブリッド検索 — ベクトル検索 + BM25（形態素解析）の併用");
console.log("=".repeat(60));

console.log(`\nチャンク設定: chunkSize=100, chunkOverlap=20`);
console.log(`チャンク数: ${chunks.length}`);
chunks.forEach((chunk, i) => {
  console.log(
    `  [${i}] (${chunk.pageContent.length}文字) ${chunk.pageContent.replace(/\n/g, " ").trim()}`,
  );
});

// --- 形態素解析デモ ---

console.log(`\n${"=".repeat(60)}`);
console.log("形態素解析デモ");
console.log("=".repeat(60));

const demoText = "VPN接続は必要ですか？";
const { tokens: demoTokens, allTokens } = await tokenizeJapanese(demoText);
console.log(`\n入力: "${demoText}"`);
console.log("全形態素:");
for (const t of allTokens) {
  console.log(`  "${t.surface_form}" (${t.pos}) → 基本形: ${t.basic_form}`);
}
console.log(
  `フィルタ後 (助詞・助動詞・記号 除去): [${demoTokens.join(", ")}]`,
);

// --- ベクトルストア & 日本語 BM25 Retriever 構築 ---

const k = 3;

const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);
const bm25Retriever = await JapaneseBM25Retriever.fromDocuments(chunks, { k });

// --- RRF (Reciprocal Rank Fusion) ---

/**
 * 2つの検索結果リストを RRF でマージして統合ランキングを生成する。
 * スコア = Σ 1 / (rank + constant) で、constant = 60 が一般的。
 */
function reciprocalRankFusion(...docLists: Document[][]): Document[] {
  const constant = 60;
  const scoreMap = new Map<string, { score: number; doc: Document }>();

  for (const docs of docLists) {
    docs.forEach((doc, rank) => {
      const key = doc.pageContent;
      const existing = scoreMap.get(key);
      const rrfScore = 1 / (rank + 1 + constant);
      if (existing) {
        existing.score += rrfScore;
      } else {
        scoreMap.set(key, { score: rrfScore, doc });
      }
    });
  }

  return Array.from(scoreMap.values())
    .sort((a, b) => b.score - a.score)
    .map((entry) => entry.doc);
}

// --- 質問 ---
// Q1: 固有名詞「VPN」を含む — BM25 がキーワード一致で有利
// Q2: 「退職届」「1ヶ月前」など専門用語 — BM25 のキーワードマッチが効く
// Q3: 語彙ミスマッチ — 「バイト」→「パートタイマー」「アルバイト」「非正規」— ベクトル検索が有利
const questions = [
  "VPN接続は必要ですか？",
  "退職届はいつまでに出せばいいですか？",
  "バイトでも有給はもらえますか？",
];

// --- 比較実行 ---

for (const question of questions) {
  console.log(`\n${"=".repeat(60)}`);
  console.log(`質問: ${question}`);

  const { tokens: qTokens } = await tokenizeJapanese(question);
  console.log(`形態素解析: [${qTokens.join(", ")}]`);
  console.log("=".repeat(60));

  // --- ベクトル検索のみ ---
  console.log(`\n【ベクトル検索のみ】(k=${k})`);
  const vectorDocs = await vectorStore.similaritySearch(question, k);
  console.log(`取得チャンク数: ${vectorDocs.length}`);
  vectorDocs.forEach((doc, i) => {
    console.log(
      `  [${i}] (${doc.pageContent.length}文字) ${doc.pageContent.replace(/\n/g, " ").trim()}`,
    );
  });

  const vectorContext = vectorDocs.map((doc) => doc.pageContent).join("\n");
  const vectorAnswer = await ragChain.invoke({
    context: vectorContext,
    question,
  });
  console.log(`回答: ${vectorAnswer}`);

  // --- BM25 のみ（形態素解析） ---
  console.log(`\n【BM25 検索（形態素解析）】(k=${k})`);
  const bm25Docs = await bm25Retriever.invoke(question);
  console.log(`取得チャンク数: ${bm25Docs.length}`);
  bm25Docs.forEach((doc, i) => {
    console.log(
      `  [${i}] (${doc.pageContent.length}文字) ${doc.pageContent.replace(/\n/g, " ").trim()}`,
    );
  });

  const bm25Context = bm25Docs.map((doc) => doc.pageContent).join("\n");
  const bm25Answer = await ragChain.invoke({
    context: bm25Context,
    question,
  });
  console.log(`回答: ${bm25Answer}`);

  // --- ハイブリッド検索 (RRF) ---
  console.log(`\n【ハイブリッド検索 (RRF)】(各 k=${k})`);
  const hybridDocs = reciprocalRankFusion(vectorDocs, bm25Docs);
  console.log(`統合結果: ${hybridDocs.length}件`);
  hybridDocs.forEach((doc, i) => {
    console.log(
      `  [${i}] (${doc.pageContent.length}文字) ${doc.pageContent.replace(/\n/g, " ").trim()}`,
    );
  });

  const hybridContext = hybridDocs.map((doc) => doc.pageContent).join("\n");
  const hybridAnswer = await ragChain.invoke({
    context: hybridContext,
    question,
  });
  console.log(`回答: ${hybridAnswer}`);

  // --- 比較サマリー ---
  const vectorSet = new Set(vectorDocs.map((d) => d.pageContent));
  const bm25Set = new Set(bm25Docs.map((d) => d.pageContent));
  const hybridSet = new Set(hybridDocs.map((d) => d.pageContent));

  const onlyVector = vectorDocs.filter((d) => !bm25Set.has(d.pageContent));
  const onlyBm25 = bm25Docs.filter((d) => !vectorSet.has(d.pageContent));
  const both = vectorDocs.filter((d) => bm25Set.has(d.pageContent));

  console.log(`\n【比較】`);
  console.log(`  ベクトル検索: ${vectorSet.size}件`);
  console.log(`  BM25 検索:    ${bm25Set.size}件`);
  console.log(`  ハイブリッド: ${hybridSet.size}件`);
  console.log(`  両方で取得:   ${both.length}件`);
  console.log(`  ベクトルのみ: ${onlyVector.length}件`);
  console.log(`  BM25のみ:     ${onlyBm25.length}件`);
}
