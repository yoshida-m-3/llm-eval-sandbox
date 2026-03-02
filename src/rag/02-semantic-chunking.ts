/**
 * セマンティックチャンキング比較検証
 *
 * 3 手法（A: semantic-chunking パッケージ / B: Ollama Embedding 自前実装 / C: 文字数ベース）で
 * 経費精算規定を分割し、「原則 NG だが例外あり」の情報が同一チャンクに保持されるかを検証。
 *
 * 検証結果:
 * - 今回の文書では 3 手法すべてが宿泊費の「原則禁止＋例外条件」を同一チャンクに保持でき、
 *   RAG 回答も同一品質（原則 NG・部長承認で 1 泊 8,000 円上限）となった。
 * - A) semantic-chunking パッケージは 3 チャンクに大きく分割し文脈が最も豊富だが、
 *   複数セクションが 1 チャンクに混在するため、検索ノイズが増えるリスクがある。
 * - B) Ollama Embedding 自前実装は 5 チャンクに分割するが粒度が不均一。
 *   見出し行（例:「【申請方法】」6 文字）が単独チャンクになる等、
 *   閾値チューニングが難しくセクション境界と意味的境界が一致しない場合がある。
 *   隣接文の類似度ログから、見出し→本文の遷移（0.49〜0.53）と
 *   同一セクション内（0.57〜0.85）の差が見て取れるが、閾値 1 つでは最適分割が困難。
 * - C) 文字数ベース（RecursiveCharacterTextSplitter）は 5 チャンクで最も均一な粒度。
 *   今回は各セクションが 100〜130 文字程度でチャンクサイズ 200 と相性が良く、
 *   結果的にセクション境界とほぼ一致する分割になった。
 * - セマンティックチャンキングの真価は、セクション境界が曖昧な自然文や
 *   長文ドキュメントで発揮される。今回のように明確な見出し構造を持つ文書では
 *   文字数ベースの分割でも十分な検索精度が得られ、手法間の差は小さかった。
 */
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { chunkit } from "semantic-chunking";
import { embeddings, ragChain } from "./shared.js";
import { expensePolicy } from "./documents/expense-policy.js";

// --- ユーティリティ ---

/** 行単位で文に分割（空行・見出し行も独立した要素として保持） */
function splitIntoSentences(text: string): string[] {
  return text
    .split("\n")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

/** コサイン類似度 */
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// =====================================================================
// A) semantic-chunking パッケージ
//    内蔵 ONNX モデル（Xenova/all-MiniLM-L6-v2）で隣接文の類似度を計算し分割
// =====================================================================

async function chunkBySemanticPackage(text: string): Promise<string[]> {
  const result = await chunkit(
    [{ document_name: "expense-policy", document_text: text }],
    {
      similarityThreshold: 0.456,
      maxTokenSize: 200,
    },
  );
  return result.map((chunk: { text: string }) => chunk.text);
}

// =====================================================================
// B) 自前実装（Ollama Embedding ベース）
//    nomic-embed-text で文ベクトルを生成 → コサイン類似度 → 閾値以下で分割
// =====================================================================

async function chunkByOllamaEmbedding(
  text: string,
  threshold: number = 0.5,
): Promise<{ chunks: string[]; similarities: { pair: string; sim: number }[] }> {
  const sentences = splitIntoSentences(text);

  // 各文の Embedding を取得
  const embeds = await Promise.all(
    sentences.map((s) => embeddings.embedQuery(s)),
  );

  const similarities: { pair: string; sim: number }[] = [];
  const chunks: string[] = [];
  let currentChunk = [sentences[0]];

  for (let i = 1; i < sentences.length; i++) {
    const sim = cosineSimilarity(embeds[i - 1], embeds[i]);
    similarities.push({
      pair: `「${sentences[i - 1].substring(0, 20)}…」→「${sentences[i].substring(0, 20)}…」`,
      sim,
    });

    if (sim < threshold) {
      // 類似度が閾値未満 → ここで分割
      chunks.push(currentChunk.join("\n"));
      currentChunk = [sentences[i]];
    } else {
      currentChunk.push(sentences[i]);
    }
  }
  if (currentChunk.length > 0) {
    chunks.push(currentChunk.join("\n"));
  }

  return { chunks, similarities };
}

// =====================================================================
// C) RecursiveCharacterTextSplitter（文字数ベースのベースライン）
// =====================================================================

async function chunkByCharacter(text: string): Promise<string[]> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 40,
  });
  const docs = await splitter.createDocuments([text]);
  return docs.map((doc) => doc.pageContent);
}

// =====================================================================
// 比較実行
// =====================================================================

// 原則NGだが例外条件がある質問 → 「原則」と「例外」が同じチャンクにあるかが鍵
const question =
  "アルバイトスタッフが宿泊を伴う出張をすることはできますか？";

console.log("=".repeat(60));
console.log("セマンティックチャンキング 比較検証");
console.log("=".repeat(60));

// --- A) semantic-chunking パッケージ ---
console.log("\n▶ A) semantic-chunking パッケージ（Xenova/all-MiniLM-L6-v2）");
const chunksA = await chunkBySemanticPackage(expensePolicy);
console.log(`  分割数: ${chunksA.length} チャンク`);
chunksA.forEach((chunk, i) => {
  const preview = chunk.replace(/\n/g, " ").substring(0, 80);
  console.log(`  [${i}] (${chunk.length}文字) ${preview}…`);
});

// --- B) Ollama Embedding ベース ---
// 閾値 0.55: 隣接文の類似度ログからセクション境界（0.49〜0.55）と
// 同一セクション内（0.57〜0.85）の間を狙った値
const OLLAMA_THRESHOLD = 0.55;
console.log(`\n▶ B) 自前実装（nomic-embed-text, 閾値=${OLLAMA_THRESHOLD}）`);
const { chunks: chunksB, similarities } = await chunkByOllamaEmbedding(
  expensePolicy,
  OLLAMA_THRESHOLD,
);

console.log("  隣接文のコサイン類似度:");
for (const { pair, sim } of similarities) {
  const marker = sim < OLLAMA_THRESHOLD ? " ← 分割" : "";
  console.log(`    ${sim.toFixed(3)} ${pair}${marker}`);
}
console.log(`  分割数: ${chunksB.length} チャンク`);
chunksB.forEach((chunk, i) => {
  const preview = chunk.replace(/\n/g, " ").substring(0, 80);
  console.log(`  [${i}] (${chunk.length}文字) ${preview}…`);
});

// --- C) 文字数ベース（ベースライン）---
console.log("\n▶ C) RecursiveCharacterTextSplitter（200文字 / overlap 40）");
const chunksC = await chunkByCharacter(expensePolicy);
console.log(`  分割数: ${chunksC.length} チャンク`);
chunksC.forEach((chunk, i) => {
  const preview = chunk.replace(/\n/g, " ").substring(0, 80);
  console.log(`  [${i}] (${chunk.length}文字) ${preview}…`);
});

// --- RAG 回答比較 ---
console.log(`\n${"=".repeat(60)}`);
console.log(`質問: ${question}`);
console.log("=".repeat(60));

const methods: [string, string[]][] = [
  ["A) semantic-chunking パッケージ", chunksA],
  ["B) Ollama Embedding（自前実装）", chunksB],
  ["C) 文字数ベース（ベースライン）", chunksC],
];

for (const [label, chunks] of methods) {
  const docs = chunks.map((text) => new Document({ pageContent: text }));
  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
  const relevantDocs = await vectorStore.similaritySearch(question, 2);

  console.log(`\n--- ${label} ---`);
  console.log("検索結果 (k=2):");
  relevantDocs.forEach((doc, i) => {
    const preview = doc.pageContent.replace(/\n/g, " ").trim().substring(0, 80);
    console.log(`  [${i}] ${preview}…`);
  });

  const context = relevantDocs.map((doc) => doc.pageContent).join("\n");
  const answer = await ragChain.invoke({ context, question });
  console.log(`回答: ${answer}`);
}
