import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { chunkit } from "semantic-chunking";
import { embeddings, ragChain } from "./shared.js";

// --- 検証用ドキュメント（01-chunk-size.ts と同一）---

const document = `
当社の経費精算規定について説明します。

【出張日当】
正社員の出張日当は1日あたり3,000円です。ただし、管理職の場合は1日あたり5,000円が支給されます。
アルバイトスタッフには出張日当は支給されません。

【交通費】
正社員は交通費の実費精算が可能です。
新幹線のグリーン車利用は管理職のみ認められています。
アルバイトスタッフの交通費は1日あたり上限1,500円までの実費精算となります。

【宿泊費】
正社員の宿泊費の上限は1泊12,000円です。管理職の場合は1泊15,000円まで認められています。
アルバイトスタッフの宿泊を伴う出張は原則として認められていません。ただし、事前に部長の承認を得た場合に限り、1泊8,000円を上限として認められます。

【懇親会費用】
正社員は1人あたり5,000円まで経費として申請できます。
アルバイトスタッフは懇親会費用の経費申請はできません。ただし、歓送迎会の場合は雇用形態に関わらず1人あたり3,000円まで申請可能です。

【申請方法】
経費精算の締め日は毎月25日です。
正社員は社内システムから申請してください。
アルバイトスタッフは専用の紙の申請書を使用してください。
領収書の原本添付が必須です。
`;

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
const chunksA = await chunkBySemanticPackage(document);
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
  document,
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
const chunksC = await chunkByCharacter(document);
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
