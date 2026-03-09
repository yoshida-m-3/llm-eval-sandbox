/**
 * LangChain.js vs LangFlow 比較検証
 *
 * 同一の RAG タスクを LangChain.js（コード）と LangFlow（ビジュアル/API）の
 * 両方で実行し、以下の観点で比較する:
 * 1. 応答品質 — 同一質問セットに対する回答内容の比較
 * 2. レイテンシ — 実行時間の計測・比較
 * 3. 開発体験 — コード量・カスタマイズ性の定性評価
 *
 * 前提:
 * - Ollama 起動済み（llama3.2:3b, nomic-embed-text）
 * - LangFlow サーバー起動済み
 * - .env: LANGFLOW_RAG_FLOW_ID=<RAG Flow ID>
 *
 * 実行: pnpm langflow:06
 */
import { LangflowClient } from "@datastax/langflow-client";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { llm, embeddings, ragChain } from "../rag/shared.js";

// --- 設定 ---

const BASE_URL = process.env.LANGFLOW_BASE_URL || "http://localhost:7860";
const API_KEY = process.env.LANGFLOW_API_KEY;
const RAG_FLOW_ID = process.env.LANGFLOW_RAG_FLOW_ID;

if (!RAG_FLOW_ID) {
  console.error("LANGFLOW_RAG_FLOW_ID が設定されていません。");
  process.exit(1);
}

const langflowClient = new LangflowClient({
  baseUrl: BASE_URL,
  apiKey: API_KEY,
});

// --- テスト用ドキュメント（LangFlow UI 側にも同じ内容をインジェスト済みであること） ---

const document = `
llm-eval-sandbox は LangChain エコシステムを学習するためのプロジェクトです。
技術スタックは TypeScript と pnpm を使用しています。
LLMプロバイダーとして Ollama（ローカル）を使用し、将来的に Amazon Bedrock に移行予定です。
マイルストーンは4つあります。
Milestone 1 は LangChain と LangSmith の学習です。
Milestone 2 は LangGraph によるマルチエージェントワークフローの構築です。
Milestone 3 は LangFlow によるビジュアルフロー構築です。
Milestone 4 は OpenTelemetry による可観測性の検証です。
RAG の手法として、チャンクサイズ調整、セマンティックチャンキング、ハイブリッド検索、リランキングなどを検証しています。
LangGraph ではスーパーバイザーパターン、ハンドオフパターン、並列実行パターンを実装しました。
`;

// --- 共通の質問セット ---

const questions = [
  "このプロジェクトで使用している技術スタックは何ですか？",
  "Milestone 2 の内容を教えてください。",
  "RAG の手法としてどのようなものを検証していますか？",
  "LangGraph で実装したパターンを教えてください。",
];

// --- ヘルパー ---

function extractLangFlowText(response: {
  outputs?: {
    outputs?: { results?: { message?: { text?: string } } }[];
  }[];
}): string {
  const text = response.outputs?.[0]?.outputs?.[0]?.results?.message?.text;
  return text ?? JSON.stringify(response);
}

interface Result {
  question: string;
  langchain: { answer: string; timeMs: number };
  langflow: { answer: string; timeMs: number };
}

// ============================================================
// 1. LangChain.js RAG のセットアップ
// ============================================================
console.log("=".repeat(60));
console.log("LangChain.js RAG セットアップ");
console.log("=".repeat(60));

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 100,
  chunkOverlap: 20,
});
const chunks = await splitter.createDocuments([document]);
const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);
console.log(`ドキュメント分割: ${chunks.length} チャンク`);
console.log(`モデル: ${llm.model}, temperature: ${llm.temperature}`);
console.log();

// ============================================================
// 2. 比較実行
// ============================================================
console.log("=".repeat(60));
console.log("比較実行: LangChain.js vs LangFlow");
console.log("=".repeat(60));

const results: Result[] = [];

for (const question of questions) {
  console.log(`\n質問: ${question}`);
  console.log("-".repeat(40));

  // --- LangChain.js ---
  const lcStart = performance.now();
  const relevantDocs = await vectorStore.similaritySearch(question, 3);
  const context = relevantDocs.map((doc) => doc.pageContent).join("\n");
  const lcAnswer = await ragChain.invoke({ context, question });
  const lcTime = performance.now() - lcStart;

  console.log(`[LangChain.js] (${lcTime.toFixed(0)}ms)`);
  console.log(`  ${lcAnswer.substring(0, 200)}`);

  // --- LangFlow ---
  const lfStart = performance.now();
  let lfAnswer: string;
  try {
    const response = await langflowClient.flow(RAG_FLOW_ID).run(question);
    lfAnswer = extractLangFlowText(response);
  } catch (error) {
    lfAnswer = `エラー: ${(error as Error).message}`;
  }
  const lfTime = performance.now() - lfStart;

  console.log(`[LangFlow]     (${lfTime.toFixed(0)}ms)`);
  console.log(`  ${lfAnswer.substring(0, 200)}`);

  results.push({
    question,
    langchain: { answer: lcAnswer, timeMs: lcTime },
    langflow: { answer: lfAnswer, timeMs: lfTime },
  });
}

// ============================================================
// 3. 比較サマリー
// ============================================================
console.log();
console.log("=".repeat(60));
console.log("比較サマリー");
console.log("=".repeat(60));

// レイテンシ比較
console.log("\n--- レイテンシ比較 (ms) ---");
console.log(
  `${"質問".padEnd(40)} ${"LangChain.js".padStart(12)} ${"LangFlow".padStart(12)} ${"差分".padStart(10)}`
);
console.log("-".repeat(76));

let lcTotalMs = 0;
let lfTotalMs = 0;

for (const r of results) {
  const diff = r.langflow.timeMs - r.langchain.timeMs;
  const diffStr = `${diff > 0 ? "+" : ""}${diff.toFixed(0)}`;
  console.log(
    `${r.question.substring(0, 38).padEnd(40)} ${r.langchain.timeMs.toFixed(0).padStart(12)} ${r.langflow.timeMs.toFixed(0).padStart(12)} ${diffStr.padStart(10)}`
  );
  lcTotalMs += r.langchain.timeMs;
  lfTotalMs += r.langflow.timeMs;
}

console.log("-".repeat(76));
const avgDiff = (lfTotalMs - lcTotalMs) / results.length;
console.log(
  `${"平均".padEnd(40)} ${(lcTotalMs / results.length).toFixed(0).padStart(12)} ${(lfTotalMs / results.length).toFixed(0).padStart(12)} ${(avgDiff > 0 ? "+" : "") + avgDiff.toFixed(0).padStart(9)}`
);

// 応答長の比較
console.log("\n--- 応答長比較 (文字数) ---");
console.log(
  `${"質問".padEnd(40)} ${"LangChain.js".padStart(12)} ${"LangFlow".padStart(12)}`
);
console.log("-".repeat(66));

for (const r of results) {
  console.log(
    `${r.question.substring(0, 38).padEnd(40)} ${String(r.langchain.answer.length).padStart(12)} ${String(r.langflow.answer.length).padStart(12)}`
  );
}

// 定性比較
console.log("\n--- 開発体験の定性比較 ---");
console.log(`
┌─────────────────┬──────────────────────────────┬──────────────────────────────┐
│ 観点            │ LangChain.js（コード）        │ LangFlow（ビジュアル）        │
├─────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 初期構築        │ コード記述が必要              │ UI でドラッグ&ドロップ        │
│ カスタマイズ    │ 自由度高（コードで何でも可能）│ コンポーネント制約あり         │
│ デバッグ        │ コード上 + LangSmith          │ UI ログ + LangSmith           │
│ パラメータ変更  │ コード修正 → 再実行           │ tweaks API で動的変更可能     │
│ チーム共有      │ Git でコード共有              │ フロー JSON エクスポート       │
│ API 化          │ 別途 API サーバー構築が必要   │ 組み込み REST API あり         │
│ 再現性          │ コード = 完全な再現性         │ UI 状態に依存                 │
└─────────────────┴──────────────────────────────┴──────────────────────────────┘
`);

// ============================================================
// 完了
// ============================================================
console.log("=".repeat(60));
console.log("比較検証完了");
console.log("=".repeat(60));
