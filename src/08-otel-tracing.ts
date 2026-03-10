/**
 * OpenTelemetry トレース可視化デモ
 *
 * LangChain.js の各ステップを OTel スパンとして記録し、
 * ローカル Grafana（Tempo）でトレースを可視化する。
 *
 * 前提条件:
 *   cd docker && docker compose up -d
 *
 * 3 つのセクションで段階的に検証:
 * 1. 手動スパン — OTel API で独自スパンを作成
 * 2. LangChain コールバック — LLM・チェーンの自動スパン化
 * 3. RAG パイプライン — 検索 → LLM の一連のフローをトレース
 *
 * 実行: pnpm otel:08
 * 確認: http://localhost:3000 (Grafana → Explore → Tempo)
 */

// OTel SDK を最初にインポート（計装を有効化）
import "./otel/setup.js";

import { trace } from "@opentelemetry/api";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { llm, embeddings, ragPrompt } from "./rag/shared.js";
import { expensePolicy } from "./rag/documents/expense-policy.js";
import { OTelCallbackHandler } from "./otel/langchain-otel-handler.js";

const otelHandler = new OTelCallbackHandler();
const tracer = trace.getTracer("otel-demo", "1.0.0");

// ============================================================
// 1. 手動スパン — OTel API を直接使用
// ============================================================
console.log("=".repeat(60));
console.log("1. 手動スパン（OTel API 直接使用）");
console.log("=".repeat(60));

await tracer.startActiveSpan("manual-greeting", async (span) => {
  span.setAttribute("input.question", "日本の首都");
  const answer = await llm.invoke("日本の首都はどこですか？", {
    callbacks: [otelHandler],
  });
  span.setAttribute("output.content", String(answer.content));
  console.log(`回答: ${answer.content}`);
  span.end();
});

console.log("→ Grafana Tempo に 'manual-greeting' スパンが記録されています\n");

// ============================================================
// 2. LangChain コールバック — チェーンの自動スパン化
// ============================================================
console.log("=".repeat(60));
console.log("2. LangChain チェーンのトレース（コールバック経由）");
console.log("=".repeat(60));

const chain = ragPrompt.pipe(llm).pipe(new StringOutputParser());

await tracer.startActiveSpan("chain-demo", async (span) => {
  const answer = await chain.invoke(
    {
      context: "東京スカイツリーの高さは634メートルです。",
      question: "東京スカイツリーの高さは？",
    },
    { callbacks: [otelHandler] },
  );
  span.setAttribute("output.answer", answer);
  console.log(`回答: ${answer}`);
  span.end();
});

console.log(
  "→ 'chain-demo' の下に Prompt / LLM / Parser の子スパンが表示されます\n",
);

// ============================================================
// 3. RAG パイプライン — 検索 → LLM のフルトレース
// ============================================================
console.log("=".repeat(60));
console.log("3. RAG パイプライン全体のトレース");
console.log("=".repeat(60));

// ベクトルストアを準備
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 40,
});
const chunks = await splitter.createDocuments([expensePolicy]);
const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);

const question = "アルバイトスタッフが宿泊を伴う出張をすることはできますか？";

await tracer.startActiveSpan("rag-pipeline", async (span) => {
  span.setAttribute("input.question", question);

  // 検索フェーズ
  const relevantDocs = await tracer.startActiveSpan(
    "retrieval",
    async (retrievalSpan) => {
      const docs = await vectorStore.similaritySearch(question, 3);
      retrievalSpan.setAttribute("retrieval.doc_count", docs.length);
      retrievalSpan.setAttribute(
        "retrieval.doc_lengths",
        docs.map((d) => d.pageContent.length).join(","),
      );
      retrievalSpan.end();
      return docs;
    },
  );

  // 生成フェーズ
  const context = relevantDocs.map((doc) => doc.pageContent).join("\n");

  const answer = await tracer.startActiveSpan(
    "generation",
    async (genSpan) => {
      genSpan.setAttribute("generation.context_length", context.length);
      const result = await chain.invoke(
        { context, question },
        { callbacks: [otelHandler] },
      );
      genSpan.setAttribute("generation.answer_length", result.length);
      genSpan.end();
      return result;
    },
  );

  span.setAttribute("output.answer_length", String(answer.length));
  console.log(`質問: ${question}`);
  console.log(`使用チャンク数: ${relevantDocs.length}`);
  console.log(`回答: ${answer}`);
  span.end();
});

console.log(
  "\n→ 'rag-pipeline' の下に 'retrieval' と 'generation' がネストされています\n",
);

// ============================================================
// 完了 — スパンのフラッシュを待つ
// ============================================================
console.log("=".repeat(60));
console.log("全セクション完了 — トレースをフラッシュ中...");

// BatchSpanProcessor のフラッシュを待つ
const { sdk } = await import("./otel/setup.js");
await sdk.shutdown();

console.log("フラッシュ完了！");
console.log("");
console.log("Grafana でトレースを確認:");
console.log("  1. http://localhost:3000 を開く");
console.log("  2. 左メニュー → Explore");
console.log("  3. データソース 'Tempo' を選択");
console.log("  4. 'Search' タブでトレースを検索");
console.log("=".repeat(60));
