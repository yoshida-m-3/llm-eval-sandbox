/**
 * LangSmith トレース可視化デモ
 *
 * 環境変数 LANGCHAIN_TRACING_V2=true を設定するだけで、LangChain.js の
 * すべての呼び出しが自動的に LangSmith へトレースされることを確認する。
 *
 * 4 つのセクションで段階的にトレース機能を検証:
 * 1. 自動トレース — llm.invoke() だけでトレースが送信される
 * 2. チェーンのトレース — prompt → llm → parser の各ステップが子スパンになる
 * 3. traceable() によるカスタムトレース — 任意の関数を親スパンとしてラップ
 * 4. 既存 RAG チェーンのトレース — shared.ts をゼロ変更で可視化
 *
 * 実行: pnpm trace:06
 */
import { traceable } from "langsmith/traceable";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { llm, embeddings, ragPrompt, ragChain } from "./rag/shared.js";
import { expensePolicy } from "./rag/documents/expense-policy.js";

// ============================================================
// 1. 自動トレース — LLM 単体呼び出し
// ============================================================
console.log("=".repeat(60));
console.log("1. 自動トレース（LLM 単体）");
console.log("=".repeat(60));

const simpleAnswer = await llm.invoke("日本の首都はどこですか？");
console.log(`回答: ${simpleAnswer.content}\n`);
console.log("→ LangSmith ダッシュボードに LLM 呼び出しのトレースが記録されています\n");

// ============================================================
// 2. チェーンのトレース — prompt | llm | parser
// ============================================================
console.log("=".repeat(60));
console.log("2. チェーンのトレース（prompt → llm → parser）");
console.log("=".repeat(60));

const chain = ragPrompt.pipe(llm).pipe(new StringOutputParser());

const chainAnswer = await chain.invoke({
  context: "東京スカイツリーの高さは634メートルです。",
  question: "東京スカイツリーの高さは？",
});
console.log(`回答: ${chainAnswer}\n`);
console.log(
  "→ LangSmith でチェーン内の各ステップ（Prompt / LLM / Parser）が子スパンとして表示されます\n",
);

// ============================================================
// 3. traceable() によるカスタムトレース
// ============================================================
console.log("=".repeat(60));
console.log("3. traceable() によるカスタムトレース（RAG パイプライン）");
console.log("=".repeat(60));

// ベクトルストアを事前準備
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 40,
});
const chunks = await splitter.createDocuments([expensePolicy]);
const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);

// RAG パイプライン全体を traceable でラップ
const ragPipeline = traceable(
  async (question: string) => {
    // 検索
    const relevantDocs = await vectorStore.similaritySearch(question, 3);
    const context = relevantDocs.map((doc) => doc.pageContent).join("\n");

    // プロンプト → LLM → パース
    const answer = await ragChain.invoke({ context, question });
    return { answer, docsUsed: relevantDocs.length };
  },
  { name: "rag-pipeline-custom" },
);

const result = await ragPipeline(
  "アルバイトスタッフが宿泊を伴う出張をすることはできますか？",
);
console.log(`回答: ${result.answer}`);
console.log(`使用チャンク数: ${result.docsUsed}\n`);
console.log(
  "→ LangSmith で 'rag-pipeline-custom' という親スパンの下に検索・LLM 呼び出しがネストされています\n",
);

// ============================================================
// 4. 既存 RAG チェーンのトレース（ゼロコード変更）
// ============================================================
console.log("=".repeat(60));
console.log("4. 既存 RAG チェーンのトレース（shared.ts をそのまま使用）");
console.log("=".repeat(60));

const question = "経費精算の締め日はいつですか？";
const relevantDocs = await vectorStore.similaritySearch(question, 2);
const context = relevantDocs.map((doc) => doc.pageContent).join("\n");

const ragAnswer = await ragChain.invoke({ context, question });
console.log(`質問: ${question}`);
console.log(`回答: ${ragAnswer}\n`);
console.log(
  "→ shared.ts の ragChain は変更なし。環境変数だけで自動トレースされています\n",
);

console.log("=".repeat(60));
console.log("全セクション完了 — LangSmith ダッシュボードでトレースを確認してください");
console.log("https://smith.langchain.com");
console.log("=".repeat(60));
