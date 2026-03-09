/**
 * LangFlow カスタムコンポーネント検証
 *
 * LangFlow UI にカスタムコンポーネントを登録し、フローに組み込んで動作確認する:
 * 1. コンポーネントファイルの確認 — components/ ディレクトリの Python ファイルを表示
 * 2. カスタムコンポーネント付きフローの実行 — 日本語プリプロセッサ経由で質問応答
 * 3. tweaks でカスタムコンポーネントのパラメータ変更 — system_instruction の動的変更
 *
 * カスタムコンポーネントの登録手順（LangFlow UI）:
 *   1. LangFlow UI の左サイドバーで「+ New Component」をクリック
 *   2. components/japanese_preprocessor.py のコードを貼り付け
 *   3. components/custom_rag_prompt.py のコードを貼り付け
 *   4. フローに組み込む:
 *      Chat Input → Japanese Preprocessor → Japanese RAG Prompt → Ollama → Chat Output
 *                                           ↑
 *                                    Retriever（コンテキスト）
 *
 * 前提:
 * - LangFlow サーバー起動済み
 * - UI でカスタムコンポーネントを使ったフローを構築し、Flow ID を .env に設定済み
 * - .env: LANGFLOW_CUSTOM_FLOW_ID=<Flow ID>
 *
 * 実行: pnpm langflow:05
 */
import { LangflowClient } from "@datastax/langflow-client";
import { readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const BASE_URL = process.env.LANGFLOW_BASE_URL || "http://localhost:7860";
const API_KEY = process.env.LANGFLOW_API_KEY;
const FLOW_ID = process.env.LANGFLOW_CUSTOM_FLOW_ID;

if (!FLOW_ID) {
  console.error("LANGFLOW_CUSTOM_FLOW_ID が設定されていません。");
  console.error(
    ".env ファイルに LANGFLOW_CUSTOM_FLOW_ID=<Flow ID> を追加してください。"
  );
  console.error("\nカスタムコンポーネントの登録手順:");
  console.error(
    "  1. LangFlow UI で components/ 内の Python コードをカスタムコンポーネントとして登録"
  );
  console.error(
    "  2. Chat Input → JapanesePreprocessor → CustomRAGPrompt → Ollama → Chat Output のフローを構築"
  );
  console.error("  3. Flow ID を .env に設定");
  process.exit(1);
}

const client = new LangflowClient({ baseUrl: BASE_URL, apiKey: API_KEY });

// FlowResponse からテキストを抽出するヘルパー
function extractText(response: {
  outputs?: {
    outputs?: { results?: { message?: { text?: string } } }[];
  }[];
}): string {
  const text = response.outputs?.[0]?.outputs?.[0]?.results?.message?.text;
  return text ?? JSON.stringify(response);
}

// ============================================================
// 1. カスタムコンポーネントのソースコード確認
// ============================================================
console.log("=".repeat(60));
console.log("1. カスタムコンポーネント一覧");
console.log("=".repeat(60));

const componentFiles = [
  "japanese_preprocessor.py",
  "custom_rag_prompt.py",
];

for (const file of componentFiles) {
  const filePath = join(__dirname, "components", file);
  try {
    const content = readFileSync(filePath, "utf-8");
    // docstring からコンポーネント概要を抽出
    const docMatch = content.match(/"""([\s\S]*?)"""/);
    const summary = docMatch
      ? docMatch[1].trim().split("\n")[0]
      : "(概要なし)";

    // display_name を抽出
    const nameMatch = content.match(/display_name\s*=\s*"([^"]+)"/);
    const displayName = nameMatch ? nameMatch[1] : file;

    console.log(`\n  [${displayName}] ${file}`);
    console.log(`  概要: ${summary}`);

    // inputs を抽出
    const inputMatches = content.matchAll(/name="(\w+)",\s*\n\s*display_name="([^"]+)"/g);
    const inputs = [...inputMatches].map((m) => `${m[2]}(${m[1]})`);
    if (inputs.length > 0) {
      console.log(`  入力: ${inputs.join(", ")}`);
    }

    // outputs を抽出
    const outputMatches = content.matchAll(
      /display_name="([^"]+)",\s*\n\s*name="(\w+)",\s*\n\s*method="(\w+)"/g
    );
    const outputs = [...outputMatches].map((m) => `${m[1]}(${m[2]})`);
    if (outputs.length > 0) {
      console.log(`  出力: ${outputs.join(", ")}`);
    }
  } catch {
    console.error(`  ${file}: ファイルが見つかりません`);
  }
}
console.log();

// ============================================================
// 2. カスタムコンポーネント付きフローの実行
// ============================================================
console.log("=".repeat(60));
console.log("2. カスタムコンポーネント付きフロー実行");
console.log("=".repeat(60));

// 全角文字を含むテスト入力（プリプロセッサの効果を確認）
const testQuestions = [
  {
    label: "全角英数字を含む質問",
    question: "このドキュメントの　主なトピックを　３つ教えてください。",
  },
  {
    label: "通常の質問",
    question: "具体的なルールや手順があれば教えてください。",
  },
];

for (const { label, question } of testQuestions) {
  console.log(`\n[${label}]`);
  console.log(`質問: ${question}`);
  try {
    const response = await client.flow(FLOW_ID).run(question);
    console.log("応答:", extractText(response));
  } catch (error) {
    console.error("エラー:", (error as Error).message);
  }
  console.log("-".repeat(40));
}
console.log();

// ============================================================
// 3. tweaks でカスタムコンポーネントのパラメータ変更
// ============================================================
console.log("=".repeat(60));
console.log("3. tweaks によるカスタムコンポーネントのパラメータ変更");
console.log("=".repeat(60));

// CustomRAGPrompt コンポーネントの system_instruction を tweaks で変更
// ※ コンポーネント ID は LangFlow UI のノード設定から確認
const RAG_PROMPT_COMPONENT_ID = "CustomRAGPrompt";

const tweakConfigs = [
  {
    label: "デフォルト（コンテキストのみで回答）",
    tweaks: {} as Record<string, Record<string, string | number>>,
  },
  {
    label: "箇条書き指示を追加",
    tweaks: {
      [RAG_PROMPT_COMPONENT_ID]: {
        system_instruction:
          "以下のコンテキスト情報のみを使って質問に回答してください。\n回答は箇条書き形式でまとめてください。",
      },
    },
  },
  {
    label: "要約モード（max_context_length を制限）",
    tweaks: {
      [RAG_PROMPT_COMPONENT_ID]: {
        system_instruction:
          "以下のコンテキスト情報のみを使って質問に簡潔に回答してください。一文で答えてください。",
        max_context_length: 500,
      },
    },
  },
];

const tweakQuestion = "ドキュメントの内容を要約してください。";

for (const config of tweakConfigs) {
  console.log(`\n[${config.label}]`);
  console.log(`質問: ${tweakQuestion}`);
  try {
    const response = await client.flow(FLOW_ID).run(tweakQuestion, {
      tweaks: config.tweaks,
    });
    console.log("応答:", extractText(response));
  } catch (error) {
    console.error("エラー:", (error as Error).message);
  }
  console.log("-".repeat(40));
}

// ============================================================
// 完了
// ============================================================
console.log();
console.log("=".repeat(60));
console.log("カスタムコンポーネント検証完了");
console.log("=".repeat(60));
