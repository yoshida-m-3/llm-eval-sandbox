/**
 * LangFlow チャットフロー呼び出し
 *
 * LangFlow UI で構築したチャットフローを TypeScript クライアントから実行する:
 * 1. 単発質問 — client.flow(flowId).run(input) で応答取得
 * 2. 複数質問の順次実行 — 異なる質問を連続で投げて応答を表示
 *
 * LangFlow UI で構築するフロー構成:
 *   Chat Input -> Prompt Template（「あなたは親切な日本語アシスタントです。{user_message}」）
 *     -> Ollama（llama3.2:3b）-> Chat Output
 * ※ コンポーネント名はバージョンにより異なる場合あり（例: Prompt / Prompt Template）
 *
 * 前提:
 * - LangFlow サーバー起動済み
 * - UI でチャットフローを構築し、Flow ID を .env に設定済み
 * - .env: LANGFLOW_CHAT_FLOW_ID=<Flow ID>
 *
 * 実行: pnpm langflow:02
 */
import { LangflowClient } from "@datastax/langflow-client";

const BASE_URL = process.env.LANGFLOW_BASE_URL || "http://localhost:7860";
const API_KEY = process.env.LANGFLOW_API_KEY;
const FLOW_ID = process.env.LANGFLOW_CHAT_FLOW_ID;

if (!FLOW_ID) {
  console.error("LANGFLOW_CHAT_FLOW_ID が設定されていません。");
  console.error(".env ファイルに LANGFLOW_CHAT_FLOW_ID=<Flow ID> を追加してください。");
  process.exit(1);
}

const client = new LangflowClient({ baseUrl: BASE_URL, apiKey: API_KEY });

// FlowResponse からテキストを抽出するヘルパー
function extractText(response: { outputs?: { outputs?: { results?: { message?: { text?: string } } }[] }[] }): string {
  const text = response.outputs?.[0]?.outputs?.[0]?.results?.message?.text;
  return text ?? JSON.stringify(response);
}

// ============================================================
// 1. 単発質問
// ============================================================
console.log("=".repeat(60));
console.log("1. 単発質問");
console.log("=".repeat(60));

try {
  const response = await client.flow(FLOW_ID).run("TypeScriptとは何ですか？簡潔に教えてください。");
  console.log("質問: TypeScriptとは何ですか？簡潔に教えてください。");
  console.log("応答:", extractText(response));
} catch (error) {
  console.error("フロー実行エラー:", (error as Error).message);
  console.error("Flow ID が正しいか、フローが正しく構築されているか確認してください。");
}
console.log();

// ============================================================
// 2. 複数質問の順次実行
// ============================================================
console.log("=".repeat(60));
console.log("2. 複数質問の順次実行");
console.log("=".repeat(60));

const questions = [
  "日本の首都はどこですか？",
  "Pythonと比べたTypeScriptの特徴を3つ挙げてください。",
  "機械学習を一言で説明してください。",
];

for (const question of questions) {
  console.log(`\n質問: ${question}`);
  try {
    const response = await client.flow(FLOW_ID).run(question);
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
console.log("チャットフロー実行完了");
console.log("=".repeat(60));
