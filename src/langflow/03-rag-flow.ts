/**
 * LangFlow RAG フロー呼び出し
 *
 * LangFlow UI で構築した RAG フローを TypeScript クライアントから実行する:
 * 1. RAG 質問応答 — client.flow(flowId).run(question) で検索+生成
 * 2. tweaks による実行時パラメータ変更 — チャンクサイズや temperature を動的に変更
 *
 * LangFlow UI で構築するフロー構成:
 *   [インジェスト側]
 *   Text Input（検証用ドキュメント）-> RecursiveCharacterTextSplitter
 *     -> Vector Store + Ollama Embeddings（nomic-embed-text）
 *   [クエリ側]
 *   Chat Input -> Retriever -> Prompt（RAGテンプレート）-> Ollama（llama3.2:3b）-> Chat Output
 *
 * 前提:
 * - LangFlow サーバー起動済み
 * - UI で RAG フローを構築し、Flow ID を .env に設定済み
 * - .env: LANGFLOW_RAG_FLOW_ID=<Flow ID>
 *
 * 実行: pnpm langflow:03
 */
import { LangflowClient } from "@datastax/langflow-client";

const BASE_URL = process.env.LANGFLOW_BASE_URL || "http://localhost:7860";
const API_KEY = process.env.LANGFLOW_API_KEY;
const FLOW_ID = process.env.LANGFLOW_RAG_FLOW_ID;

if (!FLOW_ID) {
  console.error("LANGFLOW_RAG_FLOW_ID が設定されていません。");
  console.error(".env ファイルに LANGFLOW_RAG_FLOW_ID=<Flow ID> を追加してください。");
  process.exit(1);
}

const client = new LangflowClient({ baseUrl: BASE_URL, apiKey: API_KEY });

// FlowResponse からテキストを抽出するヘルパー
function extractText(response: { outputs?: { outputs?: { results?: { message?: { text?: string } } }[] }[] }): string {
  const text = response.outputs?.[0]?.outputs?.[0]?.results?.message?.text;
  return text ?? JSON.stringify(response);
}

// ============================================================
// 1. RAG 質問応答
// ============================================================
console.log("=".repeat(60));
console.log("1. RAG 質問応答");
console.log("=".repeat(60));

const ragQuestions = [
  "このドキュメントの主なトピックは何ですか？",
  "具体的な手順やルールがあれば教えてください。",
];

for (const question of ragQuestions) {
  console.log(`\n質問: ${question}`);
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
// 2. tweaks によるパラメータ変更
// ============================================================
console.log("=".repeat(60));
console.log("2. tweaks によるパラメータ変更");
console.log("=".repeat(60));

// tweaks でコンポーネントのパラメータを実行時に上書きできる
// コンポーネント ID は LangFlow UI のノード設定から確認
// 形式: { "コンポーネントID": { パラメータ名: 値 } }
// ※ 以下は例 — 実際のコンポーネント ID（例: "Ollama-xxxxx"）に置き換えて使用
const OLLAMA_COMPONENT_ID = "Ollama";

const tweakConfigs = [
  {
    label: "デフォルト設定",
    tweaks: {} as Record<string, Record<string, string | number | null | boolean>>,
  },
  {
    label: "temperature を 0.1 に下げる（より確定的な応答）",
    tweaks: { [OLLAMA_COMPONENT_ID]: { temperature: 0.1 } },
  },
  {
    label: "temperature を 0.9 に上げる（より創造的な応答）",
    tweaks: { [OLLAMA_COMPONENT_ID]: { temperature: 0.9 } },
  },
];

const testQuestion = "ドキュメントの内容を要約してください。";

for (const config of tweakConfigs) {
  console.log(`\n[${config.label}]`);
  console.log(`質問: ${testQuestion}`);
  try {
    const response = await client.flow(FLOW_ID).run(testQuestion, {
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
console.log("RAG フロー実行完了");
console.log("=".repeat(60));
