/**
 * LangFlow ストリーミング & 応用
 *
 * LangFlow の高度な機能を検証する:
 * 1. ストリーミング — client.flow(flowId).stream() でリアルタイム応答
 * 2. セッション維持 — 同一 session_id で会話継続
 * 3. REST API 直接呼び出し — fetch で POST /api/v1/run/{flow_id} を実行
 *
 * 前提:
 * - LangFlow サーバー起動済み
 * - .env: LANGFLOW_CHAT_FLOW_ID=<Flow ID>
 *
 * 実行: pnpm langflow:04
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
// 1. ストリーミング
// ============================================================
console.log("=".repeat(60));
console.log("1. ストリーミング");
console.log("=".repeat(60));

try {
  console.log("質問: LangFlowの特徴を教えてください。");
  console.log("応答（ストリーム）:");

  const stream = await client.flow(FLOW_ID).stream("LangFlowの特徴を教えてください。");
  const reader = stream.getReader();

  while (true) {
    const { done, value: event } = await reader.read();
    if (done) break;
    // イベントの内容を表示（形式はフロー設定により異なる）
    if (typeof event === "string") {
      process.stdout.write(event);
    } else {
      console.log("イベント:", JSON.stringify(event, null, 2));
    }
  }
  console.log();
} catch (error) {
  console.error("ストリーミングエラー:", (error as Error).message);
}
console.log();

// ============================================================
// 2. セッション維持（会話の継続）
// ============================================================
console.log("=".repeat(60));
console.log("2. セッション維持（会話の継続）");
console.log("=".repeat(60));

const sessionId = `session-${Date.now()}`;
console.log(`セッション ID: ${sessionId}\n`);

const conversationMessages = [
  "私の名前は太郎です。覚えてください。",
  "私の名前を覚えていますか？",
  "ありがとうございます。では、TypeScriptについて教えてください。",
];

for (const message of conversationMessages) {
  console.log(`ユーザー: ${message}`);
  try {
    const stream = await client.flow(FLOW_ID).stream(message, {
      session_id: sessionId,
    });
    const reader = stream.getReader();

    process.stdout.write("アシスタント: ");
    while (true) {
      const { done, value: event } = await reader.read();
      if (done) break;
      if (typeof event === "string") {
        process.stdout.write(event);
      } else {
        // オブジェクト形式のイベントの場合は文字列化して表示
        console.log(JSON.stringify(event));
      }
    }
    console.log();
  } catch (error) {
    console.error("エラー:", (error as Error).message);
  }
  console.log("-".repeat(40));
}
console.log();

// ============================================================
// 3. REST API 直接呼び出し
// ============================================================
console.log("=".repeat(60));
console.log("3. REST API 直接呼び出し（fetch）");
console.log("=".repeat(60));

// @datastax/langflow-client の API が不足する場合のフォールバックとして
// REST API を直接呼び出す方法を示す
try {
  const question = "RESTful API とは何ですか？一言で教えてください。";
  console.log(`質問: ${question}`);

  const res = await fetch(`${BASE_URL}/api/v1/run/${FLOW_ID}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(API_KEY ? { "x-api-key": API_KEY } : {}),
    },
    body: JSON.stringify({
      input_value: question,
      output_type: "chat",
      input_type: "chat",
    }),
  });

  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  }

  const result = await res.json();

  // レスポンス構造を表示
  console.log("\nレスポンス構造（トップレベルキー）:", Object.keys(result));

  // outputs からテキストを抽出（LangFlow のレスポンス形式に対応）
  if (result.outputs) {
    for (const output of result.outputs) {
      for (const inner of output.outputs || []) {
        const text =
          inner.results?.message?.text ||
          inner.results?.text ||
          JSON.stringify(inner.results);
        console.log("応答:", text);
      }
    }
  } else {
    console.log("応答（raw）:", JSON.stringify(result, null, 2));
  }
} catch (error) {
  console.error("REST API エラー:", (error as Error).message);
}

// ============================================================
// 完了
// ============================================================
console.log();
console.log("=".repeat(60));
console.log("ストリーミング & 応用 完了");
console.log("=".repeat(60));
