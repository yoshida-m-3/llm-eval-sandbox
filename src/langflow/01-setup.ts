/**
 * LangFlow セットアップ確認
 *
 * LangFlow サーバーへの接続確認とフロー一覧取得を行う:
 * 1. ヘルスチェック — /health エンドポイントでサーバー稼働確認
 * 2. クライアント初期化 — LangflowClient の生成
 * 3. フロー一覧取得 — REST API でフロー名・ID を表示
 *
 * 前提:
 * - LangFlow サーバーが起動済み（uv: langflow run / Docker: docker run -p 7860:7860 langflowai/langflow:latest）
 * - Docker 利用時の Ollama URL は http://host.docker.internal:11434
 *
 * 実行: pnpm langflow:01
 */
import { LangflowClient } from "@datastax/langflow-client";

const BASE_URL = process.env.LANGFLOW_BASE_URL || "http://localhost:7860";
const API_KEY = process.env.LANGFLOW_API_KEY;

// ============================================================
// 1. ヘルスチェック
// ============================================================
console.log("=".repeat(60));
console.log("1. ヘルスチェック");
console.log("=".repeat(60));

try {
  const res = await fetch(`${BASE_URL}/health`);
  const body = await res.json();
  console.log(`ステータス: ${res.status}`);
  console.log("レスポンス:", body);
} catch (error) {
  console.error("LangFlow サーバーに接続できません。サーバーが起動しているか確認してください。");
  console.error(`URL: ${BASE_URL}/health`);
  console.error("エラー:", (error as Error).message);
  process.exit(1);
}
console.log();

// ============================================================
// 2. クライアント初期化
// ============================================================
console.log("=".repeat(60));
console.log("2. クライアント初期化");
console.log("=".repeat(60));

const client = new LangflowClient({ baseUrl: BASE_URL, apiKey: API_KEY });
console.log(`LangflowClient を初期化しました（baseUrl: ${BASE_URL}）`);
console.log();

// ============================================================
// 3. フロー一覧取得
// ============================================================
console.log("=".repeat(60));
console.log("3. フロー一覧取得");
console.log("=".repeat(60));

try {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (API_KEY) headers["x-api-key"] = API_KEY;
  const res = await fetch(`${BASE_URL}/api/v1/flows`, { headers });
  const flows = await res.json();

  if (!Array.isArray(flows) || flows.length === 0) {
    console.log("登録済みフローはありません。LangFlow UI でフローを作成してください。");
  } else {
    console.log(`登録済みフロー数: ${flows.length}\n`);
    for (const flow of flows) {
      console.log(`  名前: ${flow.name}`);
      console.log(`  ID:   ${flow.id}`);
      console.log(`  説明: ${flow.description || "(なし)"}`);
      console.log("  ---");
    }
  }
} catch (error) {
  console.error("フロー一覧の取得に失敗しました:", (error as Error).message);
}

// ============================================================
// 完了
// ============================================================
console.log();
console.log("=".repeat(60));
console.log("セットアップ確認完了");
console.log("=".repeat(60));
