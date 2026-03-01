import { ChatOllama } from "@langchain/ollama";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// ツールの定義: LLMが呼び出せる関数を作成する
// z.object でパラメータのスキーマを定義し、LLMが適切な引数を生成できるようにする
const weatherTool = tool(
  async ({ city }) => {
    // 実際にはAPIを呼ぶが、ここではダミーデータを返す
    const data: Record<string, string> = {
      Tokyo: "晴れ 25℃",
      Osaka: "曇り 22℃",
      Sapporo: "雪 -3℃",
    };
    return data[city] ?? `${city}の天気情報は見つかりませんでした`;
  },
  {
    name: "get_weather",
    description: "指定した都市の天気を取得します",
    schema: z.object({
      city: z.string().describe("天気を取得したい都市名"),
    }),
  },
);

const model = new ChatOllama({
  model: "llama3.2:3b",
  temperature: 0,
});

// モデルにツールを紐付ける（bindTools）
const modelWithTools = model.bindTools([weatherTool]);

// LLMにツールを使うべき質問を投げる
// NOTE: llama3.2:3b は小型モデルのため、複数ツール呼び出しは不安定。
//       より大きなモデル（llama3.1:8b 等）や OpenAI/Bedrock ではより正確に動作する。
const response = await modelWithTools.invoke("東京の天気を教えてください");

console.log("=== LLMの応答 ===");
console.log("content:", response.content);
console.log("tool_calls:", JSON.stringify(response.tool_calls, null, 2));

// LLMが返した tool_calls を実際に実行する
if (response.tool_calls && response.tool_calls.length > 0) {
  console.log("\n=== ツール実行結果 ===");
  for (const tc of response.tool_calls) {
    const result = await weatherTool.invoke(tc.args);
    console.log(`${tc.args.city}: ${result}`);
  }
}
