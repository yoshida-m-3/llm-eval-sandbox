import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

const model = new ChatOllama({
  model: "llama3.2:3b",
  temperature: 0,
});

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "あなたは{role}です。簡潔に1文で回答してください。"],
  ["human", "{question}"],
]);

// チェーン: テンプレート → モデル → 出力パーサー を pipe でつなぐ
// StringOutputParser は AIMessage から content の文字列だけを取り出す
const chain = prompt.pipe(model).pipe(new StringOutputParser());

// chain.invoke() 1回で テンプレート展開 → LLM呼び出し → 文字列抽出 が実行される
const result = await chain.invoke({
  role: "プログラミング講師",
  question: "TypeScriptとは？",
});

console.log("=== チェーンの結果 ===");
console.log(result); // string型（AIMessageではなく純粋な文字列）
