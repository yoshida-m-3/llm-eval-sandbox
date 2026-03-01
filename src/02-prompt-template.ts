import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const model = new ChatOllama({
  model: "llama3.2:3b",
  temperature: 0,
});

// プロンプトテンプレート: 変数 {role} と {question} を埋め込み可能なひな型
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "あなたは{role}です。簡潔に1文で回答してください。"],
  ["human", "{question}"],
]);

// テンプレートに変数を渡してメッセージを生成し、モデルに渡す
const messages = await prompt.invoke({
  role: "プログラミング講師",
  question: "TypeScriptとは？",
});
const response = await model.invoke(messages);

console.log("=== プログラミング講師として ===");
console.log(response.content);

// 同じテンプレートを変数を変えて再利用
const messages2 = await prompt.invoke({
  role: "小学校の先生",
  question: "TypeScriptとは？",
});
const response2 = await model.invoke(messages2);

console.log("\n=== 小学校の先生として ===");
console.log(response2.content);
