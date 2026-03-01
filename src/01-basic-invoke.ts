import { ChatOllama } from "@langchain/ollama";
import { HumanMessage } from "@langchain/core/messages";

const model = new ChatOllama({
  model: "llama3.2:3b",
  temperature: 0,
});

const response = await model.invoke([
  new HumanMessage("TypeScriptを1文で説明してください。日本語で回答してください。"),
]);

console.log("Response:", response.content);
