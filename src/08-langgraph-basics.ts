/**
 * LangGraph 基本グラフワークフロー
 *
 * LangGraph のコア概念を 3 つのセクションで段階的に学ぶ:
 * 1. 最小グラフ — START → ノード → END の直線フロー（LLM不使用）
 * 2. 条件分岐グラフ — addConditionalEdges で動的ルーティング
 * 3. LLM統合グラフ — ChatOllama を組み込んだ品質チェック付きワークフロー
 *
 * 実行: pnpm graph:08
 */
import {
  Annotation,
  StateGraph,
  START,
  END,
  MessagesAnnotation,
} from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import type { AIMessage } from "@langchain/core/messages";
import { llm } from "./rag/shared.js";

// ============================================================
// 1. 最小グラフ — START → greet → shout → END
// ============================================================
console.log("=".repeat(60));
console.log("1. 最小グラフ（直線フロー）");
console.log("=".repeat(60));

// ステート定義: Annotation.Root() でグラフ全体の状態を型安全に定義
const GreetingAnnotation = Annotation.Root({
  name: Annotation<string>,
  greeting: Annotation<string>,
  uppercased: Annotation<string>,
});

// ノード関数: ステートを受け取り、更新したいフィールドだけを返す
function greetNode(state: typeof GreetingAnnotation.State) {
  return { greeting: `こんにちは、${state.name}さん！` };
}

function shoutNode(state: typeof GreetingAnnotation.State) {
  return { uppercased: state.greeting.toUpperCase() };
}

// グラフ構築: addNode → addEdge → compile → invoke
const greetingGraph = new StateGraph(GreetingAnnotation)
  .addNode("greet", greetNode)
  .addNode("shout", shoutNode)
  .addEdge(START, "greet")
  .addEdge("greet", "shout")
  .addEdge("shout", END)
  .compile();

const greetingResult = await greetingGraph.invoke({ name: "taro" });
console.log("入力:", { name: "taro" });
console.log("結果:", greetingResult);
console.log();

// ============================================================
// 2. 条件分岐グラフ — スコアに応じてルーティング
// ============================================================
console.log("=".repeat(60));
console.log("2. 条件分岐グラフ（動的ルーティング）");
console.log("=".repeat(60));

const ScoreAnnotation = Annotation.Root({
  score: Annotation<number>,
  result: Annotation<string>,
});

function evaluateNode(state: typeof ScoreAnnotation.State) {
  // スコアをそのまま通す（実際のアプリではここで計算や取得を行う）
  return {};
}

// ルーター関数: ステートを見て次のノード名を返す
function scoreRouter(state: typeof ScoreAnnotation.State): string {
  if (state.score >= 80) return "excellent";
  if (state.score >= 50) return "good";
  return "needsWork";
}

function excellentNode(_state: typeof ScoreAnnotation.State) {
  return { result: "素晴らしい！合格です" };
}

function goodNode(_state: typeof ScoreAnnotation.State) {
  return { result: "まずまずです。もう少し頑張りましょう" };
}

function needsWorkNode(_state: typeof ScoreAnnotation.State) {
  return { result: "要改善。基礎から復習しましょう" };
}

const scoreGraph = new StateGraph(ScoreAnnotation)
  .addNode("evaluate", evaluateNode)
  .addNode("excellent", excellentNode)
  .addNode("good", goodNode)
  .addNode("needsWork", needsWorkNode)
  .addEdge(START, "evaluate")
  // addConditionalEdges: ルーター関数の戻り値でノードを選択
  .addConditionalEdges("evaluate", scoreRouter, {
    excellent: "excellent",
    good: "good",
    needsWork: "needsWork",
  })
  .addEdge("excellent", END)
  .addEdge("good", END)
  .addEdge("needsWork", END)
  .compile();

// 3パターンをテスト
for (const score of [90, 65, 30]) {
  const result = await scoreGraph.invoke({ score });
  console.log(`スコア ${score} → ${result.result}`);
}
console.log();

// ============================================================
// 3. LLM統合グラフ — 品質チェック付き応答生成
// ============================================================
console.log("=".repeat(60));
console.log("3. LLM統合グラフ（品質チェック付き）");
console.log("=".repeat(60));

// MessagesAnnotation を拡張してリトライカウンタを追加
const ChatAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  retryCount: Annotation<number>({
    reducer: (a, b) => a + b,
    default: () => 0,
  }),
  maxRetries: Annotation<number>,
});

// ノード: LLM で応答を生成
async function generateNode(state: typeof ChatAnnotation.State) {
  const response = await llm.invoke(state.messages);
  return { messages: [response] };
}

// ノード: 品質チェック（応答が短すぎないか確認）
function qualityCheckRouter(state: typeof ChatAnnotation.State): string {
  const lastMessage = state.messages[state.messages.length - 1];
  const content = String(lastMessage.content);

  // 応答が20文字未満なら品質不足とみなす
  if (content.length < 20 && state.retryCount < state.maxRetries) {
    console.log(
      `  [品質チェック] 応答が短すぎます（${content.length}文字）→ 再生成`,
    );
    return "retry";
  }
  console.log(`  [品質チェック] OK（${content.length}文字）`);
  return "accept";
}

// ノード: リトライ時にフィードバックを追加
function retryNode(state: typeof ChatAnnotation.State) {
  return {
    messages: [
      new HumanMessage(
        "もう少し詳しく、具体的に回答してください。最低でも50文字以上でお願いします。",
      ),
    ],
    retryCount: 1,
  };
}

const chatGraph = new StateGraph(ChatAnnotation)
  .addNode("generate", generateNode)
  .addNode("retry", retryNode)
  .addEdge(START, "generate")
  .addConditionalEdges("generate", qualityCheckRouter, {
    accept: END,
    retry: "retry",
  })
  .addEdge("retry", "generate")
  .compile();

const chatResult = await chatGraph.invoke({
  messages: [new HumanMessage("TypeScriptの型システムの利点を教えてください")],
  maxRetries: 2,
});

const lastMsg = chatResult.messages[
  chatResult.messages.length - 1
] as AIMessage;
console.log(`\nメッセージ数: ${chatResult.messages.length}`);
console.log(`リトライ回数: ${chatResult.retryCount}`);
console.log(`最終応答:\n${lastMsg.content}`);

// ============================================================
// 完了
// ============================================================
console.log();
console.log("=".repeat(60));
console.log("全セクション完了");
console.log("=".repeat(60));
