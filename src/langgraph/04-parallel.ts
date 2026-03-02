/**
 * 並列実行パターン (Map-Reduce) — Send で同一ノードを異なる入力で並列実行
 *
 * Send API で analyze ノードを 3 つの異なる観点（技術面・市場面・UX面）で
 * 並列実行し、reducer で結果を集約した後、aggregate ノードで統合レポートを作成する。
 *
 * フロー: START --(Send x3)--> analyze --> aggregate --> END
 *
 * 実行: pnpm graph:04
 */
import {
  Annotation,
  Send,
  StateGraph,
  START,
  END,
} from "@langchain/langgraph";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import { llm } from "../rag/shared.js";

// ============================================================
// ステート定義
// ============================================================
const ParallelAnnotation = Annotation.Root({
  topic: Annotation<string>,
  analyses: Annotation<string[]>({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),
  finalReport: Annotation<string>,
});

// analyze ノードが受け取るローカルステート
interface AnalyzeInput {
  topic: string;
  perspective: string;
  analyses: string[];
}

// ============================================================
// ノード定義
// ============================================================

// 各観点から分析を実行
async function analyzeNode(state: AnalyzeInput) {
  console.log(`[analyze] 分析中: ${state.perspective}`);

  const response = await llm.invoke([
    new SystemMessage(
      `あなたは${state.perspective}の専門家です。以下のトピックについて、${state.perspective}の観点から簡潔に分析してください（3〜5文程度）。`,
    ),
    new HumanMessage(state.topic),
  ]);

  const analysis = `【${state.perspective}】\n${response.content}`;
  console.log(`[analyze] ${state.perspective} 完了`);

  return { analyses: [analysis] };
}

// 全分析結果を統合してレポート作成
async function aggregateNode(state: typeof ParallelAnnotation.State) {
  console.log("\n[aggregate] 統合レポート作成中...");

  const allAnalyses = state.analyses.join("\n\n");

  const response = await llm.invoke([
    new SystemMessage(
      `あなたはビジネスアナリストです。以下の複数の観点からの分析結果を統合して、総合的なレポートを作成してください。`,
    ),
    new HumanMessage(
      `トピック: ${state.topic}\n\n分析結果:\n${allAnalyses}\n\n上記を踏まえた総合レポートを作成してください。`,
    ),
  ]);

  console.log("[aggregate] レポート作成完了");

  return { finalReport: String(response.content) };
}

// ============================================================
// 並列実行: Send で 3 つの観点を同時に analyze ノードへ
// ============================================================
const PERSPECTIVES = ["技術面", "市場面", "UX面"];

function fanOut(state: typeof ParallelAnnotation.State) {
  return PERSPECTIVES.map(
    (perspective) =>
      new Send("analyze", {
        topic: state.topic,
        perspective,
        analyses: [],
      }),
  );
}

// ============================================================
// グラフ構築
// ============================================================
const parallelGraph = new StateGraph(ParallelAnnotation)
  .addNode("analyze", analyzeNode)
  .addNode("aggregate", aggregateNode)
  .addConditionalEdges(START, fanOut)
  .addEdge("analyze", "aggregate")
  .addEdge("aggregate", END)
  .compile();

// ============================================================
// 実行
// ============================================================
console.log("=".repeat(60));
console.log("並列実行パターン (Map-Reduce)");
console.log("=".repeat(60));

const topic = "AIを活用したプログラミング学習アプリ";
console.log(`トピック: ${topic}\n`);

const result = await parallelGraph.invoke(
  { topic },
  { recursionLimit: 25 },
);

console.log("\n" + "=".repeat(60));
console.log("各観点の分析結果");
console.log("=".repeat(60));
for (const analysis of result.analyses) {
  console.log(`\n${analysis}`);
}

console.log("\n" + "=".repeat(60));
console.log("統合レポート");
console.log("=".repeat(60));
console.log(result.finalReport);
