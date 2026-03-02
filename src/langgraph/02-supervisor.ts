/**
 * Supervisor パターン — 中央の Supervisor がワーカーを選択・指示
 *
 * Supervisor が withStructuredOutput で次のワーカーを決定し、
 * Command で制御を移譲する。各ワーカーは処理後 Command で
 * Supervisor に制御を戻す。
 *
 * フロー: START → supervisor → researcher → supervisor → writer → supervisor → END
 *
 * 実行: pnpm graph:02
 */
import {
  Annotation,
  Command,
  MessagesAnnotation,
  StateGraph,
  START,
} from "@langchain/langgraph";
import { z } from "zod";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { llm } from "../rag/shared.js";

// ============================================================
// ステート定義
// ============================================================
const SupervisorAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
});

// ============================================================
// ノード定義
// ============================================================
const WORKERS = ["researcher", "writer", "FINISH"] as const;

const routerSchema = z.object({
  next: z.enum(WORKERS),
});

async function supervisorNode(
  state: typeof SupervisorAnnotation.State,
): Promise<Command> {
  console.log("\n[supervisor] ルーティング判断中...");

  const structured = llm.withStructuredOutput(routerSchema);

  try {
    const result = await structured.invoke([
      new SystemMessage(
        `あなたはタスクマネージャーです。以下のワーカーを管理しています: researcher, writer。
ユーザーの依頼に対して、次に作業すべきワーカーを選んでください。
- researcher: 情報の調査・収集が必要な場合
- writer: 調査結果をもとに文章を作成する場合
- FINISH: すべての作業が完了した場合

必ず JSON で {"next": "..."} の形式で回答してください。`,
      ),
      ...state.messages,
    ]);

    console.log(`[supervisor] 次のワーカー: ${result.next}`);

    if (result.next === "FINISH") {
      return new Command({ goto: "__end__" });
    }

    return new Command({ goto: result.next });
  } catch (e) {
    // フォールバック: メッセージ内容から決定的にルーティング
    console.log("[supervisor] 構造化出力失敗、フォールバックルーターを使用");
    return deterministicRouter(state);
  }
}

function deterministicRouter(
  state: typeof SupervisorAnnotation.State,
): Command {
  const msgs = state.messages;
  const hasResearch = msgs.some(
    (m) => m.getType() === "ai" && String(m.content).includes("[調査結果]"),
  );
  const hasArticle = msgs.some(
    (m) => m.getType() === "ai" && String(m.content).includes("[記事]"),
  );

  if (!hasResearch) {
    console.log("[supervisor] (フォールバック) → researcher");
    return new Command({ goto: "researcher" });
  }
  if (!hasArticle) {
    console.log("[supervisor] (フォールバック) → writer");
    return new Command({ goto: "writer" });
  }
  console.log("[supervisor] (フォールバック) → END");
  return new Command({ goto: "__end__" });
}

async function researcherNode(
  state: typeof SupervisorAnnotation.State,
): Promise<Command> {
  console.log("[researcher] 調査を実行中...");

  const response = await llm.invoke([
    new SystemMessage(
      "あなたは技術リサーチャーです。ユーザーの質問について要点を3つ調査してまとめてください。回答の冒頭に「[調査結果]」と付けてください。",
    ),
    ...state.messages,
  ]);

  console.log("[researcher] 調査完了");

  return new Command({
    goto: "supervisor",
    update: { messages: [response] },
  });
}

async function writerNode(
  state: typeof SupervisorAnnotation.State,
): Promise<Command> {
  console.log("[writer] 記事を作成中...");

  const response = await llm.invoke([
    new SystemMessage(
      "あなたはテクニカルライターです。これまでの調査結果をもとに、わかりやすい解説記事を作成してください。回答の冒頭に「[記事]」と付けてください。",
    ),
    ...state.messages,
  ]);

  console.log("[writer] 記事作成完了");

  return new Command({
    goto: "supervisor",
    update: { messages: [response] },
  });
}

// ============================================================
// グラフ構築
// ============================================================
const supervisorGraph = new StateGraph(SupervisorAnnotation)
  .addNode("supervisor", supervisorNode, {
    ends: ["researcher", "writer", "__end__"],
  })
  .addNode("researcher", researcherNode, {
    ends: ["supervisor"],
  })
  .addNode("writer", writerNode, {
    ends: ["supervisor"],
  })
  .addEdge(START, "supervisor")
  .compile();

// ============================================================
// 実行
// ============================================================
console.log("=".repeat(60));
console.log("Supervisor パターン");
console.log("=".repeat(60));

const result = await supervisorGraph.invoke(
  {
    messages: [
      new HumanMessage(
        "TypeScriptの型システムについて短い解説を作成してください",
      ),
    ],
  },
  { recursionLimit: 25 },
);

console.log("\n" + "=".repeat(60));
console.log("最終結果");
console.log("=".repeat(60));
console.log(`メッセージ数: ${result.messages.length}`);
const lastMessage = result.messages[result.messages.length - 1];
console.log(`\n最終メッセージ:\n${lastMessage.content}`);
