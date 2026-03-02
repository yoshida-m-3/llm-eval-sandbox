/**
 * Handoff パターン — エージェント間の直接ハンドオフ
 *
 * Supervisor なしでエージェント同士が直接制御を受け渡すパターン。
 * Triage ノードが初期分類を行い、各エージェントが Command で
 * 終了またはハンドオフする。
 *
 * フロー: START → triage → tech_support / billing_support → END
 *
 * 実行: pnpm graph:03
 */
import {
  Annotation,
  Command,
  MessagesAnnotation,
  StateGraph,
  START,
  END,
} from "@langchain/langgraph";
import { z } from "zod";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { llm } from "../rag/shared.js";

// ============================================================
// ステート定義
// ============================================================
const HandoffAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  currentAgent: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => "triage",
  }),
});

// ============================================================
// ノード定義
// ============================================================
const DEPARTMENTS = ["tech_support", "billing_support"] as const;

const triageSchema = z.object({
  department: z.enum(DEPARTMENTS),
});

async function triageNode(
  state: typeof HandoffAnnotation.State,
): Promise<Command> {
  console.log("\n[triage] 問い合わせを分類中...");

  const structured = llm.withStructuredOutput(triageSchema);

  try {
    const result = await structured.invoke([
      new SystemMessage(
        `あなたはカスタマーサポートの受付です。問い合わせを以下の部門に振り分けてください:
- tech_support: 技術的な問題（エラー、設定、使い方など）
- billing_support: 請求・支払いに関する問題（料金、プラン変更、返金など）

必ず JSON で {"department": "..."} の形式で回答してください。`,
      ),
      ...state.messages,
    ]);

    console.log(`[triage] 振り分け先: ${result.department}`);

    return new Command({
      goto: result.department,
      update: { currentAgent: result.department },
    });
  } catch (e) {
    // フォールバック: キーワードで振り分け
    console.log("[triage] 構造化出力失敗、キーワードベースで振り分け");
    const content = state.messages
      .map((m) => String(m.content))
      .join(" ")
      .toLowerCase();

    const isBilling =
      content.includes("料金") ||
      content.includes("請求") ||
      content.includes("支払") ||
      content.includes("プラン") ||
      content.includes("返金");

    const dept = isBilling ? "billing_support" : "tech_support";
    console.log(`[triage] (フォールバック) → ${dept}`);

    return new Command({
      goto: dept,
      update: { currentAgent: dept },
    });
  }
}

async function techSupportNode(
  state: typeof HandoffAnnotation.State,
): Promise<Command> {
  console.log("[tech_support] 技術サポート対応中...");

  const response = await llm.invoke([
    new SystemMessage(
      `あなたは技術サポート担当です。ユーザーの技術的な問い合わせに対して、
具体的な解決手順を提案してください。`,
    ),
    ...state.messages,
  ]);

  console.log("[tech_support] 対応完了");

  return new Command({
    goto: "__end__",
    update: {
      messages: [response],
      currentAgent: "tech_support",
    },
  });
}

async function billingSupportNode(
  state: typeof HandoffAnnotation.State,
): Promise<Command> {
  console.log("[billing_support] 請求サポート対応中...");

  const response = await llm.invoke([
    new SystemMessage(
      `あなたは請求・課金サポート担当です。ユーザーの請求に関する問い合わせに対して、
丁寧に回答してください。`,
    ),
    ...state.messages,
  ]);

  console.log("[billing_support] 対応完了");

  return new Command({
    goto: "__end__",
    update: {
      messages: [response],
      currentAgent: "billing_support",
    },
  });
}

// ============================================================
// グラフ構築
// ============================================================
const handoffGraph = new StateGraph(HandoffAnnotation)
  .addNode("triage", triageNode, {
    ends: ["tech_support", "billing_support"],
  })
  .addNode("tech_support", techSupportNode, {
    ends: ["__end__"],
  })
  .addNode("billing_support", billingSupportNode, {
    ends: ["__end__"],
  })
  .addEdge(START, "triage")
  .compile();

// ============================================================
// 実行: 2パターンでテスト
// ============================================================
const testCases = [
  {
    label: "技術系の問い合わせ",
    input: "アプリがログイン画面でエラーコード500を表示します。どうすれば直りますか？",
  },
  {
    label: "請求系の問い合わせ",
    input: "先月の請求金額が通常より高いのですが、明細を確認してもらえますか？",
  },
];

for (const testCase of testCases) {
  console.log("\n" + "=".repeat(60));
  console.log(`テスト: ${testCase.label}`);
  console.log("=".repeat(60));
  console.log(`入力: ${testCase.input}`);

  const result = await handoffGraph.invoke(
    {
      messages: [new HumanMessage(testCase.input)],
    },
    { recursionLimit: 25 },
  );

  console.log(`\n対応エージェント: ${result.currentAgent}`);
  const lastMessage = result.messages[result.messages.length - 1];
  console.log(`回答:\n${lastMessage.content}`);
}
