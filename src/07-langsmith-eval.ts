/**
 * LangSmith 評価・デバッグ実践
 *
 * LangSmith の evaluate() API を使い、RAG パイプラインを体系的に評価する。
 * 2 つの異なるチャンク戦略で実験を実行し、LangSmith の比較ビューで差分を確認する。
 *
 * 4 つのセクションで構成:
 * 1. データセット作成 — テスト質問と参照回答を LangSmith に登録
 * 2. カスタム評価器 — LLM 不要の決定的ヒューリスティクス評価
 * 3. evaluate() 実行 — チャンクサイズ 200 / k=3 で実験
 * 4. 戦略比較 — チャンクサイズ 100 / k=5 で 2 回目の実験、比較ビューで差分確認
 *
 * 実行: pnpm eval:07
 */
import { Client } from "langsmith";
import { evaluate, type EvaluationResult } from "langsmith/evaluation";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { embeddings, ragChain } from "./rag/shared.js";
import { expensePolicy } from "./rag/documents/expense-policy.js";

const client = new Client();

// ============================================================
// 1. データセット作成
// ============================================================
console.log("=".repeat(60));
console.log("1. データセット作成");
console.log("=".repeat(60));

const datasetName = "expense-policy-qa";

// テスト質問と参照回答（01-chunk-size.ts と同じ質問セット + 締め日の質問）
const examples = [
  {
    inputs: {
      question:
        "アルバイトスタッフが宿泊を伴う出張をすることはできますか？",
    },
    outputs: {
      answer:
        "原則として認められていませんが、事前に部長の承認を得た場合に限り、1泊8,000円を上限として認められます。",
      keywords: ["部長", "承認", "8,000"],
    },
  },
  {
    inputs: {
      question:
        "管理職が1泊2日で出張した場合、日当と宿泊費の合計上限はいくらですか？",
    },
    outputs: {
      answer:
        "日当5,000円 + 宿泊費15,000円 = 合計20,000円が上限です。",
      keywords: ["5,000", "15,000", "20,000"],
    },
  },
  {
    inputs: {
      question: "経費精算の締め日はいつですか？",
    },
    outputs: {
      answer: "毎月25日です。",
      keywords: ["25"],
    },
  },
];

// 重複実行対策: 既存データセットがあればそれを使う
let dataset;
try {
  dataset = await client.createDataset(datasetName, {
    description: "経費精算規定に関するQAデータセット（RAG評価用）",
  });
  console.log(`データセット '${datasetName}' を新規作成しました`);

  // サンプル登録
  await client.createExamples({
    inputs: examples.map((e) => e.inputs),
    outputs: examples.map((e) => e.outputs),
    datasetId: dataset.id,
  });
  console.log(`${examples.length} 件のサンプルを登録しました\n`);
} catch {
  // 既存データセットを取得
  dataset = await client.readDataset({ datasetName });
  console.log(
    `データセット '${datasetName}' は既に存在します（再利用）\n`,
  );
}

// ============================================================
// 2. カスタム評価器
// ============================================================
console.log("=".repeat(60));
console.log("2. カスタム評価器の定義");
console.log("=".repeat(60));

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- evaluate() の evaluator 型に合わせる
type EvalArgs = { outputs?: Record<string, any>; referenceOutputs?: Record<string, any>; [k: string]: any };

/** キーワードベースの正確性チェック */
function answerCorrectness({ outputs, referenceOutputs }: EvalArgs): EvaluationResult {
  const answer = String(outputs?.answer ?? "");
  const keywords = (referenceOutputs?.keywords ?? []) as string[];
  const matched = keywords.filter((kw: string) => answer.includes(kw));
  const score = keywords.length > 0 ? matched.length / keywords.length : 0;
  return {
    key: "answer_correctness",
    score,
    comment: `キーワード一致: ${matched.join(", ")} (${matched.length}/${keywords.length})`,
  };
}

/** 回答の簡潔さチェック（200文字以内なら満点） */
function conciseness({ outputs }: EvalArgs): EvaluationResult {
  const answer = String(outputs?.answer ?? "");
  const length = answer.length;
  const score = length <= 200 ? 1.0 : Math.max(0, 1.0 - (length - 200) / 300);
  return {
    key: "conciseness",
    score,
    comment: `回答長: ${length}文字`,
  };
}

console.log("- answerCorrectness: 参照回答のキーワード含有率");
console.log("- conciseness: 回答の簡潔さ（200文字以内で満点）\n");

// ============================================================
// 3. evaluate() 実行 — チャンクサイズ 200 / k=3
// ============================================================
console.log("=".repeat(60));
console.log("3. 実験1: チャンクサイズ 200, k=3");
console.log("=".repeat(60));

/** RAG ターゲット関数を生成 */
async function createRagTarget(chunkSize: number, chunkOverlap: number, k: number) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap,
  });
  const chunks = await splitter.createDocuments([expensePolicy]);
  const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);

  return async (inputs: Record<string, string>) => {
    const question = inputs.question;
    const relevantDocs = await vectorStore.similaritySearch(question, k);
    const context = relevantDocs.map((doc) => doc.pageContent).join("\n");
    const answer = await ragChain.invoke({ context, question });
    return { answer };
  };
}

const target1 = await createRagTarget(200, 40, 3);

const results1 = await evaluate(target1, {
  data: datasetName,
  evaluators: [answerCorrectness, conciseness],
  experimentPrefix: "rag-chunk200-k3",
});

console.log("実験1 完了");
console.log(`実験名: ${results1.experimentName}`);
console.log(`結果数: ${results1.length}\n`);

// ============================================================
// 4. 戦略比較 — チャンクサイズ 100 / k=5
// ============================================================
console.log("=".repeat(60));
console.log("4. 実験2: チャンクサイズ 100, k=5（比較用）");
console.log("=".repeat(60));

const target2 = await createRagTarget(100, 20, 5);

const results2 = await evaluate(target2, {
  data: datasetName,
  evaluators: [answerCorrectness, conciseness],
  experimentPrefix: "rag-chunk100-k5",
});

console.log("実験2 完了");
console.log(`実験名: ${results2.experimentName}`);
console.log(`結果数: ${results2.length}\n`);

// ============================================================
// 完了
// ============================================================
console.log("=".repeat(60));
console.log("全実験完了 — LangSmith で 2 つの実験を比較してください");
console.log("https://smith.langchain.com");
console.log("=".repeat(60));
