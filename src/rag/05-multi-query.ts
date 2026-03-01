import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import type { Document } from "@langchain/core/documents";
import { embeddings, llm, ragChain } from "./shared.js";
import { employeeGuide } from "./documents/employee-guide.js";

// --- チャンク分割 ---

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 100,
  chunkOverlap: 20,
});
const chunks = await splitter.createDocuments([employeeGuide]);

console.log("=".repeat(60));
console.log("マルチクエリ — 単一クエリとの検索ヒット率比較");
console.log("=".repeat(60));

console.log(`\nチャンク設定: chunkSize=100, chunkOverlap=20`);
console.log(`チャンク数: ${chunks.length}`);
chunks.forEach((chunk, i) => {
  console.log(
    `  [${i}] (${chunk.pageContent.length}文字) ${chunk.pageContent.replace(/\n/g, " ").trim()}`,
  );
});

// --- ベクトルストア構築 ---

const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);

// --- マルチクエリ生成プロンプト ---

const multiQueryPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `あなたは検索クエリを生成するアシスタントです。
ユーザーの質問に対して、異なる視点や言い回しで3つの検索クエリを生成してください。
各クエリは改行で区切って出力してください。余計な説明や番号は不要です。`,
  ],
  ["human", "{question}"],
]);

const multiQueryChain = multiQueryPrompt
  .pipe(llm)
  .pipe(new StringOutputParser());

// --- ヘルパー関数 ---

/** LLM で質問を3パターンに言い換え */
async function generateQueries(question: string): Promise<string[]> {
  const raw = await multiQueryChain.invoke({ question });
  const queries = raw
    .split("\n")
    .map((line) => line.replace(/^\d+[\.\)]\s*/, "").trim())
    .filter((line) => line.length > 0);
  return queries;
}

/** 検索結果を重複排除して統合（pageContent が同一なら重複とみなす） */
function deduplicateDocs(docs: Document[]): Document[] {
  const seen = new Set<string>();
  return docs.filter((doc) => {
    if (seen.has(doc.pageContent)) return false;
    seen.add(doc.pageContent);
    return true;
  });
}

// --- 質問 ---
// Q1: 語彙ミスマッチ — 質問は「リモートで働く」だが文書では「テレワーク」「在宅勤務」
// Q2: 複数セクション横断 — 「バイト」の権利が休暇・福利厚生・研修に分散し語彙も異なる
// Q3: 語彙ミスマッチ+曖昧 — 「PC」「パソコン」「端末」が混在、「辞めるとき」→「退職」
const questions = [
  "自宅からリモートで働くにはどうすればいいですか？",
  "バイトでも使える福利厚生や休暇の制度はありますか？",
  "会社を辞めるときにPCはどうすればいいですか？",
];

const k = 3;

// --- 比較実行 ---

for (const question of questions) {
  console.log(`\n${"=".repeat(60)}`);
  console.log(`質問: ${question}`);
  console.log("=".repeat(60));

  // --- 単一クエリ検索 ---
  console.log(`\n【単一クエリ検索】(k=${k})`);
  const singleDocs = await vectorStore.similaritySearch(question, k);
  console.log(`取得チャンク数: ${singleDocs.length}`);
  singleDocs.forEach((doc, i) => {
    console.log(
      `  [${i}] (${doc.pageContent.length}文字) ${doc.pageContent.replace(/\n/g, " ").trim()}`,
    );
  });

  const singleContext = singleDocs.map((doc) => doc.pageContent).join("\n");
  const singleAnswer = await ragChain.invoke({
    context: singleContext,
    question,
  });
  console.log(`回答: ${singleAnswer}`);

  // --- マルチクエリ検索 ---
  console.log(`\n【マルチクエリ検索】(各クエリ k=${k})`);

  const generatedQueries = await generateQueries(question);
  console.log("生成されたクエリ:");
  generatedQueries.forEach((q, i) => {
    console.log(`  ${i + 1}. ${q}`);
  });

  // 元の質問 + 生成クエリ すべてで検索
  const allQueries = [question, ...generatedQueries];
  const allDocs: Document[] = [];

  for (const query of allQueries) {
    const docs = await vectorStore.similaritySearch(query, k);
    allDocs.push(...docs);
  }

  const mergedDocs = deduplicateDocs(allDocs);
  console.log(
    `\n統合結果: ${allDocs.length}件 → 重複排除後 ${mergedDocs.length}件`,
  );
  mergedDocs.forEach((doc, i) => {
    console.log(
      `  [${i}] (${doc.pageContent.length}文字) ${doc.pageContent.replace(/\n/g, " ").trim()}`,
    );
  });

  const multiContext = mergedDocs.map((doc) => doc.pageContent).join("\n");
  const multiAnswer = await ragChain.invoke({
    context: multiContext,
    question,
  });
  console.log(`回答: ${multiAnswer}`);

  // --- 比較サマリー ---
  const singleSet = new Set(singleDocs.map((d) => d.pageContent));
  const multiSet = new Set(mergedDocs.map((d) => d.pageContent));
  const newChunks = mergedDocs.filter((d) => !singleSet.has(d.pageContent));

  console.log(`\n【比較】`);
  console.log(`  単一クエリ: ${singleSet.size}件`);
  console.log(`  マルチクエリ: ${multiSet.size}件`);
  console.log(`  マルチクエリで新たに取得: ${newChunks.length}件`);
  if (newChunks.length > 0) {
    newChunks.forEach((doc, i) => {
      console.log(
        `    + [${i}] ${doc.pageContent.replace(/\n/g, " ").trim()}`,
      );
    });
  }
}
