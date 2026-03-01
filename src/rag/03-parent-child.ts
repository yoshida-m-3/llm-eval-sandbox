import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { ParentDocumentRetriever } from "@langchain/classic/retrievers/parent_document";
import { InMemoryStore } from "@langchain/core/stores";
import { Document } from "@langchain/core/documents";
import { embeddings, ragChain } from "./shared.js";

// --- 検証用ドキュメント（01-chunk-size.ts と同一）---

const document = `
当社の経費精算規定について説明します。

【出張日当】
正社員の出張日当は1日あたり3,000円です。ただし、管理職の場合は1日あたり5,000円が支給されます。
アルバイトスタッフには出張日当は支給されません。

【交通費】
正社員は交通費の実費精算が可能です。
新幹線のグリーン車利用は管理職のみ認められています。
アルバイトスタッフの交通費は1日あたり上限1,500円までの実費精算となります。

【宿泊費】
正社員の宿泊費の上限は1泊12,000円です。管理職の場合は1泊15,000円まで認められています。
アルバイトスタッフの宿泊を伴う出張は原則として認められていません。ただし、事前に部長の承認を得た場合に限り、1泊8,000円を上限として認められます。

【懇親会費用】
正社員は1人あたり5,000円まで経費として申請できます。
アルバイトスタッフは懇親会費用の経費申請はできません。ただし、歓送迎会の場合は雇用形態に関わらず1人あたり3,000円まで申請可能です。

【申請方法】
経費精算の締め日は毎月25日です。
正社員は社内システムから申請してください。
アルバイトスタッフは専用の紙の申請書を使用してください。
領収書の原本添付が必須です。
`;

// --- 質問 ---
// Q1: 原則NGだが例外条件がある質問 → 小チャンクだと例外が欠落しやすい
// Q2: 複数セクションの情報を組み合わせる質問
const questions = [
  "アルバイトスタッフが宿泊を伴う出張をすることはできますか？",
  "管理職が1泊2日で出張した場合、日当と宿泊費の合計上限はいくらですか？",
];

// --- スプリッター設定 ---
// parentSplitter: 大きいチャンク（セクション単位に近い）
// childSplitter: 小さいチャンク（検索精度を高める）

const parentSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 0,
});
const childSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 50,
  chunkOverlap: 0,
});

console.log("=".repeat(60));
console.log("親子チャンク（Parent-Child Chunks）比較検証");
console.log("=".repeat(60));

// =====================================================================
// A) 親子チャンク（ParentDocumentRetriever）
//    小チャンクで検索 → ヒットしたチャンクの親（大チャンク）を返す
// =====================================================================

console.log("\n▶ A) 親子チャンク（ParentDocumentRetriever）");
console.log("  parentSplitter: 200文字, childSplitter: 50文字");

const vectorStore = new MemoryVectorStore(embeddings);
const byteStore = new InMemoryStore<Uint8Array>();

const retriever = new ParentDocumentRetriever({
  vectorstore: vectorStore,
  byteStore,
  parentSplitter,
  childSplitter,
  childK: 5,
  parentK: 2,
});

const docs = [new Document({ pageContent: document })];
await retriever.addDocuments(docs);

// 子チャンク（ベクトルストアに格納されたもの）を確認
const allChildDocs = await vectorStore.similaritySearch("", 100);
console.log(`\n  子チャンク数: ${allChildDocs.length}`);
allChildDocs.forEach((doc, i) => {
  console.log(
    `    [${i}] (${doc.pageContent.length}文字) ${doc.pageContent.replace(/\n/g, " ").trim()}`,
  );
});

// 親チャンクの確認
const parentChunks = await parentSplitter.createDocuments([document]);
console.log(`\n  親チャンク数: ${parentChunks.length}`);
parentChunks.forEach((doc, i) => {
  const preview = doc.pageContent.replace(/\n/g, " ").trim().substring(0, 80);
  console.log(`    [${i}] (${doc.pageContent.length}文字) ${preview}…`);
});

// =====================================================================
// B) 小チャンクのみ（ベースライン）
//    50文字チャンクで検索 → そのまま小チャンクを返す
// =====================================================================

console.log("\n▶ B) 小チャンクのみ（ベースライン: 50文字）");

const smallChunks = await childSplitter.createDocuments([document]);
const smallVectorStore = await MemoryVectorStore.fromDocuments(
  smallChunks,
  embeddings,
);
console.log(`  チャンク数: ${smallChunks.length}`);

// =====================================================================
// 比較実行
// =====================================================================

for (const question of questions) {
  console.log(`\n${"=".repeat(60)}`);
  console.log(`質問: ${question}`);
  console.log("=".repeat(60));

  // --- A) 親子チャンク ---
  console.log("\n--- A) 親子チャンク ---");
  const parentResults = await retriever.invoke(question);

  console.log(`検索結果 (${parentResults.length}件の親チャンク):`);
  parentResults.forEach((doc, i) => {
    console.log(
      `  [${i}] (${doc.pageContent.length}文字) ${doc.pageContent.replace(/\n/g, " ").trim()}`,
    );
  });

  const contextA = parentResults.map((doc) => doc.pageContent).join("\n");
  const answerA = await ragChain.invoke({ context: contextA, question });
  console.log(`回答: ${answerA}`);

  // --- B) 小チャンクのみ ---
  console.log("\n--- B) 小チャンクのみ（50文字）---");
  const smallResults = await smallVectorStore.similaritySearch(question, 2);

  console.log(`検索結果 (k=2):`);
  smallResults.forEach((doc, i) => {
    console.log(
      `  [${i}] (${doc.pageContent.length}文字) ${doc.pageContent.replace(/\n/g, " ").trim()}`,
    );
  });

  const contextB = smallResults.map((doc) => doc.pageContent).join("\n");
  const answerB = await ragChain.invoke({ context: contextB, question });
  console.log(`回答: ${answerB}`);
}
