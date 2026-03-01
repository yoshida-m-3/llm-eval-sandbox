import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { embeddings, ragChain } from "./shared.js";

// --- 検証用ドキュメント: 経費精算規定（01-chunk-size.ts と同一）---

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

// --- チャンク分割（中サイズで固定）---

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 100,
  chunkOverlap: 20,
});
const chunks = await splitter.createDocuments([document]);

console.log("=".repeat(60));
console.log("取得件数(k)の調整 — Top-K 比較検証");
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

// --- k の値 ---

const kValues = [1, 2, 5, 10];

// --- 質問 ---
// Q1: 単一セクションの情報で回答可能 → k が小さくても正確に答えられるはず
// Q2: 複数セクションの情報を組み合わせる必要がある → k を増やすと改善が期待できる
// Q3: 特定の情報だけが必要 → k を増やしすぎると無関係な情報が混入し悪化する可能性
const questions = [
  "アルバイトスタッフが宿泊を伴う出張をすることはできますか？",
  "管理職が1泊2日で出張した場合、日当と宿泊費の合計上限はいくらですか？",
  "経費精算の締め日はいつですか？",
];

// --- 比較実行 ---

for (const question of questions) {
  console.log(`\n${"=".repeat(60)}`);
  console.log(`質問: ${question}`);
  console.log("=".repeat(60));

  for (const k of kValues) {
    const searchK = Math.min(k, chunks.length);
    const relevantDocs = await vectorStore.similaritySearch(question, searchK);

    console.log(`\n--- k=${k} (取得: ${relevantDocs.length}件) ---`);
    console.log("検索結果:");
    relevantDocs.forEach((doc, i) => {
      console.log(
        `  [${i}] (${doc.pageContent.length}文字) ${doc.pageContent.replace(/\n/g, " ").trim()}`,
      );
    });

    const context = relevantDocs.map((doc) => doc.pageContent).join("\n");
    const answer = await ragChain.invoke({ context, question });
    console.log(`回答: ${answer}`);
  }
}
