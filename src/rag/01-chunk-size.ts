import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { embeddings, ragChain } from "./shared.js";

// --- 検証用ドキュメント: 経費精算規定 ---
// 「原則 NG → ただし例外あり」のパターンを含む。
// 小さいチャンクだと原則と例外が分断され、不完全な回答になることを検証する。

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

// --- チャンクサイズ・オーバーラップ調整の比較 ---

const configs = [
  { label: "小 (50/10)", chunkSize: 50, chunkOverlap: 10 },
  { label: "中 (100/20)", chunkSize: 100, chunkOverlap: 20 },
  { label: "大 (200/40)", chunkSize: 200, chunkOverlap: 40 },
];

// Q1: 原則NGだが例外条件がある質問 → 小チャンクだと例外が欠落しやすい
// Q2: 複数セクションの情報を組み合わせる質問 → 小チャンクだと片方しか取得できない
const questions = [
  "アルバイトスタッフが宿泊を伴う出張をすることはできますか？",
  "管理職が1泊2日で出張した場合、日当と宿泊費の合計上限はいくらですか？",
];

for (const config of configs) {
  console.log(`\n${"=".repeat(60)}`);
  console.log(`チャンク設定: ${config.label}`);
  console.log(
    `  chunkSize=${config.chunkSize}, chunkOverlap=${config.chunkOverlap}`,
  );
  console.log("=".repeat(60));

  // 1. チャンク分割
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: config.chunkSize,
    chunkOverlap: config.chunkOverlap,
  });
  const chunks = await splitter.createDocuments([document]);

  console.log(`\n--- 分割結果: ${chunks.length} チャンク ---`);
  chunks.forEach((chunk, i) => {
    console.log(
      `  [${i}] (${chunk.pageContent.length}文字) ${chunk.pageContent.trim()}`,
    );
  });

  for (const question of questions) {
    // 2. ベクトルストアに格納 & 検索
    const vectorStore = await MemoryVectorStore.fromDocuments(
      chunks,
      embeddings,
    );
    const relevantDocs = await vectorStore.similaritySearch(question, 2);

    console.log(`\n--- 質問: ${question} ---`);
    console.log(`検索結果 (k=2):`);
    relevantDocs.forEach((doc, i) => {
      console.log(`  [${i}] ${doc.pageContent.trim()}`);
    });

    // 3. RAG で回答生成
    const context = relevantDocs.map((doc) => doc.pageContent).join("\n");
    const answer = await ragChain.invoke({ context, question });

    console.log(`回答: ${answer}`);
  }
}
