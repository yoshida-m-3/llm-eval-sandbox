import { BaseRetriever } from "@langchain/core/retrievers";
import type { Document } from "@langchain/core/documents";
import { tokenizeJapanese } from "./japanese-tokenizer.js";

interface TokenizedDoc {
  tokens: string[];
  document: Document;
}

/**
 * 日本語形態素解析を組み込んだ BM25 Retriever。
 *
 * @langchain/community の BM25Retriever は内部で空白分割 + /\w+/g（ASCII のみ）を
 * 使用しており、日本語では機能しない。このクラスは kuromoji による形態素解析で
 * トークン化し、Okapi BM25 スコアリングを自前で実装する。
 */
export class JapaneseBM25Retriever extends BaseRetriever {
  lc_namespace = ["custom", "retrievers", "japanese_bm25"];

  private tokenizedDocs: TokenizedDoc[] = [];
  private avgDocLength = 0;
  private k: number;
  private k1: number;
  private b: number;

  private constructor(
    private docs: Document[],
    options: { k: number; k1?: number; b?: number },
  ) {
    super();
    this.k = options.k;
    this.k1 = options.k1 ?? 1.2;
    this.b = options.b ?? 0.75;
  }

  static async fromDocuments(
    documents: Document[],
    options: { k: number; k1?: number; b?: number },
  ): Promise<JapaneseBM25Retriever> {
    const retriever = new JapaneseBM25Retriever(documents, options);
    await retriever.buildIndex();
    return retriever;
  }

  private async buildIndex(): Promise<void> {
    this.tokenizedDocs = await Promise.all(
      this.docs.map(async (doc) => ({
        tokens: (await tokenizeJapanese(doc.pageContent)).tokens,
        document: doc,
      })),
    );

    const totalTokens = this.tokenizedDocs.reduce(
      (sum, d) => sum + d.tokens.length,
      0,
    );
    this.avgDocLength = totalTokens / this.tokenizedDocs.length;
  }

  async _getRelevantDocuments(query: string): Promise<Document[]> {
    const { tokens: queryTokens } = await tokenizeJapanese(query);
    const N = this.tokenizedDocs.length;

    // IDF: log((N - df + 0.5) / (df + 0.5) + 1)
    const idf = new Map<string, number>();
    for (const term of queryTokens) {
      if (idf.has(term)) continue;
      const df = this.tokenizedDocs.filter((d) =>
        d.tokens.includes(term),
      ).length;
      idf.set(term, Math.log((N - df + 0.5) / (df + 0.5) + 1));
    }

    // BM25 スコア計算
    const scored = this.tokenizedDocs.map(({ tokens, document }) => {
      let score = 0;
      const dl = tokens.length;

      for (const term of queryTokens) {
        const tf = tokens.filter((t) => t === term).length;
        const idfVal = idf.get(term) ?? 0;
        score +=
          idfVal *
          ((tf * (this.k1 + 1)) /
            (tf + this.k1 * (1 - this.b + (this.b * dl) / this.avgDocLength)));
      }

      return { score, document };
    });

    return scored
      .sort((a, b) => b.score - a.score)
      .slice(0, this.k)
      .map((item) => item.document);
  }
}
