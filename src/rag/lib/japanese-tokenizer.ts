import { tokenize } from "kuromojin";

// 助詞・助動詞・記号はBM25のノイズになるため除去
const STOP_POS = new Set(["助詞", "助動詞", "記号"]);

export interface TokenizeResult {
  /** フィルタ後のトークン（surface_form） */
  tokens: string[];
  /** フィルタ前の全形態素情報 */
  allTokens: { surface_form: string; pos: string; basic_form: string }[];
}

/**
 * 日本語テキストを kuromoji で形態素解析し、助詞・助動詞を除去したトークン列を返す。
 */
export async function tokenizeJapanese(
  text: string,
): Promise<TokenizeResult> {
  const rawTokens = await tokenize(text);

  const allTokens = rawTokens.map((t) => ({
    surface_form: t.surface_form,
    pos: t.pos,
    basic_form: t.basic_form,
  }));

  const tokens = rawTokens
    .filter((t) => !STOP_POS.has(t.pos))
    .map((t) => t.surface_form)
    .filter((s) => s.trim().length > 0);

  return { tokens, allTokens };
}
