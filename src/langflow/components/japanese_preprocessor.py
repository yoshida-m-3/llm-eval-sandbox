"""
日本語テキストプリプロセッサ — LangFlow カスタムコンポーネント

入力テキストに対して以下の正規化処理を行う:
- 全角英数字 → 半角英数字
- 半角カタカナ → 全角カタカナ
- 連続空白の圧縮
- 前後の空白トリム

LangFlow UI の「Custom Component」からこのファイルを読み込んで使用する。
フロー内で Chat Input と Prompt の間に配置し、入力テキストを正規化する。
"""

import re
import unicodedata

from langflow.custom import Component
from langflow.io import MessageTextInput, BoolInput, Output
from langflow.schema.message import Message


class JapanesePreprocessor(Component):
    display_name = "Japanese Preprocessor"
    description = "日本語テキストの正規化（全角→半角変換、空白圧縮など）を行うプリプロセッサ"
    icon = "languages"
    name = "JapanesePreprocessor"

    inputs = [
        MessageTextInput(
            name="input_text",
            display_name="Input Text",
            info="正規化する日本語テキスト",
            required=True,
        ),
        BoolInput(
            name="normalize_unicode",
            display_name="Unicode NFKC Normalize",
            info="NFKC 正規化を適用する（全角英数→半角、半角カナ→全角など）",
            value=True,
        ),
        BoolInput(
            name="compress_whitespace",
            display_name="Compress Whitespace",
            info="連続する空白文字を1つに圧縮する",
            value=True,
        ),
        BoolInput(
            name="strip_text",
            display_name="Strip Whitespace",
            info="先頭・末尾の空白を除去する",
            value=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Processed Text",
            name="processed_text",
            method="preprocess",
        ),
    ]

    def preprocess(self) -> Message:
        text = self.input_text

        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        if self.compress_whitespace:
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)

        if self.strip_text:
            text = text.strip()

        self.status = f"処理完了: {len(self.input_text)}文字 → {len(text)}文字"
        return Message(text=text)
