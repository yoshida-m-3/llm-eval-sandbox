"""
日本語 RAG プロンプトテンプレート — LangFlow カスタムコンポーネント

既存の TypeScript 実装（src/rag/shared.ts）で使用している
日本語 RAG プロンプトを LangFlow コンポーネントとして再実装する。

特徴:
- コンテキストと質問を受け取り、RAG 用プロンプトを生成
- system_instruction をカスタマイズ可能
- max_context_length でコンテキストの長さを制限可能

LangFlow UI で Retriever と LLM の間に配置して使用する。
"""

from langflow.custom import Component
from langflow.io import MessageTextInput, MultilineInput, IntInput, Output
from langflow.schema.message import Message


class CustomRAGPrompt(Component):
    display_name = "Japanese RAG Prompt"
    description = "日本語 RAG 用のプロンプトテンプレート（コンテキスト+質問→回答指示）"
    icon = "file-text"
    name = "CustomRAGPrompt"

    inputs = [
        MessageTextInput(
            name="context",
            display_name="Context",
            info="検索で取得したコンテキスト情報",
            required=True,
        ),
        MessageTextInput(
            name="question",
            display_name="Question",
            info="ユーザーの質問",
            required=True,
        ),
        MultilineInput(
            name="system_instruction",
            display_name="System Instruction",
            info="システムプロンプトのカスタマイズ",
            value="以下のコンテキスト情報のみを使って質問に回答してください。\nコンテキストに含まれない情報については「情報が見つかりません」と回答してください。",
        ),
        IntInput(
            name="max_context_length",
            display_name="Max Context Length",
            info="コンテキストの最大文字数（0=無制限）",
            value=0,
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="RAG Prompt",
            name="rag_prompt",
            method="build_prompt",
        ),
    ]

    def build_prompt(self) -> Message:
        context = self.context
        question = self.question

        # コンテキストの長さ制限
        if self.max_context_length > 0 and len(context) > self.max_context_length:
            context = context[: self.max_context_length] + "\n...(省略)"

        prompt = (
            f"{self.system_instruction}\n\n"
            f"コンテキスト:\n{context}\n\n"
            f"質問: {question}"
        )

        self.status = (
            f"プロンプト生成完了: コンテキスト {len(context)}文字, "
            f"質問 {len(question)}文字, 合計 {len(prompt)}文字"
        )
        return Message(text=prompt)
