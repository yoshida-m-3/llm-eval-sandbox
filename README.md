# llm-eval-sandbox

LangChain エコシステム（Langシリーズ）を段階的に学習・検証するためのプロジェクト。

## マイルストーン

### Milestone 1: LangChain + LangSmith
LLMアプリケーション構築の基盤となるLangChainの基本概念（プロンプト、チェーン、ツール連携）を習得し、LangSmithによるトレース・デバッグ環境を整える。

- [ ] LangChain の環境構築
- [ ] 基本的なチェーンの構築（プロンプトテンプレート、LLM呼び出し）
- [ ] ツール連携とRetrieval-Augmented Generation (RAG) の実装
- [ ] LangSmith のセットアップとトレース可視化
- [ ] LangSmith による評価・デバッグの実践

### Milestone 2: LangGraph
ステートフルなマルチエージェントワークフローをグラフ構造で構築する。

- [ ] LangGraph の環境構築
- [ ] 基本的なグラフワークフローの構築
- [ ] ステート管理と条件分岐の実装
- [ ] マルチエージェントパターンの実装

### Milestone 3: LangServe
構築したLLMアプリケーションをREST APIとしてデプロイする。

- [ ] LangServe の環境構築
- [ ] チェーン/エージェントのAPI化
- [ ] Playground UIの活用
- [ ] デプロイと動作検証

### Milestone 4: LangFlow
ビジュアルUIでフローを構築し、ノーコード/ローコード開発を検証する。

- [ ] LangFlow のセットアップ
- [ ] ビジュアルフローの構築
- [ ] カスタムコンポーネントの作成
- [ ] 既存実装との比較検証

## 技術スタック

- Python
- LangChain / LangSmith / LangGraph / LangServe / LangFlow
