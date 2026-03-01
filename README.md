# llm-eval-sandbox

LangChain エコシステム（Langシリーズ）を段階的に学習・検証するためのプロジェクト。

## マイルストーン

### Milestone 1: LangChain + LangSmith
LLMアプリケーション構築の基盤となるLangChainの基本概念（プロンプト、チェーン、ツール連携）を習得し、LangSmithによるトレース・デバッグ環境を整える。

- [x] LangChain の環境構築
- [x] 基本的なチェーンの構築（プロンプトテンプレート、LLM呼び出し）
- [x] ツール連携とRetrieval-Augmented Generation (RAG) の実装
- [ ] LangSmith のセットアップとトレース可視化
- [ ] LangSmith による評価・デバッグの実践

### Milestone 2: LangGraph
ステートフルなマルチエージェントワークフローをグラフ構造で構築する。

- [ ] LangGraph の環境構築
- [ ] 基本的なグラフワークフローの構築
- [ ] ステート管理と条件分岐の実装
- [ ] マルチエージェントパターンの実装

### Milestone 3: Hono API Server
LangChain.js で構築したチェーン/エージェントを Hono による REST API として公開する。

- [ ] Hono プロジェクトのセットアップ
- [ ] チェーン/エージェントのAPI化
- [ ] ストリーミングレスポンスの実装
- [ ] デプロイと動作検証（Cloudflare Workers / Node.js）

### Milestone 4: LangFlow
ビジュアルUIでフローを構築し、ノーコード/ローコード開発を検証する。

- [ ] LangFlow のセットアップ
- [ ] ビジュアルフローの構築
- [ ] カスタムコンポーネントの作成
- [ ] 既存実装との比較検証

## 技術スタック

- TypeScript / pnpm
- LangChain.js / LangSmith / LangGraph.js / Hono / LangFlow
- LLM: Ollama（ローカル） → Amazon Bedrock（クラウド）
