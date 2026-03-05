# llm-eval-sandbox

LangChain エコシステム（Langシリーズ）を段階的に学習・検証するためのプロジェクト。

## マイルストーン

### Milestone 1: LangChain + LangSmith
LLMアプリケーション構築の基盤となるLangChainの基本概念（プロンプト、チェーン、ツール連携）を習得し、LangSmithによるトレース・デバッグ環境を整える。

- [x] LangChain の環境構築
- [x] 基本的なチェーンの構築（プロンプトテンプレート、LLM呼び出し）
- [x] ツール連携とRetrieval-Augmented Generation (RAG) の実装
- [x] LangSmith のセットアップとトレース可視化
- [x] LangSmith による評価・デバッグの実践

### Milestone 2: LangGraph
ステートフルなマルチエージェントワークフローをグラフ構造で構築する。

- [x] LangGraph の環境構築
- [x] 基本的なグラフワークフローの構築
- [x] ステート管理と条件分岐の実装
- [x] マルチエージェントパターンの実装

### Milestone 3: LangFlow
ビジュアルUIでフローを構築し、ノーコード/ローコード開発を検証する。

- [x] LangFlow のセットアップ
- [x] ビジュアルフローの構築
- [ ] カスタムコンポーネントの作成
- [ ] 既存実装との比較検証

### Milestone 4: OpenTelemetry
LLMアプリケーションの可観測性をOTelで標準化し、複数バックエンドで検証する。

- [ ] OTel SDK のセットアップ（LangChain.js への計装追加）
- [ ] ローカル Grafana スタック（Tempo + Grafana）でのトレース可視化
- [ ] AWS CloudWatch へのエクスポート設定と比較
- [ ] メトリクス・ログの収集と可視化
- [ ] 両バックエンドの比較レポート

## 技術スタック

- TypeScript / pnpm
- LangChain.js / LangSmith / LangGraph.js / LangFlow
- OpenTelemetry SDK / OTel Collector
- Grafana / Grafana Tempo / Prometheus
- AWS CloudWatch（X-Ray / Metrics / Logs）
- LLM: Ollama（ローカル）
