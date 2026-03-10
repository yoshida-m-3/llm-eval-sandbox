# OpenTelemetry トレース計装リファレンス

## 1. アーキテクチャ

```
┌─────────────────┐    OTLP HTTP     ┌─────────────────┐    OTLP HTTP    ┌───────────┐    query    ┌─────────┐
│  Node.js App    │ ───── :4318 ────→ │  OTel Collector │ ─────────────→ │   Tempo   │ ←──────── │ Grafana │
│  (LangChain.js) │                   │  (中継・バッチ) │                │ (トレースDB)│           │  (UI)   │
└─────────────────┘                   └─────────────────┘                └───────────┘           └─────────┘
```

### OTel Collector を挟む理由

- アプリから直接 Tempo に送ることも可能だが、Collector を挟むのが OTel の標準構成
- バッチ処理・フィルタリング・複数バックエンドへの分岐を Collector 側で制御できる
- バックエンド追加時（例: AWS CloudWatch）もアプリ側の変更が不要

---

## 2. OTel の3層構造

```
┌─ resource（プロセスレベル。全スパン共通）─────────┐
│  service.name    = "llm-eval-sandbox"            │
│  service.version = "1.0.0"                       │
│                                                   │
│  ┌─ tracer（計装ライブラリレベル）──────────────┐ │
│  │  name    = "langchain-instrumentation"       │ │
│  │  version = "1.0.0"                           │ │
│  │                                               │ │
│  │  ┌─ span（個別操作レベル）──────────────┐    │ │
│  │  │  name = "llm.ChatOllama"             │    │ │
│  │  │  llm.model = "ChatOllama"            │    │ │
│  │  │  llm.prompt_length = "142"           │    │ │
│  │  └──────────────────────────────────────┘    │ │
│  └───────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────┘
```

| レベル | 設定場所 | 粒度 | 例 |
|---|---|---|---|
| **resource** | `setup.ts` の `resourceFromAttributes` | プロセス全体 | アプリ名・バージョン |
| **tracer** | `trace.getTracer("名前", "版")` | 計装ライブラリ単位 | 誰がスパンを出したか |
| **span** | `startSpan("名前", { attributes })` | 個別の操作 | LLM 呼び出し1回 |

### service.name の設定

`@opentelemetry/semantic-conventions` の `ATTR_SERVICE_NAME` は定数で、実体は文字列 `"service.name"`。
省略するとデフォルト値 `"unknown_service"` になる。

### スクリプト単位の識別

全スクリプトが同じ `service.name` を共有するため、スクリプト単位で区別するには resource 属性を追加する:

```ts
const scriptName = process.argv[1]?.split("/").pop()?.replace(".ts", "") ?? "unknown";
resource: resourceFromAttributes({
  [ATTR_SERVICE_NAME]: "llm-eval-sandbox",
  "script.name": scriptName,  // Grafana で属性フィルタ可能
})
```

---

## 3. LangChain コールバック方式による計装

### フロー

```
chain.invoke({ context, question }, { callbacks: [otelHandler] })
│
│  ① handleChainStart(runId="aaa")
│     │
│     │  ② handleLLMStart(runId="bbb")
│     │     │
│     │     │  Ollama API 呼び出し（実際の推論）
│     │     │
│     │  ③ handleLLMEnd(runId="bbb")
│     │     → span に属性を追記して end()
│     │
│  ④ handleChainEnd(runId="aaa")
│     → span.end()
```

### spans Map によるライフサイクル管理

Start と End が別メソッドで呼ばれるため、その間スパンを `Map<runId, { span, ctx }>` で保持する。

```
① Start → Map に保存  →  ③ End → 属性追記 → span.end() → Map から削除
```

### 対応するイベントとスパン

| LangChain イベント | スパン名 | 記録する属性 |
|---|---|---|
| `handleLLMStart/End` | `llm.{モデル名}` | モデル名、プロンプト長、completion 長、トークン使用量 |
| `handleChainStart/End` | `chain.{チェーン名}` | チェーンタイプ |
| `handleRetrieverStart/End` | `retriever.search` | 検索クエリ、取得ドキュメント数 |

エラー時は `SpanStatusCode.ERROR` + `recordException` でスタックトレースを記録。

### Grafana Tempo 上での見え方（RAG パイプライン）

```
┌─ rag-pipeline (手動スパン) ──────────────────────────────────────────┐
│                                                                      │
│  ┌─ retrieval (手動スパン) ─────┐                                   │
│  │  retrieval.doc_count = 3     │                                   │
│  └──────────────────────────────┘                                   │
│                                                                      │
│  ┌─ generation (手動スパン) ─────────────────────────────────────┐  │
│  │                                                               │  │
│  │  ┌─ chain.RunnableSequence (コールバック自動生成) ──────────┐ │  │
│  │  │                                                          │ │  │
│  │  │  ┌─ llm.ChatOllama (コールバック自動生成) ────────────┐ │ │  │
│  │  │  │  llm.model = "ChatOllama"                          │ │ │  │
│  │  │  │  llm.completion_length = "120"                     │ │ │  │
│  │  │  └────────────────────────────────────────────────────┘ │ │  │
│  │  │                                                          │ │  │
│  │  └──────────────────────────────────────────────────────────┘ │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

手動スパン（ビジネスロジック粒度）とコールバックスパン（LangChain 内部粒度）が入れ子になる。

---

## 4. コールバック方式 vs Python 自動計装

| | Python 自動計装 | コールバック方式（今回） |
|---|---|---|
| 仕組み | ランタイムで bytecode を解析・書き換え（monkey-patching） | LangChain が用意したフックに関数を登録 |
| 初回起動コスト | あり（モジュールスキャン） | なし |
| 設定の手軽さ | 環境変数だけで動く | `callbacks: [handler]` を渡す必要がある |
| 対象範囲 | ライブラリ全体を自動検出 | LangChain が発火するイベントのみ |
| 渡し忘れリスク | なし（全自動） | あり（`callbacks` を渡さないとスパンが出ない） |

コールバック方式はコード解析・bytecode 書き換え・monkey-patching を一切行わないため、
初回起動時のパフォーマンス劣化が発生しない。

---

## 5. ライブラリ選定

### OTel SDK パッケージ（7個）

すべて `@opentelemetry/` スコープの **CNCF 公式パッケージ**。ベンダーロックインなし。

| パッケージ | 役割 |
|---|---|
| `@opentelemetry/api` | 公開 API（tracer 取得、スパン操作） |
| `@opentelemetry/sdk-node` | Node.js 向け統合 SDK |
| `@opentelemetry/sdk-trace-node` | Node.js 向けトレーサープロバイダー |
| `@opentelemetry/sdk-trace-base` | `BatchSpanProcessor` 等の基盤クラス |
| `@opentelemetry/exporter-trace-otlp-http` | OTLP/HTTP エクスポーター |
| `@opentelemetry/resources` | サービスメタデータ付与 |
| `@opentelemetry/semantic-conventions` | 標準属性名の定数定義 |

### OTLP/HTTP を選んだ理由

- gRPC より依存が軽い（`@grpc/grpc-js` 不要）
- curl やブラウザでデバッグしやすい
- 本番でも HTTP で十分なケースが多い

### コールバックハンドラーを自作した理由

`@traceloop/node-server-sdk`（OpenLLMetry）という LLM 自動計装ライブラリも候補だったが、
学習目的のため仕組みが見えるコールバック方式を採用した。

---

## 6. LangSmith トレースとの比較

| | LangSmith | OpenTelemetry |
|---|---|---|
| セットアップ | 環境変数だけ | SDK 初期化 + コールバック |
| バックエンド | SaaS（LangSmith Cloud） | 自前（Tempo, Jaeger 等） |
| LangChain 専用 | Yes | No（汎用。HTTP/DB 等も計装可） |
| ベンダーロックイン | LangSmith に依存 | なし（CNCF 標準） |
| コスト | 無料枠あり、有料プラン | インフラ費用のみ |

---

## 7. 使い方

```bash
pnpm otel:up      # Docker スタック起動（Collector + Tempo + Grafana）
pnpm otel:08      # デモ実行（Ollama が必要）
pnpm otel:down    # スタック停止
```

- Grafana: http://localhost:3000 → Explore → Tempo
- Tempo API: http://localhost:3200
