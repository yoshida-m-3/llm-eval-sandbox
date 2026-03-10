/**
 * LangChain → OpenTelemetry コールバックハンドラー
 *
 * LangChain の各ステップ（LLM 呼び出し、チェーン、検索など）を
 * OTel スパンとして記録する。Grafana Tempo 上でトレースツリーとして可視化できる。
 */
import { trace, Span, SpanStatusCode, context } from "@opentelemetry/api";
import { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import type { Serialized } from "@langchain/core/load/serializable";
import type { LLMResult } from "@langchain/core/outputs";
import type { ChainValues } from "@langchain/core/utils/types";
import type { Document } from "@langchain/core/documents";

const tracer = trace.getTracer("langchain-instrumentation", "1.0.0");

export class OTelCallbackHandler extends BaseCallbackHandler {
  name = "OTelCallbackHandler";

  // runId → { span, parentContext } のマップ
  private spans = new Map<string, { span: Span; ctx: ReturnType<typeof context.active> }>();

  private startSpan(runId: string, name: string, attrs: Record<string, string> = {}) {
    const parentCtx = this.findParentContext(runId) ?? context.active();
    const span = tracer.startSpan(name, { attributes: attrs }, parentCtx);
    // スパンのコンテキストを保存（子スパンがネストできるように）
    const spanCtx = trace.setSpan(parentCtx, span);
    this.spans.set(runId, { span, ctx: spanCtx });
  }

  private endSpan(runId: string) {
    const entry = this.spans.get(runId);
    if (entry) {
      entry.span.end();
      this.spans.delete(runId);
    }
  }

  private findParentContext(runId: string): ReturnType<typeof context.active> | undefined {
    // LangChain は parentRunId を渡さないが、
    // 直近のアクティブコンテキストを使ってネストする
    return undefined;
  }

  // --- LLM ---
  async handleLLMStart(
    llm: Serialized,
    prompts: string[],
    runId: string,
    parentRunId?: string,
  ) {
    const modelName = llm.id?.[llm.id.length - 1] ?? "unknown";
    this.startSpan(runId, `llm.${modelName}`, {
      "llm.model": modelName,
      "llm.prompt_length": String(prompts.join("").length),
    });
  }

  async handleLLMEnd(output: LLMResult, runId: string) {
    const entry = this.spans.get(runId);
    if (entry && output.generations?.[0]?.[0]) {
      const gen = output.generations[0][0];
      entry.span.setAttribute("llm.completion_length", String(gen.text.length));
      if (output.llmOutput?.tokenUsage) {
        const usage = output.llmOutput.tokenUsage;
        entry.span.setAttribute("llm.token.prompt", String(usage.promptTokens ?? 0));
        entry.span.setAttribute("llm.token.completion", String(usage.completionTokens ?? 0));
        entry.span.setAttribute("llm.token.total", String(usage.totalTokens ?? 0));
      }
    }
    this.endSpan(runId);
  }

  async handleLLMError(err: Error, runId: string) {
    const entry = this.spans.get(runId);
    if (entry) {
      entry.span.setStatus({ code: SpanStatusCode.ERROR, message: err.message });
      entry.span.recordException(err);
    }
    this.endSpan(runId);
  }

  // --- Chain ---
  async handleChainStart(
    chain: Serialized,
    inputs: ChainValues,
    runId: string,
    parentRunId?: string,
  ) {
    const chainName = chain.id?.[chain.id.length - 1] ?? "chain";
    this.startSpan(runId, `chain.${chainName}`, {
      "chain.type": chainName,
    });
  }

  async handleChainEnd(outputs: ChainValues, runId: string) {
    this.endSpan(runId);
  }

  async handleChainError(err: Error, runId: string) {
    const entry = this.spans.get(runId);
    if (entry) {
      entry.span.setStatus({ code: SpanStatusCode.ERROR, message: err.message });
      entry.span.recordException(err);
    }
    this.endSpan(runId);
  }

  // --- Retriever ---
  async handleRetrieverStart(
    retriever: Serialized,
    query: string,
    runId: string,
    parentRunId?: string,
  ) {
    this.startSpan(runId, "retriever.search", {
      "retriever.query": query,
    });
  }

  async handleRetrieverEnd(documents: Document[], runId: string) {
    const entry = this.spans.get(runId);
    if (entry) {
      entry.span.setAttribute("retriever.document_count", String(documents.length));
    }
    this.endSpan(runId);
  }

  async handleRetrieverError(err: Error, runId: string) {
    const entry = this.spans.get(runId);
    if (entry) {
      entry.span.setStatus({ code: SpanStatusCode.ERROR, message: err.message });
      entry.span.recordException(err);
    }
    this.endSpan(runId);
  }
}
