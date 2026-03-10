/**
 * OpenTelemetry SDK セットアップ
 *
 * アプリ起動時に最初にインポートすることで、
 * 以降のすべてのスパンが OTLP エクスポーターへ送信される。
 */
import { NodeSDK } from "@opentelemetry/sdk-node";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { resourceFromAttributes } from "@opentelemetry/resources";
import {
  ATTR_SERVICE_NAME,
  ATTR_SERVICE_VERSION,
} from "@opentelemetry/semantic-conventions";

const OTEL_COLLECTOR_URL =
  process.env.OTEL_EXPORTER_OTLP_ENDPOINT ?? "http://localhost:4318";

const exporter = new OTLPTraceExporter({
  url: `${OTEL_COLLECTOR_URL}/v1/traces`,
});

const sdk = new NodeSDK({
  resource: resourceFromAttributes({
    [ATTR_SERVICE_NAME]: "llm-eval-sandbox",
    [ATTR_SERVICE_VERSION]: "1.0.0",
  }),
  spanProcessors: [
    new BatchSpanProcessor(exporter, {
      maxQueueSize: 512,
      scheduledDelayMillis: 2000,
    }),
  ],
});

sdk.start();
console.log("[OTel] SDK started — exporting traces to", OTEL_COLLECTOR_URL);

// Graceful shutdown
const shutdown = async () => {
  await sdk.shutdown();
  console.log("[OTel] SDK shut down");
  process.exit(0);
};
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);

export { sdk };
