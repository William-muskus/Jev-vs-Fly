/**
 * TypeSafe bills Jev on input tokens; output is free.
 * @see https://typesafe.ai/blog/introducing-system-one-models-and-jev
 */
export const JEV_INPUT_USD_PER_MILLION = 0.042;

export interface JevUsage {
  input_tokens: number;
  output_tokens: number;
}

export function emptyJevUsage(): JevUsage {
  return { input_tokens: 0, output_tokens: 0 };
}

export function addJevUsage(total: JevUsage, delta: JevUsage | null | undefined): JevUsage {
  if (!delta) return total;
  return {
    input_tokens: total.input_tokens + Math.max(0, Math.floor(delta.input_tokens || 0)),
    output_tokens: total.output_tokens + Math.max(0, Math.floor(delta.output_tokens || 0)),
  };
}

export function parseJevUsage(raw: unknown): JevUsage {
  if (!raw || typeof raw !== "object") return emptyJevUsage();
  const record = raw as { input_tokens?: unknown; output_tokens?: unknown };
  return {
    input_tokens: Math.max(0, Math.floor(Number(record.input_tokens) || 0)),
    output_tokens: Math.max(0, Math.floor(Number(record.output_tokens) || 0)),
  };
}

export function jevUsd(inputTokens: number): number {
  return (Math.max(0, inputTokens) / 1_000_000) * JEV_INPUT_USD_PER_MILLION;
}

export function formatJevUsd(usd: number): string {
  if (usd <= 0) return "$0.00";
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  if (usd < 1) return `$${usd.toFixed(3)}`;
  return `$${usd.toFixed(2)}`;
}

export function formatTokenCount(n: number): string {
  return Math.max(0, Math.floor(n)).toLocaleString("en-US");
}

/** One line for the victory parchment: `Jev · $0.0013 · 31,200 tokens`. */
export function formatJevSpend(usage: JevUsage): string {
  const tokens = usage.input_tokens + usage.output_tokens;
  return `Jev · ${formatJevUsd(jevUsd(usage.input_tokens))} · ${formatTokenCount(tokens)} tokens`;
}
