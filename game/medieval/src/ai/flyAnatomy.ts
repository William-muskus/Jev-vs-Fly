/**
 * Anatomy the fly worker sends with `ready`, and a thought (sampled neuron
 * activity) it streams while choosing a move.
 */

export interface FlySample {
  idx: Int32Array | number[];
  xy: Float32Array | number[];
  cls: Uint8Array | Int32Array | number[];
}

export interface FlySilhouette {
  xy: Float32Array | number[];
  cls: Uint8Array | Int32Array | number[];
}

export interface FlyAnatomy {
  sample: FlySample;
  silhouette: FlySilhouette;
  legend: string[];
  steps: number;
}

export interface FlyThought {
  activitySample: Float32Array | number[] | null;
  trace: Float32Array | number[] | null;
  traceSteps: number;
}

export function flyAnatomyFromReady(msg: {
  sample?: FlySample;
  silhouette?: FlySilhouette;
  legend?: string[];
  features?: { steps?: number };
}): FlyAnatomy | null {
  if (!msg.sample?.xy || !msg.silhouette?.xy) return null;
  return {
    sample: msg.sample,
    silhouette: msg.silhouette,
    legend: msg.legend ?? [],
    steps: msg.features?.steps && msg.features.steps > 0 ? msg.features.steps : 1,
  };
}

export function flyThoughtFromReply(msg: {
  activitySample?: Float32Array | number[] | null;
  trace?: Float32Array | number[] | null;
  traceSteps?: number;
}): FlyThought | null {
  const activity = msg.activitySample ?? null;
  const trace = msg.trace ?? null;
  if (!activity && !trace) return null;
  return {
    activitySample: activity,
    trace,
    traceSteps: msg.traceSteps && msg.traceSteps > 0 ? msg.traceSteps : 1,
  };
}
