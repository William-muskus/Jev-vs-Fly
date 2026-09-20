declare module "../../../../web/brainviz.js" {
  export const CLASS_COLORS: Record<string, [number, number, number]>;
  export const NEURON_COLOR: [number, number, number];
  export class BrainCanvas {
    raf: number;
    constructor(canvas: HTMLCanvasElement);
    setData(sample: unknown, silhouette: unknown, legend: string[]): void;
    setActivity(values: ArrayLike<number> | null): void;
    setLiveActivity(values: ArrayLike<number> | null): void;
    resetLiveScale(): void;
    setTrace(trace: ArrayLike<number>, steps: number, opts?: { autoplay?: boolean }): void;
    setThinking(on: boolean): void;
    pause(): void;
    dispose(): void;
  }
  export class ClassStrip {
    constructor(canvas: HTMLCanvasElement);
    dispose(): void;
  }
}
