import { memo, useCallback, useEffect, useRef, useState } from "react";

import { BrainCanvas } from "../../../../web/brainviz.js";
import { flyClient } from "../ai/flyClient";
import type { FlyAnatomy, FlyThought } from "../ai/flyAnatomy";
import { FLY_PLAYER_NAME } from "./jevflyFlags";

/**
 * Live FlyWire sample: sampled neurons glowing at their connectome positions.
 * Activity is painted as each recurrent timestep finishes in the worker — not
 * replayed afterwards — so the map matches Fruit Fly thinking in real time.
 */
export const FlyBrainView = memo(function FlyBrainView({
  anatomy,
  thought,
  thinking,
  liveBind,
  liveStep,
}: {
  anatomy: FlyAnatomy;
  thought: FlyThought | null;
  thinking: boolean;
  liveBind?: { current: ((sample: Float32Array | number[], step: number, steps: number) => void) | null };
  liveStep?: string | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const vizRef = useRef<BrainCanvas | null>(null);
  const sawLive = useRef(false);
  const [liveCaption, setLiveCaption] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const viz = new BrainCanvas(canvas);
    vizRef.current = viz;
    return () => {
      viz.dispose();
      vizRef.current = null;
    };
  }, []);

  useEffect(() => {
    vizRef.current?.setData(anatomy.sample, anatomy.silhouette, anatomy.legend);
  }, [anatomy]);

  useEffect(() => {
    const viz = vizRef.current;
    if (!viz) return;
    if (thinking) {
      viz.resetLiveScale();
      sawLive.current = false;
    }
    viz.setThinking(thinking);
    if (!thinking) setLiveCaption(null);
  }, [thinking]);

  const onLive = useCallback((sample: Float32Array | number[], step: number, steps: number): void => {
    sawLive.current = true;
    if (steps > 1) {
      const label = `step ${step + 1}/${steps}`;
      setLiveCaption(label);
      const el = document.querySelector(".mc-fly-brain-state");
      if (el) el.textContent = label;
    }
    vizRef.current?.setLiveActivity(sample);
  }, []);
  if (liveBind) liveBind.current = onLive;

  useEffect(() => {
    const unsub = flyClient.subscribeLive(onLive);
    return () => {
      unsub();
      if (liveBind?.current === onLive) liveBind.current = null;
    };
  }, [liveBind, onLive]);

  useEffect(() => {
    const viz = vizRef.current;
    if (!viz || !thought?.activitySample || thinking) return;
    // Live packets already painted the search. The leftover `thought` is the
    // first look at the board — snapping to it here would rewind the map.
    if (sawLive.current) return;
    viz.setLiveActivity(thought.activitySample);
  }, [thought, thinking]);

  const stateLabel = thinking
    ? liveStep
      ? `step ${liveStep}`
      : liveCaption ?? "thinking"
    : thought
      ? "thought"
      : "quiet";

  return (
    <div className="mc-fly-brain mc-slate pointer-events-none" aria-label={`${FLY_PLAYER_NAME} neural activity`}>
      <div className="mc-fly-brain-head">
        <span>{FLY_PLAYER_NAME}</span>
        <span className="mc-fly-brain-state">{stateLabel}</span>
      </div>
      <canvas ref={canvasRef} className="mc-fly-brain-canvas" />
    </div>
  );
});
