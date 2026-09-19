import { memo, useEffect, useRef } from "react";

import { BrainCanvas } from "../../../../web/brainviz.js";
import type { FlyAnatomy, FlyThought } from "../ai/flyAnatomy";
import { FLY_PLAYER_NAME } from "./jevflyFlags";

/**
 * Live FlyWire sample: sampled neurons glowing at their connectome positions.
 * A thought replays the recurrent timesteps; while Fruit Fly is still searching
 * the dots pulse.
 */
export const FlyBrainView = memo(function FlyBrainView({
  anatomy,
  thought,
  thinking,
}: {
  anatomy: FlyAnatomy;
  thought: FlyThought | null;
  thinking: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const vizRef = useRef<BrainCanvas | null>(null);

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
    vizRef.current?.setThinking(thinking);
  }, [thinking]);

  useEffect(() => {
    const viz = vizRef.current;
    if (!viz || !thought) return;
    if (thought.trace && thought.traceSteps > 1) {
      viz.setTrace(thought.trace, thought.traceSteps, { autoplay: true });
    } else if (thought.activitySample) {
      viz.setActivity(thought.activitySample);
    }
  }, [thought]);

  return (
    <div className="mc-fly-brain mc-slate pointer-events-none" aria-label={`${FLY_PLAYER_NAME} neural activity`}>
      <div className="mc-fly-brain-head">
        <span>{FLY_PLAYER_NAME}</span>
        <span className="mc-fly-brain-state">{thinking ? "thinking" : thought ? "thought" : "quiet"}</span>
      </div>
      <canvas ref={canvasRef} className="mc-fly-brain-canvas" />
    </div>
  );
});
