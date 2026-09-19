/** Graphics presets. Each one genuinely changes cost, not just a label. */

export type QualityPreset = "low" | "medium" | "high" | "ultra";

export interface QualitySettings {
  postFx: boolean;
  bloom: boolean;
  ssao: boolean;
  /** Depth of field is only ever enabled during cinematic moments. */
  dof: boolean;
  grade: boolean;
  smaa: boolean;
  msaaSamples: number;
  shadows: boolean;
  shadowMapSize: number;
  contactShadows: boolean;
  lightShafts: boolean;
  dustCount: number;
  emberCount: number;
  maxPixelRatio: number;
  captureParticles: number;
  /**
   * The ambient combat stance every figure holds — thirty-two skeletons breathing
   * every frame, which is the part of skeletal animation that actually costs
   * something. Off, the figures hold the first frame of their stance instead.
   *
   * This is deliberately *not* a switch on animation as a whole: the stride a
   * figure crosses the board on, its strike and its death are one or two mixers
   * for a second or two and run on every preset. Bundling the walk cycle in here
   * is what left phones sliding statues around the board — mobile lands on `low`
   * by default, so the entire rank read as having lost its animation.
   */
  idleAnimations: boolean;
  /** Instanced silhouette soldiers drawn up per distant army. */
  troopCount: number;
  /** Lit camp pyres outside the walls (each one is a real point light). */
  campfires: number;
  ashCount: number;
  smokeCount: number;
  /** Siege engines, banners, battlefield debris and circling crows. */
  battleProps: boolean;
}

export const QUALITY_SETTINGS: Record<QualityPreset, QualitySettings> = {
  low: {
    postFx: false,
    bloom: false,
    ssao: false,
    dof: false,
    grade: false,
    smaa: false,
    msaaSamples: 0,
    shadows: false,
    shadowMapSize: 512,
    contactShadows: true,
    lightShafts: false,
    dustCount: 0,
    emberCount: 0,
    maxPixelRatio: 1,
    captureParticles: 18,
    idleAnimations: false,
    troopCount: 60,
    campfires: 0,
    ashCount: 0,
    smokeCount: 24,
    battleProps: false,
  },
  medium: {
    postFx: true,
    bloom: true,
    ssao: false,
    dof: false,
    grade: true,
    smaa: true,
    msaaSamples: 0,
    shadows: true,
    shadowMapSize: 1024,
    contactShadows: true,
    lightShafts: true,
    dustCount: 220,
    emberCount: 60,
    maxPixelRatio: 1.5,
    captureParticles: 34,
    idleAnimations: true,
    troopCount: 140,
    campfires: 2,
    ashCount: 70,
    smokeCount: 45,
    battleProps: true,
  },
  high: {
    postFx: true,
    bloom: true,
    ssao: false,
    dof: true,
    grade: true,
    smaa: true,
    msaaSamples: 4,
    shadows: true,
    shadowMapSize: 2048,
    contactShadows: true,
    lightShafts: true,
    dustCount: 520,
    emberCount: 120,
    maxPixelRatio: 2,
    captureParticles: 60,
    idleAnimations: true,
    troopCount: 240,
    campfires: 3,
    ashCount: 130,
    smokeCount: 65,
    battleProps: true,
  },
  ultra: {
    postFx: true,
    bloom: true,
    ssao: true,
    dof: true,
    grade: true,
    smaa: true,
    msaaSamples: 4,
    shadows: true,
    shadowMapSize: 4096,
    contactShadows: true,
    lightShafts: true,
    dustCount: 900,
    emberCount: 190,
    maxPixelRatio: 2,
    captureParticles: 90,
    idleAnimations: true,
    troopCount: 320,
    campfires: 4,
    ashCount: 220,
    smokeCount: 90,
    battleProps: true,
  },
};

export const QUALITY_ORDER: QualityPreset[] = ["low", "medium", "high", "ultra"];

/**
 * First-run guess from the GPU string, core count and memory. The engine then
 * measures real frame times for a few seconds and steps down if needed.
 *
 * `navigator.deviceMemory` is Chromium-only: Safari and every iPhone report
 * nothing at all. Defaulting the unknown to 4 GiB and then demanding 6 sent
 * *every* iOS device to `low` on the first frame, which is how phones ended up
 * on the one preset that had the stride switched off.
 */
export function detectQualityPreset(): QualityPreset {
  if (typeof window === "undefined") return "high";
  const isTouch = window.matchMedia("(pointer: coarse)").matches;
  const cores = navigator.hardwareConcurrency ?? 4;
  const reported = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  // Absence is not evidence of a small device: only judge memory when it is known.
  const memory = reported ?? Number.POSITIVE_INFINITY;

  let renderer = "";
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
    if (gl) {
      const info = gl.getExtension("WEBGL_debug_renderer_info");
      if (info) renderer = String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL)).toLowerCase();
    }
  } catch {
    renderer = "";
  }

  const weakGpu = /(swiftshader|llvmpipe|software|mali-4|adreno \(tm\) [345]|intel.*hd graphics [2-4])/.test(renderer);
  const strongGpu = /(rtx|radeon rx|apple m[1-9]|geforce gtx 1[06-9]|arc a)/.test(renderer);

  if (weakGpu || cores <= 2 || memory <= 2) return "low";
  // A current phone holds `medium` comfortably, and the frame-rate watchdog
  // below steps it back down if this one cannot.
  if (isTouch) return cores >= 4 && memory >= 3 ? "medium" : "low";
  if (strongGpu && cores >= 8 && memory >= 8) return "ultra";
  if (cores >= 6 && memory >= 4) return "high";
  return "medium";
}
