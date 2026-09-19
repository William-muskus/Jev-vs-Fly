/**
 * Fitting the hall to the screen it is being played on.
 *
 * Every camera shot in the engine was authored against a wide desktop window,
 * and a perspective camera's `fov` is its *vertical* angle. So the narrower the
 * viewport, the less of the board's width fits in frame — on a phone held
 * upright (aspect ≈ 0.46) a 46° lens sees barely half the files, and the answer
 * is never "pull straight back": the colonnade stands at radius 12.5, so a shot
 * dragged out past it puts the hall's own pillars between the player and the
 * board. That is the "the map is covering the board" bug.
 *
 * The fix is to solve the framing rather than author it: work out the distance
 * and the lens that put the whole board on the narrow axis, then take the extra
 * distance as *height* instead of ground reach so the camera climbs over the
 * colonnade rather than backing into it.
 */

import * as THREE from "three";

import { TILE } from "./board";

/**
 * How far the camera may sit from the board's centre on the ground plane before
 * the colonnade (pillars at radius 12.5, each 0.62–0.72 thick) starts cutting
 * across the board. Everything beyond this has to be bought with height.
 */
export const HALL_INNER_RADIUS = 11;

/**
 * Radius of the sphere a framing has to contain: the eight files plus a margin
 * for the figures standing on the outer ranks and the shields above their heads.
 */
export const BOARD_REACH = TILE * 4 + 0.45;

/** Widest lens the engine will ever open up to, however narrow the screen is. */
export const MAX_LENS_FOV = 78;

/** Steepest the camera is allowed to sit on a phone — portrait, then landscape. */
const HANDHELD_PORTRAIT_PHI = 0.82;
const HANDHELD_LANDSCAPE_PHI = 1.02;

export interface ViewportProfile {
  width: number;
  height: number;
  /** width / height of the drawing surface. */
  aspect: number;
  /** A coarse pointer on a hand-sized screen: phone, or a small tablet. */
  handheld: boolean;
  /** Taller than it is wide. */
  portrait: boolean;
}

export interface Framing {
  position: THREE.Vector3;
  target: THREE.Vector3;
  /** Vertical field of view, in degrees. */
  fov: number;
  /** Distance from `target` the framing settled on. */
  radius: number;
}

export interface FramingOptions {
  /** The lens the shot was authored with — the framing never goes tighter. */
  fov: number;
  /** The widest this framing is allowed to open the lens. */
  maxFov: number;
  /** Radius of the sphere that has to stay in frame. */
  reach?: number;
  /** How far back the camera may be sent. */
  maxDistance?: number;
}

export interface OrbitLimits {
  minDistance: number;
  maxDistance: number;
  minPolarAngle: number;
  maxPolarAngle: number;
  rotateSpeed: number;
  /** Pixels a press may travel and still count as a tap on a square. */
  tapSlop: number;
}

/**
 * What the engine is drawing into. `handheld` is a real capability test — a
 * coarse pointer on a small screen — not a user-agent guess, so a phone in
 * desktop mode and a narrow desktop window are both handled honestly.
 */
export function readViewport(width: number, height: number): ViewportProfile {
  const safeWidth = Math.max(1, width);
  const safeHeight = Math.max(1, height);
  const coarse =
    typeof window !== "undefined" && typeof window.matchMedia === "function"
      ? window.matchMedia("(pointer: coarse)").matches
      : false;
  const shortest = Math.min(safeWidth, safeHeight);
  return {
    width: safeWidth,
    height: safeHeight,
    aspect: safeWidth / safeHeight,
    // A phone in either orientation, or a window narrow enough to play like one.
    handheld: coarse ? shortest <= 820 : safeWidth <= 620,
    portrait: safeWidth < safeHeight,
  };
}

/** Vertical fov (degrees) that holds `reach` at `distance` on *both* axes. */
export function fitFov(reach: number, distance: number, aspect: number): number {
  const half = reach / Math.max(0.001, distance);
  const forHeight = Math.atan(half);
  const forWidth = Math.atan(half / Math.max(0.05, aspect));
  return THREE.MathUtils.radToDeg(Math.max(forHeight, forWidth) * 2);
}

/** Distance at which `reach` fits inside a `fov` lens on this aspect. */
export function fitDistance(reach: number, fov: number, aspect: number): number {
  const halfHeight = Math.tan(THREE.MathUtils.degToRad(fov) / 2);
  const halfWidth = halfHeight * aspect;
  return reach / Math.max(0.05, Math.min(halfHeight, halfWidth));
}

/**
 * The widest lens a viewport is allowed. A phone held upright needs the most,
 * because its narrow axis is the one the board's width has to fit on; a normal
 * desktop window never reaches for any of this and keeps its authored lens.
 */
export function lensCeiling(view: ViewportProfile, base: number): number {
  if (view.handheld) return Math.max(base, view.portrait ? 68 : 58);
  return Math.max(base, view.aspect < 1 ? 62 : 52);
}

/**
 * Steepest polar angle (measured from straight up) at which a camera `radius`
 * from the board still stands inside the colonnade.
 */
export function groundedPhi(radius: number): number {
  return Math.asin(THREE.MathUtils.clamp(HALL_INNER_RADIUS / Math.max(0.001, radius), 0, 1));
}

/**
 * Re-solves an authored shot for the viewport actually on screen.
 *
 * The azimuth is never touched — whichever side of the board the shot was
 * looking from, it keeps looking from there. Only the distance, the elevation
 * and the lens are solved.
 */
export function frameShot(
  position: THREE.Vector3,
  target: THREE.Vector3,
  view: ViewportProfile,
  options: FramingOptions,
): Framing {
  const reach = options.reach ?? BOARD_REACH;
  const maxFov = Math.max(options.fov, options.maxFov);
  const spherical = new THREE.Spherical().setFromVector3(position.clone().sub(target));
  const authored = spherical.radius;

  // How far back this lens has to sit before the whole board fits on the narrow
  // axis. On a wide window this is shorter than the authored shot, so nothing
  // moves and the desktop framing is exactly as it always was.
  const needed = fitDistance(reach, maxFov, view.aspect);
  spherical.radius = Math.min(Math.max(authored, needed), options.maxDistance ?? 21);

  // A phone reads the board from higher up: the near rank stops hiding the far
  // rank, and a tap lands on the square the finger is over.
  if (view.handheld) {
    spherical.phi = Math.min(spherical.phi, view.portrait ? HANDHELD_PORTRAIT_PHI : HANDHELD_LANDSCAPE_PHI);
  }
  // Whatever the angle, the extra distance is taken as height: past the pillars
  // the hall itself becomes the thing standing in front of the board.
  spherical.phi = Math.min(spherical.phi, groundedPhi(spherical.radius));
  spherical.makeSafe();

  return {
    position: new THREE.Vector3().setFromSpherical(spherical).add(target),
    target: target.clone(),
    fov: THREE.MathUtils.clamp(fitFov(reach, spherical.radius, view.aspect), options.fov, MAX_LENS_FOV),
    radius: spherical.radius,
  };
}

/**
 * Orbit limits for a viewport. `fitted` is the radius the current framing
 * settled on, so a phone — which frames from further out — can still be pulled
 * back a little past its own framing instead of being pinned at the limit.
 */
export function orbitLimits(view: ViewportProfile, fitted: number): OrbitLimits {
  const handheld = view.handheld;
  return {
    // A pinch must not be able to bury the camera inside the front rank.
    minDistance: handheld ? 5.8 : 4.5,
    maxDistance: Math.max(handheld ? 19 : 17, fitted * 1.25),
    minPolarAngle: 0.12,
    // Phones never get the near-ground angle: at eye level the board is a line
    // and the whole screen is hall.
    maxPolarAngle: handheld ? Math.PI / 2 - 0.36 : Math.PI / 2 - 0.08,
    rotateSpeed: handheld ? 0.4 : 0.55,
    tapSlop: handheld ? 16 : 8,
  };
}
