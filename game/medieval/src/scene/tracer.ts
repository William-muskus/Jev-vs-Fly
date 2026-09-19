/**
 * The streak a round leaves in the air: a short 3D ribbon laid along the line
 * the ball *actually* flew, from the bore to the body.
 *
 * A round crossing this hall is a few pixels wide and gone in a fifth of a
 * second. Everything that used to make it legible was pinned to the round
 * itself — a smear cone, a glint, a heat sprite — so all of it moved with the
 * metal and none of it told the eye where the shot had *been*. The result is a
 * shot you notice arriving and never see travel.
 *
 * This is the missing half. The streak is built from the round's own flown
 * path, resampled every frame, so:
 *
 * - it is **geometry, not a billboard**: it holds its shape from any camera
 *   angle, and it goes behind figures and pillars the way a real object does;
 * - it **bends where the ball bent**. A smoothbore ball bellies off the line of
 *   sight and comes back onto the body, and that curve is now visible instead
 *   of being hidden inside a straight cone;
 * - it is deliberately **short**. A streak that reached from muzzle to victim
 *   would read as a laser, which is the one thing black powder never was. It is
 *   about a square long: enough to show the direction of travel and the speed,
 *   never enough to look like a beam.
 *
 * Cross-section is a three-bladed tube rather than a camera-facing quad — no
 * camera to consult, no flipping as the shot crosses the view axis, and cheap
 * enough (a couple of dozen rings) to run several in the air at once. The tail
 * pinches to a needle and the brightness falls off along it, so the streak
 * *dissolves* behind the round rather than ending on a cut edge.
 */

import * as THREE from "three";

/** How one round's streak is drawn. */
export interface StreakLook {
  /**
   * Visible arc length, in rendered ball diameters. Kept short on purpose: this
   * is the length of blurred air behind a round, not a tracer burning down the
   * whole line of fire.
   */
  span: number;
  /** Width at the head, as a fraction of the round's rendered diameter. */
  width: number;
  /** The body of the streak — disturbed air lit by the hall, not fire. */
  color: number;
  /** The hot filament inside it, only visible over the head of the streak. */
  core: number;
  /** Peak brightness of the body, 0–1. */
  strength: number;
}

/** Blades in the tube's cross-section: three is the cheapest solid section. */
const BLADES = 3;

const scratch = new THREE.Vector3();
const tangent = new THREE.Vector3();
const binormal = new THREE.Vector3();
const vertex = new THREE.Vector3();

/** Any unit vector square to the given one — enough to seat the first ring. */
function anyPerpendicular(axis: THREE.Vector3): THREE.Vector3 {
  return Math.abs(axis.y) > 0.9
    ? new THREE.Vector3(0, 0, 1).cross(axis).normalize()
    : new THREE.Vector3(0, 1, 0).cross(axis).normalize();
}

/**
 * One layer of the streak: a tube swept along the spine, tapering and dimming
 * towards the tail.
 *
 * Buffers are allocated once at their maximum size and rewritten in place every
 * frame; only the rings in use are drawn (`setDrawRange`), so a streak that has
 * not yet grown to full length never trails stale triangles behind it.
 */
class StreakTube {
  readonly mesh: THREE.Mesh;
  private readonly geometry = new THREE.BufferGeometry();
  private readonly material: THREE.MeshBasicMaterial;
  private readonly position: THREE.BufferAttribute;
  private readonly colour: THREE.BufferAttribute;
  private readonly normal = new THREE.Vector3(0, 1, 0);

  /**
   * @param rings most spine samples this layer will ever draw
   * @param radius half-width at the head, in world units
   * @param tint colour at full brightness
   * @param falloff how fast brightness dies towards the tail. A low number is a
   *   long even streak; a high one only lights the few calibres behind the
   *   round, which is how the hot filament is kept to the head.
   */
  constructor(
    private readonly rings: number,
    private readonly radius: number,
    private readonly tint: THREE.Color,
    private readonly falloff: number,
    order: number,
  ) {
    const vertices = rings * BLADES;
    this.position = new THREE.BufferAttribute(new Float32Array(vertices * 3), 3);
    this.position.setUsage(THREE.DynamicDrawUsage);
    this.colour = new THREE.BufferAttribute(new Float32Array(vertices * 3), 3);
    this.colour.setUsage(THREE.DynamicDrawUsage);
    const index: number[] = [];
    for (let ring = 0; ring < rings - 1; ring += 1) {
      for (let blade = 0; blade < BLADES; blade += 1) {
        const a = ring * BLADES + blade;
        const b = ring * BLADES + ((blade + 1) % BLADES);
        index.push(a, b, b + BLADES, a, b + BLADES, a + BLADES);
      }
    }
    this.geometry.setAttribute("position", this.position);
    this.geometry.setAttribute("color", this.colour);
    this.geometry.setIndex(index);
    this.geometry.setDrawRange(0, 0);
    this.material = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      // Additive so the streak reads as light on air rather than as a solid
      // grey tube, and never writes depth over the round it belongs to.
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = "shot_streak";
    // A streak spanning a square of a board can leave its own stale bounding
    // sphere behind between frames; never let one blink out mid-flight.
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = order;
    this.mesh.visible = false;
  }

  /**
   * Sweeps the tube along the spine (oldest sample first, the round itself
   * last) at the given brightness.
   *
   * The ring frame is carried forward from sample to sample instead of being
   * rebuilt from a world axis: a fresh basis per ring makes the tube twist
   * visibly wherever the path turns.
   */
  write(spine: THREE.Vector3[], brightness: number): void {
    const count = spine.length;
    if (count < 2 || brightness <= 0.001) {
      this.geometry.setDrawRange(0, 0);
      this.mesh.visible = false;
      return;
    }
    this.mesh.visible = true;
    const used = Math.min(count, this.rings);
    const first = count - used;
    for (let i = 0; i < used; i += 1) {
      const point = spine[first + i];
      const ahead = spine[Math.min(count - 1, first + i + 1)];
      const behind = spine[Math.max(first, first + i - 1)];
      tangent.subVectors(ahead, behind);
      if (tangent.lengthSq() < 1e-12) tangent.set(0, 0, 1);
      else tangent.normalize();
      // Project the carried normal back square to the new tangent.
      this.normal.addScaledVector(tangent, -this.normal.dot(tangent));
      if (this.normal.lengthSq() < 1e-8) this.normal.copy(anyPerpendicular(tangent));
      this.normal.normalize();
      binormal.crossVectors(tangent, this.normal);

      const u = used === 1 ? 1 : i / (used - 1);
      // Pinched to a needle at the tail: air closing behind the round.
      const width = this.radius * Math.pow(u, 0.55);
      const fade = brightness * Math.pow(u, this.falloff);
      for (let blade = 0; blade < BLADES; blade += 1) {
        const angle = (blade / BLADES) * Math.PI * 2;
        vertex
          .copy(point)
          .addScaledVector(this.normal, Math.cos(angle) * width)
          .addScaledVector(binormal, Math.sin(angle) * width);
        const at = i * BLADES + blade;
        this.position.setXYZ(at, vertex.x, vertex.y, vertex.z);
        this.colour.setXYZ(at, this.tint.r * fade, this.tint.g * fade, this.tint.b * fade);
      }
    }
    this.geometry.setDrawRange(0, (used - 1) * BLADES * 6);
    this.position.needsUpdate = true;
    this.colour.needsUpdate = true;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}

/**
 * The streak behind one round, in two layers: a wide faint sheath of disturbed
 * air, and a thin bright filament inside it that only lights the few calibres
 * immediately behind the metal. One layer alone reads either as fog (wide and
 * faint) or as a wire (thin and bright); together they read as speed.
 *
 * Lives in world space and is fed the round's position every frame by
 * `flyShot`; it owns no shared resources, so a shot can throw it away the
 * moment it is spent.
 */
export class TracerStreak {
  readonly object = new THREE.Group();
  private readonly sheath: StreakTube;
  private readonly filament: StreakTube;
  /** The flown path, oldest first; the last entry is always the live round. */
  private readonly points: THREE.Vector3[] = [];
  /** Arc length the streak is kept trimmed to, in world units. */
  private readonly span: number;
  /** How far the round has to travel before a new spine sample is laid down. */
  private readonly step: number;
  private readonly rings: number;
  private brightness = 1;

  /**
   * @param gauge rendered diameter of the round, so the streak is always in
   *   proportion to the thing that made it
   * @param rings spine resolution — the one knob graphics quality turns
   */
  constructor(look: StreakLook, gauge: number, rings: number) {
    this.rings = Math.max(4, Math.round(rings));
    this.span = look.span * gauge;
    this.step = this.span / (this.rings - 1);
    const head = look.width * gauge * 0.5;
    this.sheath = new StreakTube(this.rings, head, new THREE.Color(look.color), 1.6, 6);
    // Half the width, brighter, and dying far faster along the length: the part
    // of the streak that is still nearly the round itself.
    this.filament = new StreakTube(this.rings, head * 0.42, new THREE.Color(look.core), 4.2, 7);
    (this.sheath.mesh.material as THREE.MeshBasicMaterial).opacity = look.strength;
    this.object.name = "shot_trail";
    this.object.add(this.sheath.mesh, this.filament.mesh);
    this.brightness = look.strength;
  }

  /**
   * Moves the head of the streak onto the round's new position, laying a fresh
   * spine sample once it has travelled far enough to need one, then trims the
   * tail back to the streak's span.
   */
  extend(at: THREE.Vector3): void {
    const points = this.points;
    if (points.length < 2) {
      points.push(at.clone());
    } else {
      const anchor = points[points.length - 2];
      if (anchor.distanceToSquared(at) >= this.step * this.step) points.push(at.clone());
      else points[points.length - 1].copy(at);
    }
    this.trim();
    this.redraw();
  }

  /** Global brightness, for the beat the streak hangs after the round lands. */
  fade(amount: number): void {
    this.brightness = Math.max(0, amount);
    this.redraw();
  }

  dispose(): void {
    this.sheath.dispose();
    this.filament.dispose();
    this.object.removeFromParent();
    this.object.clear();
  }

  private redraw(): void {
    this.sheath.write(this.points, this.brightness);
    this.filament.write(this.points, this.brightness);
  }

  /**
   * Keeps the streak exactly `span` long by *sliding* its last sample along the
   * oldest segment rather than dropping whole samples. Popping a sample makes
   * the tail visibly stutter backwards once per step; sliding it means the tail
   * dissolves at the same speed the round is travelling.
   */
  private trim(): void {
    const points = this.points;
    if (points.length < 2) return;
    let total = 0;
    let drop = 0;
    for (let i = points.length - 1; i > 0; i -= 1) {
      const segment = points[i].distanceTo(points[i - 1]);
      if (total + segment >= this.span) {
        const keep = (this.span - total) / Math.max(1e-6, segment);
        scratch.copy(points[i - 1]);
        points[i - 1].copy(points[i]).lerp(scratch, keep);
        drop = i - 1;
        break;
      }
      total += segment;
    }
    if (drop > 0) points.splice(0, drop);
    // A slow round on a fast frame rate can bank more samples than the tube can
    // ever draw; the oldest are the ones nobody sees.
    const overflow = points.length - (this.rings + 1);
    if (overflow > 0) points.splice(0, overflow);
  }
}
