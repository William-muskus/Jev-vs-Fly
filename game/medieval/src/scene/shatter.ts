/**
 * What a round leaves behind when it arrives.
 *
 * Up to now the kill at the far end of a shot was the same warm sprite burst as
 * a sword blow: a flash, a puff, some glowing dots. That reads as *magic*, not
 * as impact — nothing about it says a lump of lead just went through a body at
 * three hundred metres a second. This module is the answer, and it is deliberately
 * built out of geometry rather than billboards, because the whole point of the
 * moment is that something **broke**.
 *
 * Four things fire on the same frame, in the order the eye reads them:
 *
 * 1. **The punch ring** — a disc of light square to the line of flight, snapped
 *    open on the surface it hit. It is the only part that is not physical, and
 *    it exists to tell the eye *where* on the body the round went in.
 * 2. **Sparks** — real stretched geometry, not dots: each spark is a sliver
 *    oriented along its own velocity, so it draws a streak that turns as it
 *    flies. They come off in a cone thrown *back* toward the shooter, which is
 *    what spall actually does, they cool white → orange → dull red over their
 *    life, and the survivors skitter along the flagstones.
 * 3. **Fragments** — chips of whatever the victim is made of: marble off a
 *    kingdom statue, glassy black flakes off Sun Empire obsidian, steel spall
 *    off a cuirass, wool and gilt braid off a Grande Armée coat. They tumble on
 *    their own axes, bounce once or twice off the board and settle.
 * 4. **Dust** — the fine haze the caller layers on top through the effects
 *    system; this module only reports the colour to use for it.
 *
 * All of it runs on the caller's tween clock inside two instanced draw calls
 * (one for sparks, one for fragments), so a whole exchange of fire costs the
 * frame loop nothing permanent and nothing pooled.
 */

import * as THREE from "three";

import type { SpellLight } from "./spells";
import { shockwaveTexture } from "./textures";
import { Ease, type TweenManager } from "./tween";

/** What the round found when it got there. */
export type ImpactBody =
  /** Carved kingdom marble: pale chips, bright dust, sparks off the plate under it. */
  | "marble"
  /** Sun Empire obsidian and jade: glassy black flakes, hard bright shatter. */
  | "obsidian"
  /** A cuirass or a helmet: steel spall and a shower of sparks, barely any dust. */
  | "plate"
  /** Napoleonic wool, leather and gilt braid: dark tatters, dust, few sparks. */
  | "uniform"
  /** The floor of the hall — a ricochet rather than a kill. */
  | "flagstone";

interface BodyRecipe {
  /** Fragment colours, drawn from at random per chip. */
  fragments: number[];
  fragmentRoughness: number;
  fragmentMetalness: number;
  /** Chips thrown at power 1. */
  fragmentCount: number;
  /** Chip size in world units at power 1. */
  fragmentSize: number;
  /** How far a chip is stretched into a sliver (1 = a cube-ish lump). */
  fragmentSliver: number;
  /** How lively the chips are off the stone. 0 = they land dead. */
  bounce: number;
  /** Sparks thrown at power 1. */
  sparkCount: number;
  /** Metres per second a spark leaves at. */
  sparkSpeed: number;
  sparkLife: number;
  /** Hot colour on the first frame, and what it cools to. */
  sparkHot: number;
  sparkCool: number;
  /** Colour of the punch ring and the light it throws. */
  ring: number;
  /** Fine haze the caller should layer over the top. */
  dust: number;
  /** How wide the spall cone opens, in radians. Tight = a clean hole. */
  spread: number;
}

/**
 * One recipe per material. The numbers are ordered by hardness: obsidian throws
 * the most fragments and the widest spread because it *shatters*; wool throws
 * the fewest and the slowest because it absorbs.
 */
const BODIES: Record<ImpactBody, BodyRecipe> = {
  marble: {
    fragments: [0xe8e2d4, 0xd6cfbe, 0xc2baa6, 0x9d9482],
    fragmentRoughness: 0.82,
    fragmentMetalness: 0.05,
    fragmentCount: 16,
    fragmentSize: 0.06,
    fragmentSliver: 2.1,
    bounce: 0.38,
    sparkCount: 14,
    sparkSpeed: 7,
    sparkLife: 0.42,
    sparkHot: 0xfff3d2,
    sparkCool: 0xff5a1c,
    ring: 0xffe6bd,
    dust: 0xd8d0be,
    spread: 1.05,
  },
  obsidian: {
    // Volcanic glass: near black with a jade fleck and a hot bronze edge.
    fragments: [0x1b1a20, 0x2b2933, 0x123d33, 0x8c6a2f],
    fragmentRoughness: 0.24,
    fragmentMetalness: 0.32,
    fragmentCount: 22,
    fragmentSize: 0.052,
    // Glass does not chip, it flakes: long, thin, razor slivers.
    fragmentSliver: 3.4,
    bounce: 0.52,
    sparkCount: 18,
    sparkSpeed: 8.4,
    sparkLife: 0.36,
    sparkHot: 0xfffaf0,
    sparkCool: 0x59f0c0,
    ring: 0xa8ffe0,
    dust: 0x4a4a52,
    spread: 1.25,
  },
  plate: {
    fragments: [0x8f959d, 0x6a707a, 0x4a4f57, 0xb9a06a],
    fragmentRoughness: 0.34,
    fragmentMetalness: 0.85,
    fragmentCount: 11,
    fragmentSize: 0.045,
    fragmentSliver: 2.6,
    bounce: 0.6,
    // Steel on steel is the brightest thing that happens on this board.
    sparkCount: 30,
    sparkSpeed: 10.5,
    sparkLife: 0.5,
    sparkHot: 0xffffff,
    sparkCool: 0xff3c08,
    ring: 0xfff0d0,
    dust: 0x77736c,
    spread: 0.82,
  },
  uniform: {
    // Navy wool, buff leather, gilt lace, and the brass off a shako plate.
    fragments: [0x1d2a4a, 0x2b3a5e, 0x6b4a2c, 0xb08a3c, 0x8e1f22],
    fragmentRoughness: 0.78,
    fragmentMetalness: 0.18,
    fragmentCount: 14,
    fragmentSize: 0.05,
    fragmentSliver: 2.8,
    // Cloth and leather land where they fall.
    bounce: 0.16,
    sparkCount: 7,
    sparkSpeed: 6,
    sparkLife: 0.3,
    sparkHot: 0xffe9b6,
    sparkCool: 0xff4a10,
    ring: 0xffd79a,
    dust: 0x6d6355,
    spread: 1.4,
  },
  flagstone: {
    fragments: [0x9a917f, 0x7d7466, 0x5d564b],
    fragmentRoughness: 0.9,
    fragmentMetalness: 0.04,
    fragmentCount: 13,
    fragmentSize: 0.055,
    fragmentSliver: 1.9,
    bounce: 0.44,
    sparkCount: 22,
    sparkSpeed: 9,
    sparkLife: 0.55,
    sparkHot: 0xfff6dd,
    sparkCool: 0xff4708,
    ring: 0xffca8a,
    dust: 0xa79d8a,
    spread: 1.5,
  },
};

/** The haze colour a caller should use for its own smoke over the hit. */
export function impactDust(body: ImpactBody): number {
  return BODIES[body].dust;
}

// ------------------------------------------------------------------ moulds

const FORWARD = new THREE.Vector3(0, 0, 1);

let chipGeometry: THREE.BufferGeometry | null = null;
let sliverGeometry: THREE.BufferGeometry | null = null;
let ringGeometry: THREE.PlaneGeometry | null = null;
let ringMap: THREE.CanvasTexture | null = null;

/**
 * One broken chip: a crushed tetrahedron. Four faces is all a fragment tumbling
 * past the camera for half a second can possibly show, and the irregular
 * silhouette is what stops a debris field looking like spilled dice.
 */
function sharedChipGeometry(): THREE.BufferGeometry {
  if (!chipGeometry) {
    const chip = new THREE.TetrahedronGeometry(0.5, 0);
    const position = chip.getAttribute("position") as THREE.BufferAttribute;
    const vertex = new THREE.Vector3();
    for (let i = 0; i < position.count; i += 1) {
      vertex.fromBufferAttribute(position, i);
      // Squashed off true so no two faces catch the torches equally.
      vertex.set(vertex.x * 1.15, vertex.y * 0.72, vertex.z * 0.94);
      position.setXYZ(i, vertex.x, vertex.y, vertex.z);
    }
    chip.computeVertexNormals();
    chipGeometry = chip;
  }
  return chipGeometry;
}

/**
 * The body of a spark: a four-sided sliver one unit long down +Z, so it can be
 * pointed along its own velocity and scaled by how fast it is going. Four faces
 * rather than a plane means it never disappears edge-on to the camera.
 */
function sharedSliverGeometry(): THREE.BufferGeometry {
  if (!sliverGeometry) {
    const sliver = new THREE.CylinderGeometry(0.5, 0.16, 1, 4, 1, false);
    sliver.rotateX(Math.PI / 2);
    sliverGeometry = sliver;
  }
  return sliverGeometry;
}

function sharedRingGeometry(): THREE.PlaneGeometry {
  if (!ringGeometry) ringGeometry = new THREE.PlaneGeometry(1, 1);
  return ringGeometry;
}

function sharedRingMap(): THREE.CanvasTexture {
  if (!ringMap) ringMap = shockwaveTexture();
  return ringMap;
}

/** Frees the shared moulds and maps (scene teardown). */
export function disposeShatterAssets(): void {
  chipGeometry?.dispose();
  sliverGeometry?.dispose();
  ringGeometry?.dispose();
  ringMap?.dispose();
  chipGeometry = null;
  sliverGeometry = null;
  ringGeometry = null;
  ringMap = null;
}

// ------------------------------------------------------------------ scatter

/**
 * A direction inside a cone about `axis`. Sampling the cosine rather than the
 * angle keeps the scatter even across the cap instead of bunching it on the
 * axis, which is the difference between spall and a firework.
 */
function inCone(axis: THREE.Vector3, spread: number, out: THREE.Vector3): THREE.Vector3 {
  const cosLimit = Math.cos(Math.min(Math.PI, spread));
  const z = cosLimit + Math.random() * (1 - cosLimit);
  const radius = Math.sqrt(Math.max(0, 1 - z * z));
  const phi = Math.random() * Math.PI * 2;
  out.set(Math.cos(phi) * radius, Math.sin(phi) * radius, z);
  return out.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(FORWARD, axis));
}

/** A tumbling chip of the victim. */
interface Fragment {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  quaternion: THREE.Quaternion;
  axis: THREE.Vector3;
  spin: number;
  scale: THREE.Vector3;
  life: number;
  maxLife: number;
  /** Set once it has stopped moving, so it is no longer integrated. */
  resting: boolean;
}

/** One spark: geometry stretched along the way it is going. */
interface Spark {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  thickness: number;
  life: number;
  maxLife: number;
  /** Individual flicker phase — sparks do not fade smoothly, they gutter. */
  flicker: number;
}

export interface ShatterOptions {
  /** What the round hit, which decides everything about the debris. */
  body: ImpactBody;
  /** Unit vector along the round's line of travel at the moment of arrival. */
  along: THREE.Vector3;
  /**
   * How hard it arrived. 1 is a musket ball in a man; a pistol is around 0.6
   * and a six-pounder round shot is over 2. Scales count, speed and size
   * together, so the ranks stay honestly ordered.
   */
  power: number;
  /** Board surface height, so fragments and sparks can bounce off it. */
  floor: number;
  /**
   * The round carried through rather than staying in the body, so there is an
   * exit as well as an entry: a second, wider, slower cone of debris going the
   * way the round was already going.
   */
  through?: boolean;
  /** Hard cap on instances, off the graphics preset. */
  budget: number;
  /** A slot borrowed from the scene's light pool, or null to run unlit. */
  light?: SpellLight | null;
}

/**
 * Breaks a body open on the frame the round arrives, and animates the wreckage
 * until the last chip stops moving.
 *
 * Resolves when the debris is gone, so a caller can fire it with `void` and
 * carry straight on with the death beat — the shatter outlives the hitstop on
 * purpose, and is still settling while the body goes down.
 */
export async function spawnImpactShatter(
  scene: THREE.Object3D,
  tweens: TweenManager,
  at: THREE.Vector3,
  options: ShatterOptions,
): Promise<void> {
  const recipe = BODIES[options.body];
  const power = Math.max(0.2, options.power);
  const budget = Math.max(6, options.budget);

  // Everything is thrown back out of the hole the round made — spall comes off
  // the face the projectile struck, not the far side.
  const back = options.along.clone().negate().normalize();
  const group = new THREE.Group();
  group.name = `shatter_${options.body}`;
  scene.add(group);

  // ---- the punch ring: where the round went in --------------------------
  const ringMaterial = new THREE.MeshBasicMaterial({
    map: sharedRingMap(),
    color: recipe.ring,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    opacity: 1,
    side: THREE.DoubleSide,
  });
  const ring = new THREE.Mesh(sharedRingGeometry(), ringMaterial);
  ring.quaternion.setFromUnitVectors(FORWARD, back);
  ring.position.copy(at).addScaledVector(back, 0.02);
  ring.renderOrder = 9;
  ring.frustumCulled = false;
  const ringSize = 0.3 + power * 0.42;
  ring.scale.setScalar(ringSize * 0.25);
  group.add(ring);

  // ---- sparks ----------------------------------------------------------
  const sparkWanted = Math.round(recipe.sparkCount * (0.55 + power * 0.55));
  const sparkTotal = Math.min(sparkWanted, Math.round(budget * 0.6));
  const sparks: Spark[] = [];
  const scratch = new THREE.Vector3();
  for (let i = 0; i < sparkTotal; i += 1) {
    const direction = inCone(back, recipe.spread, scratch.clone());
    const speed = recipe.sparkSpeed * (0.35 + Math.random() * 0.95) * (0.7 + power * 0.4);
    sparks.push({
      position: at.clone().addScaledVector(direction, 0.02 + Math.random() * 0.04),
      velocity: direction.multiplyScalar(speed),
      thickness: (0.006 + Math.random() * 0.009) * (0.8 + power * 0.3),
      life: 0,
      // A handful of sparks always outlive the rest and skitter off the stone;
      // an even spread of lifetimes makes the shower stop like a switch.
      maxLife: recipe.sparkLife * (Math.random() > 0.82 ? 1.7 + Math.random() : 0.4 + Math.random() * 0.7),
      flicker: Math.random() * Math.PI * 2,
    });
  }

  const sparkMaterial = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    opacity: 1,
  });
  const sparkMesh =
    sparks.length > 0 ? new THREE.InstancedMesh(sharedSliverGeometry(), sparkMaterial, sparks.length) : null;
  if (sparkMesh) {
    sparkMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    sparkMesh.frustumCulled = false;
    sparkMesh.renderOrder = 8;
    // Per-instance colour is the only way to cool each spark on its own clock;
    // under additive blending, driving it toward black is also how one dies.
    sparkMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(sparks.length * 3), 3);
    sparkMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    group.add(sparkMesh);
  }

  // ---- fragments -------------------------------------------------------
  const fragmentWanted = Math.round(recipe.fragmentCount * (0.5 + power * 0.6));
  const fragmentTotal = Math.min(fragmentWanted, Math.round(budget * 0.45));
  const fragments: Fragment[] = [];
  for (let i = 0; i < fragmentTotal; i += 1) {
    // Entry spall is tight and fast; the exit side, when there is one, throws
    // more mass, more slowly, over a wider angle.
    const exit = options.through === true && Math.random() > 0.55;
    const axis = exit ? options.along : back;
    const direction = inCone(axis, recipe.spread * (exit ? 1.5 : 1), new THREE.Vector3());
    const speed = (exit ? 2.1 : 3.4) * (0.3 + Math.random() * 1.1) * (0.65 + power * 0.5);
    const bulk = recipe.fragmentSize * (0.45 + Math.random() * 0.95) * (0.7 + power * 0.45);
    fragments.push({
      position: at.clone().addScaledVector(direction, 0.03),
      velocity: direction.multiplyScalar(speed).addScaledVector(new THREE.Vector3(0, 1, 0), 0.6 + Math.random()),
      quaternion: new THREE.Quaternion().random(),
      axis: new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).normalize(),
      spin: (Math.random() > 0.5 ? 1 : -1) * (7 + Math.random() * 16),
      scale: new THREE.Vector3(bulk, bulk * (0.4 + Math.random() * 0.4), bulk * recipe.fragmentSliver * (0.6 + Math.random() * 0.7)),
      life: 0,
      maxLife: 1.1 + Math.random() * 0.9,
      resting: false,
    });
  }

  const fragmentMaterial = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: recipe.fragmentRoughness,
    metalness: recipe.fragmentMetalness,
    // A chip crossing a dark hall needs a floor under its shading or it reads as
    // a hole in the frame rather than as debris.
    emissive: new THREE.Color(0x14161a),
    emissiveIntensity: 1,
  });
  const fragmentMesh =
    fragments.length > 0
      ? new THREE.InstancedMesh(sharedChipGeometry(), fragmentMaterial, fragments.length)
      : null;
  if (fragmentMesh) {
    fragmentMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    fragmentMesh.frustumCulled = false;
    fragmentMesh.castShadow = false;
    const colours = new Float32Array(fragments.length * 3);
    const tint = new THREE.Color();
    for (let i = 0; i < fragments.length; i += 1) {
      tint.setHex(recipe.fragments[Math.floor(Math.random() * recipe.fragments.length)]);
      // A little value scatter so a dozen chips of one material still read as a
      // dozen different pieces.
      tint.multiplyScalar(0.78 + Math.random() * 0.44);
      colours[i * 3] = tint.r;
      colours[i * 3 + 1] = tint.g;
      colours[i * 3 + 2] = tint.b;
    }
    fragmentMesh.instanceColor = new THREE.InstancedBufferAttribute(colours, 3);
    group.add(fragmentMesh);
  }

  // ---- the wreckage in motion ------------------------------------------
  const life = Math.max(
    0.45,
    ...sparks.map((spark) => spark.maxLife),
    ...fragments.map((fragment) => fragment.maxLife),
  );
  const matrix = new THREE.Matrix4();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  const colour = new THREE.Color();
  const hot = new THREE.Color(recipe.sparkHot);
  const cool = new THREE.Color(recipe.sparkCool);
  const spin = new THREE.Quaternion();
  let last = 0;

  try {
    await tweens.to({
      duration: life,
      easing: Ease.linear,
      onUpdate: (t) => {
        const delta = Math.max(0, Math.min(0.05, (t - last) * life));
        last = t;

        // The ring is over almost before it starts — it is a read, not an effect.
        const ringLife = Math.min(1, t * life / 0.18);
        ringMaterial.opacity = Math.pow(1 - ringLife, 1.7);
        ring.scale.setScalar(ringSize * (0.25 + ringLife * 1.15));

        if (sparkMesh) {
          const attribute = sparkMesh.instanceColor as THREE.InstancedBufferAttribute;
          for (let i = 0; i < sparks.length; i += 1) {
            const spark = sparks[i];
            spark.life += delta;
            const age = Math.min(1, spark.life / spark.maxLife);
            if (age >= 1) {
              // Additive black is invisible; no need to shuffle the instance list.
              matrix.makeScale(0, 0, 0);
              sparkMesh.setMatrixAt(i, matrix);
              attribute.setXYZ(i, 0, 0, 0);
              continue;
            }
            // Sparks are tiny and hot: they slow hard in air and fall fast.
            spark.velocity.multiplyScalar(Math.max(0, 1 - delta * 3.2));
            spark.velocity.y -= 11 * delta;
            spark.position.addScaledVector(spark.velocity, delta);
            if (spark.position.y < options.floor) {
              // Skittering off the flagstones: it keeps its run, loses its lift.
              spark.position.y = options.floor + 0.002;
              spark.velocity.y = Math.abs(spark.velocity.y) * 0.32;
              spark.velocity.x *= 0.7;
              spark.velocity.z *= 0.7;
            }
            const speed = spark.velocity.length();
            if (speed < 1e-4) {
              matrix.makeScale(0, 0, 0);
              sparkMesh.setMatrixAt(i, matrix);
              attribute.setXYZ(i, 0, 0, 0);
              continue;
            }
            scratch.copy(spark.velocity).divideScalar(speed);
            quaternion.setFromUnitVectors(FORWARD, scratch);
            // The streak *is* the motion: length tracks speed, so a spark draws
            // a long line while it is quick and a dot once it is spent.
            const length = Math.min(0.42, 0.012 + speed * 0.02);
            scale.set(spark.thickness, spark.thickness, length);
            matrix.compose(spark.position, quaternion, scale);
            sparkMesh.setMatrixAt(i, matrix);

            // White-hot to dull red, guttering as it cools.
            colour.copy(hot).lerp(cool, Math.pow(age, 0.55));
            const gutter = 0.72 + 0.28 * Math.sin(spark.flicker + spark.life * 47);
            colour.multiplyScalar(Math.pow(1 - age, 1.35) * gutter * 2.2);
            attribute.setXYZ(i, colour.r, colour.g, colour.b);
          }
          sparkMesh.instanceMatrix.needsUpdate = true;
          attribute.needsUpdate = true;
        }

        if (fragmentMesh) {
          for (let i = 0; i < fragments.length; i += 1) {
            const fragment = fragments[i];
            fragment.life += delta;
            const age = Math.min(1, fragment.life / fragment.maxLife);
            if (!fragment.resting) {
              fragment.velocity.multiplyScalar(Math.max(0, 1 - delta * 0.9));
              fragment.velocity.y -= 15 * delta;
              fragment.position.addScaledVector(fragment.velocity, delta);
              spin.setFromAxisAngle(fragment.axis, fragment.spin * delta);
              fragment.quaternion.premultiply(spin);
              if (fragment.position.y < options.floor + fragment.scale.y * 0.5) {
                fragment.position.y = options.floor + fragment.scale.y * 0.5;
                if (Math.abs(fragment.velocity.y) < 0.45) {
                  // Come to rest lying on the stone rather than jittering.
                  fragment.resting = true;
                  fragment.velocity.set(0, 0, 0);
                } else {
                  fragment.velocity.y = Math.abs(fragment.velocity.y) * recipe.bounce;
                  // Friction on the tangent, and the tumble is knocked down too.
                  fragment.velocity.x *= 0.62;
                  fragment.velocity.z *= 0.62;
                  fragment.spin *= 0.55;
                }
              }
            }
            // Held full size, then pulled under in the last quarter of its life,
            // so the board is never left littered.
            const shrink = age < 0.72 ? 1 : Math.pow(1 - (age - 0.72) / 0.28, 0.8);
            scale.copy(fragment.scale).multiplyScalar(shrink);
            matrix.compose(fragment.position, fragment.quaternion, scale);
            fragmentMesh.setMatrixAt(i, matrix);
          }
          fragmentMesh.instanceMatrix.needsUpdate = true;
        }

        // The hole itself throws light for a moment: bright on the first frame,
        // gone within a fifth of a second, like every real spark shower.
        options.light?.set(at, Math.pow(1 - ringLife, 2) * 14 * power);
      },
    });
  } finally {
    options.light?.release();
    ringMaterial.dispose();
    sparkMaterial.dispose();
    fragmentMaterial.dispose();
    sparkMesh?.dispose();
    fragmentMesh?.dispose();
    group.removeFromParent();
    group.clear();
  }
}
