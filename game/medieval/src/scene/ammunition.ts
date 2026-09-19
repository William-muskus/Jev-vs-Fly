/**
 * The magazine of the Grande Armée: one round per barrel, forged rather than
 * faked.
 *
 * Every gun on the board used to fire the same glowing dot, which is the one
 * thing black powder never did — nothing an 1805 barrel spat out was a tracer.
 * What actually left those bores was cold grey lead or hot black iron, and each
 * of the four rounds here is built to its own period drawing:
 *
 * - **pistol ball** — a small cast lead sphere off the officer's flintlock,
 *   mould seam and sprue scar still on it.
 * - **musket ball** — the line's .69 calibre ball: bigger, softer, dented by
 *   the ramrod, and unstabilised, so it wanders on the way over.
 * - **Minié bullet** — the marksman's rifled round: a conical ogive with three
 *   grease grooves and a hollow base, spun up by the rifling and dead stable.
 * - **round shot** — the battery's solid iron ball: pitted from the sand mould,
 *   still glowing from the bore, and heavy enough to keep going past the body.
 *
 * Every round is authored to the same contract so a shot only has to scale it:
 * **nose along +Z, centred on its own middle, exactly one world unit from nose
 * to base.** Geometry and materials are cached per kind and cloned per shot;
 * only a round that carries heat gets its own material so it can cool in
 * flight.
 */

import * as THREE from "three";

import type { StreakLook } from "./tracer";

/** Which round a barrel is loaded with. */
export type AmmoKind = "pistolBall" | "musketBall" | "minieBullet" | "roundShot";

/** How one round behaves once it is out of the bore. */
export interface AmmoSpec {
  /** Period name, for the record. */
  label: string;
  /**
   * Nose-to-base length as a multiple of the bore diameter. A cast ball is 1
   * (it is a sphere); the Minié bullet is nearly twice as long as it is wide.
   */
  length: number;
  /**
   * How much larger than life the round is drawn, as a multiple of the bore.
   *
   * A .69 ball is a fiftieth of a man's height: rendered true to scale on a
   * board this size it is one or two pixels, and a shot simply *cannot be seen*.
   * This is the one deliberate lie in the magazine — the round is drawn at a
   * legible gauge (still far smaller than a fist beside the figures) while its
   * flight path, wander and spin stay on the real numbers.
   */
  gauge: number;
  /**
   * Rifling. A stabilised round spins about its own nose axis and stays
   * pointing where it was sent; a ball out of a smoothbore tumbles on whatever
   * axis it happened to leave with.
   */
  stabilised: boolean;
  /** Turn rate in radians per world unit travelled. */
  twist: number;
  /**
   * How far the round wanders off the line of sight at mid-flight, in bore
   * diameters. This is the whole reason a smoothbore musket could not hit a man
   * at a hundred paces, and the rifled round is the only one here with none.
   */
  wander: number;
  /**
   * Heat still in the round as it leaves the muzzle. Lead is cold by the time
   * it is visible; a 6-pounder shot comes out of the bore dull red and cools
   * across the hall.
   */
  heat: number;
  /**
   * The motion smear *on the metal itself*: colour, strength, and its length in
   * rendered ball diameters. This is blur, not fire — no period round was a
   * tracer, so it is grey for lead and only warm for iron out of a hot bore.
   * It rides with the round; the line of flight is carried by {@link trail}.
   */
  streak: { color: number; opacity: number; stretch: number };
  /**
   * The short streak left along the path the round actually flew — the one thing
   * that lets the eye follow a shot from the bore to the body rather than only
   * see it arrive. Built as real geometry in `tracer.ts`, so it bends where a
   * smoothbore ball bellies off the line of sight.
   */
  trail: StreakLook;
  /**
   * Torchlight caught on the metal as it turns, as a fraction of full white.
   * Without it, cold lead crossing a dark hall reads as nothing at all.
   */
  glint: number;
  /** Air dragged along behind a heavy round, in bore diameters. 0 for small arms. */
  wake: number;
  /**
   * How violently the round breaks what it hits, as a multiple of a musket
   * ball. Drives the debris field at the far end — the spark shower, the number
   * of chips thrown and the size of the punch ring. This is mass times speed,
   * not spectacle: a pistol ball is under one, a six-pounder is well over two.
   */
  shatter: number;
  /**
   * Whether the round carries through the body instead of staying in it. Soft
   * lead at black-powder velocity flattens and stops; a spun conical bullet and
   * a solid iron shot both come out the other side, and that is what puts spall
   * on the far side of the man as well as the near one.
   */
  through: boolean;
}

/**
 * The four rounds, in order of what they weigh. Note how little glow the small
 * arms carry: lead is read by its shape and its motion smear, not by light.
 */
export const AMMUNITION: Record<AmmoKind, AmmoSpec> = {
  // .58 cast lead, off an officer's flintlock. Light, quick, barely visible.
  pistolBall: {
    label: "cast lead pistol ball",
    length: 1,
    gauge: 2.5,
    stabilised: false,
    twist: 5,
    wander: 0.9,
    heat: 0,
    streak: { color: 0xc9ced6, opacity: 0.34, stretch: 7 },
    // The lightest round on the board leaves the least behind it: barely half a
    // square of thin, cold air.
    trail: { span: 5, width: 0.6, color: 0xb7c0cb, core: 0xe9eff7, strength: 0.3 },
    glint: 0.4,
    wake: 0,
    shatter: 0.72,
    through: false,
  },
  // .69 Charleville ball. The heaviest thing the line carries and the least
  // accurate round on the board.
  musketBall: {
    label: ".69 Charleville musket ball",
    length: 1,
    gauge: 2.3,
    stabilised: false,
    twist: 4,
    wander: 1.6,
    heat: 0,
    streak: { color: 0xc2c7ce, opacity: 0.4, stretch: 8.5 },
    // Fat, grey and visibly curved: this is the round whose wander the streak was
    // worth building geometry for.
    trail: { span: 5.6, width: 0.74, color: 0xb2bac5, core: 0xe6edf5, strength: 0.36 },
    glint: 0.42,
    wake: 0,
    // The fattest small-arms round on the board, and soft enough to stay in.
    shatter: 1,
    through: false,
  },
  // Rifled: conical, spun hard, and the only round that goes exactly where it
  // is pointed.
  minieBullet: {
    label: "Minié bullet",
    length: 1.9,
    gauge: 2.6,
    stabilised: true,
    // Spun about its own nose, so the grease grooves flicker rather than the
    // whole round turning over — the mark of a rifled barrel.
    twist: 22,
    wander: 0,
    heat: 0,
    streak: { color: 0xdde1e7, opacity: 0.46, stretch: 11 },
    // The longest, thinnest streak in the army and the only straight one: a
    // rifled round drawing a wire across the hall.
    trail: { span: 9, width: 0.48, color: 0xccd6e2, core: 0xf4f8ff, strength: 0.42 },
    glint: 0.5,
    wake: 0,
    // Lighter than a musket ball but arriving far faster and still spinning.
    shatter: 1.24,
    through: true,
  },
  // Solid iron, out of a 6-pounder. Slow enough to watch, hot enough to see.
  roundShot: {
    label: "6-pounder round shot",
    length: 1,
    // Already the biggest thing fired on the board: it needs the least help.
    gauge: 1.7,
    stabilised: false,
    twist: 2.4,
    wander: 0.35,
    heat: 1,
    streak: { color: 0xff9a52, opacity: 0.52, stretch: 6 },
    // Short, wide and hot — iron still glowing from the bore drags a bank of
    // scorched air behind it rather than a thread.
    trail: { span: 4.2, width: 1, color: 0xff7f36, core: 0xffd9a4, strength: 0.5 },
    glint: 0.3,
    wake: 2.4,
    // Six pounds of iron. Nothing on the board survives being in its way, and
    // it is still travelling once it is through.
    shatter: 2.5,
    through: true,
  },
};

// ------------------------------------------------------------------ metals

let leadMaterial: THREE.MeshStandardMaterial | null = null;
let ironMaterial: THREE.MeshStandardMaterial | null = null;
const geometries: THREE.BufferGeometry[] = [];

/**
 * Unpolished cast lead: nearly no colour of its own and dull under a torch, but
 * kept off full metalness on purpose. A mirror-metal sphere a few pixels across
 * has nothing to reflect in a dark hall and renders as a black dot; a rougher,
 * lighter lead catches the torches and stays visible while it crosses.
 */
function lead(): THREE.MeshStandardMaterial {
  if (!leadMaterial) {
    leadMaterial = new THREE.MeshStandardMaterial({
      color: 0xb4bac2,
      metalness: 0.62,
      roughness: 0.44,
      // Never fully black, even with every torch behind it.
      emissive: new THREE.Color(0x2c3138),
      emissiveIntensity: 1,
    });
    leadMaterial.envMapIntensity = 1.3;
  }
  return leadMaterial;
}

/** Sand-cast iron: near black, matt, and the only round that glows. */
function iron(): THREE.MeshStandardMaterial {
  if (!ironMaterial) {
    ironMaterial = new THREE.MeshStandardMaterial({
      color: 0x3b3936,
      metalness: 0.68,
      roughness: 0.72,
      emissive: new THREE.Color(0xff5a1e),
      emissiveIntensity: 0,
    });
    ironMaterial.envMapIntensity = 1;
  }
  return ironMaterial;
}

function track<T extends THREE.BufferGeometry>(geometry: T): T {
  geometries.push(geometry);
  return geometry;
}

// ------------------------------------------------------------------ founding

/** A cheap, stable hash — the mould marks must be the same every time. */
function hash(i: number): number {
  const x = Math.sin(i * 127.1 + 0.5) * 43758.5453;
  return x - Math.floor(x);
}

/**
 * A ball out of a two-part mould: not quite round, a raised seam where the
 * halves met, and the stub of the sprue where it was nipped off the gate.
 *
 * @param dents how far the surface strays from a true sphere (ramrod and
 *   cartridge wear on a soft lead ball)
 */
function castBall(dents: number): THREE.BufferGeometry[] {
  const ball = track(new THREE.SphereGeometry(0.5, 20, 14));
  const position = ball.getAttribute("position") as THREE.BufferAttribute;
  const vertex = new THREE.Vector3();
  for (let i = 0; i < position.count; i += 1) {
    vertex.fromBufferAttribute(position, i);
    // Soft, low-frequency deformation: lead squashes, it does not chip.
    const wobble = 1 + (hash(i * 3.7) - 0.5) * dents;
    vertex.multiplyScalar(wobble);
    position.setXYZ(i, vertex.x, vertex.y * 0.985, vertex.z);
  }
  ball.computeVertexNormals();
  // The mould seam, standing a hair proud of the surface.
  const seam = track(new THREE.TorusGeometry(0.495, 0.013, 5, 26));
  // The sprue scar: cut flush, so a stub rather than a spike.
  const sprue = track(new THREE.CylinderGeometry(0.07, 0.085, 0.05, 8));
  sprue.translate(0, 0.5, 0);
  return [ball, seam, sprue];
}

/**
 * The Minié round in profile, turned on a lathe exactly as it was pressed: a
 * long ogive nose, a bearing body cut by three grease grooves (they carried the
 * tallow that kept the fouling soft), and the hollow base whose skirt the
 * charge blew out into the rifling.
 */
function minieProfile(): THREE.BufferGeometry {
  const halfWidth = 0.263;
  const groove = halfWidth - 0.036;
  const points: THREE.Vector2[] = [
    new THREE.Vector2(0.0, 0.0),
    new THREE.Vector2(halfWidth - 0.03, 0.0),
    // The skirt: thin-walled, slightly flared, the mark of a hollow base.
    new THREE.Vector2(halfWidth, 0.028),
    new THREE.Vector2(halfWidth, 0.15),
    new THREE.Vector2(groove, 0.181),
    new THREE.Vector2(halfWidth, 0.212),
    new THREE.Vector2(halfWidth, 0.3),
    new THREE.Vector2(groove, 0.331),
    new THREE.Vector2(halfWidth, 0.362),
    new THREE.Vector2(halfWidth, 0.45),
    new THREE.Vector2(groove, 0.481),
    new THREE.Vector2(halfWidth, 0.512),
    // Bearing surface up to the shoulder, then the ogive.
    new THREE.Vector2(halfWidth - 0.004, 0.6),
    new THREE.Vector2(halfWidth - 0.014, 0.68),
    new THREE.Vector2(halfWidth - 0.036, 0.76),
    new THREE.Vector2(halfWidth - 0.072, 0.84),
    new THREE.Vector2(halfWidth - 0.122, 0.91),
    new THREE.Vector2(halfWidth - 0.19, 0.966),
    new THREE.Vector2(0.0, 1.0),
  ];
  const bullet = track(new THREE.LatheGeometry(points, 22));
  // Authored base-to-nose along +Y; the contract is nose along +Z, centred.
  bullet.translate(0, -0.5, 0);
  bullet.rotateX(Math.PI / 2);
  bullet.computeVertexNormals();
  return bullet;
}

/**
 * Solid shot out of a sand mould: pitted all over, with the casting seam still
 * round its middle. The pits are what make it read as iron rather than as a
 * smooth game-engine sphere.
 */
function solidShot(): THREE.BufferGeometry[] {
  const shot = track(new THREE.IcosahedronGeometry(0.5, 3));
  const position = shot.getAttribute("position") as THREE.BufferAttribute;
  const vertex = new THREE.Vector3();
  for (let i = 0; i < position.count; i += 1) {
    vertex.fromBufferAttribute(position, i);
    const pit = hash(i * 5.13);
    // Most of the surface is merely uneven; one vertex in six is a real pit.
    const depth = pit > 0.84 ? 0.052 * (pit - 0.84) / 0.16 : 0;
    vertex.multiplyScalar(1 - depth + (hash(i * 2.91) - 0.5) * 0.018);
    position.setXYZ(i, vertex.x, vertex.y, vertex.z);
  }
  shot.computeVertexNormals();
  const seam = track(new THREE.TorusGeometry(0.492, 0.011, 5, 30));
  return [shot, seam];
}

interface Forged {
  geometries: THREE.BufferGeometry[];
  material: THREE.MeshStandardMaterial;
}

const forges: Partial<Record<AmmoKind, Forged>> = {};

function forge(kind: AmmoKind): Forged {
  const existing = forges[kind];
  if (existing) return existing;
  let made: Forged;
  switch (kind) {
    case "pistolBall":
      made = { geometries: castBall(0.028), material: lead() };
      break;
    case "musketBall":
      // A soft ball rammed down a fouled barrel takes a beating.
      made = { geometries: castBall(0.05), material: lead() };
      break;
    case "minieBullet":
      made = { geometries: [minieProfile()], material: lead() };
      break;
    case "roundShot":
      made = { geometries: solidShot(), material: iron() };
      break;
  }
  forges[kind] = made;
  return made;
}

/** A round built for one shot, plus the parts of it that are still hot. */
export interface Round {
  object: THREE.Object3D;
  /** Materials whose glow has to cool over the flight; empty for cold lead. */
  heated: THREE.MeshStandardMaterial[];
}

/**
 * Builds one round of the given kind, normalised nose-along-+Z and one unit
 * long, ready to be scaled by the gun's calibre. A round that carries heat gets
 * its own material clone so its glow can be faded without touching any other
 * shot in the air.
 */
export function loadRound(kind: AmmoKind): Round {
  const spec = AMMUNITION[kind];
  const { geometries: parts, material } = forge(kind);
  const hot = spec.heat > 0;
  const shared = hot ? (material.clone() as THREE.MeshStandardMaterial) : material;
  const group = new THREE.Group();
  group.name = `round_${kind}`;
  for (const geometry of parts) {
    const mesh = new THREE.Mesh(geometry, shared);
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    // A round crossing the frame in a tenth of a second must never be culled by
    // a bounding sphere that is one frame stale.
    mesh.frustumCulled = false;
    group.add(mesh);
  }
  return { object: group, heated: hot ? [shared] : [] };
}

/** Frees the shared metals and moulds (scene teardown). */
export function disposeAmmunition(): void {
  for (const geometry of geometries) geometry.dispose();
  geometries.length = 0;
  leadMaterial?.dispose();
  ironMaterial?.dispose();
  leadMaterial = null;
  ironMaterial = null;
  for (const kind of Object.keys(forges) as AmmoKind[]) delete forges[kind];
}
