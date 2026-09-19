/**
 * Black powder: what a shot looks like when it leaves a barrel.
 *
 * The Grande Armée does not fight with witchfire. Its officers, its line
 * infantry and its batteries kill at range with a flash at the muzzle, a cloud
 * of dirty white smoke and a ball that crosses the board almost too fast to
 * follow. Everything here is additive billboards on the caller's tween clock —
 * built, animated and disposed inside one shot, so nothing needs a permanent
 * slot in the frame loop.
 *
 * Lights are always *borrowed* from the scene's shared pool, never created:
 * three.js keys its shader programs on the scene's light count, so adding one
 * mid-fight recompiles every material in the hall (see {@link SpellLightPool}).
 */

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import type { Faction } from "../core/types";
import { AMMUNITION, type AmmoKind, type AmmoSpec, disposeAmmunition, loadRound } from "./ammunition";
import type { SpellLight } from "./spells";
import { fineSmokeTexture, muzzleFlashTexture, radialTexture, smokeTexture, tracerTexture } from "./textures";
import { TracerStreak } from "./tracer";

/** How one army's powder burns. Both sides use the same charge; only the
 * livery tint of the smoke differs, so a volley always reads as gunpowder
 * rather than as a spell. */
export interface GunLook {
  /** The flash at the bore. */
  flash: number;
  /** The bloom of burning powder pushed out ahead of the bore. */
  ball: number;
  /** Powder smoke rolling off the barrel. */
  smoke: number;
  /** Colour the flash throws into the hall. */
  light: number;
}

export const GUN_LOOK: Record<Faction, GunLook> = {
  w: { flash: 0xfff6dd, ball: 0xffe6b4, smoke: 0xcfd4dc, light: 0xffd9a0 },
  b: { flash: 0xfff1c8, ball: 0xffcf82, smoke: 0xc8bfae, light: 0xffb45e },
};

/** The frame a normalised ball is authored in: nose along +Z. */
const FORWARD = new THREE.Vector3(0, 0, 1);

let flashMap: THREE.CanvasTexture | null = null;
let ballMap: THREE.CanvasTexture | null = null;
let puffMap: THREE.CanvasTexture | null = null;
let finePuffMap: THREE.CanvasTexture | null = null;
let smearMap: THREE.CanvasTexture | null = null;
let smearGeometry: THREE.BufferGeometry | null = null;

function sharedFlashMap(): THREE.CanvasTexture {
  if (!flashMap) flashMap = muzzleFlashTexture();
  return flashMap;
}

function sharedBallMap(): THREE.CanvasTexture {
  if (!ballMap) ballMap = radialTexture("rgba(255,255,255,1)", "rgba(255,190,110,0)");
  return ballMap;
}

function sharedPuffMap(): THREE.CanvasTexture {
  if (!puffMap) puffMap = smokeTexture();
  return puffMap;
}

/** The paler, threadier bloom a rifled barrel leaves. */
function sharedFinePuffMap(): THREE.CanvasTexture {
  if (!finePuffMap) finePuffMap = fineSmokeTexture();
  return finePuffMap;
}

function sharedSmearMap(): THREE.CanvasTexture {
  if (!smearMap) smearMap = tracerTexture();
  return smearMap;
}

/**
 * The body of a round's motion smear: a cone one unit long, wide and bright
 * where the metal is and tapering to nothing behind it. Authored with the wide
 * end at the origin and the tip down the negative flight axis, so a shot only
 * has to point it the way it is travelling and scale it by its own calibre.
 */
function sharedSmearGeometry(): THREE.BufferGeometry {
  if (!smearGeometry) {
    const cone = new THREE.CylinderGeometry(0.34, 0, 1, 10, 1, true);
    // Wide end onto the origin, tip hanging below, then the whole thing tipped
    // from the lathe's +Y onto the flight axis.
    cone.translate(0, -0.5, 0);
    cone.rotateX(Math.PI / 2);
    smearGeometry = cone;
  }
  return smearGeometry;
}

// ---------------------------------------------------------------- the ball

/** Named model axes, as reported by the generator for every sculpt. */
type AxisName = "positiveX" | "negativeX" | "positiveY" | "negativeY" | "positiveZ" | "negativeZ";

const AXES: Record<AxisName, THREE.Vector3> = {
  positiveX: new THREE.Vector3(1, 0, 0),
  negativeX: new THREE.Vector3(-1, 0, 0),
  positiveY: new THREE.Vector3(0, 1, 0),
  negativeY: new THREE.Vector3(0, -1, 0),
  positiveZ: new THREE.Vector3(0, 0, 1),
  negativeZ: new THREE.Vector3(0, 0, -1),
};

/** The generated projectile sculpt and the axes it was authored along. */
export interface ShotModelSource {
  url: string;
  /** Which round this sculpt is the real article for. */
  ammo: AmmoKind;
  /**
   * Which way the nose of the ball points in the sculpt's own frame, when the
   * generator reported one. A cast ball is a body of revolution, so it usually
   * comes back *directionless* — leave this out and the long axis measured off
   * the mesh is used as the nose instead, which is what a bullet's shape means.
   */
  front?: AxisName;
  /** The sculpt's own up axis. Only meaningful together with `front`. */
  up?: AxisName;
}

/**
 * Generated sculpts, keyed by the round they stand in for and normalised once:
 * nose down the flight line, centred on its own middle, and one world unit
 * long, so a shot only has to scale it by its calibre. A kind with no sculpt in
 * hand is forged procedurally instead (see `ammunition.ts`), so a slow download
 * never costs the army its ammunition.
 */
const sculpts = new Map<AmmoKind, THREE.Object3D>();
const sculptJobs = new Map<AmmoKind, Promise<void>>();

function basis(front: THREE.Vector3, up: THREE.Vector3): THREE.Quaternion {
  const f = front.clone().normalize();
  const r = new THREE.Vector3().crossVectors(up, f).normalize();
  const u = new THREE.Vector3().crossVectors(f, r).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(r, u, f));
}

/**
 * The axis a sculpt is longest along, measured off its own bounds. For a cast
 * ball that is the nose-to-base line by definition, which is why a directionless
 * projectile can still be flown nose-first without guessing a yaw constant.
 */
function longestAxis(model: THREE.Object3D): THREE.Vector3 {
  const size = new THREE.Vector3();
  new THREE.Box3().setFromObject(model).getSize(size);
  if (size.x >= size.y && size.x >= size.z) return new THREE.Vector3(1, 0, 0);
  if (size.y >= size.z) return new THREE.Vector3(0, 1, 0);
  return new THREE.Vector3(0, 0, 1);
}

/** Any unit vector square to the given one — enough to complete a basis. */
function perpendicular(axis: THREE.Vector3): THREE.Vector3 {
  return Math.abs(axis.y) > 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
}

/**
 * Fetches the generated ball and prepares it for flight. Called once when the
 * hall is built; failures are swallowed on purpose — a missing sculpt must never
 * cost the army its gunfire.
 */
export function primeShotModel(source: ShotModelSource): Promise<void> {
  const running = sculptJobs.get(source.ammo);
  if (running) return running;
  const job = (async () => {
    try {
      const gltf = await new GLTFLoader().loadAsync(source.url);
      // Rotate the sculpt's own frame onto "nose along +Z, up along +Y", which is
      // the frame a shot orients along its line of travel. A directionless ball
      // has no reported front, so its own longest extent is taken as the nose.
      const oriented = new THREE.Group();
      const front = source.front ? AXES[source.front] : longestAxis(gltf.scene);
      const up = source.up ? AXES[source.up] : perpendicular(front);
      const correction = basis(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 1, 0))
        .multiply(basis(front, up).invert());
      gltf.scene.quaternion.copy(correction);
      oriented.add(gltf.scene);

      // One unit from nose to base, centred on its own middle.
      const box = new THREE.Box3().setFromObject(oriented);
      const size = new THREE.Vector3();
      const centre = new THREE.Vector3();
      box.getSize(size);
      box.getCenter(centre);
      const length = Math.max(1e-4, size.z);
      gltf.scene.position.sub(centre);
      oriented.scale.setScalar(1 / length);

      oriented.traverse((node) => {
        const mesh = node as THREE.Mesh;
        if (!mesh.isMesh) return;
        mesh.castShadow = false;
        mesh.receiveShadow = false;
        // A ball in flight is a handful of pixels crossing the screen in a tenth
        // of a second; culling it by a stale bounding sphere makes it blink.
        mesh.frustumCulled = false;
        legible(mesh.material);
      });
      sculpts.set(source.ammo, oriented);
    } catch (error) {
      console.warn(`[gunfire] ${source.ammo} sculpt unavailable, forging it instead`, error);
    }
  })();
  sculptJobs.set(source.ammo, job);
  return job;
}

/**
 * Makes a sculpt's own metal readable at speed.
 *
 * A generated round comes back as a small, dark, near-mirror body. That is
 * physically fair and visually useless: with nothing to reflect in a torch-lit
 * hall it renders as a black dot a few pixels across and the shot looks like it
 * never happened. So the metal is pulled off full mirror, roughened, given a
 * floor of self-lit grey, and told to take the environment strongly.
 */
function legible(material: THREE.Material | THREE.Material[]): void {
  const list = Array.isArray(material) ? material : [material];
  for (const entry of list) {
    const metal = entry as THREE.MeshStandardMaterial;
    if (!metal.isMeshStandardMaterial) continue;
    metal.metalness = Math.min(metal.metalness, 0.6);
    metal.roughness = THREE.MathUtils.clamp(metal.roughness, 0.35, 0.7);
    metal.envMapIntensity = 1.3;
    // A floor under the shading, so the round never goes to pure black against
    // the far wall of the hall.
    metal.emissive = new THREE.Color(0x2c3138);
    metal.emissiveIntensity = 1;
    metal.needsUpdate = true;
  }
}

/**
 * Gives one shot its own copies of a sculpt's materials and returns them, so a
 * round that leaves the bore glowing can cool on its way across without dimming
 * every other shot of the same kind still in the air.
 */
function ownMetal(round: THREE.Object3D): THREE.MeshStandardMaterial[] {
  const owned: THREE.MeshStandardMaterial[] = [];
  round.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh || Array.isArray(mesh.material)) return;
    const metal = mesh.material as THREE.MeshStandardMaterial;
    if (!metal.isMeshStandardMaterial) return;
    const copy = metal.clone();
    copy.emissive = new THREE.Color(0xff5a1e);
    copy.emissiveIntensity = 0;
    mesh.material = copy;
    owned.push(copy);
  });
  return owned;
}

/** Frees the shared maps, moulds and metals (scene teardown). */
export function disposeGunAssets(): void {
  flashMap?.dispose();
  ballMap?.dispose();
  puffMap?.dispose();
  finePuffMap?.dispose();
  smearMap?.dispose();
  smearGeometry?.dispose();
  flashMap = null;
  ballMap = null;
  puffMap = null;
  finePuffMap = null;
  smearMap = null;
  smearGeometry = null;
  sculpts.clear();
  sculptJobs.clear();
  disposeAmmunition();
}

export interface MuzzleFlashOptions {
  look: GunLook;
  /** Width of the flash in world units — a pistol is a quarter of a field gun. */
  size: number;
  /** Where the barrel is pointing, so the flame leans out of the bore. */
  direction: THREE.Vector3;
  /** How long the flame is on screen. Powder burns for a frame or three. */
  life?: number;
  /** A slot borrowed from the scene's light pool, or null to fire unlit. */
  light?: SpellLight | null;
}

/**
 * The flash at the bore, in four stacked layers.
 *
 * It used to be two billboards fading out of a single sprite, which was legible
 * next to the old glowing-dot projectile but is nowhere near enough now that a
 * real sculpted round leaves the barrel drawn several calibres wide: the ball
 * was arriving *brighter than the charge that sent it*. So:
 *
 * 1. **The star** — the ragged petal texture, billboarded, the silhouette of the
 *    flame.
 * 2. **The core** — a small disc of flat white stacked additively on top of the
 *    star's own blown-out centre. Additive layers are how you get past opacity 1:
 *    this is what clips to white, and therefore the only part the bloom pass
 *    really takes hold of.
 * 3. **The jet** — a cone of flame down the line of fire, *not* billboarded, so
 *    the flash grows along the barrel instead of only swelling as a disc. This is
 *    what tells the eye which way the round just went.
 * 4. **The lead bloom** — the old forward puff, kept, a barrel's width out.
 *
 * The envelope matters as much as the size. Powder ignites in one frame, so the
 * whole stack is held at *full* brightness for the first fifth of its life
 * (`IGNITION`) and only then falls away — a flash that starts decaying on frame
 * one never registers at 60fps. The last beat carries a flicker, because a
 * charge burns out unevenly rather than dimming on a dial.
 */
const IGNITION = 0.2;

export async function spawnMuzzleFlash(
  scene: THREE.Object3D,
  tweens: { to: (spec: { duration: number; easing: (t: number) => number; onUpdate: (t: number) => void }) => Promise<void> },
  at: THREE.Vector3,
  options: MuzzleFlashOptions,
): Promise<void> {
  const life = options.life ?? 0.11;
  const size = options.size;
  const group = new THREE.Group();
  group.name = "muzzle_flash";
  group.position.copy(at);
  scene.add(group);

  const star = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: sharedFlashMap(),
      color: options.look.flash,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      opacity: 1,
      rotation: Math.random() * Math.PI,
    }),
  );
  star.scale.setScalar(size);
  star.renderOrder = 8;
  star.frustumCulled = false;

  // Flat white over the star's centre. Small on purpose: its job is to blow the
  // core out, not to widen the flame.
  const core = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: sharedBallMap(),
      color: 0xffffff,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      opacity: 1,
    }),
  );
  core.scale.setScalar(size * 0.46);
  core.position.copy(options.direction).multiplyScalar(size * 0.08);
  core.renderOrder = 11;
  core.frustumCulled = false;

  // The second bloom sits a barrel's width forward along the line of fire.
  const lead = new THREE.Sprite((star.material as THREE.SpriteMaterial).clone());
  (lead.material as THREE.SpriteMaterial).color.setHex(options.look.ball);
  lead.position.copy(options.direction).multiplyScalar(size * 0.34);
  lead.scale.setScalar(size * 0.62);
  lead.renderOrder = 9;
  lead.frustumCulled = false;

  // The vented charge itself: a cone whose wide, bright end sits on the bore and
  // whose tip runs out down the aim. The smear cone is authored tip-down -Z, so
  // mapping +Z onto the *reverse* of the aim points the tip where the ball went.
  const jet = new THREE.Mesh(
    sharedSmearGeometry(),
    new THREE.MeshBasicMaterial({
      map: sharedSmearMap(),
      color: options.look.flash,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      opacity: 0.9,
      side: THREE.DoubleSide,
    }),
  );
  jet.quaternion.setFromUnitVectors(FORWARD, options.direction.clone().negate());
  jet.scale.set(size * 0.8, size * 0.8, size * 1.5);
  jet.renderOrder = 10;
  jet.frustumCulled = false;

  group.add(star, jet, lead, core);

  const starMaterial = star.material as THREE.SpriteMaterial;
  const leadMaterial = lead.material as THREE.SpriteMaterial;
  const coreMaterial = core.material as THREE.SpriteMaterial;
  const jetMaterial = jet.material as THREE.MeshBasicMaterial;

  try {
    await tweens.to({
      duration: life,
      easing: (t: number) => t,
      onUpdate: (t: number) => {
        // Held wide open, then gone: brightness falls off much faster than size.
        const burn = t <= IGNITION ? 1 : Math.pow(1 - (t - IGNITION) / (1 - IGNITION), 2.1);
        // Uneven burn-out — a charge guttering, not a dimmer being turned down.
        const flicker = 0.82 + 0.18 * Math.abs(Math.sin(t * 47));
        const fade = burn * flicker;
        starMaterial.opacity = fade;
        leadMaterial.opacity = fade * 0.9;
        // The core is the last thing to widen and the first thing to die, which
        // is what makes the first frame read as a detonation.
        coreMaterial.opacity = Math.pow(burn, 1.6);
        jetMaterial.opacity = fade * 0.85;
        star.scale.setScalar(size * (1 + t * 0.6));
        lead.scale.setScalar(size * (0.62 + t * 0.8));
        core.scale.setScalar(size * (0.46 + t * 0.22));
        // The jet runs *out* as it dies rather than swelling: escaping gas.
        jet.scale.set(size * (0.8 + t * 0.5), size * (0.8 + t * 0.5), size * (1.5 + t * 1.1));
        starMaterial.rotation += 0.12;
        options.light?.set(group.position, fade * 38 * size);
      },
    });
  } finally {
    options.light?.release();
    starMaterial.dispose();
    leadMaterial.dispose();
    coreMaterial.dispose();
    jetMaterial.dispose();
    group.removeFromParent();
    group.clear();
  }
}

/**
 * One round in flight.
 *
 * The round itself is a real mesh — a sculpt when one has been fetched for that
 * kind, otherwise forged from `ammunition.ts` — and everything else on it is
 * there to make the metal legible at speed rather than to make it glow.
 *
 * Three things carry the read, in order of importance:
 *
 * 1. **The smear.** A cone of blurred metal on the nose of the round, brightest
 *    where the metal is and gone a couple of calibres behind it. It rides with
 *    the round; the path it has flown is drawn separately (see
 *    {@link TracerStreak}), which is what the eye actually follows.
 * 2. **The glint.** A small billboard of caught torchlight on the metal, so the
 *    round registers even against the dark far wall of the hall.
 * 3. **The heat.** Only iron out of a field gun: a dull glow that cools as it
 *    crosses, plus a bank of air dragged along behind the shot.
 *
 * It lives in world space and is placed by {@link flyShot} every frame.
 */
class Shot {
  readonly group = new THREE.Group();
  /** Motion smear cone, pointed down the line of travel. */
  private readonly smear: THREE.Mesh;
  /** Torchlight caught on the turning metal. */
  private readonly glint: THREE.Sprite;
  /** Heat still in the metal, for a round that left the bore glowing. */
  private readonly glow: THREE.Sprite | null;
  /** Air pulled along behind a heavy round. */
  private readonly wake: THREE.Sprite | null;
  private readonly light: SpellLight | null;
  /** The round itself. */
  private readonly round: THREE.Object3D;
  /** Materials whose glow has to cool as the round crosses. */
  private readonly heated: THREE.MeshStandardMaterial[];
  private readonly spec: AmmoSpec;
  /**
   * What the round turns about. A rifled bullet spins about its own nose; a ball
   * out of a smoothbore tumbles about whatever axis it happened to leave with.
   */
  private readonly axis: THREE.Vector3;
  private readonly spin: number;
  /** Rendered diameter of the round: the bore, opened up to a legible gauge. */
  readonly gauge: number;

  constructor(kind: AmmoKind, look: GunLook, size: number, light: SpellLight | null) {
    const spec = AMMUNITION[kind];
    this.spec = spec;
    this.light = light;
    this.group.name = `shot_${kind}`;
    const gauge = size * spec.gauge;
    this.gauge = gauge;
    const sculpt = sculpts.get(kind);
    if (sculpt) {
      this.round = sculpt.clone(true);
      // A sculpt shares its metal with every other shot of the same kind, so a
      // round that has to cool needs its own copy before its glow is touched.
      this.heated = spec.heat > 0 ? ownMetal(this.round) : [];
    } else {
      const forged = loadRound(kind);
      this.round = forged.object;
      this.heated = forged.heated;
    }
    // The mesh is one unit nose-to-base, so the rendered gauge and the round's
    // own proportions are all the scale it needs.
    this.round.scale.setScalar(gauge * spec.length);
    this.group.add(this.round);

    this.axis = spec.stabilised
      ? FORWARD.clone()
      : new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).normalize();
    this.spin = (Math.random() > 0.5 ? 1 : -1) * spec.twist * (0.85 + Math.random() * 0.3);

    this.smear = new THREE.Mesh(
      sharedSmearGeometry(),
      new THREE.MeshBasicMaterial({
        map: sharedSmearMap(),
        color: spec.streak.color,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        opacity: spec.streak.opacity,
        side: THREE.DoubleSide,
      }),
    );
    // Wide as the round at the nose, a couple of calibres long behind it.
    this.smear.scale.set(gauge, gauge, gauge * spec.streak.stretch * NOSE_BLUR);
    this.smear.renderOrder = 6;
    this.smear.frustumCulled = false;
    this.group.add(this.smear);

    this.glint = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: sharedBallMap(),
        color: spec.streak.color,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        opacity: spec.glint,
      }),
    );
    this.glint.scale.setScalar(gauge * 1.6);
    this.glint.renderOrder = 7;
    this.glint.frustumCulled = false;
    this.group.add(this.glint);

    if (spec.heat > 0) {
      this.glow = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: sharedBallMap(),
          color: 0xff7a2e,
          transparent: true,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
          opacity: 0.55 * spec.heat,
        }),
      );
      this.glow.scale.setScalar(gauge * 1.7);
      this.glow.renderOrder = 7;
      this.glow.frustumCulled = false;
      this.group.add(this.glow);
    } else {
      this.glow = null;
    }

    if (spec.wake > 0) {
      this.wake = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: sharedPuffMap(),
          color: look.smoke,
          transparent: true,
          depthWrite: false,
          opacity: 0.18,
        }),
      );
      this.wake.scale.setScalar(gauge * spec.wake);
      this.wake.renderOrder = 5;
      this.wake.frustumCulled = false;
      this.group.add(this.wake);
    } else {
      this.wake = null;
    }
  }

  /**
   * @param cooling 1 the instant it leaves the bore, 0 by the time it arrives —
   *   only iron carries enough heat for this to be visible.
   */
  place(at: THREE.Vector3, cooling: number): void {
    this.group.position.copy(at);
    const heat = this.spec.heat * (0.4 + cooling * 0.6);
    if (this.glow) {
      (this.glow.material as THREE.SpriteMaterial).opacity = 0.55 * heat;
      this.glow.scale.setScalar(this.gauge * (1.6 + cooling * 0.5));
    }
    for (const material of this.heated) material.emissiveIntensity = 1.15 * heat;
    this.light?.set(at, heat * 5);
  }

  /**
   * Points the round down its line of travel and turns it as it goes: a Minié
   * bullet rolls about its nose and stays pointing where it was sent, a cast
   * ball tumbles end over end about its own axis. The smear is laid along the
   * same line so it always trails the metal rather than the camera.
   *
   * @param haste flight speed as a multiple of the reference pace, so a fast
   *   round smears longer than a lumbering one
   */
  aimAlong(direction: THREE.Vector3, travelled: number, haste: number): void {
    this.round.quaternion.setFromUnitVectors(FORWARD, direction);
    this.round.rotateOnAxis(this.axis, travelled * this.spin);
    this.smear.quaternion.setFromUnitVectors(FORWARD, direction);
    this.smear.scale.z = this.gauge * this.spec.streak.stretch * NOSE_BLUR * haste;
    // The wake of dragged air hangs behind the smear.
    this.wake?.position.copy(direction).multiplyScalar(-this.wake.scale.x * 0.42);
  }

  dispose(): void {
    this.light?.release();
    (this.glint.material as THREE.Material).dispose();
    (this.smear.material as THREE.Material).dispose();
    if (this.glow) (this.glow.material as THREE.Material).dispose();
    if (this.wake) (this.wake.material as THREE.Material).dispose();
    // Only a heated round owns its material; cold lead shares the cached one.
    for (const material of this.heated) material.dispose();
    this.group.removeFromParent();
    this.group.clear();
  }
}

export interface ShotOptions {
  look: GunLook;
  /** Which round is in the barrel. */
  ammo: AmmoKind;
  /** Diameter of the bore in world units. */
  size: number;
  /**
   * Seconds of flight. Long enough that the eye can pick the round up and
   * follow it — a true muzzle velocity would put it in the body inside two
   * frames, which is exactly why nobody could see the shot.
   */
  flight: number;
  /** A slot borrowed from the scene's light pool, or null. */
  light?: SpellLight | null;
  /** Called with the round's position every frame, for the smoke it leaves. */
  onTrail?: (at: THREE.Vector3, t: number) => void;
  /**
   * Spine samples in the streak the round draws behind it. The one knob
   * graphics quality turns on the trail; 0 leaves the round untrailed.
   */
  trailDetail?: number;
}

/**
 * How much of a round's authored nose smear is kept now that the flight path is
 * drawn as geometry (`tracer.ts`).
 *
 * The cone's job has narrowed: it is the blur *on the metal*, a couple of
 * calibres of stretched highlight at the nose. At its old full length the two
 * layers doubled up and the pair read as one fat smudge with no direction in it.
 */
const NOSE_BLUR = 0.5;

/**
 * Hands the streak over to a short fade of its own once the round has landed.
 *
 * A streak that is deleted on the frame of impact snaps off, which reads as a
 * glitch. Left to dissolve over a beat and a half of frames it reads as the
 * afterimage of something that was moving very fast, and it dies under the
 * debris and the flash of the hit.
 */
function releaseStreak(
  streak: TracerStreak,
  tweens: { to: (spec: { duration: number; easing: (t: number) => number; onUpdate: (t: number) => void }) => Promise<void> },
  strength: number,
): void {
  void (async () => {
    try {
      await tweens.to({
        duration: 0.16,
        easing: (t: number) => t,
        onUpdate: (t: number) => streak.fade(strength * Math.pow(1 - t, 1.7)),
      });
    } finally {
      streak.dispose();
    }
  })();
}

/**
 * Sends a round from a muzzle to a body: dead straight, no arc, no easing. Shot
 * travels flat over a chessboard's worth of distance, and the flatness is what
 * tells the eye this is a gun rather than a lobbed spell.
 *
 * The one thing that is *not* straight is a smoothbore ball. A ball rattling
 * down an unrifled barrel leaves it turning, and a turning sphere curves: it
 * bellies off the line of sight and comes back onto the body. That is why a
 * musket could not be trusted at a hundred paces, and the rifled Minié round is
 * the only thing in the army that flies a true line.
 */
export async function flyShot(
  scene: THREE.Object3D,
  tweens: { to: (spec: { duration: number; easing: (t: number) => number; onUpdate: (t: number) => void }) => Promise<void> },
  from: THREE.Vector3,
  to: THREE.Vector3,
  options: ShotOptions,
): Promise<void> {
  const spec = AMMUNITION[options.ammo];
  const shot = new Shot(options.ammo, options.look, options.size, options.light ?? null);
  const heading = to.clone().sub(from);
  const distance = Math.max(1e-4, heading.length());
  heading.divideScalar(distance);
  // The plane the ball bellies out into: across the line of fire, and tilted a
  // little so the drift is never a flat sideways slide.
  const drift = new THREE.Vector3(0, 1, 0).cross(heading).normalize();
  drift.addScaledVector(new THREE.Vector3(0, 1, 0), (Math.random() - 0.5) * 0.7).normalize();
  const wander = spec.wander * options.size * (Math.random() > 0.5 ? 1 : -1) * (0.6 + Math.random() * 0.8);
  // Tiles per second against a reference pace, so the smear lengthens on a fast
  // barrel and shortens on a lumbering one instead of being a fixed streak.
  const haste = THREE.MathUtils.clamp(distance / Math.max(0.01, options.flight) / 12, 0.55, 1.9);
  shot.place(from, 1);
  shot.aimAlong(heading, 0, haste);
  scene.add(shot.group);
  // The streak lives in world space, not on the round: it is the path, so it
  // must stay where the round has been rather than travel with it.
  const rings = options.trailDetail ?? 20;
  const streak = rings > 0 ? new TracerStreak(spec.trail, shot.gauge, rings) : null;
  if (streak) scene.add(streak.object);
  const at = new THREE.Vector3();
  try {
    await tweens.to({
      duration: options.flight,
      easing: (t: number) => t,
      onUpdate: (t: number) => {
        at.lerpVectors(from, to, t);
        // Peaks mid-flight and closes again: the ball still finds the body, it
        // just does not get there in a straight line.
        if (wander !== 0) at.addScaledVector(drift, wander * Math.sin(Math.PI * t));
        shot.place(at, 1 - t);
        // The smear is short as the round clears the bore, opens to full length
        // once it is up to speed: a shot has no blur before it has moved.
        shot.aimAlong(heading, t * distance, haste * Math.min(1, 0.35 + t * 6));
        streak?.extend(at);
        options.onTrail?.(at, t);
      },
    });
  } finally {
    shot.dispose();
    if (streak) releaseStreak(streak, tweens, spec.trail.strength);
  }
}

export interface PowderCloudOptions {
  look: GunLook;
  /** Width of the cloud in world units. */
  size: number;
  /** Direction the smoke is pushed, i.e. the line of fire. */
  direction: THREE.Vector3;
  /** How many puffs make up the bank. */
  count: number;
  /** Seconds from ignition to the last thread of it dissolving. */
  life?: number;
  /**
   * Overrides the faction tint. A rifled bore burns a small, tight-patched
   * charge almost completely, so its bank is a pale ash grey rather than the
   * soot of a smoothbore volley.
   */
  tint?: number;
  /** How thick the bank reads. 1 = a musket; below that you see through it. */
  density?: number;
  /**
   * Fine-grain powder: swaps in the paler, threadier map, vents in a shorter
   * sharper beat, lifts faster and tears itself apart sooner.
   */
  fine?: boolean;
  /**
   * The hall's own air, in world units per second. What finally carries the
   * bank off the square it was fired from — without it the smoke would sit
   * exactly where it was made and simply dim, which never reads as air.
   */
  draft?: THREE.Vector3;
  /**
   * World height the bank rolls out along instead of sinking through, i.e. the
   * top of the board. Smoke that reaches it stops falling and spreads.
   */
  floor?: number;
}

/**
 * One lobe of the bank, with the whole of its own history in it: when it left
 * the bore, how hard, how long it has to live, and how it churns while it does.
 *
 * Every lobe is integrated from its *absolute age* rather than stepped frame by
 * frame. That is deliberate: the bank is driven off a tween's normalised clock,
 * so a closed-form path is the only way the smoke stays identical whatever the
 * frame rate, and it lets each lobe be born and die on its own schedule inside
 * one shared timeline.
 */
interface PowderPuff {
  sprite: THREE.Sprite;
  material: THREE.SpriteMaterial;
  /** Offset from the bore the lobe is made at. */
  seat: THREE.Vector3;
  /** Speed it leaves the bore at, before drag takes it. */
  jet: THREE.Vector3;
  /** How fast that ejection speed is eaten by the air. */
  drag: number;
  /** Seconds after ignition this lobe appears. */
  born: number;
  /** Seconds it lasts once it has. */
  span: number;
  /** Width at birth, and how many times that it swells to. */
  seed: number;
  swell: number;
  /** Upward pull of hot, light gas. */
  lift: number;
  /** Amplitude and rate of the curl it turns over as it goes. */
  churn: THREE.Vector3;
  rate: number;
  phase: number;
  spin: number;
  peak: number;
}

/**
 * The bank of smoke a black-powder charge leaves hanging in front of the gun.
 *
 * The old version was a handful of sprites that all appeared on the same frame,
 * slid outward in a straight line at constant speed and dimmed together — one
 * pop, then nothing. What a charge actually does has three distinct phases, and
 * all three are modelled here:
 *
 * 1. **The vent.** Gas leaves the bore for a tenth of a second or so, not all at
 *    once, so the lobes are *born in sequence* across `vent` and the earliest
 *    ones get the hardest shove. This is what makes the bank grow out of the
 *    barrel instead of appearing around it.
 * 2. **The stall.** That ejection speed is eaten by the air almost immediately:
 *    each lobe travels `jet/drag · (1 − e^⁻ᵈʳᵃᵍ·ᵃᵍᵉ)`, i.e. it lunges forward
 *    perhaps a square and stops. From there it is only buoyancy, the hall's
 *    draft and its own churn — a cloud, not a projectile.
 * 3. **The dissolve.** Mass is conserved while volume is not: as a lobe swells
 *    it must thin, so opacity carries `(seed/width)^1.35` on top of its fade.
 *    Smoke therefore gets faint *because it is spreading*, which is why it goes
 *    from solid white to a haze you can read the board through.
 */
export async function spawnPowderCloud(
  scene: THREE.Object3D,
  tweens: { to: (spec: { duration: number; easing: (t: number) => number; onUpdate: (t: number) => void }) => Promise<void> },
  at: THREE.Vector3,
  options: PowderCloudOptions,
): Promise<void> {
  const life = options.life ?? 1.5;
  const fine = options.fine === true;
  const tint = options.tint ?? options.look.smoke;
  const density = options.density ?? 1;
  const size = options.size;
  const group = new THREE.Group();
  group.name = "powder_cloud";
  group.position.copy(at);
  scene.add(group);

  // How long gas keeps coming out of the bore. A tight-patched rifle charge is
  // spent in half the time a musket's loose one takes to finish venting.
  const vent = Math.min(life * 0.4, fine ? 0.1 : 0.17);
  // Everything below the board would be under the stone, so a bank that sinks
  // that far flattens against it instead.
  const floor = options.floor != null ? options.floor - at.y : null;
  const side = new THREE.Vector3(0, 1, 0).cross(options.direction).normalize();
  const up = new THREE.Vector3(0, 1, 0);

  const puffs: PowderPuff[] = [];
  for (let i = 0; i < options.count; i += 1) {
    // Where this lobe sits in the vent: 0 is the first gas out of the bore.
    const order = options.count <= 1 ? 0 : i / (options.count - 1);
    const material = new THREE.SpriteMaterial({
      map: fine ? sharedFinePuffMap() : sharedPuffMap(),
      color: tint,
      transparent: true,
      depthWrite: false,
      opacity: 0,
      rotation: Math.random() * Math.PI * 2,
    });
    const sprite = new THREE.Sprite(material);
    sprite.renderOrder = 5;
    sprite.frustumCulled = false;
    sprite.visible = false;
    group.add(sprite);

    // Later gas is cooler and slower: it barely clears the muzzle and is left
    // curling around it, which is the part that hangs on the barrel.
    const push = 1 - order * 0.62;
    const seed = size * (fine ? 0.16 + Math.random() * 0.2 : 0.22 + Math.random() * 0.28);
    puffs.push({
      sprite,
      material,
      seat: options.direction
        .clone()
        .multiplyScalar(size * order * (fine ? 0.16 : 0.22))
        .addScaledVector(side, (Math.random() - 0.5) * size * 0.22)
        .addScaledVector(up, (Math.random() - 0.5) * size * 0.2),
      jet: options.direction
        .clone()
        .multiplyScalar(size * (fine ? 5.4 : 4.1) * push * (0.7 + Math.random() * 0.6))
        // Gas escaping past the ball never leaves straight: it fans off the bore.
        .addScaledVector(side, (Math.random() - 0.5) * size * (fine ? 1.5 : 2.6))
        .addScaledVector(up, (Math.random() - 0.35) * size * (fine ? 1.9 : 1.5)),
      // A thin jet is stopped by the air faster than a fat sooty one.
      drag: (fine ? 5.4 : 4.2) * (0.8 + Math.random() * 0.5),
      born: vent * order * (0.55 + Math.random() * 0.9),
      // Small lobes are torn apart first; a few always outlast the rest, so the
      // bank never switches off on one frame.
      span: life * (0.5 + Math.random() * 0.5) * (Math.random() < 0.18 ? 1.25 : 1),
      seed,
      swell: fine ? 3.4 + Math.random() * 1.8 : 2.6 + Math.random() * 1.5,
      lift: size * (fine ? 0.34 + Math.random() * 0.3 : 0.15 + Math.random() * 0.22),
      churn: new THREE.Vector3(
        (Math.random() - 0.5) * size * 0.5,
        (Math.random() - 0.5) * size * 0.3,
        (Math.random() - 0.5) * size * 0.5,
      ),
      rate: 1.1 + Math.random() * 1.6,
      phase: Math.random() * Math.PI * 2,
      spin: (Math.random() - 0.5) * (fine ? 1.5 : 1),
      peak: (fine ? 0.34 : 0.56) * density * (0.7 + Math.random() * 0.6),
    });
  }

  const place = new THREE.Vector3();
  try {
    await tweens.to({
      duration: life,
      easing: (t: number) => t,
      onUpdate: (t: number) => {
        const now = t * life;
        for (const puff of puffs) {
          const age = now - puff.born;
          if (age <= 0 || age >= puff.span) {
            puff.sprite.visible = false;
            continue;
          }
          puff.sprite.visible = true;
          const u = age / puff.span;

          // Lunge, then stall: the ejection speed is gone in a few frames and
          // what is left is a cloud sitting in the air being moved around.
          const carried = (1 - Math.exp(-puff.drag * age)) / puff.drag;
          place.copy(puff.seat).addScaledVector(puff.jet, carried);
          // Buoyancy builds instead of being an initial kick: powder smoke sags
          // off the barrel first and only then climbs.
          place.y += puff.lift * age * age * 0.75;
          if (options.draft) place.addScaledVector(options.draft, age);
          // Turbulent curl — the lobe rolls over itself rather than sliding.
          const swirl = Math.min(1, age / 0.4);
          place.addScaledVector(puff.churn, Math.sin(age * puff.rate + puff.phase) * swirl);
          if (floor != null && place.y < floor) place.y = floor;
          puff.sprite.position.copy(place);

          // Entrainment: fast swell while the gas is still hot, then easing off.
          const width = puff.seed * (1 + (puff.swell - 1) * Math.pow(u, 0.55));
          puff.sprite.scale.setScalar(width);
          // Thinning as it spreads (mass over volume) with a soft tail on top, so
          // the last of it dissolves into the hall rather than being switched off.
          const bloom = Math.min(1, age / (fine ? 0.05 : 0.07));
          const thinning = Math.pow(puff.seed / width, 1.35);
          puff.material.opacity = puff.peak * bloom * thinning * Math.pow(1 - u, 0.85);
          // Angular drag: the curl slows as the lobe loses its energy.
          puff.material.rotation += puff.spin * 0.016 * (1 - u * 0.7);
        }
      },
    });
  } finally {
    for (const puff of puffs) puff.material.dispose();
    group.removeFromParent();
    group.clear();
  }
}
