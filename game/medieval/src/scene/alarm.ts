import * as THREE from "three";

import { radialTexture } from "./textures";

/** Alarm red — the same hex the threatened king's own emissive burns at. */
const ALARM_COLOUR = 0xff2a1a;

/**
 * The check alarm: a red light that stands over whichever king is in danger.
 *
 * It has two states because a check has two moments. The **flare** is the
 * instant the threat is declared — a hard surge that decays in under a second.
 * The **watch** is everything after: while the king is still in check the light
 * keeps breathing at a much lower level, so a player who looks away and back
 * still sees which crown is under the sword.
 *
 * The lamp is created once and lives in the scene forever at zero intensity.
 * three.js recompiles every material in the hall whenever a scene's light count
 * changes, so adding a light on check and removing it after would stall the
 * frame on the exact beat that has to feel sharp.
 */
export class CheckAlarm {
  readonly group = new THREE.Group();

  private lamp: THREE.PointLight;
  /** Soft ground halo, so the alarm reads even where the lamp has nothing to hit. */
  private halo: THREE.Sprite;
  private haloMap: THREE.Texture;
  private threatened = false;
  /** Eased presence, 0 when no king is in check. */
  private watch = 0;
  /** Decaying surge from the moment the check was declared. */
  private flare = 0;
  private phase = 0;

  constructor() {
    this.group.name = "check-alarm";
    this.group.visible = false;

    this.lamp = new THREE.PointLight(ALARM_COLOUR, 0, 5.5, 2);
    this.lamp.castShadow = false;
    this.group.add(this.lamp);

    this.haloMap = radialTexture("rgba(255,58,36,0.85)", "rgba(255,26,12,0)");
    const material = new THREE.SpriteMaterial({
      map: this.haloMap,
      color: ALARM_COLOUR,
      transparent: true,
      depthWrite: false,
      opacity: 0,
      blending: THREE.AdditiveBlending,
    });
    this.halo = new THREE.Sprite(material);
    this.halo.renderOrder = 3;
    this.group.add(this.halo);
  }

  /**
   * Points the alarm at the king under threat, or stands it down.
   * Safe to call every frame — moving the light does not recompile anything.
   */
  setThreat(position: THREE.Vector3 | null): void {
    this.threatened = position !== null;
    if (!position) return;
    this.lamp.position.set(position.x, position.y + 1.15, position.z);
    this.halo.position.set(position.x, position.y + 0.95, position.z);
  }

  /** The moment the check is declared: one hard pulse over the watch level. */
  strike(): void {
    this.flare = 1;
    // Restart the breath on the flare so the first swell after it is a full one
    // rather than whatever phase the sine happened to be passing through.
    this.phase = 0;
  }

  update(delta: number): void {
    const target = this.threatened ? 1 : 0;
    // Comes up fast, goes out slowly: a threat announces itself, then lifts.
    const rate = target > this.watch ? 6 : 2.6;
    this.watch += Math.sign(target - this.watch) * Math.min(Math.abs(target - this.watch), delta * rate);
    this.flare = Math.max(0, this.flare - delta * 1.8);

    if (this.watch <= 0.001 && this.flare <= 0) {
      if (this.group.visible) {
        this.group.visible = false;
        this.lamp.intensity = 0;
        (this.halo.material as THREE.SpriteMaterial).opacity = 0;
      }
      return;
    }
    this.group.visible = true;

    this.phase += delta;
    // Slow heartbeat, weighted toward the dark half so it pulses rather than glows.
    const breath = Math.pow(Math.sin(this.phase * 2.4) * 0.5 + 0.5, 1.6);
    const surge = this.flare * this.flare;
    // The watch level is what a player sees for most of a check, so it is held
    // low: a red edge on the crown, not a lamp washing the surrounding rank.
    const level = this.watch * (0.2 + breath * 0.26) + surge * 0.7;

    this.lamp.intensity = level * 2.1;
    const material = this.halo.material as THREE.SpriteMaterial;
    material.opacity = Math.min(0.3, level * 0.24);
    this.halo.scale.setScalar(1.35 + breath * 0.2 + surge * 0.45);
  }

  dispose(): void {
    (this.halo.material as THREE.Material).dispose();
    this.haloMap.dispose();
    this.group.clear();
  }
}
