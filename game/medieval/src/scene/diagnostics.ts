import * as THREE from "three";

/**
 * What the player's driver actually is, and whether it can do the two things
 * this scene depends on: filtered half-float render targets (the reflection
 * probe is built in one) and a working physically-based lighting path.
 *
 * Linux is where this matters. Mesa's software rasterisers (llvmpipe, softpipe)
 * and a few older open drivers either refuse the float targets the PMREM
 * generator needs or return NaN when the resulting cube map is sampled — and a
 * NaN in the environment term blacks out every metal/roughness surface in the
 * hall while emissive sprites keep drawing, which reads as "the game is black".
 */
export interface GpuReport {
  vendor: string;
  renderer: string;
  webgl2: boolean;
  /** llvmpipe / softpipe / SwiftShader — no GPU behind the context. */
  software: boolean;
  /** Linear filtering of float textures. Missing on several Mesa builds. */
  floatLinear: boolean;
  /** Rendering into a half-float buffer, which the reflection probe needs. */
  halfFloatBuffer: boolean;
}

const SOFTWARE_PATTERN = /(llvmpipe|softpipe|swiftshader|virgl|software rasterizer|mesa offscreen)/;

export function probeGpu(renderer: THREE.WebGLRenderer): GpuReport {
  const gl = renderer.getContext();
  let vendor = "unknown";
  let name = "unknown";
  try {
    const info = gl.getExtension("WEBGL_debug_renderer_info");
    if (info) {
      vendor = String(gl.getParameter(info.UNMASKED_VENDOR_WEBGL) ?? "unknown");
      name = String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL) ?? "unknown");
    }
  } catch {
    // Some privacy modes block the extension entirely — keep the defaults.
  }

  return {
    vendor,
    renderer: name,
    webgl2: renderer.capabilities.isWebGL2,
    software: SOFTWARE_PATTERN.test(name.toLowerCase()),
    floatLinear: renderer.extensions.has("OES_texture_float_linear"),
    halfFloatBuffer: renderer.capabilities.isWebGL2 || renderer.extensions.has("EXT_color_buffer_half_float"),
  };
}

/** One line for the settings panel and the console, e.g. `llvmpipe · WebGL2 · software`. */
export function describeGpu(report: GpuReport): string {
  const tags = [report.webgl2 ? "WebGL2" : "WebGL1"];
  if (report.software) tags.push("software");
  if (!report.floatLinear) tags.push("no float filtering");
  return `${report.renderer} · ${tags.join(" · ")}`;
}

/**
 * Renders one 8×8 frame of a white sphere lit **only** by the reflection probe
 * and reads it back. A healthy driver returns a lit sphere; a driver that
 * cannot sample the probe returns black (or NaN, which also rasterises black),
 * and that is the exact failure that makes the whole hall look unlit.
 *
 * Cheap enough to run once at boot: one 8×8 draw plus one readback.
 */
export function reflectionProbeWorks(renderer: THREE.WebGLRenderer, environment: THREE.Texture): boolean {
  const size = 8;
  const target = new THREE.WebGLRenderTarget(size, size);
  const scene = new THREE.Scene();
  scene.environment = environment;
  scene.environmentIntensity = 1;

  const geometry = new THREE.SphereGeometry(1, 16, 12);
  const material = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.35, metalness: 0 });
  scene.add(new THREE.Mesh(geometry, material));

  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 10);
  camera.position.set(0, 0, 2.2);
  camera.lookAt(0, 0, 0);

  const previousTarget = renderer.getRenderTarget();
  const previousTone = renderer.toneMapping;
  const previousExposure = renderer.toneMappingExposure;
  const pixels = new Uint8Array(size * size * 4);
  let lit = false;

  try {
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.toneMappingExposure = 1;
    renderer.setRenderTarget(target);
    renderer.clear();
    renderer.render(scene, camera);
    renderer.readRenderTargetPixels(target, 0, 0, size, size, pixels);
    for (let i = 0; i < pixels.length; i += 4) {
      if (pixels[i] > 8 || pixels[i + 1] > 8 || pixels[i + 2] > 8) {
        lit = true;
        break;
      }
    }
  } catch (error) {
    console.warn("[scene] reflection probe self-test failed", error);
    lit = false;
  } finally {
    renderer.setRenderTarget(previousTarget);
    renderer.toneMapping = previousTone;
    renderer.toneMappingExposure = previousExposure;
    geometry.dispose();
    material.dispose();
    target.dispose();
  }

  return lit;
}
