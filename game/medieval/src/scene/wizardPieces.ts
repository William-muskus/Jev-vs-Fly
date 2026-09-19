/**
 * Enchanted stone chess — the living set.
 *
 * These are original gothic-Staunton silhouettes (not a third-party scan).
 * Drop converted GLBs in `public/models/wizard/{k,q,b,n,r,p}.glb` to replace
 * them; see `python -m game.wizard_assets`.
 */

import * as THREE from "three";

import type { PieceKind } from "../core/types";

function stone(): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color: 0xe8e0cf,
    roughness: 0.58,
    metalness: 0.14,
  });
}

function mesh(geometry: THREE.BufferGeometry, material: THREE.Material, y = 0): THREE.Mesh {
  const m = new THREE.Mesh(geometry, material);
  m.castShadow = true;
  m.receiveShadow = true;
  m.position.y = y;
  m.frustumCulled = false;
  return m;
}

function pedestal(mat: THREE.MeshStandardMaterial, radius = 0.42): THREE.Group {
  const g = new THREE.Group();
  g.add(mesh(new THREE.CylinderGeometry(radius, radius * 1.08, 0.08, 28), mat, 0.04));
  g.add(mesh(new THREE.CylinderGeometry(radius * 0.86, radius * 0.98, 0.07, 28), mat, 0.1));
  g.add(mesh(new THREE.TorusGeometry(radius * 0.72, 0.035, 8, 24), mat, 0.15));
  return g;
}

function stem(mat: THREE.MeshStandardMaterial, height: number, rBottom: number, rTop: number, y0: number): THREE.Mesh {
  return mesh(new THREE.CylinderGeometry(rTop, rBottom, height, 20), mat, y0 + height / 2);
}

function king(mat: THREE.MeshStandardMaterial): THREE.Group {
  const g = new THREE.Group();
  g.add(pedestal(mat, 0.4));
  g.add(stem(mat, 0.42, 0.2, 0.13, 0.16));
  g.add(mesh(new THREE.SphereGeometry(0.2, 20, 16), mat, 0.72));
  g.add(mesh(new THREE.CylinderGeometry(0.16, 0.18, 0.12, 12), mat, 0.92));
  const coronet = mesh(new THREE.CylinderGeometry(0.2, 0.14, 0.14, 10, 1, true), mat, 1.04);
  g.add(coronet);
  for (let i = 0; i < 5; i += 1) {
    const a = (i / 5) * Math.PI * 2;
    const spike = mesh(new THREE.ConeGeometry(0.035, 0.12, 6), mat, 1.16);
    spike.position.x = Math.cos(a) * 0.14;
    spike.position.z = Math.sin(a) * 0.14;
    g.add(spike);
  }
  const crossV = mesh(new THREE.BoxGeometry(0.05, 0.22, 0.05), mat, 1.28);
  const crossH = mesh(new THREE.BoxGeometry(0.16, 0.05, 0.05), mat, 1.32);
  g.add(crossV);
  g.add(crossH);
  return g;
}

function queen(mat: THREE.MeshStandardMaterial): THREE.Group {
  const g = new THREE.Group();
  g.add(pedestal(mat, 0.4));
  g.add(stem(mat, 0.4, 0.19, 0.12, 0.16));
  g.add(mesh(new THREE.SphereGeometry(0.19, 20, 16), mat, 0.7));
  g.add(mesh(new THREE.CylinderGeometry(0.15, 0.17, 0.1, 14), mat, 0.88));
  for (let i = 0; i < 8; i += 1) {
    const a = (i / 8) * Math.PI * 2;
    const pearl = mesh(new THREE.SphereGeometry(0.035, 10, 8), mat, 1.02);
    pearl.position.x = Math.cos(a) * 0.15;
    pearl.position.z = Math.sin(a) * 0.15;
    g.add(pearl);
  }
  g.add(mesh(new THREE.SphereGeometry(0.055, 12, 10), mat, 1.1));
  return g;
}

function bishop(mat: THREE.MeshStandardMaterial): THREE.Group {
  const g = new THREE.Group();
  g.add(pedestal(mat, 0.36));
  g.add(stem(mat, 0.38, 0.16, 0.1, 0.16));
  g.add(mesh(new THREE.SphereGeometry(0.15, 18, 14), mat, 0.66));
  g.add(mesh(new THREE.ConeGeometry(0.16, 0.38, 16), mat, 0.94));
  g.add(mesh(new THREE.SphereGeometry(0.04, 10, 8), mat, 1.16));
  return g;
}

function knight(mat: THREE.MeshStandardMaterial): THREE.Group {
  const g = new THREE.Group();
  g.add(pedestal(mat, 0.38));
  const chest = mesh(new THREE.BoxGeometry(0.28, 0.36, 0.42), mat, 0.4);
  chest.position.z = 0.04;
  g.add(chest);
  const neck = mesh(new THREE.CylinderGeometry(0.09, 0.12, 0.32, 12), mat, 0.68);
  neck.rotation.x = -0.7;
  neck.position.z = 0.12;
  g.add(neck);
  const head = mesh(new THREE.BoxGeometry(0.18, 0.16, 0.32), mat, 0.86);
  head.position.z = 0.22;
  head.rotation.x = -0.25;
  g.add(head);
  const muzzle = mesh(new THREE.BoxGeometry(0.12, 0.1, 0.18), mat, 0.8);
  muzzle.position.z = 0.4;
  g.add(muzzle);
  for (const side of [-1, 1]) {
    const ear = mesh(new THREE.ConeGeometry(0.04, 0.14, 6), mat, 0.98);
    ear.position.set(side * 0.07, 0.98, 0.12);
    ear.rotation.x = -0.4;
    g.add(ear);
  }
  const mane = mesh(new THREE.BoxGeometry(0.06, 0.28, 0.22), mat, 0.82);
  mane.position.z = 0.04;
  g.add(mane);
  return g;
}

function rook(mat: THREE.MeshStandardMaterial): THREE.Group {
  const g = new THREE.Group();
  g.add(pedestal(mat, 0.4));
  g.add(mesh(new THREE.CylinderGeometry(0.22, 0.26, 0.5, 16), mat, 0.42));
  g.add(mesh(new THREE.CylinderGeometry(0.26, 0.24, 0.1, 16), mat, 0.72));
  const merlonCount = 6;
  for (let i = 0; i < merlonCount; i += 1) {
    const a = (i / merlonCount) * Math.PI * 2;
    const merlon = mesh(new THREE.BoxGeometry(0.1, 0.14, 0.1), mat, 0.84);
    merlon.position.x = Math.cos(a) * 0.2;
    merlon.position.z = Math.sin(a) * 0.2;
    g.add(merlon);
  }
  return g;
}

function pawn(mat: THREE.MeshStandardMaterial): THREE.Group {
  const g = new THREE.Group();
  g.add(pedestal(mat, 0.32));
  g.add(stem(mat, 0.22, 0.14, 0.1, 0.16));
  g.add(mesh(new THREE.TorusGeometry(0.11, 0.03, 8, 16), mat, 0.4));
  g.add(mesh(new THREE.SphereGeometry(0.15, 18, 14), mat, 0.56));
  return g;
}

const BUILDERS: Record<PieceKind, (mat: THREE.MeshStandardMaterial) => THREE.Group> = {
  k: king,
  q: queen,
  b: bishop,
  n: knight,
  r: rook,
  p: pawn,
};

/** One unrigged, unarmed stone piece. Faction tint is applied by the factory. */
export function buildWizardChessPiece(kind: PieceKind): THREE.Group {
  const mat = stone();
  const piece = BUILDERS[kind](mat);
  piece.name = `wizard_${kind}`;
  return piece;
}
