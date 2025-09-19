import type { Pose } from '@tensorflow-models/pose-detection';

/* MoveNet key-point indices */
const R_SH = 6, L_SH = 5;
const R_WR = 10, L_WR = 9;
const R_HIP = 12, L_HIP = 11;

/* helpers */
const X = (p: Pose, i: number) => p.keypoints[i]?.x ?? 0;
const Y = (p: Pose, i: number) => p.keypoints[i]?.y ?? 0;

/* ─── basic tests ───────────────────────────────────────── */

const raiseRight = (p: Pose) => Y(p, R_WR) < Y(p, R_SH);
const raiseLeft  = (p: Pose) => Y(p, L_WR) < Y(p, L_SH);

const armsCross = (p: Pose) =>
  Math.hypot(X(p, R_WR) - X(p, L_SH), Y(p, R_WR) - Y(p, L_SH)) < 80 &&
  Math.hypot(X(p, L_WR) - X(p, R_SH), Y(p, L_WR) - Y(p, R_SH)) < 80;

const spineTiltDeg = (p: Pose) => {
  const midShX = (X(p, R_SH) + X(p, L_SH)) / 2;
  const midShY = (Y(p, R_SH) + Y(p, L_SH)) / 2;
  const midHpX = (X(p, R_HIP) + X(p, L_HIP)) / 2;
  const midHpY = (Y(p, R_HIP) + Y(p, L_HIP)) / 2;
  return ((Math.atan2(midShY - midHpY, midShX - midHpX) * 180) / Math.PI);
};

const leanLeft  = (p: Pose) => spineTiltDeg(p) < -10;
const leanRight = (p: Pose) => spineTiltDeg(p) >  10;

/* ─── high-level detector (returns at most ONE label) ───────── */

export type GestureLabel =
  | 'RAISE_RIGHT'
  | 'RAISE_LEFT'
  | 'ARMS_CROSS'
  | 'LEAN_LEFT'
  | 'LEAN_RIGHT';

export function detectGesture(p: Pose): GestureLabel | null {
  if (raiseRight(p)) return 'RAISE_RIGHT';
  if (raiseLeft(p))  return 'RAISE_LEFT';
  if (armsCross(p))  return 'ARMS_CROSS';
  if (leanLeft(p))   return 'LEAN_LEFT';
  if (leanRight(p))  return 'LEAN_RIGHT';
  return null;
}
