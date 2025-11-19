// src/ai/layersModel.ts
import * as tf from '@tensorflow/tfjs';
import type { Keypoint } from '@tensorflow-models/pose-detection';

/** Path to TFJS Layers model */
const DEFAULT_URL = '/models/gesture/model.json';

/** Order must match training exactly */
export const LABELS = [
  'idle',
  'raise_right',
  'raise_left',
  'cross_arms',
  'both_arms_up',
];

export type GestureLabel = (typeof LABELS)[number];

export interface GesturePrediction {
  classIndex: number;
  label: GestureLabel | `cls_${number}`;
  confidence: number;
  probs: number[];
  nClasses: number;
}

/** Load the TFJS Layers model (Bi-GRU classifier) */
export async function loadGestureLayers(url = DEFAULT_URL) {
  // Caller already sets backend + tf.ready() (LiveQuickTest, GestureStory)
  const model = await tf.loadLayersModel(url);
  console.log('Layers model loaded:', {
    in: model.inputs[0].shape,
    out: model.outputs[0].shape,
  });
  return model;
}

/** Per-frame normalization matching training pipeline:
 * torso-centered, shoulder-width scaled, 17 keypoints × (x,y,score) → [17,3]
 *
 * MoveNet COCO-17 layout:
 *  11 = left shoulder, 6 = right shoulder, 11 = left hip, 12 = right hip
 */
export function normalizeFrame(kps: Keypoint[]): number[] | null {
  if (!kps || kps.length < 17) return null;

  // MoveNet COCO-17 indices
  const Ls = kps[5];  // left shoulder
  const Rs = kps[6];  // right shoulder
  const Lh = kps[11]; // left hip
  const Rh = kps[12]; // right hip

  if (!Ls || !Rs || !Lh || !Rh) return null;

  const cx = (Ls.x + Rs.x + Lh.x + Rh.x) / 4;
  const cy = (Ls.y + Rs.y + Lh.y + Rh.y) / 4;
  const shoulderW = Math.hypot(Ls.x - Rs.x, Ls.y - Rs.y) || 1;

  const out: number[] = [];
  for (const kp of kps.slice(0, 17)) {
    const nx = (kp.x - cx) / shoulderW;
    const ny = (kp.y - cy) / shoulderW;
    const sc = kp.score ?? 0;
    out.push(nx, ny, sc);
  }

  return out;
}


/** Simple sequence buffer to collect T frames and emit [1,T,17,3] tensor */
class SeqBuf {
  private frames: number[][] = [];
  private readonly T: number;

  constructor(T = 60) {
    this.T = T;
  }

  push(f: number[] | null) {
    if (!f) return;
    this.frames.push(f);
    if (this.frames.length > this.T) this.frames.shift();
  }

  ready() {
    return this.frames.length === this.T;
  }

  /** Returns a tensor shaped [1, T, 17, 3] as used in training. */
  toTensor4D() {
    const flat = this.frames.flat(); // length T*51
    const T = this.frames.length;
    return tf.tensor4d(flat, [1, T, 17, 3]);
  }

  reset() {
    this.frames = [];
  }
}

// // --- Simple movement-based idle detector -------------------------------
// function detectIdle(kps: Keypoint[], lastKps: Keypoint[] | null) {
//   if (!lastKps) return false;

//   let total = 0;
//   for (let i = 0; i < 17; i++) {
//     const dx = kps[i].x - lastKps[i].x;
//     const dy = kps[i].y - lastKps[i].y;
//     total += Math.hypot(dx, dy);
//   }

//   // If the body moved less than this threshold, assume idle.
//   return total < 15;   // tweak 10–20 depending on sensitivity
// }

/** Make a per-frame step function for the Layers model. */
export function makeLayersStepper(model: tf.LayersModel, T = 60) {
  const buf = new SeqBuf(T);
  const nClasses =
    (model.outputs[0].shape?.[1] as number) ?? LABELS.length;

  // --- latency tracking (unchanged) ---
  let total = 0;
  let count = 0;

  // --- motion-based idle detection --------------------------
  let lastNorm: number[] | null = null;
  let stillStreak = 0;

  // tune these vars for system testing
  const MOTION_IDLE_THRESH = 0.02;   // avg movement per joint (normalized coords)
  const STILL_FRAMES_FOR_IDLE = 8;   // how many very-still frames before idle is forced

  return (kps: Keypoint[]) => {
    // normalize to [51] or null
    const norm = normalizeFrame(kps);
    if (!norm) return null;

    // --- compute motion magnitude between frames ---
    let motion = 0;
    if (lastNorm) {
      let acc = 0;
      const len = norm.length;
      const jointCount = len / 3;

      for (let i = 0; i < len; i += 3) {
        const dx = norm[i] - lastNorm[i];
        const dy = norm[i + 1] - lastNorm[i + 1];
        acc += Math.hypot(dx, dy);
      }
      motion = acc / jointCount;
    }
    lastNorm = norm;

    const isVeryStill = motion > 0 && motion < MOTION_IDLE_THRESH;
    if (isVeryStill) {
      stillStreak++;
    } else {
      stillStreak = 0;
    }

    // push into temporal buffer
    buf.push(norm);
    if (!buf.ready()) return null;

    const x = buf.toTensor4D();

    const t0 = performance.now();
    const y = model.predict(x) as tf.Tensor;
    const probsArr = Array.from(y.dataSync()); // <- define probs here
    const t1 = performance.now();

    x.dispose();
    y.dispose();

    const elapsed = t1 - t0;
    total += elapsed;
    count++;
    if (count % 50 === 0) {
      console.log(
        `Average inference latency: ${(total / count).toFixed(2)} ms`
      );
    }

    // argmax over probs
    let bestI = 0;
    let bestP = probsArr[0] ?? 0;
    for (let i = 1; i < probsArr.length; i++) {
      if (probsArr[i] > bestP) {
        bestP = probsArr[i];
        bestI = i;
      }
    }

    // --- motion-based idle override ---------------------------------
    // index 0 in LABELS is 'idle' in your setup
    const IDLE_INDEX = 0;

    const forceIdle = stillStreak >= STILL_FRAMES_FOR_IDLE;

    let finalLabel: any;
    let finalConf: number;

    if (forceIdle) {
      finalLabel = 'idle';
      finalConf = 1.0;
      bestI = IDLE_INDEX;
    } else {
      finalLabel = (LABELS[bestI] ?? `cls_${bestI}`) as any;
      finalConf = bestP;
    }

    return {
      classIndex: bestI,
      label: finalLabel,
      confidence: finalConf,
      probs: probsArr,
      nClasses,
    };
  };
}