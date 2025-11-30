// src/ai/layersModel.ts
import * as tf from '@tensorflow/tfjs';
import type { Keypoint } from '@tensorflow-models/pose-detection';

const DEFAULT_URL = '/models/gesture/model.json';
export const LABELS = ['idle', 'raise_right', 'raise_left', 'cross_arms', 'both_arms_up'];

// Small logger that only warns once per message
const warned = new Set<string>();
function warnOnce(msg: string) {
  if (!warned.has(msg)) {
    console.warn(msg);
    warned.add(msg);
  }
}

// BlazePose-33 → COCO-17 subset mapping. Indices refer to BlazePose joints.
const BLAZEPOSE33_TO_COCO17: number[] = [
  0,     // nose
  2, 3,  // left/right eye (approx)
  7, 8,  // left/right ear (approx)
  11, 12, // shoulders
  13, 14, // elbows
  15, 16, // wrists
  23, 24, // hips
  25, 26, // knees
  27, 28, // ankles
];

// Ensure we have a 17-keypoint array in COCO order. Returns null if not possible.
function toCoco17(kps: Keypoint[] | null | undefined): Keypoint[] | null {
  if (!kps || kps.length === 0) return null;

  // If we already have 17 keypoints, COCO-17 (MoveNet).
  if (kps.length === 17) {
    return kps;
  }

  // If detector returns BlazePose-33 or anything else, map down to COCO-17 subset.
  if (kps.length >= 33) {
    const out: Keypoint[] = [];
    for (const idx of BLAZEPOSE33_TO_COCO17) {
      const kp = kps[idx];
      if (!kp) {
        warnOnce('[gesture] Missing BlazePose keypoint while mapping to COCO-17');
        return null;
      }
      out.push(kp);
    }
    return out;
  }

  warnOnce(`[gesture] Unsupported keypoint layout length=${kps.length}`);
  return null;
}

// Model loading
export async function loadGestureLayers(url = DEFAULT_URL) {
  await tf.setBackend('webgl');
  await tf.ready();
  const model = await tf.loadLayersModel(url);
  console.log('Layers model loaded:', {
    in: model.inputs[0].shape,
    out: model.outputs[0].shape,
  });
  return model;
}

/**
 * Per-frame normalization that matches with model training:
 *  - Map to COCO-17
 *  - Center on torso (average of shoulders and hips)
 *  - Scale by shoulder width
 *  - Keep (nx, ny, score) for each of the 17 joints → flat [51]
 */
function normalizeFrame(raw: Keypoint[]): number[] | null {
  const kps = toCoco17(raw);
  if (!kps || kps.length !== 17) {
    warnOnce('[gesture] Skipping frame: keypoints not in COCO-17 layout yet');
    return null;
  }

  // Shoulders / hips in COCO-17
  const Ls = kps[5];
  const Rs = kps[6];
  const Lh = kps[11];
  const Rh = kps[12];

  if (!Ls || !Rs || !Lh || !Rh) {
    warnOnce('[gesture] Skipping frame: shoulders/hips missing');
    return null;
  }

  const sx1 = Ls.x ?? 0;
  const sy1 = Ls.y ?? 0;
  const sx2 = Rs.x ?? 0;
  const sy2 = Rs.y ?? 0;

  const hx1 = Lh.x ?? 0;
  const hy1 = Lh.y ?? 0;
  const hx2 = Rh.x ?? 0;
  const hy2 = Rh.y ?? 0;

  // Torso center: average of shoulders and hips
  const cx = (sx1 + sx2 + hx1 + hx2) / 4;
  const cy = (sy1 + sy2 + hy1 + hy2) / 4;

  // Shoulder distance for scale (fallback to hip distance if too small)
  let shoulderW = Math.hypot(sx1 - sx2, sy1 - sy2);
  const hipW = Math.hypot(hx1 - hx2, hy1 - hy2);
  if (!isFinite(shoulderW) || shoulderW < 1e-3) {
    shoulderW = hipW;
  }
  if (!isFinite(shoulderW) || shoulderW < 1e-3) {
    warnOnce('[gesture] Skipping frame: invalid shoulder/hip width');
    return null;
  }

  const out: number[] = [];
  for (const kp of kps) {
    const x = kp.x ?? 0;
    const y = kp.y ?? 0;
    const score = kp.score ?? 0;

    const nx = (x - cx) / shoulderW;
    const ny = (y - cy) / shoulderW;

    out.push(nx, ny, score);
  }

  return out; // length 51
}

/**
 * Simple forward-style temporal smoothing:
 *  - If a joint score is below the threshold, reuse the previous frame's (nx, ny)
 *    for that joint.
 *  - This approximates the forward/backward fill used in the Python preprocessing for our dataset,
 *    but in a streaming way.
 */
function smoothFrameWithPrev(
  curr: number[],
  prev: number[],
  scoreThreshold = 0.3,
): number[] {
  if (curr.length !== prev.length) return curr;

  const out = curr.slice();

  const jointCount = curr.length / 3;
  for (let j = 0; j < jointCount; j++) {
    const base = j * 3;
    const score = curr[base + 2];

    if (!Number.isFinite(score) || score < scoreThreshold) {
      // Reuse previous (nx, ny). Keep the current score as-is.
      out[base + 0] = prev[base + 0];
      out[base + 1] = prev[base + 1];
    }
  }

  return out;
}

/**
 * Pose level idle detector on the normalized frame.
 * Treat the pose as clearly idle when both wrists are comfortably
 * below the shoulders with reasonable confidence.
 */
function isClearlyIdle(normFrame: number[]): boolean {
  if (normFrame.length !== 17 * 3) return false;

  const lShoulderY = normFrame[5 * 3 + 1];
  const rShoulderY = normFrame[6 * 3 + 1];
  if (!Number.isFinite(lShoulderY) || !Number.isFinite(rShoulderY)) {
    return false;
  }
  const shoulderY = (lShoulderY + rShoulderY) / 2;

  const lWristY = normFrame[9 * 3 + 1];
  const rWristY = normFrame[10 * 3 + 1];
  const lWristScore = normFrame[9 * 3 + 2];
  const rWristScore = normFrame[10 * 3 + 2];

  const minScore = 0.5;
  if (lWristScore < minScore || rWristScore < minScore) {
    return false;
  }

  // Normalized coordinates keep y increasing downward.
  // Idle arms down: wrists well below the shoulders.
  const margin = 0.4;
  if (lWristY > shoulderY + margin && rWristY > shoulderY + margin) {
    return true;
  }

  return false;
}

// Sequence buffer
class SeqBuf {
  private frames: number[][] = [];
  private lastFrame: number[] | null = null;
  private T: number;

  constructor(T = 60) {
    this.T = T;
  }

  /**
   * Push a frame into the buffer.
   * If repeat > 1, the same smoothed frame is pushed several times.
   * This is used to flush idle frames through the window more quickly.
   */
  push(f: number[] | null, repeat = 1) {
    if (!f) return;

    let frame = f;
    if (this.lastFrame) {
      frame = smoothFrameWithPrev(frame, this.lastFrame);
    }

    for (let r = 0; r < repeat; r++) {
      this.frames.push(frame);
      if (this.frames.length > this.T) {
        this.frames.shift();
      }
    }

    this.lastFrame = frame;
  }

  ready() {
    return this.frames.length === this.T;
  }

  toTensor4D(): tf.Tensor4D | null {
    if (!this.ready()) return null;
    const flat = this.frames.flat(); // T * 51
    return tf.tensor4d(flat, [1, this.frames.length, 17, 3]);
  }
}

/** Make a per-frame step function for the Layers model. */
export function makeLayersStepper(model: tf.LayersModel, T = 60) {
  const buf = new SeqBuf(T);
  const nClasses = (model.outputs[0].shape?.[1] as number) ?? LABELS.length;

  // latency tracking
  let total = 0;
  let count = 0;

  return (kps: Keypoint[]) => {
    const norm = normalizeFrame(kps);
    const isIdlePose = norm ? isClearlyIdle(norm) : false;

    // When the pose is clearly idle, push multiple copies of this frame
    // to flush gesture frames from the window more quickly and return state to idle.
    const repeat = isIdlePose ? 4 : 1;
    buf.push(norm, repeat);

    if (!buf.ready()) return null;

    const x = buf.toTensor4D();
    if (!x) return null;

    const t0 = performance.now();

    const y = model.predict(x) as tf.Tensor;
    const probs = y.dataSync();

    const t1 = performance.now();
    const elapsed = t1 - t0;

    total += elapsed;
    count++;
    if (count % 50 === 0) {
      console.log(`Average inference latency: ${(total / count).toFixed(2)} ms`);
    }

    x.dispose();
    y.dispose();

    let bestI = 0;
    let bestP = probs[0] ?? 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > bestP) {
        bestP = probs[i];
        bestI = i;
      }
    }

    return {
      classIndex: bestI,
      label: LABELS[bestI] ?? `cls_${bestI}`,
      confidence: bestP,
      probs,
      nClasses,
    };
  };
}
