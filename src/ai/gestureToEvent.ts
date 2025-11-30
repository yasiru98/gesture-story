// src/ai/gestureToEvent.ts

// One model prediction at a single time step.
export interface GestureResult {
  label: string;
  confidence: number;
  classIndex: number;
  probs: Float32Array | number[];
  nClasses: number;
}

// The events that drive the story state machine.
export type StoryEvent =
  | { type: 'CHOICE_LEFT' }
  | { type: 'CHOICE_RIGHT' }
  | { type: 'CONFIRM' }
  | { type: 'BACK' };

// Map a gesture label string to a story event.
function gestureLabelToEvent(label: string): StoryEvent | null {
  switch (label) {
    case 'raise_left':
      return { type: 'CHOICE_LEFT' };
    case 'raise_right':
      return { type: 'CHOICE_RIGHT' };
    case 'both_arms_up':
      return { type: 'CONFIRM' };
    case 'cross_arms':
      return { type: 'BACK' };
    default:
      return null;
  }
}

/**
 * GestureEventSmoother
 *
 * - Keeps a short history of recent gesture labels.
 * - Computes the dominant label and a simple dominance ratio.
 * - Only fires when a non-idle label becomes dominant and passes thresholds.
 * - Edge-triggered: the same dominant gesture will fire at most once
 *   until the dominant label changes (usually by going back to idle).
 */
export class GestureEventSmoother {
  private history: string[] = [];
  private maxLen: number;
  private minDominance: number;
  private minConfidence: number;

  // Remember the last dominant label that actually triggered an event.
  private lastFiredDominant: string | null = null;

  constructor(maxLen = 8, minDominance = 0.6, minConfidence = 0.6) {
    this.maxLen = maxLen;
    this.minDominance = minDominance;
    this.minConfidence = minConfidence;
  }

  /**
   * Push a new model result and possibly return a story event.
   * Returns null when we decide not to fire anything.
   */
  push(r: GestureResult): StoryEvent | null {
    const { label, confidence } = r;

    // Always track the label in the history so dominance is smooth.
    this.history.push(label);
    if (this.history.length > this.maxLen) {
      this.history.shift();
    }

    // If the current prediction is very weak, do not try to fire.
    if (!Number.isFinite(confidence) || confidence < this.minConfidence) {
      return null;
    }

    // Count labels in the current window.
    const counts: Record<string, number> = {};
    for (const lab of this.history) {
      counts[lab] = (counts[lab] ?? 0) + 1;
    }

    // Find the dominant label.
    let bestLabel: string | null = null;
    let bestCount = 0;
    for (const [lab, cnt] of Object.entries(counts)) {
      if (cnt > bestCount) {
        bestCount = cnt;
        bestLabel = lab;
      }
    }

    if (!bestLabel) {
      return null;
    }

    const dominance = bestCount / this.history.length;
    if (dominance < this.minDominance) {
      // No label dominates the window strongly enough.
      return null;
    }

    // If idle is dominant, treat this as a reset point but do not fire.
    if (bestLabel === 'idle') {
      this.lastFiredDominant = 'idle';
      return null;
    }

    // Edge trigger:
    // Only fire when the dominant non-idle label CHANGES compared to
    // the last dominant label that fired. This prevents repeated events
    // while the user holds the same gesture.
    if (this.lastFiredDominant === bestLabel) {
      return null;
    }

    // New dominant gesture: remember it and map to a story event.
    this.lastFiredDominant = bestLabel;

    const ev = gestureLabelToEvent(bestLabel);
    return ev;
  }
}
