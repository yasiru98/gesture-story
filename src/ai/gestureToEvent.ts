// src/ai/gestureToEvent.ts
import type { StoryEvent } from '../state/storyMachine';

export type GestureResult = {
  label: string;
  confidence: number;
  classIndex: number;
  probs: Float32Array | number[];
  nClasses: number;
};

/**
 * Smoothing + gating:
 *  - keeps last N high-confidence gestures
 *  - fires only when one gesture clearly dominates
 *  - after firing, requires a *run* of idle frames before firing again
 */
export class GestureEventSmoother {
  private history: string[] = [];
  private maxLen: number;
  private minConf: number;

  private waitingForIdle = false;
  private lastFiredLabel: string | null = null;

  // how many consecutive idle frames we require to unlock
  private idleStreak = 0;
  private idleUnlockFrames: number;

  constructor(maxLen = 8, minConf = 0.6, idleUnlockFrames = 6) {
    this.maxLen = maxLen;
    this.minConf = minConf;
    this.idleUnlockFrames = idleUnlockFrames; // ~8 frames ≈ 250–300ms at 30 FPS
  }

  push(res: GestureResult | null): StoryEvent | null {
    if (!res) return null;

    const { label, confidence } = res;

    // ---------- IDLE HANDLING ----------
    if (label === 'idle') {
      // count how long user been idle
      this.idleStreak++;

      if (this.idleStreak >= this.idleUnlockFrames) {
        // only after a *run* of idle frames -> fully unlock
        this.waitingForIdle = false;
        this.lastFiredLabel = null;
        this.history = [];
      }

      // never fire events on idle itself
      return null;
    } else {
      // as soon as system sees a non-idle label, reset the streak
      this.idleStreak = 0;
    }

    // Too low confidence -> ignore
    if (confidence < this.minConf) return null;

    // If system already fired a gesture and haven't had enough idle yet, block
    if (this.waitingForIdle) return null;

    // ---------- ACCUMULATE LABEL HISTORY ----------
    this.history.push(label);
    if (this.history.length > this.maxLen) this.history.shift();

    const counts: Record<string, number> = {};
    for (const l of this.history) {
      counts[l] = (counts[l] ?? 0) + 1;
    }

    let bestLabel = label;
    let bestCount = 0;
    for (const [l, c] of Object.entries(counts)) {
      if (c > bestCount) {
        bestCount = c;
        bestLabel = l;
      }
    }

    const dominance = bestCount / this.history.length;

    // Only fire if one gesture clearly dominates the recent window
    if (dominance < 0.5) return null;

    // Don't fire the same gesture repeatedly without a real idle in between
    if (bestLabel === this.lastFiredLabel) return null;

    // ---------- MAP GESTURE → STORY EVENT ----------
    let ev: StoryEvent | null = null;
    switch (bestLabel) {
      case 'raise_right':
        ev = { type: 'CHOICE_RIGHT' };
        break;
      case 'raise_left':
        ev = { type: 'CHOICE_LEFT' };
        break;
      case 'both_arms_up':
        ev = { type: 'CONFIRM' };
        break;
      case 'cross_arms':
        ev = { type: 'BACK' };
        break;
      default:
        ev = null;
    }

    if (ev) {
      console.log(
        '[GestureEventSmoother] firing event',
        ev,
        'from label',
        bestLabel,
        'dominance',
        dominance.toFixed(2)
      );
      // After a fire, require a new idle streak before we allow another
      this.history = [];
      this.waitingForIdle = true;
      this.lastFiredLabel = bestLabel;
      this.idleStreak = 0;
    }

    return ev;
  }
}
