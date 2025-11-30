// src/pages/GestureStory.tsx
import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import * as poseDetection from '@tensorflow-models/pose-detection';
import type { Pose, Keypoint } from '@tensorflow-models/pose-detection';

import { useMachine } from '@xstate/react';
import { loadGestureLayers, makeLayersStepper, LABELS } from '../ai/layersModel';
import { storyMachine } from '../state/storyMachine';
import { GestureEventSmoother } from '../ai/gestureToEvent';

export default function GestureStory() {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const [ready, setReady] = useState(false);
  const [lastGesture, setLastGesture] = useState<{
    label: string;
    confidence: number;
  } | null>(null);

  const [error, setError] = useState<string | null>(null);

  const [lastProbs, setLastProbs] = useState<number[] | null>(null);
  const [showDebugHud, setShowDebugHud] = useState(false);

  // While the model has not yet seen a clear idle, show a loading overlay
  const [waitingForIdle, setWaitingForIdle] = useState(true);

  // Track when we last saw a strong non idle gesture for display smoothing
  const lastStrongGestureRef = useRef<{ label: string; t: number } | null>(null);

  const [state, send] = useMachine(storyMachine);

  // Debug: log state transitions
  useEffect(() => {
    if (process.env.NODE_ENV !== 'production') {
      console.log('[StoryMachine] state:', state.value);
    }
  }, [state.value]);

  useEffect(() => {
    let raf = 0;
    let detector: poseDetection.PoseDetector | null = null;
    let cancelled = false;

    const MIN_INFER_INTERVAL_MS = 80;
    let lastInferTime = 0;

    (async () => {
      try {
        await tf.setBackend('webgl');
        await tf.ready();

        const model = await loadGestureLayers('/models/gesture/model.json');
        const step = makeLayersStepper(model, 60);
        const smoother = new GestureEventSmoother(6, 0.6);

        const video = videoRef.current;
        if (!video) {
          console.error('[GestureStory] No video element');
          setError('Video element is not available.');
          return;
        }

        // Camera setup
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user' },
          audio: false,
        });
        video.srcObject = stream;

        await new Promise<void>((res) => {
          const onMeta = () => {
            video.removeEventListener('loadedmetadata', onMeta);
            res();
          };
          if (video.readyState >= 1) res();
          else video.addEventListener('loadedmetadata', onMeta);
        });
        await video.play();

        setReady(true);

        // MoveNet detector
        detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
          { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING },
        );

        // Main loop: pose to gesture to story event
        const loop = async (ts: number) => {
          if (cancelled || !detector) return;

          const poses: Pose[] = await detector.estimatePoses(video, {
            flipHorizontal: false,
          });
          const pose = poses[0];

          if (pose && pose.keypoints && pose.keypoints.length) {
            if (ts - lastInferTime >= MIN_INFER_INTERVAL_MS) {
              lastInferTime = ts;

              const r = step(pose.keypoints as Keypoint[]);

              if (r) {
                const now = performance.now();
                const { label, confidence } = r;

                // Keep last probabilities for debug HUD
                setLastProbs(Array.from(r.probs));

                // Track last strong non idle gesture
                if (label !== 'idle' && confidence >= 0.7) {
                  lastStrongGestureRef.current = { label, t: now };
                }

                // Decide what to show as current gesture in the HUD
                let displayLabel = label;
                if (label === 'idle') {
                  displayLabel = 'idle';
                } else {
                  const lastStrong = lastStrongGestureRef.current;
                  const since = lastStrong ? now - lastStrong.t : Infinity;

                  if (since > 400 && confidence < 0.7) {
                    displayLabel = 'idle';
                  }
                }

                setLastGesture({ label: displayLabel, confidence });

                // As soon as we see idle on the display side, remove the overlay
                if (displayLabel === 'idle') {
                  setWaitingForIdle((prev) => (prev ? false : prev));
                }

                // For story logic, still use raw model label
                const ev = smoother.push({
                  label,
                  confidence,
                  classIndex: r.classIndex,
                  probs: Array.from(r.probs),
                  nClasses: r.nClasses,
                });

                if (ev) {
                  console.log('[GestureStory] sending event from gesture:', ev);
                  send(ev);
                }
              }
            }
          }

          raf = requestAnimationFrame(loop);
        };

        raf = requestAnimationFrame(loop);
      } catch (err) {
        console.error('[GestureStory] init error:', err);
        setError(
          'Unable to start camera or load models. Check camera permissions and reload the page.',
        );
      }
    })();

    return () => {
      cancelled = true;
      if (raf) cancelAnimationFrame(raf);

      const v = videoRef.current;
      const s = (v?.srcObject as MediaStream) || null;
      s?.getTracks().forEach((t) => t.stop());

      detector?.dispose?.();
    };
  }, [send]);

  const { title, body, hint } = describeState(state.value as string);

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 1.2fr) minmax(0, 1fr)',
        gap: 24,
        padding: 24,
        background: '#05060a',
        color: '#f4f4f4',
        fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, sans-serif',
        minHeight: '100vh',
      }}
    >
      {/* Left: Story + gesture guide */}
      <div>
        <h1 style={{ fontSize: 28, marginBottom: 8 }}>Gesture Driven Story</h1>
        <p style={{ opacity: 0.8, marginBottom: 24 }}>
          Use your gestures to navigate the branching narrative. Raise your left or right arm,
          cross your arms, or raise both arms. Return to idle after each gesture/story event.
        </p>

        <div
          style={{
            borderRadius: 16,
            padding: 20,
            background: 'linear-gradient(145deg, #151827, #1f2437)',
            boxShadow: '0 20px 50px rgba(0,0,0,0.4)',
            minHeight: 260,
          }}
        >
          <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 12 }}>{title}</div>
          <p style={{ fontSize: 15, lineHeight: 1.6, whiteSpace: 'pre-line' }}>{body}</p>
          <p style={{ marginTop: 18, fontSize: 13, opacity: 0.75 }}>{hint}</p>
        </div>

        {state.matches('intro') && (
          <button
            onClick={() => send({ type: 'START' })}
            style={{
              marginTop: 16,
              padding: '10px 18px',
              borderRadius: 999,
              border: 'none',
              background: '#4f46e5',
              color: '#fff',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Start story (or use gestures)
          </button>
        )}

        {/* Full gesture guide panel */}
        <div
          style={{
            marginTop: 22,
            borderRadius: 16,
            padding: 16,
            background: '#020617',
            border: '1px solid rgba(148,163,184,0.4)',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'baseline',
              marginBottom: 10,
            }}
          >
            <div style={{ fontWeight: 600 }}>Gesture guide</div>
            <div style={{ fontSize: 11, opacity: 0.75 }}>
              Sit centered in frame with shoulders and wrists visible.
            </div>
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
              gap: 10,
            }}
          >
            <GestureGuideItem
              title="Raise left arm"
              caption="Choose left path"
              kind="left"
            />
            <GestureGuideItem
              title="Raise right arm"
              caption="Choose right path"
              kind="right"
            />
            <GestureGuideItem
              title="Both arms up"
              caption="Confirm / move forward"
              kind="both"
            />
            <GestureGuideItem
              title="Cross arms"
              caption="Go back"
              kind="cross"
            />
          </div>
        </div>
      </div>

      {/* Right: Webcam + status + debug */}
      <div>
        <div
          style={{
            position: 'relative',
            borderRadius: 16,
            overflow: 'hidden',
            background: '#000',
            aspectRatio: '4 / 3',
            marginBottom: 12,
          }}
        >
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />

          {waitingForIdle && (
            <div
              style={{
                position: 'absolute',
                inset: 0,
                background: 'rgba(15,23,42,0.82)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: 16,
                textAlign: 'center',
              }}
            >
              <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>
                Waiting for idle pose
              </div>
              <div style={{ fontSize: 13, opacity: 0.85, maxWidth: 260 }}>
                Sit comfortably in front of the camera with your arms relaxed on the armrests.
                The system will start once it detects an idle posture.
              </div>
            </div>
          )}
        </div>

        <div
          style={{
            borderRadius: 12,
            padding: 12,
            background: '#111827',
            fontSize: 13,
          }}
        >
          <div style={{ marginBottom: 4, fontWeight: 600 }}>Status</div>
          <div style={{ opacity: 0.9 }}>
            <div>Webcam: {ready ? '✅ ready' : '⏳ starting…'}</div>
            <div>
              Story state: <code>{String(state.value)}</code>
            </div>
            <div>
              Last gesture{' '}
              {lastGesture
                ? `${lastGesture.label} (${lastGesture.confidence.toFixed(2)})`
                : '—'}
            </div>
            {error && (
              <div style={{ marginTop: 8, color: '#f87171' }}>
                {error}
              </div>
            )}
          </div>

          {/* Toggle button for probability HUD */}
          <button
            onClick={() => setShowDebugHud((v) => !v)}
            style={{
              marginTop: 10,
              padding: '6px 10px',
              borderRadius: 999,
              border: '1px solid #4b5563',
              background: showDebugHud ? '#1f2937' : '#111827',
              color: '#e5e7eb',
              fontSize: 12,
              cursor: 'pointer',
            }}
          >
            {showDebugHud ? 'Hide debug HUD' : 'Show debug HUD'}
          </button>

          {/* Probability HUD */}
          {showDebugHud && (
            <div
              style={{
                marginTop: 10,
                padding: 8,
                borderRadius: 8,
                background: '#020617',
                fontSize: 12,
                lineHeight: 1.4,
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: 4 }}>
                Gesture probabilities
              </div>
              {lastProbs ? (
                <div>
                  {LABELS.map((lab, i) => {
                    const p = lastProbs[i] ?? 0;
                    const barWidth = Math.max(4, Math.round(p * 100));
                    return (
                      <div
                        key={lab}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          marginBottom: 3,
                        }}
                      >
                        <div style={{ width: 90 }}>{lab}</div>
                        <div
                          style={{
                            flex: 1,
                            height: 6,
                            borderRadius: 4,
                            background: '#111827',
                            overflow: 'hidden',
                            marginRight: 6,
                          }}
                        >
                          <div
                            style={{
                              width: `${barWidth}%`,
                              height: '100%',
                              borderRadius: 4,
                              background: '#4f46e5',
                              transition: 'width 80ms linear',
                            }}
                          />
                        </div>
                        <div style={{ width: 40, textAlign: 'right' }}>
                          {p.toFixed(2)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div style={{ opacity: 0.7 }}>No predictions yet.</div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/** One gesture item with an image and text. */
const GestureGuideItem: React.FC<{
  title: string;
  caption: string;
  kind: 'left' | 'right' | 'both' | 'cross';
}> = ({ title, caption, kind }) => {
  const src = getGestureImageSrc(kind);

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        background: 'rgba(15,23,42,0.7)',
        borderRadius: 8,
        padding: 6,
      }}
    >
      <div
        style={{
          flexShrink: 0,
          width: 40,
          height: 40,
          borderRadius: 10,
          background: '#020617',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          overflow: 'hidden',
        }}
      >
        <img
          src={src}
          alt={title}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'contain',
          }}
        />
      </div>
      <div style={{ fontSize: 11 }}>
        <div style={{ fontWeight: 600 }}>{title}</div>
        <div style={{ opacity: 0.8 }}>{caption}</div>
      </div>
    </div>
  );
};

function getGestureImageSrc(kind: 'left' | 'right' | 'both' | 'cross'): string {
  switch (kind) {
    case 'left':
      return '/assets/raise_left.jpeg';
    case 'right':
      return '/assets/raise_right.jpeg';
    case 'both':
      return '/assets/both_arms_up.jpeg';
    case 'cross':
      return '/assets/cross_arms.jpeg';
    default:
      return '/assets/raise_right.jpeg';
  }
}

/** Simple text for each story state. */
function describeState(state: string): {
  title: string;
  body: string;
  hint: string;
} {
  switch (state) {
    case 'intro':
      return {
        title: 'At the Edge of the Forest',
        body:
          'You stand at the edge of a quiet forest. A narrow trail disappears between the trees.\n\n' +
          'When you are ready, gesture with rasing your right/left arm or click Start to step into the story.',
        hint: 'Raise your left or right arm to choose your first direction once the story begins.',
      };
    case 'choice1':
      return {
        title: 'The Fork in the Path',
        body:
          'The forest path splits in two.\n\nTo your LEFT, the trees grow denser and the light softens.\n' +
          'To your RIGHT, the trail climbs toward a faint glow on the horizon.',
        hint:
          'Raise LEFT arm to take the shadowed path. Raise RIGHT arm to follow the light. Cross arms to go back.',
      };
    case 'pathLeft_intro':
      return {
        title: 'Shadowed Canopy',
        body:
          'You step into the dim, cool air beneath the canopy. The sounds of the outside world fade.\n\n' +
          'Soft bioluminescent shapes flicker between the roots.',
        hint:
          'Raise BOTH arms to move deeper. Cross arms to return to the fork.',
      };
    case 'pathLeft_deep':
      return {
        title: 'Whispers Between Trees',
        body:
          'The forest seems to lean closer, branches arching overhead like vaulted ceilings.\n\n' +
          'You feel as though the forest is listening to your every move.',
        hint:
          'Raise BOTH arms to reach a reflective ending. Cross arms to step back.',
      };
    case 'pathRight_intro':
      return {
        title: 'Climbing Toward the Glow',
        body:
          'You follow the rising trail. Wind picks up, carrying the scent of rain.\n\n' +
          'Ahead, the glow pulses slowly, like a heartbeat.',
        hint:
          'Raise BOTH arms to move closer to the glow. Cross arms to return to the fork.',
      };
    case 'pathRight_deep':
      return {
        title: 'The Silent Beacon',
        body:
          'You reach a clearing where a strange, silent beacon rises from the ground.\n\n' +
          'Its surface reflects not your face, but fragments of choices you have not yet made.',
        hint:
          'Raise BOTH arms to trigger a mysterious ending. Cross arms to step back.',
      };
    case 'ending_reflective':
      return {
        title: 'Ending: Quiet Reflection',
        body:
          'You sit beneath the old trees as the forest settles around you.\n\n' +
          'With every breath, you feel more attuned to the small decisions that shaped your path.',
        hint: 'Reload the page to experience another branch.',
      };
    case 'ending_mysterious':
      return {
        title: 'Ending: The Unanswered Signal',
        body:
          'The beacon flashes once, then goes dark.\n\nLater, you will wonder whether it responded to your gesture, your presence, or something else entirely.',
        hint: 'Reload the page to explore a different route.',
      };
    default:
      return {
        title: 'Story',
        body: 'Unknown state.',
        hint: '',
      };
  }
}
