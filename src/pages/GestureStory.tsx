// src/pages/GestureStory.tsx
import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import * as poseDetection from '@tensorflow-models/pose-detection';
import type { Pose, Keypoint } from '@tensorflow-models/pose-detection';

import { useMachine } from '@xstate/react';
import { loadGestureLayers, makeLayersStepper } from '../ai/layersModel';
import { storyMachine } from '../state/storyMachine';
import { GestureEventSmoother } from '../ai/gestureToEvent';

export default function GestureStory() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [ready, setReady] = useState(false);
  const [lastGesture, setLastGesture] = useState<{
    label: string;
    confidence: number;
  } | null>(null);

  const [state, send] = useMachine(storyMachine);

  // Debug: log state transitions
  useEffect(() => {
    console.log('[StoryMachine] state:', state.value);
  }, [state.value]);

  useEffect(() => {
    let raf = 0;
    let detector: poseDetection.PoseDetector | null = null;
    let cancelled = false;

    (async () => {
      try {
        // ---- Backend + gesture model (same as LiveQuickTest) ----
        await tf.setBackend('webgl');
        await tf.ready();

        const model = await loadGestureLayers('/models/gesture/model.json');
        const step = makeLayersStepper(model, 60);

        // ---- Gesture smoother for story events with idle gating ----
        const smoother = new GestureEventSmoother();

        // ---- Camera setup ----
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user' },
        });
        const video = videoRef.current!;
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

        // ---- Canvas sizing ----
        const canvas = canvasRef.current;
        let ctx: CanvasRenderingContext2D | null = null;
        let W = 0;
        let H = 0;

        if (canvas) {
          W = video.videoWidth || 640;
          H = video.videoHeight || 480;
          canvas.width = W;
          canvas.height = H;
          ctx = canvas.getContext('2d');
        }

        // ---- MoveNet Lightning detector ----
        detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
          { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
        );

        // ---- Main loop: pose to gesture to story event ----
        const loop = async () => {
          if (cancelled || !detector) return;

          const poses: Pose[] = await detector.estimatePoses(video, {
            flipHorizontal: false,
          });
          const pose: Pose | undefined = poses[0];

          // Draw video frame + keypoints on canvas
          if (ctx && W && H) {
            ctx.clearRect(0, 0, W, H);
            ctx.drawImage(video, 0, 0, W, H);

            if (pose && pose.keypoints?.length) {
              ctx.fillStyle = '#22d3ee';
              for (const k of pose.keypoints) {
                const score = k.score ?? 0;
                if (score > 0.5 && k.x != null && k.y != null) {
                  ctx.beginPath();
                  ctx.arc(k.x, k.y, 4, 0, Math.PI * 2);
                  ctx.fill();
                }
              }
            }
          }

          // Gesture to story
          if (pose && pose.keypoints?.length) {
            const r = step(pose.keypoints as Keypoint[]);

            if (r) {
              const { label, confidence } = r;
              setLastGesture({ label, confidence });

              const ev = smoother.push({
                label,
                confidence,
                classIndex: r.classIndex,
                probs: r.probs,
                nClasses: r.nClasses,
              });

              if (ev) {
                console.log('[GestureStory] sending event from gesture:', ev);
                send(ev);
              }
            }
          }

          raf = requestAnimationFrame(loop);
        };

        loop();
      } catch (err) {
        console.error('[GestureStory] init error:', err);
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
      {/* Left: Story + instructions */}
      <div>
        <h1 style={{ fontSize: 28, marginBottom: 8 }}>Gesture-Driven Story</h1>
        <p style={{ opacity: 0.8, marginBottom: 24 }}>
          Use your gestures to navigate the branching narrative. After each
          choice, briefly return to a relaxed <strong>idle</strong> pose before
          making the next gesture.
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
          <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 12 }}>
            {title}
          </div>
          <p style={{ fontSize: 15, lineHeight: 1.6, whiteSpace: 'pre-line' }}>
            {body}
          </p>
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
      </div>

      {/* Right: Webcam + debug */}
      <div>
        <div
          style={{
            borderRadius: 16,
            overflow: 'hidden',
            background: '#000',
            aspectRatio: '4 / 3',
            marginBottom: 12,
            position: 'relative',
          }}
        >
          {/* Hidden video: draw onto the canvas */}
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            style={{ display: 'none' }}
          />
          <canvas
            ref={canvasRef}
            style={{ width: '100%', height: '100%', display: 'block' }}
          />
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
            <div>Webcam: {ready ? '✔️ ready' : '⏳ starting…'}</div>
            <div>
              Story state: <code>{String(state.value)}</code>
            </div>
            <div>
              Last gesture:{' '}
              {lastGesture
                ? `${lastGesture.label} (${lastGesture.confidence.toFixed(2)})`
                : '—'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/** simple text per state. */
function describeState(state: string): { title: string; body: string; hint: string } {
  switch (state) {
    case 'intro':
      return {
        title: 'At the Edge of the Forest',
        body:
          'You stand at the edge of a quiet forest. A narrow trail disappears between the trees.\n\n' +
          'Sit centered in front of your camera, keep your upper body and wrists visible, and make sure the room is reasonably well lit.\n\n' +
          'When you are ready, gesture or click Start to step into the story.',
        hint:
          'Gestures: raise LEFT arm, raise RIGHT arm, cross your arms, or raise BOTH arms. Return to idle after each story beat.',
      };
    case 'choice1':
      return {
        title: 'The Fork in the Path',
        body:
          'The forest path splits in two.\n\nTo your LEFT, the trees grow denser and the light softens.\n' +
          'To your RIGHT, the trail climbs toward a faint glow on the horizon.',
        hint:
          'Raise LEFT arm to take the shadowed path. Raise RIGHT arm to follow the light. Cross arms to go back. Remember to return to idle afterward.',
      };
    case 'pathLeft_intro':
      return {
        title: 'Shadowed Canopy',
        body:
          'You step into the dim, cool air beneath the canopy. The sounds of the outside world fade.\n\n' +
          'Soft bioluminescent shapes flicker between the roots.',
        hint:
          'Raise BOTH arms to move deeper. Cross arms to return to the fork. Return to idle to confirm each choice.',
      };
    case 'pathLeft_deep':
      return {
        title: 'Whispers Between Trees',
        body:
          'The forest seems to lean closer, branches arching overhead like vaulted ceilings.\n\n' +
          'You feel as though the forest is listening to your every move.',
        hint:
          'Raise BOTH arms to reach a reflective ending. Cross arms to step back. Return to idle before the next gesture.',
      };
    case 'pathRight_intro':
      return {
        title: 'Climbing Toward the Glow',
        body:
          'You follow the rising trail. Wind picks up, carrying the scent of rain.\n\n' +
          'Ahead, the glow pulses slowly, like a heartbeat.',
        hint:
          'Raise BOTH arms to move closer to the glow. Cross arms to return to the fork. Return to idle to continue.',
      };
    case 'pathRight_deep':
      return {
        title: 'The Silent Beacon',
        body:
          'You reach a clearing where a strange, silent beacon rises from the ground.\n\n' +
          'Its surface reflects not your face, but fragments of choices you have not yet made.',
        hint:
          'Raise BOTH arms to trigger a mysterious ending. Cross arms to step back. Return to idle between choices.',
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
          'The beacon flashes once, then goes dark.\n\nLater, you will wonder whether it responded to your gesture, ' +
          'your presence, or something else entirely.',
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
