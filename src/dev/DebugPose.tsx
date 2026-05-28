import { useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import * as poseDetection from '@tensorflow-models/pose-detection';
import type { Pose } from '@tensorflow-models/pose-detection';
import { detectGesture } from '../ai/heuristics';

export default function DebugPose() {
  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let detector: poseDetection.PoseDetector | null = null;
    let raf = 0;

    (async () => {
      // Backend
      await tf.setBackend('webgl');
      await tf.ready();

      // Camera
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' } // OK on laptops too
      });
      const video = videoRef.current!;
      video.srcObject = stream;

      await new Promise<void>(res => {
        const onMeta = () => { video.removeEventListener('loadedmetadata', onMeta); res(); };
        if (video.readyState >= 1) res(); else video.addEventListener('loadedmetadata', onMeta);
      });
      await video.play();

      // Canvas size = video pixels 1:1 mapping to avoid overlay/sizing/placement issues
      const W = video.videoWidth;
      const H = video.videoHeight;
      const canvas = canvasRef.current!;
      canvas.width = W;
      canvas.height = H;

      // movenet integration
      detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
      );

      const ctx = canvas.getContext('2d')!;

      const loop = async () => {
        // 1draw the current video frame
        ctx.drawImage(video, 0, 0, W, H);

        // estimate pose (no flipping)
        const poses = await detector!.estimatePoses(video);
        const pose: Pose | undefined = poses[0];

        // overlay keypoints
        if (pose) {
          ctx.fillStyle = '#22d3ee';
          for (const k of pose.keypoints) {
            if ((k.score ?? 0) > 0.5 && k.x != null && k.y != null) {
              ctx.beginPath();
              ctx.arc(k.x, k.y, 5, 0, Math.PI * 2);
              ctx.fill();
            }
          }

          // heuristics
          const label = detectGesture(pose);
          if (label) {
            console.log('Gesture:', label);
            ctx.font = '16px system-ui, sans-serif';
            const text = `Gesture: ${label}`;
            const pad = 8;
            const w = ctx.measureText(text).width + pad * 2;
            const h = 26;
            ctx.fillStyle = 'rgba(0,0,0,0.6)';
            ctx.fillRect(10, 10, w, h);
            ctx.fillStyle = '#fff';
            ctx.fillText(text, 10 + pad, 10 + h - 8);
          }
        }

        raf = requestAnimationFrame(loop);
      };

      loop();
    })();

    return () => {
      if (raf) cancelAnimationFrame(raf);
      detector?.dispose();
      const v = videoRef.current;
      const s = (v?.srcObject as MediaStream) || null;
      s?.getTracks().forEach(t => t.stop());
    };
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'grid', placeItems: 'center', background: '#18181b' }}>
      {/* hidden video drawn onto the canvas each frame */}
      <video ref={videoRef} autoPlay muted playsInline style={{ display: 'none' }} />
      {/* responsive display sizing while keeping internal pixels 1:1 to video pixels */}
      <canvas ref={canvasRef} style={{ width: 'min(75vw, 1080px)', height: 'auto', boxShadow: '0 0 0 1px #2a2a2a' }} />
    </div>
  );
}
