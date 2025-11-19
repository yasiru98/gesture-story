// src/pages/LiveQuickTest.tsx
import { useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import * as poseDetection from '@tensorflow-models/pose-detection';
import type { Pose } from '@tensorflow-models/pose-detection';
import { loadGestureLayers, makeLayersStepper } from '../ai/layersModel';

export default function LiveQuickTest() {
  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let raf = 0;
    let detector: poseDetection.PoseDetector | null = null;

    (async () => {
      // Backend & model
      await tf.setBackend('webgl');
      await tf.ready();
      const model = await loadGestureLayers('/models/gesture/model.json');
      const step  = makeLayersStepper(model, 60);

      // Camera
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
      const video = videoRef.current!;
      video.srcObject = stream;

      await new Promise<void>(res => {
        const onMeta = () => { video.removeEventListener('loadedmetadata', onMeta); res(); };
        if (video.readyState >= 1) res(); else video.addEventListener('loadedmetadata', onMeta);
      });
      await video.play();

      // Canvas sizing
      const W = video.videoWidth, H = video.videoHeight;
      const canvas = canvasRef.current!;
      canvas.width = W; canvas.height = H;
      const ctx = canvas.getContext('2d')!;

      // Pose detector (MoveNet Lightning)
      detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
      );

      const loop = async () => {
        ctx.drawImage(video, 0, 0, W, H);

        const poses = await detector!.estimatePoses(video, { flipHorizontal: false });
        const pose: Pose | undefined = poses[0];

        let label = '…', conf = 0;
        if (pose && pose.keypoints?.length) {
          // draw keypoints
          ctx.fillStyle = '#22d3ee';
          for (const k of pose.keypoints) {
            if ((k.score ?? 0) > 0.5 && k.x != null && k.y != null) {
              ctx.beginPath(); ctx.arc(k.x, k.y, 4, 0, Math.PI*2); ctx.fill();
            }
          }
          const r = step(pose.keypoints);
          if (r) { label = r.label; conf = r.confidence; }
        }

        // HUD
        const text = `ML: ${label} (${conf.toFixed(2)})`;
        const pad = 8; const h = 26; ctx.font = '16px system-ui, sans-serif';
        const w = ctx.measureText(text).width + pad*2;
        ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(10, 10, w, h);
        ctx.fillStyle = '#fff'; ctx.fillText(text, 10 + pad, 10 + h - 8);

        raf = requestAnimationFrame(loop);
      };

      loop();
    })();

    return () => {
      if (raf) cancelAnimationFrame(raf);
      const v = videoRef.current;
      const s = (v?.srcObject as MediaStream) || null;
      s?.getTracks().forEach(t => t.stop());
    };
  }, []);

  return (
    <div style={{ minHeight:'100vh', display:'grid', placeItems:'center', background:'#18181b' }}>
      <video ref={videoRef} autoPlay muted playsInline style={{ display:'none' }} />
      <canvas ref={canvasRef} style={{ width:'min(75vw, 1080px)', height:'auto', boxShadow:'0 0 0 1px #2a2a2a' }} />
    </div>
  );
}
