import { useEffect, useRef } from 'react';
import '@tensorflow/tfjs-backend-webgl';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { detectGesture } from '../ai/heuristics';

export default function DebugPose() {
  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let detector: poseDetection.PoseDetector | null = null;

    (async () => {
      /* webcam stream */
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) videoRef.current.srcObject = stream;

      /* MoveNet */
      detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
      );

      const loop = async () => {
        const video  = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || video.readyState !== 4 || !detector) {
          requestAnimationFrame(loop);
          return;
        }

        /* keep canvas buffer same as element size */
        if (
          canvas.width  !== canvas.clientWidth ||
          canvas.height !== canvas.clientHeight
        ) {
          canvas.width  = canvas.clientWidth;
          canvas.height = canvas.clientHeight;
        }

        /* scale factors: model-space to canvas-space */
        const sx = canvas.width  / video.videoWidth;
        const sy = canvas.height / video.videoHeight;

        const ctx = canvas.getContext('2d')!;
        const [pose] = await detector.estimatePoses(video);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        pose.keypoints.forEach(k => {
          if (k.score! > 0.5) {
            ctx.fillStyle = '#22d3ee';
            ctx.beginPath();
            ctx.arc(k.x * sx, k.y * sy, 5, 0, Math.PI * 2);
            ctx.fill();
          }
        });

        const label = detectGesture(pose);
        if (label) console.log(label);

        requestAnimationFrame(loop);
      };
      loop();
    })();

    return () => detector?.dispose();
  }, []);

  /* centred  wrapper */
  return (
    <div className="min-h-screen grid place-items-center bg-zinc-900">
      <div
        className="relative"
        style={{
          width:  'min(75vw, 1080px)',
          height: 'min(calc(75vw * 0.75), 810px)' // 4:3 aspect
        }}
      >
        <video
          ref={videoRef}
          className="absolute inset-0 w-full h-full object-cover bg-black"
          autoPlay
          muted
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />
      </div>
    </div>
  );
}
