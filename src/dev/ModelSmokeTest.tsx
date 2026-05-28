// src/dev/ModelSmokeTest.tsx
import { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { loadGestureLayers, LABELS } from '../ai/layersModel';

export default function ModelSmokeTest() {
  const [msg, setMsg] = useState('loading…');

  useEffect(() => {
    (async () => {
      try {
        await tf.setBackend('webgl');
        await tf.ready();

        const model = await loadGestureLayers('/models/gesture/model.json');

        // Create zeros matching [1,60,17,3]
        const x = tf.zeros([1, 60, 17, 3]);
        
        const y = model.predict(x) as tf.Tensor;   // [1,nClasses]
        const data = Array.from((await y.data()) as Float32Array);

        x.dispose(); y.dispose();

        setMsg(
          `Layers OK in=[1,60,17,3] out=[1,${data.length}] labels=${LABELS.length} ` +
          `first=[${data.slice(0,5).map(v => v.toFixed(4)).join(', ')}]`
        );
      } catch (e: any) {
        console.error(e);
        setMsg(`Error: ${e?.message || e}`);
      }
    })();
  }, []);

  return (
    <div style={{padding:16, color:'#fff', background:'#111', fontFamily:'system-ui,sans-serif'}}>
      <h2 style={{margin:0, fontSize:18}}>Model Smoke Test (Layers)</h2>
      <p style={{opacity:.9, marginTop:8}}>{msg}</p>
      <p style={{opacity:.7}}>
        This test runs a single forward pass with zeros using <code>tf.loadLayersModel</code>. 
        If it says “Layers OK”, model is workking as intended, proceed to the live webcam test.
      </p>
    </div>
  );
}
