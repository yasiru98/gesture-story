# Gesture-Driven Storytelling

A fully client-side, browser-based interactive narrative controlled entirely by upper-body gestures. No server, no cloud API, no specialised hardware — just a standard webcam and a modern browser.

**Live demo → [gesture-story.netlify.app](https://gesture-story.netlify.app)**

---

## What it does

The user navigates a branching story by performing upper-body gestures in front of their webcam. A lightweight neural network runs entirely in the browser using TensorFlow.js with WebGL acceleration, classifying gestures in real time and driving a finite state machine that controls narrative progression.

| Gesture | Story action |
|---|---|
| Raise left arm | CHOICE\_LEFT — take the left path |
| Raise right arm | CHOICE\_RIGHT — take the right path |
| Both arms up | CONFIRM — move forward / confirm |
| Cross arms | BACK — return to previous state |
| Idle | No action — resets the gesture window |

---

## Tech stack

| Layer | Technology |
|---|---|
| UI framework | React 19 + TypeScript |
| Styling | Tailwind CSS |
| Bundler | Vite |
| Pose estimation | TensorFlow.js + MoveNet Lightning (WebGL) |
| Gesture classifier | TensorFlow.js Layers (Bidirectional GRU, ~1 MB) |
| Story state machine | XState v5 |
| Deployment | Netlify |

---

## Architecture

The full pipeline runs on the client side in a single `requestAnimationFrame` loop:

```
Webcam (30 fps)
  → MoveNet Lightning          — 17 body keypoints per frame
  → normalizeFrame()           — torso-centre + shoulder-width normalisation
  → SeqBuf (60-frame buffer)   — rolling temporal window
  → Bidirectional GRU model    — softmax over 5 gesture classes (~80 ms/window)
  → GestureEventSmoother       — dominance + confidence gate, edge-triggered
  → XState storyMachine        — narrative state transitions
  → React UI                   — story text, gesture guide, debug HUD
```

### Key design decisions

**Training/inference normalisation parity**
Every frame is normalised by torso centre and shoulder width before reaching the classifier — identical to how training data was preprocessed. This makes the system scale-invariant: the model sees body-relative coordinates regardless of how far the user sits from the camera, their body proportions, or camera resolution.

**Idle-priority buffer flushing**
The GRU operates on a 60-frame (≈2 second) window. When the user relaxes after a gesture, raised-arm frames linger in the window and delay the return to idle. When MoveNet detects both wrists clearly below shoulder height with high confidence, the current frame is pushed into the buffer four times, flushing gesture frames out four times faster without retraining the model.

**Edge-triggered event smoother**
`GestureEventSmoother` maintains an 8-frame label history and only fires a story event when one label dominates ≥60% of the window with ≥60% classifier confidence. It is edge-triggered: the same gesture will not fire twice until the window returns to idle, preventing held gestures from spamming the state machine.

**Decoupled story logic**
The XState machine receives typed `StoryEvent` objects — `CHOICE_LEFT`, `CHOICE_RIGHT`, `CONFIRM`, `BACK`. It has no knowledge of the gesture layer. Swapping gestures for keyboard, voice, or any other input requires no changes to story logic.

---

## Project structure

```
src/
  App.tsx                  — root, renders GestureStory
  main.tsx                 — entry point

  pages/
    GestureStory.tsx       — main page: webcam → pipeline → UI

  ai/
    layersModel.ts         — model loading, frame normalisation, SeqBuf, stepper
    gestureToEvent.ts      — GestureEventSmoother, gesture→StoryEvent mapping
    heuristics.ts          — heuristic(rule-based) detector (reference only, not used in production)

  state/
    storyMachine.ts        — XState v5 finite state machine (8 states, 4 event types)

  dev/                     — development and testing tools (not routed in production)
    DebugPose.tsx          — live pose keypoint visualiser with heuristic overlay
    LiveQuickTest.tsx      — live gesture label + confidence tester
    ModelSmokeTest.tsx     — model load + single forward-pass sanity check

public/
  assets/                  — gesture reference images (shown in the gesture guide)
  models/gesture/          — TF.js model artefacts (model.json + weight shard ~1 MB)
```

---

## Getting started

**Prerequisites:** Node.js 18+

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Production build
npm run build
```

The app opens at `http://localhost:5173`. Grant camera permission when prompted. The model loads from `public/models/gesture/` — no network requests are made at runtime.

> **First load note:** allow 2–3 seconds on first visit while the WebGL backend compiles shaders and loads model weights. Subsequent inference runs at 62–65 ms per window.

---

## Model

The gesture classifier is a two-layer Bidirectional GRU network trained on a custom dataset of six participants performing five upper-body gestures across multiple recording sessions.

| Property | Value |
|---|---|
| Architecture | Bi-GRU (128) → Bi-GRU (64) → Dense (ReLU) → Softmax (5) |
| Input shape | (60, 17, 3) — frames × joints × (x, y, confidence) |
| Model size | ~1 MB (JSON + binary weights) |
| Inference latency | 62–65 ms / window (WebGL, RTX 4060) |
| LOPO mean accuracy | 0.989 |
| LOPO mean macro-F1 | 0.988 |

The model was trained offline in Keras and converted to TensorFlow.js Layers format. Training scripts, data preprocessing pipeline, and cross-validation runner live in the `tools/` directory of the parent repository.

---

## Performance

Measured on Windows 11, NVIDIA RTX 4060, Firefox 146, WebGL backend:

| Metric | Value |
|---|---|
| Cold start (backend + model load) | ~2.7 s |
| Warm-up inference (first call) | 240–300 ms |
| Steady-state inference | 62–65 ms / window |
| Effective gesture update rate | ~10–12 fps |
| End-to-end gesture-to-UI latency | < 100 ms |

---

## Story structure

```
intro
  └─ choice1
       ├─ pathLeft_intro
       │    └─ pathLeft_deep
       │         └─ ending_reflective  (final)
       └─ pathRight_intro
            └─ pathRight_deep
                 └─ ending_mysterious  (final)
```

Any state with a `BACK` transition returns to its parent. The machine is defined in [`src/state/storyMachine.ts`](src/state/storyMachine.ts).

---

## Future work

- Add 2D / 3D visual scenes to the narrative
- Expand gesture vocabulary to include dynamic and compound gestures
- Per-session calibration to adapt normalisation to new environments
- Structured user study measuring perceived responsiveness
- Explore multimodal interaction (gesture + audio)
