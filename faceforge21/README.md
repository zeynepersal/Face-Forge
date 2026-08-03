# FaceForge v2.0 — Setup Instructions

## Requirements
```
pip install flask flask-cors opencv-python mediapipe numpy Pillow
```

## How to Run

**Step 1 — Start the Python backend:**
```bash
python3 backend.py
```
You should see: `FaceForge backend running on http://localhost:5050`

**Step 2 — Open the frontend:**
Just open `index.html` in your browser (Chrome recommended).

## Features

| Feature | Where |
|---|---|
| Landmark detection (real MediaPipe) | Landmarks page → Detect Landmarks |
| Smile / Eyebrow / Lips / Slim warp | Expression page → toggles + sliders → Apply Warp |
| Aging / De-aging (real FFT) | Aging page → preset or custom → Apply Simulation |
| Frequency spectrum (visual) | Frequency page |
| Golden Ratio analysis + overlay | Golden Ratio page → Analyze Face |

## File Structure
```
faceforge2/
├── backend.py        ← Flask API (Python)
├── index.html        ← Frontend
├── css/style.css
├── js/
│   ├── main.js       ← Navigation + API helper
│   ├── upload.js
│   ├── landmarks.js  ← Calls /api/landmarks
│   ├── expression.js ← Calls /api/expression
│   ├── aging.js      ← Calls /api/aging
│   ├── frequency.js  ← Canvas FFT visualization
│   ├── evaluation.js ← Metric placeholders
│   └── golden.js     ← Calls /api/golden_ratio
```

## API Endpoints

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| /api/landmarks | POST | `{image}` | `{landmarks, count, overlay, width, height}` |
| /api/expression | POST | `{image, smile, eyebrow, lips, slim}` | `{result}` |
| /api/aging | POST | `{image, mode, intensity}` | `{result}` |
| /api/golden_ratio | POST | `{image}` | `{scores, overall, overlay, measurements}` |

All images are base64 data URLs.
