# Facial Skin Analysis System

End-to-end AI-assisted facial skin analysis application with a FastAPI backend and React frontend.

## File Structure

```text
Playground/
|-- backend/
|   |-- app/
|   |   |-- api/routes/analyze.py
|   |   |-- core/config.py
|   |   |-- detectors/
|   |   |   |-- dark_spots.py
|   |   |   |-- imperfections.py
|   |   |   `-- oiliness.py
|   |   |-- pipeline/
|   |   |   |-- face_detection.py
|   |   |   |-- pipeline.py
|   |   |   |-- preprocessing.py
|   |   |   `-- skin_segmentation.py
|   |   |-- schemas/analysis.py
|   |   |-- services/analysis_service.py
|   |   |-- utils/image_io.py
|   |   |-- visualization/annotator.py
|   |   `-- main.py
|   `-- requirements.txt
|-- docs/
|   `-- architecture.md
`-- frontend/
    |-- src/
    |   |-- components/
    |   |   |-- AnalysisResultPanel.tsx
    |   |   `-- ImageSourcePanel.tsx
    |   |-- types/analysis.ts
    |   |-- App.tsx
    |   |-- main.tsx
    |   `-- styles.css
    |-- index.html
    |-- package.json
    |-- tsconfig.json
    |-- tsconfig.node.json
    `-- vite.config.ts
```

## Pipeline

`Image Input -> Preprocessing -> Face Detection -> Skin Segmentation -> Multi-Detection Engine -> Visualization -> Response Generation`

The implementation uses lightweight OpenCV and NumPy heuristics for fast local inference:

- Face detection: Haar cascade face detection with a full-image fallback
- Skin segmentation: combined HSV and YCrCb masks with geometric exclusions for eyes and lips
- Pimples / whiteheads / blackheads: blob and intensity-based irregularity detection
- Dark spots: LAB lightness anomaly mask
- Oiliness: brightness and texture fusion score

## Backend Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend runs at `http://127.0.0.1:8000`.

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://127.0.0.1:5173`.

## Production Deployment

This project is now containerized for deployment with Docker Compose.

```bash
docker compose up --build
```

Production URLs:

- Frontend: `http://localhost`
- Backend health: `http://localhost/api/health`

Notes:

- The frontend is served by Nginx.
- Requests to `/api/*` are proxied to the FastAPI backend.
- The frontend uses `VITE_API_BASE_URL` when provided, otherwise it defaults to:
  - dev: `http://127.0.0.1:8000`
  - production: `/api`

## API

`POST /analyze`

Accepts either:

- multipart `file`
- JSON body with `image_base64`

Returns:

- localized detections for pimples, whiteheads, and blackheads
- dark spot area ratio and mask statistics
- oiliness score and category
- annotated image as base64 PNG
