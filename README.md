# ASL Sign Language Translator

Real-time American Sign Language recognition web application.

## Pipeline
Webcam → RTMPose (133 keypoints) → Transformer Classifier (100 signs) → Gemini 2.5 Flash (grammar) → React UI

## Stack
- **Pose Estimation:** RTMPose-L Whole-Body
- **Classifier:** Custom Transformer (~2.5M params), trained on WLASL-100
- **LLM:** Gemini 2.5 Flash (free tier)
- **Backend:** FastAPI + WebSocket
- **Frontend:** React + Tailwind + Framer Motion

## Quick Start

### 1. Clone & install
```bash
pip install -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

### 3. Download & preprocess dataset
```bash
python data/download_wlasl.py
python data/preprocess.py
```

### 4. Train (Google Colab recommended)
Open `training/colab_notebook.ipynb` and run all cells.

### 5. Run backend
```bash
uvicorn backend.main:app --reload --port 8000
```

### 6. Run frontend
```bash
cd frontend
npm install
npm run dev
```

## Project Structure
See `docs/architecture.md` for the complete technical blueprint.

## Performance Targets
| Metric | Target |
|---|---|
| Top-1 Accuracy | 55–65% |
| Top-5 Accuracy | 80–90% |
| Inference Latency | <100ms |
| Pipeline FPS | 15–25 FPS |
