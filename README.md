# ForensVision 🛡️

ForensVision is a forensic video analysis tool that detects violence and weapons in video. It combines a Python backend that runs ML models (violence/weapon detectors) with a Next.js frontend for uploading videos and viewing analysis results.

Quick start

Backend
1. Create a Python virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

2. Run the backend (from `backend/`):

```bash
python3 main.py
```

Frontend
1. Install dependencies and run dev server (from `frontend/`):

```bash
cd frontend
npm install
npm run dev
```
