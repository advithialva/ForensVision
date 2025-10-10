# ForensVision

ForensVision is a multi-part project for violence and weapon detection in video. It contains a Python backend (FastAPI/Flask style) and a Next.js frontend.

Structure
- backend/: Python API, models, and utilities
- frontend/: Next.js frontend app
- models/: stored ML models used by the backend

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

Notes
- This repository contains model files (.pt, .h5) and some sample data. Large files are ignored by `.gitignore` if placed under `data/`.
