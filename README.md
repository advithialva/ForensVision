# ForensVision: AI-powered Forensic Video Analysis Tool
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)  [![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)  [![Next.js](https://img.shields.io/badge/Next.js-14.0+-black.svg)](https://nextjs.org/)  [![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)  [![MoBiLSTM](https://img.shields.io/badge/MoBiLSTM-Violence_Detection-purple.svg)]()


<img width="1340" height="290" alt="Image" src="https://github.com/user-attachments/assets/4ac4ea14-d752-42a4-9ef0-e814fdd7c708" />

ForensVision is an intelligent video analytics platform that aids forensic investigations by automatically detecting violent activities and weapons in uploaded surveillance footage. It features a Python FastAPI backend for efficient model inference and a modern Next.js frontend for seamless video uploads and intuitive visualization of analysis results.

It is designed to assist law enforcement agencies, security professionals, and forensic investigators.


---
## ✨ Key Features

- 🎥 Automated analysis of uploaded surveillance videos for violence and weapon detection
- 🔫 Custom-trained YOLOv8 model for identifying weapons with high precision
- 🧠 Hybrid MoBiLSTM (MobileNet + BiLSTM) architecture for violence recognition
- ⚡ Fast inference powered by a Python FastAPI backend
- 🌐 Responsive Next.js frontend for video upload and result visualization

---
## 💻 Tech Stack

| **Category**           | **Technologies Used**                                     |
| ---------------------- | --------------------------------------------------------- |
| **Backend**            | FastAPI · Uvicorn · Python                                |
| **AI / ML**            | TensorFlow · Keras · PyTorch · YOLOv8                     |
| **Computer Vision**    | OpenCV                                                    |
| **Data Preprocessing** | NumPy · Pandas · Scikit-learn                             |
| **Frontend**           | Next.js · React · TypeScript                              |
| **Styling / UI**       | Tailwind CSS · Framer Motion                              |
| **Tools**              | Git/GitHub · Jupyter Notebook                             |
| **Models**             | MoBiLSTM (Violence Detection) · YOLOv8 (Weapon Detection) |

---

## 📁 Project Structure

```
ForensVision
├─ backend
│  ├─ main.py                 # Entry point for the backend server
│  ├─ config.py               # Configuration
│  ├─ models/                 # ML model scripts
│  ├─ utils/                  # Helper functions
│  ├─ requirements.txt
│  └─ yolov8n.pt
├─ frontend
│  ├─ src/
│  │  ├─ app/                 # Main app layout and styling files
│  │  ├─ components/          # Reusable UI components
│  │  └─ services/            # API service layer for backend communication
│  ├─ package.json
│  ├─ tailwind.config.ts
│  └─ tsconfig.json
├─ models                     # Trained deep learning models
│  ├─ violence_detection/
│  └─ weapon_detection/
└─ README.md                  # Project documentation
```

---

## 🚀 Installation

### 🖥️ Backend Setup

1. **Clone and navigate to backend**
   ```bash
   git clone https://github.com/advithialva/ForensVision.git
   cd ForensVision/backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate

   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the backend server**
   ```bash
   python main.py
   ```
   Server will start at `http://localhost:8000`


### 🌐 Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd ../frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```
   Frontend will be available at `http://localhost:3000`

---
## Environment Variables
```bash
# Backend Configuration
CORS_ORIGINS=http://localhost:3000
MAX_FILE_SIZE=500MB
MODEL_DEVICE=auto
```
```bash
# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
```
---
## 🤖 AI Models

### Violence Detection Model
- **Architecture**: MoBiLSTM (Mobile + Bidirectional LSTM)
- **Components**: 
  - LSTM for temporal analysis
  - YOLO for object detection
  - ResNet-50 for visual feature extraction
- **Input**: Video frames with person tracking
- **Output**: Violence probability with component scores

### Weapon Detection Model
- **Architecture**: YOLOv8n (Nano variant for speed)
- **Training**: Custom dataset with multiple weapon types
  
### Model Paths
Models are automatically loaded from the `models/` directory:
- Violence: `models/violence_detection/MoBiLSTM_violence_detection_model.h5`
- Weapons: `models/weapon_detection/weapon_detection.pt`

---
## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---
## 👥 Authors

- **Aaron Fernandes** - [GitHub](https://github.com/aaronfernandes21)
- **Advithi Alva** - [GitHub](https://github.com/advithialva)
- **Pratham R Shetty** - [GitHub](https://github.com/Prathamshettyy)
- **Ryshel Jasmi DSouza** - [GitHub](https://github.com/ryshel-jasmi)
