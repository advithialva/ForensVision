# ForensVision 🛡️

ForensVision is a forensic video analysis tool that detects violence and weapons in video. It combines a Python backend that runs ML models (violence/weapon detectors) with a Next.js frontend for uploading videos and viewing analysis results.

It's designed to assist law enforcement agencies, security professionals, and forensic investigators.

---
## 🚀 Quick Start

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
