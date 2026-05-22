# 🎤 EMOTISENSE AI
## Indian Speech Emotion Recognition System

Real-time speech emotion recognition system using MFCC-based feature extraction, CNN classification, transcript-assisted emotion enhancement, and interactive signal visualization.

---

## 📌 Project Overview

EMOTISENSE AI is a web-based Speech Emotion Recognition (SER) system developed for real-time audio analysis.

The system captures speech through the microphone, extracts acoustic features using MFCC, processes them through a trained CNN model, and displays emotion predictions with interactive analytics.

### Supported Emotions:

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

---

## 🚀 Features

### 🎙 Real-Time Audio Capture
- Microphone recording
- Start / Stop recording controls
- Real-time processing
- Speech capture system

### 🧠 AI Emotion Analysis
- MFCC feature extraction
- CNN-based emotion classification
- Transcript-assisted emotion enhancement
- Confidence score generation
- Emotion probability distribution

### 📊 Signal Processing & ECE Features
- Live frequency spectrum
- Spectrogram visualization
- Signal energy analysis
- Noise reduction
- DSP analytics

### 📈 Analytics Dashboard
- Analysis history
- Session statistics
- Timestamps
- Confidence metrics
- Emotion trends

---

## 🧠 Technical Pipeline

```text
Microphone Input
      ↓
Audio Preprocessing
      ↓
Noise Reduction
      ↓
MFCC Extraction
      ↓
Feature Scaling
      ↓
CNN Prediction
      ↓
Transcript Context Enhancement
      ↓
Final Emotion Prediction
```

---

## 🛠 Technologies Used

### Frontend
- HTML
- CSS
- JavaScript
- Web Audio API

### Backend
- Flask
- Python

### AI / ML
- TensorFlow
- CNN
- Librosa
- NumPy
- Scikit-Learn

### Signal Processing
- MFCC
- FFT
- Spectrogram Analysis

---

## 📁 Project Structure

```text
EmotiSense-Ai/

├── app.py
├── index.html
├── best_model_weights.h5
├── CNN_model.json
├── encoder.pickle
├── scaler.pickle
├── requirements.txt
├── README.md
```

---

## ⚙ Installation

Clone repository:

```bash
git clone https://github.com/Mahak5123/EmotiSense-Ai.git
```

Move to project directory:

```bash
cd EmotiSense-Ai
```

Install required libraries:

```bash
pip install -r requirements.txt
```

Run application:

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

## 📊 Emotion Classes

| Emotion | Emoji |
|----------|--------|
| Angry | 😠 |
| Disgust | 🤢 |
| Fear | 😨 |
| Happy | 😊 |
| Neutral | 😐 |
| Sad | 😢 |
| Surprise | 😲 |

---

## 🎓 ECE Concepts Used

### Digital Signal Processing
- MFCC extraction
- FFT analysis
- Spectrogram generation
- Audio preprocessing

### Artificial Intelligence
- CNN architecture
- Classification models
- Feature normalization

### Web Technologies
- Real-time microphone capture
- Interactive visual dashboard
- Dynamic UI updates

---

## 📚 Research Reference

Inspired by research paper:

**"IoT-Enabled WBAN and Machine Learning for Speech Emotion Recognition in Patients"**

Published in:

**Sensors 2023**

---

## 🔮 Future Improvements

- Multilingual emotion recognition
- Indian speech dataset fine-tuning
- Mobile application deployment
- Edge device implementation
- Healthcare emotion monitoring

---

## 👩‍💻 Developer

Mahak Salecha

---

⭐ If you found this project useful, consider starring the repository.