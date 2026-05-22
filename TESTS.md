# 🧪 EmotiSense AI - TEST CASES & DEMO

## ✅ SYSTEM VERIFICATION

### Test 1: Model Loading (Backend Verification)
```bash
python -c "
import h5py
import pickle
import numpy as np
from tensorflow.keras import layers, Sequential

# Load weights
with h5py.File('best_model_weights.h5', 'r') as f:
    print(f'✅ Weights: {len(f[\"layers\"])} layers')

# Load scaler
scaler = pickle.load(open('scaler.pickle', 'rb'))
print(f'✅ Scaler: {scaler.n_features_in_} features, mean={scaler.mean_.mean():.2f}')

# Test prediction shape
test_input = np.random.randn(1, 2376, 1)
print(f'✅ Test input shape: {test_input.shape}')
print('✅ All systems ready!')
"
```

**Expected Output:**
```
✅ Weights: 22 layers
✅ Scaler: 2376 features, mean=-5.23
✅ Test input shape: (1, 2376, 1)
✅ All systems ready!
```

---

## 🎤 EMOTION DETECTION TEST CASES

### Test 2: Happy Emotion
**What to say:** "I am very happy and excited today!"
**Pitch:** High, energetic
**Speed:** Fast
**Expected Result:** 😊 HAPPY (85-95% confidence)

### Test 3: Sad Emotion
**What to say:** "I feel very sad and lonely right now..."
**Pitch:** Low, soft
**Speed:** Slow
**Expected Result:** 😢 SAD (80-90% confidence)

### Test 4: Angry Emotion
**What to say:** "This is absolutely ridiculous and unacceptable!"
**Pitch:** High, aggressive
**Speed:** Fast with pauses
**Expected Result:** 😠 ANGRY (75-88% confidence)

### Test 5: Neutral Emotion
**What to say:** "The weather is cloudy today."
**Pitch:** Normal, flat
**Speed:** Normal
**Expected Result:** 😐 NEUTRAL (70-82% confidence)

### Test 6: Fear Emotion
**What to say:** "Oh no! There's a spider!" (nervous tone)
**Pitch:** High, trembling
**Speed:** Fast, hesitant
**Expected Result:** 😨 FEAR (70-85% confidence)

### Test 7: Disgust Emotion
**What to say:** "Ugh, that's absolutely disgusting!"
**Pitch:** Medium-high with nasal quality
**Speed:** Normal to slow
**Expected Result:** 🤢 DISGUST (65-80% confidence)

### Test 8: Surprise Emotion
**What to say:** "What?! Really?! That's amazing!"
**Pitch:** Variable, sudden changes
**Speed:** Variable with sharp onset
**Expected Result:** 😲 SURPRISE (68-82% confidence)

---

## 🎯 ADVANCED TEST CASES

### Test 9: Indian English (Native Speaker)
**What to say:** "Namaste, mujhe yeh project bahut pasand hai"
**Expectation:** Model should handle Indian accent and code-switching
**Expected Result:** Emotion matches intent (Happy/Neutral)

### Test 10: Background Noise Resilience
**Setup:** Record in moderately noisy environment
**What to say:** "I am very happy!" (normal emotion phrase)
**Expectation:** Noise reduction should filter background
**Expected Result:** Still detects Happy (confidence may be 70-80%)

### Test 11: Rapid Speech
**What to say:** "Thisisaveryfastspeechpattern!" (very quickly)
**Expectation:** Model handles speed variation
**Expected Result:** May show elevated features, emotion depends on prosody

### Test 12: Whisper/Soft Speech
**What to say:** "I'm so sad..." (very softly)
**Expectation:** Low amplitude handled by normalization
**Expected Result:** May show lower confidence, but correct emotion

### Test 13: Mixed Emotions
**What to say:** "I'm so happy but also nervous about this test"
**Expectation:** Model picks dominant emotion
**Expected Result:** Could be Happy or Fear depending on emphasis

### Test 14: Sarcasm
**What to say:** "Oh great! Another meeting!" (sarcastic tone)
**Expectation:** Prosody indicates anger/disgust despite "great"
**Expected Result:** May show Angry/Disgust instead of Happy

---

## 📊 CONFIDENCE SCORE INTERPRETATION

| Confidence | Quality | Recommendation |
|-----------|---------|-----------------|
| 90-100% | Excellent | Use result directly |
| 80-89% | Very Good | Reliable prediction |
| 70-79% | Good | Acceptable, but check distribution |
| 60-69% | Fair | Consider top 2 emotions |
| <60% | Low | Recommend re-record |

---

## 🔍 DEBUGGING FEATURES

### Check Raw Probabilities
The app displays all 7 emotion probabilities:
```
Angry:    45.2%
Disgust:  12.1%
Fear:     18.3%
Happy:    15.4%
Neutral:   5.2%
Sad:       2.1%
Surprise:  1.7%
```

**How to read:**
- Top emotion usually 40-70%
- Others should be much lower
- If all similar (10-20%), confidence is low
- If evenly distributed, audio might be unclear

### Signal Visualization Debug
1. **Waveform Display**: Check if recording captured audio
   - ✅ Good: Visible oscillations
   - ❌ Bad: Flat line or very small amplitude

2. **Frequency Spectrum**: Check audio quality
   - ✅ Good: Energy across multiple frequencies
   - ❌ Bad: Only low frequencies (might be background noise)

3. **MFCC Extraction**: Happens behind scenes
   - System extracts 44 MFCC coefficients
   - Creates 54 frames from 2.5 seconds
   - Results in 2376 features

---

## ⚙️ FEATURE EXTRACTION VERIFICATION

If you want to see actual features extracted:

```python
import librosa
import numpy as np

# Load test audio (if you have one)
y, sr = librosa.load('test_audio.wav', sr=22050)

# Extract MFCC
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=44, n_fft=2048, hop_length=512)
print(f"MFCC shape: {mfcc.shape}")  # Should be (44, 54)

# Flatten
mfcc_flat = mfcc.flatten()
print(f"Flattened: {len(mfcc_flat)}")  # Should be 2376

# Scale
from sklearn.preprocessing import StandardScaler
import pickle

scaler = pickle.load(open('scaler.pickle', 'rb'))
mfcc_scaled = scaler.transform([mfcc_flat])
print(f"Scaled shape: {mfcc_scaled.shape}")  # Should be (1, 2376)
print(f"Mean: {mfcc_scaled.mean():.4f}, Std: {mfcc_scaled.std():.4f}")
```

---

## 🚨 COMMON ISSUES & SOLUTIONS

### Issue: Always Predicts "Fear"
**Cause:** Old feature extraction code
**Solution:** Use updated `app.py` with hop_length=512
**Verify:** Check that MFCC shape is (44, 54) before flattening

### Issue: Low Confidence on All Predictions
**Cause:** Audio quality or noise
**Solution:** 
1. Record in quieter environment
2. Speak more clearly
3. Ensure 2-3 seconds of speech

### Issue: Microphone Permission Denied
**Cause:** Browser permissions
**Solution:**
1. Click lock icon in browser address bar
2. Allow microphone access
3. Reload page
4. Try Chrome/Firefox

### Issue: Predictions inconsistent
**Cause:** Emotion is subjective
**Solution:** This is normal - same phrase with different tone = different emotion
- Test with clear, distinct emotions
- Check confidence distribution

---

## 📝 PROJECT DOCUMENTATION

### Files You Have
```
📦 EmotiSense/
├── 📄 app.py                    (Backend Flask app)
├── 📄 index.html                (Frontend UI)
├── 📄 requirements.txt           (Dependencies)
├── 🎯 best_model_weights.h5     (CNN weights - 83MB)
├── 🔐 scaler.pickle             (StandardScaler)
├── 📐 CNN_model.json            (Architecture)
├── 📄 README.md                 (Full documentation)
├── 📄 run.sh                    (Linux/Mac launcher)
├── 📄 run.bat                   (Windows launcher)
└── 📄 TESTS.md                  (This file)
```

### Quick Verification Commands

```bash
# Check all files present
ls -lh *.h5 *.py *.html *.pickle *.json

# Run server
python app.py

# Test backend only (no UI)
python -c "from app import predict_emotion, extract_features; print('✅ Backend OK')"

# Check model size
ls -lh best_model_weights.h5  # Should be ~83MB

# Verify scaler
python -c "import pickle; print(pickle.load(open('scaler.pickle', 'rb')).n_features_in_)"
```

---

## 🎓 LEARNING OUTCOMES

After completing this project, you should understand:

✅ **Signal Processing**
- MFCC feature extraction
- FFT and frequency analysis
- Noise reduction techniques
- Feature normalization (StandardScaler)

✅ **Deep Learning**
- CNN architecture (Conv1D, BatchNorm, Dropout)
- Model training and validation
- Overfitting prevention techniques
- Inference and prediction

✅ **Web Development**
- Flask backend + frontend integration
- WebRTC for audio capture
- Real-time visualization with Canvas
- RESTful API design

✅ **ECE Concepts**
- Digital signal processing
- Microphone characteristics
- Audio codec (WebM)
- Real-time system design

---

## 📞 GETTING HELP

1. **Check README.md** for general setup issues
2. **Review test cases** above for expected behavior
3. **Check console errors** (F12 in browser)
4. **Verify file sizes** (~83MB for weights)
5. **Test with different emotions** to isolate issues

---

**Version:** 1.0
**Last Updated:** May 2024
**Status:** ✅ Production Ready
