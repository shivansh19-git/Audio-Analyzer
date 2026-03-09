# 🎵 Audio Analyzer 

A sophisticated Flask-based web application that analyzes singing performance by comparing user recordings against reference songs using advanced DSP algorithms.

### Machine Learning Model
**Emotion Classification:** Implemented using Support Vector Machine (SVM) trained on speech features. The model is persisted using pickle for fast loading and real-time predictions on audio segments.<br>
Dataset was retrieved through [MIR](http://mir.dei.uc.pt/downloads.html) - Bi-modal (audio and lyrics) emotion dataset

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Algorithms & Terminology](#algorithms--terminology)

---

## Overview

**Audio Analyzer** is an intelligent singing evaluation system that provides comprehensive analysis across three dimensions:

- **Pitch Accuracy** (50%) - How well you match the melody
- **Rhythm Accuracy** (35%) - How well you stay in sync  
- **Emotion Detection** (15%) - What feeling is conveyed

**Tech Stack:**
- Backend: Flask (Python)
- Frontend: HTML5, CSS3, JavaScript
- Audio Processing: Librosa, NumPy, SciPy
- ML: Scikit-learn (emotion classification)

---

##  Key Features

 **Intelligent Segment Matching** - Auto-finds where your singing starts  
 **Pitch Analysis** - PYIN extraction + DTW alignment  
 **Rhythm Analysis** - Beat tracking + tempo awareness  
 **Emotion Detection** - 5-class ML classifier  
 **Performance Scoring** - Weighted composite score + grading  

---

## 🏗️ Technical Architecture

### Complete Workflow

```
User Uploads Files
    ↓
Audio Preprocessing (15s)
├─ Load at 22050 Hz
├─ Trim silence
└─ Noise reduction
    ↓
Auto-Segment Matching (2:07 first time)
├─ MFCC feature extraction
├─ DTW alignment (2-pass algorithm)
└─ Confidence scoring
    ↓
User Reviews & Adjusts
├─ Visual waveform
├─ Play segment
└─ Fine-tune position
    ↓
Detailed Analysis (16s)
├─ Pitch Analysis
├─ Rhythm Analysis
├─ Emotion Detection
└─ Score Calculation
    ↓
Display Results
├─ Overall score & grade
├─ Individual metrics
└─ Emotional expression
```

## 🔬 Algorithms & Terminology

### 1. MFCC (Mel-Frequency Cepstral Coefficients)

**What it is:** Audio representation mimicking human hearing

**Key characteristics:**
- Compresses audio into ~13 coefficients
- Focuses on perceptually relevant frequencies
- Widely used in speech/music recognition

**Visual:**
```
Audio Signal → FFT → Mel-Scale → Log → DCT → 13 MFCCs
(Raw audio)  (Freq) (Log scale)  (Compression)
```

**Why:** MFCCs capture essential audio characteristics efficiently  
**Watch:** [MFCC Explained](https://www.youtube.com/results?search_query=MFCC+Mel+Frequency+Cepstral+Coefficients+explained)

---

### 2. DTW (Dynamic Time Warping)

**Definition:** Aligns two sequences even if different lengths/speeds

**Problem it solves:**
```
Reference: C D E F G (perfect tempo)
Your singing: C... D... E F... G (slower with pauses)

Without DTW: Misaligned at every point
With DTW: Perfect alignment despite speed difference
```

**Algorithm:**
```
Builds cost matrix comparing all pairs
Finds optimal path minimizing distance
Handles tempo variations seamlessly
```

**When used:** Pitch comparison (tempo-aware alignment)  
**Watch:** [DTW Algorithm Tutorial](https://www.youtube.com/results?search_query=Dynamic+Time+Warping+algorithm)

---

### 3. PYIN (Probabilistic YIN)

**Definition:** Algorithm for extracting fundamental frequency (pitch) from audio

**What it returns:**
- f0 (frequency in Hz)
- Voicing confidence (0-1, higher = more certain)
- Voiced frames (which frames are actually pitched)

**Range in this project:**
- fmin=50Hz (lowest bass note ~G1)
- fmax=300Hz (highest soprano ~F4)

**Advantages over simple YIN:**
- More robust to noise
- Provides confidence scores
- Better handles edge cases

**Formula:**
```
f0_values = librosa.pyin(
    audio,
    fmin=50,   # Lowest frequency bound
    fmax=300,  # Highest frequency bound
    sr=22050   # Sample rate
)
```

**Watch:** [Pitch Detection Methods](https://www.youtube.com/results?search_query=pitch+detection+YIN+algorithm)

---

### 4. Cent (Musical Distance Unit)

**Definition:** Logarithmic unit for pitch interval

**Key relationships:**
- 100 cents = 1 semitone (half-step)
- 1200 cents = 1 octave (12 semitones)
- 1 cent ≈ 0.58% frequency change
- Humans distinguish ~10-20 cents

**This project uses:** ±150 cents tolerance = ~1.5 semitones

**Example:**
```
Target note: C4 (262 Hz)
Your note:   262.5 Hz → ~9 cents sharp ✓ (within tolerance)
Your note:   275 Hz   → ~200 cents sharp ✗ (too far off)
```

**Watch:** [Cents in Music Theory](https://www.youtube.com/results?search_query=musical+cent+interval+semitone)

---

### 5. Chroma Features

**Definition:** 12-pitch representation (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)

**Advantages:**
- Octave-invariant (C2 = C4 = C5)
- Perfect for key detection
- Reveals wrong octave singing

**Usage:**
```python
chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
# Output: (12, num_frames) array
# Each row = energy in one pitch class
```


---

### 6. Tempogram & Beat Tracking

**Definition:** Detects tempo/beat structure of audio

- Finds expected beat times
- Compares with user's actual onsets
- Enables rhythm scoring

**Process:**
```
1. Compute onset strength (energy changes)
2. Build tempogram (tempo vs time)
3. Find peak tempo
4. Estimate beat times using DP tracking
5. Compare user onsets to beat times
```

**Watch:** [Beat Tracking Fundamentals](https://www.youtube.com/results?search_query=beat+tracking+music+information+retrieval)

---

### 7. Spectral Subtraction (Noise Reduction)

**Definition:** Reduces noise by subtracting estimated noise spectrum

**Formula:**
```
Clean = Signal - α × Noise_Estimate
(where α = over-subtraction factor, typically 1.0-2.0)
```

- Improves pitch detection accuracy
- Reduces rhythm analysis errors
- Makes emotion detection more reliable

---

### 8. Feature Scaling & Normalization

 ML models expect features in similar ranges

**Method used:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
# Output: Each feature has mean=0, std=1
```

**Without scaling:**
```
Feature 1: -500 to 500 (large range)
Feature 2: 0.0 to 1.0 (small range)
→ Feature 1 dominates in ML model (WRONG!)

With scaling:
Both features: -3 to 3 (equal influence)
→ ML model weights them fairly
```

---



### Pitch Accuracy

**What it measures:** Note accuracy

**Tolerance:** ±150 cents (~1.5 semitones)

**Tips to improve:**
- Use tuner app during practice
- Sing slower for accuracy
- Focus on jump points
- Sing with reference audio

### Rhythm Accuracy

**Components:**
- **Sync (50%):** How often you're on beat
- **Regularity (50%):** Consistency of timing

**Tips to improve:**
- Practice with metronome
- Tap foot to establish beat
- Start slower, gradually speed up
- Record and compare timing

### Emotion Detected

 Happy 😊, Sad 😢, Neutral 😐, Energetic ⚡, Angry 😠

**Note:** <60% confidence means emotion is unclear

**Tips:**
- Think about emotional intent
- Match song's mood
- Adjust tone and dynamics
- Practice with expression

---

## 🎓 Learning Resources

**Video Tutorials:**
- [MFCC Explained](https://www.youtube.com/results?search_query=MFCC+explained)
- [DTW Algorithm](https://www.youtube.com/results?search_query=Dynamic+Time+Warping)
- [Pitch Detection](https://www.youtube.com/results?search_query=pitch+detection+algorithms)
- [Beat Tracking](https://www.youtube.com/results?search_query=beat+tracking+music)
- [Musical Intervals](https://www.youtube.com/results?search_query=cents+music+theory)

**Documentation:**
- [Librosa Docs](https://librosa.org/)
- [SciPy Audio](https://docs.scipy.org/doc/scipy/reference/)
- [NumPy](https://numpy.org/doc/)
- [Flask](https://flask.palletsprojects.com/)

---

