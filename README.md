# Audio Analyzer

---


## Preprocessing

---
├─ Trim silence (detect where singing starts/ends)<br>
├─ Noise reduction (spectral subtraction)<br>
└─ Normalize loudness <br>
<details>
<summary>
    <strong id="1">
        Details : 
    </strong>
</summary>

### Trimming silence

---
Trimming silence means removing low-energy (near-zero amplitude) sections from the beginning and end of an audio signal.
<br>In waveform terms:
    
    Silence = very low amplitude values

We detect regions below a threshold (in dB) and cut them out

---

### mel Spectrogram

---
<table>
  <tr>
    <td width="30%">
      <img src="./images/mel_spectrogram.png" alt="Mel Spectrogram" width="100%" height="100%">
    </td>
    <td>
      A <strong>Mel-scaled spectrogram</strong> is a visual representation of audio that maps frequencies 
      to the <strong>Mel scale</strong>, which mimics human hearing by prioritizing lower frequencies.<br><br>
      It is a standard tool in AI audio tasks, converting sound into images 
      (time vs. Mel frequency) and often using <strong>decibel scales</strong> for amplitude.
    </td>
  </tr>
</table>

---

### Noise Reduction

---
Noise reduction is the process of removing unwanted background sounds (fan noise, room hum, mic hiss, street noise) while preserving the vocal signal.
<br>In signal terms:

    Audio = Voice + Noise

Goal → Estimate Noise , and subtract or suppress it.
<br>We do this through a method called **_Spectral Subtraction_**.

*Spectral subtraction : 
    
    Estimate noise from silent portion
    Reduce frequencies below threshold
---

### Normalize Loudness

---
Loudness normalization means adjusting the amplitude of an audio signal to a consistent level across recordings.

After normalization:

    Feature distributions become consistent
    Training becomes more stable

</details>

---

## Segment Matching

---
├─ Load full reference song<br>
├─ Auto-match: Find where user audio fits in reference<br>
    
    - Pass 1: Fast correlation filtering (find top 3)
    - Pass 2: Accurate DTW on top candidates

├─ Calculate match confidence (0-100%)<br>
└─ Fallback: Manual selection if low confidence

<details>
<summary>
    <strong id="1">
        Details : 
    </strong>
</summary>

### Validating user audio

---
It’s a lightweight verification stage before full feature extraction and scoring.<br>
Compute RMS energy: `rms = np.mean(librosa.feature.rms(y=audio))`

If RMS is extremely low → likely silence.

Use pitch detection: `f0, voiced_probs = librosa.pyin(user_audio, fmin=50, fmax=2000)` 

This runs the **_pYIN algorithm_**.

---

### pYIN algorithm

---
pYIN = probabilistic version of YIN (pitch detection algorithm).

It tries to estimate:
    
    f0 → fundamental frequency (pitch) per frame
    voiced_probs → probability that the frame contains voiced sound

If waveform repeats regularly → voiced.

---

### Auto matching

---
Alignment module to match:

    Reference song segment  ↔  User singing segment

**2-Pass Auto Matching Audio Segment → (Fast Correlation + DTW on MFCC)**

**Pass 1**: Fast Correlation (Coarse Alignment)

    Finds where the user singing roughly matches the reference.
    We slide one signal over another and compute similarity.
    It is Very fast and Narrows search window for DTW to perform quickly

**Pass 2**: DTW on MFCC (Precise Alignment)

DTW aligns two sequences even if:

    One is faster
    One is slower
    Notes are stretched
    
It finds the optimal warping path minimizing distance.

In our system:

    MFCC(user)  ↔  MFCC(reference)

<details>
<summary>
    <strong id="1">
        DTW : 
    </strong>
</summary>

Dynamic Time Warping aligns two time sequences of different lengths by:
    
    Stretching
    Compressing
    Warping time axis
It minimizes total distance between two sequences.

[Video Explanation](https://youtu.be/_K1OsqCicBY)
</details>
<details>
<summary>
    <strong id="1">
        MFCC : 
    </strong>
</summary>

Mel-Frequency Cepstral Coefficients are compressed representations of the spectral envelope of audio.<br>
It can:
    
    Compare tonal similarity between user & reference
    Detect pronunciation differences
    Analyze vocal texture stability

[Video Explanation](https://youtu.be/SJo7vPgRlBQ)
</details>

---

### Prepare User audio

---
Prepare audio pair for comparison by handling length mismatches, after ref audio is converted into matched segment.

</details>

---