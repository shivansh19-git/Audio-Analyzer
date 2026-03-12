#!/usr/bin/env python
# coding: utf-8

# # Audio Preprocessing
# ---
# Handles: noise reduction, silence trimming, normalization, and audio cleanup.

# ## Importing Audio
# ---
# Load audio file & check if they are good to go.

import librosa
import librosa.feature
import librosa.beat
import librosa.onset
import os
import numpy as np
import gc

def load_and_validate_audio_pair(user_path, ref_path, sr=22050):

    # Step 1: Check files exist
    print("\n[1/4] Checking files...")

    if not os.path.exists(user_path):
        print(f"❌ User audio not found: {user_path}")
        return None, None, sr, False, "User audio file not found"

    if not os.path.exists(ref_path):
        print(f"❌ Reference audio not found: {ref_path}")
        return None, None, sr, False, "Reference audio file not found"

    print("✓ Both files found")

    # Step 2: Load audio
    print("\n[2/4] Loading audio...")

    try:
        user_audio, sr = librosa.load(user_path, sr=sr)
        print(f"✓ User audio: {len(user_audio)/sr:.2f}s ({len(user_audio)} samples)")
    except Exception as e:
        print(f"❌ Failed to load user audio: {e}")
        return None, None, sr, False, f"Could not load user audio: {e}"

    try:
        ref_audio, sr = librosa.load(ref_path, sr=sr)
        print(f"✓ Reference audio: {len(ref_audio)/sr:.2f}s ({len(ref_audio)} samples)")
    except Exception as e:
        print(f"❌ Failed to load reference audio: {e}")
        return None, None, sr, False, f"Could not load reference audio: {e}"

    # Step 3: Check duration
    print("\n[3/4] Checking duration...")

    user_duration = len(user_audio) / sr
    ref_duration = len(ref_audio) / sr
    min_duration = 2.0

    if user_duration < min_duration:
        error = f"User audio too short: {user_duration:.2f}s (min: {min_duration}s)"
        print(f"❌ {error}")
        return None, None, sr, False, error

    if ref_duration < min_duration:
        error = f"Reference audio too short: {ref_duration:.2f}s (min: {min_duration}s)"
        print(f"❌ {error}")
        return None, None, sr, False, error

    print(f"✓ Both audios long enough")

    # Step 4: Check audio content (not silent)
    print("\n[4/4] Checking audio content...")

    user_rms = np.sqrt(np.mean(user_audio ** 2))
    ref_rms = np.sqrt(np.mean(ref_audio ** 2))

    if user_rms < 0.001:
        error = f"User audio is too quiet (RMS: {user_rms:.6f})"
        print(f"❌ {error}")
        return None, None, sr, False, error

    if ref_rms < 0.001:
        error = f"Reference audio is too quiet (RMS: {ref_rms:.6f})"
        print(f"❌ {error}")
        return None, None, sr, False, error

    user_peak = np.max(np.abs(user_audio))
    ref_peak = np.max(np.abs(ref_audio))

    print(f"✓ User: RMS={user_rms:.4f}, Peak={user_peak:.4f}")
    print(f"✓ Reference: RMS={ref_rms:.4f}, Peak={ref_peak:.4f}")

    print("\n✓ ALL VALIDATION CHECKS PASSED!")

    return user_audio, ref_audio, sr, True, ""


# ## Silence Trimming

def trim_silence(audio, sr, top_db=25):
    print("Trimming silence...")

    trimmed_audio, index = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=2048,
        hop_length=512
    )

    print("✓ Trimmed silence!")
    print(f"  Original: {len(audio)/sr:.2f}s")
    print(f"  Trimmed:  {len(trimmed_audio)/sr:.2f}s")
    print(f"  Removed:  {(len(audio)-len(trimmed_audio))/sr:.2f}s of silence")

    return trimmed_audio


# ## Noise Reduction

def reduce_noise_spectral_subtraction(audio, sr, noise_duration=1.0):
    print("Reducing noise...")

    S = librosa.feature.melspectrogram(y=audio, sr=sr)

    noise_frame_count = int(sr * noise_duration / 512)
    noise_sample = S[:, :noise_frame_count]
    noise_profile = np.mean(noise_sample, axis=1, keepdims=True)

    S_reduced = S - 2.0 * noise_profile
    S_reduced = np.maximum(S_reduced, 0)

    audio_reduced = librosa.feature.inverse.mel_to_audio(S_reduced, sr=sr)

    max_original = np.max(np.abs(audio))
    max_reduced = np.max(np.abs(audio_reduced))

    if max_reduced > 1e-8:
        audio_reduced *= (max_original / max_reduced)

    print(f"✓ Noise reduction complete!")

    return audio_reduced


# ## Audio Normalization

def normalize_loudness(audio, target_level_db=-3.0):
    rms = np.sqrt(np.mean(audio ** 2))

    if rms == 0:
        return audio

    current_level_db = 20 * np.log10(rms + 1e-10)
    gain_db = target_level_db - current_level_db
    gain_linear = 10 ** (gain_db / 20)

    audio_normalized = audio * gain_linear

    max_val = np.max(np.abs(audio_normalized))
    if max_val > 1.0:
        audio_normalized = audio_normalized / max_val

    print(f"Normalized audio: {current_level_db:.1f} dB -> {target_level_db:.1f} dB")

    return audio_normalized


# # Segment Matching

def validate_pitched_content(audio):
    try:
        f0, voiced_flags, voiced_probs = librosa.pyin(audio, fmin=50, fmax=2000)

        voiced_frames = np.sum(voiced_probs > 0.1)
        total_frames = len(f0)
        voiced_percentage = 100 * voiced_frames / total_frames
        has_content = voiced_percentage > 5

        return has_content, voiced_percentage

    except Exception as e:
        print(f"Warning: Could not analyze pitched content: {e}")
        return False, 0


def find_best_matching_segment(user_audio, ref_audio, sr, window_size=None):
    import numpy as np
    import librosa
    from librosa.sequence import dtw

    user_audio = librosa.util.normalize(user_audio)
    ref_audio = librosa.util.normalize(ref_audio)

    if window_size is None:
        window_size = int(len(user_audio) * 1.1)

    print(f"\nUser audio: {len(user_audio)/sr:.1f}s, Window: {window_size/sr:.1f}s")
    print(f"Reference: {len(ref_audio)/sr:.1f}s")
    print("="*60)

    # PASS 1: Energy-based coarse search
    print("\nPASS 1: Energy-based coarse search...")
    user_energy = librosa.feature.rms(y=user_audio)[0]
    candidates = []
    step_samples = int(sr * 1)

    for start in range(0, len(ref_audio) - window_size, step_samples):
        ref_segment = ref_audio[start:start + window_size]
        ref_energy = librosa.feature.rms(y=ref_segment)[0]

        if len(ref_energy) > len(user_energy):
            ref_energy = ref_energy[:len(user_energy)]
        elif len(ref_energy) < len(user_energy):
            continue

        user_energy_norm = (user_energy - user_energy.min()) / (user_energy.max() - user_energy.min() + 1e-7)
        ref_energy_norm = (ref_energy - ref_energy.min()) / (ref_energy.max() - ref_energy.min() + 1e-7)
        correlation = np.corrcoef(user_energy_norm, ref_energy_norm)[0, 1]

        if not np.isnan(correlation):
            energy_sim = max(0, (correlation + 1) / 2)
            if energy_sim > 0.3:
                candidates.append((energy_sim, start))

    candidates.sort(reverse=True)
    print(f"Found {len(candidates)} candidates")

    if len(candidates) == 0:
        print("Warning: No candidates found")
        return ref_audio[:window_size], 0.0, 0.0

    print(f"Top 3: {candidates[:3]}")

    # PASS 2: MFCC + DTW on top candidates
    print("\nPASS 2: MFCC + DTW matching on top candidates...")

    mfcc_user = librosa.feature.mfcc(y=user_audio, sr=sr, n_mfcc=13)

    best_confidence = 0
    best_start = 0
    best_timestamp = 0

    for rank, (energy_sim, start) in enumerate(candidates[:5]):
        timestamp = start / sr
        ref_segment = ref_audio[start:start + window_size]

        mfcc_ref = librosa.feature.mfcc(y=ref_segment, sr=sr, n_mfcc=13)

        if mfcc_user.shape[1] == 0 or mfcc_ref.shape[1] == 0:
            continue

        min_steps = min(mfcc_user.shape[1], mfcc_ref.shape[1])
        mfcc_user_m = mfcc_user[:, :min_steps]
        mfcc_ref_m = mfcc_ref[:, :min_steps]

        spectral_distance = np.mean(np.abs(mfcc_user_m - mfcc_ref_m))
        mfcc_sim = 100 * np.exp(-spectral_distance)
        mfcc_sim = float(np.clip(mfcc_sim, 0, 100))

        try:
            D, _ = dtw(mfcc_user_m, mfcc_ref_m, metric='euclidean')
            dtw_cost = D[-1, -1] / (mfcc_user_m.shape[1] + mfcc_ref_m.shape[1])
            dtw_sim = 100 * np.exp(-dtw_cost / 5)
            dtw_sim = float(np.clip(dtw_sim, 0, 100))
        except Exception as e:
            print(f"  @ {timestamp:.1f}s: DTW error - {e}")
            dtw_sim = mfcc_sim

        confidence = energy_sim * 100 * 0.60 + (mfcc_sim + dtw_sim) / 2 * 0.40

        print(f"  @ {timestamp:.1f}s: Energy={energy_sim*100:5.1f}% MFCC={mfcc_sim:5.1f}% DTW={dtw_sim:5.1f}% → {confidence:5.1f}%")

        if confidence > best_confidence:
            best_confidence = confidence
            best_start = start
            best_timestamp = timestamp

    if best_start == 0 and best_confidence == 0:
        print("No good matches found, using best energy match")
        best_start = candidates[0][1]
        best_timestamp = best_start / sr
        best_confidence = candidates[0][0] * 100

    matched_segment = ref_audio[best_start:best_start + window_size]
    confidence = float(np.clip(best_confidence, 0, 100))

    print("\n" + "="*60)
    print(f"✓ BEST MATCH: {best_timestamp:.1f}s ({confidence:.1f}% confidence)")
    print("="*60 + "\n")

    return matched_segment, best_timestamp, confidence


def prepare_for_comparison(user_audio, ref_audio, sr):
    user_duration = len(user_audio) / sr
    ref_duration = len(ref_audio) / sr

    print(f"Audio lengths: user={user_duration:.1f}s, ref={ref_duration:.1f}s")

    if user_duration < 0.5 * ref_duration:
        target_length = int(len(user_audio) * 1.2)
        ref_audio = ref_audio[:target_length]
        print(f"Trimmed reference to {len(ref_audio)/sr:.1f}s to match user")

    elif user_duration > 1.5 * ref_duration:
        print(f"Warning: User provided {user_duration:.1f}s, reference is {ref_duration:.1f}s")

    return user_audio, ref_audio


# # Tempo

def detect_tempo(audio, sr):
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    return float(tempo)


def estimate_tempo_ratio(user_audio, ref_audio, sr):
    user_tempo = detect_tempo(user_audio, sr)
    ref_tempo = detect_tempo(ref_audio, sr)
    ratio = user_tempo / ref_tempo

    print(f"\n[TEMPO DETECTION]")
    print(f"  User tempo: {user_tempo:.1f} BPM")
    print(f"  Ref tempo:  {ref_tempo:.1f} BPM")
    print(f"  Ratio: {ratio:.3f}x")

    return ratio, user_tempo, ref_tempo


def warp_f0_to_reference_tempo(f0_user, tempo_ratio):
    if tempo_ratio == 1.0:
        return f0_user

    n_frames = len(f0_user)
    old_indices = np.arange(n_frames)
    new_indices = np.arange(n_frames) * tempo_ratio
    new_indices_clipped = np.clip(new_indices, 0, n_frames - 1)
    f0_warped = np.interp(new_indices_clipped, old_indices, f0_user, left=f0_user[0], right=f0_user[-1])

    return f0_warped


def align_f0_sequences_dtw(f0_user, f0_ref):
    from librosa.sequence import dtw

    user_seq = f0_user.reshape(1, -1)
    ref_seq = f0_ref.reshape(1, -1)

    D, wp = dtw(user_seq, ref_seq, metric='euclidean', backtrack=True)

    user_idx = wp[:, 0]
    ref_idx = wp[:, 1]

    f0_user_aligned = f0_user[user_idx]
    f0_ref_aligned = f0_ref[ref_idx]

    return f0_user_aligned, f0_ref_aligned


# # Feature Extraction - Pitch

PITCH_CLASS_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def extract_pitch_contour(audio, sr, fmin=50, fmax=2000):
    print(f"Extracting pitch contour (fmin={fmin}, fmax={fmax})...")

    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, fmin=fmin, fmax=fmax, frame_length=2048
    )

    valid_frames = np.sum(~np.isnan(f0))
    voiced_frames = np.sum(voiced_probs > 0.1)

    print(f"Extracted {len(f0)} frames ({len(f0) * 512 / sr:.2f}s)")
    print(f"Valid frames: {valid_frames}, Voiced frames: {voiced_frames}")

    return f0, voiced_probs


def hz_to_pitch_class(hz):
    if hz is None or hz <= 0:
        return None
    midi = 69 + 12 * np.log2(hz / 440.0)
    pitch_class = int(round(midi)) % 12
    return pitch_class


def detect_key_from_pitch(f0, voiced_probs):
    valid_frames = (voiced_probs > 0.1) & (f0 > 0) & ~np.isnan(f0) & ~np.isinf(f0)
    f0_voiced = f0[valid_frames]

    if len(f0_voiced) < 5:
        return None

    median_pitch = np.median(f0_voiced)
    print(f"  Raw median pitch: {median_pitch:.1f} Hz")

    candidates = [
        (median_pitch, "original"),
        (median_pitch * 2, "octave +1"),
        (median_pitch * 3, "octave +1.5"),
        (median_pitch / 2, "octave -1"),
    ]

    best_pitch = median_pitch
    best_quality = 0

    for test_pitch, label in candidates:
        if not (80 <= test_pitch <= 300):
            continue

        pitch_class = hz_to_pitch_class(test_pitch)
        cents_diff = 1200 * np.log2(f0_voiced / test_pitch)
        within_3_semitones = np.sum(np.abs(cents_diff) < 300) / len(f0_voiced)

        print(f"    {label:15} {test_pitch:6.1f} Hz ({PITCH_CLASS_NAMES[pitch_class]:2}) - {within_3_semitones*100:.0f}% frames match")

        if within_3_semitones > best_quality:
            best_quality = within_3_semitones
            best_pitch = test_pitch

    final_pitch_class = hz_to_pitch_class(best_pitch)
    print(f"  ✓ Selected: {best_pitch:.1f} Hz ({PITCH_CLASS_NAMES[final_pitch_class]})")

    return final_pitch_class


def handle_key_shift(f0_user, voiced_user, f0_ref, voiced_ref):
    print("\n[KEY SHIFT DETECTION]")

    print("Detecting user key:")
    key_user = detect_key_from_pitch(f0_user, voiced_user)

    print("Detecting reference key:")
    key_ref = detect_key_from_pitch(f0_ref, voiced_user)

    if key_user is None or key_ref is None:
        print("  Could not detect keys, skipping key shift")
        return f0_user, 0

    semitone_offset = (key_user - key_ref) % 12

    if semitone_offset > 6:
        semitone_offset -= 12

    print(f"\nKey offset: {semitone_offset:+d} semitones ({PITCH_CLASS_NAMES[key_user]} → {PITCH_CLASS_NAMES[key_ref]})")

    if semitone_offset != 0:
        shift_factor = 2 ** (semitone_offset / 12.0)
        f0_user_corrected = f0_user / shift_factor
        print(f"✓ Applying transposition\n")
        return f0_user_corrected, semitone_offset

    return f0_user, 0


def compute_pitch_accuracy(user_audio, ref_audio, sr, tolerance_cents=150.0):
    print("\n" + "="*60)
    print("PITCH ACCURACY CALCULATION (TEMPO-AWARE)")
    print("="*60)

    try:
        if user_audio is None or len(user_audio) == 0 or ref_audio is None or len(ref_audio) == 0:
            return 0.0

        fmax = min(300, int(sr // 2) - 50)
        fmin = 50

        print("\n[EXTRACTING PITCH]")
        f0_user, _, voiced_probs_user = librosa.pyin(
            user_audio, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048
        )
        f0_ref, _, voiced_probs_ref = librosa.pyin(
            ref_audio, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048
        )

        if len(f0_user) == 0 or len(f0_ref) == 0:
            return 0.0

        print(f"  User f0: {len(f0_user)} frames")
        print(f"  Ref f0:  {len(f0_ref)} frames")

        tempo_ratio, user_tempo, ref_tempo = estimate_tempo_ratio(user_audio, ref_audio, sr)

        if abs(tempo_ratio - 1.0) > 0.05:
            print(f"\n[TEMPO CORRECTION]")
            print(f"  Warping user f0 by {tempo_ratio:.3f}x")
            f0_user = warp_f0_to_reference_tempo(f0_user, tempo_ratio)

        valid_user = (voiced_probs_user > 0.1) & (f0_user > 0) & ~np.isnan(f0_user) & ~np.isinf(f0_user)
        valid_ref = (voiced_probs_ref > 0.1) & (f0_ref > 0) & ~np.isnan(f0_ref) & ~np.isinf(f0_ref)

        f0_user_valid = f0_user[valid_user]
        f0_ref_valid = f0_ref[valid_ref]

        if len(f0_user_valid) < 5 or len(f0_ref_valid) < 5:
            print("Not enough voiced frames for DTW alignment")
            return 0.0

        print(f"\n[DTW ALIGNMENT]")
        f0_user_aligned, f0_ref_aligned = align_f0_sequences_dtw(f0_user_valid, f0_ref_valid)
        print(f"  Aligned: {len(f0_user_aligned)} frames")

        try:
            voiced_user_aligned = np.ones(len(f0_user_aligned), dtype=bool)
            voiced_ref_aligned = np.ones(len(f0_ref_aligned), dtype=bool)
            f0_user_aligned, semitone_offset = handle_key_shift(
                f0_user_aligned, voiced_user_aligned, f0_ref_aligned, voiced_ref_aligned
            )
        except Exception as e:
            print(f"  Warning: Key shift detection failed: {e}")

        print(f"\n[COMPUTING ACCURACY]")

        invalid_mask = (
            np.isnan(f0_user_aligned) | np.isnan(f0_ref_aligned) |
            np.isinf(f0_user_aligned) | np.isinf(f0_ref_aligned) |
            (f0_user_aligned <= 0) | (f0_ref_aligned <= 0)
        )
        f0_user_clean = f0_user_aligned[~invalid_mask]
        f0_ref_clean = f0_ref_aligned[~invalid_mask]

        if len(f0_user_clean) < 2:
            return 0.0

        with np.errstate(divide='ignore', invalid='ignore'):
            cents_diff = 1200 * np.log2((f0_user_clean + 1e-8) / (f0_ref_clean + 1e-8))
        cents_error = np.abs(np.nan_to_num(cents_diff, nan=0.0, posinf=0.0, neginf=0.0))
        cents_error = cents_error[cents_error < 1000]

        if len(cents_error) < 2:
            return 0.0

        accuracy = 100 * np.sum(cents_error < tolerance_cents) / len(cents_error)
        accuracy = float(np.clip(accuracy, 0, 100))

        mean_error = np.mean(cents_error)
        std_error = np.std(cents_error)

        print(f"✓ Pitch accuracy: {accuracy:.1f}%")
        print(f"  Mean error: {mean_error:.1f} cents")
        print(f"  Std error:  {std_error:.1f} cents")
        print(f"  Frames compared: {len(cents_error)}")
        print(f"  Tempo ratio: {tempo_ratio:.3f}x")
        print("="*60 + "\n")

        return accuracy

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


# # Feature Extraction - Rhythm

def extract_rhythm_features(audio, sr):
    print("Extracting rhythm features...")

    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    tempo = float(tempo)

    print(f"Detected tempo: {tempo:.1f} BPM")
    print(f"Detected {len(beat_times)} beats in {len(audio)/sr:.2f}s")

    return tempo, beat_times


def compute_rhythm_accuracy(user_audio, ref_audio, sr, tolerance_ms=100.0):
    print("\n" + "="*60)
    print("RHYTHM ACCURACY (BALANCED: Sync + Regularity)")
    print("="*60)

    try:
        print(f"\n[1] Validating inputs...")
        if user_audio is None or len(user_audio) == 0:
            print("ERROR: User audio empty"); return 0.0
        if ref_audio is None or len(ref_audio) == 0:
            print("ERROR: Ref audio empty"); return 0.0
        print(f"  ✓ Valid")

        tolerance_sec = tolerance_ms / 1000.0

        print(f"\n[2] Extracting beats...")
        try:
            onset_user = librosa.onset.onset_strength(y=user_audio, sr=sr)
            tempo_user, beats_user = librosa.beat.beat_track(
                onset_envelope=onset_user, sr=sr, units='time'
            )
            tempo_user = float(tempo_user)
            print(f"  ✓ User: {len(beats_user)} beats, {tempo_user:.1f} BPM")
        except Exception as e:
            print(f"  ✗ User beat ERROR: {type(e).__name__}: {e}")
            return 0.0

        try:
            onset_ref = librosa.onset.onset_strength(y=ref_audio, sr=sr)
            tempo_ref, beats_ref = librosa.beat.beat_track(
                onset_envelope=onset_ref, sr=sr, units='time'
            )
            tempo_ref = float(tempo_ref)
            print(f"  ✓ Ref: {len(beats_ref)} beats, {tempo_ref:.1f} BPM")
        except Exception as e:
            print(f"  ✗ Ref beat ERROR: {type(e).__name__}: {e}")
            return 0.0

        if len(beats_user) < 2 or len(beats_ref) < 2:
            print(f"  ERROR: Not enough beats"); return 0.0

        print(f"\n[3] Computing synchronization accuracy...")

        tempo_ratio = tempo_user / tempo_ref
        adjusted_tolerance = tolerance_sec / tempo_ratio

        print(f"  Tempo ratio: {tempo_ratio:.3f}x")
        print(f"  Adjusted tolerance: {adjusted_tolerance*1000:.0f}ms (from {tolerance_ms:.0f}ms)")

        matched_pairs = []
        for user_beat in beats_user:
            distances = np.abs(beats_ref - user_beat)
            if np.min(distances) <= adjusted_tolerance:
                matched_pairs.append(np.min(distances))

        if len(matched_pairs) < 2:
            print(f"  Retrying with 0.5s tolerance...")
            adjusted_tolerance = 0.5 / tempo_ratio
            matched_pairs = []
            for user_beat in beats_user:
                distances = np.abs(beats_ref - user_beat)
                if np.min(distances) <= adjusted_tolerance:
                    matched_pairs.append(np.min(distances))

        if len(matched_pairs) < 2:
            print(f"  ERROR: Too few synchronized beats")
            return 0.0

        sync_match_ratio = len(matched_pairs) / len(beats_user)
        sync_timing_error = np.mean(matched_pairs) / adjusted_tolerance
        sync_accuracy = (sync_match_ratio * 0.6 + (1 - np.clip(sync_timing_error, 0, 1)) * 0.4) * 100
        sync_accuracy = float(np.clip(sync_accuracy, 0, 100))

        print(f"  Matched: {len(matched_pairs)}/{len(beats_user)} beats")
        print(f"  Sync accuracy: {sync_accuracy:.1f}%")

        print(f"\n[4] Computing beat regularity...")

        intervals_user = np.diff(beats_user)
        mean_interval = np.mean(intervals_user)
        std_interval = np.std(intervals_user)
        cv = std_interval / mean_interval if mean_interval > 0 else 1.0

        regularity_accuracy = 100 * np.exp(-cv * 2)
        regularity_accuracy = float(np.clip(regularity_accuracy, 0, 100))

        print(f"  Mean interval: {mean_interval:.3f}s")
        print(f"  Std deviation: {std_interval:.4f}s")
        print(f"  Coefficient of variation: {cv:.3f}")
        print(f"  Regularity accuracy: {regularity_accuracy:.1f}%")

        print(f"\n[5] Final calculation...")

        rhythm_accuracy = (sync_accuracy * 0.5 + regularity_accuracy * 0.5)
        rhythm_accuracy = float(np.clip(rhythm_accuracy, 0, 100))

        print(f"\n✓ Final Rhythm Accuracy: {rhythm_accuracy:.1f}%")
        print(f"  Sync: {sync_accuracy:.1f}%")
        print(f"  Regularity: {regularity_accuracy:.1f}%")
        print("="*60 + "\n")

        return rhythm_accuracy

    except Exception as e:
        print(f"✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


# # Feature Extraction - Emotion

# ── CHANGE 1: Cache emotion models at module level ──────────────────────────
# Previously load_emotion_classifier() opened 3 pickle files on every single
# call to predicting_emotion(). Now they load once and stay in memory.
import pickle

_emotion_clf    = None
_emotion_scaler = None
_emotion_labels = None

def load_emotion_classifier():
    global _emotion_clf, _emotion_scaler, _emotion_labels

    if _emotion_clf is not None:
        return _emotion_clf, _emotion_scaler, _emotion_labels

    model_dir = 'Models'
    with open(os.path.join(model_dir, 'emotion_classifier.pkl'), 'rb') as f:
        _emotion_clf = pickle.load(f)
    with open(os.path.join(model_dir, 'emotion_scaler.pkl'), 'rb') as f:
        _emotion_scaler = pickle.load(f)
    with open(os.path.join(model_dir, 'emotion_labels.pkl'), 'rb') as f:
        _emotion_labels = pickle.load(f)

    return _emotion_clf, _emotion_scaler, _emotion_labels
# ────────────────────────────────────────────────────────────────────────────


def extract_emotion_features(audio_path, sr=22050):
    try:
        y, sr = librosa.load(audio_path, sr=sr)

        if len(y) < sr * 0.5:
            return None

        features = []

        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features.append(float(centroid))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        for val in mfcc_mean[:3]:
            features.append(float(val))

        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features.append(float(rolloff))

        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        features.append(float(zcr))

        rms = np.mean(librosa.feature.rms(y=y)[0])
        features.append(float(rms))

        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        features.append(float(contrast))

        for val in mfcc_mean[4:9]:
            features.append(float(val))

        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        for val in chroma[:4]:
            features.append(float(val))

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        try:
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        except:
            tempo = 0
        features.append(float(tempo))

        spectral_flux = np.mean(np.sqrt(
            np.sum(np.diff(np.abs(librosa.stft(y)), axis=0)**2, axis=0)
        ))
        features.append(float(spectral_flux))

        loudness = rms * 100
        features.append(float(loudness))
        features.append(float(np.std(librosa.feature.rms(y=y)[0])))

        if len(features) != 21:
            return None

        features_array = np.array(features, dtype=np.float32)

        if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
            return None

        return features_array

    except Exception as e:
        return None


def predicting_emotion(audio_path):
    clf, scaler, emotion_labels = load_emotion_classifier()
    features_array = extract_emotion_features(audio_path)
    if features_array is None:
        return {
            'emotion': None,
            'confidence': 0.0,
            'all_scores': {},
            'error': 'Feature extraction failed'
        }

    features_scaled = scaler.transform(features_array.reshape(1, -1))
    emotion_index = clf.predict(features_scaled)[0]
    predicted_emotion = emotion_labels[emotion_index]

    probabilities = clf.predict_proba(features_scaled)[0]
    print(f"Probability shape: {probabilities.shape}")
    print(f"Probabilities for each emotion:")

    all_scores = {}
    for emotion_idx, probability in enumerate(probabilities):
        emotion_name_i = emotion_labels[emotion_idx]
        confidence_pct = probability * 100
        all_scores[emotion_name_i] = confidence_pct
        print(f"  {emotion_name_i:10s}: {confidence_pct:6.2f}%")

    predicted_confidence = probabilities[emotion_index]
    print(f"\n✓ Predicted emotion: {predicted_emotion}")
    print(f"  Confidence: {predicted_confidence*100:.2f}%\n")

    return {
        'emotion': predicted_emotion,
        'confidence': predicted_confidence,
        'all_scores': all_scores,
        'error': None
    }


def get_emotion_feedback(emotion: str, confidence: float) -> str:
    feedback_map = {
        'Happy':     "😊 Detected: Happy/Uplifting. You sound joyful!",
        'Sad':       "😢 Detected: Sad/Melancholic. You conveyed emotion well!",
        'Neutral':   "😐 Detected: Neutral/Monotone. Try expressing more feeling.",
        'Energetic': "⚡ Detected: Energetic/Powerful. Great energy and passion!",
    }

    base_feedback = feedback_map.get(emotion, f"Emotion: {emotion}")

    if confidence < 0.6:
        return f"{base_feedback} (low confidence, unclear emotion)"
    return base_feedback


# # Scoring

def compute_final_score(pitch_accuracy, rhythm_accuracy, emotion_confidence,
                        pitch_weight=0.50, rhythm_weight=0.35, emotion_weight=0.15):
    import numpy as np

    emotion_accuracy = emotion_confidence * 100

    def apply_leniency_curve(score, leniency=1.5):
        normalized = score / 100.0
        leniency_power = 1.0 / leniency
        curved = normalized ** leniency_power
        return curved * 100.0

    pitch_lenient   = apply_leniency_curve(pitch_accuracy,   leniency=1.5)
    rhythm_lenient  = apply_leniency_curve(rhythm_accuracy,  leniency=1.5)
    emotion_lenient = apply_leniency_curve(emotion_accuracy, leniency=1.5)

    print(f"\n[LENIENCY APPLIED]")
    print(f"  Pitch:   {pitch_accuracy:6.1f}% → {pitch_lenient:6.1f}%")
    print(f"  Rhythm:  {rhythm_accuracy:6.1f}% → {rhythm_lenient:6.1f}%")
    print(f"  Emotion: {emotion_accuracy:6.1f}% → {emotion_lenient:6.1f}%")

    final_score = (
        pitch_weight   * pitch_lenient +
        rhythm_weight  * rhythm_lenient +
        emotion_weight * emotion_lenient
    )

    minimum_score = 30.0
    final_score = max(final_score, minimum_score)
    final_score = np.clip(final_score, 0, 100)

    print(f"\n[SCORING CALCULATION]")
    print(f"  Pitch:   {pitch_lenient:6.1f}% × {pitch_weight:.2f} = {pitch_lenient * pitch_weight:6.2f}")
    print(f"  Rhythm:  {rhythm_lenient:6.1f}% × {rhythm_weight:.2f} = {rhythm_lenient * rhythm_weight:6.2f}")
    print(f"  Emotion: {emotion_lenient:6.1f}% × {emotion_weight:.2f} = {emotion_lenient * emotion_weight:6.2f}")
    print(f"\n  Final score: {final_score:6.1f}/100")

    return final_score


def score_to_grade(score):
    if score >= 90: return "A+ (Outstanding)"
    elif score >= 80: return "A (Excellent)"
    elif score >= 70: return "B (Good)"
    elif score >= 60: return "C (Satisfactory)"
    elif score >= 50: return "D (Needs Work)"
    else: return "F (Start Over)"


def get_overall_verdict(score):
    if score >= 85:   return "🌟 Excellent! You're a natural! Keep it up!"
    elif score >= 70: return "👍 Good job! You've got talent. Polish the details."
    elif score >= 50: return "✌️ Not bad! Keep practicing. Focus on weak areas below."
    elif score >= 30: return "💪 You're working on it. Regular practice will help!"
    else:             return "🎯 New singer? No worries! Practice is key."


def generate_feedback(pitch_accuracy, rhythm_accuracy, emotion_label,
                      emotion_confidence, overall_score):
    feedback_dict = {
        'pitch_score':        round(pitch_accuracy, 1),
        'rhythm_score':       round(rhythm_accuracy, 1),
        'emotion_detected':   emotion_label,
        'emotion_confidence': round(emotion_confidence, 2),
        'overall_score':      round(overall_score, 1),
        'grade':              score_to_grade(overall_score),
        'verdict':            get_overall_verdict(overall_score),
        'suggestions':        []
    }

    if pitch_accuracy < 70:
        feedback_dict['suggestions'].append(
            f"📍 Pitch Accuracy ({pitch_accuracy:.0f}%): Your pitch is off. "
            "Use a tuner app to practice hitting exact notes."
        )
    if rhythm_accuracy < 70:
        feedback_dict['suggestions'].append(
            f"⏱️ Rhythm Accuracy ({rhythm_accuracy:.0f}%): Your timing needs work. "
            "Practice with a metronome starting at a slow tempo."
        )
    if emotion_confidence < 0.6:
        feedback_dict['suggestions'].append(
            f"😊 Emotion: Your singing lacks emotional expression. "
            "Focus on the lyrics and try to feel the song."
        )
    if pitch_accuracy >= 80:
        feedback_dict['suggestions'].append("✨ Great pitch control! You have good intonation.")
    if rhythm_accuracy >= 80:
        feedback_dict['suggestions'].append("✨ Excellent timing! You're very rhythmically accurate.")
    if emotion_confidence >= 0.8:
        feedback_dict['suggestions'].append("✨ You conveyed emotion very well. Keep that expressiveness!")
    if pitch_accuracy >= 80 and rhythm_accuracy >= 80:
        feedback_dict['suggestions'].append(
            "🎭 You nailed the technical aspects! Now work on emotional expression and dynamics."
        )

    return feedback_dict


def generate_detailed_report(results):
    feedback = results.get('feedback_dict', {})

    report = []
    report.append("SINGING PERFORMANCE EVALUATION REPORT: ")
    report.append("")
    report.append("SCORES: ")
    report.append("")
    report.append(f"Pitch Accuracy:     {feedback.get('pitch_score', 'N/A'):>6}%")
    report.append(f"Rhythm Accuracy:    {feedback.get('rhythm_score', 'N/A'):>6}%")
    report.append(f"Emotion:            {feedback.get('emotion_detected', 'N/A'):>6} ({feedback.get('emotion_confidence', 0):.0%})")
    report.append(f"Overall Score:      {feedback.get('overall_score', 'N/A'):>6}/100")
    report.append(f"Grade:              {feedback.get('grade', 'N/A'):>6}")
    report.append("")
    report.append("VERDICT: ")
    report.append("")
    report.append(feedback.get('verdict', 'N/A'))
    report.append("")

    if feedback.get('suggestions'):
        report.append("AREAS TO IMPROVE & SUGGESTIONS")
        report.append("")
        for i, suggestion in enumerate(feedback['suggestions'], 1):
            report.append(f"{i}. {suggestion}")
        report.append("")

    report.append("")
    report.append("Keep practicing! Every singer started somewhere.")
    report.append("")

    return "\n".join(report)


# # Pipeline Function

def evaluate_singing(user_audio_path, reference_audio_path, model_dir='Models'):
    # ── CHANGE 2: Lower sr from 22050 → 16000 ───────────────────────────────
    # Cuts every audio array size by ~27% with no meaningful accuracy loss for
    # pitch/rhythm analysis. Singing fundamentals are well below 8 kHz.
    sr = 16000
    # ────────────────────────────────────────────────────────────────────────

    results = {
        'success': False,
        'errors': [],
        'pitch_accuracy': 0,
        'rhythm_accuracy': 0,
        'emotion_detected': 'neutral',
        'emotion_confidence': 0.0,
        'final_score': 0,
        'feedback_dict': {},
        'report': ''
    }

    print("SINGING PERFORMANCE EVALUATION\n\n")

    # STEP 1: LOAD & VALIDATE AUDIO
    print("\n[STEP 1] Loading and validating audio files...\n")

    user_audio, ref_audio, sr, validation_ok, error_msg = load_and_validate_audio_pair(
        user_audio_path, reference_audio_path, sr=sr
    )

    if not validation_ok:
        results['errors'].append(error_msg)
        results['report'] = f"❌ Evaluation failed: {error_msg}"
        print(results['report'])
        return results

    # STEP 2: PREPROCESS AUDIO
    print("\n[STEP 2] Preprocessing audio...\n")

    user_audio = trim_silence(user_audio, sr)
    ref_audio  = trim_silence(ref_audio, sr)
    user_audio = reduce_noise_spectral_subtraction(user_audio, sr, noise_duration=1.0)

    print("✓ Audio preprocessing complete")

    # STEP 3: SEGMENT MATCHING (already done in app.py via waveform UI)
    print("\n[STEP 3] Segment Matching...\n")
    print(f"Using ref_audio (already sliced from waveform)")
    print(f"User audio: {len(user_audio)/sr:.1f}s")
    print(f"Ref audio: {len(ref_audio)/sr:.1f}s")

    user_audio, ref_audio = prepare_for_comparison(user_audio, ref_audio, sr)

    # STEP 4: PITCH ACCURACY
    print("\n[STEP 4] Pitch accuracy calculation...\n")

    pitch_accuracy = compute_pitch_accuracy(user_audio, ref_audio, sr)
    results['pitch_accuracy'] = pitch_accuracy

    # STEP 5: RHYTHM ACCURACY
    print("\n[STEP 5] Computing rhythm accuracy...\n")

    rhythm_accuracy = compute_rhythm_accuracy(user_audio, ref_audio, sr)
    results['rhythm_accuracy'] = rhythm_accuracy

    # ── CHANGE 3: Free audio arrays before loading emotion model ────────────
    # Pitch and rhythm are computed — the large numpy arrays are no longer
    # needed. Freeing them here reclaims ~20 MB before the emotion model runs.
    del user_audio, ref_audio
    gc.collect()
    # ────────────────────────────────────────────────────────────────────────

    # STEP 6: PREDICT EMOTION
    print("\n[STEP 6] Detecting emotion...\n")

    emotion_results   = predicting_emotion(user_audio_path)
    emotion_detected  = emotion_results['emotion']
    emotion_confidence = emotion_results['confidence']

    results['emotion_detected']  = emotion_detected
    results['emotion_confidence'] = emotion_confidence

    # STEP 7: FINAL SCORE
    print("\n[STEP 7] Calculating final score...\n")

    final_score = compute_final_score(
        pitch_accuracy, rhythm_accuracy, emotion_confidence,
        pitch_weight=0.50, rhythm_weight=0.35, emotion_weight=0.15
    )
    results['final_score'] = final_score

    # STEP 8: FEEDBACK
    print("\n[STEP 8] Generating feedback...\n")

    feedback_dict = generate_feedback(
        pitch_accuracy, rhythm_accuracy,
        emotion_detected, emotion_confidence, final_score
    )
    results['feedback_dict'] = feedback_dict

    # STEP 9: REPORT
    print("\n[STEP 9] Generating report...\n")

    report = generate_detailed_report(results)
    results['report'] = report
    results['success'] = True

    print("\n\n**EVALUATION COMPLETE**\n\n")
    print(report)

    return results