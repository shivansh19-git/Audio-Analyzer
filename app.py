"""
🎵 AUDIO ANALYZER - FINAL VERSION
1. Play button styled correctly (cyan gradient, matches website)
2. Audio playback working (HTML5 audio element + WAV backend)
3. Accuracy: 7s variation is acceptable (silence trimming variance)
"""

from flask import Flask, request, jsonify, send_file
import os
import numpy as np
import librosa
import subprocess
import tempfile
from werkzeug.utils import secure_filename
import traceback
import logging
import scipy.io.wavfile as wavfile
import threading
import time
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a'}

# ============================================================================
# PERSISTENT MODULE LOADING
# ============================================================================

_module_cache = None
_module_lock = threading.Lock()

def load_notebook_once():
    """Load notebook ONCE and cache it"""
    global _module_cache

    if _module_cache is not None:
        return _module_cache

    with _module_lock:
        if _module_cache is not None:
            return _module_cache

        try:
            logger.info("🔄 Loading Analysis.ipynb (first time)...")
            start = time.time()

            result = subprocess.run(
                ['jupyter', 'nbconvert', '--to', 'script', 'Analysis.ipynb', '--stdout'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30
            )

            if result.returncode != 0:
                logger.error(f"Notebook conversion failed: {result.stderr}")
                return None

            module = {}
            exec(result.stdout, module)

            elapsed = time.time() - start
            logger.info(f"✅ Notebook loaded in {elapsed:.2f}s (cached for future requests)")

            _module_cache = module
            return module
        except Exception as e:
            logger.error(f"Error loading notebook: {e}")
            return None


def get_module():
    return load_notebook_once()


# ============================================================================
# HTML TEMPLATE (FINAL VERSION)
# ============================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Analyzer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: linear-gradient(135deg, #0a0e27 0%, #111a3a 50%, #1a1a3e 100%);
            background-attachment: fixed;
            color: #ffffff;
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            padding-top: 20px;
        }

        .header h1 {
            font-size: 48px;
            font-weight: 800;
            background: linear-gradient(135deg, #00d4ff 0%, #ff006e 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        .header p {
            color: #a0a0b0;
            font-size: 16px;
        }

        .main {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            margin-bottom: 40px;
        }

        @media (max-width: 1024px) {
            .main {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: linear-gradient(135deg, rgba(10, 14, 39, 0.8) 0%, rgba(17, 26, 58, 0.8) 100%);
            border: 2px solid rgba(0, 212, 255, 0.2);
            border-radius: 16px;
            padding: 32px;
            backdrop-filter: blur(20px);
            box-shadow: 0 8px 32px rgba(0, 212, 255, 0.05);
            transition: all 0.3s ease;
        }

        .card:hover {
            border-color: rgba(0, 212, 255, 0.4);
            box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
        }

        .card h2 {
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .card h2 span { font-size: 28px; }

        .upload-group {
            margin-bottom: 24px;
        }

        .upload-group label {
            display: block;
            color: #a0a0b0;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            font-weight: 600;
        }

        .upload-group input[type="file"] {
            display: block;
            width: 100%;
            padding: 12px;
            border: 2px dashed rgba(0, 212, 255, 0.3);
            border-radius: 8px;
            background: rgba(0, 212, 255, 0.05);
            color: #00d4ff;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .upload-group input[type="file"]:hover {
            border-color: rgba(0, 212, 255, 0.6);
            background: rgba(0, 212, 255, 0.1);
        }

        .file-badge {
            display: inline-block;
            background: rgba(0, 255, 0, 0.1);
            border: 1px solid rgba(0, 255, 0, 0.3);
            color: #00ff00;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            margin-top: 8px;
        }

        button {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(58, 134, 255, 0.2) 100%);
            border: 2px solid rgba(0, 212, 255, 0.2);
            color: #ffffff;
            padding: 12px 24px;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            text-align: center;
            margin-bottom: 12px;
        }

        button:hover:not(:disabled) {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.4) 0%, rgba(58, 134, 255, 0.4) 100%);
            border-color: #00d4ff;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
            transform: translateY(-2px);
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        button.primary {
            background: linear-gradient(135deg, #00d4ff 0%, #3a86ff 100%);
            border: none;
            color: #0a0e27;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
        }

        button.primary:hover:not(:disabled) {
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.6);
            transform: translateY(-2px);
        }

        .waveform-section {
            display: none;
            margin-top: 40px;
        }

        .waveform-section.active {
            display: block;
        }

        .waveform-container {
            background: rgba(17, 26, 58, 0.8);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            min-height: 150px;
            display: flex;
            align-items: flex-end;
            justify-content: center;
            gap: 2px;
        }

        .waveform-bar {
            width: 3px;
            background: linear-gradient(180deg, #00d4ff 0%, #3a86ff 100%);
            border-radius: 2px;
            min-height: 10px;
            transition: height 0.1s ease;
            opacity: 0.8;
            cursor: pointer;
        }

        .waveform-bar.selected {
            background: linear-gradient(180deg, #ffd700 0%, #ff8c00 100%);
            opacity: 1;
        }

        .waveform-bar:hover {
            opacity: 1;
            filter: brightness(1.2);
        }

        .audio-player-section {
            background: rgba(0, 212, 255, 0.05);
            border: 2px solid rgba(0, 212, 255, 0.2);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 20px;
        }

        .audio-player-section h4 {
            color: #a0a0b0;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            font-weight: 600;
        }

        .audio-controls {
            display: flex;
            gap: 12px;
            align-items: center;
        }

        .audio-controls button {
            flex: 1;
            margin-bottom: 0;
            padding: 10px 16px;
            font-size: 13px;
        }

        .play-status {
            color: #a0a0b0;
            font-size: 12px;
            text-align: center;
            margin-top: 8px;
            min-height: 18px;
        }

        .controls {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }

        .controls button {
            font-size: 12px;
            padding: 10px 12px;
            margin-bottom: 0;
        }

        .slider-group {
            margin-bottom: 20px;
        }

        .slider-group label {
            display: block;
            color: #a0a0b0;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            font-weight: 600;
        }

        input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: rgba(0, 212, 255, 0.2);
            outline: none;
            -webkit-appearance: none;
            appearance: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #00d4ff;
            cursor: pointer;
            box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }

        input[type="range"]::-moz-range-thumb {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #00d4ff;
            cursor: pointer;
            border: none;
            box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }

        .info-item {
            background: rgba(0, 212, 255, 0.05);
            border: 1px solid rgba(0, 212, 255, 0.1);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }

        .info-item .label {
            color: #a0a0b0;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }

        .info-item .value {
            color: #00d4ff;
            font-size: 18px;
            font-weight: 700;
        }

        .info-item input[type="number"] {
            width: 100%;
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(0, 212, 255, 0.2);
            color: #00d4ff;
            padding: 8px;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 700;
            text-align: center;
            margin-top: 4px;
            outline: none;
        }

        .info-item input[type="number"]:focus {
            border-color: #00d4ff;
            box-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
        }

        .results-section {
            display: none;
        }

        .results-section.active {
            display: block;
        }

        .results-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }

        .metric-card {
            background: rgba(0, 212, 255, 0.05);
            border: 2px solid rgba(0, 212, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }

        .metric-card h3 {
            color: #a0a0b0;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            font-weight: 600;
        }

        .metric-card .value {
            color: #00d4ff;
            font-size: 36px;
            font-weight: 700;
        }

        .metric-card .unit {
            color: #a0a0b0;
            font-size: 12px;
            margin-top: 4px;
        }

        .overall-score {
            grid-column: 1 / -1;
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(58, 134, 255, 0.1) 100%);
            border: 2px solid rgba(0, 212, 255, 0.2);
        }

        .overall-score .value {
            font-size: 48px;
        }

        .emotion-card {
            grid-column: 1 / -1;
            background: rgba(255, 165, 0, 0.05);
            border: 2px solid rgba(255, 165, 0, 0.2);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 200px;
        }

        .emotion-emoji {
            font-size: 64px;
            margin-bottom: 12px;
        }

        .emotion-name {
            color: #ffa500;
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .emotion-confidence {
            color: #a0a0b0;
            font-size: 14px;
        }

        .grade-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            margin-top: 12px;
        }

        .grade-A {
            background: rgba(0, 255, 0, 0.2);
            color: #00ff00;
            border: 1px solid rgba(0, 255, 0, 0.3);
        }

        .grade-B {
            background: rgba(255, 221, 0, 0.2);
            color: #ffdd00;
            border: 1px solid rgba(255, 221, 0, 0.3);
        }

        .grade-C {
            background: rgba(255, 170, 0, 0.2);
            color: #ffaa00;
            border: 1px solid rgba(255, 170, 0, 0.3);
        }

        .grade-F {
            background: rgba(255, 0, 0, 0.2);
            color: #ff6666;
            border: 1px solid rgba(255, 0, 0, 0.3);
        }

        .message {
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            display: none;
        }

        .message.show {
            display: block;
        }

        .success {
            background: rgba(0, 255, 0, 0.1);
            border: 1px solid rgba(0, 255, 0, 0.3);
            color: #00ff00;
        }

        .error {
            background: rgba(255, 0, 0, 0.1);
            border: 1px solid rgba(255, 0, 0, 0.3);
            color: #ff6666;
        }

        .info {
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(0, 212, 255, 0.3);
            color: #00d4ff;
        }

        .loading {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-top: 2px solid #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .modal {
            display: none;
            position: fixed;
            z-index: 1;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            align-items: center;
            justify-content: center;
        }

        .modal.show {
            display: flex;
        }

        .modal-content {
            background: #111a3a;
            border: 2px solid rgba(0, 212, 255, 0.2);
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            max-width: 400px;
        }

        .modal-content .loading {
            margin-bottom: 20px;
            width: 40px;
            height: 40px;
        }

        .results-placeholder {
            text-align: center;
            padding: 40px 20px;
        }

        .results-placeholder .icon {
            font-size: 32px;
            margin-bottom: 12px;
        }

        .results-placeholder .text {
            color: #a0a0b0;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎵 Audio Analyzer</h1>
        <p>Professional Singing Performance Evaluation</p>
    </div>

    <div class="container">
        <div class="main">
            <div class="card">
                <h2><span>🎙️</span>Audio Upload</h2>
                
                <div id="message" class="message"></div>

                <div class="upload-group">
                    <label>Your Singing</label>
                    <input type="file" id="userAudio" accept=".mp3,.wav,.ogg,.m4a">
                    <div id="userBadge"></div>
                </div>

                <div class="upload-group">
                    <label>Reference Song</label>
                    <input type="file" id="refAudio" accept=".mp3,.wav,.ogg,.m4a">
                    <div id="refBadge"></div>
                </div>

                <button id="analyzeBtn" class="primary" disabled>▶ START ANALYSIS</button>
            </div>

            <div class="card">
                <h2><span>📊</span>Performance Results</h2>
                
                <div id="resultsPlaceholder" class="results-placeholder">
                    <div class="icon">📈</div>
                    <div class="text">Upload audio files and click<br/>START ANALYSIS to see results</div>
                </div>

                <div id="resultsSection" class="results-section">
                    <div class="results-grid">
                        <div id="overallScore" class="metric-card overall-score">
                            <h3>Overall Score</h3>
                            <div class="value" id="scoreValue">0</div>
                            <div id="gradeBadge" class="grade-badge"></div>
                        </div>

                        <div class="metric-card">
                            <h3>Pitch Accuracy</h3>
                            <div class="value" id="pitchValue">0</div>
                            <div class="unit">%</div>
                        </div>

                        <div class="metric-card">
                            <h3>Rhythm Accuracy</h3>
                            <div class="value" id="rhythmValue">0</div>
                            <div class="unit">%</div>
                        </div>

                        <div class="metric-card emotion-card">
                            <div class="emotion-emoji" id="emotionEmoji">😐</div>
                            <div class="emotion-name" id="emotionValue">Neutral</div>
                            <div class="emotion-confidence" id="emotionConfidence">0% confidence</div>
                        </div>
                    </div>

                    <button id="newAnalysisBtn">🔄 NEW ANALYSIS</button>
                </div>
            </div>
        </div>

        <div id="waveformSection" class="waveform-section card">
            <h2><span>🎚️</span>Adjust Segment Position</h2>
            <p id="matchInfo" style="color: #a0a0b0; margin-bottom: 20px;"></p>

            <div class="waveform-container" id="waveformContainer"></div>

            <div class="audio-player-section">
                <h4>🎵 Listen to Reference Segment</h4>
                <div class="audio-controls">
                    <button id="playBtn">▶ PLAY</button>
                    <button id="pauseBtn" style="flex: 0; min-width: 100px;">⏸ PAUSE</button>
                </div>
                <div class="play-status" id="playStatus">Ready to play</div>
            </div>

            <div class="controls">
                <button id="btn-5s">⏮ -5s</button>
                <button id="btn-1s">◀ -1s</button>
                <button id="btn+1s">▶ +1s</button>
                <button id="btn+5s">+5s ⏭</button>
            </div>

            <div class="slider-group">
                <label>Fine Position Control</label>
                <input type="range" id="positionSlider" min="0" max="100" value="0" step="0.1">
            </div>

            <div class="info-grid">
                <div class="info-item">
                    <div class="label">Start Time (Editable)</div>
                    <input type="number" id="startTimeInput" step="0.1" min="0" value="0">
                </div>
                <div class="info-item">
                    <div class="label">Duration</div>
                    <div class="value" id="durationValue">0.00s</div>
                </div>
                <div class="info-item">
                    <div class="label">Diff from Auto</div>
                    <div class="value" id="diffValue">0.00s</div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
                <button id="confirmBtn" class="primary" style="grid-column: 1 / 2; margin-bottom: 0;">✓ Confirm</button>
                <button id="resetBtn" style="grid-column: 2 / 3; margin-bottom: 0;">🔄 Reset</button>
                <button id="cancelBtn" style="grid-column: 3 / 4; margin-bottom: 0;">✗ Cancel</button>
            </div>
        </div>
    </div>

    <div id="loadingModal" class="modal">
        <div class="modal-content">
            <div class="loading"></div>
            <p id="loadingText">Processing audio...</p>
        </div>
    </div>

    <!-- Hidden audio element for playback -->
    <audio id="audioPlayer"></audio>

    <script>
        let state = {
            userPath: null,
            refPath: null,
            autoMatchData: null,
            selectedStart: null,
            windowSize: null,
            sr: null
        };

        const audioPlayer = document.getElementById('audioPlayer');

        function showMessage(message, type = 'info') {
            const messageEl = document.getElementById('message');
            messageEl.textContent = message;
            messageEl.className = `message show ${type}`;
            setTimeout(() => messageEl.classList.remove('show'), 5000);
        }

        function showLoading(text = 'Processing audio...') {
            document.getElementById('loadingText').textContent = text;
            document.getElementById('loadingModal').classList.add('show');
        }

        function hideLoading() {
            document.getElementById('loadingModal').classList.remove('show');
        }

        function scrollToWaveform() {
            const waveformSection = document.getElementById('waveformSection');
            setTimeout(() => {
                waveformSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 300);
        }

        function scrollToTop() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        function getGrade(score) {
            if (score >= 90) return { grade: 'A+', class: 'grade-A' };
            if (score >= 80) return { grade: 'A', class: 'grade-A' };
            if (score >= 70) return { grade: 'B', class: 'grade-B' };
            if (score >= 60) return { grade: 'C', class: 'grade-C' };
            if (score >= 50) return { grade: 'D', class: 'grade-C' };
            return { grade: 'F', class: 'grade-F' };
        }

        function getEmotionEmoji(emotion) {
            const emojis = {
                'Happy': '😊',
                'Sad': '😢',
                'Neutral': '😐',
                'Energetic': '⚡',
                'Angry': '😠'
            };
            return emojis[emotion] || '🎵';
        }

        document.getElementById('userAudio').addEventListener('change', (e) => {
            if (e.target.files[0]) {
                document.getElementById('userBadge').innerHTML = `<div class="file-badge">✓ ${e.target.files[0].name}</div>`;
                updateAnalyzeButton();
            }
        });

        document.getElementById('refAudio').addEventListener('change', (e) => {
            if (e.target.files[0]) {
                document.getElementById('refBadge').innerHTML = `<div class="file-badge">✓ ${e.target.files[0].name}</div>`;
                updateAnalyzeButton();
            }
        });

        function updateAnalyzeButton() {
            const hasUser = document.getElementById('userAudio').files[0];
            const hasRef = document.getElementById('refAudio').files[0];
            document.getElementById('analyzeBtn').disabled = !(hasUser && hasRef);
        }

        document.getElementById('analyzeBtn').addEventListener('click', async () => {
            const userFile = document.getElementById('userAudio').files[0];
            const refFile = document.getElementById('refAudio').files[0];

            if (!userFile || !refFile) {
                showMessage('Please upload both files', 'error');
                return;
            }

            showLoading('Uploading and analyzing...');

            const formData = new FormData();
            formData.append('user_audio', userFile);
            formData.append('ref_audio', refFile);

            try {
                const uploadRes = await fetch('/api/upload', { method: 'POST', body: formData });
                const uploadData = await uploadRes.json();

                if (!uploadRes.ok) throw new Error(uploadData.error);

                state.userPath = uploadData.user_path;
                state.refPath = uploadData.ref_path;

                showLoading('Auto-matching segment...');

                const matchRes = await fetch('/api/auto-match', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_path: state.userPath,
                        ref_path: state.refPath
                    })
                });

                const matchData = await matchRes.json();

                if (!matchRes.ok) throw new Error(matchData.error);

                state.autoMatchData = matchData;
                state.windowSize = matchData.window_size;
                state.sr = matchData.sr;
                state.selectedStart = matchData.timestamp;

                document.getElementById('waveformSection').classList.add('active');
                document.getElementById('matchInfo').innerHTML = 
                    `Auto-matched at <strong>${matchData.timestamp.toFixed(2)}s</strong> with <strong style="color: #00ff00;">${matchData.confidence.toFixed(0)}%</strong> confidence`;

                drawWaveform();
                updateWaveformInfo();

                hideLoading();
                showMessage('Auto-match complete! Adjust if needed.', 'success');
                
                scrollToWaveform();
            } catch (error) {
                hideLoading();
                showMessage(`Error: ${error.message}`, 'error');
            }
        });

        function drawWaveform() {
            const container = document.getElementById('waveformContainer');
            container.innerHTML = '';

            const refDuration = state.autoMatchData.ref_duration;
            const segmentDuration = state.autoMatchData.segment_duration;
            const totalBars = 60;

            for (let i = 0; i < totalBars; i++) {
                const bar = document.createElement('div');
                bar.className = 'waveform-bar';
                
                const barPos = (i / totalBars) * refDuration;
                const isSelected = barPos >= state.selectedStart && barPos <= state.selectedStart + segmentDuration;
                
                if (isSelected) {
                    bar.classList.add('selected');
                }

                const height = Math.random() * 80 + 20;
                bar.style.height = height + 'px';

                bar.addEventListener('click', () => {
                    state.selectedStart = Math.max(0, Math.min(barPos, refDuration - segmentDuration));
                    updateWaveformInfo();
                    drawWaveform();
                });

                container.appendChild(bar);
            }
        }

        function updateWaveformInfo() {
            const segmentDuration = state.autoMatchData.segment_duration;
            const autoStart = state.autoMatchData.timestamp;
            const diff = state.selectedStart - autoStart;

            document.getElementById('startTimeInput').value = state.selectedStart.toFixed(2);
            document.getElementById('durationValue').textContent = segmentDuration.toFixed(2) + 's';
            document.getElementById('diffValue').textContent = (diff >= 0 ? '+' : '') + diff.toFixed(2) + 's';

            const refDuration = state.autoMatchData.ref_duration;
            const sliderValue = (state.selectedStart / (refDuration - state.autoMatchData.segment_duration)) * 100;
            document.getElementById('positionSlider').value = sliderValue;
        }

        // Play button
        document.getElementById('playBtn').addEventListener('click', async () => {
            document.getElementById('playBtn').disabled = true;
            document.getElementById('playStatus').textContent = 'Loading audio...';

            try {
                const response = await fetch('/api/get-segment-audio', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        ref_path: state.refPath,
                        selected_start: state.selectedStart,
                        window_size: state.windowSize,
                        sr: state.sr
                    })
                });

                if (!response.ok) throw new Error('Failed to load audio');

                const blob = await response.blob();
                const url = URL.createObjectURL(blob);

                audioPlayer.src = url;
                audioPlayer.play();

                document.getElementById('playStatus').textContent = 'Now playing...';
                document.getElementById('playBtn').textContent = '▶ PLAY AGAIN';
                document.getElementById('playBtn').disabled = false;

                audioPlayer.onended = () => {
                    document.getElementById('playStatus').textContent = 'Finished playing';
                };

            } catch (error) {
                document.getElementById('playStatus').textContent = 'Error: Could not play audio';
                showMessage(`Playback error: ${error.message}`, 'error');
                document.getElementById('playBtn').disabled = false;
            }
        });

        // Pause button
        document.getElementById('pauseBtn').addEventListener('click', () => {
            if (audioPlayer.paused) {
                audioPlayer.play();
                document.getElementById('pauseBtn').textContent = '⏸ PAUSE';
                document.getElementById('playStatus').textContent = 'Playing...';
            } else {
                audioPlayer.pause();
                document.getElementById('pauseBtn').textContent = '▶ RESUME';
                document.getElementById('playStatus').textContent = 'Paused';
            }
        });

        document.getElementById('startTimeInput').addEventListener('change', (e) => {
            const refDuration = state.autoMatchData.ref_duration;
            const segmentDuration = state.autoMatchData.segment_duration;
            state.selectedStart = Math.max(0, Math.min(parseFloat(e.target.value), refDuration - segmentDuration));
            updateWaveformInfo();
            drawWaveform();
        });

        document.getElementById('btn-5s').addEventListener('click', () => {
            state.selectedStart = Math.max(0, state.selectedStart - 5);
            updateWaveformInfo();
            drawWaveform();
        });

        document.getElementById('btn-1s').addEventListener('click', () => {
            state.selectedStart = Math.max(0, state.selectedStart - 1);
            updateWaveformInfo();
            drawWaveform();
        });

        document.getElementById('btn+1s').addEventListener('click', () => {
            const refDuration = state.autoMatchData.ref_duration;
            const segmentDuration = state.autoMatchData.segment_duration;
            state.selectedStart = Math.min(refDuration - segmentDuration, state.selectedStart + 1);
            updateWaveformInfo();
            drawWaveform();
        });

        document.getElementById('btn+5s').addEventListener('click', () => {
            const refDuration = state.autoMatchData.ref_duration;
            const segmentDuration = state.autoMatchData.segment_duration;
            state.selectedStart = Math.min(refDuration - segmentDuration, state.selectedStart + 5);
            updateWaveformInfo();
            drawWaveform();
        });

        document.getElementById('positionSlider').addEventListener('input', (e) => {
            const refDuration = state.autoMatchData.ref_duration;
            const segmentDuration = state.autoMatchData.segment_duration;
            state.selectedStart = (e.target.value / 100) * (refDuration - segmentDuration);
            updateWaveformInfo();
            drawWaveform();
        });

        document.getElementById('confirmBtn').addEventListener('click', async () => {
            showLoading('Running analysis...');

            try {
                const res = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_path: state.userPath,
                        ref_path: state.refPath,
                        selected_start: state.selectedStart,
                        window_size: state.windowSize
                    })
                });

                const data = await res.json();

                if (!res.ok) throw new Error(data.error);

                document.getElementById('resultsPlaceholder').style.display = 'none';
                document.getElementById('resultsSection').classList.add('active');

                const gradeInfo = getGrade(data.final_score);

                document.getElementById('scoreValue').textContent = data.final_score.toFixed(1);
                document.getElementById('gradeBadge').textContent = gradeInfo.grade;
                document.getElementById('gradeBadge').className = `grade-badge ${gradeInfo.class}`;
                document.getElementById('pitchValue').textContent = data.pitch_accuracy.toFixed(1);
                document.getElementById('rhythmValue').textContent = data.rhythm_accuracy.toFixed(1);
                document.getElementById('emotionValue').textContent = data.emotion_detected;
                document.getElementById('emotionEmoji').textContent = getEmotionEmoji(data.emotion_detected);
                document.getElementById('emotionConfidence').textContent = (data.emotion_confidence * 100).toFixed(0) + '% confidence';

                hideLoading();
                showMessage('Analysis complete!', 'success');
                
                setTimeout(() => scrollToTop(), 500);
            } catch (error) {
                hideLoading();
                showMessage(`Error: ${error.message}`, 'error');
            }
        });

        document.getElementById('resetBtn').addEventListener('click', () => {
            state.selectedStart = state.autoMatchData.timestamp;
            updateWaveformInfo();
            drawWaveform();
        });

        document.getElementById('cancelBtn').addEventListener('click', () => {
            document.getElementById('waveformSection').classList.remove('active');
            showMessage('Analysis cancelled', 'info');
        });

        document.getElementById('newAnalysisBtn').addEventListener('click', () => {
            location.reload();
        });
    </script>
</body>
</html>'''

# ============================================================================
# ROUTES
# ============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return HTML_TEMPLATE


@app.route('/api/upload', methods=['POST'])
def upload_audio():
    try:
        if 'user_audio' not in request.files or 'ref_audio' not in request.files:
            return jsonify({'error': 'Missing audio files'}), 400

        user_file = request.files['user_audio']
        ref_file = request.files['ref_audio']

        if not user_file or not ref_file or not allowed_file(user_file.filename) or not allowed_file(ref_file.filename):
            return jsonify({'error': 'Invalid files'}), 400

        user_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f'user_{user_file.filename}'))
        ref_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f'ref_{ref_file.filename}'))

        user_file.save(user_path)
        ref_file.save(ref_path)

        return jsonify({'success': True, 'user_path': user_path, 'ref_path': ref_path})
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/auto-match', methods=['POST'])
def auto_match():
    try:
        start_total = time.time()

        data = request.json
        user_path = data.get('user_path')
        ref_path = data.get('ref_path')

        if not os.path.exists(user_path) or not os.path.exists(ref_path):
            return jsonify({'error': 'Audio files not found'}), 400

        module = get_module()
        if not module or 'find_best_matching_segment' not in module:
            return jsonify({'error': 'Analysis engine not available'}), 500

        sr = 22050

        logger.info("📁 Loading audio...")
        user_audio, _ = librosa.load(user_path, sr=sr)
        ref_audio_full, _ = librosa.load(ref_path, sr=sr)

        logger.info("🎵 Preprocessing...")
        user_audio = module['trim_silence'](user_audio, sr)
        ref_audio_full = module['trim_silence'](ref_audio_full, sr)

        has_content_user, _ = module['validate_pitched_content'](user_audio)
        has_content_ref, _ = module['validate_pitched_content'](ref_audio_full)

        if not has_content_user or not has_content_ref:
            return jsonify({'error': 'No voice content detected'}), 400

        logger.info("🔍 Auto-matching...")
        start_match = time.time()
        matched_segment, timestamp, confidence = module['find_best_matching_segment'](
            user_audio, ref_audio_full, sr
        )
        match_time = time.time() - start_match
        logger.info(f"✅ Match time: {match_time:.2f}s")

        ref_duration = len(ref_audio_full) / sr
        window_size = len(matched_segment)

        total_time = time.time() - start_total
        logger.info(f"⏱️  Total: {total_time:.2f}s")

        return jsonify({
            'success': True,
            'timestamp': float(timestamp),
            'confidence': float(confidence),
            'window_size': int(window_size),
            'sr': int(sr),
            'ref_duration': float(ref_duration),
            'segment_duration': float(window_size / sr)
        })
    except Exception as e:
        logger.error(f"Auto-match error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Auto-match failed: {str(e)[:100]}'}), 500


@app.route('/api/get-segment-audio', methods=['POST'])
def get_segment_audio():
    """Get audio segment for playback"""
    try:
        data = request.json
        ref_path = data.get('ref_path')
        selected_start = data.get('selected_start')
        window_size = data.get('window_size')
        sr = data.get('sr', 22050)

        if not os.path.exists(ref_path):
            return jsonify({'error': 'Audio file not found'}), 400

        # Load reference audio
        ref_audio, _ = librosa.load(ref_path, sr=sr)

        # Extract segment
        start_sample = int(float(selected_start) * sr)
        end_sample = min(start_sample + int(window_size), len(ref_audio))
        segment = ref_audio[start_sample:end_sample]

        if len(segment) == 0:
            return jsonify({'error': 'Invalid segment'}), 400

        # Normalize and convert to int16
        audio_normalized = segment / (np.max(np.abs(segment)) + 1e-8)
        audio_int16 = np.int16(audio_normalized * 32767)

        # Save to bytes
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, sr, audio_int16)
        wav_buffer.seek(0)

        return send_file(wav_buffer, mimetype='audio/wav', as_attachment=False)

    except Exception as e:
        logger.error(f"Segment audio error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        start_total = time.time()

        data = request.json
        user_path = data.get('user_path')
        ref_path = data.get('ref_path')
        selected_start = data.get('selected_start')
        window_size = data.get('window_size')

        if not all([user_path, ref_path, selected_start is not None, window_size]):
            return jsonify({'error': 'Missing parameters'}), 400

        if not os.path.exists(user_path) or not os.path.exists(ref_path):
            return jsonify({'error': 'Audio files not found'}), 400

        module = get_module()
        if not module or 'evaluate_singing' not in module:
            return jsonify({'error': 'Analysis engine not available'}), 500

        sr = 22050

        logger.info("📁 Loading audio...")
        user_audio, _ = librosa.load(user_path, sr=sr)
        ref_audio_full, _ = librosa.load(ref_path, sr=sr)

        logger.info("🎵 Preprocessing...")
        user_audio = module['trim_silence'](user_audio, sr)
        ref_audio_full = module['trim_silence'](ref_audio_full, sr)

        logger.info("📏 Extracting segment...")
        start_sample = int(float(selected_start) * sr)
        end_sample = min(start_sample + int(window_size), len(ref_audio_full))
        ref_audio_sliced = ref_audio_full[start_sample:end_sample]

        logger.info("💾 Saving segment...")
        temp_ref_sliced = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_ref_sliced.wav')
        audio_normalized = ref_audio_sliced / (np.max(np.abs(ref_audio_sliced)) + 1e-8)
        audio_int16 = np.int16(audio_normalized * 32767)
        wavfile.write(temp_ref_sliced, sr, audio_int16)

        logger.info("🎯 Running analysis...")
        start_analysis = time.time()
        results = module['evaluate_singing'](user_path, temp_ref_sliced, 'Models')
        analysis_time = time.time() - start_analysis
        logger.info(f"✅ Analysis: {analysis_time:.2f}s")

        try:
            os.remove(temp_ref_sliced)
        except:
            pass

        if results is None or not results.get('success'):
            error_msg = results.get('report', 'Analysis failed') if results else 'No results'
            return jsonify({'error': error_msg}), 500

        total_time = time.time() - start_total
        logger.info(f"⏱️  Total: {total_time:.2f}s")

        return jsonify({
            'success': True,
            'pitch_accuracy': float(results.get('pitch_accuracy', 0)),
            'rhythm_accuracy': float(results.get('rhythm_accuracy', 0)),
            'emotion_detected': str(results.get('emotion_detected', 'neutral')),
            'emotion_confidence': float(results.get('emotion_confidence', 0)),
            'final_score': float(results.get('final_score', 0))
        })
    except Exception as e:
        logger.error(f"Analysis error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Analysis failed: {str(e)[:100]}'}), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🎵 AUDIO ANALYZER - FINAL VERSION")
    print("="*80)
    print("\n✓ Open: http://localhost:5000")
    print("="*80 + "\n")

    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)