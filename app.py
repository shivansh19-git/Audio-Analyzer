"""
🎵 AUDIO ANALYZER - MEMORY OPTIMIZED VERSION
For 512MB Render Free Tier
Optimizations:
1. Lower sample rate (11025 Hz for matching)
2. Aggressive garbage collection
3. Cache limit = 1 file only
4. Early cleanup of temp files
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
import gc
import glob
from collections import OrderedDict

# ============================================================================
# SETUP & CONFIGURATION
# ============================================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # OPTIMIZATION: Hard cap at 10 MB
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable aggressive garbage collection
gc.enable()

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a'}

# ============================================================================
# MEMORY OPTIMIZATION: AGGRESSIVE CACHING
# ============================================================================

_module_cache = None
_module_lock = threading.Lock()
_module_loading = False
_module_failed  = False

# OPTIMIZATION: Limit audio cache to 1 file only
_audio_cache = OrderedDict()
MAX_CACHE_SIZE = 1

def get_cached_audio(path, sr):
    """Cache only 1 file to minimize memory usage"""
    cache_key = f"{path}_{sr}"

    if cache_key in _audio_cache:
        logger.info(f"📦 Cache hit for {os.path.basename(path)}")
        return _audio_cache[cache_key]

    logger.info(f"📁 Loading audio: {os.path.basename(path)}")
    audio, _ = librosa.load(path, sr=sr)

    # Aggressively clear old cache to free memory
    if len(_audio_cache) >= MAX_CACHE_SIZE:
        removed_key = _audio_cache.popitem(last=False)
        logger.info(f"🗑️  Cache cleared: {removed_key[0]}")

    _audio_cache[cache_key] = audio
    return audio


# ============================================================================
# PERSISTENT MODULE LOADING
# ============================================================================

def load_notebook_once():
    """
    Load the pre-converted Analysis.py module ONCE and cache it.

    WHY PRE-CONVERTED:
    - Runtime nbconvert spawns a jupyter subprocess (~100MB RAM) PLUS holds
      the entire script as a Python string PLUS exec() loads all imports/models.
      All three overlap in memory → OOM kill on 512 MB Render free tier.
    - Pre-converting offline and committing Analysis.py eliminates the subprocess
      entirely. Loading is just a standard Python import: fast, predictable RAM.

    HOW TO GENERATE Analysis.py (run this ONCE on your local machine):
        jupyter nbconvert --to script Analysis.ipynb
    Then commit Analysis.py alongside app.py.
    """
    global _module_cache, _module_loading, _module_failed

    if _module_cache is not None:
        return _module_cache
    if _module_failed:
        return None

    with _module_lock:
        if _module_cache is not None:
            return _module_cache
        if _module_failed:
            return None

        _module_loading = True
        try:
            logger.info("🔄 Loading Analysis module...")
            start = time.time()
            gc.collect()

            # ── Method 1: import pre-converted Analysis.py (preferred) ──────
            analysis_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Analysis.py')
            if os.path.exists(analysis_py):
                logger.info("   Found Analysis.py — using direct import (no subprocess)")
                import importlib.util
                spec = importlib.util.spec_from_file_location("Analysis", analysis_py)
                mod = importlib.util.module_from_spec(spec)

                # Free everything we can before the heavy import
                gc.collect()
                spec.loader.exec_module(mod)
                gc.collect()

                # Expose as dict for compatibility with existing module['func'] calls
                module = {k: getattr(mod, k) for k in dir(mod) if not k.startswith('__')}

                # ── Diagnostic: log what functions were actually loaded ──────
                fn_names = [k for k, v in module.items() if callable(v) and not k.startswith('_')]
                logger.info(f"   Functions available in module: {fn_names}")

                # ── Check for the required functions ─────────────────────────
                required = ['find_best_matching_segment', 'evaluate_singing', 'trim_silence', 'validate_pitched_content']
                missing  = [f for f in required if f not in module]
                if missing:
                    logger.error(f"   ❌ Missing required functions: {missing}")
                    logger.error(f"   This usually means Analysis.py has IPython magic commands")
                    logger.error(f"   that failed silently. See fix instructions below.")
                    # Try stripping IPython artifacts and re-executing
                    logger.info("   🔧 Attempting to strip IPython artifacts and re-load...")
                    import re as _re
                    with open(analysis_py, 'r', encoding='utf-8', errors='replace') as _f:
                        raw = _f.read()
                    # Remove IPython magic lines and get_ipython() calls
                    cleaned = _re.sub(r'^get_ipython\(\).*$', '', raw, flags=_re.MULTILINE)
                    cleaned = _re.sub(r'^# In\[.*?\].*$', '', cleaned, flags=_re.MULTILINE)
                    cleaned = _re.sub(r'^#\s*coding:.*$', '', cleaned, flags=_re.MULTILINE)
                    module2 = {}
                    exec(compile(cleaned, 'Analysis.py', 'exec'), module2)
                    del cleaned, raw
                    gc.collect()
                    fn_names2 = [k for k, v in module2.items() if callable(v) and not k.startswith('_')]
                    logger.info(f"   After strip — functions available: {fn_names2}")
                    missing2 = [f for f in required if f not in module2]
                    if not missing2:
                        logger.info("   ✅ Strip successful — using cleaned module")
                        module = module2
                    else:
                        logger.error(f"   ❌ Still missing after strip: {missing2}")
                        logger.error(f"   Check that Analysis.ipynb actually defines: {missing2}")

            # ── Method 2: fallback — runtime nbconvert (last resort) ─────────
            else:
                logger.warning("   Analysis.py not found — falling back to nbconvert (high RAM!)")
                logger.warning("   Run: jupyter nbconvert --to script Analysis.ipynb")
                logger.warning("   and commit Analysis.py to fix OOM crashes.")

                tmp_script = os.path.join(tempfile.gettempdir(), '_analysis_converted.py')
                result = subprocess.run(
                    ['jupyter', 'nbconvert', '--to', 'script',
                     'Analysis.ipynb', f'--output={tmp_script[:-3]}'],
                    capture_output=True, text=True,
                    encoding='utf-8', errors='replace', timeout=180
                )
                if result.returncode != 0:
                    logger.error(f"nbconvert failed: {result.stderr[:300]}")
                    _module_failed = True
                    return None

                del result
                gc.collect()

                if not os.path.exists(tmp_script):
                    logger.error("Converted script not found after nbconvert")
                    _module_failed = True
                    return None

                with open(tmp_script, 'r', encoding='utf-8', errors='replace') as f:
                    script_code = f.read()
                try:
                    os.remove(tmp_script)
                except Exception:
                    pass

                module = {}
                exec(compile(script_code, 'Analysis.ipynb', 'exec'), module)
                del script_code
                gc.collect()

            elapsed = time.time() - start
            logger.info(f"✅ Analysis module loaded in {elapsed:.2f}s (cached)")
            _module_cache = module
            return module

        except MemoryError:
            logger.error("❌ OOM while loading Analysis module — 512 MB limit hit")
            _module_failed = True
            gc.collect()
            return None
        except Exception as e:
            logger.error(f"Error loading Analysis module: {e}")
            _module_failed = True
            return None
        finally:
            _module_loading = False


def prewarm_module():
    """Load the module eagerly so the first HTTP request never has to wait."""
    logger.info("🔥 Pre-warming Analysis module...")
    load_notebook_once()
    if _module_cache:
        logger.info("✅ Pre-warm complete — Analysis module ready")
    else:
        logger.error("❌ Pre-warm FAILED — check logs above for details")


# Start pre-warm as soon as this module is imported.
# daemon=True so it doesn't block server shutdown.
_prewarm_thread = threading.Thread(target=prewarm_module, daemon=True, name='prewarm')
_prewarm_thread.start()


def get_module():
    return load_notebook_once()


def is_module_ready():
    return _module_cache is not None


def is_module_loading():
    return _module_loading


def is_module_failed():
    return _module_failed


# ============================================================================
# HTML TEMPLATE (FINAL VERSION)
# ============================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title> Audio Analyzer</title>
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='75' font-size='75'>🎵</text></svg>">
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

        // ── Engine readiness check ──────────────────────────────────────────
        // Poll /api/ready so we never fire a request while the server is still
        // loading the notebook — which causes the OOM-truncated-JSON bug.
        let engineReady = false;

        async function pollUntilReady() {
            const btn = document.getElementById('analyzeBtn');
            btn.disabled = true;
            btn.textContent = '⏳ Engine warming up…';

            const maxAttempts = 40;   // 40 × 5s = 3 min 20 s max wait
            for (let i = 0; i < maxAttempts; i++) {
                try {
                    const res = await fetch('/api/ready');
                    const data = await res.json();

                    if (data.ready) {
                        engineReady = true;
                        btn.textContent = '🎵 Auto-Match & Analyze';
                        updateAnalyzeButton();   // re-apply file-presence check
                        showMessage('Analysis engine ready!', 'success');
                        return;
                    } else if (data.status === 'failed') {
                        btn.textContent = '❌ Engine failed to load';
                        showMessage('Analysis engine failed to load. Please redeploy.', 'error');
                        return;
                    } else {
                        const dots = '.'.repeat((i % 3) + 1);
                        btn.textContent = `⏳ Warming up${dots}`;
                        showMessage(`Warming up analysis engine… (${i + 1}/${maxAttempts})`, 'info');
                    }
                } catch (e) {
                    // network blip — keep trying
                }
                await new Promise(r => setTimeout(r, 5000));  // wait 5 s
            }

            btn.textContent = '⚠️ Timed out — refresh page';
            showMessage('Engine warm-up timed out. Please refresh.', 'error');
        }

        // Start polling immediately when page loads
        pollUntilReady();

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
            document.getElementById('analyzeBtn').disabled = !(hasUser && hasRef && engineReady);
        }

        document.getElementById('analyzeBtn').addEventListener('click', async () => {
            if (!engineReady) {
                showMessage('Engine is still warming up, please wait…', 'error');
                return;
            }
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
                        window_size: state.windowSize,
                        sr_match: state.sr   // FIX: needed to scale window_size to analysis sr
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


@app.route('/health', methods=['GET'])
def health():
    """Lightweight health check endpoint for uptime monitoring"""
    return jsonify({'status': 'ok', 'message': 'App is running'}), 200


@app.route('/api/ready', methods=['GET'])
def ready():
    """
    Tells the frontend whether the analysis engine has finished loading.
    The frontend polls this before sending files, so it never fires a request
    into a half-loaded server and gets truncated JSON back.
    """
    if is_module_ready():
        return jsonify({'ready': True, 'status': 'ready'}), 200
    elif is_module_failed():
        return jsonify({'ready': False, 'status': 'failed',
                        'message': 'Analysis engine failed to load. Please redeploy.'}), 503
    else:
        return jsonify({'ready': False, 'status': 'loading',
                        'message': 'Analysis engine is warming up, please wait…'}), 503


@app.route('/api/upload', methods=['POST'])
def upload_audio():
    try:
        if 'user_audio' not in request.files or 'ref_audio' not in request.files:
            return jsonify({'error': 'Missing audio files'}), 400

        user_file = request.files['user_audio']
        ref_file = request.files['ref_audio']

        if not user_file or not ref_file or not allowed_file(user_file.filename) or not allowed_file(ref_file.filename):
            return jsonify({'error': 'Invalid files'}), 400

        # OPTIMIZATION: Reject oversized files before they consume memory
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB max per file
        user_file.seek(0, 2)   # Seek to end
        user_size = user_file.tell()
        user_file.seek(0)      # Reset to start

        ref_file.seek(0, 2)
        ref_size = ref_file.tell()
        ref_file.seek(0)

        if user_size > MAX_FILE_SIZE:
            logger.warning(f"⚠️  User file too large: {user_size / 1024 / 1024:.1f} MB")
            return jsonify({'error': f'User audio too large ({user_size / 1024 / 1024:.1f} MB). Max is 10 MB.'}), 400

        if ref_size > MAX_FILE_SIZE:
            logger.warning(f"⚠️  Reference file too large: {ref_size / 1024 / 1024:.1f} MB")
            return jsonify({'error': f'Reference audio too large ({ref_size / 1024 / 1024:.1f} MB). Max is 10 MB.'}), 400

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

        # OPTIMIZATION 2: Use LOWER sample rate for matching to save memory
        sr = 11025  # Half the size of 22050, still accurate enough

        logger.info("📁 Loading audio (11025 Hz - optimized)...")
        user_audio = get_cached_audio(user_path, sr)
        ref_audio_full = get_cached_audio(ref_path, sr)

        logger.info("🎵 Preprocessing...")
        user_audio = module['trim_silence'](user_audio, sr)
        ref_audio_full = module['trim_silence'](ref_audio_full, sr)

        has_content_user, _ = module['validate_pitched_content'](user_audio)
        has_content_ref, _ = module['validate_pitched_content'](ref_audio_full)

        if not has_content_user or not has_content_ref:
            return jsonify({'error': 'No voice content detected'}), 400

        logger.info("🔍 Auto-matching (chunked DTW)...")
        start_match = time.time()

        # ── Chunked DTW ────────────────────────────────────────────────────────
        # ROOT CAUSE OF OOM: passing the full 181s reference to find_best_matching_segment
        # at once forces DTW to hold a [user_frames × ref_frames] matrix for the
        # entire reference — easily 200-400 MB for long songs on 512 MB Render.
        #
        # FIX: downsample both signals, then slide a 30s chunk window across the
        # reference with 5s overlap.  DTW runs on one small chunk at a time;
        # peak RAM is bounded by chunk size, not total reference length.
        #
        # Memory budget per chunk (30s ref, 14s user, ~2756 Hz after 4x DS):
        #   user_frames  ≈ 14 * 2756 / 512 ≈ 75 MFCC frames
        #   chunk_frames ≈ 30 * 2756 / 512 ≈ 161 MFCC frames
        #   DTW matrix   ≈ 75 × 161 × 8 B  ≈ 96 KB  ✅ (vs ~300 MB for full ref)

        DOWNSAMPLE_FACTOR = 4
        sr_ds = sr // DOWNSAMPLE_FACTOR          # ~2756 Hz

        user_audio_ds = user_audio[::DOWNSAMPLE_FACTOR]

        CHUNK_SECS    = 30                        # seconds of ref per DTW call
        OVERLAP_SECS  = 5                         # overlap so we don't miss a boundary match
        chunk_samples = CHUNK_SECS  * sr_ds
        step_samples  = (CHUNK_SECS - OVERLAP_SECS) * sr_ds

        ref_ds = ref_audio_full[::DOWNSAMPLE_FACTOR]
        ref_duration = len(ref_audio_full) / sr

        best_confidence  = -1.0
        best_timestamp   = 0.0
        best_matched_seg = user_audio_ds          # fallback

        num_chunks = max(1, int(np.ceil((len(ref_ds) - chunk_samples) / step_samples)) + 1)
        logger.info(
            f"   user={len(user_audio_ds):,} smp  ref={len(ref_ds):,} smp  "
            f"chunk={CHUNK_SECS}s  chunks={num_chunks}"
        )

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * step_samples
            chunk_end   = min(chunk_start + chunk_samples, len(ref_ds))
            ref_chunk   = ref_ds[chunk_start:chunk_end]

            # Skip chunks shorter than the user audio (can't match)
            if len(ref_chunk) < len(user_audio_ds):
                continue

            try:
                seg, ts_in_chunk, conf = module['find_best_matching_segment'](
                    user_audio_ds, ref_chunk, sr_ds
                )
            except Exception as chunk_err:
                logger.warning(f"   Chunk {chunk_idx} failed: {chunk_err}")
                continue

            # ts_in_chunk is seconds from start of this chunk — offset to full ref
            chunk_offset_secs = chunk_start / sr_ds
            ts_global = ts_in_chunk + chunk_offset_secs

            logger.info(
                f"   Chunk {chunk_idx+1}/{num_chunks}: "
                f"offset={chunk_offset_secs:.1f}s  ts={ts_global:.2f}s  conf={conf:.3f}"
            )

            if conf > best_confidence:
                best_confidence  = conf
                best_timestamp   = ts_global
                best_matched_seg = seg

            # Free chunk from memory immediately
            del ref_chunk, seg
            gc.collect()

        # Free downsampled ref — no longer needed
        del ref_ds
        gc.collect()

        timestamp  = float(best_timestamp)
        confidence = float(best_confidence)
        # Scale window_size back from sr_ds domain to sr (11025 Hz) domain
        window_size = len(best_matched_seg) * DOWNSAMPLE_FACTOR

        match_time = time.time() - start_match
        logger.info(
            f"✅ Best match: t={timestamp:.2f}s  confidence={confidence:.3f}  "
            f"window={window_size} smp ({window_size/sr:.2f}s)  took {match_time:.2f}s"
        )

        total_time = time.time() - start_total
        logger.info(f"⏱️  Total: {total_time:.2f}s")

        # Final cleanup
        del user_audio_ds, best_matched_seg
        gc.collect()

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
        gc.collect()
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
        # sr_match = sr used during auto-match (11025). window_size is in those samples.
        # We load audio at sr=22050 here, so we must scale window_size proportionally.
        sr_match = int(data.get('sr_match', 11025))

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
        # window_size is in sr_match (11025 Hz) samples; scale to analysis sr (22050 Hz)
        scaled_window = int(window_size * sr / sr_match)
        end_sample = min(start_sample + scaled_window, len(ref_audio_full))
        logger.info(f"   window_size={window_size} @ {sr_match}Hz  →  {scaled_window} samples @ {sr}Hz")
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
            logger.info("🗑️  Temp file deleted")
        except Exception as e:
            logger.error(f"Failed to delete temp: {e}")

        # OPTIMIZATION 3: Clean ALL old temp files after analysis
        for old_file in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], 'temp_*')):
            try:
                os.remove(old_file)
                logger.info(f"🗑️  Cleaned: {os.path.basename(old_file)}")
            except:
                pass

        # Clear cache and force garbage collection
        _audio_cache.clear()
        gc.collect()

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
        gc.collect()
        logger.error(f"Analysis error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Analysis failed: {str(e)[:100]}'}), 500


# ============================================================================
# CLEANUP & STARTUP
# ============================================================================

def cleanup_temp_files():
    """Aggressive cleanup on startup to free memory"""
    logger.info("🧹 Starting aggressive cleanup...")

    # Clear cache
    _audio_cache.clear()
    gc.collect()

    cleaned = 0
    try:
        temp_files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], 'temp_*'))

        for old_file in temp_files:
            try:
                if os.path.exists(old_file):
                    os.remove(old_file)
                    logger.info(f"✅ Deleted: {os.path.basename(old_file)}")
                    cleaned += 1
            except Exception as e:
                logger.warning(f"⚠️  Could not delete {os.path.basename(old_file)}: {e}")

        gc.collect()
        logger.info(f"✅ Cleanup complete - {cleaned} temp files removed")

    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")


if __name__ == '__main__':
    cleanup_temp_files()

    print("\n" + "="*80)
    print("🎵 AUDIO ANALYZER - MEMORY OPTIMIZED VERSION")
    print("="*80)
    print("\n✅ OPTIMIZATIONS ENABLED:")
    print("   ✓ Lower sample rate for matching (11025 Hz)")
    print("   ✓ DTW downsampling 4x  --  ~16x smaller matrix, ~16x less RAM")
    print("   ✓ File size hard cap (10 MB per file, rejected at upload)")
    print("   ✓ MAX_CONTENT_LENGTH reduced to 10 MB (Flask-level guard)")
    print("   ✓ Aggressive garbage collection")
    print("   ✓ Cache limit = 1 file only")
    print("   ✓ Temp file cleanup after analysis")
    print("   ✓ Health endpoint for uptime monitoring")
    print("\n✓ Open: http://localhost:5000")
    print("✓ Health: http://localhost:5000/health")
    print("="*80 + "\n")

    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)