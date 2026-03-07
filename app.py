"""
AUDIO ANALYZER - Single Page Aesthetic Design
Compact, professional, with animations
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import os
import time
import subprocess
import warnings
import tempfile

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Audio Analyzer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# ADVANCED CSS STYLING - AESTHETIC & COMPACT
# ============================================================================

custom_css = """
<style>
    /* Variables */
    :root {
        --primary: #0a0e27;
        --secondary: #111a3a;
        --tertiary: #1a2d5c;
        --accent-cyan: #00d4ff;
        --accent-pink: #ff006e;
        --accent-purple: #7b2cbf;
        --accent-blue: #3a86ff;
        --accent-gold: #ffa500;
        --text-primary: #ffffff;
        --text-secondary: #a0a0b0;
        --success: #00ff00;
        --border: rgba(0, 212, 255, 0.1);
    }

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #111a3a 50%, #1a1a3e 100%);
        background-attachment: fixed;
        color: var(--text-primary);
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
    }

    [data-testid="stAppViewContainer"] { background: transparent !important; }
    .stApp { background: transparent !important; }
    .main { background: transparent !important; padding: 0 !important; }

    /* ======== TOP TOOLBAR ======== */
    [data-testid="stAppViewContainer"] header {
        background: linear-gradient(90deg, rgba(10, 14, 39, 0.95) 0%, rgba(17, 26, 58, 0.95) 100%) !important;
        border-bottom: 1px solid var(--border) !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Hide default header elements */
    [data-testid="stToolbar"] { display: none !important; }

    /* Custom toolbar */
    .toolbar-container {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 70px;
        background: linear-gradient(90deg, rgba(10, 14, 39, 0.98) 0%, rgba(17, 26, 58, 0.98) 100%);
        border-bottom: 1px solid var(--border);
        backdrop-filter: blur(20px);
        z-index: 999;
        display: flex;
        align-items: center;
        padding: 0 40px;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.05);
    }

    .toolbar-logo {
        font-size: 24px;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-pink) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .toolbar-logo::before {
        content: "🎵";
        font-size: 28px;
    }

    .toolbar-spacer { flex: 1; }

    .toolbar-info {
        display: flex;
        gap: 30px;
        align-items: center;
    }

    .toolbar-stat {
        text-align: right;
        font-size: 12px;
    }

    .toolbar-stat-value {
        font-size: 18px;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-blue) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .toolbar-stat-label {
        color: var(--text-secondary);
        margin-top: 2px;
    }

    /* ======== MAIN CONTENT ======== */
    .main-container {
        margin-top: 70px;
        padding: 40px;
        max-width: 1200px;
        margin-left: auto;
        margin-right: auto;
    }

    /* ======== HEADINGS ======== */
    h1, h2, h3, h4, h5, h6 {
        letter-spacing: 0.5px;
    }

    h1 {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-pink) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 !important;
        padding: 0 !important;
    }

    h2 {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 !important;
        padding: 0 !important;
    }

    h3 {
        font-size: 18px;
        font-weight: 600;
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-blue) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 !important;
        padding: 0 !important;
    }

    .subtitle {
        color: var(--text-secondary);
        font-size: 16px;
        margin-top: 12px !important;
        line-height: 1.5;
    }

    /* ======== LAYOUT SECTIONS ======== */
    .section-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 20px;
    }

    .section-icon {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(58, 134, 255, 0.1) 100%);
        border: 1px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }

    /* ======== UPLOAD AREA ======== */
    .upload-container {
        background: linear-gradient(135deg, rgba(17, 26, 58, 0.5) 0%, rgba(26, 45, 92, 0.5) 100%);
        border: 2px solid var(--border);
        border-radius: 16px;
        padding: 28px;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }

    .upload-container:hover {
        border-color: rgba(0, 212, 255, 0.3);
        background: linear-gradient(135deg, rgba(17, 26, 58, 0.7) 0%, rgba(26, 45, 92, 0.7) 100%);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    }

    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed var(--accent-cyan) !important;
        border-radius: 12px !important;
        background: transparent !important;
    }

    /* ======== BUTTONS ======== */
    .stButton > button {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-blue) 100%) !important;
        color: var(--primary) !important;
        border: none !important;
        padding: 16px 42px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2) !important;
        cursor: pointer !important;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s ease;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.4) !important;
        letter-spacing: 1px !important;
    }

    .stButton > button:active {
        transform: translateY(-1px) !important;
    }

    /* ======== ANALYSIS BUTTON SPECIAL ======== */
    .analyze-btn {
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-pink) 100%) !important;
        font-size: 16px !important;
        padding: 20px 50px !important;
        min-width: 280px;
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.3) !important;
    }

    .analyze-btn:hover {
        transform: translateY(-6px) scale(1.02) !important;
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.5) !important;
        letter-spacing: 1.5px !important;
    }

    /* ======== CARDS & METRICS ======== */
    .metric-card {
        background: linear-gradient(135deg, rgba(17, 26, 58, 0.6) 0%, rgba(26, 45, 92, 0.6) 100%);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 24px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: rgba(0, 212, 255, 0.3);
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    }

    .metric-value {
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-pink) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 12px 0;
    }

    .metric-label {
        color: var(--text-secondary);
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ======== STATUS INDICATORS ======== */
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .status-success {
        background: rgba(0, 255, 0, 0.1);
        color: var(--success);
        border: 1px solid rgba(0, 255, 0, 0.2);
    }

    .status-pending {
        background: rgba(255, 165, 0, 0.1);
        color: var(--accent-gold);
        border: 1px solid rgba(255, 165, 0, 0.2);
    }

    /* ======== PROGRESS BAR ======== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent-cyan) 0%, var(--accent-pink) 100%) !important;
        border-radius: 10px !important;
    }

    /* ======== DIVIDER ======== */
    .divider {
        height: 1px;
        background: var(--border);
        margin: 20px 0;
    }

    /* ======== TEXT & LISTS ======== */
    p {
        color: var(--text-secondary);
        line-height: 1.6;
        margin: 0 !important;
    }

    /* ======== AUDIO PLAYER ======== */
    audio {
        width: 100%;
        border-radius: 8px;
        margin: 12px 0;
    }

    /* ======== GRID LAYOUT ======== */
    .compact-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
    }

    /* ======== ANIMATION KEYFRAMES ======== */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.1); }
        50% { box-shadow: 0 0 40px rgba(0, 212, 255, 0.2); }
    }

    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }

    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }

    .glowing {
        animation: glow 2s ease-in-out infinite;
    }

    /* ======== RESPONSIVE ======== */
    @media (max-width: 768px) {
        .toolbar-container {
            padding: 0 20px;
            height: 60px;
        }

        .toolbar-info {
            display: none;
        }

        .main-container {
            margin-top: 60px;
            padding: 20px;
        }

        h1 { font-size: 32px; }
        h2 { font-size: 22px; }
        h3 { font-size: 16px; }

        .compact-grid {
            grid-template-columns: 1fr;
        }

        .analyze-btn {
            min-width: 100% !important;
            padding: 16px 30px !important;
        }
    }

    /* ======== SCROLLBAR ======== */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(0, 212, 255, 0.05);
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-cyan) 0%, var(--accent-blue) 100%);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--accent-pink) 0%, var(--accent-cyan) 100%);
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

if 'results' not in st.session_state:
    st.session_state.results = None
if 'user_file' not in st.session_state:
    st.session_state.user_file = None
if 'ref_file' not in st.session_state:
    st.session_state.ref_file = None
if 'show_results' not in st.session_state:
    st.session_state.show_results = False


# ============================================================================
# CUSTOM TOOLBAR
# ============================================================================

def render_toolbar():
    """Render top toolbar"""
    st.markdown("""
    <div class="toolbar-container">
        <div class="toolbar-logo">Audio Analyzer</div>
        <div class="toolbar-spacer"></div>
        <div class="toolbar-info">
            <div class="toolbar-stat">
                <div class="toolbar-stat-value">✓ Ready</div>
                <div class="toolbar-stat-label">System Status</div>
            </div>
            <div class="toolbar-stat">
                <div class="toolbar-stat-value">v1.0</div>
                <div class="toolbar-stat-label">Version</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# LOAD NOTEBOOK
# ============================================================================

@st.cache_resource
def load_notebook():
    """Load Analysis.ipynb using nbconvert"""
    try:
        if not os.path.exists("Analysis.ipynb"):
            st.error("Analysis.ipynb not found")
            return None

        with tempfile.TemporaryDirectory() as tmpdir:

            result = subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "Analysis.ipynb",
                    "--output",
                    "analysis_temp",
                    "--output-dir",
                    tmpdir,
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                st.error(result.stderr)
                return None

            # find the generated file
            files = os.listdir(tmpdir)

            script_file = None
            for f in files:
                if f.startswith("analysis_temp"):
                    script_file = os.path.join(tmpdir, f)
                    break

            if script_file is None:
                st.error("Converted script not found")
                return None

            with open(script_file, "r", encoding="utf-8") as f:
                script = f.read()

            namespace = {}
            exec(script, namespace)

            return namespace

    except Exception as e:
        st.error(f"Notebook loading error: {e}")
        return None


def run_analysis(user_path, ref_path):
    try:
        st.write("Loading notebook...")

        module = load_notebook()

        if not module:
            st.error("Notebook failed to load")
            return None

        if 'evaluate_singing' not in module:
            st.error("evaluate_singing function not found")
            return None

        st.write("Running evaluation...")

        results = module['evaluate_singing'](user_path, ref_path, 'Models')

        st.write("Evaluation finished")

        return results

    except Exception as e:
        st.error(f"Error: {e}")
        return None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_emotion_emoji(emotion):
    return {'Happy': '😊', 'Sad': '😢', 'Neutral': '😐', 'Energetic': '⚡'}.get(emotion, '🎵')


def get_grade(score):
    if score >= 90:
        return "A+", "#00ff00"
    elif score >= 80:
        return "A", "#00ff00"
    elif score >= 70:
        return "B", "#ffdd00"
    elif score >= 60:
        return "C", "#ffaa00"
    elif score >= 50:
        return "D", "#ff6600"
    else:
        return "F", "#ff0000"


def create_gauge_chart(value, title):
    """Create compact gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00ff00" if value >= 80 else "#ffdd00" if value >= 60 else "#ff0000"},
            'steps': [
                {'range': [0, 50], 'color': "rgba(255, 0, 0, 0.1)"},
                {'range': [50, 80], 'color': "rgba(255, 200, 0, 0.1)"},
                {'range': [80, 100], 'color': "rgba(0, 255, 0, 0.1)"}
            ]
        }
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff', size=12),
        height=280,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


# ============================================================================
# MAIN PAGE
# ============================================================================

render_toolbar()

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header section - Compact
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h1 style="margin: 0;">Audio Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Real-time singing performance evaluation with AI-powered analysis</p>',
                unsafe_allow_html=True)

with col2:
    if st.session_state.show_results and st.session_state.results:
        st.markdown("""
        <div style="text-align: right;">
            <div class="status-badge status-success">✓ Analysis Complete</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Main two-column layout - Compact
col_upload, col_results = st.columns([1, 1], gap="large")

# ============================================================================
# LEFT COLUMN - UPLOAD
# ============================================================================

with col_upload:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📤</div>
        <h3>Upload Audio</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-container">
        <p style="margin-bottom: 16px; color: #a0a0b0; font-size: 13px;">User's Singing Audio</p>
    </div>
    """, unsafe_allow_html=True)

    user_file = st.file_uploader(
        "User audio",
        type=['mp3', 'wav', 'ogg', 'm4a'],
        key='user',
        label_visibility="collapsed"
    )

    if user_file:
        st.session_state.user_file = user_file
        st.audio(user_file, format='audio/mp3')
        st.markdown(f'<div class="status-badge status-success" style="margin-top: 8px;">✓ {user_file.name}</div>',
                    unsafe_allow_html=True)

    st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-container">
        <p style="margin-bottom: 16px; color: #a0a0b0; font-size: 13px;">Reference Song Audio</p>
    </div>
    """, unsafe_allow_html=True)

    ref_file = st.file_uploader(
        "Reference audio",
        type=['mp3', 'wav', 'ogg', 'm4a'],
        key='ref',
        label_visibility="collapsed"
    )

    if ref_file:
        st.session_state.ref_file = ref_file
        st.audio(ref_file, format='audio/mp3')
        st.markdown(f'<div class="status-badge status-success" style="margin-top: 8px;">✓ {ref_file.name}</div>',
                    unsafe_allow_html=True)

    # Analyze button
    st.markdown('<div style="margin: 28px 0;"></div>', unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([0.5, 1, 0.5])
    with col_btn2:
        if st.session_state.user_file and st.session_state.ref_file:
            if st.button("▶ Analyze Audio", key="analyze", use_container_width=True):

                progress = st.progress(0)
                status = st.empty()

                user_path = "temp_user.mp3"
                ref_path = "temp_ref.mp3"

                # STEP 1 — Saving files
                status.text("Saving uploaded files...")
                progress.progress(10)

                with open(user_path, "wb") as f:
                    f.write(st.session_state.user_file.getbuffer())

                with open(ref_path, "wb") as f:
                    f.write(st.session_state.ref_file.getbuffer())

                time.sleep(0.3)

                # STEP 2 — Loading analysis engine
                status.text("Loading analysis engine...")
                progress.progress(25)

                module = load_notebook()

                if not module or 'evaluate_singing' not in module:
                    st.error("Analysis engine failed to load")
                    progress.empty()
                    st.stop()

                time.sleep(0.3)

                # STEP 3 — Running evaluation
                status.text("Running singing evaluation...")
                progress.progress(60)

                with st.spinner("Analyzing pitch, rhythm and emotion..."):
                    results = module['evaluate_singing'](user_path, ref_path, 'Models')

                time.sleep(0.3)

                # STEP 4 — Finalizing
                status.text("Finalizing results...")
                progress.progress(90)

                try:
                    os.remove(user_path)
                    os.remove(ref_path)
                except:
                    pass

                progress.progress(100)
                status.text("Analysis complete!")

                if results and results.get('success'):
                    st.session_state.results = results
                    st.session_state.show_results = True
                    st.success("✓ Analysis Complete!")
                    st.rerun()
        else:
            st.button("▶ Analyze Audio", key="analyze", use_container_width=True, disabled=True)

# ============================================================================
# RIGHT COLUMN - RESULTS
# ============================================================================

with col_results:
    if st.session_state.show_results and st.session_state.results:
        r = st.session_state.results

        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📊</div>
            <h3>Performance Results</h3>
        </div>
        """, unsafe_allow_html=True)

        # Overall Score - Prominent
        grade, color = get_grade(r['final_score'])

        st.markdown(f"""
        <div class="metric-card" style="text-align: center; padding: 32px 24px;">
            <div style="font-size: 14px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px;">Overall Score</div>
            <div class="metric-value">{r['final_score']:.1f}</div>
            <div style="font-size: 12px; background: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2); color: {color}; padding: 6px 12px; border-radius: 6px; display: inline-block; margin-top: 8px;">{grade}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="margin: 16px 0;"></div>', unsafe_allow_html=True)

        # Compact metrics
        st.markdown(
            '<h3 style="font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-secondary);">Accuracy Metrics</h3>',
            unsafe_allow_html=True)

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.plotly_chart(create_gauge_chart(r['pitch_accuracy'], "Pitch"), use_container_width=True)

        with col_m2:
            st.plotly_chart(create_gauge_chart(r['rhythm_accuracy'], "Rhythm"), use_container_width=True)

        st.markdown('<div style="margin: 16px 0;"></div>', unsafe_allow_html=True)

        # Emotion detection
        st.markdown(
            '<h3 style="font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-secondary);">Emotion Detection</h3>',
            unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="text-align: center; padding: 24px;">
            <div style="font-size: 40px; margin-bottom: 8px;">{get_emotion_emoji(r['emotion_detected'])}</div>
            <div style="font-size: 18px; font-weight: 600; color: var(--text-primary);">{r['emotion_detected']}</div>
            <div style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">{r['emotion_confidence'] * 100:.0f}% Confidence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="margin: 16px 0;"></div>', unsafe_allow_html=True)

        # Raw results
        st.markdown(
            '<h3 style="font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-secondary);">Raw Data</h3>',
            unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="font-size: 13px;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                <div><span style="color: var(--text-secondary);">Pitch:</span> <span style="color: #00d4ff; font-weight: 600;">{r['pitch_accuracy']:.1f}%</span></div>
                <div><span style="color: var(--text-secondary);">Rhythm:</span> <span style="color: #00d4ff; font-weight: 600;">{r['rhythm_accuracy']:.1f}%</span></div>
                <div><span style="color: var(--text-secondary);">Emotion:</span> <span style="color: #00d4ff; font-weight: 600;">{r['emotion_detected']}</span></div>
                <div><span style="color: var(--text-secondary);">Final:</span> <span style="color: #00d4ff; font-weight: 600;">{r['final_score']:.1f}/100</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)

        col_reset1, col_reset2, col_reset3 = st.columns([0.5, 1, 0.5])
        with col_reset2:
            if st.button("🔄 New Analysis", use_container_width=True):
                st.session_state.results = None
                st.session_state.show_results = False
                st.session_state.user_file = None
                st.session_state.ref_file = None
                st.rerun()

    else:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📊</div>
            <h3>Performance Results</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="text-align: center; padding: 40px 24px; border: 2px dashed rgba(0, 212, 255, 0.2);">
            <div style="font-size: 32px; margin-bottom: 12px;">📈</div>
            <div style="color: var(--text-secondary); font-size: 14px;">Upload audio files and click Analyze<br/>to see detailed performance metrics</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)