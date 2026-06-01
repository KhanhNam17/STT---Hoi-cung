import streamlit as st

st.set_page_config(
    page_title="Smart Meeting — Trợ lý Cuộc họp",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    footer { visibility: hidden; }

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        background: #e8520a !important;
        border: none !important;
        font-weight: 600 !important;
        color: white !important;
        letter-spacing: 0.5px;
    }
    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button[kind="primary"]:hover {
        background: #c44008 !important;
    }

    .stDownloadButton > button:not([kind="primary"]) {
        background: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        font-weight: 600 !important;
        border: 1px solid rgba(128,128,128,0.2) !important;
    }

    .stProgress > div > div > div {
        background: #e8520a !important;
    }

    hr { border-color: rgba(128,128,128,0.2) !important; }
    
    .feature-card {
        background: var(--background-color);
        border: 1px solid rgba(128,128,128,0.2);
        border-top: 2px solid #e8520a;
        border-radius: 6px;
        padding: 36px 32px 28px;
        margin-bottom: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:16px 0 24px;">
<div style="font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:3px;text-transform:uppercase;color:#e8520a;margin-bottom:8px;">// SMART MEETING</div>
<div style="font-size:18px;font-weight:700;color:var(--text-color);letter-spacing:-0.3px;">Trợ lý Cuộc họp</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--text-color);opacity:0.6;margin-top:6px;">STT · DIARIZATION · DOCX</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    st.page_link("app.py",               label="🏠  Trang chủ")
    st.page_link("pages/2_Live_Mode.py", label="🎙️  Smart Meeting (Live)")

    st.divider()

    st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--text-color);line-height:2.2;">
<div>RUNTIME &nbsp;&nbsp;<span style="color:#2e8b2e;font-weight:bold;">● OFFLINE</span></div>
<div style="opacity:0.7;">STT &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Zipformer (streaming)</div>
<div style="opacity:0.7;">DIARIZE &nbsp;&nbsp;NeMo Sortformer</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="max-width:780px;margin:64px auto 0;padding:0 8px;">
<div style="font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:4px;text-transform:uppercase;color:#e8520a;margin-bottom:20px;display:flex;align-items:center;gap:10px;">
<span style="display:inline-block;width:28px;height:1px;background:#e8520a;"></span>
SMART MEETING ASSISTANT
</div>

<h1 style="font-size:36px;font-weight:700;color:var(--text-color);letter-spacing:-0.8px;line-height:1.2;margin:0 0 16px;">
Trợ lý Cuộc họp<br>
<span style="color:#e8520a;">Phiên âm & Phân tách Người nói</span>
</h1>

<p style="font-size:14px;color:var(--text-color);opacity:0.7;line-height:1.9;margin:0 0 52px;max-width:520px;">
Ghi âm cuộc họp → phiên âm realtime → phân tách người nói (Sortformer) → tóm tắt → xuất biên bản DOCX.<br>
Chạy hoàn toàn cục bộ, không cần internet sau khi cài đặt.
</p>
</div>
""", unsafe_allow_html=True)

# ── Single feature card ──────────────────────────────────────────────────────
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.markdown("""
<div class="feature-card">
<div style="display:flex;align-items:flex-start;gap:18px;margin-bottom:28px;">
<div style="font-size:28px;line-height:1;margin-top:2px;">🎙️</div>
<div>
<div style="font-size:17px;font-weight:700;color:var(--text-color);margin-bottom:5px;">Smart Meeting — Ghi âm Cuộc họp Trực tiếp</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:2px;color:#e8520a;text-transform:uppercase;">Live Mode</div>
</div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px 24px;margin-bottom:28px;flex-grow:1;">
<div style="font-size:12px;color:var(--text-color);opacity:0.8;display:flex;align-items:center;gap:7px;">
<span style="color:#e8520a;font-size:10px;">▸</span> Ghi âm realtime từ micro
</div>
<div style="font-size:12px;color:var(--text-color);opacity:0.8;display:flex;align-items:center;gap:7px;">
<span style="color:#e8520a;font-size:10px;">▸</span> Phiên âm tiếng Việt streaming
</div>
<div style="font-size:12px;color:var(--text-color);opacity:0.8;display:flex;align-items:center;gap:7px;">
<span style="color:#e8520a;font-size:10px;">▸</span> Phân tách người nói (Sortformer)
</div>
<div style="font-size:12px;color:var(--text-color);opacity:0.8;display:flex;align-items:center;gap:7px;">
<span style="color:#e8520a;font-size:10px;">▸</span> Tóm tắt nội dung (Qwen NPU)
</div>
<div style="font-size:12px;color:var(--text-color);opacity:0.8;display:flex;align-items:center;gap:7px;">
<span style="color:#e8520a;font-size:10px;">▸</span> Tạm dừng / tiếp tục giữa cuộc họp
</div>
<div style="font-size:12px;color:var(--text-color);opacity:0.8;display:flex;align-items:center;gap:7px;">
<span style="color:#e8520a;font-size:10px;">▸</span> Xuất biên bản DOCX
</div>
</div>
<div style="display:flex;gap:6px;flex-wrap:wrap;padding-top:18px;border-top:1px solid rgba(128,128,128,0.2);">
<span style="font-family:'IBM Plex Mono',monospace;font-size:10px;padding:3px 10px;border-radius:3px;background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);color:var(--text-color);opacity:0.8;">.mic</span>
<span style="font-family:'IBM Plex Mono',monospace;font-size:10px;padding:3px 10px;border-radius:3px;background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);color:var(--text-color);opacity:0.8;">realtime</span>
<span style="font-family:'IBM Plex Mono',monospace;font-size:10px;padding:3px 10px;border-radius:3px;background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);color:var(--text-color);opacity:0.8;">zipformer</span>
<span style="font-family:'IBM Plex Mono',monospace;font-size:10px;padding:3px 10px;border-radius:3px;background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);color:var(--text-color);opacity:0.8;">sortformer</span>
<span style="font-family:'IBM Plex Mono',monospace;font-size:10px;padding:3px 10px;border-radius:3px;background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);color:var(--text-color);opacity:0.8;">qwen</span>
</div>
</div>
""", unsafe_allow_html=True)

    st.page_link("pages/2_Live_Mode.py", label="🎙️  Bắt đầu cuộc họp", use_container_width=True)

# ── Status bar ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="max-width:780px;margin:40px auto 0;padding:0 8px;">
<div style="display:flex;gap:32px;padding:14px 20px;background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);border-radius:4px;">
<div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--text-color);">
<span style="opacity:0.7;">STATUS</span> &nbsp;<span style="color:#2e8b2e;font-weight:bold;">● READY</span>
</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--text-color);">
<span style="opacity:0.7;">MODE</span> &nbsp;<span style="opacity:0.9;font-weight:bold;">OFFLINE / LOCAL</span>
</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--text-color);">
<span style="opacity:0.7;">ENGINE</span> &nbsp;<span style="opacity:0.9;font-weight:bold;">Zipformer · NeMo Sortformer · Qwen NPU (tóm tắt)</span>
</div>
</div>
</div>
""", unsafe_allow_html=True)