# app.py — GEO Drift Tracker Streamlit App
# RULE: st.set_page_config must be the very first Streamlit call

import streamlit as st

st.set_page_config(
    page_title="GEO Drift Tracker",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

import streamlit.components.v1 as components

# ── All other imports AFTER set_page_config ──
import os
import sys
import datetime
import re
from html import escape as html_escape
from textwrap import dedent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Load API keys from Streamlit secrets if deployed, else from .env
try:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    os.environ["GROQ_API_KEY"]   = st.secrets["GROQ_API_KEY"]
except Exception:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

# Add src/ to path if project uses that layout
sys.path.insert(0, os.path.dirname(__file__))


def is_plausible_brand(name: str) -> bool:
    """Filter out generic labels that are not actual brand/product names."""
    value = str(name).strip()
    if not value:
        return False
    if value.lower().startswith(("best ", "top ", "free ", "cheap ", "better ", "candidate ")):
        return False
    normalized = re.sub(r"[^A-Za-z0-9]+", "", value).lower()
    if not normalized:
        return False
    if normalized in {
        "see",
        "real",
        "explore",
        "reviewed",
        "discover",
        "startup",
        "startups",
        "tools",
        "software",
        "bestcrm",
        "bestcrmsoftware",
    }:
        return False
    if normalized in {
        "better",
        "candidate",
        "devex",
        "growth",
        "accelerate",
        "software",
        "platform",
        "platforms",
        "tools",
        "product",
        "products",
        "crmsoftware",
        "bestcrm",
        "bestcrmsoftware",
    }:
        return False
    if len(normalized) >= 10 and sum(ch.isdigit() for ch in normalized) >= 4:
        return False
    return True

# ── CSS ──────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Raleway:wght@600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: #f1f2f8 !important;
}
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #0a0b14 !important;
}
.stApp {
    background-color: #0a0b14 !important;
}
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section.main > div {
    padding: 0 !important;
}
.element-container {
    margin-bottom: 0 !important;
    padding: 0 !important;
}
.stVerticalBlock,
div[data-testid="stVerticalBlock"] {
    gap: 0rem !important;
}
.stColumns {
    gap: 12px !important;
}
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="collapseSidebar"],
section[data-testid="stSidebar"],
#MainMenu, footer, header {
    display: none !important;
}
iframe {
    display: block !important;
    border: none !important;
    background: #0a0b14 !important;
}
[data-testid="stIframe"] {
    background: #0a0b14 !important;
}
.stMarkdown p, .stMarkdown div, .stMarkdown span {
    color: #f1f2f8 !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton {
    display: flex !important;
    justify-content: center !important;
}
.stButton > button {
    min-width: 160px !important;
    padding: 13px 32px !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    letter-spacing: 0.01em !important;
    width: auto !important;
}
.stButton > button[kind="primary"] {
    background-color: #6366f1 !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #818cf8 !important;
}
.stButton > button[kind="secondary"] {
    background-color: transparent !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #9294a8 !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: rgba(255,255,255,0.3) !important;
    color: #f1f2f8 !important;
}
[data-testid="stButton"] button[kind="primary"] {
    background: #6366f1 !important;
    color: white !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 12px 28px !important;
    border-radius: 10px !important;
    font-size: 14px !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: #818cf8 !important;
}
.stSpinner > div {
    border-top-color: #6366f1 !important;
}
.stExpander {
    background-color: #12141f !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}
.stExpander summary {
    color: #f1f2f8 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stDataFrame {
    background-color: #12141f !important;
}
.stDownloadButton > button {
    background-color: transparent !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #9294a8 !important;
    border-radius: 8px !important;
}
</style>
"""


def base_styles():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Raleway:wght@600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');
    * { box-sizing: border-box; margin: 0; padding: 0; }
    * {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        box-sizing: border-box;
    }
    html, body {
        width: 100%;
        height: 100%;
        background: #0a0b14 !important;
        color: #f1f2f8;
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        line-height: 1.6;
        -webkit-font-smoothing: antialiased;
        overflow: hidden;
    }
    body {
        background: #0a0b14 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Raleway', sans-serif !important;
    }
    [style*="font-family:'Syne'"],
    [style*="font-family: 'Syne'"] {
        font-family: 'Raleway', sans-serif !important;
    }
    [style*="font-family:'DM Sans'"],
    [style*="font-family: 'DM Sans'"] {
        font-family: 'Inter', sans-serif !important;
    }
    [style*="font-family:'DM Mono'"],
    [style*="font-family: 'DM Mono'"] {
        font-family: 'IBM Plex Mono', monospace !important;
    }
    a {
        color: inherit;
        text-decoration: none;
    }
    </style>
    """


def render_component(markup: str, height: int) -> None:
    components.html(base_styles() + dedent(markup), height=height, scrolling=False)

# ── Session state ────────────────────────────────────────────
def init_state():
    defaults = {
        "page":     "home",
        "results":  None,
        "last_run": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ── Navigation bar ───────────────────────────────────────────
def render_nav(page: str):
    render_component(
        f"""
        <style>
        @keyframes pulse {{
            0%,100% {{ opacity:1; transform:scale(1); }}
            50% {{ opacity:0.5; transform:scale(0.75); }}
        }}
        </style>
        <div style="background:rgba(10,11,20,0.95);
                    border-bottom:1px solid rgba(255,255,255,0.07);
                    padding:0 48px;height:68px;
                    display:flex;align-items:center;
                    justify-content:space-between;
                    position:relative;z-index:10;
                    font-family:'DM Sans',sans-serif">
            <div style="display:flex;align-items:center;gap:10px">
                <div style="width:30px;height:30px;border-radius:8px;
                            background:linear-gradient(135deg,#6366f1,#2dd4bf);
                            flex-shrink:0"></div>
                <span style="font-family:'Syne',sans-serif;font-weight:700;
                             font-size:17px;color:#f1f2f8">GEO Drift</span>
                <span style="font-family:'DM Mono',monospace;font-size:13px;
                             color:#52546a">/tracker</span>
            </div>
            <div style="display:flex;align-items:center;gap:20px">
                <div style="display:flex;align-items:center;gap:8px">
                    <div style="width:8px;height:8px;border-radius:50%;
                                background:#2dd4bf;
                                animation:pulse 2s infinite;
                                box-shadow:0 0 6px #2dd4bf"></div>
                    <span style="font-family:'DM Mono',monospace;font-size:12px;
                                 color:#9294a8">2 engines live</span>
                </div>
            </div>
        </div>
        """,
        height=68,
    )

# ── HOME PAGE ────────────────────────────────────────────────
def render_home():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    render_nav("home")

    # Hero
    render_component("""
    <div style="background:#0a0b14;padding:64px 48px 64px;
                text-align:center;font-family:'DM Sans',sans-serif;
                position:relative;overflow:hidden">
        <div style="position:absolute;width:500px;height:500px;
                    border-radius:50%;
                    background:rgba(99,102,241,0.07);
                    filter:blur(120px);
                    top:-200px;right:-100px;pointer-events:none"></div>
        <div style="position:absolute;width:400px;height:400px;
                    border-radius:50%;
                    background:rgba(45,212,191,0.05);
                    filter:blur(120px);
                    bottom:-100px;left:-100px;pointer-events:none"></div>

        <div style="display:inline-block;
                    background:rgba(99,102,241,0.10);
                    border:1px solid rgba(99,102,241,0.28);
                    color:#818cf8;
                    font-family:'DM Mono',monospace;font-size:12px;
                    padding:6px 16px;border-radius:24px;
                    margin-bottom:32px;letter-spacing:0.06em">
            Built for Peec AI &nbsp;·&nbsp; GEO Intelligence Layer
        </div>

        <div style="font-family:'Syne',sans-serif;font-weight:800;
                    font-size:clamp(38px,4.5vw,58px);line-height:1.1;
                    letter-spacing:-0.03em;margin-bottom:20px">
            <div style="color:#f1f2f8">Know which brands</div>
            <div style="background:linear-gradient(120deg,#6366f1,#2dd4bf);
                        -webkit-background-clip:text;
                        -webkit-text-fill-color:transparent;
                        background-clip:text">
                AI actually trusts.
            </div>
        </div>

        <div style="font-family:'DM Sans',sans-serif;font-weight:300;
                    font-size:16px;color:#9294a8;line-height:1.75;
                    max-width:520px;margin:0 auto 40px">
            Ask an AI the same question five times and you get five different
            answers. GEO Drift Tracker runs those queries, measures how much
            the answers change, and tells you which brands are genuinely
            consistent — and which ones just got lucky.
        </div>
    </div>
    """, height=460)

    # CTA Buttons
    _, c1, gap, c2, _ = st.columns([3, 1, 0.1, 1, 3])
    with c1:
        if st.button("Open Dashboard", key="hero_dashboard", type="primary"):
            st.session_state.page = "dashboard"
            st.rerun()
    with c2:
        if st.button("How it works", key="hero_howto", type="secondary"):
            st.session_state.page = "how_it_works"
            st.rerun()

    # Stats strip
    render_component("""
    <div style="border-top:1px solid rgba(255,255,255,0.07);
                border-bottom:1px solid rgba(255,255,255,0.07);
                display:grid;grid-template-columns:repeat(4,1fr);
                margin-top:72px;background:#0a0b14">
        <div style="text-align:center;padding:44px 20px;height:200px;
                    border-right:1px solid rgba(255,255,255,0.07)">
            <div style="font-family:'Syne',sans-serif;font-weight:800;
                        font-size:52px;line-height:1;color:#6366f1">2</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;
                        text-transform:uppercase;letter-spacing:0.09em;
                        color:#9294a8;margin-top:14px;display:block">AI Engines</div>
        </div>
        <div style="text-align:center;padding:44px 20px;height:200px;
                    border-right:1px solid rgba(255,255,255,0.07)">
            <div style="font-family:'Syne',sans-serif;font-weight:800;
                        font-size:52px;line-height:1;color:#2dd4bf">5x</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;
                        text-transform:uppercase;letter-spacing:0.09em;
                        color:#9294a8;margin-top:14px;display:block">Runs Per Prompt</div>
        </div>
        <div style="text-align:center;padding:44px 20px;height:200px;
                    border-right:1px solid rgba(255,255,255,0.07)">
            <div style="font-family:'Syne',sans-serif;font-weight:800;
                        font-size:52px;line-height:1;color:#fbbf24">0–1</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;
                        text-transform:uppercase;letter-spacing:0.09em;
                        color:#9294a8;margin-top:14px;display:block">Stability Score</div>
        </div>
        <div style="text-align:center;padding:44px 20px;height:200px">
            <div style="font-family:'Syne',sans-serif;font-weight:800;
                        font-size:52px;line-height:1;color:#a78bfa">$0</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;
                        text-transform:uppercase;letter-spacing:0.09em;
                        color:#9294a8;margin-top:14px;display:block">Cost To Run</div>
        </div>
    </div>
    """, height=200)

    # Problem section
    render_component("""
    <div style="padding:96px 48px 0;background:#0a0b14;
                font-family:'DM Sans',sans-serif">
        <div style="font-family:'DM Mono',monospace;font-size:11px;
                    color:#6366f1;text-transform:uppercase;
                    letter-spacing:0.12em;margin-bottom:12px">The Problem</div>
        <div style="font-family:'Syne',sans-serif;font-weight:700;
                    font-size:clamp(30px,4vw,44px);color:#f1f2f8;
                    letter-spacing:-0.03em;margin-bottom:52px">
            One snapshot lies to you.
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px">
            <div style="background:#12141f;border:1px solid rgba(255,255,255,0.07);
                        border-top:3px solid #f472b6;border-radius:0 0 14px 14px;
                        padding:32px 28px;min-height:180px;transition:transform 0.2s">
                <div style="font-family:'Syne',sans-serif;font-weight:600;
                            font-family:'Raleway',sans-serif;font-weight:600;
                            font-size:17px;color:#f1f2f8;margin-bottom:14px">
                    Different every time
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:14px;
                            color:#9294a8;line-height:1.75;font-weight:300">
                    Ask the same question twice and you get different brand rankings.
                    Which result do you report to your client?
                </div>
            </div>
            <div style="background:#12141f;border:1px solid rgba(255,255,255,0.07);
                        border-top:3px solid #fbbf24;border-radius:0 0 14px 14px;
                        padding:32px 28px;min-height:180px">
                <div style="font-family:'Syne',sans-serif;font-weight:600;
                            font-family:'Raleway',sans-serif;font-weight:600;
                            font-size:17px;color:#f1f2f8;margin-bottom:14px">
                    Engines disagree
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:14px;
                            color:#9294a8;line-height:1.75;font-weight:300">
                    Gemini and Groq often rank the same brand in completely different
                    positions. Both cannot be right at the same time.
                </div>
            </div>
            <div style="background:#12141f;border:1px solid rgba(255,255,255,0.07);
                        border-top:3px solid #6366f1;border-radius:0 0 14px 14px;
                        padding:32px 28px;min-height:180px">
                <div style="font-family:'Syne',sans-serif;font-weight:600;
                            font-family:'Raleway',sans-serif;font-weight:600;
                            font-size:17px;color:#f1f2f8;margin-bottom:14px">
                    Snapshots mislead
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:14px;
                            color:#9294a8;line-height:1.75;font-weight:300">
                    A brand at position 1 today may be at position 6 tomorrow
                    with no real change. You need variance data, not spot checks.
                </div>
            </div>
        </div>
    </div>
    """, height=520)

    # Solution card
    render_component("""
    <div style="padding:24px 48px 0;background:#0a0b14;
                font-family:'DM Sans',sans-serif">
        <div style="background:rgba(99,102,241,0.06);
                    border:1px solid rgba(99,102,241,0.18);
                    border-radius:16px;padding:40px 44px;min-height:220px;
                    display:flex;justify-content:space-between;
                    align-items:center;gap:48px">
            <div style="flex:1">
                <div style="font-family:'DM Mono',monospace;font-size:11px;
                            color:#6366f1;text-transform:uppercase;
                            letter-spacing:0.12em;margin-bottom:12px">
                    The Solution
                </div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;
                            font-size:28px;color:#f1f2f8;margin-bottom:24px">
                    Measure drift. Find what is real.
                </div>
                <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:14px">
                    <div style="width:8px;height:8px;border-radius:2px;
                                background:#2dd4bf;margin-top:6px;flex-shrink:0"></div>
                    <span style="font-size:15px;color:#9294a8;font-weight:400">
                        Runs each query 5 times to expose natural variance
                    </span>
                </div>
                <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:14px">
                    <div style="width:8px;height:8px;border-radius:2px;
                                background:#2dd4bf;margin-top:6px;flex-shrink:0"></div>
                    <span style="font-size:15px;color:#9294a8;font-weight:400">
                        Queries Gemini and Groq simultaneously
                    </span>
                </div>
                <div style="display:flex;align-items:flex-start;gap:12px">
                    <div style="width:8px;height:8px;border-radius:2px;
                                background:#2dd4bf;margin-top:6px;flex-shrink:0"></div>
                    <span style="font-size:15px;color:#9294a8;font-weight:400">
                        Gives every brand a 0 to 1 reliability score
                    </span>
                </div>
            </div>
            <div style="text-align:right;flex-shrink:0">
                <div style="font-family:'Syne',sans-serif;font-weight:800;
                            font-size:60px;color:#2dd4bf;line-height:1">97%</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;
                            color:#9294a8;line-height:1.8;margin-top:8px">
                    of brand drift goes undetected
                </div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;
                            color:#9294a8;line-height:1.8">
                    without run-to-run comparison
                </div>
            </div>
        </div>
    </div>
    """, height=300)

    # Features
    render_component("""
    <div style="padding:96px 48px 0;background:#0a0b14;
                font-family:'DM Sans',sans-serif">
        <div style="font-family:'DM Mono',monospace;font-size:11px;
                    color:#6366f1;text-transform:uppercase;
                    letter-spacing:0.12em;margin-bottom:12px;text-align:center">
            Features
        </div>
        <div style="font-family:'Syne',sans-serif;font-weight:700;
                    font-size:clamp(30px,4vw,44px);color:#f1f2f8;
                    letter-spacing:-0.03em;margin-bottom:52px;text-align:center">
            Everything you need, nothing extra.
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px">

            <div style="background:#12141f;border:1px solid rgba(255,255,255,0.07);
                        border-radius:16px;padding:32px">
                <div style="height:3px;background:#6366f1;border-radius:14px 14px 0 0;
                            margin:-32px -32px 24px -32px"></div>
                <div style="font-family:'Syne',sans-serif;font-weight:600;
                            font-size:17px;color:#f1f2f8;margin-bottom:12px">
                    Stability Score
                </div>
                <div style="font-size:14px;color:#9294a8;line-height:1.75;font-weight:300">
                    Every brand receives a score from 0 to 1. A score of 1 means it
                    appears consistently at the same position every time.
                </div>
            </div>

            <div style="background:#12141f;border:1px solid rgba(255,255,255,0.07);
                        border-radius:16px;padding:32px">
                <div style="height:3px;background:#2dd4bf;border-radius:14px 14px 0 0;
                            margin:-32px -32px 24px -32px"></div>
                <div style="font-family:'Syne',sans-serif;font-weight:600;
                            font-size:17px;color:#f1f2f8;margin-bottom:12px">
                    Two AI Engines
                </div>
                <div style="font-size:14px;color:#9294a8;line-height:1.75;font-weight:300">
                    Runs on Google Gemini and Groq simultaneously. See exactly where
                    the two engines agree and where they completely disagree.
                </div>
            </div>

            <div style="background:#12141f;border:1px solid rgba(255,255,255,0.07);
                        border-radius:16px;padding:32px">
                <div style="height:3px;background:#a78bfa;border-radius:14px 14px 0 0;
                            margin:-32px -32px 24px -32px"></div>
                <div style="font-family:'Syne',sans-serif;font-weight:600;
                            font-size:17px;color:#f1f2f8;margin-bottom:12px">
                    Sentiment Detection
                </div>
                <div style="font-size:14px;color:#9294a8;line-height:1.75;font-weight:300">
                    Every brand mention is automatically labeled as positive, neutral,
                    or negative based on how the AI describes it.
                </div>
            </div>

            <div style="background:#12141f;border:1px solid rgba(255,255,255,0.07);
                        border-radius:16px;padding:32px">
                <div style="height:3px;background:#fbbf24;border-radius:14px 14px 0 0;
                            margin:-32px -32px 24px -32px"></div>
                <div style="font-family:'Syne',sans-serif;font-weight:600;
                            font-size:17px;color:#f1f2f8;margin-bottom:12px">
                    Completely Free
                </div>
                <div style="font-size:14px;color:#9294a8;line-height:1.75;font-weight:300">
                    Built entirely on free API tiers. Google Gemini Flash and Groq
                    both offer generous free plans. No credit card required.
                </div>
            </div>

            <div style="background:#12141f;border:1px solid rgba(255,255,255,0.07);
                        border-radius:16px;padding:32px">
                <div style="height:3px;background:#6366f1;border-radius:14px 14px 0 0;
                            margin:-32px -32px 24px -32px"></div>
                <div style="font-family:'Syne',sans-serif;font-weight:600;
                            font-size:17px;color:#f1f2f8;margin-bottom:12px">
                    Built to Extend
                </div>
                <div style="font-size:14px;color:#9294a8;line-height:1.75;font-weight:300">
                    Adding a new AI engine like OpenAI requires writing one new
                    function. The rest of the system adapts automatically.
                </div>
            </div>

            <div style="background:#12141f;border:1px solid rgba(255,255,255,0.07);
                        border-radius:16px;padding:32px">
                <div style="height:3px;background:#2dd4bf;border-radius:14px 14px 0 0;
                            margin:-32px -32px 24px -32px"></div>
                <div style="font-family:'Syne',sans-serif;font-weight:600;
                            font-size:17px;color:#f1f2f8;margin-bottom:12px">
                    Export Everything
                </div>
                <div style="font-size:14px;color:#9294a8;line-height:1.75;font-weight:300">
                    Download a full CSV of all brand data. Charts saved as PNG.
                    Ready for any presentation or analytics workflow.
                </div>
            </div>
        </div>
    </div>
    """, height=620)

    # How it works
    render_component("""
    <div style="padding:96px 48px 104px;background:#0a0b14;
                font-family:'DM Sans',sans-serif">
        <div style="font-family:'DM Mono',monospace;font-size:11px;
                    color:#6366f1;text-transform:uppercase;
                    letter-spacing:0.12em;margin-bottom:12px;text-align:center">
            Process
        </div>
        <div style="font-family:'Syne',sans-serif;font-weight:700;
                    font-size:clamp(30px,4vw,44px);color:#f1f2f8;
                    letter-spacing:-0.03em;margin-bottom:64px;text-align:center">
            Four steps to brand clarity.
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0;
                    position:relative">
            <div style="position:absolute;top:25px;left:12.5%;width:75%;height:1px;
                        background:rgba(99,102,241,0.25)"></div>
            <div style="text-align:center;padding:0 20px;position:relative">
                <div style="width:52px;height:52px;border-radius:50%;
                            border:1.5px solid rgba(99,102,241,0.5);
                            background:rgba(99,102,241,0.12);
                            display:flex;align-items:center;justify-content:center;
                            margin:0 auto 20px;
                            font-family:'IBM Plex Mono',monospace;font-size:15px;
                            color:#818cf8;font-weight:500">01</div>
                <div style="font-family:'Raleway',sans-serif;font-weight:600;
                            font-size:16px;color:#f1f2f8;margin-bottom:10px">
                    Query 5 times
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:13px;
                            color:#9294a8;line-height:1.65;font-weight:400;
                            max-width:180px;margin:0 auto">
                    Each prompt goes to both Gemini and Groq, five times each, independently
                </div>
            </div>
            <div style="text-align:center;padding:0 20px;position:relative">
                <div style="width:52px;height:52px;border-radius:50%;
                            border:1.5px solid rgba(99,102,241,0.5);
                            background:rgba(99,102,241,0.12);
                            display:flex;align-items:center;justify-content:center;
                            margin:0 auto 20px;
                            font-family:'IBM Plex Mono',monospace;font-size:15px;
                            color:#818cf8;font-weight:500">02</div>
                <div style="font-family:'Raleway',sans-serif;font-weight:600;
                            font-size:16px;color:#f1f2f8;margin-bottom:10px">
                    Parse responses
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:13px;
                            color:#9294a8;line-height:1.65;font-weight:400;
                            max-width:180px;margin:0 auto">
                    Brand names, positions, and sentiment pulled from every single response
                </div>
            </div>
            <div style="text-align:center;padding:0 20px;position:relative">
                <div style="width:52px;height:52px;border-radius:50%;
                            border:1.5px solid rgba(99,102,241,0.5);
                            background:rgba(99,102,241,0.12);
                            display:flex;align-items:center;justify-content:center;
                            margin:0 auto 20px;
                            font-family:'IBM Plex Mono',monospace;font-size:15px;
                            color:#818cf8;font-weight:500">03</div>
                <div style="font-family:'Raleway',sans-serif;font-weight:600;
                            font-size:16px;color:#f1f2f8;margin-bottom:10px">
                    Calculate variance
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:13px;
                            color:#9294a8;line-height:1.65;font-weight:400;
                            max-width:180px;margin:0 auto">
                    We measure how much the answers differ across all five runs per engine
                </div>
            </div>
            <div style="text-align:center;padding:0 20px;position:relative">
                <div style="width:52px;height:52px;border-radius:50%;
                            border:1.5px solid rgba(99,102,241,0.5);
                            background:rgba(99,102,241,0.12);
                            display:flex;align-items:center;justify-content:center;
                            margin:0 auto 20px;
                            font-family:'IBM Plex Mono',monospace;font-size:15px;
                            color:#818cf8;font-weight:500">04</div>
                <div style="font-family:'Raleway',sans-serif;font-weight:600;
                            font-size:16px;color:#f1f2f8;margin-bottom:10px">
                    Score stability
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:13px;
                            color:#9294a8;line-height:1.65;font-weight:400;
                            max-width:180px;margin:0 auto">
                    Each brand scored 0 to 1 based on how consistently it appears across all runs
                </div>
            </div>
        </div>
    </div>
    """, height=540)

    st.markdown("<div style='height:28px;background:#0a0b14'></div>", unsafe_allow_html=True)

    # Footer
    render_component("""
    <div style="border-top:1px solid rgba(255,255,255,0.07);
                padding:32px 48px;background:#0a0b14;
                display:flex;justify-content:space-between;align-items:center;
                font-family:'DM Sans',sans-serif">
        <div>
            <div style="font-family:'Syne',sans-serif;font-weight:600;
                        font-size:15px;color:#f1f2f8">GEO Drift Tracker</div>
            <div style="font-family:'DM Mono',monospace;font-size:12px;
                        color:#52546a;margin-top:4px">
                Niranjan Tapasvi  ©  2026
            </div>
        </div>
        <div style="display:flex;gap:8px">
            <span style="background:rgba(255,255,255,0.04);
                         border:1px solid rgba(255,255,255,0.08);
                         border-radius:20px;padding:5px 16px;
                         font-family:'DM Mono',monospace;font-size:11px;
                         color:#9294a8">Python 3.11</span>
            <span style="background:rgba(255,255,255,0.04);
                         border:1px solid rgba(255,255,255,0.08);
                         border-radius:20px;padding:5px 16px;
                         font-family:'DM Mono',monospace;font-size:11px;
                         color:#9294a8">Gemini + Groq</span>
            <span style="background:rgba(255,255,255,0.04);
                         border:1px solid rgba(255,255,255,0.08);
                         border-radius:20px;padding:5px 16px;
                         font-family:'DM Mono',monospace;font-size:11px;
                         color:#9294a8">Open Source</span>
        </div>
    </div>
    """, height=120)


def render_how_it_works():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    render_nav("how_it_works")
    st.markdown("<div style='height:20px;background:#0a0b14'></div>", unsafe_allow_html=True)

    col_back, _, _ = st.columns([1, 4, 1])
    with col_back:
        if st.button("← Back to Home", key="how_back_home", type="secondary"):
            st.session_state.page = "home"
            st.rerun()

    st.markdown("<div style='height:24px;background:#0a0b14'></div>", unsafe_allow_html=True)

    render_component("""
    <div style="padding:28px 48px 0;background:#0a0b14;text-align:center">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                    color:#6366f1;text-transform:uppercase;
                    letter-spacing:0.12em;margin-bottom:12px">
            How it works
        </div>
        <div style="font-family:'Raleway',sans-serif;font-weight:700;
                    font-size:44px;color:#f1f2f8;letter-spacing:-0.03em">
            Simple enough for anyone.
        </div>
    </div>
    """, height=120)

    st.markdown("<div style='height:24px;background:#0a0b14'></div>", unsafe_allow_html=True)

    sections = [
        (
            "01",
            "You give it a question",
            "Type in any question about software or products, the kind you might search for online. Something like what are the best tools for managing a small business. That is all you need to get started.",
        ),
        (
            "02",
            "It asks two AI systems the same thing",
            "The tool sends your question to Google Gemini and Groq at the same time. It does this five times each so it gets multiple different answers from both. This is how it catches inconsistency.",
        ),
        (
            "03",
            "It finds the patterns",
            "Every response is read automatically. The tool picks out which brand names appear, where they show up in the list, and whether the AI is speaking about them positively or negatively. This happens across all ten responses.",
        ),
        (
            "04",
            "It tells you who to trust",
            "Each brand gets a score between 0 and 1. A score close to 1 means that brand shows up consistently in a similar position every time. A score close to 0 means the results jump around. Only the consistent ones are worth reporting on.",
        ),
    ]

    for number, title, text in sections:
        render_component(f"""
        <div style="padding:36px 40px;background:#12141f;
                    border:1px solid rgba(255,255,255,0.07);
                    border-radius:16px;margin:0 48px 16px 48px;
                    display:flex;align-items:flex-start;gap:28px">
            <div style="width:44px;height:44px;flex-shrink:0;border-radius:50%;
                        border:1.5px solid rgba(99,102,241,0.4);
                        background:rgba(99,102,241,0.10);
                        display:flex;align-items:center;justify-content:center;
                        font-family:'IBM Plex Mono',monospace;font-size:14px;
                        color:#818cf8;margin-top:4px">{number}</div>
            <div style="flex:1;text-align:left">
                <div style="font-family:'Raleway',sans-serif;font-weight:600;
                            font-size:20px;color:#f1f2f8;margin-bottom:10px">
                    {html_escape(title)}
                </div>
                <div style="font-family:'Inter',sans-serif;font-weight:300;
                            font-size:15px;color:#9294a8;line-height:1.75">
                    {html_escape(text)}
                </div>
            </div>
        </div>
        """, height=190)

    render_component("""
    <div style="padding:8px 48px 0;background:#0a0b14">
        <div style="background:rgba(99,102,241,0.06);
                    border:1px solid rgba(99,102,241,0.18);
                    border-radius:16px;padding:36px 40px">
            <div style="font-family:'Raleway',sans-serif;font-weight:700;
                        font-size:22px;color:#f1f2f8;margin-bottom:14px">
                Why this matters
            </div>
            <div style="font-family:'Inter',sans-serif;font-weight:300;
                        font-size:16px;color:#9294a8;line-height:1.8">
                Most tools take one snapshot and report it as fact. But AI answers are not consistent. They change with every request. This tool measures that change so you know what is real signal and what is just noise. That is the difference between a reliable report and a misleading one.
            </div>
        </div>
    </div>
    """, height=220)


# ── DASHBOARD PAGE ───────────────────────────────────────────
def render_dashboard():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    render_nav("dashboard")

    st.markdown("<div style='height:24px;background:#0a0b14'></div>", unsafe_allow_html=True)

    col_back, col_spacer, col_run = st.columns([1, 1, 1])
    with col_back:
        if st.button("← Back to Home", key="back_home", type="secondary"):
            st.session_state.page = "home"
            st.rerun()
    with col_run:
        run_clicked = st.button("Run Analysis", key="run_btn", type="primary")

    st.markdown("<div style='height:8px;background:#0a0b14'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.session_state.last_run:
            st.markdown(
                f"<div style='text-align:right;font-family:\"IBM Plex Mono\",monospace;font-size:12px;color:#52546a'>Last run: {st.session_state.last_run}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='text-align:right;font-family:\"IBM Plex Mono\",monospace;font-size:12px;color:#52546a'>Takes approx. 60 seconds</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:32px;background:#0a0b14'></div>", unsafe_allow_html=True)

    # Header
    render_component("""
    <div style="padding:0 48px 24px;background:#0a0b14;
                font-family:'DM Sans',sans-serif">
        <div style="font-family:'DM Mono',monospace;font-size:11px;
                    color:#6366f1;text-transform:uppercase;
                    letter-spacing:0.12em;margin-bottom:10px">
            Brand Intelligence
        </div>
        <div style="font-family:'Syne',sans-serif;font-weight:700;
                    font-size:40px;color:#f1f2f8;letter-spacing:-0.03em;
                    margin-bottom:6px">Dashboard</div>
        <div style="font-size:17px;color:#9294a8;font-weight:300">
            Run-to-run drift analysis across AI engines
        </div>
    </div>
    """, height=120)

    st.markdown("<div style='height:24px;background:#0a0b14'></div>", unsafe_allow_html=True)

    # Run pipeline
    if run_clicked:
        try:
            from providers import PROVIDERS
            from parser import parse_response
            from aggregator import aggregate_all
            from config import PROMPTS, N_RUNS

            with st.spinner("Querying Gemini and Groq — about 60 seconds..."):
                runs_by_provider = {
                    p: {prompt: [] for prompt in PROMPTS}
                    for p in PROVIDERS
                }
                for provider_name, query_fn in PROVIDERS.items():
                    for prompt in PROMPTS:
                        for _ in range(N_RUNS):
                            raw = query_fn(prompt)
                            parsed = parse_response(raw) if raw else []
                            runs_by_provider[provider_name][prompt].append(parsed)

                results = aggregate_all(runs_by_provider, N_RUNS)
                st.session_state.results = results
                st.session_state.last_run = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                )
            st.success("Analysis complete!")
            st.rerun()
        except ModuleNotFoundError:
            try:
                from src.geo_drift_tracker import (
                    PROMPTS,
                    N_RUNS,
                    combine_and_report,
                    run_pipeline,
                )

                with st.spinner("Querying Gemini and Groq — about 60 seconds..."):
                    all_results, raw_outputs = run_pipeline(
                        PROMPTS, n_runs=N_RUNS, providers=("gemini", "groq")
                    )
                    combined_info = combine_and_report(all_results, raw_outputs, PROMPTS)
                    prompt_name = PROMPTS[0]
                    safe_prompt = re.sub(r"[^0-9A-Za-z]+", "_", prompt_name)[:50]
                    combined_path = combined_info[prompt_name]["combined_csv"]

                    combined = pd.read_csv(combined_path)
                    combined["stability_score"] = combined[
                        ["gemini_stability", "groq_stability"]
                    ].mean(axis=1)
                    combined["mention_frequency"] = combined[
                        ["gemini_mention_rate", "groq_mention_rate"]
                    ].mean(axis=1).fillna(0.0)
                    combined["avg_position"] = combined[
                        ["gemini_avg_position", "groq_avg_position"]
                    ].mean(axis=1)
                    combined["std_position"] = combined[
                        ["gemini_position_std", "groq_position_std"]
                    ].mean(axis=1).fillna(0.0)
                    combined = combined.sort_values(
                        ["stability_score", "mention_frequency"], ascending=[False, False]
                    ).reset_index(drop=True)

                    gemini_df = all_results.get("gemini", {}).get(
                        prompt_name, pd.DataFrame()
                    ).copy()
                    groq_df = all_results.get("groq", {}).get(
                        prompt_name, pd.DataFrame()
                    ).copy()
                    if not gemini_df.empty:
                        gemini_df = gemini_df.rename(
                            columns={"stability": "stability_score", "position_std": "std_position"}
                        )
                    if not groq_df.empty:
                        groq_df = groq_df.rename(
                            columns={"stability": "stability_score", "position_std": "std_position"}
                        )

                    st.session_state.results = {
                        "combined": combined,
                        "gemini": gemini_df,
                        "groq": groq_df,
                        "prompt": prompt_name,
                        "combined_path": combined_path,
                        "safe_prompt": safe_prompt,
                    }
                    st.session_state.last_run = datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    )
                st.success("Analysis complete!")
                st.rerun()
            except Exception as fallback_error:
                st.error(f"Run failed: {fallback_error}")
        except Exception as e:
            st.error(f"Run failed: {e}")

    results = st.session_state.results

    # Pre-run state
    if results is None:
        render_component(
            """
            <div style="max-width:480px; margin:64px auto;
                        background:#12141f;
                        border:1px dashed rgba(99,102,241,0.3);
                        border-radius:20px; padding:56px 44px;
                        text-align:center; font-family:'Inter',sans-serif">
                <div style="font-family:'Raleway',sans-serif;
                            font-weight:600; font-size:22px;
                            color:#f1f2f8; margin-bottom:16px;
                            line-height:1.3">
                    Ready to run your first analysis
                </div>
                <div style="font-size:15px; color:#9294a8;
                            line-height:1.75; font-weight:300;
                            margin-bottom:32px">
                    Click Run Analysis to query both Gemini
                    and Groq across your prompts. Your full
                    results will appear here once complete.
                </div>
                <div style="background:rgba(251,191,36,0.07);
                            border:1px solid rgba(251,191,36,0.2);
                            border-radius:12px; padding:20px 24px;
                            text-align:left">
                    <div style="font-family:'IBM Plex Mono',monospace;
                                font-size:11px; color:#fbbf24;
                                text-transform:uppercase;
                                letter-spacing:0.08em; margin-bottom:8px">
                        Tip
                    </div>
                    <div style="font-size:13px; color:#9294a8;
                                line-height:1.65; font-weight:300">
                        The free API tier allows around 30 calls
                        per session. Keeping runs per prompt at 5
                        keeps you well within the daily limit.
                    </div>
                </div>
            </div>
            """,
            height=440,
        )
        return

    # Results exist — show dashboard
    combined = results.get("combined", pd.DataFrame())
    gemini_df = results.get("gemini", pd.DataFrame())
    groq_df   = results.get("groq",   pd.DataFrame())

    if not combined.empty and "brand" in combined.columns:
        combined = combined[combined["brand"].apply(is_plausible_brand)].copy()
    if not gemini_df.empty and "brand" in gemini_df.columns:
        gemini_df = gemini_df[gemini_df["brand"].apply(is_plausible_brand)].copy()
    if not groq_df.empty and "brand" in groq_df.columns:
        groq_df = groq_df[groq_df["brand"].apply(is_plausible_brand)].copy()

    if combined.empty:
        st.warning("No results to display. Try running the analysis again.")
        return

    st.markdown("<div style='height:24px;background:#0a0b14'></div>", unsafe_allow_html=True)

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)

    def metric_card(col, value, label, sub, color):
        value_text = str(value)
        is_number = value_text.replace(".", "", 1).isdigit()
        size = "40px" if is_number else "18px"
        weight = "800" if is_number else "600"
        ff = "'Raleway', sans-serif" if is_number else "'Inter', sans-serif"
        with col:
            render_component(f"""
            <div style="background:#12141f;border:1px solid rgba(255,255,255,0.08);
                        border-radius:14px;padding:28px 22px;text-align:center;
                        font-family:'DM Sans',sans-serif;height:180px">
                <div style="font-family:'Syne',sans-serif;font-weight:800;
                            font-family:{ff};font-weight:{weight};
                            font-size:{size};
                            color:{color};line-height:1;margin-bottom:8px">
                    {html_escape(str(value))}
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:11px;
                            text-transform:uppercase;letter-spacing:0.07em;
                            color:#52546a">{html_escape(str(label))}</div>
                <div style="font-size:13px;color:#9294a8;margin-top:4px">{html_escape(str(sub))}</div>
            </div>
            """, height=180)

    n_brands = len(combined)
    total_calls = 30
    best  = combined.iloc[0]["brand"] if not combined.empty else "—"
    worst = combined.sort_values("stability_score").iloc[0]["brand"] \
            if not combined.empty else "—"

    metric_card(m1, n_brands,  "Brands Tracked",   "across all prompts",      "#6366f1")
    metric_card(m2, total_calls,"API Calls Made",   "within free quota",       "#2dd4bf")
    metric_card(m3, best,       "Most Stable",      "safest to report on",     "#2dd4bf")
    metric_card(m4, worst,      "Most Volatile",    "treat with caution",      "#f472b6")

    st.markdown("<div style='height:24px;background:#0a0b14'></div>", unsafe_allow_html=True)

    gap_df = pd.DataFrame()
    gap_brand = "—"
    gap_value = None
    gap_note = "No overlapping brands between Gemini and Groq yet."
    if not gemini_df.empty and not groq_df.empty:
        common = sorted(set(gemini_df["brand"]) & set(groq_df["brand"]))
        if common:
            gem_idx = gemini_df.set_index("brand")["avg_position"]
            grq_idx = groq_df.set_index("brand")["avg_position"]
            gap_rows = []
            for brand in common:
                gap = abs(float(gem_idx[brand]) - float(grq_idx[brand]))
                gap_rows.append(
                    {
                        "brand": brand,
                        "gap": gap,
                        "gemini": float(gem_idx[brand]),
                        "groq": float(grq_idx[brand]),
                    }
                )
            gap_df = pd.DataFrame(gap_rows).sort_values("gap", ascending=False)
            top_gap = gap_df.iloc[0]
            gap_brand = str(top_gap["brand"])
            gap_value = float(top_gap["gap"])
            gap_note = (
                f"Gemini and Groq disagree most on {gap_brand} "
                f"with a position gap of {gap_value:.1f}."
            )

    avg_stability = float(combined["stability_score"].mean()) if "stability_score" in combined.columns and not combined.empty else 0.0
    stable_spot = str(combined.iloc[0]["brand"]) if not combined.empty else "—"
    caution_spot = str(combined.sort_values("stability_score").iloc[0]["brand"]) if not combined.empty else "—"

    render_component(f"""
    <div style="padding:26px 28px;background:rgba(99,102,241,0.06);
                border:1px solid rgba(99,102,241,0.18);
                border-radius:16px;margin:0 48px 12px 48px;
                font-family:'DM Sans',sans-serif">
        <div style="font-family:'DM Mono',monospace;font-size:11px;
                    color:#6366f1;text-transform:uppercase;
                    letter-spacing:0.12em;margin-bottom:10px">
            Run takeaway
        </div>
        <div style="font-family:'Raleway',sans-serif;font-size:22px;
                    font-weight:700;color:#f1f2f8;margin-bottom:16px">
            Signal readout
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px">
            <div style="background:#12141f;border:1px solid rgba(255,255,255,0.07);
                        border-radius:14px;padding:18px 18px 16px 18px">
                <div style="font-family:'DM Mono',monospace;font-size:11px;
                            color:#9294a8;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:8px">
                    Safe to cite
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:16px;
                            font-weight:600;color:#f1f2f8;line-height:1.4">
                    {html_escape(stable_spot)}
                </div>
                <div style="font-size:13px;color:#9294a8;line-height:1.6;margin-top:6px">
                    Highest stability and safest for a one-slide summary.
                </div>
            </div>
            <div style="background:#12141f;border:1px solid rgba(255,255,255,0.07);
                        border-radius:14px;padding:18px 18px 16px 18px">
                <div style="font-family:'DM Mono',monospace;font-size:11px;
                            color:#9294a8;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:8px">
                    Watch closely
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:16px;
                            font-weight:600;color:#f1f2f8;line-height:1.4">
                    {html_escape(caution_spot)}
                </div>
                <div style="font-size:13px;color:#9294a8;line-height:1.6;margin-top:6px">
                    Lowest stability and most likely to shift between runs.
                </div>
            </div>
            <div style="background:#12141f;border:1px solid rgba(255,255,255,0.07);
                        border-radius:14px;padding:18px 18px 16px 18px">
                <div style="font-family:'DM Mono',monospace;font-size:11px;
                            color:#9294a8;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:8px">
                    Biggest split
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:16px;
                            font-weight:600;color:#f1f2f8;line-height:1.4">
                    {html_escape(gap_brand)}
                </div>
                <div style="font-size:13px;color:#9294a8;line-height:1.6;margin-top:6px">
                    {html_escape(gap_note)}
                </div>
            </div>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;
                    color:#52546a;margin-top:14px;line-height:1.6">
            Signal strength {avg_stability:.2f} · Stable brands are the safest call.
        </div>
    </div>
    """, height=270)
    st.markdown("<div style='height:22px;background:#0a0b14'></div>", unsafe_allow_html=True)

    # Stability table
    render_component("""
    <div style="padding:40px 48px 0;background:#0a0b14;
                font-family:'DM Sans',sans-serif">
        <div style="font-family:'DM Mono',monospace;font-size:11px;
                    color:#6366f1;text-transform:uppercase;
                    letter-spacing:0.12em;margin-bottom:6px">
            Stability Ranking
        </div>
        <div style="font-size:14px;color:#9294a8;margin-bottom:20px">
            How consistently each brand appears across repeated queries
        </div>
    </div>
    """, height=80)

    st.markdown("<div style='height:10px;background:#0a0b14'></div>", unsafe_allow_html=True)

    def stability_color(score):
        if score >= 0.7: return "#2dd4bf"
        if score >= 0.4: return "#fbbf24"
        return "#f472b6"

    def verdict_label(score):
        if score >= 0.7: return "stable"
        if score >= 0.4: return "moderate"
        return "volatile"

    rows_html = ""
    for i, row in combined.iterrows():
        score = row["stability_score"]
        color = stability_color(score)
        verdict = verdict_label(score)
        bg = "rgba(255,255,255,0.018)" if i % 2 == 0 else "transparent"
        rows_html += f"""
        <tr style="background:{bg};border-bottom:1px solid rgba(255,255,255,0.05)">
            <td style="padding:14px 20px;font-weight:500;color:#f1f2f8">
                {row['brand']}
            </td>
            <td style="padding:14px 20px;font-family:'DM Mono',monospace;
                       font-size:13px;color:#9294a8">
                {int(row['mention_frequency']*100)}%
            </td>
            <td style="padding:14px 20px;font-family:'DM Mono',monospace;
                       font-size:13px;color:#9294a8">
                {row['avg_position']:.1f}
            </td>
            <td style="padding:14px 20px">
                <div style="display:flex;align-items:center;gap:10px">
                    <div style="width:120px;height:6px;
                                background:rgba(255,255,255,0.07);
                                border-radius:3px;overflow:hidden">
                        <div style="width:{score*100:.0f}%;height:100%;
                                    background:{color};border-radius:3px">
                        </div>
                    </div>
                    <span style="font-family:'DM Mono',monospace;font-size:12px;
                                 color:{color}">{score:.2f}</span>
                </div>
            </td>
            <td style="padding:14px 20px">
                <span style="background:rgba({
                    '45,212,191' if verdict=='stable' else
                    '251,191,36' if verdict=='moderate' else
                    '244,114,182'
                },0.10);
                border:1px solid rgba({
                    '45,212,191' if verdict=='stable' else
                    '251,191,36' if verdict=='moderate' else
                    '244,114,182'
                },0.22);
                color:{color};
                font-family:'DM Mono',monospace;font-size:11px;
                padding:3px 12px;border-radius:20px;
                text-transform:uppercase;letter-spacing:0.05em">
                    {verdict}
                </span>
            </td>
        </tr>
        """

    render_component(f"""
    <div style="padding:0 48px;background:#0a0b14">
        <table style="width:100%;border-collapse:collapse;
                      background:#12141f;border-radius:14px;
                      overflow:hidden;
                      border:1px solid rgba(255,255,255,0.07);
                      font-family:'DM Sans',sans-serif">
            <thead>
                <tr style="background:#1a1d2e">
                    <th style="padding:14px 20px;text-align:left;
                               font-family:'DM Mono',monospace;font-size:11px;
                               color:#52546a;text-transform:uppercase;
                               letter-spacing:0.07em;font-weight:400">Brand</th>
                    <th style="padding:14px 20px;text-align:left;
                               font-family:'DM Mono',monospace;font-size:11px;
                               color:#52546a;text-transform:uppercase;
                               letter-spacing:0.07em;font-weight:400">
                        Mentioned<br>
                        <span style="font-size:9px;opacity:0.6">out of 5 runs</span>
                    </th>
                    <th style="padding:14px 20px;text-align:left;
                               font-family:'DM Mono',monospace;font-size:11px;
                               color:#52546a;text-transform:uppercase;
                               letter-spacing:0.07em;font-weight:400">
                        Avg Position<br>
                        <span style="font-size:9px;opacity:0.6">lower is better</span>
                    </th>
                    <th style="padding:14px 20px;text-align:left;
                               font-family:'DM Mono',monospace;font-size:11px;
                               color:#52546a;text-transform:uppercase;
                               letter-spacing:0.07em;font-weight:400">
                        Stability<br>
                        <span style="font-size:9px;opacity:0.6">1.0 = always consistent</span>
                    </th>
                    <th style="padding:14px 20px;text-align:left;
                               font-family:'DM Mono',monospace;font-size:11px;
                               color:#52546a;text-transform:uppercase;
                               letter-spacing:0.07em;font-weight:400">Status</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """, height=92 + (52 * len(combined)))

    # Charts
    st.markdown("<div style='height:14px;background:#0a0b14'></div>", unsafe_allow_html=True)
    render_component(
        """
        <div style='padding:28px 48px 0;background:#0a0b14'>
        <div style='font-family:DM Mono,monospace;font-size:11px;
        color:#6366f1;text-transform:uppercase;letter-spacing:0.12em;
        margin-bottom:14px'>Charts</div></div>
        """,
        height=54,
    )

    chart_l, chart_r = st.columns(2)

    with chart_l:
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        fig.patch.set_facecolor("#12141f")
        ax.set_facecolor("#12141f")
        brands_sorted = combined.sort_values("stability_score")
        colors = [stability_color(s) for s in brands_sorted["stability_score"]]
        ax.barh(brands_sorted["brand"], brands_sorted["stability_score"],
                color=colors, alpha=0.92,
                xerr=brands_sorted.get("std_position",
                     pd.Series([0]*len(brands_sorted))),
                error_kw={"ecolor": (1, 1, 1, 0.25), "capsize": 3})
        for idx, score in enumerate(brands_sorted["stability_score"]):
            ax.text(min(score + 0.03, 1.25), idx, f"{score:.2f}",
                    va="center", ha="left", color=colors[idx],
                    fontsize=10, fontfamily="DM Mono")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors="#9294a8", labelsize=10)
        ax.xaxis.label.set_color("#52546a")
        ax.set_xlabel("Score  (0 = unreliable · 1 = consistent)",
                      color="#52546a", size=10, labelpad=12)
        ax.set_title("Brand Stability Scores",
                     color="#f1f2f8", size=13, pad=12, loc="left",
                     fontweight="600")
        ax.grid(axis="x", color=(1, 1, 1, 0.06),
                linestyle="--", linewidth=0.5)
        ax.margins(x=0.15)
        plt.setp(ax.get_yticklabels(), color="#9294a8")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with chart_r:
        if not gemini_df.empty and not groq_df.empty:
            common = sorted(set(gemini_df["brand"]) & set(groq_df["brand"]))
            if common:
                gem_idx = gemini_df.set_index("brand")["avg_position"]
                grq_idx = groq_df.set_index("brand")["avg_position"]
                x = np.arange(len(common))
                fig2, ax2 = plt.subplots(figsize=(6.8, 4.6))
                fig2.patch.set_facecolor("#12141f")
                ax2.set_facecolor("#12141f")
                ax2.bar(x - 0.18, [gem_idx[b] for b in common],
                        0.35, label="Gemini", color="#2dd4bf", alpha=0.85)
                ax2.bar(x + 0.18, [grq_idx[b] for b in common],
                        0.35, label="Groq",   color="#a78bfa", alpha=0.85)
                ax2.set_xticks(x)
                ax2.set_xticklabels(common, rotation=25, ha="right",
                                    color="#9294a8", fontsize=10)
                ax2.invert_yaxis()
                for spine in ax2.spines.values():
                    spine.set_visible(False)
                ax2.tick_params(colors="#9294a8", labelsize=10)
                ax2.set_ylabel("Position  (1 = top of answer)",
                               color="#52546a", size=10)
                ax2.set_title("Gemini vs Groq — Avg Position",
                              color="#f1f2f8", size=13, pad=12, loc="left",
                              fontweight="600")
                ax2.grid(axis="y", color=(1, 1, 1, 0.06),
                         linestyle="--", linewidth=0.5)
                legend = ax2.legend(
                    facecolor="#1a1d2e",
                    edgecolor="#2a2d3e",
                    labelcolor="#9294a8",
                    fontsize=10
                )
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

    if not gap_df.empty:
        st.markdown("<div style='height:10px;background:#0a0b14'></div>", unsafe_allow_html=True)
        render_component(
            """
            <div style='padding:24px 48px 0;background:#0a0b14'>
            <div style='font-family:DM Mono,monospace;font-size:11px;
            color:#6366f1;text-transform:uppercase;letter-spacing:0.12em;
            margin-bottom:10px'>Drift Gap</div>
            <div style='font-size:14px;color:#9294a8'>
            Top brands where Gemini and Groq disagree most
            </div></div>
            """,
            height=66,
        )

        gap_focus = gap_df.head(6).copy()
        fig3, ax3 = plt.subplots(figsize=(12.4, 3.1))
        fig3.patch.set_facecolor("#12141f")
        ax3.set_facecolor("#12141f")
        bar_colors = []
        for gap in gap_focus["gap"]:
            if gap >= 2:
                bar_colors.append("#f472b6")
            elif gap >= 1:
                bar_colors.append("#fbbf24")
            else:
                bar_colors.append("#2dd4bf")
        ax3.barh(gap_focus["brand"], gap_focus["gap"], color=bar_colors, alpha=0.92)
        for idx, gap in enumerate(gap_focus["gap"]):
            ax3.text(gap + 0.05, idx, f"{gap:.1f}",
                     va="center", ha="left", color=bar_colors[idx],
                     fontsize=10, fontfamily="IBM Plex Mono")
        for spine in ax3.spines.values():
            spine.set_visible(False)
        ax3.tick_params(colors="#9294a8", labelsize=10)
        ax3.set_xlabel("Absolute position gap between engines",
                       color="#52546a", size=10, labelpad=12)
        ax3.set_title("Where the engines split most",
                      color="#f1f2f8", size=13, pad=12, loc="left",
                      fontweight="600")
        ax3.grid(axis="x", color=(1, 1, 1, 0.06),
                 linestyle="--", linewidth=0.5)
        plt.setp(ax3.get_yticklabels(), color="#9294a8")
        ax3.margins(x=0.15)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # Raw data expander
    st.markdown("<div style='height:24px;background:#0a0b14'></div>", unsafe_allow_html=True)
    render_component("""
    <div style="padding:28px 48px 18px;background:#0a0b14;
                font-family:'DM Sans',sans-serif">
        <div style="font-family:'DM Mono',monospace;font-size:11px;
                    color:#6366f1;text-transform:uppercase;
                    letter-spacing:0.12em;margin-bottom:6px">
            Raw Data
        </div>
        <div style="font-size:14px;color:#9294a8">
            View the underlying table and download a CSV
        </div>
    </div>
    """, height=90)
    st.markdown("<div style='height:16px;background:#0a0b14'></div>", unsafe_allow_html=True)
    with st.expander("View raw data"):
        st.dataframe(combined, use_container_width=True)
        csv = combined.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="geo_drift_results.csv",
            mime="text/csv"
        )


# ── Entry point ──────────────────────────────────────────────
try:
    init_state()
    if st.session_state.get("page", "home") == "how_it_works":
        render_how_it_works()
    elif st.session_state.get("page", "home") == "dashboard":
        render_dashboard()
    else:
        render_home()
except Exception as e:
    st.error(f"Something went wrong: {e}")
