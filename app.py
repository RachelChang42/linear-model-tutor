
import argparse
import hashlib
import json
import math
import os
import re
import sys
from html import escape
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI


st.set_page_config(
    page_title="Linear Model Tutor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

VAR_COLOR = "#404040"
VAR_BG = "#F4F4F4"
VAR_BORDER = "#D9D9D9"
DEFAULT_LOCAL_PROBLEM = "I want to study the relationship between mother's height and child's height."


APP_CSS = """
<style>
    :root {
        --chat-bg: #ffffff;
        --chat-text: #0d0d0d;
        --chat-muted: #6e6e6e;
        --chat-border: #e5e5e5;
        --chat-soft: #f4f4f4;
        --chat-soft-hover: #ececec;
        --chat-card: #ffffff;
        --chat-code: #f4f4f4;
        --chat-input: #ffffff;
        --chat-shadow: rgba(0, 0, 0, 0.04);
        --chat-focus: #0d0d0d;
    }

    html, body, .stApp {
        background: var(--chat-bg) !important;
        color: var(--chat-text) !important;
    }

    .block-container {
        max-width: 820px;
        padding-top: 0.65rem;
        padding-bottom: 7rem;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .top-icon-row {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        height: 2.15rem;
        margin-bottom: 0.15rem;
    }

    .hero-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: -0.035em;
        color: var(--chat-text);
        margin-top: 20vh;
        margin-bottom: 0.55rem;
    }

    .hero-subtitle {
        text-align: center;
        color: var(--chat-muted);
        font-size: 0.98rem;
        line-height: 1.5;
        max-width: 620px;
        margin: 0 auto 1.4rem auto;
    }

    .chat-shell {
        border: none;
        background: transparent;
        box-shadow: none;
        padding: 0;
    }

    .chat-message-row {
        display: flex;
        width: 100%;
        margin: 0.5rem 0;
    }

    .chat-message-row.user {
        justify-content: flex-end;
    }

    .chat-message-row.assistant {
        justify-content: flex-start;
    }

    .chat-bubble {
        max-width: min(78%, 680px);
        padding: 0.72rem 0.95rem;
        line-height: 1.55;
        font-size: 0.98rem;
        word-wrap: break-word;
        white-space: normal;
    }

    .chat-bubble.user {
        background: var(--chat-soft);
        color: var(--chat-text);
        border-radius: 22px 22px 6px 22px;
    }

    .chat-bubble.assistant {
        background: transparent;
        color: var(--chat-text);
        border-radius: 0;
        padding-left: 0.15rem;
    }

    .chat-bubble code {
        background: var(--chat-code);
        border: 1px solid var(--chat-border);
        border-radius: 6px;
        padding: 0.08rem 0.28rem;
        font-size: 0.92em;
        color: var(--chat-text);
    }

    .assistant-markdown-spacer {
        height: 0.12rem;
    }

    .assistant-message-row {
        width: 100%;
        margin: 0.75rem 0 0.9rem 0;
    }

    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        font-weight: 650;
        letter-spacing: -0.02em;
        line-height: 1.25;
        margin-top: 0.9rem;
        margin-bottom: 0.45rem;
        color: var(--chat-text);
    }

    [data-testid="stMarkdownContainer"] h1 {
        font-size: 1.36rem;
    }

    [data-testid="stMarkdownContainer"] h2 {
        font-size: 1.18rem;
    }

    [data-testid="stMarkdownContainer"] h3 {
        font-size: 1.04rem;
    }

    [data-testid="stMarkdownContainer"] p {
        line-height: 1.72;
        margin-bottom: 0.72rem;
    }

    [data-testid="stMarkdownContainer"] ul,
    [data-testid="stMarkdownContainer"] ol {
        margin-top: 0.25rem;
        margin-bottom: 0.85rem;
    }

    [data-testid="stMarkdownContainer"] li {
        margin-bottom: 0.28rem;
        line-height: 1.62;
    }

    [data-testid="stMarkdownContainer"] blockquote {
        border-left: 3px solid var(--chat-border);
        margin: 0.75rem 0;
        padding: 0.2rem 0 0.2rem 0.85rem;
        color: var(--chat-muted);
        background: transparent;
    }

    [data-testid="stMarkdownContainer"] pre {
        background: var(--chat-code) !important;
        border: 1px solid var(--chat-border);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
    }

    [data-testid="stMarkdownContainer"] code {
        background: var(--chat-code);
        border: 1px solid var(--chat-border);
        border-radius: 6px;
        padding: 0.08rem 0.25rem;
    }

    .new-chat-link {
        width: 2rem;
        height: 2rem;
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: var(--chat-muted);
        text-decoration: none;
        border: 1px solid transparent;
        transition: background 0.12s ease, border-color 0.12s ease, color 0.12s ease;
    }

    .new-chat-link:hover {
        background: var(--chat-soft);
        border-color: var(--chat-border);
        color: var(--chat-text);
    }

    .new-chat-link svg {
        width: 18px;
        height: 18px;
        stroke: currentColor;
        stroke-width: 1.8;
        fill: none;
        stroke-linecap: round;
        stroke-linejoin: round;
    }

    .attachment-card,
    .metric-box {
        border: 1px solid var(--chat-border);
        border-radius: 14px;
        padding: 0.85rem 0.95rem;
        background: var(--chat-card);
        box-shadow: 0 1px 2px var(--chat-shadow);
        margin-top: 0.7rem;
        color: var(--chat-text);
        line-height: 1.75;
        overflow-wrap: anywhere;
    }

    .attachment-title {
        font-weight: 600;
        color: var(--chat-text);
        font-size: 0.95rem;
        margin-bottom: 0.45rem;
    }

    .formula-line {
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--chat-text);
        padding: 0.3rem 0;
    }

    .metric-symbol {
        font-weight: 600;
        color: var(--chat-text);
    }

    .metric-value,
    .var-chip {
        display: inline-block;
        color: var(--chat-text);
        background: var(--chat-soft);
        border: 1px solid var(--chat-border);
        border-radius: 8px;
        padding: 0.02rem 0.28rem;
        margin: 0 0.04rem;
        font-weight: 500;
        white-space: nowrap;
    }

    div[data-testid="stButton"] button {
        border-radius: 8px;
        border: 1px solid var(--chat-border);
        background: var(--chat-card);
        color: var(--chat-text);
        font-weight: 500;
        min-height: 2.2rem;
        box-shadow: none;
    }

    div[data-testid="stButton"] button:hover {
        border-color: #c7c7c7;
        background: var(--chat-soft);
        color: var(--chat-text);
    }

    div[data-testid="stButton"] button:focus {
        box-shadow: 0 0 0 1px var(--chat-focus) inset !important;
        border-color: var(--chat-focus) !important;
    }

    [data-testid="stChatInput"] {
        max-width: 820px;
        margin: 0 auto;
    }

    [data-testid="stChatInput"] textarea {
        background: var(--chat-input) !important;
        color: var(--chat-text) !important;
        border: 1px solid var(--chat-border) !important;
        border-radius: 24px !important;
        box-shadow: 0 1px 8px var(--chat-shadow) !important;
        caret-color: var(--chat-text) !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: #c7c7c7 !important;
        box-shadow: 0 1px 8px var(--chat-shadow) !important;
    }

    textarea, input {
        border-radius: 12px !important;
        background: transparent !important;
        background-color: transparent !important;
        color: var(--chat-text) !important;
        border-color: var(--chat-border) !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: var(--chat-muted) !important;
    }

    [data-testid="stDataFrame"],
    [data-testid="stDataEditor"] {
        border-radius: 12px;
        overflow: hidden;
    }

    .stPlotlyChart {
        border: 1px solid var(--chat-border);
        border-radius: 14px;
        overflow: hidden;
        background: var(--chat-card);
        margin-top: 0.65rem;
    }

    [data-testid="stChatInput"] button,
    [data-testid="stChatInput"] button:hover,
    [data-testid="stChatInput"] button:focus,
    [data-testid="stChatInput"] button:active,
    [data-testid="stChatInputSubmitButton"],
    [data-testid="stChatInputSubmitButton"]:hover,
    [data-testid="stChatInputSubmitButton"]:focus,
    [data-testid="stChatInputSubmitButton"]:active {
        background: var(--chat-text) !important;
        border-color: var(--chat-text) !important;
        color: var(--chat-bg) !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] button svg,
    [data-testid="stChatInputSubmitButton"] svg {
        fill: var(--chat-bg) !important;
        stroke: var(--chat-bg) !important;
        color: var(--chat-bg) !important;
    }

    button[kind="primary"],
    button[kind="secondary"] {
        color: var(--chat-text) !important;
        border-color: var(--chat-border) !important;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --chat-bg: #212121;
            --chat-text: #ececec;
            --chat-muted: #b4b4b4;
            --chat-border: #3d3d3d;
            --chat-soft: #303030;
            --chat-soft-hover: #3a3a3a;
            --chat-card: #212121;
            --chat-code: #303030;
            --chat-input: #2f2f2f;
            --chat-shadow: rgba(0, 0, 0, 0.22);
            --chat-focus: #ececec;
        }

        html, body, .stApp {
            background: var(--chat-bg) !important;
            color: var(--chat-text) !important;
        }

        .stApp, .stApp p, .stApp div, .stApp span, .stApp label,
        .stApp [data-testid="stMarkdownContainer"] {
            color: var(--chat-text);
        }

        .hero-subtitle {
            color: var(--chat-muted) !important;
        }

        .attachment-card,
        .metric-box,
        .stPlotlyChart {
            background: var(--chat-card) !important;
            border-color: var(--chat-border) !important;
            color: var(--chat-text) !important;
        }

        .metric-value,
        .var-chip,
        .chat-bubble code {
            color: var(--chat-text) !important;
            background: var(--chat-soft) !important;
            border-color: var(--chat-border) !important;
        }

        div[data-testid="stButton"] button {
            background: var(--chat-card) !important;
            color: var(--chat-text) !important;
            border-color: var(--chat-border) !important;
        }

        div[data-testid="stButton"] button:hover {
            background: var(--chat-soft-hover) !important;
        }

        [data-testid="stChatInput"] textarea,
        input, textarea {
            color: var(--chat-text) !important;
            background-color: transparent !important;
            border-color: var(--chat-border) !important;
        }

        [data-testid="stSegmentedControl"] button,
        [data-testid="stRadio"] label {
            background: var(--chat-card) !important;
            color: var(--chat-text) !important;
            border-color: var(--chat-border) !important;
        }

        [data-testid="stSegmentedControl"] button[aria-pressed="true"],
        [data-testid="stRadio"] label:has(input:checked) {
            background: var(--chat-soft) !important;
            border-color: #666666 !important;
        }
    }


    /* v23 GPT-like composer overrides */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] * {
        display: none !important;
    }

    .composer-note {
        height: 0.05rem;
    }

    div[data-testid="stForm"] {
        max-width: 820px;
        margin: 0 auto 0.18rem auto;
        padding: 0.48rem 0.7rem 0.5rem 0.9rem !important;
        border: 1px solid var(--chat-border) !important;
        border-radius: 28px !important;
        background: var(--chat-bg) !important;
        box-shadow: 0 2px 14px var(--chat-shadow) !important;
    }

    div[data-testid="stForm"] textarea {
        min-height: 34px !important;
        height: auto !important;
        max-height: 170px !important;
        field-sizing: content !important;
        overflow-y: auto !important;
        border: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        background-color: transparent !important;
        color: var(--chat-text) !important;
        box-shadow: none !important;
        outline: none !important;
        padding: 0.2rem 0.1rem 0.1rem 0.1rem !important;
        font-size: 1rem !important;
        line-height: 1.35 !important;
        resize: none !important;
    }

    div[data-testid="stForm"] textarea:focus,
    div[data-testid="stForm"] textarea:active {
        border: 0 !important;
        box-shadow: none !important;
        outline: none !important;
    }

    div[data-testid="stForm"] textarea::placeholder {
        color: var(--chat-muted) !important;
        opacity: 1 !important;
    }

    div[data-testid="stForm"] div[data-testid="stSelectbox"] {
        max-width: 170px;
        min-width: 150px;
    }

    div[data-testid="stForm"] div[data-testid="stSelectbox"] label {
        display: none !important;
    }

    div[data-testid="stForm"] div[data-baseweb="select"] > div {
        min-height: 34px !important;
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        color: var(--chat-muted) !important;
        padding-left: 0 !important;
    }

    div[data-testid="stForm"] div[data-baseweb="select"] > div:hover,
    div[data-testid="stForm"] div[data-baseweb="select"] > div:focus,
    div[data-testid="stForm"] div[data-baseweb="select"] > div:focus-within {
        border: 0 !important;
        box-shadow: none !important;
        outline: none !important;
        background: transparent !important;
    }

    div[data-testid="stForm"] div[data-baseweb="select"] span,
    div[data-testid="stForm"] div[data-baseweb="select"] svg {
        color: var(--chat-muted) !important;
        fill: var(--chat-muted) !important;
    }

    div[data-testid="stFormSubmitButton"] {
        display: flex !important;
        justify-content: flex-end !important;
    }

    div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stFormSubmitButton"] button:hover,
    div[data-testid="stFormSubmitButton"] button:focus,
    div[data-testid="stFormSubmitButton"] button:active {
        width: 42px !important;
        height: 42px !important;
        min-height: 42px !important;
        padding: 0 !important;
        border-radius: 999px !important;
        border: 0 !important;
        background: var(--chat-text) !important;
        color: var(--chat-bg) !important;
        box-shadow: none !important;

        position: relative !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    div[data-testid="stFormSubmitButton"] button p {
        display: none !important;
    }

    div[data-testid="stFormSubmitButton"] button::before {
        content: "";
        position: absolute;
        left: 50%;
        top: 50%;
        width: 27px;
        height: 27px;
        display: block;
        background: var(--chat-bg);

        mask-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M12 19V5M12 5L5.5 11.5M12 5L18.5 11.5' stroke='black' stroke-width='2.25' stroke-linecap='round' stroke-linejoin='round' fill='none'/%3E%3C/svg%3E");
        mask-repeat: no-repeat;
        mask-position: center;
        mask-size: contain;

        -webkit-mask-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M12 19V5M12 5L5.5 11.5M12 5L18.5 11.5' stroke='black' stroke-width='2.25' stroke-linecap='round' stroke-linejoin='round' fill='none'/%3E%3C/svg%3E");
        -webkit-mask-repeat: no-repeat;
        -webkit-mask-position: center;
        -webkit-mask-size: contain;

        transform: translate(-50%, -50%);
    }

    div[data-testid="stForm"] [data-baseweb="input"],
    div[data-testid="stForm"] [data-baseweb="textarea"] {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }

    div[data-baseweb="popover"],
    div[role="listbox"] {
        border-color: var(--chat-border) !important;
        background: var(--chat-card) !important;
        color: var(--chat-text) !important;
    }

    div[role="option"] {
        color: var(--chat-text) !important;
    }

    div[role="option"][aria-selected="true"],
    div[role="option"]:hover {
        background: var(--chat-soft) !important;
        color: var(--chat-text) !important;
    }

    /* v25 composer final overrides */
    div[data-testid="stForm"] [data-baseweb="textarea"] textarea,
    div[data-testid="stForm"] [data-baseweb="textarea"] {
        background: transparent !important;
        background-color: transparent !important;
    }

    div[data-testid="stForm"] [data-testid="stTextArea"] {
        background: transparent !important;
    }

</style>
"""


# ==============================
# Runtime LLM config
# ==============================

def parse_runtime_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


RUNTIME_ARGS = parse_runtime_args()

MODEL_TIERS = {
    "medium": {
        "label": "Medium",
        "model": "deepseek-v4-flash",
        "api_key_names": ["MEDIUM_LLM_API_KEY", "DEEPSEEK_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"],
        "base_url_names": ["MEDIUM_LLM_BASE_URL", "DEEPSEEK_BASE_URL", "LLM_BASE_URL", "OPENAI_BASE_URL"],
        "model_names": ["MEDIUM_LLM_MODEL"],
    },
    "high": {
        "label": "High",
        "model": "deepseek-v4-pro",
        "api_key_names": ["HIGH_LLM_API_KEY", "DEEPSEEK_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"],
        "base_url_names": ["HIGH_LLM_BASE_URL", "DEEPSEEK_BASE_URL", "LLM_BASE_URL", "OPENAI_BASE_URL"],
        "model_names": ["HIGH_LLM_MODEL"],
    },
}


def get_active_model_tier() -> str:
    tier = str(st.session_state.get("model_tier", "medium")).lower()
    return tier if tier in MODEL_TIERS else "medium"


def first_config_value(names: List[str]) -> Optional[str]:
    for name in names:
        value = os.getenv(name) or secret_value(name)
        if value:
            return str(value)
    return None


def secret_value(name: str) -> Optional[str]:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        return None
    return None


def get_llm_config() -> Dict[str, Optional[str]]:
    tier = get_active_model_tier()
    tier_cfg = MODEL_TIERS[tier]

    api_key = RUNTIME_ARGS.api_key or first_config_value(tier_cfg["api_key_names"])
    base_url = RUNTIME_ARGS.base_url or first_config_value(tier_cfg["base_url_names"])
    model = first_config_value(tier_cfg["model_names"]) or tier_cfg["model"]

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "tier": tier,
        "tier_label": tier_cfg["label"],
    }


def get_client() -> Optional[OpenAI]:
    cfg = get_llm_config()
    if not cfg["api_key"]:
        return None
    if cfg["base_url"]:
        return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    return OpenAI(api_key=cfg["api_key"])


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_llm_json(messages: List[Dict[str, str]], temperature: float = 0.45) -> Optional[Dict[str, Any]]:
    client = get_client()
    if client is None:
        st.session_state.last_error = "API key is not configured."
        return None

    cfg = get_llm_config()
    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return extract_json_object(content)
    except Exception:
        try:
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content or "{}"
            return extract_json_object(content)
        except Exception as error:
            st.session_state.last_error = f"Could not connect to the API: {error}"
            return None


# ==============================
# State
# ==============================

def init_state() -> None:
    defaults = {
        "lesson": None,
        "data": pd.DataFrame(columns=["x", "y"]),
        "phase": "intake",
        "learner_state": "new",
        "depth_mode": "core",
        "learning_goal": "intuition",
        "misconception_type": "none",
        "introduced_concepts": [],
        "model_tier": "medium",
        "slope": None,
        "intercept": None,
        "best_slope": None,
        "best_intercept": None,
        "last_error": None,
        "pending_user_text": None,
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "I can teach linear models step by step. "
                    "Don’t worry if you’re not sure where to begin — we’ll build the idea together. "
                    "If anything feels confusing, tell me right away and I’ll slow down.\n\n"
                    "What example would you like to start with: weather, sports, animals, school, daily life, or your own idea?"
                ),
                "artifact": "none",
            }
        ],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_all() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()
    st.rerun()


# ==============================
# Lesson generation
# ==============================

LESSON_GENERATOR_SYSTEM = """
You create one-input, one-output linear-model learning examples for a high-school student.
Return only valid JSON.
Use exactly 7 data points.
The hidden data relationship should be roughly linear but not perfect.
Do not use advanced statistics language.
Important: the opening story must NOT reveal the linear pattern. It should only set up a concrete situation.
Make the data-collection story realistic. Do not force "each day" or "over a week" unless the x variable is genuinely time measured by days.
""".strip()

LESSON_GENERATOR_USER = """
The student wrote:
{student_request}

Create or translate this into a linear-model learning example.

Return this JSON object:
{{
  "title": "short title",
  "story": "one friendly paragraph that is ONLY a concrete story. Bold the two measured quantities using markdown **like this**. Do not explain the pattern. Do not say linear, line, roughly, trend, increase, decrease, relationship, correlation, slope, prediction, model, input, or output in the story.",
  "question": "one practical question",
  "data_intro": "one realistic sentence saying how the data were collected. If the data are about several objects or people, say they measured several objects/people once; do not mention days or a week unless the x variable is day/time. Do not explain the pattern here either.",
  "x_name": "input variable name",
  "x_unit": "unit, or empty string",
  "y_name": "output variable name",
  "y_unit": "unit, or empty string",
  "x_sentence": "In this example, x is ...",
  "y_sentence": "In this example, y is ...",
  "data": [{{"x": number, "y": number}}, ... exactly 7 rows]
}}

The story should feel like the first scene of a lesson, not a statistics explanation.
Good story style:
"During a zombie outbreak, a student wrote down the **number of zombies nearby** and the **number of people eaten** in several neighborhoods. We will use these records as our example."
"For a tree study, a student measured the **age of each tree** and the **height of each tree** for seven maple trees in a park."

Bad story style:
"As the number of zombies increases, the number of people eaten tends to increase roughly linearly."

If the request is vague, choose a simple beginner-friendly example.
If the request is a real problem, keep the original situation recognizable.
""".strip()


STORY_FORBIDDEN_WORDS = {
    "linear", "line", "straight", "roughly", "trend", "increase", "increases", "decrease", "decreases",
    "tends", "relationship", "correlation", "slope", "prediction", "predict", "model", "pattern"
}


def clean_opening_story(story: str, lesson: Dict[str, Any]) -> str:
    """Keep the first story concrete. Remove sentences that prematurely reveal the pattern."""
    raw = (story or "").strip()
    pieces = re.split(r"(?<=[.!?])\s+", raw)
    kept = []
    for piece in pieces:
        plain = re.sub(r"[*_`]", "", piece).lower()
        if any(word in plain for word in STORY_FORBIDDEN_WORDS):
            continue
        if piece.strip():
            kept.append(piece.strip())

    cleaned = " ".join(kept).strip()
    if len(cleaned) >= 40:
        return cleaned

    x_name = str(lesson.get("x_name", "one quantity")).strip() or "one quantity"
    y_name = str(lesson.get("y_name", "another quantity")).strip() or "another quantity"
    title = str(lesson.get("title", "this situation")).strip() or "this situation"
    return (
        f"In {title.lower()}, someone wrote down **{x_name}** and **{y_name}** each day. "
        "We will use these records as our example."
    )


def clean_data_intro(data_intro: str, lesson: Dict[str, Any]) -> str:
    raw = (data_intro or "").strip()
    plain = re.sub(r"[*_`]", "", raw).lower()

    x_name = str(lesson.get("x_name", "")).lower()
    # Allow "over a week/each day" only when the x variable is literally a calendar/time index.
    # Variables such as tree age, house age, or hours studied should not force a daily/weekly collection story.
    calendar_time_words = {"day", "days", "date", "week", "weeks", "time index"}
    x_is_calendar_time_like = any(word in x_name for word in calendar_time_words)

    if any(word in plain for word in STORY_FORBIDDEN_WORDS):
        raw = ""

    if ("week" in plain or "each day" in plain or "every day" in plain) and not x_is_calendar_time_like:
        raw = ""

    if raw:
        return raw

    x_label = str(lesson.get("x_name", "x")).strip() or "x"
    y_label = str(lesson.get("y_name", "y")).strip() or "y"
    return f"Someone collected seven observations of **{x_label}** and **{y_label}** for this example."


def sanitize_lesson(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        st.session_state.last_error = "The API did not return a valid JSON object."
        return None

    required_text_fields = [
        "title",
        "story",
        "question",
        "data_intro",
        "x_name",
        "x_unit",
        "y_name",
        "y_unit",
        "x_sentence",
        "y_sentence",
    ]

    lesson: Dict[str, Any] = {}
    for key in required_text_fields:
        value = raw.get(key)
        if not isinstance(value, str):
            st.session_state.last_error = f"The API response is missing the field: {key}."
            return None
        lesson[key] = value.strip()

    lesson["story"] = clean_opening_story(lesson.get("story", ""), lesson)
    lesson["data_intro"] = clean_data_intro(lesson.get("data_intro", ""), lesson)

    rows = raw.get("data")
    if not isinstance(rows, list):
        st.session_state.last_error = "The API response is missing the data table."
        return None

    cleaned = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            x = float(row.get("x"))
            y = float(row.get("y"))
            if math.isfinite(x) and math.isfinite(y):
                cleaned.append({"x": round(x, 3), "y": round(y, 3)})
        except Exception:
            continue

    if len(cleaned) < 2:
        st.session_state.last_error = "The API response did not include enough valid data points."
        return None

    rng = np.random.default_rng(abs(hash(lesson.get("title", ""))) % (2**32))
    rng.shuffle(cleaned)
    lesson["data"] = cleaned[:7]
    return lesson


def make_lesson(student_request: str) -> Optional[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": LESSON_GENERATOR_SYSTEM},
        {"role": "user", "content": LESSON_GENERATOR_USER.format(student_request=student_request)},
    ]
    raw = call_llm_json(messages, temperature=0.65)
    if raw is None:
        return None
    return sanitize_lesson(raw)


# ==============================
# Data/model helpers
# ==============================

def lesson_to_df(lesson: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(lesson.get("data", []))
    if df.empty:
        return pd.DataFrame(columns=["x", "y"])
    df = df[["x", "y"]].copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df.dropna().reset_index(drop=True)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["x", "y"])
    out = df.copy()
    if "x" not in out.columns:
        out["x"] = np.nan
    if "y" not in out.columns:
        out["y"] = np.nan
    out = out[["x", "y"]]
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    return out.dropna().reset_index(drop=True)


def fit_line(df: pd.DataFrame) -> Tuple[float, float]:
    data = clean_df(df)
    if len(data) < 2 or data["x"].nunique() < 2:
        return 1.0, 0.0
    slope, intercept = np.polyfit(data["x"].to_numpy(), data["y"].to_numpy(), 1)
    return float(slope), float(intercept)


def predictions(df: pd.DataFrame, slope: float, intercept: float) -> pd.DataFrame:
    out = clean_df(df).copy()
    if out.empty:
        out["y_hat"] = []
        out["error"] = []
        return out
    out["y_hat"] = slope * out["x"] + intercept
    out["error"] = out["y"] - out["y_hat"]
    return out


def sse(df: pd.DataFrame, slope: float, intercept: float) -> float:
    pe = predictions(df, slope, intercept)
    if pe.empty:
        return float("nan")
    return float(np.sum(pe["error"] ** 2))


def mse(df: pd.DataFrame, slope: float, intercept: float) -> float:
    pe = predictions(df, slope, intercept)
    if pe.empty:
        return float("nan")
    return float(np.mean(pe["error"] ** 2))


def r_squared(df: pd.DataFrame, slope: float, intercept: float) -> float:
    data = clean_df(df)
    if data.empty:
        return float("nan")
    ss_res = sse(data, slope, intercept)
    ss_tot = float(np.sum((data["y"] - data["y"].mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def ranges(df: pd.DataFrame) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    data = clean_df(df)
    if data.empty:
        return (0.0, 10.0), (0.0, 10.0)
    xmin, xmax = float(data["x"].min()), float(data["x"].max())
    ymin, ymax = float(data["y"].min()), float(data["y"].max())
    if xmin == xmax:
        xmin -= 1
        xmax += 1
    if ymin == ymax:
        ymin -= 1
        ymax += 1
    xpad = 0.08 * (xmax - xmin)
    ypad = 0.16 * (ymax - ymin)
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)


def axis_title(name: str, unit: str) -> str:
    return f"{name} ({unit})" if unit else name


def fmt_num(value: Optional[float]) -> str:
    if value is None:
        return "a value inside the data range"
    value = float(value)
    if not math.isfinite(value):
        return "a value inside the data range"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def prediction_practice_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Pick a safe prediction-practice x: inside the observed range and not already in the data."""
    data = clean_df(df)
    if data.empty:
        return {
            "x_values_observed": [],
            "x_range": {"min": None, "max": None},
            "suggested_prediction_x": None,
            "prediction_policy_note": "No data yet.",
        }

    xs = sorted({float(x) for x in data["x"].to_list() if math.isfinite(float(x))})
    if not xs:
        return {
            "x_values_observed": [],
            "x_range": {"min": None, "max": None},
            "suggested_prediction_x": None,
            "prediction_policy_note": "No valid x values yet.",
        }

    suggested = None
    if len(xs) >= 2:
        center = (xs[0] + xs[-1]) / 2
        intervals = [(abs(((xs[i] + xs[i + 1]) / 2) - center), xs[i + 1] - xs[i], xs[i], xs[i + 1]) for i in range(len(xs) - 1) if xs[i + 1] > xs[i]]
        if intervals:
            intervals.sort(key=lambda item: (item[0], -item[1]))
            _, _, left, right = intervals[0]
            suggested = round((left + right) / 2, 3)

    if suggested is None or any(abs(suggested - x) < 1e-9 for x in xs):
        span = xs[-1] - xs[0]
        if span > 0:
            for frac in [0.55, 0.45, 0.65, 0.35, 0.75, 0.25]:
                candidate = round(xs[0] + frac * span, 3)
                if xs[0] <= candidate <= xs[-1] and all(abs(candidate - x) > 1e-9 for x in xs):
                    suggested = candidate
                    break

    note = (
        "Use suggested_prediction_x for ordinary prediction practice. It is inside the observed x range and not an observed x value. "
        "Do not ask for predictions far outside the observed x range unless teaching extrapolation as an extension."
    )
    return {
        "x_values_observed": [round(x, 3) for x in xs],
        "x_range": {"min": round(xs[0], 3), "max": round(xs[-1], 3)},
        "suggested_prediction_x": suggested,
        "prediction_policy_note": note,
    }


def apply_shift(direction: str) -> None:
    df = clean_df(st.session_state.data)
    if st.session_state.slope is None or st.session_state.intercept is None:
        st.session_state.slope, st.session_state.intercept = fit_line(df)
    _, yr = ranges(df)
    dy = 0.01 * (yr[1] - yr[0])
    if direction == "down":
        dy = -dy
    st.session_state.intercept = float(st.session_state.intercept + dy)


def apply_rotation(direction: str) -> None:
    df = clean_df(st.session_state.data)
    if df.empty:
        return
    if st.session_state.slope is None or st.session_state.intercept is None:
        st.session_state.slope, st.session_state.intercept = fit_line(df)

    slope = float(st.session_state.slope)
    intercept = float(st.session_state.intercept)
    pivot_x = float(df["x"].mean())
    pivot_y = slope * pivot_x + intercept

    angle = math.atan(slope)
    delta = math.radians(1)
    new_angle = angle - delta if direction == "clockwise" else angle + delta
    new_angle = max(math.radians(-80), min(math.radians(80), new_angle))

    new_slope = math.tan(new_angle)
    st.session_state.slope = float(new_slope)
    st.session_state.intercept = float(pivot_y - new_slope * pivot_x)


def plot_model(
    df: pd.DataFrame,
    lesson: Dict[str, Any],
    show_line: bool = False,
    show_errors: bool = False,
    slope: Optional[float] = None,
    intercept: Optional[float] = None,
    height: int = 390,
) -> go.Figure:
    data = clean_df(df)
    xr, yr = ranges(data)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="markers",
            name="real y",
            marker=dict(size=7, color="#111111", opacity=0.92),
            hovertemplate="x=%{x}<br>y=%{y}<extra></extra>",
        )
    )

    if show_line and slope is not None and intercept is not None and not data.empty:
        x_line = np.linspace(xr[0], xr[1], 100)
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="model line",
                line=dict(width=2.4, color="#111111"),
                hoverinfo="skip",
            )
        )

        pe = predictions(data, slope, intercept)
        if show_errors:
            seg_x, seg_y = [], []
            x_span = xr[1] - xr[0]
            y_span = yr[1] - yr[0]
            for _, row in pe.iterrows():
                actual_x = float(row["x"])
                actual_y = float(row["y"])
                pred_y = float(row["y_hat"])
                err = float(row["error"])
                seg_x.extend([actual_x, actual_x, None])
                seg_y.extend([actual_y, pred_y, None])

                error_side = 1 if err >= 0 else -1
                normal_length = math.sqrt(float(slope) ** 2 + 1.0)
                label_x = actual_x + error_side * (-float(slope) / normal_length) * 0.045 * x_span
                label_y = actual_y + error_side * (1.0 / normal_length) * 0.085 * y_span

                fig.add_annotation(
                    x=label_x,
                    y=label_y,
                    text=f"<b>ε={err:.1f}</b>",
                    showarrow=False,
                    font=dict(size=11, color=VAR_COLOR),
                    xanchor="center",
                    opacity=0.98,
                )

            fig.add_trace(
                go.Scatter(
                    x=seg_x,
                    y=seg_y,
                    mode="lines",
                    name="ε",
                    line=dict(width=1.35, color="#111111", dash="dot"),
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=pe["x"],
                    y=pe["y_hat"],
                    mode="markers",
                    name="predicted ŷ",
                    marker=dict(size=7, color="#6B7280", opacity=0.95),
                    hovertemplate="x=%{x}<br>ŷ=%{y:.2f}<extra></extra>",
                )
            )

    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=24, b=12),
        xaxis_title=axis_title(lesson.get("x_name", "x"), lesson.get("x_unit", "")),
        yaxis_title=axis_title(lesson.get("y_name", "y"), lesson.get("y_unit", "")),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(range=xr, zeroline=False, showgrid=True, gridcolor="rgba(128,128,128,0.22)")
    fig.update_yaxes(range=yr, zeroline=False, showgrid=True, gridcolor="rgba(128,128,128,0.22)")
    return fig


def chip(value: str) -> str:
    return f"<span class='metric-value'>{escape(value)}</span>"


def metric_html(df: pd.DataFrame, slope: float, intercept: float) -> str:
    pe = predictions(df, slope, intercept)
    errors = [float(e) for e in pe["error"].to_list()]
    squared = [e * e for e in errors]
    n = len(errors)
    current_sse = float(sum(squared))
    current_mse = current_sse / n if n else float("nan")
    current_r2 = r_squared(df, slope, intercept)
    terms = " + ".join([f"{chip(f'{e:.1f}')}²" for e in errors])
    values = " + ".join([chip(f"{v:.2f}") for v in squared])

    return f"""
<div class='metric-box'>
<span class='metric-symbol'>Sum of squared errors (SSE)</span> = {terms} = {values} = {chip(f'{current_sse:.2f}')}<br>
<span class='metric-symbol'>Mean squared error (MSE)</span> = SSE / n = {chip(f'{current_sse:.2f}')} / {chip(str(n))} = {chip(f'{current_mse:.2f}')}<br>
<span class='metric-symbol'>R²</span> = 1 − SSE / total variation in y = {chip(f'{current_r2:.3f}') if math.isfinite(current_r2) else 'not defined'}<br><br>
R² is a number we calculate from the sum of squared errors. It indicates how well the model explains y. The closer to 1, the better.
</div>
"""


# ==============================
# Teacher agent
# ==============================

TEACHER_SYSTEM_PROMPT = """
You are one warm one-on-one teaching agent for high-school students learning linear models.

You have two internal responsibilities:
1. Teach the mathematical/statistical content clearly.
2. Use the adaptive teaching policy to decide what depth, path, and next learning goal fits the student now.

Do not split these into separate agents. The final reply must directly respond to the student's actual words.

Core idea:
For most students, a linear model may be their first statistical model. Teach this central idea early and clearly:
A model is not reality. A linear model is a simple approximation that uses a straight-line pattern to make useful predictions, while real data can still miss the line.

Curriculum map:
Use this as a flexible map, not a progress checklist. The student does not need to finish the whole route.
- example_context: the concrete situation chosen by the student.
- data_table: a small table of observed x and y values.
- scatter_plot: a graph of the data points.
- x_variable: x as the input / independent variable.
- y_variable: y as the output / dependent variable.
- pattern_direction: whether y tends to increase, decrease, or show no clear direction as x changes.
- roughly_linear_pattern: the idea that points can roughly follow a straight-line pattern without being perfect.
- line_as_summary: a line can summarize the overall pattern.
- model_as_approximation: a model is a useful simplification of reality, not reality itself.
- model_as_prediction: a model is a rule for predicting y from x.
- y_hat: ŷ means the model's predicted y.
- straight_line_formula: ŷ = A x + b.
- slope_A: A controls how steeply the prediction changes.
- intercept_b: b is where the line would cross the y-axis at x = 0, but it may or may not have a meaningful real-world interpretation.
- prediction_vs_real_value: predicted y and real y are different ideas.
- error_epsilon: epsilon is the gap between real y and predicted y.
- full_model_equation: Y = AX + b + epsilon.
- manual_model_adjustment: changing A and b changes the line and the errors.
- squared_error: squaring errors avoids cancellation and penalizes large errors.
- SSE: sum of squared errors.
- MSE: mean squared error.
- R_squared: a rough evaluation measure; useful later, but easy to overinterpret.
- best_fit: the fitted line is the best line according to a criterion.
- least_squares: least squares finds the line with small squared errors.
- extension_topic: optional deeper topics such as R, Python, multiple inputs, assumptions, causation, extrapolation, uncertainty, or interpretation.

Learning path targets:
Choose a path target based on the student's current evidence.

Gentle path target:
The student should understand that data points can show a trend, and a straight line can give a rough but imperfect guess. Use the "model is an approximation, not reality" idea in plain language.

Core path target:
The student should understand that a linear model uses a straight line to predict y from x, that ŷ is the predicted y, and that errors show how far predictions miss real values. The model-as-approximation idea belongs here, not only in deep mode.

Stretch path target:
The student should begin to explain why fitting means choosing a line with small errors, why extrapolation is risky, and why a pattern does not automatically prove causation.

Deep path target:
The student can handle the real statistical/mathematical structure: least squares as minimizing a loss, residuals, assumptions, uncertainty, multiple regression, and limits of interpretation. Do not turn this into a detached mini-lecture; answer the student's actual question and add only one meaningful deeper move.

Learning goals:
Return learning_goal every time. This is the main thing the current reply is trying to accomplish.
Allowed values:
- intuition: build the picture of data, trend, line, and model-as-approximation.
- prediction: use the line/model to predict a new y and understand ŷ.
- formula: interpret ŷ = A x + b, A, and b.
- error: distinguish real y from predicted ŷ and understand error/residual.
- fitting: understand that fitting chooses a line that misses the data less.
- evaluation: discuss quality, R², diagnostics, extrapolation, or causation limits.
- deep_extension: answer a deeper student-led question, such as optimization, assumptions, uncertainty, or multiple inputs.

Misconception handling:
If the student is wrong or confused, identify the likely misconception internally and return a misconception_type.
Allowed values:
- none
- graph_reading
- x_y_reversal
- prediction_vs_actual
- deterministic_model
- slope_intercept
- causation
- extrapolation
- formula_manipulation
- overfitting
- metric_misinterpretation
Address only that misconception in the reply. Do not pile on multiple corrections.

Depth mode:
Use depth_mode to choose how deep the next response should be. It is a temporary teaching mode, not a permanent label.
Allowed values:
- gentle: use when the student is confused, overloaded, or low-confidence. Use everyday words, fewer symbols, and one very small question.
- core: use for the normal main route. Give clear intuition, one useful takeaway, and one next step.
- stretch: use when the student is following well, answers direct checks correctly, or can handle a small challenge. Add one deeper idea or one challenge question.
- deep: use when the student asks "why", mentions optimization/least squares/residuals/assumptions/R²/causation/extrapolation, or shows strong mathematical/statistical curiosity. Give real statistical meaning, but keep it readable for a high-school student.

Depth switching rules:
- If the student says "I don't understand", "what does this mean", gives an unrelated answer, or seems frustrated: set learner_state = "confused" and depth_mode = "gentle".
- If the student says only "continue", "ok", or "yes": keep depth_mode = "core" unless the context already clearly supports a deeper mode.
- If the student answers a concrete check correctly: do not ask for the same confirmation again; move one small step forward. Usually set depth_mode = "core" or "stretch".
- If the student asks a relevant why/how/limits question: set learner_state = "curious" and depth_mode = "stretch" or "deep".
- If the student asks about optimization, least squares, residuals, matrix form, assumptions, uncertainty, causation, or extrapolation: set depth_mode = "deep".
- If the student is practicing with controls or calculations: set learner_state = "practicing"; depth_mode can be core or stretch depending on the student's response.

Takeaway policy:
Every response should leave one useful understanding behind. If the student stops after this response, they should still have learned something.
Examples by path:
- Gentle: A straight line is a simple way to summarize a trend, not a perfect description of reality.
- Core: ŷ is the model's predicted y, while y is the real observed y.
- Core: A linear model gives useful predictions, but the real data can still be above or below the line.
- Stretch: Fitting means choosing the line whose errors are small overall.
- Stretch: A strong pattern does not automatically prove that x causes y.
- Deep: Least squares chooses parameters by minimizing a loss over residuals.

Conversational deepening:
Do not create a separate "mini-lesson" unless the student explicitly asks for a full explanation. When the student asks a deeper question, answer it directly and keep it tied to the current example.
Deep/high mode may move faster, and it may connect several related ideas, but the structure must be clear:
- use short paragraphs,
- use bold section labels when helpful,
- order ideas in a logical chain,
- avoid one dense paragraph of terminology,
- end with a concrete, answerable next step.
Depth means clearer structure and stronger ideas, not a wall of text.
Good:
"**What fitting means**\n\nA fitted line is not the true reality; it is the best approximation under a chosen rule.\n\n**The rule here**\n\nLeast squares chooses A and b to make Σ(yᵢ − (Axᵢ+b))² as small as possible.\n\nWant to compare your line with the fitted line using this rule?"
Bad:
"Mini-lesson 1: Loss functions. Mini-lesson 2: Projection. Mini-lesson 3: Assumptions."

Formatting policy:
- Use markdown for readability.
- For gentle/core replies, keep the reply visually simple: 1 to 3 short paragraphs.
- For stretch/deep replies, use short section labels such as **Idea**, **Why it matters**, or **How to read it** when it improves clarity.
- Put formulas on their own line or in a short code block when helpful.
- Do not write a technical paragraph longer than about 3 lines.
- If a reply has more than 3 sentences or introduces technical content, use blank lines to separate ideas.
- Use bullets only for steps or comparisons, not for every reply.
- Do not over-format simple encouragement or simple checks.
- If Data workspace is shown, do not reproduce the data table in the text. The workspace already contains the table.
- Do not output markdown tables for the lesson data unless the student explicitly asks for a separate table.

Early-story rule:
- The opening story should not teach "linear", "line", "roughly linear", slope, formula, prediction, model, or model-as-approximation.
- If the student asks about "linear", "line", "roughly linear", "trend", or similar terms before seeing the data, treat this as confusion caused by introducing the term too early.
- In that case, do not merely paraphrase the word. Briefly say that the word is easier to understand after seeing the data, then show the data and ask the student to notice what they see.

Teaching loop:
1. Read the full available context, especially the latest student message and the previous assistant question.
2. Decide the current learning_goal and depth_mode.
3. Teach one small idea or respond to one student need.
4. Leave one useful takeaway.
5. End with exactly one concrete question or one clear next action.

Avoid illogical next questions:
- Do not ask whether the student wants to see a visual that you are already showing in the same response.
- If artifact = "line", do not end by asking "Would you like to see a line?" or "Would you like to see a line that fits the data?"
- Do not ask a question whose answer you just stated in the previous sentence.
- If you just explained what slope means, do not ask "What does this slope mean?" Instead ask the student to apply it to a small change, or simply move on.
- Do not introduce "fit", "fitted line", or "best fit" before the fitting idea has actually been introduced.
- Before fitting is introduced, call the line a simple line, model line, or line summary, not "the fitted line."

Evidence policy:
Strong evidence:
- The student correctly answers a concrete check question.
- The student connects the idea to the example in their own words.
- The student asks a relevant deeper question.
Moderate evidence:
- The student gives a short but correct answer.
- The student says "continue" after a simple idea.
Weak evidence:
- The student only says "ok", "yes", or gives a vague answer after a hard idea.
Negative evidence:
- The student says they are confused.
- The answer contradicts the graph/data.
- The student mixes up actual y and predicted ŷ, x and y, association and causation, or interpolation and extrapolation.

Decision rules:
- With strong evidence, move forward one useful step.
- With moderate evidence, build on the correct part and clarify only the missing part.
- With weak evidence, do not introduce a hard new concept. Ask a concrete check or give one small takeaway.
- With negative evidence, reduce difficulty, return to the example/visual, and address one misconception.
- Do not force the student to explain every obvious answer.
- Do not ask "why?" after every correct answer.
- Do not over-teach intuitive steps.

Prediction and extrapolation policy:
- For ordinary prediction practice, choose an x value inside the observed data range, or very close to it.
- Prefer the suggested_prediction_x from the current tutoring context.
- suggested_prediction_x is chosen to be inside the observed x range and not already present in the data.
- Do not ask the student to predict y for an x value that is already in the data, unless the goal is to compare real y with predicted ŷ.
- Do not choose a round number such as x = 10 just because it is convenient.
- Do not ask for predictions far outside the observed x range during the ordinary lesson.
- Extrapolation means predicting outside the observed data range. Treat extrapolation as an evaluation/extension topic.
- Only discuss extrapolation after the student understands model_as_prediction and y_hat, or if the student explicitly asks about predicting beyond the data.
- If you discuss extrapolation, clearly say that it is riskier because it is outside the data we observed.

Formula and intercept policy:
- Do not introduce ŷ = A x + b before the student understands prediction.
- Do not introduce A and b before the student understands ŷ.
- When explaining slope, do not immediately ask the student to repeat the exact meaning you just gave. Ask an application question instead, such as how much ŷ changes when x increases by 2.
- When explaining b/intercept, say it is where the line would cross the y-axis at x = 0.
- Add a caution: sometimes x = 0 has a real meaning, but sometimes it is outside the meaningful data range, so b should not be over-interpreted.
- If the reply asks about b as the value at x = 0, use artifact = "line" so the student can see the line.

Error and fitting policy:
- Do not introduce error epsilon before the student has seen that prediction and real value can differ.
- Do not use error_plot when asking the student to calculate an error.
- Use error_plot only after the student has already tried calculating or identifying the error.
- For most students, teach error before SSE/MSE/R².
- Do not introduce SSE, MSE, or R² in the first hands-on model-control step.
- R² is not part of the default core path. Use it only in evaluation/stretch/deep mode or when the student asks how to measure model quality.
- Least squares belongs in stretch/deep, or after the student understands individual errors and fitting intuition.
- In stretch/deep mode, SSE, MSE, R², and least squares may be discussed faster, but only with clear structure and a logical chain. Do not dump them into one dense paragraph.
- For least squares, prefer the chain: error → squared error → total squared error/SSE → least squares as minimizing that total.
- For MSE and R², make clear that they answer different questions: MSE is about size of misses; R² is about explained variation. Include a caution that R² can be overinterpreted.

Challenge question bank:
Use these naturally, not as a quiz list.
Gentle:
- "Is the line meant to hit every point perfectly, or just summarize the pattern?"
- "Is this a perfect rule, or a useful rough guess?"
Core:
- "What does ŷ mean in this example?"
- "Is this number the real y or the predicted ŷ?"
Stretch:
- "Would you trust the same line far outside the data range?"
- "Does a pattern prove that x causes y?"
- "If two lines both look reasonable, how could we decide which one is better?"
Deep:
- "What quantity are we minimizing when we fit the line?"
- "Why might a curve through every point predict worse later?"

Pacing rules:
- In each response, introduce at most one new formal concept.
- Keep each teaching response short, usually 2 to 5 sentences.
- If a graph, table, or model-control artifact is shown, make the text even shorter because the visual already contains information.
- Do not combine formula, prediction, slope, and intercept in one response.
- Do not introduce y-hat before the student understands prediction.
- Do not introduce A and b before the student understands y-hat.
- Do not introduce error epsilon before the student has seen that prediction and real value can differ.
- Do not introduce SSE, MSE, or R-squared before the student understands individual errors.
- Never say "to be continued."

Learner state:
Set learner_state as a temporary judgment about the current moment, not a permanent label.
Allowed values:
- new: the student has just started.
- confused: the student seems lost, overloaded, frustrated, or gives an unrelated/contradictory answer.
- following: the student is following the current idea but has not necessarily shown readiness for a harder concept.
- ready: the student has shown enough evidence to move one micro-step forward.
- curious: the student asks a relevant deeper question or wants an extension.
- practicing: the student is using data, graphs, controls, or calculations.

Artifact policy:
An artifact is optional visual support for the current reply. It is not a curriculum button.
Choose an artifact only when it naturally supports what the reply says.
If an artifact is shown, the reply should not ask whether the student wants to see that same artifact.
If artifact = "data", keep the text brief and do not repeat the table values in prose or a markdown table.
Allowed artifacts:
- none: no visual support needed.
- story: show the example story.
- data: show the data table and scatter plot.
- line: show a line visual.
- error_plot: show a line with error gaps.
- model_controls: show controls for changing A and b and seeing errors.
- best_fit: show the fitted line.

Use artifact rules:
- If the reply asks the student to read the situation, use artifact = "story".
- If the reply asks the student to inspect data or notice a pattern, use artifact = "data".
- If the student is confused by "linear" or "line" before seeing data, use artifact = "data", not "story".
- If the reply introduces or visually discusses a straight line, use artifact = "line".
- If the reply asks the student to compare predicted and real values or notice gaps after the student has already tried identifying an error, use artifact = "error_plot".
- If the reply asks the student to calculate or identify an error for the first time, use artifact = "none", not "error_plot".
- If the reply asks the student to try changing the line, use artifact = "model_controls".
- If the reply discusses fitting or the best line, use artifact = "best_fit".
- If no visual support is needed, use artifact = "none".

Phase policy:
Allowed phases:
- intake
- story
- data
- pattern
- prediction
- formula
- error
- fitting
- evaluation
- extension

Phase is a rough location in the curriculum, not a required next step. Do not advance phase unless the student response gives enough evidence or the student explicitly asks to move on.

introduced_concepts policy:
Return the full updated list of concepts that have actually been introduced to the student.
Only include a concept after it has appeared in the student-facing reply or in an artifact shown with the reply.
Do not add future concepts just because they are on the curriculum map.

Allowed concept names:
example_context, data_table, scatter_plot, x_variable, y_variable, pattern_direction, roughly_linear_pattern, line_as_summary, model_as_approximation, model_as_prediction, y_hat, straight_line_formula, slope_A, intercept_b, prediction_vs_real_value, error_epsilon, full_model_equation, manual_model_adjustment, squared_error, SSE, MSE, R_squared, best_fit, least_squares, extension_topic

Return only valid JSON:
{
  "reply": "teacher's message to the student",
  "phase": "intake|story|data|pattern|prediction|formula|error|fitting|evaluation|extension",
  "learner_state": "new|confused|following|ready|curious|practicing",
  "depth_mode": "gentle|core|stretch|deep",
  "learning_goal": "intuition|prediction|formula|error|fitting|evaluation|deep_extension",
  "misconception_type": "none|graph_reading|x_y_reversal|prediction_vs_actual|deterministic_model|slope_intercept|causation|extrapolation|formula_manipulation|overfitting|metric_misinterpretation",
  "introduced_concepts": ["concept_name", "..."],
  "artifact": "none|story|data|line|error_plot|model_controls|best_fit"
}
""".strip()


def local_teacher_response(user_text: str) -> Dict[str, Any]:
    phase = st.session_state.phase
    lesson = st.session_state.lesson
    if phase == "intake" or lesson is None:
        return {
            "reply": "Great — that’s a fun example. Let’s start with a short story first. Read it, then say “continue”, and we’ll see how a linear model starts to help step by step.",
            "phase": "story",
            "learner_state": "new",
            "depth_mode": "core",
            "learning_goal": "intuition",
            "misconception_type": "none",
            "introduced_concepts": ["example_context"],
            "artifact": "story",
        }
    return {
        "reply": "Let’s continue one small step at a time. What part should we look at next?",
        "phase": phase,
        "learner_state": st.session_state.learner_state,
        "depth_mode": st.session_state.depth_mode,
        "learning_goal": st.session_state.learning_goal,
        "misconception_type": st.session_state.misconception_type,
        "introduced_concepts": st.session_state.introduced_concepts,
        "artifact": "none",
    }


ALLOWED_PHASES = {
    "intake", "story", "data", "pattern", "prediction", "formula", "error", "fitting", "evaluation", "extension"
}

ALLOWED_LEARNER_STATES = {
    "new", "confused", "following", "ready", "curious", "practicing"
}

ALLOWED_DEPTH_MODES = {
    "gentle", "core", "stretch", "deep"
}

ALLOWED_LEARNING_GOALS = {
    "intuition", "prediction", "formula", "error", "fitting", "evaluation", "deep_extension"
}

ALLOWED_MISCONCEPTION_TYPES = {
    "none", "graph_reading", "x_y_reversal", "prediction_vs_actual", "deterministic_model",
    "slope_intercept", "causation", "extrapolation", "formula_manipulation",
    "overfitting", "metric_misinterpretation"
}

ALLOWED_ARTIFACTS = {
    "none", "story", "data", "line", "error_plot", "model_controls", "best_fit"
}

ALLOWED_CONCEPTS = [
    "example_context",
    "data_table",
    "scatter_plot",
    "x_variable",
    "y_variable",
    "pattern_direction",
    "roughly_linear_pattern",
    "line_as_summary",
    "model_as_approximation",
    "model_as_prediction",
    "y_hat",
    "straight_line_formula",
    "slope_A",
    "intercept_b",
    "prediction_vs_real_value",
    "error_epsilon",
    "full_model_equation",
    "manual_model_adjustment",
    "squared_error",
    "SSE",
    "MSE",
    "R_squared",
    "best_fit",
    "least_squares",
    "extension_topic",
]
ALLOWED_CONCEPT_SET = set(ALLOWED_CONCEPTS)


def merge_introduced_concepts(raw: Any) -> List[str]:
    existing = st.session_state.get("introduced_concepts", [])
    merged: List[str] = []
    for concept in existing:
        if concept in ALLOWED_CONCEPT_SET and concept not in merged:
            merged.append(concept)

    if isinstance(raw, list):
        for concept in raw:
            if isinstance(concept, str):
                concept = concept.strip()
                if concept in ALLOWED_CONCEPT_SET and concept not in merged:
                    merged.append(concept)

    return merged


def normalize_artifact(raw: Any) -> str:
    artifact = str(raw or "none").strip()
    old_to_new = {
        "show_story": "story",
        "show_data": "data",
        "show_line": "line",
        "show_error_plot": "error_plot",
        "show_quality_controls": "model_controls",
        "show_best_fit": "best_fit",
    }
    artifact = old_to_new.get(artifact, artifact)
    return artifact if artifact in ALLOWED_ARTIFACTS else "none"


def get_teacher_response(user_text: str) -> Dict[str, Any]:
    if st.session_state.lesson is None:
        return local_teacher_response(user_text)

    lesson = st.session_state.lesson
    data = clean_df(st.session_state.data)
    line_text = "No line yet."
    if st.session_state.slope is not None and st.session_state.intercept is not None:
        line_text = (
            f"Current line: y_hat = {st.session_state.slope:.4f}x + {st.session_state.intercept:.4f}; "
            f"SSE={sse(data, st.session_state.slope, st.session_state.intercept):.4f}; "
            f"MSE={mse(data, st.session_state.slope, st.session_state.intercept):.4f}; "
            f"R2={r_squared(data, st.session_state.slope, st.session_state.intercept):.4f}."
        )

    pred_ctx = prediction_practice_context(data)
    context = {
        "phase": st.session_state.phase,
        "learner_state": st.session_state.learner_state,
        "depth_mode": st.session_state.depth_mode,
        "learning_goal": st.session_state.learning_goal,
        "misconception_type": st.session_state.misconception_type,
        "introduced_concepts": st.session_state.introduced_concepts,
        "lesson": lesson,
        "data": data.to_dict(orient="records"),
        "x_values_observed": pred_ctx.get("x_values_observed", []),
        "x_range": pred_ctx.get("x_range", {"min": None, "max": None}),
        "suggested_prediction_x": pred_ctx.get("suggested_prediction_x"),
        "prediction_policy_note": pred_ctx.get("prediction_policy_note", ""),
        "line_status": line_text,
        "recent_dialogue": st.session_state.messages[-16:],
        "latest_student_message": user_text,
    }

    messages = [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "system", "content": "Current tutoring context: " + json.dumps(context, ensure_ascii=False)},
        {"role": "user", "content": user_text},
    ]
    raw = call_llm_json(messages, temperature=0.45)
    if not isinstance(raw, dict):
        detail = st.session_state.get("last_error") or "The API did not return a valid response."
        return {
            "reply": f"I can’t connect to the model right now, so I can’t answer this properly. Details: {detail}",
            "phase": st.session_state.phase,
            "learner_state": st.session_state.learner_state,
            "depth_mode": st.session_state.depth_mode,
            "learning_goal": st.session_state.learning_goal,
            "misconception_type": st.session_state.misconception_type,
            "introduced_concepts": st.session_state.introduced_concepts,
            "artifact": "none",
        }

    phase = str(raw.get("phase", st.session_state.phase)).strip()
    learner_state = str(raw.get("learner_state", st.session_state.learner_state)).strip()
    depth_mode = str(raw.get("depth_mode", st.session_state.depth_mode)).strip()
    learning_goal = str(raw.get("learning_goal", st.session_state.learning_goal)).strip()
    misconception_type = str(raw.get("misconception_type", "none")).strip()
    artifact = normalize_artifact(raw.get("artifact", "none"))

    if phase not in ALLOWED_PHASES:
        phase = st.session_state.phase
    if learner_state not in ALLOWED_LEARNER_STATES:
        learner_state = st.session_state.learner_state
    if depth_mode not in ALLOWED_DEPTH_MODES:
        depth_mode = st.session_state.depth_mode
    if learning_goal not in ALLOWED_LEARNING_GOALS:
        learning_goal = st.session_state.learning_goal
    if misconception_type not in ALLOWED_MISCONCEPTION_TYPES:
        misconception_type = "none"

    return {
        "reply": str(raw.get("reply", "")).strip() or "I can’t answer properly because the model returned an empty response. Could you try again?",
        "phase": phase,
        "learner_state": learner_state,
        "depth_mode": depth_mode,
        "learning_goal": learning_goal,
        "misconception_type": misconception_type,
        "introduced_concepts": merge_introduced_concepts(raw.get("introduced_concepts")),
        "artifact": artifact,
    }


def markdown_bold_to_html(text: str) -> str:
    safe = escape(text or "")
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)


# ==============================
# UI attachments
# ==============================

def render_story_attachment() -> None:
    lesson = st.session_state.lesson
    if not lesson:
        return
    story_html = markdown_bold_to_html(lesson.get("story", ""))
    st.markdown(
        f"""
<div class='attachment-card'>
<div class='attachment-title'>{escape(lesson.get('title', 'Example'))}</div>
{story_html}
</div>
""",
        unsafe_allow_html=True,
    )


def render_data_attachment(message_index: int) -> None:
    lesson = st.session_state.lesson
    if not lesson:
        return
    st.markdown(
        f"""
<div class='attachment-card'>
<div class='attachment-title'>Data workspace</div>
{escape(lesson.get('data_intro', 'Someone collected data over a week for this example.'))}<br>
You can click to edit, add, or remove data.
</div>
""",
        unsafe_allow_html=True,
    )
    col_table, col_plot = st.columns([0.36, 0.64], gap="medium")
    with col_table:
        edited = st.data_editor(
            st.session_state.data,
            key=f"data_editor_{message_index}",
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            height=275,
            column_config={
                "x": st.column_config.NumberColumn("x", step=0.1, width="small"),
                "y": st.column_config.NumberColumn("y", step=0.1, width="small"),
            },
        )
        st.session_state.data = clean_df(edited)
    with col_plot:
        st.plotly_chart(plot_model(st.session_state.data, lesson, height=310), use_container_width=True, key=f"data_plot_{message_index}")


def render_line_attachment(message_index: int) -> None:
    lesson = st.session_state.lesson
    if not lesson:
        return
    data = clean_df(st.session_state.data)
    if st.session_state.slope is None or st.session_state.intercept is None:
        st.session_state.slope, st.session_state.intercept = fit_line(data)

    st.markdown(
        """
<div class='attachment-card'>
<div class='attachment-title'>Line visual</div>
A straight line is drawn as a simple visual summary of the overall pattern.
</div>
""",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        plot_model(data, lesson, show_line=True, slope=st.session_state.slope, intercept=st.session_state.intercept, height=380),
        use_container_width=True,
        key=f"line_plot_{message_index}",
    )


def render_error_attachment(message_index: int) -> None:
    lesson = st.session_state.lesson
    if not lesson:
        return
    data = clean_df(st.session_state.data)
    if st.session_state.slope is None or st.session_state.intercept is None:
        st.session_state.slope, st.session_state.intercept = fit_line(data)

    st.markdown(
        """
<div class='attachment-card'>
<div class='attachment-title'>Error ε</div>
The real <strong>y</strong> and the predicted <strong>ŷ</strong> are not always the same.
Their difference is called an <span class='var-chip'>error</span>, and we write it as <strong>ε</strong>.<br>
<div class='formula-line'>ε = y − ŷ</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        plot_model(
            data, lesson, show_line=True, show_errors=True,
            slope=st.session_state.slope, intercept=st.session_state.intercept, height=390
        ),
        use_container_width=True,
        key=f"error_plot_{message_index}",
    )



def render_quality_attachment(message_index: int) -> None:
    lesson = st.session_state.lesson
    if not lesson:
        return
    data = clean_df(st.session_state.data)
    if st.session_state.slope is None or st.session_state.intercept is None:
        st.session_state.slope, st.session_state.intercept = fit_line(data)

    st.markdown(
        """
<div class='attachment-card'>
<div class='attachment-title'>Try changing the model</div>
Every straight line refers to a different model. Click ↑ or ↓ to move the line, click ↺ or ↻ to rotate it, or type to change the parameters.
</div>
""",
        unsafe_allow_html=True,
    )

    c_a, c_b, c_up, c_down, c_ccw, c_cw, c_reset = st.columns([1.2, 1.2, 0.62, 0.62, 0.62, 0.62, 0.78])
    with c_a:
        st.session_state.slope = st.number_input("A", value=float(st.session_state.slope), step=0.05, format="%.4f", key=f"slope_input_{message_index}")
    with c_b:
        st.session_state.intercept = st.number_input("b", value=float(st.session_state.intercept), step=0.5, format="%.4f", key=f"intercept_input_{message_index}")
    with c_up:
        st.write("")
        if st.button("↑", use_container_width=True, key=f"shift_up_{message_index}"):
            apply_shift("up")
            st.rerun()
    with c_down:
        st.write("")
        if st.button("↓", use_container_width=True, key=f"shift_down_{message_index}"):
            apply_shift("down")
            st.rerun()
    with c_ccw:
        st.write("")
        if st.button("↺", use_container_width=True, key=f"rotate_ccw_{message_index}"):
            apply_rotation("counterclockwise")
            st.rerun()
    with c_cw:
        st.write("")
        if st.button("↻", use_container_width=True, key=f"rotate_cw_{message_index}"):
            apply_rotation("clockwise")
            st.rerun()
    with c_reset:
        st.write("")
        if st.button("Reset", use_container_width=True, key=f"reset_line_{message_index}"):
            st.session_state.slope, st.session_state.intercept = fit_line(data)
            st.rerun()

    slope = float(st.session_state.slope)
    intercept = float(st.session_state.intercept)

    st.plotly_chart(
        plot_model(data, lesson, show_line=True, show_errors=True, slope=slope, intercept=intercept, height=390),
        use_container_width=True,
        key=f"quality_plot_{message_index}",
    )



def render_bestfit_attachment(message_index: int) -> None:
    lesson = st.session_state.lesson
    if not lesson:
        return
    data = clean_df(st.session_state.data)
    best_slope, best_intercept = fit_line(data)
    st.session_state.best_slope = best_slope
    st.session_state.best_intercept = best_intercept

    st.markdown(
        f"""
<div class='attachment-card'>
<div class='attachment-title'>Fitted line</div>
This line is the one that makes the overall squared errors as small as possible for this data.<br>
For this dataset, the fitted line has A = <strong>{best_slope:.3f}</strong> and b = <strong>{best_intercept:.3f}</strong>.
</div>
""",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        plot_model(data, lesson, show_line=True, show_errors=True, slope=best_slope, intercept=best_intercept, height=405),
        use_container_width=True,
        key=f"bestfit_plot_{message_index}",
    )


def render_artifact(artifact: str, idx: int) -> None:
    artifact = normalize_artifact(artifact)
    if artifact == "story":
        render_story_attachment()
    elif artifact == "data":
        render_data_attachment(idx)
    elif artifact == "line":
        render_line_attachment(idx)
    elif artifact == "error_plot":
        render_error_attachment(idx)
    elif artifact == "model_controls":
        render_quality_attachment(idx)
    elif artifact == "best_fit":
        render_bestfit_attachment(idx)



# ==============================
# Main chat UI
# ==============================

def render_header() -> None:
    st.markdown(
        """
<div class='top-icon-row'>
  <a class='new-chat-link' href='?new_chat=1' title='New conversation' aria-label='New conversation'>
    <svg viewBox='0 0 24 24'>
      <path d='M12 20h9'/>
      <path d='M16.5 3.5a2.1 2.1 0 0 1 3 3L8 18l-4 1 1-4Z'/>
    </svg>
  </a>
</div>
""",
        unsafe_allow_html=True,
    )

    show_intro = st.session_state.lesson is None and not any(m.get("role") == "user" for m in st.session_state.messages)
    if show_intro:
        st.markdown("<div class='hero-title'>Linear Model Tutor</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='hero-subtitle'>I can teach linear models step by step. Start with an example, a question, or a real-world problem.</div>",
            unsafe_allow_html=True,
        )


def message_text_to_html(content: str) -> str:
    safe = escape(content or "")
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = re.sub(r"`([^`]+)`", r"<code>\1</code>", safe)
    safe = safe.replace("\n", "<br>")
    return safe


def render_messages() -> None:
    st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
    for idx, msg in enumerate(st.session_state.messages):
        role = msg.get("role", "assistant")
        if role == "user":
            content_html = message_text_to_html(msg.get("content", ""))
            st.markdown(
                f"<div class='chat-message-row user'><div class='chat-bubble user'>{content_html}</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='assistant-markdown-spacer'></div>", unsafe_allow_html=True)
            st.markdown(msg.get("content", ""))
            render_artifact(msg.get("artifact", "none"), idx)
    st.markdown("</div>", unsafe_allow_html=True)


GRAPH_ARTIFACTS = {"data", "line", "error_plot", "model_controls", "best_fit"}


def stable_float(value: Any, digits: int = 6) -> Optional[float]:
    try:
        value = float(value)
        if math.isfinite(value):
            return round(value, digits)
    except Exception:
        pass
    return None


def data_hash(df: pd.DataFrame) -> str:
    data = clean_df(df)
    if data.empty:
        records: List[Dict[str, float]] = []
    else:
        rounded = data.copy()
        rounded["x"] = rounded["x"].round(6)
        rounded["y"] = rounded["y"].round(6)
        rounded = rounded.sort_values(["x", "y"]).reset_index(drop=True)
        records = rounded.to_dict(orient="records")
    payload = json.dumps(records, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def artifact_fingerprint(artifact: str) -> Optional[str]:
    artifact = normalize_artifact(artifact)
    if artifact not in GRAPH_ARTIFACTS:
        return None

    data = clean_df(st.session_state.data)
    payload: Dict[str, Any] = {
        "artifact": artifact,
        "data_hash": data_hash(data),
    }

    if artifact in {"line", "error_plot", "model_controls"}:
        payload["slope"] = stable_float(st.session_state.slope)
        payload["intercept"] = stable_float(st.session_state.intercept)

    if artifact == "best_fit":
        payload["best_slope"] = stable_float(st.session_state.best_slope)
        payload["best_intercept"] = stable_float(st.session_state.best_intercept)

    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def artifact_fingerprint_seen(fingerprint: Optional[str]) -> bool:
    if not fingerprint:
        return False
    for msg in st.session_state.get("messages", []):
        if msg.get("role") == "assistant" and msg.get("artifact_fingerprint") == fingerprint:
            return True
    return False


def reply_asks_for_error_calculation(reply: str) -> bool:
    text = (reply or "").lower()
    patterns = [
        "what is the error",
        "what's the error",
        "calculate the error",
        "compute the error",
        "find the error",
        "identify the error",
        "what does the error equal",
        "what does that equal",
        "real y minus predicted",
        "actual minus predicted",
        "observed minus predicted",
        "y − ŷ",
        "y - ŷ",
        "y minus y-hat",
        "actual y minus predicted",
    ]
    return any(pattern in text for pattern in patterns)


def reply_asks_to_see_line(reply: str) -> bool:
    text = (reply or "").lower()
    patterns = [
        "would you like to see a line",
        "want to see a line",
        "shall we draw a line",
        "should we draw a line",
        "would you like to see the line",
        "want to see the line",
    ]
    return any(pattern in text for pattern in patterns)


def sanitize_artifact_for_reply(artifact: str, reply: str) -> Tuple[str, Optional[str]]:
    artifact = normalize_artifact(artifact)

    if artifact == "error_plot" and reply_asks_for_error_calculation(reply):
        return "none", None

    if artifact == "line" and reply_asks_to_see_line(reply):
        return "none", None

    fingerprint = artifact_fingerprint(artifact)
    if artifact_fingerprint_seen(fingerprint):
        return "none", None

    return artifact, fingerprint


def process_user_message(user_text: str) -> None:
    if st.session_state.lesson is None:
        with st.spinner("Connecting to the model..."):
            lesson = make_lesson(user_text)
        if lesson is None:
            detail = st.session_state.get("last_error") or "The API is unavailable."
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"I can’t connect to the model right now, so I can’t answer or generate a lesson. Details: {detail}",
                    "artifact": "none",
                }
            )
            return
        st.session_state.lesson = lesson
        st.session_state.data = lesson_to_df(lesson)
        st.session_state.slope = None
        st.session_state.intercept = None
        response = local_teacher_response(user_text)
    else:
        with st.spinner("Thinking..."):
            response = get_teacher_response(user_text)

    st.session_state.phase = response.get("phase", st.session_state.phase)
    st.session_state.learner_state = response.get("learner_state", st.session_state.learner_state)
    st.session_state.depth_mode = response.get("depth_mode", st.session_state.depth_mode)
    st.session_state.learning_goal = response.get("learning_goal", st.session_state.learning_goal)
    st.session_state.misconception_type = response.get("misconception_type", st.session_state.misconception_type)
    st.session_state.introduced_concepts = merge_introduced_concepts(response.get("introduced_concepts"))

    raw_artifact = normalize_artifact(response.get("artifact", "none"))
    reply_text = response.get("reply", "Let's continue.")

    if raw_artifact in {"line", "error_plot", "model_controls"}:
        if st.session_state.slope is None or st.session_state.intercept is None:
            st.session_state.slope, st.session_state.intercept = fit_line(st.session_state.data)

    if raw_artifact == "best_fit":
        st.session_state.best_slope, st.session_state.best_intercept = fit_line(st.session_state.data)

    artifact, artifact_fp = sanitize_artifact_for_reply(raw_artifact, reply_text)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply_text,
            "artifact": artifact,
            "artifact_fingerprint": artifact_fp,
        }
    )


def queue_user_message(user_text: str) -> None:
    user_text = user_text.strip()
    if not user_text:
        return

    if user_text.lower() in {"start over", "reset conversation", "new conversation", "/reset"}:
        reset_all()
        return

    st.session_state.messages.append({"role": "user", "content": user_text, "artifact": "none"})
    st.session_state.pending_user_text = user_text
    st.rerun()


def process_pending_user_message() -> None:
    pending = st.session_state.get("pending_user_text")
    if not pending:
        return

    st.session_state.pending_user_text = None
    process_user_message(pending)
    st.rerun()


def handle_new_chat_query() -> None:
    try:
        if st.query_params.get("new_chat"):
            st.query_params.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_state()
    except Exception:
        pass


def render_composer() -> None:
    label_to_tier = {cfg["label"]: tier for tier, cfg in MODEL_TIERS.items()}
    labels = [MODEL_TIERS["medium"]["label"], MODEL_TIERS["high"]["label"]]
    current_tier = get_active_model_tier()
    current_label = MODEL_TIERS[current_tier]["label"]
    current_index = labels.index(current_label) if current_label in labels else 0

    st.markdown("<div class='composer-note'></div>", unsafe_allow_html=True)

    with st.form("chat_composer_form", clear_on_submit=True):
        prompt = st.text_area(
            "Message",
            key="composer_prompt",
            label_visibility="collapsed",
            placeholder="Type your answer, question, topic, or problem here",
            height=40,
        )

        col_model, col_space, col_send = st.columns([1.65, 5.15, 0.78], gap="small")
        with col_model:
            choice = st.selectbox(
                "Model",
                options=labels,
                index=current_index,
                label_visibility="collapsed",
                key="composer_model_label",
            )
        with col_send:
            submitted = st.form_submit_button(" ", use_container_width=True)

    if submitted:
        if choice in label_to_tier:
            st.session_state.model_tier = label_to_tier[choice]
        if prompt and prompt.strip():
            queue_user_message(prompt)


def main() -> None:
    init_state()
    handle_new_chat_query()
    st.markdown(APP_CSS, unsafe_allow_html=True)
    render_header()
    render_messages()
    process_pending_user_message()
    render_composer()


if __name__ == "__main__":
    main()
