"""
Multi-Style TTS Agent — Streamlit App
--------------------------------------
LangGraph workflow:  classify  →  (general | poem | news | joke)  →  TTS
Powered by OpenAI (gpt-4o-mini for text, tts-1 for audio).

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this file + requirements.txt to a GitHub repo
    2. Create a new app pointing to app.py
    3. In "Settings → Secrets" add:
        OPENAI_API_KEY = "sk-..."
"""

import os
import io
import re
from typing import TypedDict

import streamlit as st
from openai import OpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv


# ---------- Page config ----------
st.set_page_config(
    page_title="TTS Agent | Classify + Speak",
    page_icon="🎙️",
    layout="centered",
)

load_dotenv()  # picks up local .env in dev


def get_api_key() -> str:
    """Priority: Streamlit secrets → env var → empty string."""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")


# ---------- Agent state ----------
class AgentState(TypedDict):
    input_text: str
    processed_text: str
    audio_data: bytes
    audio_path: str
    content_type: str


VOICE_MAP = {
    "general": "alloy",
    "poem": "nova",
    "news": "onyx",
    "joke": "shimmer",
}


# ---------- Build the LangGraph workflow ----------
def build_workflow(client: OpenAI):
    """Build and compile the LangGraph workflow with the given OpenAI client."""

    def classify_content(state: AgentState) -> AgentState:
        # If user already picked a specific type in the sidebar, skip auto-classification
        if state.get("content_type") and state["content_type"] != "auto":
            return state

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the content as exactly ONE of: "
                        "'general', 'poem', 'news', 'joke'. "
                        "Reply with only that single word, lowercase, no punctuation."
                    ),
                },
                {"role": "user", "content": state["input_text"]},
            ],
        )
        ctype = response.choices[0].message.content.strip().lower()
        if ctype not in VOICE_MAP:
            ctype = "general"
        state["content_type"] = ctype
        return state

    def process_general(state: AgentState) -> AgentState:
        state["processed_text"] = state["input_text"]
        return state

    def process_poem(state: AgentState) -> AgentState:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Rewrite the following text as a short, beautiful poem:"},
                {"role": "user", "content": state["input_text"]},
            ],
        )
        state["processed_text"] = response.choices[0].message.content.strip()
        return state

    def process_news(state: AgentState) -> AgentState:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Rewrite the following text in a formal news anchor style:"},
                {"role": "user", "content": state["input_text"]},
            ],
        )
        state["processed_text"] = response.choices[0].message.content.strip()
        return state

    def process_joke(state: AgentState) -> AgentState:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Turn the following text into a short, funny joke:"},
                {"role": "user", "content": state["input_text"]},
            ],
        )
        state["processed_text"] = response.choices[0].message.content.strip()
        return state

    def text_to_speech(state: AgentState) -> AgentState:
        voice = VOICE_MAP.get(state["content_type"], "alloy")
        audio_buf = io.BytesIO()
        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=voice,
            input=state["processed_text"],
        ) as response:
            for chunk in response.iter_bytes():
                audio_buf.write(chunk)
        state["audio_data"] = audio_buf.getvalue()
        state["audio_path"] = ""
        return state

    workflow = StateGraph(AgentState)
    workflow.add_node("classify_content", classify_content)
    workflow.add_node("process_general", process_general)
    workflow.add_node("process_poem", process_poem)
    workflow.add_node("process_news", process_news)
    workflow.add_node("process_joke", process_joke)
    workflow.add_node("text_to_speech", text_to_speech)

    workflow.set_entry_point("classify_content")
    workflow.add_conditional_edges(
        "classify_content",
        lambda x: x["content_type"],
        {
            "general": "process_general",
            "poem": "process_poem",
            "news": "process_news",
            "joke": "process_joke",
        },
    )
    workflow.add_edge("process_general", "text_to_speech")
    workflow.add_edge("process_poem", "text_to_speech")
    workflow.add_edge("process_news", "text_to_speech")
    workflow.add_edge("process_joke", "text_to_speech")
    workflow.add_edge("text_to_speech", END)

    return workflow.compile()


def sanitize_filename(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9_\- ]+", "", text).strip().replace(" ", "_")
    return s[:max_len] or "tts_output"


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")

    default_key = get_api_key()
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=default_key,
        help="Used only for this session. You can also set OPENAI_API_KEY in .env or in Streamlit Secrets.",
    )

    mode = st.radio(
        "Content type",
        options=["auto", "general", "poem", "news", "joke"],
        index=0,
        help="Pick `auto` to let the agent classify the text for you.",
    )

    st.divider()
    st.markdown("**Voice mapping**")
    st.markdown(
        "- `general` → **alloy**\n"
        "- `poem` → **nova**\n"
        "- `news` → **onyx**\n"
        "- `joke` → **shimmer**"
    )

    st.divider()
    st.caption("Built with LangGraph + OpenAI + Streamlit")


# ---------- Main UI ----------
st.title("🎙️ MANISH SINGH - Multi-Style TTS Agent")
st.caption("Classify → Rewrite → Speak. The agent picks the best voice for the content.")

EXAMPLES = {
    "— pick an example —": "",
    "General": "The quick brown fox jumps over the lazy dog.",
    "Poem": "Roses are red, violets are blue, AI is amazing, and so are you!",
    "News": "Breaking news: Scientists discover a new species of deep-sea creature in the Mariana Trench.",
    "Joke": "Why don't scientists trust atoms? Because they make up everything!",
}

example_pick = st.selectbox("Quick examples", list(EXAMPLES.keys()), index=0)
default_text = EXAMPLES[example_pick]

input_text = st.text_area(
    "Enter your text",
    value=default_text,
    height=160,
    placeholder="Type or paste text to convert into speech...",
)

generate = st.button("🎧 Generate Speech", type="primary", use_container_width=True)


# ---------- Run pipeline ----------
if generate:
    api_key = api_key_input.strip()
    if not api_key:
        st.error("Please provide your OpenAI API key in the sidebar.")
        st.stop()
    if not input_text.strip():
        st.warning("Please enter some text first.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI(api_key=api_key)
    app = build_workflow(client)

    with st.spinner("Classifying → rewriting → synthesizing speech..."):
        try:
            result = app.invoke(
                {
                    "input_text": input_text,
                    "processed_text": "",
                    "audio_data": b"",
                    "audio_path": "",
                    "content_type": mode,
                }
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    # Result summary
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Detected type", result["content_type"].upper())
    with c2:
        st.metric("Voice", VOICE_MAP.get(result["content_type"], "alloy"))

    st.subheader("📝 Processed text")
    st.write(result["processed_text"])

    st.subheader("🔊 Audio")
    st.audio(result["audio_data"], format="audio/mp3")

    file_name = f"{result['content_type']}_{sanitize_filename(input_text)}.mp3"
    st.download_button(
        label="⬇️ Download MP3",
        data=result["audio_data"],
        file_name=file_name,
        mime="audio/mp3",
        use_container_width=True,
    )

    st.success("Done!")


# ---------- Footer ----------
st.divider()
with st.expander("📦 requirements.txt (for deployment)"):
    st.code(
        "streamlit>=1.32\n"
        "openai>=1.30\n"
        "langgraph>=0.2\n"
        "python-dotenv>=1.0\n",
        language="text",
    ) 
