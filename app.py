# app.py
import os, tempfile, logging, base64
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa, librosa.sequence as seq
import streamlit as st
import openai
import matplotlib.ticker as tk
from jiwer import wer
from difflib import ndiff
from Levenshtein import distance as levenshtein_distance
from audio_recorder_streamlit import audio_recorder

# ─── Page config must be first ───────────────────────────────
st.set_page_config(page_title="🗣️ AI Pronunciation Trainer", layout="wide")

# ─── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

# ─── OpenAI key from Secrets ─────────────────────────────────
if not st.secrets.get("OPENAI_API_KEY"):
    st.error("⚠️ Please set OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ─── Helpers ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def whisper_bytes(raw: bytes, lang: str = "he") -> str:
    """Transcribe raw audio via Whisper‑1, return lowercase text."""
    logging.info("→ whisper_bytes: writing temp file…")
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(raw); tmp.flush()
        logging.info("→ whisper_bytes: calling Whisper…")
        rsp = openai.audio.transcriptions.create(
            model="whisper-1",
            file=open(tmp.name, "rb"),
            language=lang,
            response_format="text"
        )
    text = rsp.strip().lower()
    logging.info(f"← whisper_bytes: received {len(text)} chars")
    return text

def load_pcm(raw: bytes, sr=16_000, min_sec=0.5):
    """Load raw bytes, resample/pad to fixed length."""
    logging.info("→ load_pcm: writing temp file…")
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(raw); tmp.flush()
        y, r = sf.read(tmp.name)
    if y.ndim > 1: y = y.mean(1)
    if r != sr:
        logging.info(f"→ load_pcm: resampling {r}→{sr}")
        y = librosa.resample(y, r, sr)
    if len(y) < min_sec * sr:
        pad_amt = int(min_sec * sr) - len(y)
        logging.info(f"→ load_pcm: padding {pad_amt} samples")
        y = np.pad(y, (0, pad_amt))
    return y.astype(np.float32)

def mfcc_dtw(y1, y2, sr=16_000):
    logging.info("→ mfcc_dtw: extracting MFCCs")
    m1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=13)
    m2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=13)
    D, _ = seq.dtw(m1, m2, metric="euclidean")
    cost = D[-1, -1] / D.shape[0]
    logging.info(f"← mfcc_dtw: cost={cost:.3f}")
    return cost

def feats(y, sr=16_000):
    logging.info("→ feats: extracting audio features")
    f = {}
    f["FFT mag"]       = float(np.mean(np.abs(np.fft.rfft(y))))
    f0, _, _           = librosa.pyin(y,
                             fmin=librosa.note_to_hz("C2"),
                             fmax=librosa.note_to_hz("C7"))
    f["Median f₀ Hz"]  = float(np.nanmedian(f0))
    melspec            = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    f["Mel (dB)"]      = float(np.mean(librosa.power_to_db(melspec)))
    mfccs              = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    f["MFCC mean"]     = float(np.mean(mfccs))
    f["Centroid"]      = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    f["Bandwidth"]     = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    f["Rolloff"]       = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    f["ZCR"]           = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    logging.info("← feats: done")
    return f

def diff_html(ref, hyp):
    out = []
    for seg in ndiff(ref.split(), hyp.split()):
        if seg.startswith("  "):
            out.append(f"<span style='color:#21c55d'>{seg[2:]}</span>")
        elif seg.startswith(("+ ","- ")):
            out.append(f"<span style='color:#ef4444'>{seg[2:]}</span>")
    return " ".join(out)

def mel_fig(y, sr=16_000):
    S = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128),
        ref=np.max
    )
    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(S, origin="lower", aspect="auto", cmap="magma")
    ax.set_xlabel("Frames"); ax.set_ylabel("Mel bins")
    plt.colorbar(im, ax=ax, pad=0.01)
    fig.tight_layout(); return fig

# ─── UI ──────────────────────────────────────────────────────
st.title("🗣️ AI Pronunciation Trainer")

col1, col2 = st.columns(2)
def audio_widget(label, col):
    with col:
        st.subheader(label)
        rec = audio_recorder(
            text=f"Record {label}",
            icon_size="2x",
            recording_color="#e91e63",
            neutral_color="#6c757d",
            sample_rate=16_000,
            pause_threshold=2.0
        )
        up = st.file_uploader(f"or upload {label}", type=["wav","mp3","flac"], key=label)
        if rec:
            st.audio(rec, format="audio/wav")
            return rec
        if up:
            st.audio(up)
            return up.read()
        st.info(f"Awaiting {label} audio")
        return None

teacher_raw = audio_widget("Teacher", col1)
student_raw = audio_widget("Student", col2)

run = st.button("▶ Analyse", type="primary",
                disabled=not (teacher_raw and student_raw))

if run:
    with st.spinner("Transcribing & scoring…"):
        logging.info("----- ANALYSIS START -----")

        # transcripts
        logging.info("Transcribing teacher…")
        t_txt = whisper_bytes(teacher_raw, lang="he")
        logging.info("Transcribing student…")
        s_txt = whisper_bytes(student_raw, lang="he")

        # audio→PCM
        t_y = load_pcm(teacher_raw)
        s_y = load_pcm(student_raw)

        # metrics
        lev = 1 - levenshtein_distance(t_txt, s_txt) / max(len(t_txt), len(s_txt), 1)
        dtw = mfcc_dtw(t_y, s_y)
        score = 0.7 * lev + 0.3 * (1 - min(dtw / 100, 1))

        # feature diffs
        f_ref = feats(t_y)
        diff  = {k: abs(f_ref[k] - feats(s_y)[k]) for k in f_ref}

        logging.info("----- ANALYSIS END -----")

    # render
    st.subheader("Transcripts")
    a, b = st.columns(2)
    a.write(t_txt); b.write(s_txt)

    st.subheader("Word‑level diff")
    st.markdown(diff_html(t_txt, s_txt), unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m3.metric("Similarity Score",    f"{lev:.2f}")
    m1.metric("Composite Score", f"{score*100:.1f}%")
    m2.metric("WER",            f"{wer(t_txt, s_txt):.0%}")


    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("#### Audio‑feature differences")
        st.table({k: [f"{v:.2f}"] for k, v in diff.items()})
    with c2:
        st.markdown("#### Mel‑spectrograms")
        p, q = st.columns(2)
        p.pyplot(mel_fig(t_y)); q.pyplot(mel_fig(s_y))

    # CSV report
    csv = "metric,value\n"
    for k, v in {
        **diff,
        "WER": wer(t_txt, s_txt),
        "Levenshtein": lev,
        "DTW": dtw,
        "Composite": score
    }.items():
        csv += f"{k},{v}\n"
    st.download_button("⬇ Download report", csv, "report.csv", "text/csv")
