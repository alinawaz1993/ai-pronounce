# app.py
import os
import io
import logging
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.sequence as seq
import streamlit as st
import openai
from jiwer import wer
from difflib import ndiff
from Levenshtein import distance as levenshtein_distance
from audio_recorder_streamlit import audio_recorder
from concurrent.futures import ThreadPoolExecutor

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

# ─── Helpers ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def whisper_bytes(raw: bytes, lang: str = "he") -> str:
    """Transcribe raw audio via Whisper‑1, return lowercase text."""
    logging.info("→ whisper_bytes: in‑memory transcription")
    bio = io.BytesIO(raw)
    bio.name = "audio.wav"
    rsp = openai.audio.transcriptions.create(
        model="whisper-1",
        file=bio,
        language=lang,
        response_format="text"
    )
    text = rsp.strip().lower()
    logging.info(f"← whisper_bytes: received {len(text)} chars")
    return text

def load_pcm(raw: bytes, sr: int = 16_000, min_sec: float = 0.5):
    """Load raw bytes, resample/pad to fixed length without disk I/O."""
    logging.info("→ load_pcm: loading from memory")
    bio = io.BytesIO(raw)
    bio.name = "audio.wav"
    y, orig_sr = sf.read(bio)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if orig_sr != sr:
        logging.info(f"→ load_pcm: resampling {orig_sr}→{sr}")
        # keyword args for newer librosa
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    if len(y) < sr * min_sec:
        pad_amt = int(sr * min_sec) - len(y)
        logging.info(f"→ load_pcm: padding {pad_amt} samples")
        y = np.pad(y, (0, pad_amt))
    return y.astype(np.float32)

@st.cache_data(show_spinner=False)
def feats(y: np.ndarray, sr: int = 16_000) -> dict:
    """Extract audio features in one STFT pass."""
    logging.info("→ feats: extracting audio features")
    f = {}
    f["FFT mag"] = float(np.mean(np.abs(np.fft.rfft(y))))
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    melspec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64, power=2.0
    )
    mel_db = librosa.power_to_db(melspec)
    f["Mel (dB)"] = float(np.mean(mel_db))
    mfccs = librosa.feature.mfcc(S=mel_db, n_mfcc=13)
    f["MFCC mean"] = float(np.mean(mfccs))
    f["Centroid"]  = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    f["Bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
    f["Rolloff"]   = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr)))
    f["ZCR"]       = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            hop_length=256
        )
        f["Median f₀ Hz"] = float(np.nanmedian(f0))
    except Exception as e:
        logging.warning(f"pyin failed: {e}")
        f["Median f₀ Hz"] = 0.0
    logging.info("← feats: done")
    return f

def mfcc_dtw(y1: np.ndarray, y2: np.ndarray, sr: int = 16_000):
    """Compute DTW cost between two MFCC sequences."""
    logging.info("→ mfcc_dtw: extracting MFCCs for DTW")
    m1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=13)
    m2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=13)
    D, _ = seq.dtw(m1, m2, metric="euclidean")
    cost = D[-1, -1] / D.shape[0]
    logging.info(f"← mfcc_dtw: cost={cost:.3f}")
    return cost

def diff_html(ref: str, hyp: str) -> str:
    out = []
    for seg in ndiff(ref.split(), hyp.split()):
        if seg.startswith("  "):
            out.append(f"<span style='color:#21c55d'>{seg[2:]}</span>")
        elif seg.startswith(("+ ", "- ")):
            out.append(f"<span style='color:#ef4444'>{seg[2:]}</span>")
    return " ".join(out)

def mel_fig(y: np.ndarray, sr: int = 16_000):
    S = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max
    )
    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(S, origin="lower", aspect="auto", cmap="magma")
    ax.set_xlabel("Frames"); ax.set_ylabel("Mel bins")
    plt.colorbar(im, ax=ax, pad=0.01)
    fig.tight_layout()
    return fig

# ─── UI ──────────────────────────────────────────────────────────────────────
st.title("🗣️ AI Pronunciation Trainer")
col1, col2 = st.columns(2)

def audio_widget(label: str, col):
    with col:
        st.subheader(label)
        rec = audio_recorder(
            text=f"Record {label}", icon_size="2x",
            recording_color="#e91e63", neutral_color="#6c757d",
            sample_rate=16_000, pause_threshold=2.0
        )
        up = st.file_uploader(f"or upload {label}", type=["wav","mp3","flac"], key=label)
        if rec:
            st.audio(rec, format="audio/wav"); return rec
        if up:
            st.audio(up); return up.read()
        st.info(f"Awaiting {label} audio"); return None

teacher_raw = audio_widget("Teacher", col1)
student_raw = audio_widget("Student", col2)
run = st.button("▶ Analyse", type="primary", disabled=not (teacher_raw and student_raw))

if run:
    with st.spinner("Transcribing & scoring…"):
        logging.info("----- ANALYSIS START -----")

        # 1) Transcribe
        t_txt = whisper_bytes(teacher_raw, lang="he")
        s_txt = whisper_bytes(student_raw, lang="he")

        # 2) PCM
        t_y = load_pcm(teacher_raw)
        s_y = load_pcm(student_raw)

        # 3) Parallel feats + DTW
        with ThreadPoolExecutor() as pool:
            f_ref  = pool.submit(feats, t_y).result()
            f_stud = pool.submit(feats, s_y).result()
            dtw    = pool.submit(mfcc_dtw, t_y, s_y).result()

        # 4) Metrics
        lev       = 1 - levenshtein_distance(t_txt, s_txt) / max(len(t_txt), len(s_txt), 1)
        wer_val   = wer(t_txt, s_txt)
        per       = levenshtein_distance(t_txt, s_txt) / max(len(t_txt), 1)
        composite = 0.7 * lev + 0.3 * (1 - min(dtw / 100, 1))
        diff      = {k: abs(f_ref[k] - f_stud[k]) for k in f_ref}

        logging.info("----- ANALYSIS END -----")

    st.subheader("Transcripts")
    a, b = st.columns(2); a.write(t_txt); b.write(s_txt)

    st.subheader("Word‑level diff")
    st.markdown(diff_html(t_txt, s_txt), unsafe_allow_html=True)

    st.subheader("Pronunciation Scores")
    c1, c2, c3 = st.columns(3)
    c1.metric("Composite 70/30", f"{composite*100:.1f}%")
    c2.metric("WER", f"{wer_val:.0%}")
    c3.metric("PER", f"{per:.0%}")

    st.markdown("#### Audio‑feature differences")
    st.table({k: [f"{v:.2f}"] for k, v in diff.items()})

    st.markdown("#### Mel‑spectrograms")
    p, q = st.columns(2)
    p.pyplot(mel_fig(t_y)); q.pyplot(mel_fig(s_y))

    # CSV report
    csv = "metric,value\n"
    for k, v in {
        **diff,
        "WER": wer_val,
        "Levenshtein": lev,
        "DTW": dtw,
        "Composite": composite,
        "PER": per,
    }.items():
        csv += f"{k},{v}\n"
    st.download_button("⬇ Download report", csv, "report.csv", "text/csv")
