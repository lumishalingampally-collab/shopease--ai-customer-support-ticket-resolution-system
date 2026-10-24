"""Voice, translation and avatar helpers for the ShopEase demo.

All external services are optional:
- Groq Whisper is used for microphone speech-to-text when GROQ_API_KEY is set.
- NLLB can be enabled by setting NLLB_MODEL_PATH to a local NLLB model.
- Groq text translation is a fallback when NLLB is unavailable.
- gTTS is used for spoken answers when installed.
"""

from __future__ import annotations

import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import requests


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def transcribe_with_groq(audio_file, language: Optional[str] = None) -> tuple[str, str]:
    """Transcribe browser-recorded audio with Groq Whisper.

    Returns (text, detected_language_hint). Groq's transcription output is kept
    in the spoken language; translation happens separately so the original
    language can be preserved for the final answer.
    """
    key = _env("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not configured in .env.")

    audio_bytes = audio_file.getvalue()
    filename = getattr(audio_file, "name", "customer_voice.wav") or "customer_voice.wav"

    data = {
        "model": _env("GROQ_WHISPER_MODEL", "whisper-large-v3"),
        "response_format": "verbose_json",
        "temperature": "0",
    }
    if language and language.lower() in {"en", "hi", "te"}:
        data["language"] = language.lower()

    r = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {key}"},
        files={"file": (filename, audio_bytes, "audio/wav")},
        data=data,
        timeout=90,
    )
    if not r.ok:
        raise RuntimeError(f"Groq transcription failed ({r.status_code}): {r.text[:500]}")
    payload = r.json()
    text = (payload.get("text") or "").strip()
    if not text:
        raise RuntimeError("Groq returned an empty transcription.")

    return text, detect_script(text)


def detect_script(text: str) -> str:
    """Lightweight script detection for English/Hindi/Telugu."""
    import re
    if re.search(r"[\u0C00-\u0C7F]", text):
        return "Telugu"
    if re.search(r"[\u0900-\u097F]", text):
        return "Hindi"
    return "English"


def _nllb_available() -> bool:
    return bool(_env("NLLB_MODEL_PATH"))


@lru_cache(maxsize=1)
def _load_nllb():
    """Load a local NLLB model only when explicitly configured.

    This avoids downloading a large model automatically on interview/demo
    machines. The model path may point to a local Hugging Face NLLB directory.
    """
    model_path = _env("NLLB_MODEL_PATH")
    if not model_path:
        return None
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return tokenizer, model, device
    except Exception as exc:
        raise RuntimeError(f"NLLB could not be loaded from NLLB_MODEL_PATH: {exc}")


_NLLB_CODES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Telugu": "tel_Telu",
}


def translate_with_nllb(text: str, source_language: str, target_language: str) -> str:
    """Translate with a configured local NLLB model."""
    loaded = _load_nllb()
    if loaded is None:
        raise RuntimeError("NLLB_MODEL_PATH is not configured.")
    tokenizer, model, device = loaded
    source_code = _NLLB_CODES[source_language]
    target_code = _NLLB_CODES[target_language]

    tokenizer.src_lang = source_code
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    forced_id = tokenizer.convert_tokens_to_ids(target_code)
    output = model.generate(**encoded, forced_bos_token_id=forced_id, max_new_tokens=256)
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()


def translate_with_groq(text: str, source_language: str, target_language: str) -> str:
    """Reliable fallback translation through the user's Groq API."""
    key = _env("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    model = _env("GROQ_TRANSLATION_MODEL", "openai/gpt-oss-20b")
    prompt = (
        f"Translate the following customer-support text from {source_language} "
        f"to {target_language}. Preserve the meaning exactly. Do not add facts, "
        f"policies, names, numbers, or advice. Return only the translation.\n\n"
        f"TEXT:\n{text}"
    )
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You are a precise translation engine."},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"Groq translation failed ({r.status_code}): {r.text[:500]}")
    return r.json()["choices"][0]["message"]["content"].strip()


def translate_text(text: str, source_language: str, target_language: str) -> tuple[str, str]:
    """NLLB first, Groq fallback. Returns (translation, engine_name)."""
    if not text or source_language == target_language:
        return text, "none"
    if _nllb_available():
        try:
            return translate_with_nllb(text, source_language, target_language), "NLLB"
        except Exception:
            pass
    return translate_with_groq(text, source_language, target_language), "Groq"


def synthesize_gtts(text: str, language: str) -> Optional[bytes]:
    """Generate MP3 bytes for the answer. Returns None if gTTS fails."""
    try:
        from gtts import gTTS
        lang_code = {"English": "en", "Hindi": "hi", "Telugu": "te"}.get(language, "en")
        fp = io.BytesIO()
        gTTS(text=text, lang=lang_code, slow=False).write_to_fp(fp)
        return fp.getvalue()
    except Exception:
        return None


def talking_avatar_html(audio_bytes: Optional[bytes], language: str = "English") -> str:
    """Interactive animated support avatar with mouth/eye motion."""
    audio_src = ""
    if audio_bytes:
        encoded = base64.b64encode(audio_bytes).decode("ascii")
        audio_src = f'data:audio/mpeg;base64,{encoded}'

    play = ""
    if audio_src:
        play = f"""
        <audio id="shopAudio" controls preload="auto" src="{audio_src}"></audio>
        <script>
        const audio = document.getElementById("shopAudio");
        const avatar = document.getElementById("shopAvatar");
        function talk(on) {{
            avatar.classList.toggle("talking", on);
        }}
        audio.addEventListener("play", () => talk(true));
        audio.addEventListener("pause", () => talk(false));
        audio.addEventListener("ended", () => talk(false));
        </script>
        """
    return f"""
    <div id="shopAvatar" class="shop-avatar-wrap">
      <div class="orb-glow"></div>
      <div class="shop-avatar">
        <div class="ear left"></div><div class="ear right"></div>
        <div class="face">
          <div class="eye left-eye"><span></span></div>
          <div class="eye right-eye"><span></span></div>
          <div class="brow left-brow"></div><div class="brow right-brow"></div>
          <div class="nose"></div>
          <div class="mouth"><span></span></div>
          <div class="cheek left-cheek"></div><div class="cheek right-cheek"></div>
        </div>
        <div class="neck"></div>
        <div class="body"><div class="badge-dot">S</div></div>
      </div>
      <div class="avatar-status">● ShopEase AI · {language} support</div>
      {play}
    </div>
    <style>
      .shop-avatar-wrap {{ position:relative; min-height:280px; display:flex; flex-direction:column;
        align-items:center; justify-content:center; overflow:hidden; border-radius:28px;
        background:radial-gradient(circle at 50% 40%,#ece9ff 0,#f8f7ff 38%,#ffffff 75%);
        border:1px solid #e8e5ff; box-shadow:0 16px 45px rgba(88,70,190,.12); }}
      .orb-glow {{position:absolute;width:190px;height:190px;border-radius:50%;
        background:rgba(109,93,252,.18);filter:blur(25px);animation:floatGlow 3.5s ease-in-out infinite;}}
      .shop-avatar {{position:relative;width:150px;height:205px;animation:floatAvatar 3.2s ease-in-out infinite;}}
      .face {{position:absolute;left:20px;top:8px;width:110px;height:125px;border-radius:48% 48% 45% 45%;
        background:linear-gradient(145deg,#fff,#ecebff);border:3px solid #b9b1ff;box-shadow:inset 0 -8px 18px rgba(109,93,252,.10);}}
      .eye {{position:absolute;top:45px;width:19px;height:13px;border-radius:50%;background:#20243b;animation:blink 5s infinite;}}
      .eye span {{position:absolute;width:5px;height:5px;border-radius:50%;background:#fff;left:4px;top:2px;}}
      .left-eye {{left:24px}} .right-eye {{right:24px}}
      .brow {{position:absolute;top:35px;width:22px;height:5px;border-radius:5px;background:#8177cf;}}
      .left-brow {{left:22px;transform:rotate(-7deg)}} .right-brow {{right:22px;transform:rotate(7deg)}}
      .nose {{position:absolute;left:51px;top:60px;width:8px;height:18px;border-right:3px solid #aaa4df;border-bottom:3px solid #aaa4df;border-radius:0 0 8px 0;}}
      .mouth {{position:absolute;left:39px;top:88px;width:32px;height:9px;border-radius:0 0 18px 18px;background:#4a416f;overflow:hidden;transition:.12s;}}
      .mouth span {{display:block;width:16px;height:4px;background:#ff8dba;border-radius:50%;margin:5px auto 0;}}
      .talking .mouth {{height:23px;border-radius:10px 10px 18px 18px;animation:mouthTalk .16s infinite alternate;}}
      .cheek {{position:absolute;top:78px;width:14px;height:8px;border-radius:50%;background:#ffd4e5;opacity:.6;}}
      .left-cheek {{left:9px}} .right-cheek {{right:9px}}
      .ear {{position:absolute;top:58px;width:24px;height:42px;border-radius:50%;background:#ecebff;border:3px solid #b9b1ff;}}
      .ear.left {{left:6px}} .ear.right {{right:6px}}
      .neck {{position:absolute;left:65px;top:124px;width:22px;height:25px;background:#d5d1f4;border-radius:5px;}}
      .body {{position:absolute;left:28px;top:143px;width:94px;height:61px;border-radius:35px 35px 15px 15px;
        background:linear-gradient(135deg,#5d51d9,#8d7fff);box-shadow:0 10px 20px rgba(80,66,190,.2);}}
      .badge-dot {{width:27px;height:27px;border-radius:50%;background:#fff;color:#6357db;display:flex;
        align-items:center;justify-content:center;font-weight:800;position:absolute;left:34px;top:12px;}}
      .avatar-status {{position:absolute;bottom:14px;font-size:12px;color:#625d82;font-weight:700;}}
      audio {{width:90%;max-width:300px;height:34px;margin-top:8px;}}
      @keyframes blink {{0%,46%,48%,100%{{transform:scaleY(1)}}47%{{transform:scaleY(.08)}}}}
      @keyframes mouthTalk {{from{{height:10px}}to{{height:23px}}}}
      @keyframes floatAvatar {{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-8px)}}}}
      @keyframes floatGlow {{0%,100%{{transform:scale(.9);opacity:.65}}50%{{transform:scale(1.15);opacity:1}}}}
    </style>
    """


def avatar_idle_html(language: str = "English") -> str:
    return talking_avatar_html(None, language)
