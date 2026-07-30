import os
import re

VOICE_PRESETS = {
    "jake": (
        "Jake_Rivers_intro.mp3",
        "A male speaker with an energetic and slightly dramatic tone, conveying "
        "passion and enthusiasm, especially about sports.",
    ),
    "chloe": (
        "Chloe_Adams_intro.mp3",
        "A friendly and clear female voice with an engaging, informative tone "
        "and a standard American accent.",
    ),
    "eleanor": (
        "Eleanor_Reed_intro.mp3",
        "A calm, articulate female voice with a clear, professional Standard "
        "American English accent, perfect for educational or explanatory content.",
    ),
    "marcus": (
        "Marcus_Webb_intro.mp3",
        "A male speaker with an enthusiastic, confident, and slightly "
        "professorial American delivery.",
    ),
    "nora": (
        "Nora_Vance_intro.mp3",
        "A female speaker with a calm, clear, and narrative voice, using "
        "standard American English.",
    ),
    "oliver": (
        "Oliver_Grant_intro.mp3",
        "A calm, articulate male voice with a thoughtful American accent, "
        "well-suited for explanatory or reflective content.",
    ),
}

# Boson's public demo clips for their hosted API's named presets (chloe/eleanor/
# jake/marcus/nora/oliver) -- not the actual server-side reference clips, but
# real audio of each named voice, usable locally as HiggsV3VoiceClone reference
# audio to approximate the preset via zero-shot cloning.
VOICE_SAMPLE_BASE_URL = "https://docs.boson.ai/public/audio/higgs-audio-tts-voices"

HIGGS_SAMPLE_RATE = 24000


def _decode_audio_via_ffmpeg(path: str, sample_rate: int = HIGGS_SAMPLE_RATE):
    """
    Decode an audio file to a mono waveform via an ffmpeg subprocess.

    torchaudio.load() routes through torchcodec's AudioDecoder on recent
    torchaudio versions, which is broken on at least some Graydient instances
    (`_AudioDecoder() takes no arguments`) -- confirmed on a live g1_embedded
    run 2026-07-29. ffmpeg is already a hard dependency of this project's
    video pipelines, so shelling out to it directly sidesteps torchcodec
    entirely rather than hoping torchaudio's backend dispatch recovers.

    Raw PCM output (not WAV) is used deliberately: ffmpeg writes a WAV header
    with a placeholder max-int frame count when streaming to a pipe (it can't
    know the final size upfront), which makes Python's `wave` module report
    a bogus multi-hour duration if you trust the header instead of the actual
    byte count.
    """
    import subprocess
    import numpy as np
    import torch

    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}: {proc.stderr.decode(errors='replace')}")

    arr = np.frombuffer(proc.stdout, dtype="<i2").astype("float32") / 32768.0
    waveform = torch.from_numpy(arr.copy()).unsqueeze(0)  # [channels=1, samples]
    return waveform, sample_rate


def _download(url: str, dest_path: str) -> None:
    """
    Download a URL to a local path with a browser-like User-Agent.

    urllib's default User-Agent ("Python-urllib/x.y") gets a 403 from at
    least docs.boson.ai's WAF (confirmed testing this node) -- a real risk
    for any arbitrary user-supplied custom_audio_url too, not just the
    preset clips.
    """
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())


class HiggsV3VoicePreset:
    """
    Loads a reference voice as AUDIO for HiggsV3VoiceClone's reference_audio
    input -- either one of six named Higgs v3 demo voices, or an arbitrary
    user-supplied audio URL via custom_audio_url (takes priority over voice
    when non-empty, same pattern as BakeVertexColorsFromViews's back_image_url
    in ComfyUI-TripoSG).

    Preset clips read the concept_mapping-staged local copy if present; fall
    back to downloading directly, since concept_mapping is not guaranteed to
    persist across Graydient machine classes. custom_audio_url always
    downloads fresh (it's per-job, not worth pre-staging).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice": (list(VOICE_PRESETS.keys()), {"default": "jake"}),
                "custom_audio_url": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("reference_audio",)
    FUNCTION = "load"
    CATEGORY = "Higgs v3 TTS"

    def load(self, voice, custom_audio_url=""):
        custom_audio_url = (custom_audio_url or "").strip()

        if custom_audio_url:
            import tempfile

            suffix = os.path.splitext(custom_audio_url.split("?")[0])[1] or ".audio"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            _download(custom_audio_url, tmp_path)
            waveform, sample_rate = _decode_audio_via_ffmpeg(tmp_path)
            return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)

        import folder_paths

        filename, _description = VOICE_PRESETS[voice]
        local_path = os.path.join(folder_paths.models_dir, "higgsv3tts", "voices", filename)

        if not os.path.isfile(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            _download(f"{VOICE_SAMPLE_BASE_URL}/{filename}", local_path)

        waveform, sample_rate = _decode_audio_via_ffmpeg(local_path)
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)


class HiggsV3TagBuilder:
    """
    Prepends Higgs v3 control tags to text WITHOUT the caller ever typing a
    literal '<' or '>' character.

    Exists because the Telegram/Graydient chat frontend used to submit these
    jobs mangles bare '<...>' in message text before it reaches the workflow
    (confirmed live 2026-07-29 -- <|emotion:sad|> came out as the model
    literally saying the words "emotion sad", meaning the frontend stripped
    the tag delimiters and left literal leftover text). `tags` takes plain
    "key:value" pairs (no brackets) -- e.g. "emotion:sad;style:whispering" --
    and this node adds the <|...|> wrapper itself, server-side, so the
    angle brackets never pass through chat as raw characters.

    Tags are sentence-level only (Higgs v3's own style/emotion tags must sit
    at the start of the sentence they affect per the model card) -- this
    node always prepends, it doesn't support positioning a tag mid-sentence.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello! This is Higgs Audio v3 running natively inside ComfyUI."}),
                "tags": ("STRING", {"default": "", "tooltip": "Bracket-free tag list, e.g. emotion:sad;style:whispering -- semicolon or comma separated, no < > needed"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_text",)
    FUNCTION = "build"
    CATEGORY = "Higgs v3 TTS"

    def build(self, text, tags=""):
        tags = (tags or "").strip()
        working_text = text

        # Fallback path: fold tags into the single `text`/prompt field using a
        # "tags::speech" marker, since `negative_prompt` -> `tags` has proven
        # unreliable in practice (confirmed live 2026-07-29: negative_prompt is
        # a real, fillable field in the chat UI, but tags placed there never
        # affected output across repeated tests -- root cause unconfirmed,
        # likely a Graydient-side quirk specific to that field for
        # supports_txt2wav workflows). `prompt` has been reliable in every
        # test, so this keeps both paths live: whichever one actually works,
        # works. Only treated as a marker if `tags` is empty (negative_prompt
        # didn't supply anything) and the part before '::' looks like tag
        # syntax, not just a sentence that happens to contain '::'.
        if not tags and "::" in text:
            marker_part, _, rest = text.partition("::")
            marker_part = marker_part.strip()
            if marker_part and len(marker_part) < 200 and ":" in marker_part:
                tags = marker_part
                working_text = rest.strip()

        prefix = ""
        if tags:
            for raw in re.split(r"[;,]", tags):
                raw = raw.strip().strip("<|>").strip()
                if raw:
                    prefix += f"<|{raw}|>"
        return (prefix + working_text,)


NODE_CLASS_MAPPINGS = {
    "HiggsV3VoicePreset": HiggsV3VoicePreset,
    "HiggsV3TagBuilder": HiggsV3TagBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiggsV3VoicePreset": "Higgs v3 Voice Preset",
    "HiggsV3TagBuilder": "Higgs v3 Tag Builder",
}
