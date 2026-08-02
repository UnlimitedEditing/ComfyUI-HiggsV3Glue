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
# Matches HiggsV3VoiceClone's own trim_silence_edges() threshold (native.py,
# Saganaki22/Higgs_v3-TTS-ComfyUI) -- logging against the same threshold they
# actually use, not a guessed one, so this diagnostic is directly comparable
# to what their trimming will do to this exact clip.
SILENCE_THRESHOLD_DB = -42.0


def _audio_stats(waveform, sample_rate: int) -> str:
    """
    Signal-level diagnostics for a reference clip, logged to the ComfyUI
    console. Added 2026-08-02 after a real clone attempt produced no
    audible cloning despite the pipeline running error-free end to end --
    source-code review of HiggsV3VoiceClone confirmed reference_audio is
    structurally used for conditioning in our pipeline (never silently
    skipped, since our AUDIO dict is never None), so if cloning still isn't
    audible, these numbers are the next thing to check: a clip that's mostly
    below Saganaki's own -42dB trim threshold, unusually short after
    trimming, or clipped/near-silent overall would explain weak cloning at
    the model level rather than a wiring bug.
    """
    import math
    import torch

    flat = waveform.reshape(-1).float()
    n = flat.numel()
    if n == 0:
        return "EMPTY (0 samples)"

    def _to_db(x: float) -> float:
        return 20.0 * math.log10(x) if x > 1e-12 else float("-inf")

    peak_db = _to_db(flat.abs().max().item())
    rms_db = _to_db(flat.pow(2).mean().sqrt().item())
    duration = n / sample_rate

    frame = max(1, int(sample_rate * 0.01))  # 10ms frames, matching trim_silence_edges
    n_frames = (n + frame - 1) // frame
    quiet_frames = 0
    for i in range(n_frames):
        seg = flat[i * frame:(i + 1) * frame]
        if seg.numel() and _to_db(seg.pow(2).mean().sqrt().item()) < SILENCE_THRESHOLD_DB:
            quiet_frames += 1
    quiet_pct = 100.0 * quiet_frames / max(1, n_frames)

    return (
        f"duration={duration:.2f}s peak={peak_db:.1f}dBFS rms={rms_db:.1f}dBFS "
        f"frames_below_{SILENCE_THRESHOLD_DB:.0f}dB={quiet_pct:.0f}%"
    )


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


_MANGLED_TELEGRAM_FILE_RE = re.compile(
    r'^(?:[a-z_]+__)?(https?)api\.telegram\.orgfilebot(\d+)([A-Za-z0-9_-]+?)'
    r'(voice|photo|video_note|video|audio|document|animation|sticker)file(\d+)\.([a-z0-9]+)$',
    re.IGNORECASE,
)


def _unmangle_telegram_file_url(value: str):
    """
    Reconstruct a real Telegram Bot API file URL from the mangled,
    delimiter-stripped form Graydient produces for an un-uploaded voice/photo
    message reply -- confirmed live 2026-08-01, e.g.
    "init_audio__httpsapi.telegram.orgfilebot5714594430AAFo...voicefile2033054.oga"
    for the real URL
    "https://api.telegram.org/file/bot5714594430:AAFo.../voice/file_2033054.oga".

    This is deliberately narrow: it only recognizes this one well-known,
    fixed URL shape (Telegram's Bot API file-download convention), not a
    general "guess any mangled URL" heuristic -- the mangling is lossy for
    arbitrary URLs and can't be reversed in general. Returns None if `value`
    doesn't match this specific pattern.
    """
    m = _MANGLED_TELEGRAM_FILE_RE.match(value)
    if not m:
        return None
    scheme, bot_id, bot_secret, media_kind, file_id, ext = m.groups()
    return f"{scheme}://api.telegram.org/file/bot{bot_id}:{bot_secret}/{media_kind}/file_{file_id}.{ext}"


def _download(url: str, dest_path: str) -> None:
    """
    Download a URL to a local path with a browser-like User-Agent.

    urllib's default User-Agent ("Python-urllib/x.y") gets a 403 from at
    least docs.boson.ai's WAF (confirmed testing this node) -- a real risk
    for any arbitrary user-supplied custom_audio_url too, not just the
    preset clips.
    """
    import urllib.request

    if not (url.startswith("http://") or url.startswith("https://")):
        # Confirmed live 2026-08-01: a Telegram voice-message reply produced
        # a mangled non-URL value here (every ':'/'/'/'_' stripped, e.g.
        # "init_audio__httpsapi.telegram.org...oga") instead of a real
        # fetchable URL -- Graydient has no upload endpoint of its own, so
        # whatever submitted this skipped uploading the attachment first.
        # Try to reconstruct the real Telegram file URL from the known
        # mangled shape before giving up.
        reconstructed = _unmangle_telegram_file_url(url)
        if reconstructed is None:
            raise ValueError(
                f"custom_audio_url/init_audio is not a valid http(s) URL: {url!r}. "
                "This usually means the submitting client passed a raw/internal "
                "file reference (e.g. an un-uploaded Telegram voice-message "
                "attachment) instead of uploading it to a public URL first -- "
                "Graydient has no upload endpoint of its own, so this can't be "
                "resolved server-side, and it didn't match the known Telegram "
                "file-URL mangling pattern this node can reconstruct."
            )
        print(f"[HiggsV3VoicePreset] reconstructed mangled Telegram file URL: {url!r} -> {reconstructed!r}")
        url = reconstructed

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())


def _resolve_reference_audio(value: str, source_label: str):
    """
    Load a reference clip from an arbitrary Graydient-supplied string, which
    might be a real URL, a mangled Telegram file reference, or a bare local
    filename/path -- we don't control which shape the submitting client uses,
    so try each in turn rather than assuming one.
    """
    import tempfile

    value = value.strip()

    if value.startswith("http://") or value.startswith("https://"):
        print(f"[HiggsV3VoicePreset] {source_label}={value!r} -> treating as URL")
        suffix = os.path.splitext(value.split("?")[0])[1] or ".audio"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        _download(value, tmp_path)
        return _decode_audio_via_ffmpeg(tmp_path)

    reconstructed = _unmangle_telegram_file_url(value)
    if reconstructed is not None:
        print(f"[HiggsV3VoicePreset] {source_label}={value!r} -> reconstructed mangled Telegram URL {reconstructed!r}")
        suffix = os.path.splitext(reconstructed.split("?")[0])[1] or ".audio"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        _download(reconstructed, tmp_path)
        return _decode_audio_via_ffmpeg(tmp_path)

    # Not a URL and not a reconstructable Telegram reference -- try it as a
    # bare local filename/path. Check ComfyUI's own input/ directory (where
    # an already-uploaded-to-Graydient file would plausibly land) as well as
    # the literal path as given.
    candidates = [value]
    try:
        import folder_paths
        candidates.append(os.path.join(folder_paths.get_input_directory(), value))
    except Exception:
        pass

    for candidate in candidates:
        if os.path.isfile(candidate):
            print(f"[HiggsV3VoicePreset] {source_label}={value!r} -> found local file {candidate!r}")
            return _decode_audio_via_ffmpeg(candidate)

    raise ValueError(
        f"{source_label}={value!r} is not a valid http(s) URL, a reconstructable "
        "Telegram file reference, or an existing local file (checked ComfyUI's "
        f"input/ directory and the literal value). Tried: {candidates!r}"
    )


class HiggsV3VoicePreset:
    """
    Loads a reference voice as AUDIO for HiggsV3VoiceClone's reference_audio
    input -- either one of six named Higgs v3 demo voices, or a user-supplied
    reference clip, same pattern as BakeVertexColorsFromViews's
    back_image_url in ComfyUI-TripoSG.

    Three separate custom-clip inputs exist because it's unclear which field
    name a given Graydient submission path (chat frontend vs CLI vs pasted
    URL vs file attachment) actually populates -- this project's own
    established convention for images is "init_image_url" (with a "_url"
    suffix), which is a *different* literal string than the raw API body key
    ("init_image", no suffix) graydient-cli uses. Confirmed live 2026-08-02
    that "init_audio" (matching the raw body key) does work via both the CLI
    and a real chat-frontend attachment flow, but that doesn't rule out other
    input paths using "init_audio_url" or "init_audio_filename" instead --
    mapping all three to separate slots costs nothing and avoids silently
    dropping whichever one Graydient actually happens to populate for a given
    submission method. Checked in order: custom_audio_url, then
    custom_audio_url_alt, then custom_audio_filename; first non-empty wins.

    Preset clips read the concept_mapping-staged local copy if present; fall
    back to downloading directly, since concept_mapping is not guaranteed to
    persist across Graydient machine classes. Custom clips always resolve
    fresh (they're per-job, not worth pre-staging).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice": (list(VOICE_PRESETS.keys()), {"default": "jake"}),
                "custom_audio_url": ("STRING", {"default": ""}),
                "custom_audio_url_alt": ("STRING", {"default": ""}),
                "custom_audio_filename": ("STRING", {"default": ""}),
            },
            "optional": {
                # Deliberately "optional", not "required" -- ComfyUI's prompt
                # validation demands every "required" input be explicitly
                # present in the submitted API JSON regardless of whether the
                # Python function has a default, so adding this as "required"
                # broke every already-uploaded workflow with
                # "Required input is missing: require_clone_source" (confirmed
                # live 2026-08-02). "optional" inputs don't have that
                # requirement -- the Python default below still applies.
                "require_clone_source": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "If true, fail loudly when none of the three custom_audio_* "
                        "inputs resolved to a clip, instead of silently falling back "
                        "to the named voice preset. For when you need certainty that "
                        "a clone source was actually used, not just that the job "
                        "succeeded."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("reference_audio",)
    FUNCTION = "load"
    CATEGORY = "Higgs v3 TTS"

    def load(self, voice, custom_audio_url="", custom_audio_url_alt="", custom_audio_filename="", require_clone_source=False):
        for value, label in (
            (custom_audio_url, "custom_audio_url"),
            (custom_audio_url_alt, "custom_audio_url_alt"),
            (custom_audio_filename, "custom_audio_filename"),
        ):
            value = (value or "").strip()
            if not value:
                continue
            waveform, sample_rate = _resolve_reference_audio(value, label)
            print(
                f"[HiggsV3VoicePreset] loaded custom clip via {label}: "
                f"shape={tuple(waveform.shape)} sample_rate={sample_rate} "
                f"stats=({_audio_stats(waveform, sample_rate)})"
            )
            return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)

        if require_clone_source:
            raise ValueError(
                "require_clone_source is true, but none of custom_audio_url / "
                "custom_audio_url_alt / custom_audio_filename resolved to a clip "
                "(all empty). Refusing to silently fall back to the voice preset -- "
                "either supply a clone source or set require_clone_source to false."
            )

        import folder_paths

        filename, _description = VOICE_PRESETS[voice]
        local_path = os.path.join(folder_paths.models_dir, "higgsv3tts", "voices", filename)
        print(f"[HiggsV3VoicePreset] voice={voice!r}, no custom clip supplied -> source='preset' file={local_path!r}")

        if not os.path.isfile(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            _download(f"{VOICE_SAMPLE_BASE_URL}/{filename}", local_path)

        waveform, sample_rate = _decode_audio_via_ffmpeg(local_path)
        print(
            f"[HiggsV3VoicePreset] loaded preset clip: shape={tuple(waveform.shape)} "
            f"sample_rate={sample_rate} stats=({_audio_stats(waveform, sample_rate)})"
        )
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)


_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")

# Matches a "::tags::" marker ANYWHERE in the text (not just at the start),
# used to split a message into multiple tag/speech segments -- e.g.
# "emotion:happy:: Hi there! ::emotion:sad:: Goodbye now." The FIRST segment's
# tags have no leading "::" (nothing precedes them to delimit against); every
# segment after that needs both a leading and trailing "::" so the parser can
# tell "a new tag block starts here" apart from an em-dash-style "::" a user
# might type as punctuation. Captured content deliberately allows single ':'
# characters (tags themselves look like "emotion:sadness,style:whispering" --
# excluding ':' from the capture, an earlier version of this regex, meant it
# could never match real tag content at all). Non-greedy so it stops at the
# NEAREST following "::", not the last one in the message.
_INLINE_MARKER_RE = re.compile(r"::(.+?)::")


def _tags_to_prefix(tags: str) -> str:
    prefix = ""
    for raw in re.split(r"[;,]", tags):
        raw = raw.strip().strip("<|>").strip()
        if raw:
            prefix += f"<|{raw}|>"
    return prefix


def _apply_tags_after_first_word(tags: str, speech_text: str) -> str:
    """
    Insert this segment's <|...|> tag prefix after the first word of
    speech_text rather than before all of it -- see build()'s docstring for
    why (Higgs v3 speaker-conditioning drift when a tag precedes every word
    of the utterance). If speech_text has no tags, returned unchanged.
    """
    prefix = _tags_to_prefix(tags)
    if not prefix:
        return speech_text
    parts = speech_text.split(None, 1)
    if len(parts) == 2:
        first_word, rest = parts
        return f"{first_word} {prefix}{rest}"
    if len(parts) == 1:
        return f"{parts[0]} {prefix}".rstrip()
    return prefix


class HiggsV3TagBuilder:
    """
    Prepends Higgs v3 control tags to text WITHOUT the caller ever typing a
    literal '<' or '>' character.

    Background: the Telegram/Graydient chat frontend mangles bare '<...>' in
    message text before it reaches the workflow (confirmed live 2026-07-29 --
    <|emotion:sad|> came out as the model literally saying the words "emotion
    sad", meaning the frontend stripped the tag delimiters and left literal
    leftover text).

    Two ways to supply tags, tried in this priority order:
      1. `[...]` square-bracket groups found anywhere in `text` -- this is
         Graydient's actual native negative-prompt syntax (confirmed via the
         graydient-cli docs: "there's no separate negative field over the raw
         API ... applied inline in the same string"). Each group's inner
         content is extracted, the group is removed from the spoken text, and
         multiple groups are joined with commas as ONE global tag list --
         this path does not support multiple independently-tagged segments,
         see (2) below for that.
      2. `tags::speech` marker(s) (own convention, not a Graydient one) --
         the confirmed-reliable path in real Graydient testing. Supports
         MULTIPLE segments in one message, each with its own tags, confirmed
         needed 2026-08-02 after a user tried putting a second tag block
         mid-message and it came out spoken literally instead of applied:
           `firsttag:thing:: speech ... ::secondtag:thing:: more speech ...`
         The FIRST segment's tags have no leading `::` (nothing precedes them
         to delimit against) -- just `tags:: speech`, same as before. Every
         segment AFTER the first needs `::` on BOTH sides of its tag block --
         `::tags::` -- so the parser can tell "a new tag block starts here"
         apart from a literal `::` a user might type as punctuation
         mid-sentence. A message with no `::` anywhere is left completely
         untouched (plain speech, no tag processing at all).

    The `tags` external input (previously fed by `negative_prompt`) has been
    REMOVED as a tag source entirely -- confirmed live 2026-08-02 as a dead
    end that never actually delivered a usable value ("negative_prompt ->
    TagBuilder.tags STILL doesn't apply tags... confirmed dead end", per
    gen_higgs_v2.py's own history), yet still being checked FIRST meant that
    whenever Graydient populated that field with anything at all (even
    unreliably/accidentally), steps 1/2 above were skipped -- so a `tags::`
    marker typed directly in `text` was never stripped, and the literal
    marker string got spoken aloud instead of being parsed as control tags.
    A remaining `tags` kwarg is kept on `build()` for backward compatibility
    with the field_mapping possibly still targeting this slot on an
    already-uploaded workflow, but it is now ignored -- `text` is always
    checked for `[...]` / `tags::` regardless of what's passed there.

    Tags are sentence-level only (Higgs v3's own style/emotion tags must sit
    at the start of the sentence they affect per the model card) -- this
    node always prepends, it doesn't support positioning a tag mid-sentence.

    Logs every input/output to the ComfyUI console (visible in Graydient job
    logs) so a failed run shows exactly what this node received and produced,
    instead of requiring guesswork from the final spoken audio.
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
        # `tags` (external input, previously fed by negative_prompt) is
        # deliberately ignored -- see class docstring. text is always
        # checked for [...] / tags:: regardless.
        ignored_tags_input = (tags or "").strip()
        working_text = text
        source = None
        segments_log = []

        bracket_matches = _BRACKET_RE.findall(text)
        if bracket_matches:
            # Single global tag block applied to the whole message -- brackets
            # don't support the multi-segment tags::speech::tags::speech...
            # shape below, on purpose (this path isn't confirmed reliable
            # through a real chat frontend anyway, see class docstring).
            tags = ",".join(m.strip() for m in bracket_matches if m.strip())
            working_text = _BRACKET_RE.sub("", text).strip()
            source = "[...] bracket groups in text"
            combined = _apply_tags_after_first_word(tags, working_text)
            segments_log.append((tags, working_text))

        elif "::" in working_text:
            source = "tags::speech marker(s)"

            # Leading segment: tags with NO leading "::" (nothing precedes
            # them), terminated by the first "::". If what's before the
            # first "::" doesn't look like a tag list (empty, too long, or
            # no ":" in it), treat the whole message as segment text with no
            # leading tags -- it might still contain later ::tags:: blocks.
            head, _, rest = working_text.partition("::")
            head_stripped = head.strip()
            if head_stripped and len(head_stripped) < 200 and ":" in head_stripped:
                leading_tags = head_stripped
                remainder = rest
            else:
                leading_tags = ""
                remainder = working_text

            # Remaining segments: each introduced by a "::tags::" block
            # (delimiters on BOTH sides). Scanned manually (not
            # _INLINE_MARKER_RE.split()) so a match whose inner content
            # doesn't look like a real tag list (no ':' in it -- e.g. a
            # literal "::" a user typed as punctuation) is left as literal
            # text instead of being force-split into a bogus segment.
            pieces = []
            last_end = 0
            for m in _INLINE_MARKER_RE.finditer(remainder):
                candidate = m.group(1).strip()
                if not candidate or len(candidate) >= 200 or ":" not in candidate:
                    continue
                pieces.append(remainder[last_end:m.start()])
                pieces.append(candidate)
                last_end = m.end()
            pieces.append(remainder[last_end:])
            combined_parts = [_apply_tags_after_first_word(leading_tags, pieces[0].strip())]
            segments_log.append((leading_tags, pieces[0].strip()))
            i = 1
            while i < len(pieces):
                seg_tags = pieces[i].strip()
                seg_text = pieces[i + 1].strip() if i + 1 < len(pieces) else ""
                combined_parts.append(_apply_tags_after_first_word(seg_tags, seg_text))
                segments_log.append((seg_tags, seg_text))
                i += 2
            combined = " ".join(p for p in combined_parts if p)

        else:
            combined = working_text

        print(
            f"[HiggsV3TagBuilder] text={text!r} ignored_tags_input={ignored_tags_input!r} "
            f"-> source={source!r} segments={segments_log!r} -> combined_text={combined!r}"
        )
        return (combined,)


NODE_CLASS_MAPPINGS = {
    "HiggsV3VoicePreset": HiggsV3VoicePreset,
    "HiggsV3TagBuilder": HiggsV3TagBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiggsV3VoicePreset": "Higgs v3 Voice Preset",
    "HiggsV3TagBuilder": "Higgs v3 Tag Builder",
}
