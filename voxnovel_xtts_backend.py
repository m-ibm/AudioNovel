#!/usr/bin/env python3
"""Standalone XTTS backend following VoxNovel's upstream generation branches.

This module deliberately does NOT import voxnovel_audio_only.py or
headless_voxnovel.py. The latter executes generate_audio() at module load.

Supported routes for this experiment/regenerator:
- normal XTTS-v2 with references in tortoise/voices/<voice_actor>/
- fine-tuned XTTS when tortoise/voices/<voice_actor>/model/ exists
- VoxNovel's three optional VCTK-style multi-speaker voice IDs
- direct single-speaker TTS model names in voice_actor (regeneration only;
  padded experiment is intentionally restricted to the selected XTTS model)
"""
from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torchaudio
import nltk
from nltk.tokenize import sent_tokenize
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

CHAPTER_MARKERS = ("NEWCHAPTERABC",)
VOICE_DIR = Path("tortoise/voices")
DEMO_DIR = Path("tortoise/_model_demo_voices")
DEMO_TEXT = "Imagine a world where endless possibilities await around every corner."
MULTI_VOICE_MODELS = (
    "tts_models/en/vctk/vits",
    "tts_models/en/vctk/fast_pitch",
    "tts_models/ca/custom/vits",
)


def ensure_nltk_punkt() -> None:
    try:
        sent_tokenize("A short test sentence.")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        sent_tokenize("A short test sentence.")


def configure_cpu_threads() -> None:
    count = max(1, int(os.environ.get("TORCH_NUM_THREADS", "4")))
    torch.set_num_threads(count)
    try:
        torch.set_num_interop_threads(max(1, min(2, count)))
    except RuntimeError:
        pass


def choose_device() -> torch.device:
    # In the Colima Linux container, use CUDA only if actually available.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prompt_for_model() -> str:
    configured = os.environ.get("VOXNOVEL_TTS_MODEL", "").strip()
    if configured:
        print(f"Using VOXNOVEL_TTS_MODEL={configured}")
        return configured
    default = "tts_models/multilingual/multi-dataset/xtts_v2"
    answer = input(f"TTS model used for the original book [{default}]: ").strip()
    return answer or default


def prompt_for_language() -> str:
    configured = os.environ.get("VOXNOVEL_LANGUAGE", "").strip()
    if configured:
        print(f"Using VOXNOVEL_LANGUAGE={configured}")
        return configured
    answer = input("TTS language used for the original book [en]: ").strip()
    return answer or "en"


def load_voice_override() -> dict[str, str]:
    """Optional Speaker -> voice_actor JSON mapping, matching the earlier helper."""
    value = os.environ.get("VOXNOVEL_VOICE_MAP", "").strip()
    if not value:
        return {}
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"VOXNOVEL_VOICE_MAP does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in data.items()
    ):
        raise ValueError("VOXNOVEL_VOICE_MAP must be a JSON object of Speaker -> voice_actor strings")
    return data


def split_long_sentence(sentence: str, max_length: int = 230, max_pauses: int = 8) -> list[str]:
    """Same splitting strategy as the audio-only helper used for this book."""
    parts: list[str] = []
    while len(sentence) > max_length or sum(sentence.count(c) for c in ",;.") > max_pauses:
        possible = [i for i, ch in enumerate(sentence) if ch in ",;." and i < max_length]
        split_at = possible[-1] + 1 if possible else max_length
        parts.append(sentence[:split_at].strip())
        sentence = sentence[split_at:].strip()
    if sentence:
        parts.append(sentence)
    return parts


class VoxNovelTTSBackend:
    """Direct, explicit implementation of VoxNovel's voice/model dispatch."""

    def __init__(self, selected_model: str, device: torch.device, project: Path | str = Path(".")) -> None:
        self.selected_model = selected_model
        self.device = device
        self.project = Path(project).resolve()
        self.voice_dir = self.project / "tortoise" / "voices"
        self.demo_dir = self.project / "tortoise" / "_model_demo_voices"
        self.api_models: dict[str, Any] = {}
        self.multi_voice_speakers: dict[str, set[str]] = {}
        self.active_custom_voice: str | None = None
        self.custom_tts: Any | None = None

    def get_api_tts(self, model_name: str, device: torch.device | str | None = None) -> Any:
        target = str(device or self.device)
        key = f"{model_name}|{target}"
        if key not in self.api_models:
            self.api_models[key] = TTS(model_name, progress_bar=True).to(target)
        return self.api_models[key]

    def prepare_special_voice_lists(self, voice_actors: set[str]) -> None:
        # VoxNovel's original list_reference_files/generate_audio code supports
        # these three multi-speaker Coqui models. Load speaker lists only if an
        # actor is not an ordinary voice folder and is not a model name itself.
        needs_lookup = any(
            not (self.voice_dir / actor).is_dir() and not actor.startswith("tts_models/")
            for actor in voice_actors
        )
        if not needs_lookup:
            return
        for model_name in MULTI_VOICE_MODELS:
            try:
                model = self.get_api_tts(model_name, "cpu")
                self.multi_voice_speakers[model_name] = set(getattr(model, "speakers", None) or [])
            except Exception as exc:
                print(f"Warning: could not inspect speakers for {model_name}: {exc}")
                self.multi_voice_speakers[model_name] = set()

    def special_model_for_actor(self, voice_actor: str) -> str | None:
        for model_name, speakers in self.multi_voice_speakers.items():
            if voice_actor in speakers:
                return model_name
        return None

    def reference_files(self, voice_actor: str) -> list[str]:
        special_model = self.special_model_for_actor(voice_actor)
        if special_model:
            folder = self.demo_dir / special_model / voice_actor
        elif voice_actor.startswith("tts_models/"):
            folder = self.demo_dir / voice_actor
        else:
            folder = self.voice_dir / voice_actor

        if not folder.is_dir():
            raise FileNotFoundError(
                f"No VoxNovel reference-voice directory for voice_actor={voice_actor!r}: {folder}"
            )
        refs = sorted(
            str(p) for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".wav", ".mp3"}
        )
        if not refs and (special_model or voice_actor.startswith("tts_models/")):
            folder.mkdir(parents=True, exist_ok=True)
            if special_model:
                demo_model = self.get_api_tts(special_model, "cpu")
                demo_model.tts_to_file(
                    text=DEMO_TEXT, file_path=str(folder / "demo.wav"), speaker=voice_actor
                )
            else:
                demo_model = self.get_api_tts(voice_actor, self.device)
                demo_model.tts_to_file(text=DEMO_TEXT, file_path=str(folder / "demo.wav"))
            refs = sorted(
                str(p) for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in {".wav", ".mp3"}
            )
        return refs

    def _load_finetuned_xtts(self, voice_actor: str) -> Any:
        if self.active_custom_voice == voice_actor and self.custom_tts is not None:
            return self.custom_tts
        model_dir = self.voice_dir / voice_actor / "model"
        config_path = model_dir / "config.json"
        checkpoint_path = model_dir / "model.pth"
        vocab_candidates = [model_dir / "vocab.json_", model_dir / "vocab.json"]
        vocab_path = next((p for p in vocab_candidates if p.is_file()), None)
        missing = [str(p) for p in (config_path, checkpoint_path) if not p.is_file()]
        if vocab_path is None:
            missing.extend(str(p) for p in vocab_candidates)
        if missing:
            raise FileNotFoundError("Fine-tuned XTTS files missing: " + ", ".join(missing))

        config = XttsConfig()
        config.load_json(str(config_path))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=str(checkpoint_path),
            vocab_path=str(vocab_path),
            use_deepspeed=False,
        )
        # Match VoxNovel upstream fine-tuned XTTS handling: CUDA when present,
        # otherwise CPU. Do not assume this branch supports Apple MPS.
        if self.device.type == "cuda":
            model.cuda()
        else:
            if self.device.type == "mps":
                print("NOTE: VoxNovel's fine-tuned XTTS branch is kept on CPU; upstream selects only CUDA or CPU.", flush=True)
            model.cpu()
        self.custom_tts = model
        self.active_custom_voice = voice_actor
        return model

    def generate_finetuned_xtts(
        self, text: str, output_file: Path, reference_wav: str,
        language: str, voice_actor: str, temperature: float = 0.7,
    ) -> None:
        model = self._load_finetuned_xtts(voice_actor)
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[reference_wav]
        )
        result = model.inference(
            text, language, gpt_cond_latent, speaker_embedding,
            temperature=float(temperature),
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_file), torch.tensor(result["wav"]).unsqueeze(0), 24000)

    def generate_fragment(
        self, fragment: str, output_file: Path, voice_actor: str, language: str,
        *, split_sentences: bool = True, temperature: float | None = None,
    ) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        special_model = self.special_model_for_actor(voice_actor)
        if special_model:
            tts = self.get_api_tts(special_model, "cpu")
            kwargs: dict[str, Any] = {"text": fragment, "file_path": str(output_file), "speaker": voice_actor}
            # This VCTK/VITS route does not use XTTS sampling temperature.
            tts.tts_to_file(**kwargs)
            return

        if voice_actor.startswith("tts_models/") and "multi-dataset" not in voice_actor:
            tts = self.get_api_tts(voice_actor, self.device)
            try:
                kwargs = {"text": fragment, "file_path": str(output_file)}
                if temperature is not None:
                    kwargs["temperature"] = float(temperature)
                tts.tts_to_file(**kwargs)
            except ValueError as exc:
                if "Model is multi-lingual but no `language` is provided." not in str(exc):
                    raise
                kwargs["language"] = language
                tts.tts_to_file(**kwargs)
            return

        custom_dir = VOICE_DIR / voice_actor / "model"
        if custom_dir.is_dir() and "xtts" in self.selected_model.lower():
            refs = self.reference_files(voice_actor)
            if not refs:
                raise FileNotFoundError(f"No reference WAV/MP3 found for fine-tuned voice {voice_actor!r}")
            self.generate_finetuned_xtts(
                fragment, output_file, refs[0], language, voice_actor,
                temperature=0.7 if temperature is None else float(temperature),
            )
            return

        if "multilingual" in self.selected_model and "multi-dataset" in self.selected_model:
            refs = self.reference_files(voice_actor)
            if not refs:
                raise FileNotFoundError(f"No reference WAV/MP3 found for voice_actor={voice_actor!r}")
            tts = self.get_api_tts(self.selected_model, self.device)
            kwargs = {
                "text": fragment,
                "file_path": str(output_file),
                "speaker_wav": refs,
                "language": language,
                "split_sentences": split_sentences,
            }
            if temperature is not None:
                kwargs["temperature"] = float(temperature)
            tts.tts_to_file(**kwargs)
            return

        raise RuntimeError(
            "This standalone helper currently implements the XTTS voice-cloning route. "
            f"Selected model={self.selected_model!r}, voice_actor={voice_actor!r}."
        )

    def cleanup(self) -> None:
        self.api_models.clear()
        self.custom_tts = None
        self.active_custom_voice = None
        gc.collect()
