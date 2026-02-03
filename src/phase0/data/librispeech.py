"""
LibriSpeech data loading utilities.

Handles parsing the LibriSpeech directory structure and loading audio files.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torchaudio

_RESAMPLER_CACHE: dict[tuple[int, int], torchaudio.transforms.Resample] = {}


@dataclass
class UtteranceInfo:
    """Information about a single utterance."""

    utterance_id: str
    speaker_id: int
    chapter_id: int
    audio_path: Path
    duration_sec: float


def get_speaker_ids(librispeech_path: str | Path, subset: str = "train-clean-100") -> list[int]:
    """
    Get all speaker IDs from a LibriSpeech subset.

    Args:
        librispeech_path: Path to LibriSpeech root directory
        subset: Subset name (e.g., 'train-clean-100', 'dev-clean')

    Returns:
        Sorted list of speaker IDs
    """
    subset_path = Path(librispeech_path) / subset
    if not subset_path.exists():
        raise FileNotFoundError(f"LibriSpeech subset not found: {subset_path}")

    speaker_ids = []
    for speaker_dir in subset_path.iterdir():
        if speaker_dir.is_dir() and speaker_dir.name.isdigit():
            speaker_ids.append(int(speaker_dir.name))

    return sorted(speaker_ids)


def get_utterances(
    librispeech_path: str | Path,
    speaker_ids: list[int],
    subset: str = "train-clean-100",
) -> list[UtteranceInfo]:
    """
    Get all utterances for the specified speakers.

    Args:
        librispeech_path: Path to LibriSpeech root directory
        speaker_ids: List of speaker IDs to include
        subset: Subset name

    Returns:
        List of UtteranceInfo objects
    """
    subset_path = Path(librispeech_path) / subset
    speaker_set = set(speaker_ids)
    utterances = []

    for speaker_dir in subset_path.iterdir():
        if not speaker_dir.is_dir():
            continue
        if not speaker_dir.name.isdigit():
            continue

        speaker_id = int(speaker_dir.name)
        if speaker_id not in speaker_set:
            continue

        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir():
                continue
            if not chapter_dir.name.isdigit():
                continue

            chapter_id = int(chapter_dir.name)

            for audio_file in chapter_dir.glob("*.flac"):
                utterance_id = audio_file.stem
                info = torchaudio.info(str(audio_file))
                duration_sec = info.num_frames / info.sample_rate

                utterances.append(
                    UtteranceInfo(
                        utterance_id=utterance_id,
                        speaker_id=speaker_id,
                        chapter_id=chapter_id,
                        audio_path=audio_file,
                        duration_sec=duration_sec,
                    )
                )

    return utterances


def parse_librispeech_structure(
    librispeech_path: str | Path,
    subset: str = "train-clean-100",
) -> dict[int, list[UtteranceInfo]]:
    """
    Parse LibriSpeech directory structure into a dict mapping speaker IDs to utterances.

    Args:
        librispeech_path: Path to LibriSpeech root directory
        subset: Subset name

    Returns:
        Dict mapping speaker_id -> list of UtteranceInfo
    """
    speaker_ids = get_speaker_ids(librispeech_path, subset)
    utterances = get_utterances(librispeech_path, speaker_ids, subset)

    by_speaker: dict[int, list[UtteranceInfo]] = {}
    for utt in utterances:
        if utt.speaker_id not in by_speaker:
            by_speaker[utt.speaker_id] = []
        by_speaker[utt.speaker_id].append(utt)

    return by_speaker


def load_audio(
    audio_path: str | Path,
    target_sr: int = 24000,
) -> tuple[torch.Tensor, int]:
    """
    Load audio file and resample to target sample rate.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default 24000 for Mimi)

    Returns:
        Tuple of (audio tensor [1, T], sample_rate)
    """
    waveform, sr = torchaudio.load(str(audio_path))

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        key = (int(sr), int(target_sr))
        resampler = _RESAMPLER_CACHE.get(key)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            _RESAMPLER_CACHE[key] = resampler
        waveform = resampler(waveform)

    return waveform, target_sr
