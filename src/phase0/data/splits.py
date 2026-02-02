"""
Speaker and utterance split management.

Creates reproducible train/eval splits for Phase 0 analysis.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass

from .librispeech import UtteranceInfo


@dataclass
class SplitInfo:
    """Information about data splits."""

    train_speakers: list[int]
    eval_speakers: list[int]
    train_utterances: list[str]
    eval_utterances: list[str]
    train_utt_train: list[str]  # k-means fitting subset
    train_utt_val: list[str]  # k-means validation subset


def create_speaker_splits(
    speaker_ids: list[int],
    n_train: int = 200,
    n_eval: int = 51,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Create train/eval speaker splits.

    Args:
        speaker_ids: All available speaker IDs
        n_train: Number of training speakers
        n_eval: Number of evaluation speakers
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_speaker_ids, eval_speaker_ids)
    """
    if len(speaker_ids) < n_train + n_eval:
        raise ValueError(
            f"Not enough speakers: have {len(speaker_ids)}, "
            f"need {n_train + n_eval}"
        )

    # Deterministic shuffle
    rng = random.Random(seed)
    shuffled = sorted(speaker_ids)  # Sort first for reproducibility
    rng.shuffle(shuffled)

    train_speakers = sorted(shuffled[:n_train])
    eval_speakers = sorted(shuffled[n_train : n_train + n_eval])

    return train_speakers, eval_speakers


def create_utterance_splits(
    utterances: list[UtteranceInfo],
    train_speakers: list[int],
    eval_speakers: list[int],
    holdout_frac: float = 0.1,
    seed: int = 42,
) -> SplitInfo:
    """
    Create utterance splits based on speaker splits.

    Also creates a holdout within training utterances for k-means validation.

    Args:
        utterances: All utterances
        train_speakers: Training speaker IDs
        eval_speakers: Evaluation speaker IDs
        holdout_frac: Fraction of train utterances to hold out for validation
        seed: Random seed

    Returns:
        SplitInfo with all split assignments
    """
    train_speaker_set = set(train_speakers)
    eval_speaker_set = set(eval_speakers)

    train_utts = [
        u.utterance_id for u in utterances if u.speaker_id in train_speaker_set
    ]
    eval_utts = [
        u.utterance_id for u in utterances if u.speaker_id in eval_speaker_set
    ]

    # Sort for reproducibility
    train_utts = sorted(train_utts)
    eval_utts = sorted(eval_utts)

    # Create holdout within training utterances
    rng = random.Random(seed)
    shuffled_train = train_utts.copy()
    rng.shuffle(shuffled_train)

    n_holdout = int(len(shuffled_train) * holdout_frac)
    train_utt_val = sorted(shuffled_train[:n_holdout])
    train_utt_train = sorted(shuffled_train[n_holdout:])

    return SplitInfo(
        train_speakers=train_speakers,
        eval_speakers=eval_speakers,
        train_utterances=train_utts,
        eval_utterances=eval_utts,
        train_utt_train=train_utt_train,
        train_utt_val=train_utt_val,
    )


def save_splits(splits: SplitInfo, output_dir: str | Path) -> None:
    """
    Save splits to text files.

    Args:
        splits: SplitInfo object
        output_dir: Directory to save split files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save speaker lists
    with open(output_dir / "train_speakers.txt", "w") as f:
        for sid in splits.train_speakers:
            f.write(f"{sid}\n")

    with open(output_dir / "eval_speakers.txt", "w") as f:
        for sid in splits.eval_speakers:
            f.write(f"{sid}\n")

    # Save utterance lists
    with open(output_dir / "train_utt_ids.txt", "w") as f:
        for uid in splits.train_utterances:
            f.write(f"{uid}\n")

    with open(output_dir / "eval_utt_ids.txt", "w") as f:
        for uid in splits.eval_utterances:
            f.write(f"{uid}\n")

    # Save k-means train/val splits
    with open(output_dir / "train_utt_ids_train.txt", "w") as f:
        for uid in splits.train_utt_train:
            f.write(f"{uid}\n")

    with open(output_dir / "train_utt_ids_val.txt", "w") as f:
        for uid in splits.train_utt_val:
            f.write(f"{uid}\n")

    # Save metadata
    metadata = {
        "n_train_speakers": len(splits.train_speakers),
        "n_eval_speakers": len(splits.eval_speakers),
        "n_train_utterances": len(splits.train_utterances),
        "n_eval_utterances": len(splits.eval_utterances),
        "n_train_utt_train": len(splits.train_utt_train),
        "n_train_utt_val": len(splits.train_utt_val),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_splits(splits_dir: str | Path) -> SplitInfo:
    """
    Load splits from saved files.

    Args:
        splits_dir: Directory containing split files

    Returns:
        SplitInfo object
    """
    splits_dir = Path(splits_dir)

    def read_list(filename: str) -> list[str]:
        with open(splits_dir / filename) as f:
            return [line.strip() for line in f if line.strip()]

    def read_int_list(filename: str) -> list[int]:
        return [int(x) for x in read_list(filename)]

    return SplitInfo(
        train_speakers=read_int_list("train_speakers.txt"),
        eval_speakers=read_int_list("eval_speakers.txt"),
        train_utterances=read_list("train_utt_ids.txt"),
        eval_utterances=read_list("eval_utt_ids.txt"),
        train_utt_train=read_list("train_utt_ids_train.txt"),
        train_utt_val=read_list("train_utt_ids_val.txt"),
    )
