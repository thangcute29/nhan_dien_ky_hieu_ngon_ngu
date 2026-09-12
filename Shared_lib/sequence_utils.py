"""Shared preprocessing for dynamic sign-language sequences.

The GRU contract is a float32 array with shape ``(30, 126)``. Each frame
stores 21 xyz landmarks for the MediaPipe Left hand followed by 21 xyz
landmarks for the Right hand. Missing hands are represented by zeros.
"""

from __future__ import annotations

import numpy as np


HAND_FEATURES = 21 * 3
TOTAL_FEATURES = HAND_FEATURES * 2
DEFAULT_SEQUENCE_LENGTH = 30


def landmarks_to_vector(multi_hand_landmarks, multi_handedness) -> np.ndarray:
    """Convert MediaPipe results to the stable ``[Left | Right]`` layout."""
    vector = np.zeros(TOTAL_FEATURES, dtype=np.float32)
    handedness = multi_handedness or []
    for index, hand_landmarks in enumerate(multi_hand_landmarks or []):
        label = "Left"
        if index < len(handedness):
            label = handedness[index].classification[0].label
        offset = 0 if label == "Left" else HAND_FEATURES
        for point_index, landmark in enumerate(hand_landmarks.landmark):
            base = offset + point_index * 3
            vector[base : base + 3] = (landmark.x, landmark.y, landmark.z)
    return vector


def resample_sequence(sequence, length: int = DEFAULT_SEQUENCE_LENGTH) -> np.ndarray:
    """Uniformly resample a sequence, repeating nearest frames when needed."""
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != TOTAL_FEATURES:
        raise ValueError(f"Expected (frames, {TOTAL_FEATURES}), got {array.shape}")
    if len(array) == 0:
        raise ValueError("Cannot resample an empty sequence")
    indices = np.linspace(0, len(array) - 1, length).round().astype(np.int64)
    return array[indices].astype(np.float32, copy=False)


def normalize_sequence(sequence) -> np.ndarray:
    """Use one fixed wrist anchor while preserving motion and hand distance."""
    normalized = np.asarray(sequence, dtype=np.float32).copy()
    if normalized.ndim != 2 or normalized.shape[1] != TOTAL_FEATURES:
        raise ValueError(f"Expected (frames, {TOTAL_FEATURES}), got {normalized.shape}")

    anchor = None
    for frame in normalized:
        left = frame[:HAND_FEATURES]
        right = frame[HAND_FEATURES:]
        if np.any(right):
            anchor = right[:3].copy()
            break
        if np.any(left):
            anchor = left[:3].copy()
            break
    if anchor is None:
        return normalized

    for frame in normalized:
        for offset in (0, HAND_FEATURES):
            hand = frame[offset : offset + HAND_FEATURES]
            if np.any(hand):
                frame[offset : offset + HAND_FEATURES] = (
                    hand.reshape(21, 3) - anchor
                ).reshape(-1)
    return normalized


def prepare_sequence(sequence, length: int = DEFAULT_SEQUENCE_LENGTH) -> np.ndarray:
    """Resample and normalize raw landmarks for one GRU inference."""
    return normalize_sequence(resample_sequence(sequence, length))
