"""
MLX Whisper model wrapper for audio transcription.

This module provides a wrapper class for MLX Whisper models with audio
transcription and processing capabilities.
"""

from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache
from typing import Any

import librosa
from loguru import logger
from mlx_whisper.transcribe import transcribe
import numpy as np

SAMPLING_RATE = 16000
CHUNK_SIZE = 30


@lru_cache(maxsize=32)
def load_audio(fname: str) -> np.ndarray:  # type: ignore[type-arg]
    """Load and cache audio file. Cache size limited to 32 recent files."""
    a, _ = librosa.load(fname, sr=SAMPLING_RATE, dtype=np.float32)
    return a


@lru_cache(maxsize=32)
def calculate_audio_duration(audio_path: str) -> float:
    """Calculate the duration of the audio file in seconds."""
    audio = load_audio(audio_path)
    return len(audio) / SAMPLING_RATE


class MLX_Whisper:
    """
    Wrapper class for MLX Whisper audio transcription models.

    Provides methods for loading models and transcribing audio files
    using MLX Whisper implementations.
    """

    def __init__(self, model_path: str) -> None:
        """
        Initialize the MLX_Whisper wrapper with the model identifier or local path.

        Parameters:
            model_path (str): Local filesystem path or Hugging Face repository identifier for the Whisper model used for transcription.
        """
        self.model_path = model_path

    def _transcribe_generator(
        self,
        audio_path: str,
        **kwargs: Any,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Stream the audio file as sequential CHUNK_SIZE-second segments and yield transcription results for each segment.

        Each yielded dictionary contains the transcription fields produced by the underlying transcribe call, with two additional keys:
        - `chunk_start` (float): segment start time in seconds.
        - `chunk_end` (float): segment end time in seconds.

        Parameters:
            audio_path (str): Path to the audio file to transcribe.
            **kwargs: Passed through to the underlying `transcribe` call.

        Returns:
            Generator[dict[str, Any], None, None]: Generator that yields transcription dictionaries for each audio segment.
        """
        # Load the audio file
        audio = load_audio(audio_path)
        duration = calculate_audio_duration(audio_path)

        beg = 0.0
        while beg < duration:
            # Calculate chunk boundaries
            chunk_end = min(beg + CHUNK_SIZE, duration)

            # Extract audio chunk
            beg_samples = int(beg * SAMPLING_RATE)
            end_samples = int(chunk_end * SAMPLING_RATE)
            audio_chunk = audio[beg_samples:end_samples]

            # Transcribe chunk
            result = transcribe(audio_chunk, path_or_hf_repo=self.model_path, **kwargs)

            # Add timing information
            result["chunk_start"] = beg
            result["chunk_end"] = chunk_end

            yield result

            beg += CHUNK_SIZE

    def __call__(
        self,
        audio_path: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Generator[dict[str, Any], None, None] | dict[str, Any]:
        """
        Transcribe an audio file either as a single result or as a generator of chunked results.

        Parameters:
            audio_path (str): Path to the audio file to transcribe.
            stream (bool): If True, return a generator that yields transcription dictionaries for fixed-size time chunks; if False, return a single transcription dictionary for the whole file.
            **kwargs: Additional keyword arguments forwarded to the underlying transcribe() call.

        Returns:
            Generator[dict[str, Any], None, None] | dict[str, Any]:
                - If `stream` is True: a generator that yields transcription dictionaries for each chunk. Each chunk dict includes `chunk_start` and `chunk_end` timestamps plus the transcription fields produced by transcribe().
                - If `stream` is False: a single transcription dictionary for the full audio. The returned dict will include a `duration` key if not already present.
        """
        if stream:
            return self._transcribe_generator(audio_path, **kwargs)

        tc: dict[str, Any] = transcribe(
            audio_path,
            path_or_hf_repo=self.model_path,
            **kwargs,
        )
        # Attach duration so callers do not need to reload audio
        tc.setdefault("duration", calculate_audio_duration(audio_path))
        return tc


if __name__ == "__main__":
    model = MLX_Whisper("mlx-community/whisper-tiny")
    # Non-streaming (fastest for most use cases)
    result = model("examples/audios/podcast.wav", stream=True)
    for chunk in result:
        chunk_dict = chunk if isinstance(chunk, dict) else {}
        logger.info(
            f"[{chunk_dict.get('chunk_start', 0):.1f}s - {chunk_dict.get('chunk_end', 0):.1f}s]: {chunk_dict.get('text', '')}",
        )
