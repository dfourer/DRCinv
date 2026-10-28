"""Estimate DRC parameters from processed audio with the MEE network.

The neural checkpoint is intentionally not included in the PyPI wheel.  By
default it is downloaded from the project's GitHub Release and cached on the
first inference.  A local checkpoint can always be supplied for offline use.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Optional, Sequence, Tuple, Union
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np


__all__ = [
    "DRCParameters",
    "MEEParameterEstimator",
    "download_checkpoint",
    "estimate_mee_parameters",
]

__version__ = "0.3.0"

DEFAULT_CHECKPOINT_URL = (
    "https://github.com/dfourer/DRCinv/releases/download/"
    "v0.2.0/MEE_compression_large.pt"
)
DEFAULT_CHECKPOINT_SHA256 = (
    "c471092038d81a0f4a23ea8daec6f98afb5f0936b0924c805dc9b9343f1c41e2"
)
DEFAULT_DURATION_SECONDS = 5.0

# The MEE head predicts six values in [0, 1].  These are the ranges used to
# train the 30-profile compression checkpoint.  Time constants are in ms in
# the training data and are converted to seconds at the public API boundary.
COMPRESSION_LARGE_RANGES = np.asarray(
    [
        [-60.0, 0.0],
        [0.0, 15.0],
        [0.0, 130.0],
        [0.0, 127.4],
        [0.0, 500.0],
        [0.0, 1959.7],
    ],
    dtype=np.float32,
)

_MEE_CONFIG = {
    "channels": [16, 32, 64, 128, 256, 256, 512, 512, 1024, 1024, 2048, 2048],
    "kernels": [25, 25, 15, 15, 10, 10, 10, 10, 5, 5, 5, 5],
    "strides": [4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
    "dilation": [1] * 12,
}


PathLike = Union[str, os.PathLike]


@dataclass(frozen=True)
class DRCParameters:
    """Physical parameters produced by a DRC parameter estimator.

    The four time constants use seconds, matching :func:`drcinv.drc_inverse`.
    ``normalized`` preserves the six raw network outputs for diagnostics.
    """

    threshold_dbfs: float
    ratio: float
    envelope_attack_s: float
    envelope_release_s: float
    gain_attack_s: float
    gain_release_s: float
    detector_order: int = 2
    normalized: Tuple[float, ...] = ()

    def as_vector(self, dtype=np.float64) -> np.ndarray:
        """Return ``[L, R, tau_v_att, tau_v_rel, tau_g_att, tau_g_rel, p]``."""

        return np.asarray(
            [
                self.threshold_dbfs,
                self.ratio,
                self.envelope_attack_s,
                self.envelope_release_s,
                self.gain_attack_s,
                self.gain_release_s,
                self.detector_order,
            ],
            dtype=dtype,
        )

    def as_dict(self) -> dict:
        """Return names compatible with the notation used by ``drcinv``."""

        return {
            "L": self.threshold_dbfs,
            "R": self.ratio,
            "tau_v_att": self.envelope_attack_s,
            "tau_v_rel": self.envelope_release_s,
            "tau_g_att": self.gain_attack_s,
            "tau_g_rel": self.gain_release_s,
            "p": self.detector_order,
        }


def _default_cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME")
    if root:
        return Path(root).expanduser() / "drcinv"
    return Path.home() / ".cache" / "drcinv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_checkpoint(
    url: str = DEFAULT_CHECKPOINT_URL,
    *,
    sha256: Optional[str] = DEFAULT_CHECKPOINT_SHA256,
    cache_dir: Optional[PathLike] = None,
) -> Path:
    """Download a Release checkpoint once and return its cached path.

    The environment variable ``DRCINV_MEE_CHECKPOINT_URL`` overrides ``url``.
    Downloads are written atomically and verified when ``sha256`` is given.
    """

    url = os.environ.get("DRCINV_MEE_CHECKPOINT_URL", url)
    destination_dir = Path(cache_dir or _default_cache_dir()).expanduser()
    filename = Path(urlparse(url).path).name or "MEE_compression_large.pt"
    destination = destination_dir / filename
    destination_dir.mkdir(parents=True, exist_ok=True)

    if destination.is_file():
        if sha256 is None or _sha256(destination) == sha256.lower():
            return destination
        destination.unlink()

    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=filename + ".",
            suffix=".part",
            dir=destination_dir,
            delete=False,
        ) as temporary:
            temporary_name = Path(temporary.name)
            with urlopen(url) as response:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    temporary.write(block)

        if sha256 is not None:
            actual = _sha256(temporary_name)
            if actual != sha256.lower():
                raise ValueError(
                    "Checkpoint checksum mismatch: "
                    f"expected {sha256.lower()}, received {actual}."
                )
        temporary_name.replace(destination)
    except Exception:
        if temporary_name is not None and temporary_name.exists():
            temporary_name.unlink()
        raise

    return destination


def _import_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as error:
        raise ImportError(
            "MEE parameter estimation requires PyTorch. Install it with "
            "`pip install 'drcinv[param]'`."
        ) from error
    return torch, nn


def _build_mee_model():
    torch, nn = _import_torch()

    class Conv1dLayer(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
        ):
            super().__init__()
            pad = (kernel_size - 1) * dilation
            left = pad // 2
            right = pad - left
            self.conv1d = nn.Sequential()
            self.conv1d.add_module("conv1d_pad", nn.ReflectionPad1d((left, right)))
            self.conv1d.add_module(
                "conv1d",
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                ),
            )
            self.conv1d.add_module("batch_norm", nn.BatchNorm1d(out_channels))
            self.conv1d.add_module("relu", nn.ReLU())

        def forward(self, inputs):
            return self.conv1d(inputs)

    class ResidualConvBlock(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
        ):
            super().__init__()
            self.conv1 = Conv1dLayer(
                in_channels, in_channels, kernel_size, dilation=dilation
            )
            self.conv2 = Conv1dLayer(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
            )

        def forward(self, inputs):
            return self.conv2(self.conv1(inputs) + inputs)

    class MEE(nn.Module):
        def __init__(self):
            super().__init__()
            channels = [1] + list(_MEE_CONFIG["channels"])
            blocks = []
            for index, kernel_size in enumerate(_MEE_CONFIG["kernels"]):
                blocks.append(
                    ResidualConvBlock(
                        channels[index],
                        channels[index + 1],
                        kernel_size,
                        _MEE_CONFIG["strides"][index],
                        _MEE_CONFIG["dilation"][index],
                    )
                )
            self.encoder = nn.Sequential(*blocks)
            self.glob_pool = nn.AdaptiveAvgPool1d(1)
            self.out_dim = channels[-1]

            self.MLP = nn.Sequential()
            dimensions = [self.out_dim, 2048, 1024, 512]
            module_index = 0
            for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
                self.MLP.add_module(
                    str(module_index), nn.Linear(input_dim, output_dim)
                )
                module_index += 1
                self.MLP.add_module(str(module_index), nn.BatchNorm1d(output_dim))
                module_index += 1
                self.MLP.add_module(str(module_index), nn.PReLU())
                module_index += 1
            self.MLP.add_module(str(module_index), nn.Linear(dimensions[-1], 6))
            self.MLP.add_module(str(module_index + 1), nn.Sigmoid())

        def forward(self, inputs):
            encoded = self.encoder(inputs)
            return self.MLP(self.glob_pool(encoded).squeeze(-1))

    return MEE()


def _prepare_audio(
    audio: Union[np.ndarray, Sequence[float]],
    sample_rate: int,
    duration_seconds: float,
) -> np.ndarray:
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be strictly positive.")

    waveform = np.asarray(audio)
    if waveform.ndim == 2:
        if waveform.shape[1] <= 8:
            waveform = waveform.mean(axis=1)
        elif waveform.shape[0] <= 8:
            waveform = waveform.mean(axis=0)
        else:
            raise ValueError(
                "For 2-D audio, one dimension must contain at most 8 channels."
            )
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("audio must be a non-empty mono or multichannel waveform.")
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    if not np.all(np.isfinite(waveform)):
        raise ValueError("audio contains NaN or Inf.")

    target_samples = int(round(sample_rate * duration_seconds))
    if waveform.size < target_samples:
        waveform = np.pad(waveform, (0, target_samples - waveform.size))
    elif waveform.size > target_samples:
        start = (waveform.size - target_samples) // 2
        waveform = waveform[start : start + target_samples]
    return np.ascontiguousarray(waveform)


class MEEParameterEstimator:
    """Reusable MEE estimator for compression parameters.

    Audio is converted to mono and center-cropped or zero-padded to five
    seconds at its native sample rate.  No resampling or peak normalization is
    applied because signal level is meaningful to a DRC parameter estimator.
    """

    def __init__(
        self,
        checkpoint_path: Optional[PathLike] = None,
        *,
        checkpoint_url: str = DEFAULT_CHECKPOINT_URL,
        checkpoint_sha256: Optional[str] = DEFAULT_CHECKPOINT_SHA256,
        cache_dir: Optional[PathLike] = None,
        device: str = "auto",
        duration_seconds: float = DEFAULT_DURATION_SECONDS,
        control_ranges: np.ndarray = COMPRESSION_LARGE_RANGES,
    ):
        self.checkpoint_path = (
            None if checkpoint_path is None else Path(checkpoint_path).expanduser()
        )
        self.checkpoint_url = checkpoint_url
        self.checkpoint_sha256 = checkpoint_sha256
        self.cache_dir = cache_dir
        self.device_request = device
        self.duration_seconds = float(duration_seconds)
        ranges = np.asarray(control_ranges, dtype=np.float32)
        if ranges.shape != (6, 2) or np.any(ranges[:, 1] < ranges[:, 0]):
            raise ValueError("control_ranges must have shape (6, 2).")
        self.control_ranges = ranges.copy()
        self.model = None
        self.device = None

    def load(self) -> "MEEParameterEstimator":
        """Load the model and checkpoint, downloading it only when necessary."""

        if self.model is not None:
            return self
        torch, _ = _import_torch()
        if self.device_request == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_request)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

        checkpoint_path = self.checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = download_checkpoint(
                self.checkpoint_url,
                sha256=self.checkpoint_sha256,
                cache_dir=self.cache_dir,
            )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"MEE checkpoint does not exist: {checkpoint_path}")

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError("MEE checkpoint must contain a state dictionary.")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }

        model = _build_mee_model()
        model.load_state_dict(state_dict, strict=True)
        self.model = model.to(self.device).eval()
        return self

    def estimate(
        self,
        audio: Union[np.ndarray, Sequence[float]],
        sample_rate: int,
    ) -> DRCParameters:
        """Estimate seven parameters accepted directly by ``drcinv``."""

        self.load()
        torch, _ = _import_torch()
        waveform = _prepare_audio(audio, sample_rate, self.duration_seconds)
        inputs = torch.from_numpy(waveform).reshape(1, 1, -1).to(self.device)
        with torch.inference_mode():
            normalized = self.model(inputs).squeeze(0).detach().cpu().numpy()

        lows = self.control_ranges[:, 0]
        highs = self.control_ranges[:, 1]
        physical = lows + (highs - lows) * normalized
        return DRCParameters(
            threshold_dbfs=float(physical[0]),
            ratio=float(physical[1]),
            envelope_attack_s=float(physical[2] / 1000.0),
            envelope_release_s=float(physical[3] / 1000.0),
            gain_attack_s=float(physical[4] / 1000.0),
            gain_release_s=float(physical[5] / 1000.0),
            detector_order=2,
            normalized=tuple(float(value) for value in normalized),
        )


def estimate_mee_parameters(
    audio: Union[np.ndarray, Sequence[float]],
    sample_rate: int,
    *,
    checkpoint_path: Optional[PathLike] = None,
    device: str = "auto",
) -> np.ndarray:
    """One-shot MEE wrapper returning a seven-element DRC vector."""

    estimator = MEEParameterEstimator(checkpoint_path, device=device)
    return estimator.estimate(audio, sample_rate).as_vector()
