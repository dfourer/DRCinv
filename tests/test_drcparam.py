import hashlib

import numpy as np
import pytest

from drcinv import drcparam
from drcinv.drcparam.mee import _prepare_audio


def test_parameter_vector_and_dict_use_seconds():
    parameters = drcparam.DRCParameters(
        threshold_dbfs=-24.0,
        ratio=4.0,
        envelope_attack_s=0.005,
        envelope_release_s=0.050,
        gain_attack_s=0.010,
        gain_release_s=0.100,
        normalized=(0.1,) * 6,
    )
    np.testing.assert_allclose(
        parameters.as_vector(), [-24, 4, 0.005, 0.050, 0.010, 0.100, 2]
    )
    assert parameters.as_dict()["tau_g_rel"] == 0.100


@pytest.mark.parametrize("sample_rate", [8000, 22050, 44100, 48000])
def test_prepare_audio_uses_native_sample_rate(sample_rate):
    stereo = np.ones((sample_rate * 6, 2), dtype=np.float64)
    prepared = _prepare_audio(stereo, sample_rate, 5.0)
    assert prepared.shape == (sample_rate * 5,)
    assert prepared.dtype == np.float32


def test_prepare_audio_pads_without_resampling():
    audio = np.arange(10, dtype=np.float32)
    prepared = _prepare_audio(audio, 10, 2.0)
    np.testing.assert_array_equal(prepared[:10], audio)
    np.testing.assert_array_equal(prepared[10:], 0)


def test_download_checkpoint_uses_verified_cache(tmp_path):
    payload = b"checkpoint bytes"
    source = tmp_path / "source.pt"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    cache = tmp_path / "cache"

    first = drcparam.download_checkpoint(
        source.as_uri(), sha256=digest, cache_dir=cache
    )
    source.unlink()
    second = drcparam.download_checkpoint(
        source.as_uri(), sha256=digest, cache_dir=cache
    )

    assert first == second
    assert second.read_bytes() == payload


def test_invalid_control_ranges_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        drcparam.MEEParameterEstimator(control_ranges=np.zeros((5, 2)))
