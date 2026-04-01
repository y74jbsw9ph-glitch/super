"""
Tests for bearing_diagnosis.py and generate_signal.py
"""

import math

import numpy as np
import pytest

from bearing_diagnosis import (
    BearingGeometry,
    DiagnosisResult,
    FaultType,
    Features,
    bandpass_filter,
    diagnose,
    envelope,
    envelope_spectrum,
    extract_features,
    power_spectrum,
)
from generate_signal import generate_signal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def geometry():
    """Standard 6205 deep-groove ball bearing geometry."""
    return BearingGeometry(
        n_balls=9,
        ball_diameter=7.938,
        pitch_diameter=38.5,
        contact_angle=0.0,
    )


@pytest.fixture
def shaft_hz():
    return 25.0   # 1500 RPM


@pytest.fixture
def fs():
    return 12_000.0   # Hz


@pytest.fixture
def duration():
    return 2.0   # seconds


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# BearingGeometry.characteristic_frequencies
# ---------------------------------------------------------------------------

class TestCharacteristicFrequencies:
    def test_known_values(self, geometry, shaft_hz):
        freqs = geometry.characteristic_frequencies(shaft_hz)
        assert set(freqs.keys()) == {"ftf", "bpfo", "bpfi", "bsf"}
        for name, val in freqs.items():
            assert val > 0, f"{name} must be positive"

    def test_bpfi_greater_than_bpfo(self, geometry, shaft_hz):
        freqs = geometry.characteristic_frequencies(shaft_hz)
        assert freqs["bpfi"] > freqs["bpfo"]

    def test_bpfo_equals_n_times_ftf(self, geometry, shaft_hz):
        freqs = geometry.characteristic_frequencies(shaft_hz)
        assert math.isclose(freqs["bpfo"], geometry.n_balls * freqs["ftf"], rel_tol=1e-9)

    def test_invalid_shaft_speed(self, geometry):
        with pytest.raises(ValueError):
            geometry.characteristic_frequencies(-1.0)
        with pytest.raises(ValueError):
            geometry.characteristic_frequencies(0.0)

    def test_scales_with_shaft_hz(self, geometry):
        f1 = geometry.characteristic_frequencies(10.0)
        f2 = geometry.characteristic_frequencies(20.0)
        for key in f1:
            assert math.isclose(f2[key], 2.0 * f1[key], rel_tol=1e-9)


# ---------------------------------------------------------------------------
# bandpass_filter
# ---------------------------------------------------------------------------

class TestBandpassFilter:
    def test_output_length(self, fs, rng):
        x = rng.standard_normal(1000)
        y = bandpass_filter(x, fs, 1000.0, 5000.0)
        assert len(y) == len(x)

    def test_attenuates_out_of_band(self, fs):
        t = np.arange(int(fs)) / fs
        # Pure 100 Hz tone — well below the bandpass [1000, 5000]
        x = np.sin(2.0 * np.pi * 100.0 * t)
        y = bandpass_filter(x, fs, 1000.0, 5000.0)
        assert np.sqrt(np.mean(y ** 2)) < 0.01 * np.sqrt(np.mean(x ** 2))

    def test_passes_in_band(self, fs):
        t = np.arange(int(fs)) / fs
        # Pure 3000 Hz tone — inside the bandpass [1000, 5000]
        x = np.sin(2.0 * np.pi * 3000.0 * t)
        y = bandpass_filter(x, fs, 1000.0, 5000.0)
        assert np.sqrt(np.mean(y ** 2)) > 0.5 * np.sqrt(np.mean(x ** 2))

    def test_invalid_cutoffs(self, fs, rng):
        x = rng.standard_normal(512)
        with pytest.raises(ValueError):
            bandpass_filter(x, fs, 5000.0, 1000.0)   # swapped
        with pytest.raises(ValueError):
            bandpass_filter(x, fs, 0.0, 1000.0)       # zero lowcut
        with pytest.raises(ValueError):
            bandpass_filter(x, fs, 1000.0, fs / 2.0)  # highcut == Nyquist


# ---------------------------------------------------------------------------
# envelope & envelope_spectrum
# ---------------------------------------------------------------------------

class TestEnvelope:
    def test_non_negative(self, rng):
        x = rng.standard_normal(512)
        env = envelope(x)
        assert np.all(env >= 0)

    def test_length_preserved(self, rng):
        x = rng.standard_normal(512)
        assert len(envelope(x)) == len(x)

    def test_am_signal(self, fs):
        """Envelope of an AM signal should recover the modulating frequency."""
        t = np.arange(int(fs)) / fs
        carrier_freq = 3000.0
        mod_freq = 50.0
        # AM: (1 + m·cos(2π·fm·t))·cos(2π·fc·t)
        x = (1.0 + 0.8 * np.cos(2.0 * np.pi * mod_freq * t)) * np.cos(
            2.0 * np.pi * carrier_freq * t
        )
        env_freqs, env_amps = envelope_spectrum(x, fs)
        # Dominant frequency in the envelope spectrum should be near mod_freq
        dominant_idx = np.argmax(env_amps[1:]) + 1
        dominant_freq = env_freqs[dominant_idx]
        assert abs(dominant_freq - mod_freq) < 5.0, (
            f"Expected ~{mod_freq} Hz, got {dominant_freq:.1f} Hz"
        )


# ---------------------------------------------------------------------------
# power_spectrum
# ---------------------------------------------------------------------------

class TestPowerSpectrum:
    def test_output_shapes(self, rng, fs):
        x = rng.standard_normal(2048)
        freqs, psd = power_spectrum(x, fs)
        assert len(freqs) == len(psd)
        assert freqs[0] == 0.0
        assert freqs[-1] <= fs / 2.0

    def test_single_tone_peak(self, fs):
        t = np.arange(int(2 * fs)) / fs
        tone_freq = 500.0
        x = np.sin(2.0 * np.pi * tone_freq * t)
        freqs, psd = power_spectrum(x, fs)
        peak_freq = freqs[np.argmax(psd)]
        assert abs(peak_freq - tone_freq) < 20.0


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_returns_features_instance(self, rng, fs):
        x = rng.standard_normal(1024)
        feat = extract_features(x, fs)
        assert isinstance(feat, Features)

    def test_rms_correct(self, fs):
        t = np.arange(int(fs)) / fs
        x = np.sin(2.0 * np.pi * 50.0 * t)   # unit sine → RMS = 1/√2
        feat = extract_features(x, fs)
        assert math.isclose(feat.rms, 1.0 / math.sqrt(2.0), rel_tol=1e-3)

    def test_kurtosis_gaussian(self, rng, fs):
        x = rng.standard_normal(10_000)
        feat = extract_features(x, fs)
        # Excess kurtosis of Gaussian ≈ 0
        assert abs(feat.kurtosis) < 0.5

    def test_high_kurtosis_impulse(self, fs, rng):
        x = rng.standard_normal(1024) * 0.01
        x[200] = 10.0    # single large impulse → high kurtosis
        feat = extract_features(x, fs)
        assert feat.kurtosis > 5.0

    def test_empty_raises(self, fs):
        with pytest.raises(ValueError):
            extract_features(np.array([]), fs)

    def test_char_freq_energies_populated(self, geometry, shaft_hz, fs, rng):
        x = rng.standard_normal(4096)
        char_freqs = geometry.characteristic_frequencies(shaft_hz)
        feat = extract_features(x, fs, char_freqs=char_freqs)
        # All energy values should be non-negative floats
        for attr in ("energy_bpfo", "energy_bpfi", "energy_bsf", "energy_ftf"):
            assert getattr(feat, attr) >= 0.0


# ---------------------------------------------------------------------------
# generate_signal
# ---------------------------------------------------------------------------

class TestGenerateSignal:
    def test_output_length(self, geometry, shaft_hz, fs, duration, rng):
        x = generate_signal(geometry, shaft_hz, fs, duration, rng=rng)
        assert len(x) == int(fs * duration)

    def test_normal_is_quiet(self, geometry, shaft_hz, fs, duration, rng):
        x = generate_signal(
            geometry, shaft_hz, fs, duration,
            fault_type=FaultType.NORMAL, noise_std=0.05, rng=rng,
        )
        # RMS should be small (no fault impulses)
        assert np.sqrt(np.mean(x ** 2)) < 0.5

    def test_faulty_higher_kurtosis(self, geometry, shaft_hz, fs, duration):
        """Faulty signal should have higher kurtosis than normal signal."""
        rng_normal = np.random.default_rng(0)
        rng_faulty = np.random.default_rng(0)
        x_normal = generate_signal(
            geometry, shaft_hz, fs, duration,
            fault_type=FaultType.NORMAL, noise_std=0.05, rng=rng_normal,
        )
        x_faulty = generate_signal(
            geometry, shaft_hz, fs, duration,
            fault_type=FaultType.OUTER_RACE, fault_amplitude=2.0,
            noise_std=0.05, rng=rng_faulty,
        )
        from scipy.stats import kurtosis as scipy_kurtosis
        k_normal = scipy_kurtosis(x_normal, fisher=True)
        k_faulty = scipy_kurtosis(x_faulty, fisher=True)
        assert k_faulty > k_normal

    @pytest.mark.parametrize("fault_type", list(FaultType))
    def test_all_fault_types_run(self, geometry, shaft_hz, fs, duration, rng, fault_type):
        x = generate_signal(
            geometry, shaft_hz, fs, duration,
            fault_type=fault_type, rng=rng,
        )
        assert len(x) == int(fs * duration)


# ---------------------------------------------------------------------------
# diagnose (end-to-end)
# ---------------------------------------------------------------------------

class TestDiagnose:
    def test_returns_diagnosis_result(self, geometry, shaft_hz, fs, duration, rng):
        x = generate_signal(geometry, shaft_hz, fs, duration, rng=rng)
        result = diagnose(x, fs, geometry, shaft_hz)
        assert isinstance(result, DiagnosisResult)
        assert isinstance(result.fault_type, FaultType)
        assert 0.0 <= result.confidence <= 1.0

    def test_normal_bearing_classified_normal(self, geometry, shaft_hz, fs, duration):
        """A low-kurtosis signal should be classified as NORMAL."""
        rng = np.random.default_rng(7)
        x = generate_signal(
            geometry, shaft_hz, fs, duration,
            fault_type=FaultType.NORMAL, noise_std=0.05, rng=rng,
        )
        result = diagnose(x, fs, geometry, shaft_hz)
        assert result.fault_type == FaultType.NORMAL

    def test_outer_race_fault_detected(self, geometry, shaft_hz, fs, duration):
        """A strong outer-race fault signal should be identified."""
        rng = np.random.default_rng(42)
        x = generate_signal(
            geometry, shaft_hz, fs, duration,
            fault_type=FaultType.OUTER_RACE,
            fault_amplitude=5.0,
            noise_std=0.02,
            rng=rng,
        )
        result = diagnose(x, fs, geometry, shaft_hz)
        assert result.fault_type == FaultType.OUTER_RACE

    def test_inner_race_fault_detected(self, geometry, shaft_hz, fs, duration):
        rng = np.random.default_rng(42)
        x = generate_signal(
            geometry, shaft_hz, fs, duration,
            fault_type=FaultType.INNER_RACE,
            fault_amplitude=5.0,
            noise_std=0.02,
            rng=rng,
        )
        result = diagnose(x, fs, geometry, shaft_hz)
        assert result.fault_type == FaultType.INNER_RACE

    def test_scores_sum_close_to_one_for_faulty(self, geometry, shaft_hz, fs, duration):
        rng = np.random.default_rng(42)
        x = generate_signal(
            geometry, shaft_hz, fs, duration,
            fault_type=FaultType.OUTER_RACE, fault_amplitude=5.0,
            noise_std=0.02, rng=rng,
        )
        result = diagnose(x, fs, geometry, shaft_hz)
        # Scores for the four fault types (excluding NORMAL) should sum to 1.0
        fault_scores = [
            result.scores[ft.value]
            for ft in FaultType
            if ft != FaultType.NORMAL
        ]
        assert math.isclose(sum(fault_scores), 1.0, rel_tol=1e-6)

    def test_features_attached(self, geometry, shaft_hz, fs, duration, rng):
        x = generate_signal(geometry, shaft_hz, fs, duration, rng=rng)
        result = diagnose(x, fs, geometry, shaft_hz)
        assert result.features is not None
        assert isinstance(result.features, Features)
