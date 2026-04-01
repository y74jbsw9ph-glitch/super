"""
轴承故障诊断 (Bearing Fault Diagnosis)

This module provides tools for diagnosing rolling-element bearing faults
using vibration signal analysis, including:

  - Bearing characteristic frequency calculation
  - Signal preprocessing (bandpass filter, Hilbert envelope)
  - Time-domain and frequency-domain feature extraction
  - Rule-based fault classification
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew


# ---------------------------------------------------------------------------
# Bearing geometry & characteristic frequencies
# ---------------------------------------------------------------------------

class FaultType(Enum):
    """Enumeration of bearing fault types."""
    NORMAL = "normal"
    OUTER_RACE = "outer_race"   # 外圈故障
    INNER_RACE = "inner_race"   # 内圈故障
    BALL = "ball"               # 滚动体故障
    CAGE = "cage"               # 保持架故障


@dataclass
class BearingGeometry:
    """Geometric parameters of a rolling-element bearing.

    Attributes:
        n_balls:        Number of rolling elements (balls or rollers).
        ball_diameter:  Ball (roller) diameter [mm].
        pitch_diameter: Bearing pitch diameter [mm].
        contact_angle:  Contact angle [degrees].
    """
    n_balls: int
    ball_diameter: float
    pitch_diameter: float
    contact_angle: float = 0.0

    def characteristic_frequencies(self, shaft_hz: float) -> Dict[str, float]:
        """Compute the four bearing characteristic frequencies.

        Parameters
        ----------
        shaft_hz:
            Shaft rotation frequency [Hz].

        Returns
        -------
        dict with keys ``bpfo``, ``bpfi``, ``bsf``, ``ftf`` [Hz].

        Notes
        -----
        Formulae (standard rolling-element bearing theory):

        * FTF  = fr/2 · (1 − Bd/Pd · cos θ)
        * BPFO = n/2 · fr · (1 − Bd/Pd · cos θ)
        * BPFI = n/2 · fr · (1 + Bd/Pd · cos θ)
        * BSF  = Pd/(2·Bd) · fr · [1 − (Bd/Pd · cos θ)²]
        """
        if shaft_hz <= 0:
            raise ValueError("shaft_hz must be positive")
        fr = shaft_hz
        bd_pd = self.ball_diameter / self.pitch_diameter
        cos_theta = math.cos(math.radians(self.contact_angle))
        ratio = bd_pd * cos_theta

        ftf = fr / 2.0 * (1.0 - ratio)
        bpfo = self.n_balls / 2.0 * fr * (1.0 - ratio)
        bpfi = self.n_balls / 2.0 * fr * (1.0 + ratio)
        bsf = (self.pitch_diameter / (2.0 * self.ball_diameter)) * fr * (1.0 - ratio ** 2)

        return {"ftf": ftf, "bpfo": bpfo, "bpfi": bpfi, "bsf": bsf}


# ---------------------------------------------------------------------------
# Signal preprocessing
# ---------------------------------------------------------------------------

def bandpass_filter(
    x: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 5,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    x:       Input signal array.
    fs:      Sampling frequency [Hz].
    lowcut:  Lower cutoff frequency [Hz].
    highcut: Upper cutoff frequency [Hz].
    order:   Filter order (default 5).

    Returns
    -------
    Filtered signal (same length as *x*).
    """
    nyq = fs / 2.0
    if not (0 < lowcut < highcut < nyq):
        raise ValueError(
            f"Require 0 < lowcut ({lowcut}) < highcut ({highcut}) < Nyquist ({nyq})"
        )
    low = lowcut / nyq
    high = highcut / nyq
    sos = sp_signal.butter(order, [low, high], btype="band", output="sos")
    return sp_signal.sosfiltfilt(sos, x)


def envelope(x: np.ndarray) -> np.ndarray:
    """Compute the analytic envelope of *x* via the Hilbert transform.

    Parameters
    ----------
    x: Input signal array.

    Returns
    -------
    Envelope signal (non-negative, same length as *x*).
    """
    analytic = sp_signal.hilbert(x)
    return np.abs(analytic)


def envelope_spectrum(
    x: np.ndarray, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the one-sided envelope spectrum.

    Parameters
    ----------
    x:  Input signal array.
    fs: Sampling frequency [Hz].

    Returns
    -------
    (freqs, amplitudes) — both one-sided (0 … fs/2).
    """
    env = envelope(x)
    env_ac = env - env.mean()           # remove DC component
    n = len(env_ac)
    fft_vals = np.fft.rfft(env_ac)
    amplitudes = (2.0 / n) * np.abs(fft_vals)
    amplitudes[0] /= 2.0                # DC bin is single-sided already
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, amplitudes


def power_spectrum(
    x: np.ndarray, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the one-sided power spectral density using Welch's method.

    Parameters
    ----------
    x:  Input signal array.
    fs: Sampling frequency [Hz].

    Returns
    -------
    (freqs, psd) — both one-sided.
    """
    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=min(256, len(x)))
    return freqs, psd


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class Features:
    """Container for extracted bearing vibration features."""

    # Time-domain features
    rms: float = 0.0
    peak: float = 0.0
    crest_factor: float = 0.0
    kurtosis: float = 0.0
    skewness: float = 0.0
    shape_factor: float = 0.0

    # Frequency-domain features
    dominant_freq: float = 0.0
    spectral_centroid: float = 0.0
    spectral_kurtosis: float = 0.0

    # Envelope-spectrum energy at characteristic frequencies (if computed)
    energy_bpfo: float = 0.0
    energy_bpfi: float = 0.0
    energy_bsf: float = 0.0
    energy_ftf: float = 0.0


def extract_features(
    x: np.ndarray,
    fs: float,
    char_freqs: Optional[Dict[str, float]] = None,
    freq_tol_hz: float = 2.0,
) -> Features:
    """Extract time-domain and frequency-domain features from a vibration signal.

    Parameters
    ----------
    x:            Vibration signal array.
    fs:           Sampling frequency [Hz].
    char_freqs:   Dictionary of characteristic frequencies returned by
                  :meth:`BearingGeometry.characteristic_frequencies`.
                  When provided, envelope-spectrum energy at each fault
                  frequency is also computed.
    freq_tol_hz:  Bandwidth [Hz] around each characteristic frequency used
                  when summing envelope-spectrum energy (default 2 Hz).

    Returns
    -------
    :class:`Features` dataclass instance.
    """
    if len(x) == 0:
        raise ValueError("Signal array must not be empty")

    feat = Features()

    # --- Time-domain ---
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    mean_abs = float(np.mean(np.abs(x)))

    feat.rms = rms
    feat.peak = peak
    feat.crest_factor = peak / rms if rms > 0 else 0.0
    feat.kurtosis = float(kurtosis(x, fisher=True))   # excess kurtosis
    feat.skewness = float(skew(x))
    feat.shape_factor = rms / mean_abs if mean_abs > 0 else 0.0

    # --- Frequency-domain (Welch PSD) ---
    freqs_psd, psd = power_spectrum(x, fs)
    if psd.sum() > 0:
        feat.dominant_freq = float(freqs_psd[np.argmax(psd)])
        feat.spectral_centroid = float(np.sum(freqs_psd * psd) / psd.sum())
        # Spectral kurtosis approximation via the PSD shape
        mu = feat.spectral_centroid
        variance = float(np.sum((freqs_psd - mu) ** 2 * psd) / psd.sum())
        if variance > 0:
            fourth = float(np.sum((freqs_psd - mu) ** 4 * psd) / psd.sum())
            feat.spectral_kurtosis = fourth / (variance ** 2)

    # --- Envelope-spectrum energy at characteristic frequencies ---
    if char_freqs is not None:
        env_freqs, env_amps = envelope_spectrum(x, fs)
        df = env_freqs[1] - env_freqs[0] if len(env_freqs) > 1 else 1.0
        bins = max(1, int(round(freq_tol_hz / df)))

        def _energy_at(fc: float) -> float:
            idx = int(round(fc / df))
            lo = max(0, idx - bins)
            hi = min(len(env_amps), idx + bins + 1)
            return float(np.sum(env_amps[lo:hi] ** 2))

        feat.energy_bpfo = _energy_at(char_freqs["bpfo"])
        feat.energy_bpfi = _energy_at(char_freqs["bpfi"])
        feat.energy_bsf = _energy_at(char_freqs["bsf"])
        feat.energy_ftf = _energy_at(char_freqs["ftf"])

    return feat


# ---------------------------------------------------------------------------
# Fault classification
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisResult:
    """Result returned by :func:`diagnose`."""
    fault_type: FaultType
    confidence: float                      # 0–1 normalised score
    scores: Dict[str, float] = field(default_factory=dict)
    features: Optional[Features] = None


def diagnose(
    x: np.ndarray,
    fs: float,
    geometry: BearingGeometry,
    shaft_hz: float,
    bandpass_lowcut: float = 1000.0,
    bandpass_highcut: Optional[float] = None,
    kurtosis_threshold: float = 3.0,
) -> DiagnosisResult:
    """Diagnose bearing condition from a raw vibration signal.

    The algorithm follows the industry-standard Bearing Envelope Analysis
    (High-Frequency Resonance Technique, HFRT) work-flow:

    1. Bandpass-filter the raw signal around the structural resonance.
    2. Extract the analytic envelope and compute its spectrum.
    3. Compare spectral energy peaks against the four characteristic fault
       frequencies (BPFO, BPFI, BSF, FTF).
    4. Apply rule-based scoring with a kurtosis gating check.

    Parameters
    ----------
    x:                Raw vibration signal (acceleration) [any consistent unit].
    fs:               Sampling frequency [Hz].
    geometry:         Bearing geometric parameters.
    shaft_hz:         Shaft rotation frequency [Hz].
    bandpass_lowcut:  Lower edge of the resonance bandpass filter [Hz].
    bandpass_highcut: Upper edge; defaults to ``fs/2 - 1``.
    kurtosis_threshold:
                      Minimum kurtosis value above which an impulsive fault
                      is considered possible (default 3.0).

    Returns
    -------
    :class:`DiagnosisResult` with fault type, confidence score, and per-class
    raw scores.
    """
    if bandpass_highcut is None:
        bandpass_highcut = fs / 2.0 - 1.0

    char_freqs = geometry.characteristic_frequencies(shaft_hz)

    # 1. Pre-filter
    x_filtered = bandpass_filter(x, fs, bandpass_lowcut, bandpass_highcut)

    # 2. Feature extraction
    feat = extract_features(x_filtered, fs, char_freqs=char_freqs)

    # 3. Rule-based scoring
    # If kurtosis is below threshold, bearing is likely healthy
    if feat.kurtosis < kurtosis_threshold:
        scores = {
            FaultType.NORMAL.value: 1.0,
            FaultType.OUTER_RACE.value: 0.0,
            FaultType.INNER_RACE.value: 0.0,
            FaultType.BALL.value: 0.0,
            FaultType.CAGE.value: 0.0,
        }
        return DiagnosisResult(
            fault_type=FaultType.NORMAL,
            confidence=1.0,
            scores=scores,
            features=feat,
        )

    # Score each fault type by its envelope-spectrum energy
    energy_map = {
        FaultType.OUTER_RACE: feat.energy_bpfo,
        FaultType.INNER_RACE: feat.energy_bpfi,
        FaultType.BALL: feat.energy_bsf,
        FaultType.CAGE: feat.energy_ftf,
    }

    total = sum(energy_map.values())
    if total == 0:
        # Impulsive signal but no dominant fault frequency → report unknown
        scores = {k.value: 0.0 for k in FaultType}
        scores[FaultType.NORMAL.value] = 1.0
        return DiagnosisResult(
            fault_type=FaultType.NORMAL,
            confidence=0.5,
            scores=scores,
            features=feat,
        )

    # Normalise to probabilities
    raw_scores = {ft: e / total for ft, e in energy_map.items()}
    best_ft = max(raw_scores, key=lambda k: raw_scores[k])
    confidence = raw_scores[best_ft]

    scores_named = {ft.value: raw_scores[ft] for ft in raw_scores}
    scores_named[FaultType.NORMAL.value] = 0.0

    return DiagnosisResult(
        fault_type=best_ft,
        confidence=confidence,
        scores=scores_named,
        features=feat,
    )
