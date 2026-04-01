"""
轴承振动信号模拟生成工具 (Bearing Vibration Signal Generator)

Generates synthetic vibration signals for healthy bearings and four common
fault types (outer race, inner race, ball, cage) to support testing and
demonstration of the bearing fault diagnosis module.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from bearing_diagnosis import BearingGeometry, FaultType


def generate_signal(
    geometry: BearingGeometry,
    shaft_hz: float,
    fs: float,
    duration: float,
    fault_type: FaultType = FaultType.NORMAL,
    fault_amplitude: float = 1.0,
    noise_std: float = 0.05,
    resonance_freq: float = 3000.0,
    resonance_damping: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a synthetic bearing vibration signal.

    The model consists of:

    * Background Gaussian noise representing structural vibration.
    * A shaft harmonic at *shaft_hz* (imbalance / misalignment).
    * For faulty bearings: periodic impulses at the relevant characteristic
      frequency, each convolved with a decaying sinusoidal resonance response
      (simulating the structural resonance excited by each impact).

    Parameters
    ----------
    geometry:          Bearing geometric parameters.
    shaft_hz:          Shaft rotation frequency [Hz].
    fs:                Sampling frequency [Hz].
    duration:          Signal duration [seconds].
    fault_type:        Type of fault to simulate (default: NORMAL).
    fault_amplitude:   Amplitude of fault impulses relative to noise (default 1.0).
    noise_std:         Standard deviation of background Gaussian noise.
    resonance_freq:    Structural resonance frequency [Hz] excited by impacts.
    resonance_damping: Damping ratio of the resonance response (0–1).
    rng:               Optional :class:`numpy.random.Generator` for reproducibility.

    Returns
    -------
    1-D NumPy array of length ``int(fs * duration)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = int(fs * duration)
    t = np.arange(n) / fs

    # Background noise + shaft harmonic
    x = rng.normal(0.0, noise_std, n)
    x += 0.1 * np.sin(2.0 * np.pi * shaft_hz * t)

    if fault_type == FaultType.NORMAL:
        return x

    # Characteristic frequencies
    char_freqs = geometry.characteristic_frequencies(shaft_hz)
    fault_freq_map = {
        FaultType.OUTER_RACE: char_freqs["bpfo"],
        FaultType.INNER_RACE: char_freqs["bpfi"],
        FaultType.BALL: char_freqs["bsf"],
        FaultType.CAGE: char_freqs["ftf"],
    }
    fault_freq = fault_freq_map[fault_type]

    # Build the resonance impulse response h(t)
    # h(t) = exp(-ζ·ωn·t) · sin(ωd·t), t ≥ 0
    omega_n = 2.0 * np.pi * resonance_freq
    omega_d = omega_n * np.sqrt(max(1.0 - resonance_damping ** 2, 1e-6))
    h_len = int(min(fs / fault_freq * 2, n // 4))   # two fault periods
    t_h = np.arange(h_len) / fs
    h = np.exp(-resonance_damping * omega_n * t_h) * np.sin(omega_d * t_h)
    h /= np.max(np.abs(h)) if np.max(np.abs(h)) > 0 else 1.0

    # Place impulses at fault period, with slight random jitter (slip)
    period = fs / fault_freq
    impulse_positions = []
    pos = 0.0
    while pos < n:
        jitter = rng.uniform(-0.05 * period, 0.05 * period)
        idx = int(round(pos + jitter))
        if 0 <= idx < n:
            impulse_positions.append(idx)
        pos += period

    # Superpose each convolved impulse onto the signal
    for idx in impulse_positions:
        end = min(idx + h_len, n)
        x[idx:end] += fault_amplitude * h[: end - idx]

    return x
