# 轴承故障诊断 (Bearing Fault Diagnosis)

A Python library for rolling-element bearing fault diagnosis using vibration
signal analysis.

## Features

* **Bearing characteristic frequency calculation** — FTF, BPFO, BPFI, BSF
* **Signal preprocessing** — zero-phase Butterworth bandpass filter, Hilbert-transform envelope
* **Feature extraction** — time-domain (RMS, kurtosis, crest factor, …) and frequency-domain (PSD, spectral centroid, envelope-spectrum energy)
* **Rule-based fault classification** — classifies bearing condition as *normal*, *outer-race fault*, *inner-race fault*, *ball fault*, or *cage fault*
* **Synthetic signal generator** — for testing and demonstration

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Example

```python
from bearing_diagnosis import BearingGeometry, diagnose
from generate_signal import generate_signal, FaultType

# 1. Define bearing geometry (e.g. 6205 deep-groove ball bearing)
geometry = BearingGeometry(
    n_balls=9,
    ball_diameter=7.938,   # mm
    pitch_diameter=38.5,   # mm
    contact_angle=0.0,     # degrees
)

shaft_hz = 25.0   # 1500 RPM
fs = 12_000.0     # sampling frequency [Hz]
duration = 2.0    # seconds

# 2. Generate a synthetic outer-race fault signal
import numpy as np
rng = np.random.default_rng(42)
x = generate_signal(
    geometry, shaft_hz, fs, duration,
    fault_type=FaultType.OUTER_RACE,
    fault_amplitude=5.0,
    noise_std=0.02,
    rng=rng,
)

# 3. Diagnose
result = diagnose(x, fs, geometry, shaft_hz)
print(result.fault_type)    # FaultType.OUTER_RACE
print(result.confidence)    # e.g. 0.87
print(result.scores)        # per-class normalised scores
```

## API

### `BearingGeometry`

| Parameter | Description |
|---|---|
| `n_balls` | Number of rolling elements |
| `ball_diameter` | Ball diameter [mm] |
| `pitch_diameter` | Pitch diameter [mm] |
| `contact_angle` | Contact angle [degrees] |

```python
freqs = geometry.characteristic_frequencies(shaft_hz)
# Returns dict with keys: ftf, bpfo, bpfi, bsf  [Hz]
```

### `diagnose(x, fs, geometry, shaft_hz, ...)`

Diagnoses bearing condition from a raw vibration signal.

Returns a `DiagnosisResult` with:
* `fault_type` — one of `FaultType.{NORMAL, OUTER_RACE, INNER_RACE, BALL, CAGE}`
* `confidence` — normalised confidence score (0–1)
* `scores` — per-class raw scores
* `features` — extracted `Features` dataclass

### `extract_features(x, fs, char_freqs=None)`

Extracts time-domain and frequency-domain features from a signal array.

### `generate_signal(geometry, shaft_hz, fs, duration, fault_type, ...)`

Generates a synthetic bearing vibration signal for any `FaultType`.

## Tests

```bash
pytest test_bearing_diagnosis.py -v
```
