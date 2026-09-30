# FEL-helper

Fast extraction of selected measurement records from the observed Fluke `.fel`
binary layout. Converts event records, sampled waveforms, high-speed transients,
and selected trend fields into NumPy arrays and CSV files.

This is an independently developed, **partial decoder**, not a Fluke product or
a complete implementation of the FEL format. It accepts the layout documented
in [FORMAT.md](docs/FORMAT.md). Other layouts are rejected rather than guessed.

## Install

Python 3.10 or newer and NumPy are required. From a downloaded checkout:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
fel-helper --help
```

To get the source with Git first:

```sh
git clone https://github.com/thewhitebuffalo/FEL-helper.git
cd FEL-helper
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.

## Use

```sh
# Inspect record counts without interpreting measurement payloads.
fel-helper inspect recording.fel --json

# Fast default: uncompressed NPZ arrays, events CSV, record inventory, manifest.
fel-helper decode recording.fel --output decoded-recording

# Also export the large waveform, transient, and trend tables as CSV.
fel-helper decode recording.fel --output decoded-csv --csv

# Trade additional CPU time for smaller NPZ files.
fel-helper decode recording.fel --output decoded-compressed --compress
```

Use a **new output directory** for each run. Existing directories are refused;
input recordings are never modified. A successful export ends with
`manifest.json`. If an export fails, an incomplete output directory may remain
without that completion marker. Remove it or choose a new destination before
retrying. CSV output is optional because large text tables are slower and larger
than binary arrays.

```python
from fel_decoder import decode_file

recording = decode_file("recording.fel")
waveforms = recording.groups["waveforms"]

# First waveform record, preserving its capture boundary.
begin = int(waveforms["sample_start"][0])
count = int(waveforms["sample_count"][0])
samples = waveforms["values"][begin:begin + count]
offset_us = waveforms["offset_us"][begin:begin + count]
start_ticks = int(waveforms["start_ticks"][0])

# Exact sample timestamps: avoid floating-point Unix epoch seconds.
sample_ticks = [start_ticks + int(us) * 10 for us in offset_us]
```

The example requires at least one waveform record. Files without that record
type have an empty waveform group. NPZ files contain ordinary numerical arrays;
load with `numpy.load(path, allow_pickle=False)`.

## What is retained

| Group | Extracted fields |
| --- | --- |
| Events (111) | Start/end ticks, ID, type code, channel mask, value, depth, severity code, capture-window timestamps |
| Waveforms (105) | Eight numbered slots, original signed samples, per-record scale/offset coefficients, calibrated values, microsecond offsets |
| Transients (112) | Four numbered slots, original signed samples, coefficients, calibrated values, sample rate |
| Trends (119) | First six numbered slots, each with three statistics in stored order: minimum, maximum, average |
| Energy trends (70) | Nominal period and 180 named fields, including RMS current and total active/apparent power min/max/average |
| Demand (71) | Nominal period and 20 named fields from the verified vendor schema |

See [named energy layouts](docs/ENERGY_LAYOUTS.md) for tag-independent measurement
access, stored units, export compatibility and the native-validation scope.

All record offsets, tags, and sizes appear in the inventory. Unknown tags are
counted explicitly, but their payloads are not decoded. Even supported tags
contain fields whose meaning remains unknown. Keep the original recording;
these exports are not a replacement for it.

## Interpretation limits

- Numbered channel slots have **no automatic L–L, L–G, current, phase-angle, or unit
  assignment**. The new energy trend/demand fields use verified vendor names;
  physical wiring and instrument configuration still require verification.
- Waveform and transient samples are instantaneous quantities. This tool does
  not calculate RMS current, identify breaker operation, or infer trip causes.
- Timestamps preserve the instrument clock. They do not establish correct wall
  time or synchronization between meters. No timezone shift or alignment is
  silently applied.
- Record boundaries are retained. Consecutive array rows from different
  records may be separated by a long gap; do not treat the entire array as a
  continuous waveform.
- Missing coefficients yield `NaN`, not zero. Raw values remain available.
- Event channel codes are bitmasks, not phase ordinals. Event records retain
  file order, which is not necessarily chronological.
- The decoder reads the file into memory and uses additional memory for arrays.
  It is vectorized for speed, but it is not a bounded-memory streaming decoder.
- File hashes provide provenance, not authenticity or verification of an
  undocumented internal checksum. Plausible bit corruption can escape framing
  checks; retain and compare authoritative exports where available.

## Test

```sh
python -m pip install ".[test]"
python -m pytest -q
```

Tests generate synthetic bytes in memory. No client recordings or extracted
client data are included. See [VALIDATION.md](docs/VALIDATION.md) for the tested
scope and for independent validation against locally held vendor exports.

The source can be run before installation using
`PYTHONPATH=src python -m fel_decoder --help` on macOS/Linux.
