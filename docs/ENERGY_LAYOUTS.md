# Named energy trends and demand

The container scanner and measurement decoders have separate responsibilities.
`scan_bytes` validates framing for every record in the observed header profile.
The registry in `layouts.py` selects a measurement layout by **profile, tag and
record length**, never by payload plausibility or length alone. Unknown tags stay
in the inventory. Unknown lengths of recognized tags fail decoding; `inspect`
still reports them as uninterpreted, including their sizes.

## Supported additions

| Tag | Bytes | Output group | Payload |
| --- | ---: | --- | --- |
| 70 | 744 | `energy_trends` | Nominal period and 180 named float32 fields |
| 71 | 104 | `demand` | Nominal period and 20 named float32 fields |

Both layouts contain high-word-first 100 ns timestamps at offsets 4 and 12, a
uint32 nominal interval in seconds at offset 20, and little-endian float32 fields
beginning at offset 24. The ordered field tuples in `layouts.py` specify every
float. Names and ordering were checked against native vendor Parquet tables.

For example, phase A/B/C RMS current maxima in tag 70 are at byte offsets
100/112/124; total active and apparent power maxima are at 256/304. These are
**total power channels**, not sums of independently timed phase maxima. Stored
power units are W and VA; divide by 1,000 for kW and kVA. Current is RMS amperes,
not an instantaneous waveform crest. Average and maximum channels have different
measurement windows. A trend interval is not the precise instant of its maximum.

## Access without wire-tag knowledge

```python
from fel_decoder import decode_file

recording = decode_file("recording.fel")
current_c = recording.measurement("current_rms_c_max")
total_kw = recording.measurement("power_rms_active_total_max") / 1000
intervals = recording.groups["energy_trends"]
start_ticks = intervals["start_ticks"]
end_ticks = intervals["end_ticks"]
```

This accessor returns the named field's float32 array in stored units and raises
`KeyError` when the field is unavailable. It does not invent semantic mappings for
the six unnamed tag-119 slots. There is no automatic maxima aggregation, NaN
filtering, phase summation, unit conversion, clock correction or timezone shift.

`energy_trends.npz` and `demand.npz` contain one numerical array per named field,
plus `record_offset`, `start_ticks`, `end_ticks`, and the nominal `trend_period` or
`demand_period`. `--csv` adds tables with those same fields and the global
`record_index`. Actual start/end boundaries take precedence over nominal periods;
boundary records may be partial. NaNs remain NaNs and integer ticks stay exact.

## Compatibility and limits

Existing `trends`, `events`, `waveforms` and `transients` arrays and CSV columns
remain unchanged. The new groups are present only when their records occur.
Mixed-layout files retain separate groups and original record order/offsets.
Manifest schema version 2 adds per-layout coverage and named-field units. A null
unit means the decoder does not assert a unit for that field; auxiliary units,
physical wiring, calibration and clock correctness require configuration evidence.

This is still a partial, empirically verified decoder. The accepted header
signature and record length cannot prove that all firmware variants share field
meanings. Only these registered layouts are supported. A new variant requires
native-export validation and independent synthetic fixtures before registration;
do not alias an unknown tag to an existing decoder based on matching length.

## Verification

Synthetic tests cover named fields at independently specified wire offsets,
signed power, NaN preservation, integer timestamp precision, partial/mixed records,
wrong sizes and reversed timestamps, unknown tags, and CSV/NPZ round trips.

Local vendor comparisons additionally checked every trend and demand cell in two
native imports: 10,040 trend rows with 183 columns, and 20,080 demand rows with 23
columns. The underlying FEL bytes were identical to each import's embedded source.
No timestamp or numerical differences were found in the independent extraction
used to establish the mapping. Six unique local files shared the 744-byte trend
layout. Only two had native exports; generalization to the other four is based on
that shared layout, not independent vendor confirmation for every file.

Client files, paths, instrument identifiers, hashes and extracted measurements are
not included in this repository. Synthetic CI tests do not replace validation
against new vendor exports when another profile is added.
