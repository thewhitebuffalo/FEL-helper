# Observed binary layout

These layouts were inferred from a limited collection of recordings and
cross-checked against available vendor text exports. They are not an official
format specification. All offsets below are decimal bytes. All tag numbers are
decimal integers.

## Container

The header is 32 bytes. The decoder requires the observed bytes at offsets 0–3
(`0bba1400`) and 24–31 (`0000090000000000`). These are profile signatures, not a
claim that either is a documented format version. The little-endian uint32 at
offset 4 must equal the number of bytes after the header. Header timestamps at
8 and 16 are retained in the header hex but are not used for clock correction.

Each record starts with `<HH`: uint16 tag, uint16 total length, including those
four bytes. The next record begins at the current offset plus its length.
Malformed lengths, incomplete headers, and truncated records are errors. Every
byte must belong to the header or an inventoried record. Unknown tags are
allowed; known tags with unrecognized lengths are rejected by decoding.

## Time fields

A timestamp consists of two little-endian uint32 words, **high word first**:

```python
high, low = struct.unpack_from("<II", data, offset)
ticks = (high << 32) | low
```

The epoch is 1601-01-01 and the unit is 100 ns. Subtract
`116444736000000000` to obtain ticks relative to the Unix epoch. Integer storage
avoids the precision loss of absolute floating-point epoch seconds.

Record start/end fields are at offsets 4 and 12 in all four supported tags.
Waveform/transient ends correspond to the last sample in the observed files,
not an exclusive end boundary. Event capture windows can be zero when absent.
The parser validates start ≤ end but does not sort, align, or correct clocks.

## Tag 105: waveform block

Supported length: 5,208 bytes.

| Offset | Storage | Interpretation |
| --- | --- | --- |
| 20 | 16 float32 | Eight `(scale, offset)` pairs |
| 84 | uint32 | Sample count, from 1 through 256 |
| 88 | Repeated 20-byte row | uint32 microsecond offset, eight signed int16 samples |

The calibrated value is `raw * scale + offset`, evaluated in float64.
NaN coefficients are retained. An infinite coefficient is rejected. Sample
offsets must not decrease inside a record. Any unused capacity after the declared
samples is not interpreted; it never becomes fabricated extra samples.

## Tag 112: transient block

Supported length: 32,064 bytes.

| Offset | Storage | Interpretation |
| --- | --- | --- |
| 20 | 8 float32 | Four `(scale, offset)` pairs |
| 52 | uint32 | Uninterpreted field |
| 56 | uint32 | Nonzero sample rate, Hz |
| 60 | uint32 | Sample count, from 1 through 4,000 |
| 64 | Repeated 8-byte row | Four signed int16 samples |

Local sample index `i` occurs `i / sample_rate_hz` seconds after start. Exact
ticks would be `start_ticks + i * 10000000 / sample_rate_hz`; this is not always
an integer for arbitrary rates, so exports retain the integer index and rate
rather than round it. The tested recordings use 1,000,000 Hz.

## Tag 111: event

Supported length: 104 bytes.

| Offset | Storage | Interpretation |
| --- | --- | --- |
| 20 | uint32 | Event ID (not assumed unique) |
| 24 | uint16 | Event type code |
| 26 | uint16 | Channel bitmask, exposed as `channel_code` |
| 28 | float32 | Event value |
| 32 | float32 | Event depth |
| 36 | uint32 | Severity code |
| 40, 48 | Timestamp | Waveform start/end |
| 56, 64 | Timestamp | RMS capture start/end |
| 72, 80 | Timestamp | MSV capture start/end |
| 88, 96 | Timestamp | Transient start/end |

Observed vendor labels: 0 DIP, 2 INTERRUPTION, 4 RVC, 6 WAVESHAPE,
7 RVC_CANCELED, 8 TRANSIENT. Observed severities: 0 NOT_SPECIFIED, 2 MEDIUM,
3 LOW. The exports preserve numeric codes to avoid implying an exhaustive enum.
Units and channel meaning depend on configuration. The MSV abbreviation is
retained from the vendor export; its measurement meaning is not inferred.

## Tag 119: selected trend statistics

Supported length: 1,724 bytes. The uint32 at offset 20 is not interpreted. The
first 18 float32 values at offset 24 are exported as six numbered slots × three
statistics in stored order `(min, max, average)`. This order is supported by
internal consistency checks in the test corpus; vendor-value validation is
still needed for these fields. The rest of the record is **not decoded**.

## Array schema

Every group retains `record_offset`, `start_ticks`, and `end_ticks`, all uint64.
Waveform/transient groups retain per-record `sample_start` (uint64),
`sample_count` (uint32), and `coefficients` (float32, record × channel × 2).
Concatenated sample arrays are `raw` (int16) and `values` (float64). Use each
record's start/count to select its samples without bridging gaps.

Waveforms additionally store sample-level `offset_us` (uint32); transients store
record-level `sample_rate_hz` (uint32). Trends use `values` shaped
`(records, 6, 3)` and float32. Empty groups retain these shapes, for example
`(0, 4)` for absent transient samples. No arrays contain pickled objects.
