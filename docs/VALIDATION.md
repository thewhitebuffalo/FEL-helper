# Validation history and release scope

## Version 0.2.0 release candidate

Version 0.2.0 adds named tag-70/71 layouts. Their regression tests and independent
native-export comparison scope are documented in
[ENERGY_LAYOUTS.md](ENERGY_LAYOUTS.md). Independent comparison of the new layouts
with native exports is pending; the historical results below are not a PASS for
those new extraction paths or a completed 0.2.0 release validation.

## Historical validation of version 0.1.0

The checks and measurements below were completed for version 0.1.0 and its four
original extraction paths. They do not establish support for every FEL variant
or every field. Client recordings, vendor exports, filenames, source hashes, and
detailed local results are not included in the repository.

## Automated checks

57 tests, including 135 unittest subtests, pass locally on macOS ARM64:

- Python 3.10 with NumPy 1.24.4 (minimum supported NumPy series).
- Python 3.12 with NumPy 2.5.3.
- Python 3.14 with NumPy 2.4.6.

Coverage includes all four extracted record types; high-word-first unsigned
timestamps; sub-microsecond precision; signed sample extrema; nonzero and
negative calibration coefficients; missing values; per-record scaling and sample
rates; long capture gaps; duplicate event IDs; partial-capacity blocks; unknown
records; truncated inputs; invalid lengths/counts/rates; backward sample offsets;
reversed record times; malformed headers; exact NPZ round trips; CSV integer
precision; output overwrite refusal; and interrupted export handling.

The fixtures are generated from synthetic numbers with a separate `struct`
writer. An independent scalar arithmetic oracle checks randomized scaling cases.
The [initial GitHub Actions run](https://github.com/thewhitebuffalo/FEL-helper/actions/runs/36654771150)
also passed all six hosted jobs: Linux with Python 3.10, 3.12, 3.13 and 3.14,
plus macOS and Windows with Python 3.12. Each job installed the package, ran the
tests, and checked the command-line entry point. These checks establish
compatibility for the tested environments and synthetic fixtures; the private
recording comparisons described below were performed locally on macOS.

## Checks against locally held recordings

All eight available recordings passed complete record framing and independent
scalar extraction comparisons:

| Check | Quantity |
| --- | ---: |
| Input bytes | 285,374,692 |
| Inventoried records | 86,460 |
| Event records | 223 |
| Waveform sample rows | 367,104 |
| Waveform scalar values | 2,936,832 |
| Transient sample rows | 1,500,000 |
| Transient scalar values | 6,000,000 |
| Trend records | 56,353 |
| Extracted trend triples | 338,118 |

Every extracted raw sample, calibrated value, selected trend triple, event
field, timestamp, and record boundary was compared with a separate scalar
reader. This checks implementation consistency with the inferred layout;
independent vendor exports provide the stronger measurement checks below.

## Comparisons with vendor exports

| Comparison | Result |
| --- | --- |
| 188 events across seven recordings | All exported IDs, type labels, channel masks and severity labels match |
| 1,190 nonempty event/window timestamps | Exact equality in integer 100 ns ticks; absent windows also checked |
| Event values | Maximum absolute difference 0.0004931641 in exported units |
| Event depths | Maximum absolute difference 0.0004887695313 in exported units |
| 500,000 transient sample instants | Exact timestamp and capture-boundary equality |
| 1,500,000 transient scalar voltage values (three slots) | Maximum absolute difference 0.0005138148367 V |

The comparison tolerance is 0.0006, sufficient for decimal rounding in these
vendor exports. It is a software comparison tolerance, **not an instrument
accuracy specification**. Event values and depths do not share a universal unit.

Ordinary waveform values (tag 105), selected trend values (tag 119), and the
fourth transient slot do not have an independent vendor-value comparison in the
available exports. Their extraction and arithmetic checks pass, but their
electrical meaning and full vendor agreement remain unverified.

## Performance

On an Apple M3 Max, repeated read + decode + SHA-256 runs took a summed per-file
median of **0.279 seconds** for the eight recordings (285.4 MB), approximately
1.02 GB/s. Each file was measured three times. These are warm-cache measurements;
they exclude interpreter startup, output writing, and vendor comparison work.
They are not a cold-disk or universal performance claim.

End-to-end export checks also passed:

| Export path | Measured elapsed time | Output size |
| --- | ---: | ---: |
| All eight files, default NPZ + event/inventory CSV | 0.386 seconds total | 98.63 MB |
| All eight files, compressed NPZ + event/inventory CSV | 2.75 seconds total | 27.29 MB |
| One complete recording, including all optional CSV tables | 3.89 seconds | 124.62 MB |

These runs include source reading, decoding, hashing and output writing, with
warm source reads; verification work is excluded from those timings. In each
NPZ mode, all 312 saved arrays were reloaded without pickle and compared for
exact dtype and NaN-aware value equality. The complete CSV run was streamed back
and checked against the decoded arrays: 52,736 waveform rows, 500,000 transient
rows, 9,270 trend rows, 16,846 inventory rows, and 35 events. Temporary exports
were removed after validation.

## Reproduce with private files

Keep the manifest and result outside the repository. Paths can be absolute or
relative to the manifest:

```json
{
  "recordings": [
    {
      "fel": "recording.fel",
      "events_csv": "events.txt",
      "transient_csv": "transients.txt",
      "timezone": "America/Los_Angeles"
    }
  ]
}
```

Omit either CSV field when that export is unavailable. Specify the timezone
explicitly when comparing local-time vendor exports; their header label alone
is not sufficient to resolve daylight-saving time.

```sh
python tools/validate_local.py --manifest /private/path/manifest.json \
  --output /private/path/results.json
```

Add `--export-output /private/path/export-results.json` to run the binary and
CSV export round-trip checks too. Those tests create temporary output directories
outside the repository and remove them afterward.

The harness exits unsuccessfully on any failed check. It checks every scalar
sample, not just selected points. Its vendor comparison currently expects the
observed semicolon-delimited export columns and event enums, unique event IDs
for reference matching, and full coverage of the selected export. Its scalar
corpus check expects fully occupied waveform/transient blocks; the decoder
itself also accepts smaller valid declared counts, covered by synthetic tests.

Before expanding support, add independent exports from other instrument
configurations and firmware versions. In particular, obtain waveform and trend
exports to validate those mappings directly. Do not silently broaden accepted
record layouts or infer L–L/L–G labels from channel positions.
