# Validate against native energy and demand exports

`tools/validate_energy_exports.py` compares the actual `decode_file` output with
the native Parquet tables inside an OLE `.fca2` import. It checks all 180 energy
fields, all 20 demand fields, both timestamps, and the nominal period for every
row. It also requires the original FEL bytes to equal the raw FEL embedded in the
same container as the two tables.

This validates extraction against the supplied native tables. It does not
authenticate the container's author, prove instrument calibration, or establish
physical wiring. Column mappings must come from inspection of the native schema;
do not rearrange values to make a comparison pass.

## Run from the committed checkout

Install the optional validation dependencies:

```sh
python -m pip install -e '.[validation]'
python tools/validate_energy_exports.py \
  --manifest /private/validation/manifest.json \
  --output /private/validation/native-result.json
```

The checkout must be clean and the validator must be committed. The command
verifies that the running package, decoder, layout registry, version module, and
validator are the files in the cited Git commit. It checks provenance again after
the comparison. A dirty, stale, or untracked implementation cannot produce a
passing report. Python source uses the repository's LF line-ending policy;
existing CRLF checkouts may need their tracked files restored after that policy
is applied.

Keep the manifest, recordings, containers, and output outside the checkout. The
output is created exclusively: an existing file is never overwritten. Inputs are
opened for reading only. Exit code 0 means all comparisons passed; 1 indicates
validation failure; 2 indicates an output-path error.

## Private manifest

The manifest is explicit about every source column. The abbreviated example
below illustrates the structure; replace each `fields` object with **every**
named field for that group before running. Its paths and stream names are
illustrative, not automatic defaults.

```json
{
  "schema_version": 1,
  "evidence_kind": "native_vendor_export",
  "recordings": [{
    "fel": "recording.fel",
    "native_container": {
      "path": "native-import.fca2",
      "embedded_fel_stream": ["session#1", "session#1.fel.raw"]
    },
    "energy_trends": {
      "parquet_stream": ["session#1", "session#1_70.parquet"],
      "start_ticks": {"column": "timestamp_start", "encoding": "filetime_100ns"},
      "end_ticks": {"column": "timestamp_end", "encoding": "filetime_100ns"},
      "period": {"column": "trend_period", "encoding": "seconds_integer"},
      "fields": {"current_rms_c_max": "current_rms_c_max"}
    },
    "demand": {
      "parquet_stream": ["session#1", "session#1_71.parquet"],
      "start_ticks": {"column": "timestamp_start", "encoding": "filetime_100ns"},
      "end_ticks": {"column": "timestamp_end", "encoding": "filetime_100ns"},
      "period": {"column": "demand_period", "encoding": "seconds_integer"},
      "fields": {"current_rms_c": "current_rms_c"}
    }
  }]
}
```

Each `fields` key is a decoder field; its value is the independently inspected
vendor column name. When every vendor name exactly matches the decoder name, the
complete mapping can be assembled as follows, after verifying that fact:

```python
from fel_decoder.layouts import ENERGY_TREND_FIELDS, DEMAND_FIELDS

energy_mapping = {name: name for name in ENERGY_TREND_FIELDS}
demand_mapping = {name: name for name in DEMAND_FIELDS}
```

This builds a column mapping, not reference values. The reference values always
come from the native Parquet streams. Relative paths resolve beside the manifest.
Both groups are required for every recording. A missing mapping, repeated source
column, unexpected column, duplicate column name, or differing row count fails.
Additional non-measurement columns may be explicitly listed in `ignored_columns`;
mapped columns cannot be ignored.

## Exactness and timestamps

Comparisons use zero tolerance. Vendor float64 values are never rounded to
float32 before comparison. NaN must match NaN; a Parquet null is not silently
converted to NaN. Row order is retained, including ordered timestamp boundaries.
No sorting, clock alignment, unit conversion, or phase interpretation is applied.

Timestamp encodings must be declared:

| Encoding | Accepted reference column |
| --- | --- |
| `filetime_100ns` | Signed or unsigned integer ticks since 1601; nonnegative values only |
| `unix_integer` | Integer since 1970, with an explicit `unit`: `s`, `ms`, `us`, `ns`, or `100ns` |
| `arrow_timestamp_utc` | Arrow timestamp, explicitly interpreted as a UTC instant; timezone-naive input is asserted UTC by this choice |

All conversions use integer arithmetic. Nanosecond timestamps must be exact
multiples of 100 ns. Floating-point epoch timestamps, null timestamps, and
out-of-range values fail. Periods require `seconds_integer` and an integer column.

## Report and evidence boundaries

The JSON report contains the full tested commit, clean/dirty status, package
version, manifest schema, counts, maximum errors, missing-value comparison counts,
and anonymous decoder-field diagnostics. It contains no input paths, filenames,
instrument identifiers, source hashes, vendor column names, or actual readings.
The embedded-FEL identity result is recorded without publishing its hash.

A native result is marked `passed` only after a successful container/source
identity check and every exact comparison. A failure has a controlled reason code;
exception text from private files is not copied to the report. The operator's
native-export declaration is recorded, not treated as proof of vendor authorship.

Synthetic tests use `evidence_kind: "synthetic"` and produce the distinct status
`synthetic_passed`. In that mode, each group may use `parquet: "fixture.parquet"`
instead of a container stream. Those results test the validator and cannot serve
as native validation evidence. CI constructs independent wire fixtures and
reference tables; it includes failures for swapped columns, float64 differences
hidden by float32 rounding, one-tick errors, malformed mappings, missing/extra
rows, NaN mismatches, and unclean source provenance.

Run the native comparison again after the final code commit. Keep its sanitized
report as a release attachment or PR evidence, rather than committing a report
that purports to identify its own containing commit.
