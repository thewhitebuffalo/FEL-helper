"""Command-line interface for FEL-helper."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

from . import __version__
from .core import FelError, KNOWN_SIZES, decode_file, scan_bytes
from .export import export_decoded
from .layouts import layout_inventory, resolve_layout


def _parser():
    parser = argparse.ArgumentParser(
        prog="fel-helper",
        description="Decode the observed FEL profile into arrays without assuming channel meanings.",
    )
    parser.add_argument("--version", action="version", version=f"FEL-helper {__version__}")
    commands = parser.add_subparsers(dest="command")
    inspect = commands.add_parser("inspect", help="Validate framing and inventory records without decoding samples")
    inspect.add_argument("input", type=Path)
    inspect.add_argument("--json", action="store_true", help="Print machine-readable inventory")
    decode = commands.add_parser("decode", help="Decode supported records into a new directory")
    decode.add_argument("input", type=Path)
    decode.add_argument("--output", type=Path, required=True, metavar="DIR", help="New directory; existing paths are refused")
    decode.add_argument("--csv", action="store_true", help="Also export waveform, transient and trend CSVs (slower and larger)")
    decode.add_argument("--compress", action="store_true", help="Compress NPZ files; slower than the default uncompressed export")
    return parser


def _inspect(path, as_json):
    data = path.read_bytes()
    records = scan_bytes(data)
    counts = Counter(record.tag for record in records)
    inventory = {
        "source_filename": path.name,
        "source_bytes": len(data),
        "record_count": len(records),
        "record_counts": {str(tag): count for tag, count in sorted(counts.items())},
        "uninterpreted_record_counts": {
            str(tag): count for tag, count in sorted(Counter(
                r.tag for r in records if resolve_layout(r.tag, r.size) is None
            ).items())
        },
        "record_layouts": layout_inventory(records),
    }
    if as_json:
        print(json.dumps(inventory, indent=2))
    else:
        print(f"{path.name}: {len(data):,} bytes, {len(records):,} records")
        for item in inventory["record_layouts"]:
            status = "selected fields supported" if item["status"] == "supported" else "uninterpreted"
            print(f"  tag {item['tag']}: {item['count']:,} ({status}); size {item['size']}")
        print("Inventory validates framing only; decoding also checks supported record layouts.")


def main(argv=None):
    """Run the CLI, returning a process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "inspect":
            _inspect(args.input, args.json)
        else:
            started = time.perf_counter()
            decoded = decode_file(args.input)
            output = export_decoded(
                decoded, args.output, source_name=args.input.name,
                csv_samples=args.csv, compress=args.compress,
            )
            elapsed = time.perf_counter() - started
            print(f"Processed {len(decoded.records):,} records; exported selected fields in {elapsed:.3f} s. Export: {output}")
            print("Partial decoder; see manifest.json for uninterpreted record counts.")
    except (FelError, OSError) as error:
        print(f"fel-helper: {error}", file=sys.stderr)
        return 1
    return 0
