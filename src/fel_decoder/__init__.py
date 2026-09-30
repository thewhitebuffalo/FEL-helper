"""Decode the observed FEL record profile without assuming wiring or clock accuracy."""

from .core import DecodedFile, FelError, Record, decode_bytes, decode_file, scan_bytes

__version__ = "0.1.0"
__all__ = ["DecodedFile", "FelError", "Record", "decode_bytes", "decode_file", "scan_bytes"]
