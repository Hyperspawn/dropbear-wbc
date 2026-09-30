"""Unitree-compatible CRC32 over 32-bit words.

Unitree's SDK (``unitree_sdk2py/utils/crc.py``, ``crc32_core``) computes the CRC
of a message by packing the C struct, reading it as little-endian ``uint32``
words (excluding the trailing ``crc`` word) and running a bitwise MSB-first
CRC with polynomial ``0x04C11DB7``, initial value ``0xFFFFFFFF``, no reflection
and no final XOR over the 32 bits of each word.

That bitwise loop is algebraically the standard CRC-32/MPEG-2 over the bytes of
each word in big-endian order, so this module uses a byte table for speed.
``crc32_core_bitwise`` keeps the literal bit loop as a reference for tests.
"""
from __future__ import annotations

import numpy as np

POLYNOMIAL = 0x04C11DB7
_MASK = 0xFFFFFFFF


def _make_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ POLYNOMIAL) if crc & 0x80000000 else (crc << 1)
            crc &= _MASK
        table.append(crc)
    return table


_TABLE = _make_table()


def crc32_core(words: np.ndarray | list[int]) -> int:
    """CRC32 of a sequence of uint32 words, bit-identical to Unitree's ``crc32_core``.

    Args:
        words: 1-D sequence of unsigned 32-bit integers.

    Returns:
        The CRC as a Python int in ``[0, 2**32)``.
    """
    data = np.asarray(words, dtype=np.uint32).astype(">u4").tobytes()
    crc = _MASK
    table = _TABLE
    for byte in data:
        crc = ((crc << 8) & _MASK) ^ table[((crc >> 24) ^ byte) & 0xFF]
    return crc


def crc32_core_bitwise(words: np.ndarray | list[int]) -> int:
    """Literal bit-by-bit transcription of Unitree's algorithm (slow; reference only)."""
    crc = _MASK
    for current in np.asarray(words, dtype=np.uint32).tolist():
        bit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & _MASK) ^ POLYNOMIAL
            else:
                crc = (crc << 1) & _MASK
            if current & bit:
                crc ^= POLYNOMIAL
            bit >>= 1
    return crc


def crc32_bytes(payload: bytes) -> int:
    """CRC32 of ``payload`` read as little-endian uint32 words (zero-padded to 4 bytes).

    This mirrors Unitree's ``__Trans``: the packed struct bytes are grouped into
    little-endian words before ``crc32_core``.
    """
    pad = (-len(payload)) % 4
    if pad:
        payload = payload + b"\x00" * pad
    return crc32_core(np.frombuffer(payload, dtype="<u4"))
