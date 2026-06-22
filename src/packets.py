import struct
import time
from typing import Dict, Optional, Tuple

import numpy as np

from .config import MAGIC, MAGIC_LE, REASSEMBLY_TTL_S
from .models import CamFrame, IRFrame


class JpegReassembler:
    def __init__(self, ttl: float = REASSEMBLY_TTL_S):
        self._ttl = ttl
        self._seqs: Dict[int, dict] = {}

    def feed(
        self, seq: int, timestamp: int, total_len: int, chunk: bytes
    ) -> Optional[CamFrame]:
        if len(chunk) == total_len:
            return CamFrame(seq=seq, timestamp=timestamp, jpeg=bytes(chunk))

        now = time.monotonic()
        stale = [s for s, v in self._seqs.items() if now - v["t"] > self._ttl]
        for s in stale:
            del self._seqs[s]

        if seq not in self._seqs:
            self._seqs[seq] = {
                "buf": bytearray(total_len),
                "written": 0,
                "total": total_len,
                "ts": timestamp,
                "t": now,
            }

        entry = self._seqs[seq]
        offset = entry["written"]
        end = offset + len(chunk)

        if end > total_len:
            del self._seqs[seq]
            return None

        entry["buf"][offset:end] = chunk
        entry["written"] = end

        if end >= total_len:
            frame = CamFrame(seq=seq, timestamp=entry["ts"], jpeg=bytes(entry["buf"]))
            del self._seqs[seq]
            return frame

        return None


# ─────────────────────────────────────────────
#  PACKET PARSERS
# ─────────────────────────────────────────────
def parse_ir(data: bytes) -> Optional[IRFrame]:
    if len(data) < 145:
        return None
    if struct.unpack_from(">I", data, 0)[0] != MAGIC or data[4] != 0x01:
        return None
    seq = struct.unpack_from(">I", data, 5)[0]
    ts = struct.unpack_from(">Q", data, 9)[0]
    raw = struct.unpack_from(">64h", data, 17)
    temps = np.array(raw, dtype=np.float32).reshape(8, 8) / 100.0
    return IRFrame(seq=seq, timestamp=ts, temps=temps)


def describe_cam_packet(data: bytes) -> str:
    if len(data) >= 2 and data[:2] == b"\xff\xd8":
        return f"raw JPEG packet size={len(data)}"
    if len(data) < 21:
        return f"short packet size={len(data)}"

    magic = struct.unpack_from(">I", data, 0)[0]
    packet_type = data[4]
    jpeg_len = struct.unpack_from(">I", data, 17)[0]
    payload_len = len(data) - 21
    first8 = data[:8].hex(" ")

    if magic != MAGIC:
        if magic == MAGIC_LE:
            return f"little-endian ES32 magic first8={first8}"
        return f"bad magic=0x{magic:08x} first8={first8}"
    if packet_type != 0x02:
        return f"bad type=0x{packet_type:02x} first8={first8}"
    if jpeg_len != payload_len:
        return (
            f"jpeg_len mismatch header={jpeg_len} payload={payload_len} first8={first8}"
        )
    if payload_len >= 2 and data[21:23] != b"\xff\xd8":
        return (
            f"payload does not start with JPEG SOI first_payload={data[21:29].hex(' ')}"
        )
    return "ok"


def parse_cam_hdr(data: bytes) -> Optional[Tuple[int, int, int, bytes]]:
    if len(data) >= 2 and data[:2] == b"\xff\xd8":
        ts = time.monotonic_ns() // 1000
        seq = ts & 0xFFFFFFFF
        return seq, ts, len(data), data

    if len(data) < 21:
        return None

    magic = struct.unpack_from(">I", data, 0)[0]
    if magic == MAGIC:
        seq = struct.unpack_from(">I", data, 5)[0]
        ts = struct.unpack_from(">Q", data, 9)[0]
        tot_len = struct.unpack_from(">I", data, 17)[0]
    elif magic == MAGIC_LE:
        seq = struct.unpack_from("<I", data, 5)[0]
        ts = struct.unpack_from("<Q", data, 9)[0]
        tot_len = struct.unpack_from("<I", data, 17)[0]
    else:
        return None

    if data[4] != 0x02:
        return None

    return seq, ts, tot_len, data[21:]
