from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import ClassVar


CCSDS_EPOCH_DAYS_FROM_1958 = 0


class PacketType(IntEnum):
    TM = 0
    TC = 1


class SeqFlags(IntEnum):
    CONTINUATION = 0b00
    FIRST        = 0b01
    LAST         = 0b10
    UNSEGMENTED  = 0b11


@dataclass
class PrimaryHeader:
    apid: int
    sequence_count: int
    data_length: int
    packet_type: PacketType = PacketType.TM
    secondary_header_flag: bool = True
    sequence_flags: SeqFlags = SeqFlags.UNSEGMENTED
    version: int = 0

    SIZE: ClassVar[int] = 6

    def to_bytes(self):
        if not 0 <= self.apid < (1 << 11):
            raise ValueError("APID out of range (0..2047)")
        if not 0 <= self.sequence_count < (1 << 14):
            raise ValueError("sequence count out of range (0..16383)")
        if not 0 <= self.data_length < (1 << 16):
            raise ValueError("data length out of range")

        w0 = ((self.version & 0b111) << 13
              | (int(self.packet_type) & 1) << 12
              | (1 if self.secondary_header_flag else 0) << 11
              | (self.apid & 0x7FF))
        w1 = (int(self.sequence_flags) & 0b11) << 14 | (self.sequence_count & 0x3FFF)
        return struct.pack(">HHH", w0, w1, self.data_length)

    @classmethod
    def from_bytes(cls, buf):
        if len(buf) < cls.SIZE:
            raise ValueError("primary header truncated")
        w0, w1, dl = struct.unpack(">HHH", buf[:cls.SIZE])
        return cls(
            version=(w0 >> 13) & 0b111,
            packet_type=PacketType((w0 >> 12) & 1),
            secondary_header_flag=bool((w0 >> 11) & 1),
            apid=w0 & 0x7FF,
            sequence_flags=SeqFlags((w1 >> 14) & 0b11),
            sequence_count=w1 & 0x3FFF,
            data_length=dl,
        )


@dataclass
class CdsTimeCode:
    days: int
    ms_of_day: int

    SIZE: ClassVar[int] = 6
    _MS_PER_DAY: ClassVar[int] = 86400 * 1000

    @classmethod
    def from_seconds(cls, t):
        if t < 0:
            raise ValueError("negative mission time")
        total_ms = int(round(t * 1000.0))
        days, ms = divmod(total_ms, cls._MS_PER_DAY)
        return cls(days=days + CCSDS_EPOCH_DAYS_FROM_1958, ms_of_day=ms)

    def to_bytes(self):
        return struct.pack(">HI", self.days & 0xFFFF, self.ms_of_day & 0xFFFFFFFF)

    @classmethod
    def from_bytes(cls, buf):
        d, ms = struct.unpack(">HI", buf[:cls.SIZE])
        return cls(days=d, ms_of_day=ms)

    @property
    def seconds(self):
        return (self.days - CCSDS_EPOCH_DAYS_FROM_1958) * 86400.0 + self.ms_of_day / 1000.0


@dataclass
class SpacePacket:
    apid: int
    sequence_count: int
    payload: bytes = b""
    timestamp: CdsTimeCode | None = None
    packet_type: PacketType = PacketType.TM

    def to_bytes(self):
        sec = self.timestamp.to_bytes() if self.timestamp is not None else b""
        data = sec + self.payload
        if not data:
            raise ValueError("CCSDS data field must be at minimum 1 octet")

        ph = PrimaryHeader(
            apid=self.apid,
            sequence_count=self.sequence_count,
            data_length=len(data) - 1,
            packet_type=self.packet_type,
            secondary_header_flag=self.timestamp is not None,
        )
        return ph.to_bytes() + data

    @classmethod
    def from_bytes(cls, buf):
        ph = PrimaryHeader.from_bytes(buf)
        end = PrimaryHeader.SIZE + ph.data_length + 1
        data = buf[PrimaryHeader.SIZE:end]

        ts = None
        if ph.secondary_header_flag:
            ts = CdsTimeCode.from_bytes(data[:CdsTimeCode.SIZE])
            payload = data[CdsTimeCode.SIZE:]
        else:
            payload = data
        return cls(apid=ph.apid, sequence_count=ph.sequence_count,
                   payload=payload, timestamp=ts, packet_type=ph.packet_type)


class APID(IntEnum):
    CHASER_TRUTH = 0x100
    CHASER_NAV   = 0x101
    SENSOR_MEAS  = 0x102
    GUIDANCE_CMD = 0x103
    CAMPAIGN_SUMMARY = 0x200
    EVENT        = 0x1FF


def encode_state(state, dv_total):
    return struct.pack(">7d", *state, dv_total)

def decode_state(payload):
    return struct.unpack(">7d", payload)


def encode_nav(state, cov_trace_pos, cov_trace_vel):
    return struct.pack(">8d", *state, cov_trace_pos, cov_trace_vel)

def decode_nav(payload):
    return struct.unpack(">8d", payload)


def encode_meas(z):
    return struct.pack(">3d", *z)

def decode_meas(payload):
    return struct.unpack(">3d", payload)


def encode_guidance(dv):
    return struct.pack(">3d", *dv)

def decode_guidance(payload):
    return struct.unpack(">3d", payload)


def encode_campaign(n_trials, capture_rate, dv_mean, dv_p95, dv_p99,
                    anees_mean):
    return struct.pack(">I5d", int(n_trials), capture_rate,
                       dv_mean, dv_p95, dv_p99, anees_mean)

def decode_campaign(payload):
    n_trials, capture_rate, dv_mean, dv_p95, dv_p99, anees_mean = \
        struct.unpack(">I5d", payload)
    return {
        "n_trials": n_trials,
        "capture_rate": capture_rate,
        "dv_mean": dv_mean,
        "dv_p95": dv_p95,
        "dv_p99": dv_p99,
        "anees_mean": anees_mean,
    }


def encode_event(text):
    data = text.encode("ascii", errors="replace")[:240]
    return struct.pack(">B", len(data)) + data

def decode_event(payload):
    n = payload[0]
    return payload[1:1 + n].decode("ascii", errors="replace")


@dataclass
class TelemetryStream:
    packets: list[bytes] = field(default_factory=list)
    _seq_counters: dict = field(default_factory=dict)

    def emit(self, apid, payload, t):
        seq = self._seq_counters.get(int(apid), 0)
        self._seq_counters[int(apid)] = (seq + 1) & 0x3FFF
        pkt = SpacePacket(
            apid=int(apid),
            sequence_count=seq,
            payload=payload,
            timestamp=CdsTimeCode.from_seconds(t),
        ).to_bytes()
        self.packets.append(pkt)
        return pkt

    def write(self, path):
        with open(path, "wb") as f:
            for p in self.packets:
                f.write(p)

    @staticmethod
    def read(path):
        with open(path, "rb") as f:
            buf = f.read()
        out: list[SpacePacket] = []
        i = 0
        while i < len(buf):
            ph = PrimaryHeader.from_bytes(buf[i:i + 6])
            length = 6 + ph.data_length + 1
            out.append(SpacePacket.from_bytes(buf[i:i + length]))
            i += length
        return out
