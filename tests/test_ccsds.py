import numpy as np
import pytest

from rpo import ccsds


def test_primary_header_round_trip():
    h = ccsds.PrimaryHeader(
        apid=0x1A2,
        sequence_count=12345,
        data_length=199,
        packet_type=ccsds.PacketType.TM,
        secondary_header_flag=True,
    )
    raw = h.to_bytes()
    assert len(raw) == 6
    h2 = ccsds.PrimaryHeader.from_bytes(raw)
    assert h2.apid == h.apid
    assert h2.sequence_count == h.sequence_count
    assert h2.data_length == h.data_length
    assert h2.packet_type == h.packet_type
    assert h2.secondary_header_flag == h.secondary_header_flag


def test_primary_header_field_widths():
    """APID, sequence count, version, type, sec-hdr flag must all live in
    the right bit positions per CCSDS 133.0-B-2."""
    h = ccsds.PrimaryHeader(apid=0x7FF, sequence_count=0x3FFF, data_length=0)
    raw = h.to_bytes()
    word0 = (raw[0] << 8) | raw[1]
    word1 = (raw[2] << 8) | raw[3]
    assert word0 & 0x07FF == 0x7FF                # APID lives in low 11 bits of word0
    assert (word1 >> 14) & 0b11 == 0b11           # default unsegmented
    assert word1 & 0x3FFF == 0x3FFF


def test_apid_overflow_rejected():
    with pytest.raises(ValueError):
        ccsds.PrimaryHeader(apid=2048, sequence_count=0, data_length=0).to_bytes()


def test_cds_time_round_trip():
    t = 1234567.89
    cds = ccsds.CdsTimeCode.from_seconds(t)
    raw = cds.to_bytes()
    assert len(raw) == 6
    cds2 = ccsds.CdsTimeCode.from_bytes(raw)
    assert abs(cds2.seconds - t) < 1e-3


def test_space_packet_round_trip():
    payload = ccsds.encode_state(
        np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]), dv_total=0.5)
    pkt = ccsds.SpacePacket(
        apid=int(ccsds.APID.CHASER_TRUTH),
        sequence_count=7,
        payload=payload,
        timestamp=ccsds.CdsTimeCode.from_seconds(123.45),
    )
    raw = pkt.to_bytes()
    pkt2 = ccsds.SpacePacket.from_bytes(raw)
    assert pkt2.apid == pkt.apid
    assert pkt2.sequence_count == 7
    assert abs(pkt2.timestamp.seconds - 123.45) < 1e-3
    state, dv = ccsds.decode_state(pkt2.payload)[:6], ccsds.decode_state(pkt2.payload)[6]
    assert np.allclose(state, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    assert dv == pytest.approx(0.5)


def test_telemetry_stream_round_trip(tmp_path):
    stream = ccsds.TelemetryStream()
    for k in range(5):
        stream.emit(ccsds.APID.CHASER_TRUTH,
                    ccsds.encode_state(np.zeros(6), 0.0), float(k))
    stream.emit(ccsds.APID.EVENT, ccsds.encode_event("HELLO"), 5.0)

    path = tmp_path / "tlm.bin"
    stream.write(str(path))
    pkts = ccsds.TelemetryStream.read(str(path))
    assert len(pkts) == 6
    seqs = [p.sequence_count for p in pkts if p.apid == int(ccsds.APID.CHASER_TRUTH)]
    assert seqs == list(range(5))
    event = next(p for p in pkts if p.apid == int(ccsds.APID.EVENT))
    assert ccsds.decode_event(event.payload) == "HELLO"
