"""
radar_capture.py
----------------
Reads raw point-cloud frames from the IWR6843AOP FMCW radar
over serial USB on the Raspberry Pi.

The IWR6843 outputs a binary TLV (Type-Length-Value) stream.
This parser handles the standard mmWave SDK output format.
"""

import serial
import struct
import time
import numpy as np
from config import SERIAL_PORT_DATA, SERIAL_PORT_CFG, SERIAL_BAUD

# ── Magic word that starts every IWR6843 frame ──────────────────────────────
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'

# ── TLV type IDs ─────────────────────────────────────────────────────────────
TLV_DETECTED_POINTS   = 1
TLV_RANGE_PROFILE     = 2
TLV_POINT_CLOUD_2D    = 6   # some firmware versions use this
TLV_POINT_CLOUD_EXT   = 7

# ── Default config script for IWR6843AOP ─────────────────────────────────────
# Adjust if your sensor uses a different profile
DEFAULT_CFG = """\
sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 7 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 60 167 7 57.14 0 0 70 1 128 2500 0 0 30
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 4
chirpCfg 2 2 0 0 0 0 0 4
frameCfg 0 2 16 0 55 1 0
lowPower 0 0
guiMonitor -1 1 0 0 0 0 1
cfarCfg -1 0 2 8 4 3 0 15 1
cfarCfg -1 1 0 4 2 3 1 15 1
multiObjBeamForming -1 1 0.5
calibDcRangeSig -1 0 -5 8 256
clutterRemoval -1 0
compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
measureRangeBiasAndRxChanPhase 0 1.5 0.2
extendedMaxVelocity -1 0
CQRxSatMonitor 0 3 5 121 0
CQSigImgMonitor 0 127 4
analogMonitor 0 0
lvdsStreamCfg -1 0 0 0
sensorStart
"""


def send_config(cfg_port_name: str, config_str: str = DEFAULT_CFG):
    """
    Send configuration commands to the IWR6843 config port.
    Call once before starting to read data.
    """
    try:
        cfg_port = serial.Serial(cfg_port_name, 115200, timeout=1)
        time.sleep(0.5)
        for line in config_str.strip().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            cfg_port.write((line + '\n').encode())
            time.sleep(0.05)
            response = cfg_port.read(cfg_port.in_waiting or 1)
        cfg_port.close()
        print(f"[radar] Config sent to {cfg_port_name}")
    except serial.SerialException as e:
        print(f"[radar] WARNING: Could not open config port {cfg_port_name}: {e}")
        print("[radar] Continuing — sensor may already be configured.")


def _parse_tlv(data: bytes, offset: int, num_tlvs: int):
    """
    Parse TLV blocks from the raw byte stream.
    Returns a list of detected point dicts: {x, y, z, doppler, snr}
    """
    points = []
    for _ in range(num_tlvs):
        if offset + 8 > len(data):
            break
        tlv_type   = struct.unpack_from('<I', data, offset)[0]; offset += 4
        tlv_length = struct.unpack_from('<I', data, offset)[0]; offset += 4

        if tlv_type == TLV_DETECTED_POINTS:
            # Each detected point: x(f), y(f), z(f), doppler(f) = 16 bytes
            n = tlv_length // 16
            for i in range(n):
                base = offset + i * 16
                if base + 16 > len(data):
                    break
                x, y, z, dop = struct.unpack_from('<ffff', data, base)
                points.append([x, y, z, dop, 15.0])   # snr=15 placeholder

        elif tlv_type == TLV_POINT_CLOUD_EXT:
            # Extended point cloud: x(f), y(f), z(f), doppler(f), snr(f) = 20 bytes
            n = tlv_length // 20
            for i in range(n):
                base = offset + i * 20
                if base + 20 > len(data):
                    break
                x, y, z, dop, snr = struct.unpack_from('<fffff', data, base)
                points.append([x, y, z, dop, snr])

        offset += tlv_length

    return points


def frame_generator(data_port_name: str = SERIAL_PORT_DATA, baud: int = SERIAL_BAUD):
    """
    Generator that yields one radar frame at a time as a list of points.

    Each point is [x, y, z, doppler, snr].

    Usage:
        for frame_points in frame_generator():
            process(frame_points)
    """
    ser = serial.Serial(data_port_name, baud, timeout=1)
    print(f"[radar] Listening on {data_port_name} @ {baud} baud …")

    buf = bytearray()
    while True:
        chunk = ser.read(ser.in_waiting or 512)
        if chunk:
            buf.extend(chunk)

        # Search for magic word
        idx = buf.find(MAGIC_WORD)
        if idx == -1:
            if len(buf) > 4096:
                buf = buf[-8:]   # keep tail in case magic is split
            continue

        # Drop bytes before magic word
        if idx > 0:
            buf = buf[idx:]

        # Need at least the full header (40 bytes after magic)
        if len(buf) < 40:
            continue

        # Parse frame header
        try:
            (
                magic,                        # 8 bytes (already found)
                version, total_len,
                platform, frame_num,
                cpu_cycles, num_obj, num_tlvs,
                subframe_num
            ) = struct.unpack_from('<8sIIIIIIII', buf, 0)
        except struct.error:
            buf = buf[8:]
            continue

        # Wait until the full frame has arrived
        if len(buf) < total_len:
            continue

        frame_data = bytes(buf[:total_len])
        buf = buf[total_len:]           # consume frame from buffer

        # Parse TLVs (start after 40-byte header)
        points = _parse_tlv(frame_data, offset=40, num_tlvs=num_tlvs)
        yield {
            "pointCloud": points,
            "trackData": [],    # Not extracted in binary parser yet
            "heightData": []    # Not extracted in binary parser yet
        }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("[test] Sending sensor config …")
    send_config(SERIAL_PORT_CFG)

    print("[test] Reading frames (Ctrl-C to stop) …")
    for i, frame_dict in enumerate(frame_generator()):
        pts = frame_dict["pointCloud"]
        print(f"  Frame {i:04d}: {len(pts)} points detected")
        if i >= 20:
            break
