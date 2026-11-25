#!/usr/bin/python3
import sys

MAX_PAYLOAD = 4096


def main():
    payload = sys.stdin.buffer.read(MAX_PAYLOAD + 1)
    if len(payload) > MAX_PAYLOAD or any(not 32 <= byte <= 126 for byte in payload):
        return 1

    checksum = 2166136261
    for byte in payload:
        checksum = ((checksum ^ byte) * 16777619) & 0xFFFFFFFF
    print(f"{checksum:08x}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
