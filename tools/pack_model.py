#!/usr/bin/env python3
"""
NeuroEdge Model Packer

Prepends a 64-byte ne_model_header_t to a raw model file (.gguf or .tflite)
so the firmware's model_loader.c can validate and mmap it.

Usage:
  python3 tools/pack_model.py \\
      --input tinyllama-1.1b-q4_k_m.gguf \\
      --output model_0.bin \\
      --model-id tinyllama-q4

  python3 tools/pack_model.py \\
      --input ds_cnn_kws.tflite \\
      --output model_1.bin \\
      --model-id dscnn-kws

Then flash with esptool:
  esptool.py --chip esp32s3 --port /dev/ttyUSB0 --baud 921600 \\
      write_flash 0x500000 model_0.bin
"""

import argparse
import struct
import zlib
import sys
from pathlib import Path

# Must match ne_model_header_t in model_loader.h exactly
MODEL_MAGIC          = 0x4E454D4C   # "NEML"
MODEL_HEADER_VERSION = 1
MODEL_TYPE_GGUF      = 1
MODEL_TYPE_TFLITE    = 2
HEADER_SIZE          = 64           # bytes, must stay fixed


def detect_model_type(filename: str) -> int:
    f = filename.lower()
    if f.endswith(".gguf"):   return MODEL_TYPE_GGUF
    if f.endswith(".tflite"): return MODEL_TYPE_TFLITE
    raise ValueError(f"Cannot detect model type from filename: {filename}\n"
                     f"Use a .gguf or .tflite file, or add --type gguf|tflite")


def pack_model(input_path: Path, output_path: Path,
               model_id: str, model_type: int) -> None:
    data = input_path.read_bytes()
    crc  = zlib.crc32(data) & 0xFFFFFFFF

    # Pack the header:
    #   uint32  magic
    #   uint8   version
    #   uint8   model_type
    #   uint16  reserved
    #   uint32  data_size
    #   uint32  crc32
    #   char[32] model_id
    #   uint8[16] padding
    model_id_bytes = model_id.encode("ascii")[:31].ljust(32, b"\x00")

    header = struct.pack(
        "<IBBHII32s16s",
        MODEL_MAGIC,
        MODEL_HEADER_VERSION,
        model_type,
        0,                        # reserved
        len(data),
        crc,
        model_id_bytes,
        b"\x00" * 16,             # padding
    )
    assert len(header) == HEADER_SIZE, f"Header size mismatch: {len(header)}"

    output_path.write_bytes(header + data)

    size_mb = (len(data) + HEADER_SIZE) / (1024 * 1024)
    print(f"  Input:    {input_path}  ({len(data):,} bytes)")
    print(f"  Output:   {output_path}  ({size_mb:.2f} MB)")
    print(f"  Model ID: {model_id}")
    print(f"  Type:     {'GGUF' if model_type == MODEL_TYPE_GGUF else 'TFLite'}")
    print(f"  CRC32:    0x{crc:08X}")
    print()
    print(f"Flash command:")
    print(f"  esptool.py --chip esp32s3 --port /dev/ttyUSB0 --baud 921600 \\")
    print(f"      write_flash 0x500000 {output_path.name}")


def main():
    p = argparse.ArgumentParser(description="Pack a model file for NeuroEdge flash")
    p.add_argument("--input",    "-i", required=True)
    p.add_argument("--output",   "-o", required=True)
    p.add_argument("--model-id", "-m", required=True,
                   help="Short identifier, e.g. tinyllama-q4")
    p.add_argument("--type",     "-t", choices=["gguf", "tflite"],
                   help="Force model type (auto-detected from extension if omitted)")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    if not inp.exists():
        print(f"ERROR: Input file not found: {inp}", file=sys.stderr)
        sys.exit(1)

    model_type = (MODEL_TYPE_GGUF   if args.type == "gguf"   else
                  MODEL_TYPE_TFLITE if args.type == "tflite" else
                  detect_model_type(args.input))

    pack_model(inp, out, args.model_id, model_type)


if __name__ == "__main__":
    main()
