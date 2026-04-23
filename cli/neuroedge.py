#!/usr/bin/env python3
"""
NeuroEdge Developer CLI

Usage:
  neuroedge flash   --model tinyllama-q4.gguf --port /dev/ttyUSB0
  neuroedge ask     --port /dev/ttyUSB0 "What is 2+2?"
  neuroedge ping    --port /dev/ttyUSB0
  neuroedge info    --port /dev/ttyUSB0
  neuroedge benchmark --port /dev/ttyUSB0
  neuroedge monitor --port /dev/ttyUSB0
  neuroedge list-models

Install:
  pip install pyserial tqdm rich
"""

import argparse
import json
import os
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Optional

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("ERROR: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ---------------------------------------------------------------------------
# CRC16-CCITT (matches firmware implementation exactly)
# ---------------------------------------------------------------------------

def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        for _ in range(8):
            if (crc >> 15) ^ (byte >> 7):
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
            byte = (byte << 1) & 0xFF
    return crc


def crc16_hex(data: bytes) -> str:
    return f"{crc16(data):04X}"


# ---------------------------------------------------------------------------
# NeuroEdge serial transport
# ---------------------------------------------------------------------------

class NeuroEdgeError(Exception):
    pass


class NeuroEdgeClient:
    """Low-level JSON-over-UART client."""

    DEFAULT_BAUD    = 115200
    DEFAULT_TIMEOUT = 10.0  # seconds

    def __init__(self, port: str, baud: int = DEFAULT_BAUD,
                 timeout: float = DEFAULT_TIMEOUT):
        self.port    = port
        self.baud    = baud
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None
        self._next_id = 1

    def open(self):
        try:
            self._ser = serial.Serial(
                port     = self.port,
                baudrate = self.baud,
                bytesize = serial.EIGHTBITS,
                parity   = serial.PARITY_NONE,
                stopbits = serial.STOPBITS_ONE,
                timeout  = 0.05,
            )
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
        except serial.SerialException as e:
            raise NeuroEdgeError(f"Cannot open {self.port}: {e}") from e

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    def _send(self, obj: dict):
        """Serialize obj to JSON, append CRC, send over UART."""
        # Build body without CRC first, then append
        body = json.dumps(obj, separators=(",", ":"))
        crc  = crc16_hex(body.encode())
        # Insert crc field before closing brace
        frame = body[:-1] + f',"crc":"{crc}"}}' + "\r\n"
        self._ser.write(frame.encode())
        self._ser.flush()

    def _recv(self, timeout: Optional[float] = None) -> dict:
        """Read a newline-terminated JSON response, verify CRC, return dict."""
        deadline = time.monotonic() + (timeout or self.timeout)
        buf = b""

        while time.monotonic() < deadline:
            chunk = self._ser.read(256)
            if chunk:
                buf += chunk
                if b"\n" in buf:
                    line, _ = buf.split(b"\n", 1)
                    line = line.rstrip(b"\r")
                    break
        else:
            raise NeuroEdgeError(
                f"Timeout waiting for response ({timeout or self.timeout:.1f}s)")

        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            raise NeuroEdgeError("Empty response received")

        try:
            resp = json.loads(text)
        except json.JSONDecodeError as e:
            raise NeuroEdgeError(f"Invalid JSON response: {text[:80]}") from e

        # Verify CRC
        if "crc" in resp:
            crc_pos = text.find('"crc"')
            if crc_pos >= 0:
                check_text = text[:crc_pos].rstrip(", ")
                expected   = crc16_hex(check_text.encode())
                if resp["crc"].upper() != expected:
                    raise NeuroEdgeError(
                        f"CRC mismatch: got {resp['crc']} expected {expected}")

        if resp.get("status") == "error":
            code = resp.get("code", "unknown")
            msg  = resp.get("message", "")
            raise NeuroEdgeError(f"Module error [{code}]: {msg}")

        return resp

    def command(self, cmd: dict, timeout: Optional[float] = None) -> dict:
        """Send a command dict, return response dict."""
        if not self._ser or not self._ser.is_open:
            raise NeuroEdgeError("Port not open")
        cmd["id"] = self._next_id
        self._next_id += 1
        self._send(cmd)
        return self._recv(timeout)

    # ------------------------------------------------------------------
    # High-level commands
    # ------------------------------------------------------------------

    def ping(self) -> float:
        """Return round-trip latency in ms."""
        t0   = time.monotonic()
        self.command({"cmd": "ping"})
        return (time.monotonic() - t0) * 1000.0

    def ask(self, prompt: str, max_tokens: int = 64, timeout: float = 30.0) -> dict:
        return self.command(
            {"cmd": "ask", "prompt": prompt, "max_tokens": max_tokens},
            timeout=timeout,
        )

    def info(self) -> dict:
        return self.command({"cmd": "info"})

    def set_model(self, model_name: str) -> dict:
        return self.command({"cmd": "set_model", "model": model_name})

    def sleep(self, duration_ms: int = 0) -> dict:
        return self.command({"cmd": "sleep", "duration_ms": duration_ms})


# ---------------------------------------------------------------------------
# Known models catalog
# ---------------------------------------------------------------------------

KNOWN_MODELS = [
    {
        "id":          "tinyllama-q4",
        "filename":    "tinyllama-1.1b-q4_k_m.gguf",
        "description": "TinyLlama 1.1B Q4_K_M — general intelligence, intent parsing",
        "size_mb":     600,
        "backend":     "llama.cpp",
    },
    {
        "id":          "mobilenet-v3",
        "filename":    "mobilenet_v3_small.tflite",
        "description": "MobileNet v3 Small — image/sensor classification",
        "size_mb":     10,
        "backend":     "TFLite Micro",
    },
    {
        "id":          "dscnn-kws",
        "filename":    "ds_cnn_kws.tflite",
        "description": "DS-CNN — keyword spotting (yes/no/stop/go/…)",
        "size_mb":     1,
        "backend":     "TFLite Micro",
    },
]


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_list_models(args):
    if HAS_RICH:
        c = Console()
        t = Table(title="NeuroEdge Available Models", show_header=True)
        t.add_column("ID",          style="cyan",  no_wrap=True)
        t.add_column("Description", style="white")
        t.add_column("Size",        style="green", justify="right")
        t.add_column("Backend",     style="yellow")
        for m in KNOWN_MODELS:
            t.add_row(m["id"], m["description"],
                      f"{m['size_mb']} MB", m["backend"])
        c.print(t)
    else:
        for m in KNOWN_MODELS:
            print(f"  {m['id']:<20} {m['description']} ({m['size_mb']} MB)")


def cmd_ping(args):
    with NeuroEdgeClient(args.port, args.baud) as client:
        samples = []
        count = getattr(args, "count", 5)
        for i in range(count):
            try:
                ms = client.ping()
                samples.append(ms)
                print(f"  pong {i+1}/{count} — {ms:.1f} ms")
                time.sleep(0.1)
            except NeuroEdgeError as e:
                print(f"  ERROR: {e}")
        if samples:
            print(f"\n  avg={sum(samples)/len(samples):.1f} ms  "
                  f"min={min(samples):.1f} ms  max={max(samples):.1f} ms")


def cmd_ask(args):
    with NeuroEdgeClient(args.port, args.baud, timeout=args.timeout) as client:
        print(f"Asking: {args.prompt!r}")
        t0   = time.monotonic()
        resp = client.ask(args.prompt, max_tokens=args.max_tokens,
                          timeout=args.timeout)
        elapsed = (time.monotonic() - t0) * 1000.0
        print(f"\nAnswer: {resp.get('text', '')}")
        print(f"Latency: {resp.get('ms', elapsed):.0f} ms")


def cmd_info(args):
    with NeuroEdgeClient(args.port, args.baud) as client:
        resp = client.info()
        print(f"  Firmware:  {resp.get('firmware', '?')}")
        print(f"  Model:     {resp.get('model', '?')}")
        print(f"  Heap free: {resp.get('heap_free', '?')} bytes")


def cmd_benchmark(args):
    prompts = [
        ("Math",    "What is 7 times 8?",                        10),
        ("Logic",   "Is 17 a prime number? Answer yes or no.",   10),
        ("General", "Name the capital of France in one word.",   10),
        ("Code",    "What does LED stand for?",                   10),
    ]

    print(f"NeuroEdge Benchmark — {args.port}")
    print("-" * 55)

    results = []
    with NeuroEdgeClient(args.port, args.baud, timeout=60.0) as client:
        for category, prompt, max_tok in prompts:
            try:
                t0   = time.monotonic()
                resp = client.ask(prompt, max_tokens=max_tok, timeout=60.0)
                ms   = (time.monotonic() - t0) * 1000.0
                text = resp.get("text", "").strip()
                results.append((category, prompt, text, ms))
                print(f"  [{category:<8}] {ms:6.0f} ms | {text[:40]}")
            except NeuroEdgeError as e:
                print(f"  [{category:<8}] ERROR: {e}")

    if results:
        avg = sum(r[3] for r in results) / len(results)
        print(f"\n  Average latency: {avg:.0f} ms over {len(results)} queries")


def cmd_monitor(args):
    """Raw serial monitor — print everything from the module."""
    print(f"Monitoring {args.port} @ {args.baud} baud. Ctrl-C to quit.\n")
    try:
        with serial.Serial(args.port, args.baud, timeout=0.1) as ser:
            while True:
                data = ser.read(256)
                if data:
                    sys.stdout.buffer.write(data)
                    sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
    except serial.SerialException as e:
        print(f"ERROR: {e}")


def cmd_flash(args):
    """
    Flash a model file to the NeuroEdge module using esptool (SPIFFS/LittleFS).
    Requires: pip install esptool
    """
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Model:  {model_path.name}  ({size_mb:.1f} MB)")
    print(f"Port:   {args.port}")
    print(f"Target: ESP32-S3")
    print()

    # Resolve model ID from filename
    model_id = None
    for m in KNOWN_MODELS:
        if model_path.name == m["filename"] or args.model_id == m["id"]:
            model_id = m["id"]
            break
    if not model_id:
        print("WARNING: Unrecognized model filename — proceeding anyway")
        model_id = model_path.stem

    # Check esptool availability
    import shutil
    if not shutil.which("esptool.py") and not shutil.which("esptool"):
        print("ERROR: esptool not found. Install with: pip install esptool")
        sys.exit(1)

    esptool = shutil.which("esptool.py") or "esptool"

    # SPIFFS partition starts at 0x290000 on the 16MB flash layout
    # (adjust in partitions.csv if your layout differs)
    SPIFFS_OFFSET = "0x290000"

    print(f"Flashing to SPIFFS @ {SPIFFS_OFFSET} ...")
    import subprocess
    result = subprocess.run(
        [
            esptool,
            "--chip", "esp32s3",
            "--port", args.port,
            "--baud", "921600",
            "write_flash", SPIFFS_OFFSET, str(model_path),
        ],
        capture_output=False,
    )

    if result.returncode != 0:
        print("ERROR: Flash failed.")
        sys.exit(result.returncode)

    print(f"\nFlash complete. Model '{model_id}' is now on the module.")
    print(f"Activate with:  neuroedge set-model --port {args.port} {model_id}")


def cmd_set_model(args):
    with NeuroEdgeClient(args.port, args.baud) as client:
        resp = client.set_model(args.model_id)
        print(f"Active model: {resp.get('model', '?')}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neuroedge",
        description="NeuroEdge Developer CLI — local AI module toolkit",
    )
    p.add_argument("--port",    "-p", default="/dev/ttyUSB0",
                   help="UART port (default: /dev/ttyUSB0)")
    p.add_argument("--baud",    "-b", type=int, default=115200,
                   help="Baud rate (default: 115200)")
    p.add_argument("--timeout", "-t", type=float, default=10.0,
                   help="Command timeout in seconds (default: 10)")

    sub = p.add_subparsers(dest="command", required=True)

    # list-models
    sub.add_parser("list-models", help="List supported AI models")

    # ping
    sp = sub.add_parser("ping", help="Check module connectivity")
    sp.add_argument("--count", "-n", type=int, default=5)

    # ask
    sp = sub.add_parser("ask", help="Send a text prompt to the LLM")
    sp.add_argument("prompt")
    sp.add_argument("--max-tokens", type=int, default=64)
    sp.add_argument("--timeout",    type=float, default=30.0)

    # info
    sub.add_parser("info", help="Show firmware version and resource usage")

    # benchmark
    sub.add_parser("benchmark", help="Run inference latency benchmark")

    # monitor
    sub.add_parser("monitor", help="Raw UART monitor (Ctrl-C to quit)")

    # flash
    sp = sub.add_parser("flash", help="Flash a model file to the module")
    sp.add_argument("--model",    "-m", required=True,
                    help="Path to model file (.gguf or .tflite)")
    sp.add_argument("--model-id", default=None,
                    help="Override model ID (auto-detected from filename)")

    # set-model
    sp = sub.add_parser("set-model", help="Switch active model on the module")
    sp.add_argument("model_id", metavar="MODEL_ID",
                    help="e.g. tinyllama-q4, mobilenet-v3, dscnn-kws")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "list-models": cmd_list_models,
        "ping":        cmd_ping,
        "ask":         cmd_ask,
        "info":        cmd_info,
        "benchmark":   cmd_benchmark,
        "monitor":     cmd_monitor,
        "flash":       cmd_flash,
        "set-model":   cmd_set_model,
    }

    fn = dispatch.get(args.command)
    if fn:
        try:
            fn(args)
        except NeuroEdgeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nInterrupted.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
