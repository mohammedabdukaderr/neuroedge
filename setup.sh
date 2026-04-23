#!/usr/bin/env bash
# NeuroEdge — one-shot setup script
# Run once after cloning: ./setup.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$REPO_ROOT/firmware/components/llama_cpp/llama.cpp"
TFLITE_DIR="$REPO_ROOT/firmware/components/tflite_micro/tflite-micro"

echo "=== NeuroEdge Setup ==="

# 1. Clone external dependencies
if [ ! -d "$LLAMA_DIR/.git" ]; then
    echo "[1/4] Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
else
    echo "[1/4] llama.cpp already present — skipping clone"
fi

if [ ! -d "$TFLITE_DIR/.git" ]; then
    echo "[1/4] Cloning esp-tflite-micro..."
    git clone --depth 1 https://github.com/espressif/esp-tflite-micro.git "$TFLITE_DIR"
else
    echo "[1/4] esp-tflite-micro already present — skipping clone"
fi

# 2. Python deps for CLI + tools
echo "[2/4] Installing Python dependencies..."
pip install --quiet pyserial rich tqdm esptool 2>/dev/null || \
    echo "       pip install failed — run manually if needed"

# 3. Make CLI executable
chmod +x "$REPO_ROOT/cli/neuroedge.py"

# 4. Source ESP-IDF if available
if [ -f "$HOME/esp/esp-idf/export.sh" ]; then
    echo "[3/4] ESP-IDF found at ~/esp/esp-idf"
    echo "      To build firmware, run:"
    echo "        source ~/esp/esp-idf/export.sh"
    echo "        cd firmware && idf.py set-target esp32s3 build"
else
    echo "[3/4] WARNING: ESP-IDF not found at ~/esp/esp-idf"
    echo "      Install from: https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/"
fi

echo "[4/4] Done."
echo ""
echo "=== Quick start ==="
echo "  1. Get a model:"
echo "     python3 tools/pack_model.py \\"
echo "         --input tinyllama-1.1b-q4_k_m.gguf \\"
echo "         --output model_0.bin --model-id tinyllama-q4"
echo ""
echo "  2. Flash model (ESP32-S3 connected via USB):"
echo "     esptool.py --chip esp32s3 --port /dev/ttyUSB0 --baud 921600 \\"
echo "         write_flash 0x500000 model_0.bin"
echo ""
echo "  3. Build & flash firmware:"
echo "     source ~/esp/esp-idf/export.sh"
echo "     cd firmware && idf.py -p /dev/ttyUSB0 flash monitor"
echo ""
echo "  4. Test from host:"
echo "     python3 cli/neuroedge.py ask --port /dev/ttyUSB0 'What is 2+2?'"
