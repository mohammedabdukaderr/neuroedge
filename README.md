# NeuroEdge

Run quantized LLMs (GGUF) on an ESP32-S3 over UART.

## Requirements

- ESP32-S3 board with PSRAM
- [ESP-IDF v5.x](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/)
- Python 3.8+

## Clone & setup

```bash
git clone <repo-url>
cd neuroedge
./setup.sh
```

`setup.sh` will:
1. Clone llama.cpp into `firmware/components/llama_cpp/llama.cpp`
2. Install Python dependencies (`pyserial`, `rich`, `tqdm`, `esptool`)
3. Make the CLI executable

## Usage

### 1. Pack a model

```bash
python3 tools/pack_model.py \
    --input tinyllama-1.1b-q4_k_m.gguf \
    --output model_0.bin \
    --model-id tinyllama-q4
```

### 2. Flash the model

```bash
esptool.py --chip esp32s3 --port /dev/ttyUSB0 --baud 921600 \
    write_flash 0x500000 model_0.bin
```

### 3. Build & flash firmware

```bash
source ~/esp/esp-idf/export.sh
cd firmware
idf.py set-target esp32s3
idf.py -p /dev/ttyUSB0 flash monitor
```

### 4. Run inference

```bash
python3 cli/neuroedge.py ask --port /dev/ttyUSB0 "What is 2+2?"
```

## Project structure

```
cli/          Host CLI (neuroedge.py)
examples/     Arduino sketches
firmware/     ESP-IDF firmware (ESP32-S3)
sdk/          C and Arduino SDK headers
tools/        Model packing utility
setup.sh      One-shot setup script
```
