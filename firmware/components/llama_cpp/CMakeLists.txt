cmake_minimum_required(VERSION 3.16)

# NeuroEdge — llama_cpp ESP-IDF component
#
# Expects the llama.cpp source tree at:
#   ${COMPONENT_DIR}/llama.cpp/
#
# Clone with:
#   cd firmware/components/llama_cpp
#   git clone https://github.com/ggerganov/llama.cpp.git
#   cd llama.cpp && git checkout b3617  # last known stable for ESP32
#
# Key compile flags for ESP32-S3 (Xtensa LX7, no SIMD, PSRAM):
#   -DGGML_NO_METAL / CUDA / VULKAN etc
#   -DGGML_CONFIG_FILE points to our ggml_esp32_config.h

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_LIST_DIR}/llama.cpp")

if(NOT EXISTS "${LLAMA_CPP_DIR}/ggml/src/ggml.c")
    message(FATAL_ERROR
        "llama.cpp source not found at ${LLAMA_CPP_DIR}\n"
        "Run: cd firmware/components/llama_cpp && "
        "git clone https://github.com/ggerganov/llama.cpp.git"
    )
endif()

# -----------------------------------------------------------------------
# Source files — include only what is needed for CPU inference
# -----------------------------------------------------------------------

set(GGML_SRCS
    "${LLAMA_CPP_DIR}/ggml/src/ggml.c"
    "${LLAMA_CPP_DIR}/ggml/src/ggml-alloc.c"
    "${LLAMA_CPP_DIR}/ggml/src/ggml-backend.c"
    "${LLAMA_CPP_DIR}/ggml/src/ggml-quants.c"
)

set(LLAMA_SRCS
    "${LLAMA_CPP_DIR}/src/llama.cpp"
    "${LLAMA_CPP_DIR}/src/llama-grammar.cpp"
    "${LLAMA_CPP_DIR}/src/llama-sampling.cpp"
    "${LLAMA_CPP_DIR}/src/llama-vocab.cpp"
    "${LLAMA_CPP_DIR}/src/unicode.cpp"
    "${LLAMA_CPP_DIR}/src/unicode-data.cpp"
    "${LLAMA_CPP_DIR}/common/common.cpp"
    "${LLAMA_CPP_DIR}/common/sampling.cpp"
)

# -----------------------------------------------------------------------
# Include directories
# -----------------------------------------------------------------------

set(LLAMA_INCLUDES
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/ggml/src"
    "${LLAMA_CPP_DIR}/common"
    "${CMAKE_CURRENT_LIST_DIR}"    # for ggml_esp32_config.h
)

# -----------------------------------------------------------------------
# Compile definitions — disable everything we don't have hardware for
# -----------------------------------------------------------------------

set(LLAMA_DEFS
    # Disable all accelerators
    GGML_NO_METAL=1
    GGML_NO_ACCELERATE=1
    GGML_NO_OPENMP=1
    GGML_NO_CUDA=1
    GGML_NO_VULKAN=1
    GGML_NO_KOMPUTE=1
    GGML_NO_SYCL=1
    GGML_NO_LLAMAFILE=1
    # Route memory to PSRAM
    GGML_USE_CPU=1
    # Point to our ESP32 config header
    GGML_CONFIG_FILE=\"ggml_esp32_config.h\"
    # Reduce static memory
    GGML_MAX_NODES=4096
    # llama.cpp options
    LLAMA_NO_METAL=1
    _GNU_SOURCE=1
)

# -----------------------------------------------------------------------
# Compile flags — safe for Xtensa LX7
# -----------------------------------------------------------------------

set(LLAMA_COMPILE_FLAGS
    -O2
    -fno-exceptions           # saves ~10 KB
    -Wno-unused-function
    -Wno-deprecated-declarations
    -Wno-missing-field-initializers
    # Xtensa LX7 has no x86/ARM SIMD — no flags needed
)

# -----------------------------------------------------------------------
# Register as ESP-IDF component
# -----------------------------------------------------------------------

idf_component_register(
    SRCS
        ${GGML_SRCS}
        ${LLAMA_SRCS}
    INCLUDE_DIRS
        ${LLAMA_INCLUDES}
    REQUIRES
        esp_system
        esp_psram
        freertos
        nvs_flash
)

# Apply definitions and flags to all sources in this component
target_compile_definitions(${COMPONENT_LIB} PRIVATE ${LLAMA_DEFS})
target_compile_options(${COMPONENT_LIB}     PRIVATE ${LLAMA_COMPILE_FLAGS})

# C++ standard — llama.cpp requires C++17
target_compile_features(${COMPONENT_LIB} PRIVATE cxx_std_17)

# Allow large files (model loading code uses size_t heavily)
target_compile_options(${COMPONENT_LIB} PRIVATE -D_FILE_OFFSET_BITS=64)

message(STATUS "NeuroEdge: llama_cpp component configured (CPU-only, PSRAM)")
