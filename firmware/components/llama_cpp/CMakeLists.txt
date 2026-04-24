cmake_minimum_required(VERSION 3.16)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_LIST_DIR}/llama.cpp")

if(NOT EXISTS "${LLAMA_CPP_DIR}/ggml/src/ggml.c")
    message(FATAL_ERROR
        "llama.cpp source not found at ${LLAMA_CPP_DIR}\n"
        "Run: ./setup.sh")
endif()

# -----------------------------------------------------------------------
# ggml core — exclude files that need dlfcn/filesystem (not on ESP32)
# ggml-backend-reg.cpp is replaced by our ESP32 stub
# -----------------------------------------------------------------------
set(GGML_SRCS
    "${LLAMA_CPP_DIR}/ggml/src/ggml.c"
    "${LLAMA_CPP_DIR}/ggml/src/ggml-alloc.c"
    "${LLAMA_CPP_DIR}/ggml/src/ggml-backend.cpp"
    "${LLAMA_CPP_DIR}/ggml/src/ggml-backend-meta.cpp"
    "${LLAMA_CPP_DIR}/ggml/src/ggml-threading.cpp"
    "${LLAMA_CPP_DIR}/ggml/src/ggml-quants.c"
    "${LLAMA_CPP_DIR}/ggml/src/ggml-cpu/ggml-cpu.c"
    "${LLAMA_CPP_DIR}/ggml/src/ggml-cpu/ggml-cpu.cpp"
    "${LLAMA_CPP_DIR}/ggml/src/ggml-cpu/quants.c"
    "${CMAKE_CURRENT_LIST_DIR}/ggml-backend-reg-esp32.cpp"
)

# -----------------------------------------------------------------------
# llama.cpp src — all files present in this version
# -----------------------------------------------------------------------
set(LLAMA_SRCS
    "${LLAMA_CPP_DIR}/src/llama.cpp"
    "${LLAMA_CPP_DIR}/src/llama-adapter.cpp"
    "${LLAMA_CPP_DIR}/src/llama-arch.cpp"
    "${LLAMA_CPP_DIR}/src/llama-batch.cpp"
    "${LLAMA_CPP_DIR}/src/llama-chat.cpp"
    "${LLAMA_CPP_DIR}/src/llama-context.cpp"
    "${LLAMA_CPP_DIR}/src/llama-cparams.cpp"
    "${LLAMA_CPP_DIR}/src/llama-grammar.cpp"
    "${LLAMA_CPP_DIR}/src/llama-graph.cpp"
    "${LLAMA_CPP_DIR}/src/llama-hparams.cpp"
    "${LLAMA_CPP_DIR}/src/llama-impl.cpp"
    "${LLAMA_CPP_DIR}/src/llama-io.cpp"
    "${LLAMA_CPP_DIR}/src/llama-kv-cache.cpp"
    "${LLAMA_CPP_DIR}/src/llama-kv-cache-iswa.cpp"
    "${LLAMA_CPP_DIR}/src/llama-memory.cpp"
    "${LLAMA_CPP_DIR}/src/llama-memory-hybrid.cpp"
    "${LLAMA_CPP_DIR}/src/llama-memory-hybrid-iswa.cpp"
    "${LLAMA_CPP_DIR}/src/llama-memory-recurrent.cpp"
    "${LLAMA_CPP_DIR}/src/llama-mmap.cpp"
    "${LLAMA_CPP_DIR}/src/llama-model.cpp"
    "${LLAMA_CPP_DIR}/src/llama-model-loader.cpp"
    "${LLAMA_CPP_DIR}/src/llama-model-saver.cpp"
    "${LLAMA_CPP_DIR}/src/llama-quant.cpp"
    "${LLAMA_CPP_DIR}/src/llama-sampler.cpp"
    "${LLAMA_CPP_DIR}/src/llama-vocab.cpp"
    "${LLAMA_CPP_DIR}/src/unicode.cpp"
    "${LLAMA_CPP_DIR}/src/unicode-data.cpp"
)

# -----------------------------------------------------------------------
# Include paths
# -----------------------------------------------------------------------
set(LLAMA_INCLUDES
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/ggml/src"
    "${LLAMA_CPP_DIR}/ggml/src/ggml-cpu"
    "${LLAMA_CPP_DIR}/src"
    "${LLAMA_CPP_DIR}/common"
    "${CMAKE_CURRENT_LIST_DIR}"
)

# -----------------------------------------------------------------------
# Compile definitions
# -----------------------------------------------------------------------
set(LLAMA_DEFS
    GGML_CONFIG_FILE="ggml_esp32_config.h"
    GGML_USE_CPU=1
    GGML_NO_METAL=1
    GGML_NO_ACCELERATE=1
    GGML_NO_OPENMP=1
    GGML_NO_LLAMAFILE=1
    GGML_NO_CUDA=1
    GGML_NO_VULKAN=1
    GGML_NO_KOMPUTE=1
    GGML_NO_SYCL=1
    GGML_MAX_NODES=2048
    GGML_COMMIT="esp32-embedded"
    GGML_VERSION="esp32-embedded"
    LLAMA_NO_METAL=1
    _GLIBCXX_HAVE_POSIX_SEMAPHORE
    _GLIBCXX_USE_POSIX_SEMAPHORE
    _GNU_SOURCE=1
    _FILE_OFFSET_BITS=64
    _POSIX_READER_WRITER_LOCKS
)

# -----------------------------------------------------------------------
# Compile flags
# -----------------------------------------------------------------------
set(LLAMA_FLAGS
    -O2
    -fno-exceptions
    -fno-rtti
    -Wno-unused-function
    -Wno-deprecated-declarations
    -Wno-missing-field-initializers
    -Wno-sign-compare
    -Wno-error=format
    -Wno-format
    -Wno-error=return-type
    -fpermissive
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

target_compile_definitions(${COMPONENT_LIB} PRIVATE ${LLAMA_DEFS})
target_compile_options(${COMPONENT_LIB}     PRIVATE ${LLAMA_FLAGS})
target_compile_features(${COMPONENT_LIB}    PRIVATE cxx_std_17)

message(STATUS "NeuroEdge: llama_cpp configured for ESP32 (CPU-only, ${CMAKE_SYSTEM_PROCESSOR})")
