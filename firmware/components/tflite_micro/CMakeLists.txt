cmake_minimum_required(VERSION 3.16)

# NeuroEdge — tflite_micro ESP-IDF component
#
# Expects TensorFlow Lite Micro source tree at:
#   ${COMPONENT_DIR}/tflite-micro/
#
# Clone with:
#   cd firmware/components/tflite_micro
#   git clone --depth 1 https://github.com/tensorflow/tflite-micro.git
#
# Alternatively, use Espressif's pre-ported fork (recommended for ESP32):
#   git clone --depth 1 https://github.com/espressif/esp-tflite-micro.git tflite-micro

set(TFLITE_DIR "${CMAKE_CURRENT_LIST_DIR}/tflite-micro")

if(NOT EXISTS "${TFLITE_DIR}/tensorflow/lite/micro/micro_interpreter.h")
    message(FATAL_ERROR
        "TFLite Micro source not found at ${TFLITE_DIR}\n"
        "Run: cd firmware/components/tflite_micro && "
        "git clone --depth 1 "
        "https://github.com/espressif/esp-tflite-micro.git tflite-micro"
    )
endif()

# -----------------------------------------------------------------------
# Source files — micro interpreter + ops needed for DS-CNN and MobileNet
# -----------------------------------------------------------------------

set(TFLM_KERNEL_DIR "${TFLITE_DIR}/tensorflow/lite/micro/kernels")
set(TFLM_CORE_DIR   "${TFLITE_DIR}/tensorflow/lite/micro")

# Core interpreter
set(TFLM_CORE_SRCS
    "${TFLM_CORE_DIR}/micro_interpreter.cc"
    "${TFLM_CORE_DIR}/micro_allocator.cc"
    "${TFLM_CORE_DIR}/micro_context.cc"
    "${TFLM_CORE_DIR}/micro_log.cc"
    "${TFLM_CORE_DIR}/micro_profiler.cc"
    "${TFLM_CORE_DIR}/micro_resource_variable.cc"
    "${TFLM_CORE_DIR}/micro_utils.cc"
    "${TFLM_CORE_DIR}/recording_micro_allocator.cc"
    "${TFLM_CORE_DIR}/arena_allocator/non_persistent_arena_buffer_allocator.cc"
    "${TFLM_CORE_DIR}/arena_allocator/persistent_arena_buffer_allocator.cc"
    "${TFLM_CORE_DIR}/arena_allocator/single_arena_buffer_allocator.cc"
    "${TFLM_CORE_DIR}/memory_helpers.cc"
    "${TFLM_CORE_DIR}/flatbuffer_utils.cc"
)

# TFLite C API (needed by our C model_runner)
set(TFLM_C_SRCS
    "${TFLITE_DIR}/tensorflow/lite/micro/micro_interpreter_context.cc"
    "${TFLITE_DIR}/tensorflow/lite/micro/micro_op_resolver.cc"
)

# Ops required for DS-CNN (depthwise separable convolutions) + MobileNet v3
set(TFLM_OP_SRCS
    "${TFLM_KERNEL_DIR}/conv.cc"
    "${TFLM_KERNEL_DIR}/depthwise_conv.cc"
    "${TFLM_KERNEL_DIR}/fully_connected.cc"
    "${TFLM_KERNEL_DIR}/pooling.cc"
    "${TFLM_KERNEL_DIR}/pooling_common.cc"
    "${TFLM_KERNEL_DIR}/reshape.cc"
    "${TFLM_KERNEL_DIR}/softmax.cc"
    "${TFLM_KERNEL_DIR}/add.cc"
    "${TFLM_KERNEL_DIR}/mul.cc"
    "${TFLM_KERNEL_DIR}/activations.cc"
    "${TFLM_KERNEL_DIR}/activations_common.cc"
    "${TFLM_KERNEL_DIR}/pad.cc"
    "${TFLM_KERNEL_DIR}/quantize.cc"
    "${TFLM_KERNEL_DIR}/dequantize.cc"
    "${TFLM_KERNEL_DIR}/hard_swish.cc"
    # Reference implementations (fallback when no SIMD available)
    "${TFLM_KERNEL_DIR}/conv_common.cc"
    "${TFLM_KERNEL_DIR}/depthwise_conv_common.cc"
    "${TFLM_KERNEL_DIR}/fully_connected_common.cc"
)

# Common TFLite utilities
set(TFLM_COMMON_SRCS
    "${TFLITE_DIR}/tensorflow/lite/kernels/internal/quantization_util.cc"
    "${TFLITE_DIR}/tensorflow/lite/kernels/kernel_util.cc"
    "${TFLITE_DIR}/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cc"
    "${TFLITE_DIR}/tensorflow/compiler/mlir/lite/schema/schema_utils.cc"
    "${TFLITE_DIR}/tensorflow/lite/micro/mock_micro_graph.cc"
)

# -----------------------------------------------------------------------
# Includes
# -----------------------------------------------------------------------

set(TFLM_INCLUDES
    "${TFLITE_DIR}"
    "${TFLITE_DIR}/tensorflow/lite/micro"
    "${TFLITE_DIR}/tensorflow/lite/micro/kernels"
    "${TFLITE_DIR}/tensorflow/lite"
    "${TFLITE_DIR}/third_party/flatbuffers/include"
    "${TFLITE_DIR}/third_party/gemmlowp"
    "${TFLITE_DIR}/third_party/ruy"
)

# -----------------------------------------------------------------------
# Definitions for ESP32-S3 (Xtensa LX7, no SIMD)
# -----------------------------------------------------------------------

set(TFLM_DEFS
    TF_LITE_STATIC_MEMORY=1
    TF_LITE_DISABLE_TELEMETRY=1
    FLATBUFFERS_LOCALE_INDEPENDENT=1
    GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK=1
)

# -----------------------------------------------------------------------
# Compile flags
# -----------------------------------------------------------------------

set(TFLM_FLAGS
    -O2
    -fno-exceptions
    -fno-rtti
    -Wno-unused-function
    -Wno-sign-compare
    -Wno-missing-field-initializers
    -Wno-deprecated-declarations
)

# -----------------------------------------------------------------------
# Register as ESP-IDF component
# -----------------------------------------------------------------------

idf_component_register(
    SRCS
        ${TFLM_CORE_SRCS}
        ${TFLM_C_SRCS}
        ${TFLM_OP_SRCS}
        ${TFLM_COMMON_SRCS}
    INCLUDE_DIRS
        ${TFLM_INCLUDES}
    REQUIRES
        esp_system
        esp_psram
        freertos
)

target_compile_definitions(${COMPONENT_LIB} PRIVATE ${TFLM_DEFS})
target_compile_options(${COMPONENT_LIB}     PRIVATE ${TFLM_FLAGS})
target_compile_features(${COMPONENT_LIB}    PRIVATE cxx_std_11)

message(STATUS "NeuroEdge: tflite_micro component configured (DS-CNN + MobileNet ops)")
