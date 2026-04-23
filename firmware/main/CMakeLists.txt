idf_component_register(
    SRCS
        "main.c"
        "uart_server.c"
        "model_runner.c"
        "model_loader.c"
        "mfcc.c"
        "kws.cc"
        "crc16.c"
    INCLUDE_DIRS
        "."
    REQUIRES
        driver
        esp_system
        esp_partition
        esp_psram
        nvs_flash
        freertos
        espressif__cjson
        llama_cpp
        tflite_micro
)

# Enable llama.cpp + TFLite Micro integrations
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    NE_LLAMA_CPP_AVAILABLE=1
    NE_TFLITE_AVAILABLE=1
)
