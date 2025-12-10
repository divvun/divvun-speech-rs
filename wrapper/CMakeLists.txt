cmake_minimum_required(VERSION 3.19)
project(tts_wrapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build for performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ExecuTorch paths from EXECUTORCH_SYSROOT environment variable
set(EXECUTORCH_ROOT $ENV{EXECUTORCH_SYSROOT})
if(NOT EXECUTORCH_ROOT)
    message(FATAL_ERROR "EXECUTORCH_SYSROOT environment variable not set")
endif()

set(EXECUTORCH_INCLUDE "${EXECUTORCH_ROOT}/include")
set(EXECUTORCH_LIB "${EXECUTORCH_ROOT}/lib")

# Custom TTS ops (fused ISTFT for Vocos vocoder, LayerNorm for XNNPACK gaps)
add_library(custom_ops STATIC
    custom_ops/op_istft.cpp
    custom_ops/op_layer_norm.cpp
    custom_ops/register_ops.cpp
    custom_ops/pffft.c
)
target_include_directories(custom_ops PRIVATE
    ${EXECUTORCH_INCLUDE}
    ${EXECUTORCH_INCLUDE}/executorch/runtime/core/portable_type/c10
    ${EXECUTORCH_INCLUDE}/executorch/runtime/core/portable_type/c10/torch/headeronly
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_ops
)
target_compile_definitions(custom_ops PRIVATE
    C10_USING_CUSTOM_GENERATED_MACROS
)
# Cross-platform compiler flags
target_compile_options(custom_ops PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /DNDEBUG /fp:fast>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -DNDEBUG -ffast-math>
)

# Create the wrapper library
add_library(tts_wrapper STATIC wrapper.cpp)

# Include directories
target_include_directories(tts_wrapper PRIVATE
    ${EXECUTORCH_INCLUDE}
    ${EXECUTORCH_INCLUDE}/executorch/runtime/core/portable_type/c10
    ${EXECUTORCH_INCLUDE}/executorch/runtime/core/portable_type/c10/torch/headeronly
)

# Compiler definitions
target_compile_definitions(tts_wrapper PRIVATE
    C10_USING_CUSTOM_GENERATED_MACROS
)

# Cross-platform compiler flags
target_compile_options(tts_wrapper PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /DNDEBUG /fp:fast /W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -DNDEBUG -ffast-math -Wall -Wextra -Wno-unused-parameter>
)

# Install targets
install(TARGETS tts_wrapper custom_ops
    ARCHIVE DESTINATION lib
)
