# ggml/src/ggml-iluvatar/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)

# CoreX SDK 路径（默认 /usr/local/corex，可通过环境变量 COREX_PATH 或 -DILUVATAR_CUDA_PATH= 覆盖）
if (DEFINED ENV{COREX_PATH})
    set(ILUVATAR_CUDA_PATH_DEFAULT "$ENV{COREX_PATH}")
else()
    set(ILUVATAR_CUDA_PATH_DEFAULT "/usr/local/corex")
endif()
set(ILUVATAR_CUDA_PATH "${ILUVATAR_CUDA_PATH_DEFAULT}" CACHE PATH "Iluvatar CoreX SDK path")

# 验证 CoreX SDK 存在
if (NOT EXISTS "${ILUVATAR_CUDA_PATH}/bin/clang++")
    message(FATAL_ERROR "Iluvatar CoreX SDK not found at ${ILUVATAR_CUDA_PATH}. "
                        "Set -DILUVATAR_CUDA_PATH=<path> or COREX_PATH env var.")
endif()

# 设置编译器为 CoreX Clang
set(CMAKE_C_COMPILER "${ILUVATAR_CUDA_PATH}/bin/clang")
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_COMPILER "${ILUVATAR_CUDA_PATH}/bin/clang++")
set(CMAKE_CXX_EXTENSIONS OFF)

# 复用 ggml-cuda 的头文件和源文件
file(GLOB GGML_HEADERS_ILUVATAR "../ggml-cuda/*.cuh")
list(APPEND GGML_HEADERS_ILUVATAR "../../include/ggml-cuda.h")

file(GLOB GGML_SOURCES_ILUVATAR "../ggml-cuda/*.cu")

# 模板实例化文件
file(GLOB SRCS "../ggml-cuda/template-instances/fattn-tile*.cu")
list(APPEND GGML_SOURCES_ILUVATAR ${SRCS})
# fattn-mma 使用 Tensor Core MMA 指令，Iluvatar 不支持，不编译
file(GLOB SRCS "../ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_ILUVATAR ${SRCS})
file(GLOB SRCS "../ggml-cuda/template-instances/mmf*.cu")
list(APPEND GGML_SOURCES_ILUVATAR ${SRCS})

# 默认包含常用的 fattn-vec 实例
if (GGML_CUDA_FA_ALL_QUANTS)
    file(GLOB SRCS "../ggml-cuda/template-instances/fattn-vec*.cu")
    list(APPEND GGML_SOURCES_ILUVATAR ${SRCS})
else()
    list(APPEND GGML_SOURCES_ILUVATAR
        ../ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-q8_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-bf16-bf16.cu)
endif()

# 将 .cu 文件作为 CXX 编译（CoreX Clang 处理）
set_source_files_properties(${GGML_SOURCES_ILUVATAR} PROPERTIES LANGUAGE CXX)

# CoreX Clang 编译标志
foreach(SOURCE ${GGML_SOURCES_ILUVATAR})
    set(COMPILE_FLAGS "-fno-strict-aliasing -ffast-math -fsigned-char -fno-strict-float-cast-overflow -x ivcore --cuda-gpu-arch=ivcore11 --cuda-path=${ILUVATAR_CUDA_PATH} -std=c++17")
    set_property(SOURCE ${SOURCE} PROPERTY COMPILE_FLAGS ${COMPILE_FLAGS})
endforeach()

# 创建后端库
ggml_add_backend_library(ggml-iluvatar
                         ${GGML_HEADERS_ILUVATAR}
                         ${GGML_SOURCES_ILUVATAR}
                        )

# 兼容 CUDA 后端的宏定义
if (NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
endif()

target_compile_definitions(ggml-iluvatar PRIVATE
    GGML_USE_ILUVATAR
    GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE}
)

if (GGML_CUDA_FORCE_MMQ)
    target_compile_definitions(ggml-iluvatar PRIVATE GGML_CUDA_FORCE_MMQ)
endif()

if (GGML_CUDA_FORCE_CUBLAS)
    target_compile_definitions(ggml-iluvatar PRIVATE GGML_CUDA_FORCE_CUBLAS)
endif()

if (GGML_CUDA_NO_VMM)
    target_compile_definitions(ggml-iluvatar PRIVATE GGML_CUDA_NO_VMM)
endif()

if (NOT GGML_CUDA_FA)
    target_compile_definitions(ggml-iluvatar PRIVATE GGML_CUDA_NO_FA)
endif()

if (GGML_CUDA_NO_PEER_COPY)
    target_compile_definitions(ggml-iluvatar PRIVATE GGML_CUDA_NO_PEER_COPY)
endif()

if (GGML_CUDA_FA_ALL_QUANTS)
    target_compile_definitions(ggml-iluvatar PRIVATE GGML_CUDA_FA_ALL_QUANTS)
endif()

if (GGML_CUDA_GRAPHS)
    target_compile_definitions(ggml-iluvatar PRIVATE GGML_CUDA_USE_GRAPHS USE_CUDA_GRAPH)
endif()

# 链接 CoreX 兼容库
target_link_directories(ggml-iluvatar PRIVATE "${ILUVATAR_CUDA_PATH}/lib64")
if (GGML_STATIC)
    target_link_libraries(ggml-iluvatar PRIVATE cudart_static cublas_static)
else()
    target_link_libraries(ggml-iluvatar PRIVATE cudart cublas)
endif()

if (NOT GGML_CUDA_NO_VMM)
    target_link_libraries(ggml-iluvatar PRIVATE cuda)  # CoreX Driver API
endif()
