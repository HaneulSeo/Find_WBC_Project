cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(LibTorchApp)

set(CMAKE_CXX_STANDARD 17)

# 프로젝트 루트의 libtorch 폴더를 참조하도록 상대 경로 설정
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")

# OpenCV 탐색 (Mac은 brew, Ubuntu는 apt 등으로 설치되어 있어야 함)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(LibTorchApp main.cpp)
target_link_libraries(LibTorchApp "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
set_property(TARGET LibTorchApp PROPERTY CXX_STANDARD 17)