cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(Inspection3dEngine LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(PathConfig)
message("CMAKE_PREFIX_PATH : " ${CMAKE_PREFIX_PATH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
# list(APPEND LIBS ${TORCH_LIBRARIES})

find_package(Eigen3 REQUIRED)

add_executable(dcp
     deep_learning/src/main.cpp
     deep_learning/src/dcp.cpp
     deep_learning/src/dgcnn.cpp
     deep_learning/src/transformer.cpp
     deep_learning/src/multihead_attention.cpp
     deep_learning/src/position_ffd.cpp
     deep_learning/src/svd_head.cpp
     deep_learning/src/loss.cpp
)

list(APPEND INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/deep_learning/include)
include_directories(${INCLUDE_DIRS})

target_link_libraries(dcp ${TORCH_LIBRARIES})
set_property(TARGET dcp PROPERTY CXX_STANDARD 17)