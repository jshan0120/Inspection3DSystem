cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Inspection3dEngine LANGUAGES C CXX CUDA)
add_subdirectory(third_party/HighFive)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(PathConfig)
message("CMAKE_PREFIX_PATH : " ${CMAKE_PREFIX_PATH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
# list(APPEND LIBS ${TORCH_LIBRARIES})

find_package(HDF5 REQUIRED COMPONENTS C CXX)

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
     deep_learning/src/data_loader.cpp
)

target_include_directories(dcp PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/deep_learning/include 
    ${HDF5_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(dcp ${TORCH_LIBRARIES} Eigen3::Eigen HDF5::HDF5 HighFive)
set_property(TARGET dcp PROPERTY CXX_STANDARD 17)

add_executable(inference
     deep_learning/src/dcp.cpp
     deep_learning/src/dgcnn.cpp
     deep_learning/src/transformer.cpp
     deep_learning/src/multihead_attention.cpp
     deep_learning/src/position_ffd.cpp
     deep_learning/src/svd_head.cpp
     inspection_system/src/main.cpp
     inspection_system/src/dcp_inference.cpp
     inspection_system/src/core.cpp)

target_link_libraries(inference ${TORCH_LIBRARIES} Eigen3::Eigen HighFive)
target_include_directories(inference PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/inspection_system/include
    ${CMAKE_CURRENT_SOURCE_DIR}/deep_learning/include 
    ${EIGEN3_INCLUDE_DIRS}
)