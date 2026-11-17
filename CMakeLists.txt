cmake_minimum_required(VERSION 3.17)
project(ORB_SLAM3 LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)

if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
endif()

message("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -w")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
# Keep Eigen/PCL alignment compatible with the distribution PCL binaries.
# -O3 remains enabled by CMake's Release defaults.

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX14)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
        add_definitions(-DCOMPILEDWITHC11)
        message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
        add_definitions(-DCOMPILEDWITHC0X)
        message(STATUS "Using flag -std=c++0x.")
else()
        message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 4.4 not found.")
endif()

message("OPENCV VERSION:")
message(${OpenCV_VERSION})

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(PCL 1.12 REQUIRED COMPONENTS common io filters visualization)
find_package(CUDAToolkit REQUIRED)

set(TENSORRT_ROOT "/home/leslie/TensorRT-8.6.1.6" CACHE PATH "TensorRT 8.x installation directory")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
        PATH_SUFFIXES include include/x86_64-linux-gnu include/aarch64-linux-gnu)
find_library(TENSORRT_LIBRARY nvinfer
        HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)
find_library(TENSORRT_ONNX_LIBRARY nvonnxparser
        HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT TENSORRT_ONNX_LIBRARY)
        message(FATAL_ERROR "TensorRT 8.x not found. Set -DTENSORRT_ROOT=/path/to/TensorRT-8.x")
endif()

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/CameraModels
        ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
)

link_directories(
        ${PROJECT_SOURCE_DIR}/lib
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Build the detector and its consumers with the same YoloDetector definition.
add_library(yolo_infer SHARED
        src/yolo/infer.cpp
        src/yolo/calibrator.cpp
        src/yolo/preprocess.cu
        src/yolo/postprocess.cu
        src/yolo/draw.cu)
target_include_directories(yolo_infer PUBLIC
        ${PROJECT_SOURCE_DIR}/include/yolo
        ${OpenCV_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR})
target_link_libraries(yolo_infer PUBLIC
        ${OpenCV_LIBS}
        CUDA::cudart
        ${TENSORRT_LIBRARY}
        PRIVATE ${TENSORRT_ONNX_LIBRARY})
set_target_properties(yolo_infer PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED YES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED YES)

add_library(${PROJECT_NAME} SHARED
        src/System.cc
        src/Tracking.cc
        src/LocalMapping.cc
        src/LoopClosing.cc
        src/ORBextractor.cc
        src/ORBmatcher.cc
        src/FrameDrawer.cc
        src/Converter.cc
        src/MapPoint.cc
        src/KeyFrame.cc
        src/Atlas.cc
        src/Map.cc
        src/MapDrawer.cc
        src/Optimizer.cc
        src/Frame.cc
        src/KeyFrameDatabase.cc
        src/Sim3Solver.cc
        src/Viewer.cc
        src/ImuTypes.cc
        src/G2oTypes.cc
        src/CameraModels/Pinhole.cpp
        src/CameraModels/KannalaBrandt8.cpp
        src/OptimizableTypes.cpp
        src/MLPnPsolver.cpp
        src/GeometricTools.cc
        src/TwoViewReconstruction.cc
        src/Config.cc
        src/Settings.cc
        src/PointCloudMapper.cpp
        src/PointCloudViewer.cpp
        include/System.h
        include/Tracking.h
        include/LocalMapping.h
        include/LoopClosing.h
        include/ORBextractor.h
        include/ORBmatcher.h
        include/FrameDrawer.h
        include/Converter.h
        include/MapPoint.h
        include/KeyFrame.h
        include/Atlas.h
        include/Map.h
        include/MapDrawer.h
        include/Optimizer.h
        include/Frame.h
        include/KeyFrameDatabase.h
        include/Sim3Solver.h
        include/Viewer.h
        include/ImuTypes.h
        include/G2oTypes.h
        include/CameraModels/GeometricCamera.h
        include/CameraModels/Pinhole.h
        include/CameraModels/KannalaBrandt8.h
        include/OptimizableTypes.h
        include/MLPnPsolver.h
        include/GeometricTools.h
        include/TwoViewReconstruction.h
        include/SerializationUtils.h
        include/Config.h
        include/Settings.h)

add_subdirectory(Thirdparty/g2o)

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        ${PCL_LIBRARIES}
        -lboost_serialization
        -lcrypto
        yolo_infer
)
# Build examples

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
        Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})
