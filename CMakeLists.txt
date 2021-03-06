cmake_minimum_required(VERSION 3.10)

option(USE_CUDA "enable CUDA" ON)
option(GUI_VIEWER "enable GUI viewer" OFF)

project(impala CXX)

find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(Boost COMPONENTS python3 numpy3 filesystem system regex REQUIRED)

if(${USE_CUDA})
    find_package(CUDA REQUIRED)
endif()

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wextra -pedantic -Wconversion -Wcast-qual")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

set(impala_source
    main.cpp
    envs/g2048/g2048_env.cpp)

if(${GUI_VIEWER})
    set(impala_source
        ${impala_source}
        viewer/texture.cpp
        viewer/gl_util.cpp
        viewer/load_png.cpp)
endif()

if(${USE_CUDA})
    cuda_add_executable(train2048 ${impala_source} cuda/cuda_memory.cu)
else()
    add_executable(train2048 ${impala_source})
endif()

target_include_directories(train2048 PRIVATE .)
target_include_directories(train2048 SYSTEM PRIVATE ./range-v3/include)
target_include_directories(train2048 SYSTEM PRIVATE ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(train2048 ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} Threads::Threads)

if(${USE_CUDA})
    target_compile_definitions(train2048 PRIVATE IMPALA_USE_CUDA)
endif()

if(${GUI_VIEWER})
    target_compile_definitions(train2048 PRIVATE IMPALA_USE_GUI_VIEWER)
    target_link_libraries(train2048 glfw GL png)
endif()
