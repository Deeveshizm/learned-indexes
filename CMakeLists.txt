cmake_minimum_required(VERSION 3.10)

# Project name
project(LearnedIndexBenchmark)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable more warnings (optional but recommended)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif (MSVC)
    add_compile_options(/W4)
endif()

# Add the executable (UPDATED FILE NAME HERE)
add_executable(learned_index_benchmark learned_index_prototype.cpp)

