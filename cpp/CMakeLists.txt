cmake_minimum_required(VERSION 3.16)
# OSQP 0.6.x still declares an old cmake_minimum_required; allow on CMake 4+.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(safe_corridor_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
  osqp
  GIT_REPOSITORY https://github.com/osqp/osqp.git
  GIT_TAG v0.6.3
  GIT_SHALLOW TRUE
)
set(UNITTESTS OFF CACHE BOOL "" FORCE)
set(PROFILING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(osqp)

add_library(safe_corridor
  src/plant.cpp
  src/mpc.cpp
  src/gate.cpp
)
target_include_directories(safe_corridor
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${eigen_SOURCE_DIR}
)
target_link_libraries(safe_corridor PUBLIC osqpstatic)

add_executable(demo_line src/demo_line.cpp)
target_link_libraries(demo_line PRIVATE safe_corridor)

install(TARGETS safe_corridor demo_line
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
