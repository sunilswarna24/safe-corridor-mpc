cmake_minimum_required(VERSION 3.16)
project(safe_corridor_mpc)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Core library lives in ../../cpp (FetchContent OSQP + Eigen there).
set(SAFE_CORRIDOR_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../cpp")
if(NOT TARGET safe_corridor)
  add_subdirectory("${SAFE_CORRIDOR_CPP_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/safe_corridor_cpp")
endif()

add_executable(mpc_node src/mpc_node.cpp)
target_link_libraries(mpc_node PRIVATE safe_corridor)
ament_target_dependencies(mpc_node rclcpp geometry_msgs nav_msgs std_msgs)

install(TARGETS mpc_node DESTINATION lib/${PROJECT_NAME})
ament_package()
