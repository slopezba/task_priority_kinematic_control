cmake_minimum_required(VERSION 3.8)
project(task_priority_kinematic_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(controller_interface REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(Threads REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(interactive_markers REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sura_msgs REQUIRED)
find_package(urdf REQUIRED)
find_package(visualization_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ControllerOutput.msg"
  "msg/TaskState.msg"
  "msg/TaskStatus.msg"
  "msg/HierarchyState.msg"
  "msg/SolverDiagnostics.msg"
  "srv/SetTaskEnabled.srv"
  "srv/SetTaskDisabled.srv"
  "srv/ReorderTasks.srv"
  "srv/ListTasks.srv"
  "srv/SwitchBackend.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

set(core_target ${PROJECT_NAME}_core)

add_library(${core_target} SHARED
  src/core/whole_body_model.cpp
  src/core/hierarchy_solver.cpp
  src/core/task_manager.cpp
  src/core/task_priority_controller.cpp
  src/core/runtime_node.cpp
  src/kinematics/kdl_kinematics_backend.cpp
  src/kinematics/pinocchio_kinematics_backend.cpp
  src/tasks/task_base.cpp
  src/tasks/end_effector_position_task.cpp
  src/tasks/end_effector_orientation_task.cpp
  src/tasks/end_effector_pose_task.cpp
  src/tasks/frame_pose_task.cpp
  src/tasks/joint_nominal_task.cpp
  src/tasks/joint_limits_task.cpp
  src/tasks/base_yaw_task.cpp
)

set_target_properties(${core_target} PROPERTIES OUTPUT_NAME ${PROJECT_NAME})

target_include_directories(${core_target} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${core_target}
  Eigen3::Eigen
)

ament_target_dependencies(${core_target}
  controller_interface
  geometry_msgs
  hardware_interface
  kdl_parser
  pluginlib
  rclcpp
  realtime_tools
  sensor_msgs
  std_msgs
  std_srvs
  sura_msgs
  urdf
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${core_target} ${cpp_typesupport_target})

add_executable(task_priority_runtime src/main.cpp)
target_link_libraries(task_priority_runtime ${core_target})
ament_target_dependencies(task_priority_runtime rclcpp)

add_executable(task_priority_target_markers src/task_priority_target_markers.cpp)
ament_target_dependencies(task_priority_target_markers
  geometry_msgs
  interactive_markers
  rclcpp
  visualization_msgs
)

pluginlib_export_plugin_description_file(controller_interface task_priority_kinematic_control_plugins.xml)
pluginlib_export_plugin_description_file(task_priority_kinematic_control task_priority_kinematic_control_core_plugins.xml)

install(
  TARGETS ${core_target} task_priority_runtime task_priority_target_markers
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES
    task_priority_kinematic_control_plugins.xml
    task_priority_kinematic_control_core_plugins.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  controller_interface
  geometry_msgs
  hardware_interface
  interactive_markers
  kdl_parser
  pluginlib
  rclcpp
  realtime_tools
  sensor_msgs
  std_msgs
  std_srvs
  sura_msgs
  urdf
  visualization_msgs
  rosidl_default_runtime
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_hierarchy_solver test/test_hierarchy_solver.cpp)
  if(TARGET test_hierarchy_solver)
    target_link_libraries(test_hierarchy_solver ${core_target})
    ament_target_dependencies(test_hierarchy_solver rclcpp)
  endif()
endif()

ament_package()
