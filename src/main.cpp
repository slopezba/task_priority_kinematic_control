#include "task_priority_kinematic_control/core/runtime_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<task_priority_kinematic_control::RuntimeNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
