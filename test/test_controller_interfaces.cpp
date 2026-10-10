#include "task_priority_kinematic_control/core/task_priority_controller.hpp"

#include <gtest/gtest.h>

namespace task_priority_kinematic_control
{
class ControllerInterfacesTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void SetUp() override
  {
    controller_ = std::make_unique<TaskPriorityController>();
    ASSERT_EQ(controller_->init("interface_test"), controller_interface::return_type::OK);
    controller_->get_node()->set_parameters({
      rclcpp::Parameter("left_arm_joints", std::vector<std::string>{"left_joint"}),
      rclcpp::Parameter("right_arm_joints", std::vector<std::string>{"right_joint"}),
      rclcpp::Parameter("body_velocity_controller_name", "vehicle_velocity")});
  }
  std::unique_ptr<TaskPriorityController> controller_;
};

TEST_F(ControllerInterfacesTest, DefaultModeRetainsVehicleAndArmInterfaces)
{
  EXPECT_EQ(controller_->command_interface_configuration().names, (std::vector<std::string>{
    "vehicle_velocity/linear.x", "vehicle_velocity/linear.y", "vehicle_velocity/linear.z",
    "vehicle_velocity/angular.x", "vehicle_velocity/angular.y", "vehicle_velocity/angular.z",
    "left_joint/velocity", "right_joint/velocity"}));
}

TEST_F(ControllerInterfacesTest, TopicModeClaimsOnlyArmHardware)
{
  controller_->get_node()->set_parameter(rclcpp::Parameter("base_command_mode", "topic"));
  EXPECT_EQ(controller_->command_interface_configuration().names,
    (std::vector<std::string>{"left_joint/velocity", "right_joint/velocity"}));
  EXPECT_EQ(controller_->state_interface_configuration().names,
    (std::vector<std::string>{"left_joint/position", "right_joint/position",
      "left_joint/velocity", "right_joint/velocity"}));
}
}  // namespace task_priority_kinematic_control
