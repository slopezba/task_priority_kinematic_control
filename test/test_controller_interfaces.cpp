#include "task_priority_kinematic_control/core/task_priority_controller.hpp"

#include <gtest/gtest.h>
#include <chrono>
#include <future>
#include <thread>

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

// Exercise the real action callbacks without requiring hardware or a kinematics plugin.
class MinimalTrajectoryController : public TaskPriorityController
{
public:
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State &) override
  {return controller_interface::CallbackReturn::SUCCESS;}
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State &) override
  {return controller_interface::CallbackReturn::SUCCESS;}
};

class TrajectoryActionReadinessTest : public ::testing::Test
{
protected:
  using Action = control_msgs::action::FollowJointTrajectory;
  using Handle = rclcpp_action::ClientGoalHandle<Action>;

  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void SetUp() override
  {
    controller_ = std::make_unique<MinimalTrajectoryController>();
    ASSERT_EQ(controller_->init("trajectory_readiness", "/readiness_case"),
      controller_interface::return_type::OK);
    const auto node = controller_->get_node();
    node->set_parameter(rclcpp::Parameter("task_ids", std::vector<std::string>{"trajectory"}));
    controller_->declare_task_parameters();
    node->set_parameters({
      rclcpp::Parameter("tasks.trajectory.plugin", "task_priority_kinematic_control/JointTrajectoryTask"),
      rclcpp::Parameter("tasks.trajectory.enabled", true),
      rclcpp::Parameter("tasks.trajectory.joint_names", std::vector<std::string>{"joint_a", "joint_b"})});
    auto model = std::make_shared<WholeBodyModel>();
    model->configure("world", "base", "left_tip", "right_tip", {"joint_a", "joint_b"}, {}, {});
    controller_->task_manager_ = std::make_unique<TaskManager>(
      node->get_node_parameters_interface(), node->get_logger());
    controller_->task_manager_->configure(TaskContext{model});
    controller_->configure_external_interfaces();
    client_node_ = std::make_shared<rclcpp::Node>("trajectory_readiness_client");
    client_ = rclcpp_action::create_client<Action>(client_node_,
      "/readiness_case/task_priority/tasks/trajectory/follow_joint_trajectory");
    executor_.add_node(node->get_node_base_interface());
    executor_.add_node(client_node_);
    spin_ = std::thread([this]() {executor_.spin();});
    ASSERT_TRUE(client_->wait_for_action_server(std::chrono::seconds(2)));
  }

  void TearDown() override
  {
    controller_->stop_task_trajectory_goals("Test cleanup");
    executor_.cancel();
    if (spin_.joinable()) {spin_.join();}
    executor_.remove_node(controller_->get_node()->get_node_base_interface());
    executor_.remove_node(client_node_);
  }

  void activate()
  {
    if (controller_->get_state().label() == "unconfigured") {controller_->get_node()->configure();}
    controller_->get_node()->activate();
    std::scoped_lock lock(controller_->task_mutex_);
    controller_->trajectory_execution_ready_ = true;
  }

  void enabled(bool value)
  {
    std::string message;
    std::scoped_lock lock(controller_->task_mutex_);
    ASSERT_TRUE(controller_->task_manager_->set_task_enabled("trajectory", value, message));
  }

  JointTrajectoryTaskStatus status()
  {
    std::scoped_lock lock(controller_->task_mutex_);
    return controller_->task_manager_->get_task_joint_trajectory_status("trajectory");
  }

  bool task_enabled()
  {
    std::scoped_lock lock(controller_->task_mutex_);
    return controller_->trajectory_task_enabled_locked("trajectory");
  }

  std::shared_ptr<Handle> send()
  {
    Action::Goal goal;
    goal.trajectory.joint_names = {"joint_a", "joint_b"};
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = {0.0, 0.0};
    goal.trajectory.points.push_back(point);
    point.positions = {1.0, 1.0};
    point.time_from_start.sec = 10;
    goal.trajectory.points.push_back(point);
    auto future = client_->async_send_goal(goal);
    if (future.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {return nullptr;}
    const auto handle = future.get();
    if (handle) {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (!status().accepted && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    }
    return handle;
  }

  std::unique_ptr<MinimalTrajectoryController> controller_;
  rclcpp::Node::SharedPtr client_node_;
  rclcpp_action::Client<Action>::SharedPtr client_;
  rclcpp::executors::MultiThreadedExecutor executor_;
  std::thread spin_;
};

TEST_F(TrajectoryActionReadinessTest, InactiveControllerRejectsWithoutStoringTrajectory)
{
  EXPECT_FALSE(send());
  EXPECT_FALSE(status().accepted);
}

TEST_F(TrajectoryActionReadinessTest, DisabledTaskRejectsAndLaterEnableHasNoLatentTrajectory)
{
  activate();
  enabled(false);
  EXPECT_FALSE(send());
  enabled(true);
  EXPECT_FALSE(status().accepted);
  EXPECT_FALSE(status().active);
}

TEST_F(TrajectoryActionReadinessTest, SecondGoalRejectedAndOwnerCancellationClearsTrajectory)
{
  activate();
  const auto first = send();
  ASSERT_TRUE(first);
  EXPECT_FALSE(send());
  auto cancel = client_->async_cancel_goal(first);
  ASSERT_EQ(cancel.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  auto result = client_->async_get_result(first);
  ASSERT_EQ(result.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  EXPECT_EQ(result.get().code, rclcpp_action::ResultCode::CANCELED);
  EXPECT_FALSE(status().accepted);
}

TEST_F(TrajectoryActionReadinessTest, DisablingAnAcceptedGoalAbortsAndReenableDoesNotResume)
{
  activate();
  const auto handle = send();
  ASSERT_TRUE(handle);
  enabled(false);
  enabled(true);
  auto result = client_->async_get_result(handle);
  ASSERT_EQ(result.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  EXPECT_EQ(result.get().code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_FALSE(status().accepted);
}

TEST_F(TrajectoryActionReadinessTest, DeactivationAbortsWithoutChangingTaskEnabled)
{
  activate();
  const auto handle = send();
  ASSERT_TRUE(handle);
  controller_->get_node()->deactivate();
  auto result = client_->async_get_result(handle);
  ASSERT_EQ(result.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  EXPECT_EQ(result.get().code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_FALSE(status().accepted);
  EXPECT_TRUE(task_enabled());
  activate();
  EXPECT_FALSE(status().active);
  EXPECT_FALSE(status().accepted);
}
}  // namespace task_priority_kinematic_control
