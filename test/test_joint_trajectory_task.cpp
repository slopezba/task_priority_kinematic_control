#include "task_priority_kinematic_control/tasks/joint_trajectory_task.hpp"

#include <gtest/gtest.h>

namespace task_priority_kinematic_control
{
namespace
{
class MockBackend : public KinematicsBackend
{
public:
  void configure(const WholeBodyModel &, const std::string &, const rclcpp::Logger &) override {}
  void update(const WholeBodyState &) override {}
  FrameState get_frame_state(const std::string &) const override { return {}; }
  Eigen::Isometry3d get_relative_transform(const std::string &, const std::string &) const override
  {
    return Eigen::Isometry3d::Identity();
  }
  const std::vector<CollisionCapsule> & collision_capsules() const override { return capsules_; }
  std::string name() const override { return "mock"; }

private:
  std::vector<CollisionCapsule> capsules_;
};

builtin_interfaces::msg::Duration seconds(double value)
{
  builtin_interfaces::msg::Duration duration;
  duration.sec = static_cast<int32_t>(value);
  duration.nanosec = static_cast<uint32_t>((value - duration.sec) * 1.0e9);
  return duration;
}

std::map<std::string, rclcpp::Parameter> task_parameters()
{
  return {
    {"plugin", rclcpp::Parameter("plugin", "task_priority_kinematic_control/JointTrajectoryTask")},
    {"enabled", rclcpp::Parameter("enabled", true)},
    {"priority", rclcpp::Parameter("priority", 1.0)},
    {"group", rclcpp::Parameter("group", "left")},
    {"joint_names", rclcpp::Parameter("joint_names", std::vector<std::string>{"joint_a", "joint_b"})},
    {"gain", rclcpp::Parameter("gain", std::vector<double>{2.0, 2.0})},
    {"goal_tolerance", rclcpp::Parameter("goal_tolerance", std::vector<double>{0.01, 0.01})},
    {"trajectory_timeout", rclcpp::Parameter("trajectory_timeout", 0.5)},
    {"hold_last_point", rclcpp::Parameter("hold_last_point", true)},
  };
}

trajectory_msgs::msg::JointTrajectory valid_trajectory()
{
  trajectory_msgs::msg::JointTrajectory trajectory;
  trajectory.joint_names = {"joint_b", "joint_a"};

  trajectory_msgs::msg::JointTrajectoryPoint first;
  first.positions = {-1.0, 1.0};
  first.velocities = {-0.2, 0.2};
  first.time_from_start = seconds(0.0);
  trajectory.points.push_back(first);

  trajectory_msgs::msg::JointTrajectoryPoint second;
  second.positions = {-2.0, 2.0};
  second.velocities = {-0.1, 0.1};
  second.time_from_start = seconds(1.0);
  trajectory.points.push_back(second);
  return trajectory;
}
}  // namespace

TEST(JointTrajectoryTask, TracksInitialTrajectoryPointWithFeedforward)
{
  auto model = std::make_shared<WholeBodyModel>();
  model->configure("world", "base", "left_tip", "right_tip", {"joint_a", "joint_b"}, {}, {});

  JointTrajectoryTask task;
  task.configure(
    "left_joint_trajectory",
    "task_priority_kinematic_control/JointTrajectoryTask",
    task_parameters(),
    TaskContext{model});

  std::string message;
  ASSERT_TRUE(task.set_joint_trajectory(valid_trajectory(), message));

  WholeBodyState state;
  state.joints_valid = true;
  state.joint_positions = Eigen::VectorXd::Zero(2);
  state.joint_velocities = Eigen::VectorXd::Zero(2);
  MockBackend backend;
  const auto output = task.update(state, backend);

  ASSERT_TRUE(output.active);
  ASSERT_EQ(output.desired_velocity.size(), 2);
  EXPECT_NEAR(output.desired_velocity(0), 2.2, 0.1);
  EXPECT_NEAR(output.desired_velocity(1), -2.2, 0.1);
}

TEST(JointTrajectoryTask, RejectsNonIncreasingTimes)
{
  auto model = std::make_shared<WholeBodyModel>();
  model->configure("world", "base", "left_tip", "right_tip", {"joint_a", "joint_b"}, {}, {});

  JointTrajectoryTask task;
  task.configure(
    "left_joint_trajectory",
    "task_priority_kinematic_control/JointTrajectoryTask",
    task_parameters(),
    TaskContext{model});

  auto trajectory = valid_trajectory();
  trajectory.points[1].time_from_start = seconds(0.0);
  std::string message;
  EXPECT_FALSE(task.set_joint_trajectory(trajectory, message));
}

}  // namespace task_priority_kinematic_control
