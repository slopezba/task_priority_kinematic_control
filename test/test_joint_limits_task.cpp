#include "task_priority_kinematic_control/tasks/joint_limits_task.hpp"

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

std::map<std::string, rclcpp::Parameter> task_parameters()
{
  return {
    {"plugin", rclcpp::Parameter("plugin", "task_priority_kinematic_control/JointLimitsTask")},
    {"enabled", rclcpp::Parameter("enabled", true)},
    {"priority", rclcpp::Parameter("priority", 1.0)},
    {"group", rclcpp::Parameter("group", "global")},
    {"alpha", rclcpp::Parameter("alpha", 0.1)},
    {"delta", rclcpp::Parameter("delta", 0.15)},
    {"eps", rclcpp::Parameter("eps", 1e-4)},
    {"gain_scalar", rclcpp::Parameter("gain_scalar", 0.5)},
    {"lower_limits", rclcpp::Parameter("lower_limits", std::vector<double>{0.0, 0.0, 0.0})},
    {"upper_limits", rclcpp::Parameter("upper_limits", std::vector<double>{1.0, 1.0, 1.0})},
  };
}

JointLimitsTask make_task(const std::shared_ptr<WholeBodyModel> & model)
{
  JointLimitsTask task;
  task.configure(
    "joint_limits",
    "task_priority_kinematic_control/JointLimitsTask",
    task_parameters(),
    TaskContext{model});
  return task;
}

}  // namespace

TEST(JointLimitsTask, DoesNotConsumeJacobianWhenAllJointsAreSafe)
{
  auto model = std::make_shared<WholeBodyModel>();
  model->configure("world", "base", "left_tip", "right_tip", {"joint_a", "joint_b", "joint_c"}, {}, {});
  auto task = make_task(model);

  WholeBodyState state;
  state.joints_valid = true;
  state.joint_positions = Eigen::Vector3d(0.5, 0.5, 0.5);
  state.joint_velocities = Eigen::Vector3d::Zero();

  MockBackend backend;
  const auto output = task.update(state, backend);

  EXPECT_FALSE(output.active);
  EXPECT_EQ(output.status_message, "limits_inactive");
  EXPECT_EQ(output.jacobian.rows(), 0);
  EXPECT_EQ(output.desired_velocity.size(), 0);
}

TEST(JointLimitsTask, OnlyConsumesRowsForJointsInsideActivationMargin)
{
  auto model = std::make_shared<WholeBodyModel>();
  model->configure("world", "base", "left_tip", "right_tip", {"joint_a", "joint_b", "joint_c"}, {}, {});
  auto task = make_task(model);

  WholeBodyState state;
  state.joints_valid = true;
  state.joint_positions = Eigen::Vector3d(0.05, 0.5, 0.95);
  state.joint_velocities = Eigen::Vector3d::Zero();

  MockBackend backend;
  const auto output = task.update(state, backend);

  ASSERT_TRUE(output.active);
  ASSERT_EQ(output.jacobian.rows(), 2);
  ASSERT_EQ(output.jacobian.cols(), 3);
  EXPECT_DOUBLE_EQ(output.jacobian(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(output.jacobian(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(output.jacobian(0, 2), 0.0);
  EXPECT_DOUBLE_EQ(output.jacobian(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(output.jacobian(1, 1), 0.0);
  EXPECT_DOUBLE_EQ(output.jacobian(1, 2), 1.0);
  EXPECT_NEAR(output.desired_velocity(0), 0.05, 1e-9);
  EXPECT_NEAR(output.desired_velocity(1), -0.05, 1e-9);
}

TEST(JointLimitsTask, KeepsActiveUntilJointPassesDeactivationMargin)
{
  auto model = std::make_shared<WholeBodyModel>();
  model->configure("world", "base", "left_tip", "right_tip", {"joint_a", "joint_b", "joint_c"}, {}, {});
  auto task = make_task(model);

  WholeBodyState state;
  state.joints_valid = true;
  state.joint_positions = Eigen::Vector3d(0.05, 0.5, 0.5);
  state.joint_velocities = Eigen::Vector3d::Zero();

  MockBackend backend;
  auto output = task.update(state, backend);
  ASSERT_TRUE(output.active);

  state.joint_positions = Eigen::Vector3d(0.12, 0.5, 0.5);
  output = task.update(state, backend);
  ASSERT_TRUE(output.active);
  ASSERT_EQ(output.jacobian.rows(), 1);
  EXPECT_NEAR(output.desired_velocity(0), 0.015, 1e-9);

  state.joint_positions = Eigen::Vector3d(0.16, 0.5, 0.5);
  output = task.update(state, backend);
  EXPECT_FALSE(output.active);
  EXPECT_EQ(output.status_message, "limits_inactive");
}

TEST(JointLimitsTask, DeactivatesWhenJointReachesDeactivationThresholdWithinEpsilon)
{
  auto model = std::make_shared<WholeBodyModel>();
  model->configure("world", "base", "left_tip", "right_tip", {"joint_a", "joint_b", "joint_c"}, {}, {});
  auto task = make_task(model);

  WholeBodyState state;
  state.joints_valid = true;
  state.joint_positions = Eigen::Vector3d(0.05, 0.5, 0.5);
  state.joint_velocities = Eigen::Vector3d::Zero();

  MockBackend backend;
  auto output = task.update(state, backend);
  ASSERT_TRUE(output.active);

  state.joint_positions = Eigen::Vector3d(0.14995, 0.5, 0.5);
  output = task.update(state, backend);
  EXPECT_FALSE(output.active);
  EXPECT_EQ(output.status_message, "limits_inactive");
}

}  // namespace task_priority_kinematic_control
