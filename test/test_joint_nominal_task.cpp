#include "task_priority_kinematic_control/tasks/joint_nominal_task.hpp"

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
    {"plugin", rclcpp::Parameter("plugin", "task_priority_kinematic_control/JointNominalTask")},
    {"enabled", rclcpp::Parameter("enabled", true)},
    {"priority", rclcpp::Parameter("priority", 1.0)},
    {"group", rclcpp::Parameter("group", "left")},
    {"joint_names", rclcpp::Parameter("joint_names", std::vector<std::string>{"joint_a", "joint_b", "joint_c"})},
    {"target", rclcpp::Parameter("target", std::vector<double>{1.0, 2.0, 3.0})},
    {"gain", rclcpp::Parameter("gain", std::vector<double>{0.5, 0.5, 0.5})},
    {"activation", rclcpp::Parameter("activation", std::vector<bool>{true, false, true})},
  };
}

}  // namespace

TEST(JointNominalTask, ActivationMaskRemovesInactiveJointRows)
{
  auto model = std::make_shared<WholeBodyModel>();
  model->configure("world", "base", "left_tip", "right_tip", {"joint_a", "joint_b", "joint_c"}, {}, {});

  JointNominalTask task;
  task.configure(
    "joint_nominal",
    "task_priority_kinematic_control/JointNominalTask",
    task_parameters(),
    TaskContext{model});

  WholeBodyState state;
  state.joints_valid = true;
  state.joint_positions = Eigen::Vector3d(0.0, 0.0, 1.0);
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
  EXPECT_DOUBLE_EQ(output.desired_velocity(0), 0.5);
  EXPECT_DOUBLE_EQ(output.desired_velocity(1), 1.0);

  auto status = task.build_status();
  ASSERT_EQ(status.joint_activation.size(), 3U);
  EXPECT_TRUE(status.joint_activation[0]);
  EXPECT_FALSE(status.joint_activation[1]);
  EXPECT_TRUE(status.joint_activation[2]);
}

TEST(JointNominalTask, RuntimeActivationCanDisableAllJointRows)
{
  auto model = std::make_shared<WholeBodyModel>();
  model->configure("world", "base", "left_tip", "right_tip", {"joint_a", "joint_b", "joint_c"}, {}, {});

  JointNominalTask task;
  task.configure(
    "joint_nominal",
    "task_priority_kinematic_control/JointNominalTask",
    task_parameters(),
    TaskContext{model});

  std::string message;
  ASSERT_TRUE(task.set_joint_activation({false, false, false}, message));

  WholeBodyState state;
  state.joints_valid = true;
  state.joint_positions = Eigen::Vector3d::Zero();
  state.joint_velocities = Eigen::Vector3d::Zero();

  MockBackend backend;
  const auto output = task.update(state, backend);

  EXPECT_FALSE(output.active);
  EXPECT_EQ(output.status_message, "no_active_joints");
}

}  // namespace task_priority_kinematic_control
