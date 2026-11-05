#include "task_priority_kinematic_control/tasks/end_effectors_relative_pose_task.hpp"

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

  FrameState get_frame_state(const std::string & frame_id) const override
  {
    return frames_.at(frame_id);
  }

  Eigen::Isometry3d get_relative_transform(
    const std::string & from_frame,
    const std::string & to_frame) const override
  {
    return get_frame_state(from_frame).pose.inverse() * get_frame_state(to_frame).pose;
  }

  const std::vector<CollisionCapsule> & collision_capsules() const override { return capsules_; }
  std::string name() const override { return "mock"; }

  std::map<std::string, FrameState> frames_;

private:
  std::vector<CollisionCapsule> capsules_;
};

std::shared_ptr<WholeBodyModel> model()
{
  auto out = std::make_shared<WholeBodyModel>();
  out->configure("world", "base", "left_tip", "right_tip", {"left_joint"}, {"right_joint"}, {});
  return out;
}

std::map<std::string, rclcpp::Parameter> task_parameters(
  const std::vector<double> & activation = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0})
{
  return {
    {"plugin", rclcpp::Parameter("plugin", "task_priority_kinematic_control/EndEffectorsRelativePoseTask")},
    {"enabled", rclcpp::Parameter("enabled", true)},
    {"priority", rclcpp::Parameter("priority", 1.0)},
    {"group", rclcpp::Parameter("group", "bimanual")},
    {"reference_frame", rclcpp::Parameter("reference_frame", "left_tip")},
    {"controlled_frame", rclcpp::Parameter("controlled_frame", "right_tip")},
    {"gain", rclcpp::Parameter("gain", std::vector<double>{2.0, 2.0, 2.0, 1.0, 1.0, 1.0})},
    {"activation", rclcpp::Parameter("activation", activation)},
    {"default_relative_pose", rclcpp::Parameter(
      "default_relative_pose",
      std::vector<double>{2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0})},
  };
}

MockBackend backend()
{
  MockBackend out;

  FrameState left;
  left.pose = Eigen::Isometry3d::Identity();
  left.jacobian = Eigen::MatrixXd::Zero(6, 2);
  left.jacobian(0, 0) = 1.0;
  left.jacobian(5, 0) = 1.0;

  FrameState right;
  right.pose = Eigen::Isometry3d::Identity();
  right.pose.translation() = Eigen::Vector3d(1.0, 0.0, 0.0);
  right.jacobian = Eigen::MatrixXd::Zero(6, 2);
  right.jacobian(0, 1) = 1.0;
  right.jacobian(5, 1) = 1.0;

  out.frames_["left_tip"] = left;
  out.frames_["right_tip"] = right;
  return out;
}

}  // namespace

TEST(EndEffectorsRelativePoseTask, FullActivationReturnsSixRows)
{
  EndEffectorsRelativePoseTask task;
  task.configure(
    "EndEffectors_Relative_pose",
    "task_priority_kinematic_control/EndEffectorsRelativePoseTask",
    task_parameters(),
    TaskContext{model()});

  WholeBodyState state;
  const auto output = task.update(state, backend());

  ASSERT_TRUE(output.active);
  ASSERT_EQ(output.error.size(), 6);
  ASSERT_EQ(output.desired_velocity.size(), 6);
  ASSERT_EQ(output.jacobian.rows(), 6);
  ASSERT_EQ(output.jacobian.cols(), 2);
  EXPECT_NEAR(output.error(0), 1.0, 1e-9);
  EXPECT_NEAR(output.desired_velocity(0), 2.0, 1e-9);
}

TEST(EndEffectorsRelativePoseTask, ReportsPoseTargetType)
{
  EndEffectorsRelativePoseTask task;
  task.configure(
    "EndEffectors_Relative_pose",
    "task_priority_kinematic_control/EndEffectorsRelativePoseTask",
    task_parameters(),
    TaskContext{model()});

  const auto status = task.build_status();

  EXPECT_EQ(status.id, "EndEffectors_Relative_pose");
  EXPECT_EQ(status.target_type, "pose");
}

TEST(EndEffectorsRelativePoseTask, PartialActivationReturnsSelectedRows)
{
  EndEffectorsRelativePoseTask task;
  task.configure(
    "EndEffectors_Relative_pose",
    "task_priority_kinematic_control/EndEffectorsRelativePoseTask",
    task_parameters({1.0, 0.0, 1.0, 0.0, 0.0, 1.0}),
    TaskContext{model()});

  WholeBodyState state;
  const auto output = task.update(state, backend());

  ASSERT_TRUE(output.active);
  ASSERT_EQ(output.error.size(), 3);
  ASSERT_EQ(output.jacobian.rows(), 3);
  ASSERT_EQ(output.jacobian.cols(), 2);
  EXPECT_NEAR(output.error(0), 1.0, 1e-9);
  EXPECT_NEAR(output.error(1), 0.0, 1e-9);
  EXPECT_NEAR(output.error(2), 0.0, 1e-9);
}

TEST(EndEffectorsRelativePoseTask, AcceptsIntegerActivationValues)
{
  auto params = task_parameters();
  params["activation"] = rclcpp::Parameter(
    "activation",
    std::vector<int64_t>{1, 0, 0, 0, 0, 1});

  EndEffectorsRelativePoseTask task;
  task.configure(
    "EndEffectors_Relative_pose",
    "task_priority_kinematic_control/EndEffectorsRelativePoseTask",
    params,
    TaskContext{model()});

  WholeBodyState state;
  const auto output = task.update(state, backend());

  ASSERT_TRUE(output.active);
  ASSERT_EQ(output.error.size(), 2);
  ASSERT_EQ(output.jacobian.rows(), 2);
}

TEST(EndEffectorsRelativePoseTask, NoActiveAxesIsInactive)
{
  EndEffectorsRelativePoseTask task;
  task.configure(
    "EndEffectors_Relative_pose",
    "task_priority_kinematic_control/EndEffectorsRelativePoseTask",
    task_parameters({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
    TaskContext{model()});

  WholeBodyState state;
  const auto output = task.update(state, backend());

  EXPECT_FALSE(output.active);
  EXPECT_EQ(output.status_message, "no active axes");
}

TEST(EndEffectorsRelativePoseTask, RejectsInvalidGain)
{
  EndEffectorsRelativePoseTask task;
  task.configure(
    "EndEffectors_Relative_pose",
    "task_priority_kinematic_control/EndEffectorsRelativePoseTask",
    task_parameters(),
    TaskContext{model()});

  std::string message;
  EXPECT_FALSE(task.set_gain({1.0, 1.0}, message));
  EXPECT_FALSE(task.set_gain({1.0, 1.0, 1.0, 1.0, 1.0, -1.0}, message));
  EXPECT_TRUE(task.set_gain({1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, message));
}

TEST(EndEffectorsRelativePoseTask, DisabledTaskIsInactive)
{
  auto params = task_parameters();
  params["enabled"] = rclcpp::Parameter("enabled", false);

  EndEffectorsRelativePoseTask task;
  task.configure(
    "EndEffectors_Relative_pose",
    "task_priority_kinematic_control/EndEffectorsRelativePoseTask",
    params,
    TaskContext{model()});

  WholeBodyState state;
  const auto output = task.update(state, backend());

  EXPECT_FALSE(output.active);
  EXPECT_EQ(output.status_message, "disabled");
}

}  // namespace task_priority_kinematic_control
