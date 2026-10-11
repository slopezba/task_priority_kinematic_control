#include "task_priority_kinematic_control/tasks/joint_limits_task.hpp"

#include <gtest/gtest.h>
#include "task_priority_kinematic_control/core/hierarchy_solver.hpp"

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

std::shared_ptr<JointLimitsTask> make_task(const std::shared_ptr<WholeBodyModel> & model)
{
  auto task = std::make_shared<JointLimitsTask>();
  task->configure(
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
  const auto output = task->update(state, backend);

  EXPECT_FALSE(output.active);
  EXPECT_EQ(output.status_message, "limits_inactive");
  EXPECT_EQ(output.jacobian.rows(), 3);
  EXPECT_DOUBLE_EQ(output.jacobian.norm(), 0);
  EXPECT_EQ(output.desired_velocity.size(), 3);
  EXPECT_DOUBLE_EQ(output.desired_velocity.norm(), 0);
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
  const auto output = task->update(state, backend);

  ASSERT_TRUE(output.active);
  ASSERT_EQ(output.jacobian.rows(), 3);
  ASSERT_EQ(output.jacobian.cols(), 3);
  EXPECT_DOUBLE_EQ(output.jacobian(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(output.jacobian(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(output.jacobian(0, 2), 0.0);
  EXPECT_DOUBLE_EQ(output.jacobian(2, 0), 0.0);
  EXPECT_DOUBLE_EQ(output.jacobian(2, 1), 0.0);
  EXPECT_DOUBLE_EQ(output.jacobian(2, 2), 1.0);
  EXPECT_NEAR(output.desired_velocity(0), 0.05, 1e-9);
  EXPECT_NEAR(output.desired_velocity(2), -0.05, 1e-9);
  EXPECT_DOUBLE_EQ(output.jacobian.row(1).norm(), 0);
  EXPECT_DOUBLE_EQ(output.error(1), 0);
  EXPECT_DOUBLE_EQ(output.desired_velocity(1), 0);
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
  auto output = task->update(state, backend);
  ASSERT_TRUE(output.active);

  state.joint_positions = Eigen::Vector3d(0.12, 0.5, 0.5);
  output = task->update(state, backend);
  ASSERT_TRUE(output.active);
  ASSERT_EQ(output.jacobian.rows(), 3);
  EXPECT_NEAR(output.desired_velocity(0), 0.015, 1e-9);

  state.joint_positions = Eigen::Vector3d(0.16, 0.5, 0.5);
  output = task->update(state, backend);
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
  auto output = task->update(state, backend);
  ASSERT_TRUE(output.active);

  state.joint_positions = Eigen::Vector3d(0.14995, 0.5, 0.5);
  output = task->update(state, backend);
  EXPECT_FALSE(output.active);
  EXPECT_EQ(output.status_message, "limits_inactive");
}

TEST(JointLimitsTask, PreparedOutputsKeepStorageWhileActiveJointsAndStatusesChange)
{
  auto model = std::make_shared<WholeBodyModel>();
  model->configure("world", "base", "left_tip", "right_tip", {"joint_a", "joint_b", "joint_c"}, {}, {0, 1});
  auto task = make_task(model);
  TaskComputation output;
  output.has_frame_pose = true; output.frame_id = "previous_task_frame";
  task->prepare_computation(output);
  EXPECT_FALSE(output.has_frame_pose); EXPECT_TRUE(output.frame_id.empty());
  const double * jacobian = output.jacobian.data();
  const double * error = output.error.data();
  const double * velocity = output.desired_velocity.data();
  const char * status = output.status_message.data();
  const size_t capacity = output.status_message.capacity();
  WholeBodyState state; state.joint_positions = Eigen::Vector3d::Constant(.5);
  MockBackend backend;
  for (int iteration = 0; iteration < 10000; ++iteration) {
    state.joint_positions.setConstant(.5);
    if (iteration % 3 == 0) {state.joint_positions(0) = .05;}
    if (iteration % 3 == 1) {state.joint_positions(2) = .95;}
    task->update_into(state, backend, output);
    EXPECT_EQ(output.jacobian.data(), jacobian); EXPECT_EQ(output.error.data(), error);
    EXPECT_EQ(output.desired_velocity.data(), velocity);
    EXPECT_EQ(output.status_message.data(), status); EXPECT_EQ(output.status_message.capacity(), capacity);
    ASSERT_EQ(output.jacobian.rows(), 3); ASSERT_EQ(output.jacobian.cols(), 5);
    EXPECT_DOUBLE_EQ(output.jacobian.leftCols(2).norm(), 0);
    for (int joint = 0; joint < 3; ++joint) {
      const bool active = (iteration % 3 == 0 && joint == 0) || (iteration % 3 == 1 && joint == 2);
      EXPECT_DOUBLE_EQ(output.jacobian(joint, 2 + joint), active ? 1 : 0);
      if (!active) {
        EXPECT_DOUBLE_EQ(output.jacobian.row(joint).norm(), 0);
        EXPECT_DOUBLE_EQ(output.error(joint), 0); EXPECT_DOUBLE_EQ(output.desired_velocity(joint), 0);
      }
    }
  }
  for (const Eigen::Index count : {0, 2, 4}) {
    state.joint_positions.setConstant(count, .05);
    task->update_into(state, backend, output);
    EXPECT_EQ(output.jacobian.data(), jacobian); EXPECT_EQ(output.error.data(), error);
    EXPECT_EQ(output.desired_velocity.data(), velocity); EXPECT_EQ(output.status_message.data(), status);
    EXPECT_FALSE(output.active); EXPECT_FALSE(output.stop_arm_motion);
    EXPECT_DOUBLE_EQ(output.jacobian.norm(), 0); EXPECT_DOUBLE_EQ(output.error.norm(), 0);
    EXPECT_DOUBLE_EQ(output.desired_velocity.norm(), 0);
    EXPECT_EQ(output.status_message, count == 0 ? "waiting_for_joints" : "invalid_joint_state");
  }
  state.joint_positions.setConstant(3, .05);
  task->set_enabled(false); task->update_into(state, backend, output);
  EXPECT_EQ(output.status_message, "disabled"); EXPECT_FALSE(output.active);
  EXPECT_DOUBLE_EQ(output.jacobian.norm(), 0); EXPECT_EQ(output.jacobian.data(), jacobian);
}

TEST(JointLimitsTask, ReleasedRowsRestoreLowerTaskWithOtherLimitsStillActive)
{
  auto model = std::make_shared<WholeBodyModel>();
  model->configure("world", "base", "left_tip", "right_tip", {"joint_a", "joint_b", "joint_c"}, {}, {});
  auto task = make_task(model);
  MockBackend backend; WholeBodyState state; state.joint_positions = Eigen::Vector3d(.05, .5, .95);
  TaskComputation output; task->prepare_computation(output); task->update_into(state, backend, output);
  TaskComputation lower;
  lower.active = true; lower.jacobian = Eigen::MatrixXd::Identity(3, 3);
  lower.desired_velocity = Eigen::VectorXd::Constant(3, .2);
  HierarchySolver solver; solver.configure(3); solver.set_method(SolverMethod::kPinv);
  auto command = solver.solve({output, lower});
  EXPECT_NEAR(command.generalized_velocity(0), .05, 1e-12);
  EXPECT_NEAR(command.generalized_velocity(1), .2, 1e-12);
  EXPECT_NEAR(command.generalized_velocity(2), -.05, 1e-12);
  state.joint_positions(0) = .14995; task->update_into(state, backend, output);
  EXPECT_TRUE(output.active); EXPECT_DOUBLE_EQ(output.jacobian.row(0).norm(), 0);
  EXPECT_DOUBLE_EQ(output.desired_velocity(0), 0); EXPECT_DOUBLE_EQ(output.error(0), 0);
  command = solver.solve({output, lower});
  EXPECT_NEAR(command.generalized_velocity(0), .2, 1e-12);
  EXPECT_NEAR(command.generalized_velocity(2), -.05, 1e-12);
  task->set_enabled(false); task->update_into(state, backend, output);
  command = solver.solve({output, lower});
  EXPECT_NEAR((command.generalized_velocity - lower.desired_velocity).norm(), 0, 1e-12);
}

TEST(JointLimitsTask, UpperHysteresisAndLowerPrecedenceRemainUnchanged)
{
  auto model = std::make_shared<WholeBodyModel>();
  model->configure("world", "base", "left_tip", "right_tip", {"joint_a", "joint_b", "joint_c"}, {}, {});
  auto task = make_task(model); MockBackend backend;
  WholeBodyState state; state.joint_positions = Eigen::Vector3d(.5, .5, .95);
  auto output = task->update(state, backend); ASSERT_TRUE(output.active);
  state.joint_positions(2) = .88; output = task->update(state, backend);
  EXPECT_TRUE(output.active); EXPECT_NEAR(output.desired_velocity(2), -.015, 1e-12);
  state.joint_positions(2) = .85005; output = task->update(state, backend);
  EXPECT_FALSE(output.active); EXPECT_DOUBLE_EQ(output.jacobian.norm(), 0);
  auto params = task_parameters();
  params["upper_limits"] = rclcpp::Parameter("upper_limits", std::vector<double>{.18, 1, 1});
  task->configure("joint_limits", "task_priority_kinematic_control/JointLimitsTask", params, TaskContext{model});
  state.joint_positions = Eigen::Vector3d(.09, .5, .5);
  output = task->update(state, backend); ASSERT_TRUE(output.active);
  EXPECT_NEAR(output.desired_velocity(0), .03, 1e-12);
  std::string message; ASSERT_TRUE(task->set_gain_scalar(0, message));
  output = task->update(state, backend);
  EXPECT_FALSE(output.active); EXPECT_DOUBLE_EQ(output.jacobian.norm(), 0);
  EXPECT_DOUBLE_EQ(output.desired_velocity.norm(), 0); EXPECT_DOUBLE_EQ(output.error.norm(), 0);
}

}  // namespace task_priority_kinematic_control
