#include "task_priority_kinematic_control/tasks/frame_pose_limits_task.hpp"
#include "task_priority_kinematic_control/core/hierarchy_solver.hpp"

#include <gtest/gtest.h>
#include <pluginlib/class_loader.hpp>

#include <cmath>

namespace task_priority_kinematic_control
{
namespace
{

class FrameBackend : public KinematicsBackend
{
public:
  void configure(const WholeBodyModel &, const std::string &, const rclcpp::Logger &) override {}
  void update(const WholeBodyState &) override {}
  FrameState get_frame_state(const std::string &) const override {return frame;}
  Eigen::Isometry3d get_relative_transform(const std::string &, const std::string &) const override
  {return Eigen::Isometry3d::Identity();}
  const std::vector<CollisionCapsule> & collision_capsules() const override {return capsules_;}
  std::string name() const override {return "frame_test";}

  FrameState frame;

private:
  std::vector<CollisionCapsule> capsules_;
};

std::map<std::string, rclcpp::Parameter> limits_parameters()
{
  return {
    {"enabled", rclcpp::Parameter("enabled", true)},
    {"priority", rclcpp::Parameter("priority", 15.0)},
    {"group", rclcpp::Parameter("group", "right")},
    {"frame_id", rclcpp::Parameter("frame_id", "tool")},
    {"axes", rclcpp::Parameter("axes", std::vector<bool>{true, true, true, false, false, true})},
    {"threshold", rclcpp::Parameter("threshold", std::vector<double>{0.1, 0.1, 0.1, 0.3, 0.3, 0.3})},
    {"activation_margin", rclcpp::Parameter("activation_margin",
      std::vector<double>{0.02, 0.02, 0.02, 0.05, 0.05, 0.05})},
    {"release_margin", rclcpp::Parameter("release_margin",
      std::vector<double>{0.04, 0.04, 0.04, 0.1, 0.1, 0.1})},
    {"eps", rclcpp::Parameter("eps", 1e-4)},
    {"gain", rclcpp::Parameter("gain", std::vector<double>{1.0, 2.0, 1.0, 1.0, 1.0, 0.5})},
    {"max_repulsive_velocity", rclcpp::Parameter("max_repulsive_velocity", 0.05)}};
}

geometry_msgs::msg::PoseStamped goal_at(
  const Eigen::Vector3d & position, const Eigen::Quaterniond & orientation = Eigen::Quaterniond::Identity(),
  const std::string & frame = "world")
{
  geometry_msgs::msg::PoseStamped goal;
  goal.header.frame_id = frame;
  goal.pose.position.x = position.x();
  goal.pose.position.y = position.y();
  goal.pose.position.z = position.z();
  goal.pose.orientation.x = orientation.x();
  goal.pose.orientation.y = orientation.y();
  goal.pose.orientation.z = orientation.z();
  goal.pose.orientation.w = orientation.w();
  return goal;
}

}  // namespace

class FramePoseLimitsTest : public ::testing::Test
{
protected:
  std::shared_ptr<WholeBodyModel> model = std::make_shared<WholeBodyModel>();
  FramePoseLimitsTask task;
  FrameBackend backend;
  WholeBodyState state;

  void SetUp() override
  {
    model->configure("world", "base", "left", "right", {"left_joint"}, {"right_joint"}, {0, 1, 2, 3, 4, 5});
    // Frame moves exactly with the six base DoFs; arm joints do not affect it.
    backend.frame.jacobian = Eigen::MatrixXd::Zero(6, model->total_dofs());
    backend.frame.jacobian.leftCols(6).setIdentity();
    backend.frame.pose.setIdentity();
    task.configure("frame_pose_limits", "task_priority_kinematic_control/FramePoseLimitsTask",
      limits_parameters(), TaskContext{model});
    ASSERT_TRUE(task.set_pose_goal(goal_at(Eigen::Vector3d::Zero())));
  }

  TaskComputation at(const Eigen::Vector3d & position, double yaw = 0.0)
  {
    backend.frame.pose.setIdentity();
    backend.frame.pose.translation() = position;
    backend.frame.pose.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    return task.update(state, backend);
  }
};

TEST_F(FramePoseLimitsTest, InsideBoxLeavesAllDegreesOfFreedomFree)
{
  const auto out = at({0.05, -0.05, 0.07}, 0.2);
  EXPECT_FALSE(out.active);
  EXPECT_TRUE(out.structural_zero_rows);
  EXPECT_EQ(out.jacobian.rows(), 4);  // x, y, z, rz
  EXPECT_TRUE(out.jacobian.isZero());
  EXPECT_TRUE(out.desired_velocity.isZero());
  EXPECT_EQ(out.status_message, "inside_bounds");

  TaskComputation lower;
  lower.active = true;
  lower.jacobian = Eigen::MatrixXd::Identity(model->total_dofs(), model->total_dofs());
  lower.desired_velocity = Eigen::VectorXd::Constant(model->total_dofs(), 0.01);
  HierarchySolver solver;
  solver.configure(model->total_dofs());
  solver.set_method(SolverMethod::kPinv);
  const auto command = solver.solve({out, lower});
  EXPECT_TRUE(command.generalized_velocity.isApprox(lower.desired_velocity, 1e-9));
}

TEST_F(FramePoseLimitsTest, OnlyTheAxisAtTheBoundIsActivatedWithInwardPush)
{
  // x error = target - current = -0.085, beyond threshold - alpha = 0.08.
  const auto out = at({0.085, 0.0, 0.0});
  ASSERT_TRUE(out.active);
  EXPECT_EQ(out.status_message, "holding_bounds");
  EXPECT_FALSE(out.jacobian.row(0).isZero());
  EXPECT_TRUE(out.jacobian.row(1).isZero());
  EXPECT_TRUE(out.jacobian.row(2).isZero());
  EXPECT_TRUE(out.jacobian.row(3).isZero());
  // Push back to the inner bound (current x -> 0.06): negative x velocity.
  EXPECT_NEAR(out.error(0), -0.025, 1e-12);
  EXPECT_NEAR(out.desired_velocity(0), -0.025, 1e-12);

  const auto other = at({0.0, -0.09, 0.0});
  ASSERT_TRUE(other.active);
  EXPECT_TRUE(other.jacobian.row(0).isZero());
  EXPECT_NEAR(other.desired_velocity(1), 0.05, 1e-12);  // gain 2 * 0.03 = 0.06, clamped
}

TEST_F(FramePoseLimitsTest, HysteresisAvoidsChattering)
{
  EXPECT_FALSE(at({0.079, 0, 0}).active);
  EXPECT_TRUE(at({0.081, 0, 0}).active);
  // Between the inner band and the activation bound the axis stays active.
  EXPECT_TRUE(at({0.07, 0, 0}).active);
  EXPECT_TRUE(at({0.0605, 0, 0}).active);
  EXPECT_FALSE(at({0.0599, 0, 0}).active);
  // Once released, it does not reactivate until reaching the activation bound again.
  EXPECT_FALSE(at({0.075, 0, 0}).active);
  EXPECT_TRUE(at({0.081, 0, 0}).active);
  task.reset();
  EXPECT_FALSE(at({0.075, 0, 0}).active);
}

TEST_F(FramePoseLimitsTest, MaskedAxesNeverProduceRows)
{
  // Rotation about x is masked: even a huge roll error yields no row.
  backend.frame.pose.setIdentity();
  backend.frame.pose.linear() = Eigen::AngleAxisd(1.0, Eigen::Vector3d::UnitX()).toRotationMatrix();
  const auto out = task.update(state, backend);
  EXPECT_FALSE(out.active);
  EXPECT_EQ(out.jacobian.rows(), 4);

  const auto yaw = at({0, 0, 0}, 0.29);
  ASSERT_TRUE(yaw.active);
  EXPECT_FALSE(yaw.jacobian.row(3).isZero());
  EXPECT_NEAR(yaw.jacobian(3, 5), 1.0, 1e-12);
  EXPECT_LT(yaw.desired_velocity(3), 0.0);
}

TEST_F(FramePoseLimitsTest, ErrorAndJacobianAreExpressedInTargetAxes)
{
  const Eigen::Quaterniond target(Eigen::AngleAxisd(std::acos(-1.0) / 2, Eigen::Vector3d::UnitZ()));
  ASSERT_TRUE(task.set_pose_goal(goal_at(Eigen::Vector3d::Zero(), target)));
  // World +y is target +x after a 90 deg yaw. Keep the frame aligned with the target.
  backend.frame.pose.setIdentity();
  backend.frame.pose.linear() = target.toRotationMatrix();
  backend.frame.pose.translation() = Eigen::Vector3d(0.0, 0.09, 0.0);
  const auto out = task.update(state, backend);
  ASSERT_TRUE(out.active);
  EXPECT_FALSE(out.jacobian.row(0).isZero());  // target x
  EXPECT_TRUE(out.jacobian.row(1).isZero());   // target y untouched
  EXPECT_NEAR(out.jacobian(0, 1), 1.0, 1e-12);  // maps world y velocity
  EXPECT_NEAR(out.jacobian(0, 0), 0.0, 1e-12);
  EXPECT_LT(out.desired_velocity(0), 0.0);
}

TEST_F(FramePoseLimitsTest, CapturesCurrentPoseWithoutGoalAndRejectsForeignFrames)
{
  FramePoseLimitsTask fresh;
  fresh.configure("frame_pose_limits", "task_priority_kinematic_control/FramePoseLimitsTask",
    limits_parameters(), TaskContext{model});
  backend.frame.pose.setIdentity();
  backend.frame.pose.translation() = Eigen::Vector3d(1.0, 2.0, 3.0);
  auto out = fresh.update(state, backend);
  EXPECT_FALSE(out.active);
  EXPECT_TRUE(out.has_updated_target);
  const auto target = fresh.current_target();
  ASSERT_EQ(target.size(), 7u);
  EXPECT_DOUBLE_EQ(target[0], 1.0);
  EXPECT_DOUBLE_EQ(target[2], 3.0);
  backend.frame.pose.translation() = Eigen::Vector3d(1.09, 2.0, 3.0);
  EXPECT_TRUE(fresh.update(state, backend).active);

  EXPECT_FALSE(fresh.set_pose_goal(goal_at(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), "base")));
  EXPECT_EQ(fresh.build_status().target_type, "pose");
}

TEST_F(FramePoseLimitsTest, ResetForgetsReceivedGoalAndCapturesCurrentPose)
{
  ASSERT_TRUE(task.set_pose_goal(goal_at(Eigen::Vector3d(0.5, 0.0, 0.0))));
  EXPECT_TRUE(at({0.0, 0.0, 0.0}).active);  // far from the received goal
  task.reset();
  EXPECT_TRUE(task.current_target().empty());
  const auto out = at({0.0, 0.0, 0.0});
  EXPECT_FALSE(out.active);  // the current pose is now the target
  EXPECT_TRUE(out.has_updated_target);
  const auto target = task.current_target();
  ASSERT_EQ(target.size(), 7u);
  EXPECT_DOUBLE_EQ(target[0], 0.0);
  EXPECT_TRUE(at({0.09, 0.0, 0.0}).active);
}

TEST_F(FramePoseLimitsTest, DisabledTaskIsInactiveAndInvalidMarginsAreRejected)
{
  task.set_enabled(false);
  EXPECT_FALSE(at({0.2, 0, 0}).active);
  EXPECT_EQ(task.build_status().status_message, "disabled");

  auto parameters = limits_parameters();
  parameters["release_margin"] = rclcpp::Parameter("release_margin", std::vector<double>{0.01});
  FramePoseLimitsTask bad;
  EXPECT_THROW(bad.configure("bad", "p", parameters, TaskContext{model}), std::invalid_argument);
  parameters = limits_parameters();
  parameters["axes"] = rclcpp::Parameter("axes", std::vector<bool>{false, false, false, false, false, false});
  EXPECT_THROW(bad.configure("bad", "p", parameters, TaskContext{model}), std::invalid_argument);
  parameters = limits_parameters();
  parameters["axes"] = rclcpp::Parameter("axes", std::vector<int64_t>{1, 0, 0, 0, 0, 0});
  EXPECT_NO_THROW(bad.configure("ok", "p", parameters, TaskContext{model}));
  EXPECT_EQ(bad.rows(), 1);
}

TEST(FramePoseLimitsPlugin, LoadsThroughPluginlib)
{
  pluginlib::ClassLoader<TaskBase> loader("task_priority_kinematic_control",
    "task_priority_kinematic_control::TaskBase");
  EXPECT_NO_THROW({
    auto task = loader.createSharedInstance("task_priority_kinematic_control/FramePoseLimitsTask");
    EXPECT_NE(task, nullptr);
  });
}

}  // namespace task_priority_kinematic_control
