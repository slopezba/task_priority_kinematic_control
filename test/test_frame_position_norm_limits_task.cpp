#include "task_priority_kinematic_control/tasks/frame_position_norm_limits_task.hpp"
#include "task_priority_kinematic_control/core/capsule_observer.hpp"
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

// radius 0.1, activation at 0.08, release at 0.06.
std::map<std::string, rclcpp::Parameter> norm_parameters()
{
  return {
    {"enabled", rclcpp::Parameter("enabled", true)},
    {"priority", rclcpp::Parameter("priority", 18.0)},
    {"group", rclcpp::Parameter("group", "right")},
    {"frame_id", rclcpp::Parameter("frame_id", "tool")},
    {"threshold", rclcpp::Parameter("threshold", std::vector<double>{0.1})},
    {"activation_margin", rclcpp::Parameter("activation_margin", std::vector<double>{0.02})},
    {"release_margin", rclcpp::Parameter("release_margin", std::vector<double>{0.04})},
    {"eps", rclcpp::Parameter("eps", 1e-4)},
    {"gain", rclcpp::Parameter("gain", std::vector<double>{1.0})},
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

class PositionNormTest : public ::testing::Test
{
protected:
  std::shared_ptr<WholeBodyModel> model = std::make_shared<WholeBodyModel>();
  FramePositionNormLimitsTask task;
  FrameBackend backend;
  WholeBodyState state;

  void SetUp() override
  {
    model->configure("world", "base", "left", "right", {"left_joint"}, {"right_joint"}, {0, 1, 2, 3, 4, 5});
    // The frame moves exactly with the six base DoFs.
    backend.frame.jacobian = Eigen::MatrixXd::Zero(6, model->total_dofs());
    backend.frame.jacobian.leftCols(6).setIdentity();
    backend.frame.pose.setIdentity();
    configure(task, norm_parameters());
  }

  void configure(FramePositionNormLimitsTask & target, const std::map<std::string, rclcpp::Parameter> & p)
  {
    target.configure("position_norm", "task_priority_kinematic_control/FramePositionNormLimitsTask",
      p, TaskContext{model});
    ASSERT_TRUE(target.set_pose_goal(goal_at(Eigen::Vector3d::Zero())));
  }

  TaskComputation at(const Eigen::Vector3d & position)
  {
    backend.frame.pose.setIdentity();
    backend.frame.pose.translation() = position;
    return task.update(state, backend);
  }
};

TEST_F(PositionNormTest, InsideSphereIsInactiveAndLeavesDofsFree)
{
  const auto out = at({0.04, 0.04, 0.04});  // norm ~0.069 < 0.08
  EXPECT_FALSE(out.active);
  EXPECT_TRUE(out.structural_zero_rows);
  EXPECT_EQ(out.jacobian.rows(), 1);
  EXPECT_TRUE(out.jacobian.isZero());
  EXPECT_EQ(out.status_message, "inside_limit");

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

TEST_F(PositionNormTest, ActivatesNearTheSphereAndPushesRadiallyInward)
{
  const Eigen::Vector3d direction = Eigen::Vector3d(1.0, 1.0, 0.0).normalized();
  const auto out = at(0.085 * direction);
  ASSERT_TRUE(out.active);
  EXPECT_EQ(out.status_message, "holding_limit");
  // d(norm)/dq along the base translation = radial direction.
  EXPECT_TRUE(out.jacobian.row(0).head<3>().transpose().isApprox(direction, 1e-12));
  EXPECT_TRUE(out.jacobian.row(0).tail(model->total_dofs() - 3).isZero());
  EXPECT_NEAR(out.error(0), 0.06 - 0.085, 1e-12);
  EXPECT_NEAR(out.desired_velocity(0), -0.025, 1e-12);
  EXPECT_NEAR(task.norm(), 0.085, 1e-12);

  // Velocity clamp.
  EXPECT_NEAR(at({0.2, 0.0, 0.0}).desired_velocity(0), -0.05, 1e-12);
}

TEST_F(PositionNormTest, HysteresisAvoidsChattering)
{
  EXPECT_FALSE(at({0.079, 0, 0}).active);
  EXPECT_TRUE(at({0.081, 0, 0}).active);
  EXPECT_TRUE(at({0.0, 0.07, 0}).active);   // still active inside the band
  EXPECT_TRUE(at({0.0, 0.0, 0.0605}).active);
  EXPECT_FALSE(at({0.0599, 0, 0}).active);
  EXPECT_FALSE(at({0.075, 0, 0}).active);   // released until the activation bound again
  EXPECT_TRUE(at({0.081, 0, 0}).active);
}

TEST_F(PositionNormTest, MaskedAxisIsFreeCylinder)
{
  auto parameters = norm_parameters();
  parameters["axes"] = rclcpp::Parameter("axes", std::vector<bool>{true, true, false});
  configure(task, parameters);
  EXPECT_FALSE(at({0.0, 0.0, 5.0}).active);       // z free
  const auto out = at({0.0, 0.09, 3.0});
  ASSERT_TRUE(out.active);
  EXPECT_NEAR(out.jacobian(0, 1), 1.0, 1e-12);
  EXPECT_NEAR(out.jacobian(0, 2), 0.0, 1e-12);
}

TEST_F(PositionNormTest, MaskUsesTargetAxes)
{
  auto parameters = norm_parameters();
  parameters["axes"] = rclcpp::Parameter("axes", std::vector<int64_t>{1, 0, 0});
  configure(task, parameters);
  // Target rotated 90 deg about z: target x == world y.
  const Eigen::Quaterniond yaw(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()));
  ASSERT_TRUE(task.set_pose_goal(goal_at(Eigen::Vector3d::Zero(), yaw)));
  EXPECT_FALSE(at({0.5, 0.0, 0.0}).active);   // world x == target -y: free
  const auto out = at({0.0, 0.09, 0.0});
  ASSERT_TRUE(out.active);
  EXPECT_NEAR(out.jacobian(0, 1), 1.0, 1e-12);
}

TEST_F(PositionNormTest, AlwaysActiveDrivesNormToZero)
{
  auto parameters = norm_parameters();
  parameters["always_active"] = rclcpp::Parameter("always_active", true);
  configure(task, parameters);
  const auto out = at({0.01, 0.0, 0.0});
  ASSERT_TRUE(out.active);
  EXPECT_EQ(out.status_message, "tracking");
  EXPECT_NEAR(out.error(0), -0.01, 1e-12);
  EXPECT_NEAR(out.desired_velocity(0), -0.01, 1e-12);
  // At the target the direction is undefined: structural zero row, no NaN.
  const auto at_target = at({0.0, 0.0, 0.0});
  EXPECT_FALSE(at_target.active);
  EXPECT_TRUE(at_target.jacobian.allFinite());
  EXPECT_TRUE(at_target.jacobian.isZero());
}

TEST_F(PositionNormTest, BaseFrameUsesNavigationState)
{
  auto parameters = norm_parameters();
  parameters["frame_id"] = rclcpp::Parameter("frame_id", "base");
  FramePositionNormLimitsTask base_task;
  base_task.configure("base_norm", "p", parameters, TaskContext{model});
  state.base_position = Eigen::Vector3d(1.0, 2.0, 3.0);
  state.base_rpy.setZero();
  EXPECT_FALSE(base_task.update(state, backend).active);  // captures the current pose
  state.base_position = Eigen::Vector3d(1.0, 2.0, 3.09);
  const auto out = base_task.update(state, backend);
  ASSERT_TRUE(out.active);
  EXPECT_GT(std::abs(out.jacobian(0, 2)), 0.5);
}

TEST_F(PositionNormTest, CopyKeepsStateAndDebugFlag)
{
  auto parameters = norm_parameters();
  parameters["debug"] = rclcpp::Parameter("debug", true);
  configure(task, parameters);
  EXPECT_TRUE(at({0.085, 0, 0}).active);
  FramePositionNormLimitsTask copy(task);
  EXPECT_TRUE(copy.debug());
  EXPECT_TRUE(copy.row_active());
  EXPECT_EQ(copy.current_target(), task.current_target());
}

TEST_F(PositionNormTest, DebugSnapshotAndMarkers)
{
  auto parameters = norm_parameters();
  parameters["debug"] = rclcpp::Parameter("debug", true);
  configure(task, parameters);
  ASSERT_TRUE(at({0.085, 0, 0}).active);
  task.observe_command(WholeBodyCommand{}, 1000);
  const auto * snapshot = task.consume_snapshot();
  ASSERT_NE(snapshot, nullptr);
  EXPECT_TRUE(snapshot->valid);
  EXPECT_TRUE(snapshot->active);
  EXPECT_NEAR(snapshot->norm, 0.085, 1e-12);
  const auto markers = frame_position_norm_markers(*snapshot, "world", "position_norm", 2.0);
  ASSERT_EQ(markers.markers.size(), 4u);
  EXPECT_EQ(markers.markers[0].type, visualization_msgs::msg::Marker::SPHERE);
  EXPECT_EQ(markers.markers[0].action, visualization_msgs::msg::Marker::ADD);
  EXPECT_NEAR(markers.markers[0].scale.x, 0.2, 1e-12);
  EXPECT_FLOAT_EQ(markers.markers[0].color.r, 1.0f);  // active -> red
  const auto cleared = delete_frame_position_norm_markers("world", "position_norm");
  for (const auto & m : cleared.markers) {
    EXPECT_EQ(m.action, visualization_msgs::msg::Marker::DELETE);
  }
}

TEST_F(PositionNormTest, InvalidParametersAreRejected)
{
  FramePositionNormLimitsTask bad;
  auto parameters = norm_parameters();
  parameters["release_margin"] = rclcpp::Parameter("release_margin", std::vector<double>{0.01});
  EXPECT_THROW(bad.configure("bad", "p", parameters, TaskContext{model}), std::invalid_argument);
  parameters = norm_parameters();
  parameters["threshold"] = rclcpp::Parameter("threshold", std::vector<double>{0.1, 0.1});
  EXPECT_THROW(bad.configure("bad", "p", parameters, TaskContext{model}), std::invalid_argument);
  parameters = norm_parameters();
  parameters["axes"] = rclcpp::Parameter("axes", std::vector<bool>{false, false, false});
  EXPECT_THROW(bad.configure("bad", "p", parameters, TaskContext{model}), std::invalid_argument);
  EXPECT_FALSE(task.set_pose_goal(goal_at(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), "base")));
}

TEST(FramePositionNormPlugin, LoadsThroughPluginlib)
{
  pluginlib::ClassLoader<TaskBase> loader("task_priority_kinematic_control",
    "task_priority_kinematic_control::TaskBase");
  EXPECT_NO_THROW({
    auto task = loader.createSharedInstance("task_priority_kinematic_control/FramePositionNormLimitsTask");
    EXPECT_NE(task, nullptr);
  });
}

}  // namespace task_priority_kinematic_control
