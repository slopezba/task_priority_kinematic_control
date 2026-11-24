#include "task_priority_kinematic_control/tasks/frame_orientation_norm_limits_task.hpp"
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

// Max angle 0.3 rad, activation at 0.25, release at 0.2.
std::map<std::string, rclcpp::Parameter> norm_parameters()
{
  return {
    {"enabled", rclcpp::Parameter("enabled", true)},
    {"priority", rclcpp::Parameter("priority", 19.0)},
    {"group", rclcpp::Parameter("group", "base")},
    {"frame_id", rclcpp::Parameter("frame_id", "tool")},
    {"threshold", rclcpp::Parameter("threshold", std::vector<double>{0.3})},
    {"activation_margin", rclcpp::Parameter("activation_margin", std::vector<double>{0.05})},
    {"release_margin", rclcpp::Parameter("release_margin", std::vector<double>{0.1})},
    {"eps", rclcpp::Parameter("eps", 1e-4)},
    {"gain", rclcpp::Parameter("gain", std::vector<double>{1.0})},
    {"max_repulsive_velocity", rclcpp::Parameter("max_repulsive_velocity", 0.2)}};
}

geometry_msgs::msg::PoseStamped goal_at(
  const Eigen::Quaterniond & orientation, const std::string & frame = "world")
{
  geometry_msgs::msg::PoseStamped goal;
  goal.header.frame_id = frame;
  goal.pose.orientation.x = orientation.x();
  goal.pose.orientation.y = orientation.y();
  goal.pose.orientation.z = orientation.z();
  goal.pose.orientation.w = orientation.w();
  return goal;
}

Eigen::Matrix3d rotation(double angle, const Eigen::Vector3d & axis)
{
  return Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
}

}  // namespace

class OrientationNormTest : public ::testing::Test
{
protected:
  std::shared_ptr<WholeBodyModel> model = std::make_shared<WholeBodyModel>();
  FrameOrientationNormLimitsTask task;
  FrameBackend backend;
  WholeBodyState state;

  void SetUp() override
  {
    model->configure("world", "base", "left", "right", {"left_joint"}, {"right_joint"}, {0, 1, 2, 3, 4, 5});
    backend.frame.jacobian = Eigen::MatrixXd::Zero(6, model->total_dofs());
    backend.frame.jacobian.leftCols(6).setIdentity();
    backend.frame.pose.setIdentity();
    configure(task, norm_parameters());
  }

  void configure(FrameOrientationNormLimitsTask & target,
    const std::map<std::string, rclcpp::Parameter> & p)
  {
    target.configure("orientation_norm", "task_priority_kinematic_control/FrameOrientationNormLimitsTask",
      p, TaskContext{model});
    ASSERT_TRUE(target.set_pose_goal(goal_at(Eigen::Quaterniond::Identity())));
  }

  TaskComputation at(const Eigen::Matrix3d & r)
  {
    backend.frame.pose.setIdentity();
    backend.frame.pose.linear() = r;
    return task.update(state, backend);
  }
};

TEST_F(OrientationNormTest, InsideIsInactive)
{
  const auto out = at(rotation(0.2, {1, 1, 1}));
  EXPECT_FALSE(out.active);
  EXPECT_EQ(out.jacobian.rows(), 1);
  EXPECT_TRUE(out.jacobian.isZero());
  EXPECT_EQ(out.status_message, "inside_limit");
}

TEST_F(OrientationNormTest, ActivatesNearTheLimitAndRotatesBack)
{
  const Eigen::Vector3d axis = Eigen::Vector3d(1, 2, 0.5).normalized();
  const auto out = at(rotation(0.27, axis));
  ASSERT_TRUE(out.active);
  EXPECT_EQ(out.status_message, "holding_limit");
  // d(angle)/dt = axis^T omega: the row is the current rotation axis on the angular DoFs.
  EXPECT_TRUE(out.jacobian.row(0).segment<3>(3).transpose().isApprox(axis, 1e-9));
  EXPECT_TRUE(out.jacobian.row(0).head<3>().isZero());
  EXPECT_NEAR(out.error(0), 0.2 - 0.27, 1e-9);
  EXPECT_NEAR(out.desired_velocity(0), -0.07, 1e-9);
  EXPECT_NEAR(task.norm(), 0.27, 1e-9);

  // Integrating the commanded angular velocity reduces the angle.
  HierarchySolver solver;
  solver.configure(model->total_dofs());
  solver.set_method(SolverMethod::kPinv);
  const auto command = solver.solve({out});
  const Eigen::Vector3d omega = command.generalized_velocity.segment<3>(3);
  EXPECT_LT(axis.dot(omega), 0.0);
}

TEST_F(OrientationNormTest, HysteresisAvoidsChattering)
{
  const Eigen::Vector3d z = Eigen::Vector3d::UnitZ();
  EXPECT_FALSE(at(rotation(0.249, z)).active);
  EXPECT_TRUE(at(rotation(0.251, z)).active);
  EXPECT_TRUE(at(rotation(0.21, z)).active);
  EXPECT_FALSE(at(rotation(0.199, z)).active);
  EXPECT_FALSE(at(rotation(0.24, z)).active);
}

TEST_F(OrientationNormTest, TiltOnlyMaskLeavesYawFree)
{
  auto parameters = norm_parameters();
  parameters["axes"] = rclcpp::Parameter("axes", std::vector<bool>{true, true, false});
  configure(task, parameters);
  EXPECT_FALSE(at(rotation(1.5, Eigen::Vector3d::UnitZ())).active);  // yaw free
  const auto out = at(rotation(0.27, Eigen::Vector3d::UnitX()));
  ASSERT_TRUE(out.active);
  EXPECT_NEAR(out.jacobian(0, 3), 1.0, 1e-9);
  EXPECT_NEAR(out.jacobian(0, 5), 0.0, 1e-9);
}

TEST_F(OrientationNormTest, AlwaysActiveDrivesAngleToZero)
{
  auto parameters = norm_parameters();
  parameters["always_active"] = rclcpp::Parameter("always_active", true);
  configure(task, parameters);
  const auto out = at(rotation(0.05, Eigen::Vector3d::UnitY()));
  ASSERT_TRUE(out.active);
  EXPECT_EQ(out.status_message, "tracking");
  EXPECT_NEAR(out.desired_velocity(0), -0.05, 1e-9);
  const auto at_target = at(Eigen::Matrix3d::Identity());
  EXPECT_FALSE(at_target.active);
  EXPECT_TRUE(at_target.jacobian.allFinite());
  EXPECT_TRUE(at_target.jacobian.isZero());
}

TEST_F(OrientationNormTest, BaseFrameUsesNavigationState)
{
  auto parameters = norm_parameters();
  parameters["frame_id"] = rclcpp::Parameter("frame_id", "base");
  FrameOrientationNormLimitsTask base_task;
  base_task.configure("base_norm", "p", parameters, TaskContext{model});
  state.base_position.setZero();
  state.base_rpy.setZero();
  EXPECT_FALSE(base_task.update(state, backend).active);  // captures the current attitude
  state.base_rpy = Eigen::Vector3d(0.0, 0.0, 0.27);
  EXPECT_TRUE(base_task.update(state, backend).active);
}

TEST_F(OrientationNormTest, CopyKeepsStateAndDebugFlag)
{
  auto parameters = norm_parameters();
  parameters["debug"] = rclcpp::Parameter("debug", true);
  configure(task, parameters);
  EXPECT_TRUE(at(rotation(0.27, Eigen::Vector3d::UnitZ())).active);
  FrameOrientationNormLimitsTask copy(task);
  EXPECT_TRUE(copy.debug());
  EXPECT_TRUE(copy.row_active());
  EXPECT_EQ(copy.current_target(), task.current_target());
}

TEST_F(OrientationNormTest, DebugSnapshotAndMarkers)
{
  auto parameters = norm_parameters();
  parameters["debug"] = rclcpp::Parameter("debug", true);
  configure(task, parameters);
  ASSERT_TRUE(at(rotation(0.27, Eigen::Vector3d::UnitX())).active);
  task.observe_command(WholeBodyCommand{}, 1000);
  const auto * snapshot = task.consume_snapshot();
  ASSERT_NE(snapshot, nullptr);
  EXPECT_TRUE(snapshot->valid);
  EXPECT_TRUE(snapshot->active);
  const auto markers = frame_orientation_norm_markers(*snapshot, "world", "orientation_norm", 2.0);
  ASSERT_EQ(markers.markers.size(), 11u);
  EXPECT_EQ(markers.markers[0].action, visualization_msgs::msg::Marker::ADD);   // cone
  EXPECT_EQ(markers.markers[3].action, visualization_msgs::msg::Marker::DELETE);  // no yaw ring
  EXPECT_EQ(markers.markers[7].action, visualization_msgs::msg::Marker::ADD);   // gauge

  parameters["axes"] = rclcpp::Parameter("axes", std::vector<bool>{false, false, true});
  configure(task, parameters);
  ASSERT_TRUE(at(rotation(0.27, Eigen::Vector3d::UnitZ())).active);
  task.observe_command(WholeBodyCommand{}, 1000);
  const auto * yaw_snapshot = task.consume_snapshot();
  ASSERT_NE(yaw_snapshot, nullptr);
  const auto yaw = frame_orientation_norm_markers(*yaw_snapshot, "world", "orientation_norm", 2.0);
  EXPECT_EQ(yaw.markers[0].action, visualization_msgs::msg::Marker::DELETE);  // no cone
  EXPECT_EQ(yaw.markers[3].action, visualization_msgs::msg::Marker::ADD);     // yaw ring
  for (const auto & m : delete_frame_orientation_norm_markers("world", "orientation_norm").markers) {
    EXPECT_EQ(m.action, visualization_msgs::msg::Marker::DELETE);
  }
}

TEST_F(OrientationNormTest, InvalidParametersAreRejected)
{
  FrameOrientationNormLimitsTask bad;
  auto parameters = norm_parameters();
  parameters["threshold"] = rclcpp::Parameter("threshold", std::vector<double>{3.2});
  EXPECT_THROW(bad.configure("bad", "p", parameters, TaskContext{model}), std::invalid_argument);
  parameters = norm_parameters();
  parameters["activation_margin"] = rclcpp::Parameter("activation_margin", std::vector<double>{0.2});
  EXPECT_THROW(bad.configure("bad", "p", parameters, TaskContext{model}), std::invalid_argument);
  EXPECT_FALSE(task.set_pose_goal(goal_at(Eigen::Quaterniond(0, 0, 0, 0))));
  EXPECT_FALSE(task.set_pose_goal(goal_at(Eigen::Quaterniond::Identity(), "base")));
}

TEST(FrameOrientationNormPlugin, LoadsThroughPluginlib)
{
  pluginlib::ClassLoader<TaskBase> loader("task_priority_kinematic_control",
    "task_priority_kinematic_control::TaskBase");
  EXPECT_NO_THROW({
    auto task = loader.createSharedInstance(
      "task_priority_kinematic_control/FrameOrientationNormLimitsTask");
    EXPECT_NE(task, nullptr);
  });
}

}  // namespace task_priority_kinematic_control
