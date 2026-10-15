#include "task_priority_kinematic_control/tasks/stabilize_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/frame_pose_task.hpp"
#include "task_priority_kinematic_control/core/hierarchy_solver.hpp"

#include <gtest/gtest.h>
#include <pluginlib/class_loader.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace task_priority_kinematic_control
{
namespace
{
double radians(double degrees) {return degrees * std::acos(-1.0) / 180;}

class LimitsBackend : public KinematicsBackend
{
public:
  void configure(const WholeBodyModel &, const std::string &, const rclcpp::Logger &) override {}
  void update(const WholeBodyState &) override {}
  FrameState get_frame_state(const std::string &) const override {return {};}
  Eigen::Isometry3d get_relative_transform(const std::string &, const std::string &) const override
  {return Eigen::Isometry3d::Identity();}
  const std::vector<CollisionCapsule> & collision_capsules() const override {return capsules_;}
  std::string name() const override {return "limits_test";}
private:
  std::vector<CollisionCapsule> capsules_;
};

std::map<std::string, rclcpp::Parameter> limits_parameters()
{
  return {
    {"enabled", rclcpp::Parameter("enabled", true)},
    {"priority", rclcpp::Parameter("priority", 2.0)},
    {"group", rclcpp::Parameter("group", "base")},
    {"lower_limits", rclcpp::Parameter("lower_limits", std::vector<double>{radians(-20), radians(-20)})},
    {"upper_limits", rclcpp::Parameter("upper_limits", std::vector<double>{radians(20), radians(20)})},
    {"alpha", rclcpp::Parameter("alpha", radians(2))},
    {"delta", rclcpp::Parameter("delta", radians(4))},
    {"eps", rclcpp::Parameter("eps", 0.0001)},
    {"gain_scalar", rclcpp::Parameter("gain_scalar", 0.5)},
    {"max_repulsive_velocity", rclcpp::Parameter("max_repulsive_velocity", 0.1)}};
}
}  // namespace

class StabilizeLimitsTest : public ::testing::Test
{
protected:
  std::shared_ptr<WholeBodyModel> model = std::make_shared<WholeBodyModel>();
  StabilizeLimitsTask task;
  LimitsBackend backend;
  WholeBodyState state;

  void SetUp() override
  {
    model->configure("world", "base", "left", "right", {"left_joint"}, {"right_joint"}, {0,1,2,3,4,5});
    state.navigation_valid = true;
    task.configure("stabilize_limits", "task_priority_kinematic_control/StabilizeLimitsTask",
      limits_parameters(), TaskContext{model});
  }
  TaskComputation at(double roll, double pitch = 0, double yaw = 0)
  {
    state.base_rpy << radians(roll), radians(pitch), radians(yaw);
    return task.update(state, backend);
  }
  WholeBodyCommand solve_with_lower_task(const TaskComputation & limit, const Eigen::VectorXd & velocity)
  {
    TaskComputation lower;
    lower.active = true;
    lower.jacobian = Eigen::MatrixXd::Identity(model->total_dofs(), model->total_dofs());
    lower.desired_velocity = velocity;
    HierarchySolver solver;
    solver.configure(model->total_dofs());
    solver.set_method(SolverMethod::kPinv);
    return solver.solve({limit, lower});
  }
};

TEST_F(StabilizeLimitsTest, InteriorLeavesAllDegreesOfFreedomAvailable)
{
  const auto out = at(5, -10, 30);
  EXPECT_FALSE(out.active);
  EXPECT_TRUE(out.jacobian.isZero());
  EXPECT_TRUE(out.desired_velocity.isZero());
  EXPECT_EQ(out.feedforward.size(), 0);
  EXPECT_FALSE(out.has_updated_target);
  EXPECT_FALSE(task.build_status().active);
  const Eigen::VectorXd velocity = Eigen::VectorXd::Constant(model->total_dofs(), 0.1);
  EXPECT_TRUE(solve_with_lower_task(out, velocity).generalized_velocity.isApprox(velocity, 1e-9));
}

TEST_F(StabilizeLimitsTest, DefaultsUseFourAndEightDegreeMarginsGainTwoAndPointThreeSaturation)
{
  auto parameters = limits_parameters();
  for (const auto * name : {"alpha", "delta", "gain_scalar", "max_repulsive_velocity"}) {
    parameters.erase(name);
  }
  task.configure("stabilize_limits", "task_priority_kinematic_control/StabilizeLimitsTask",
    parameters, TaskContext{model});
  EXPECT_FALSE(at(15).active);
  auto out = at(17);
  ASSERT_TRUE(out.active);
  EXPECT_NEAR(out.desired_velocity.x(), -2*radians(5), 1e-7);
  EXPECT_TRUE(at(13).active);
  out = at(12.5);
  EXPECT_TRUE(out.active);
  EXPECT_NEAR(out.desired_velocity.x(), -2*radians(.5), 1e-7);
  EXPECT_FALSE(at(11).active);
  EXPECT_FALSE(at(13).active);
  out = at(40, -40);
  EXPECT_DOUBLE_EQ(out.desired_velocity.x(), -.3);
  EXPECT_DOUBLE_EQ(out.desired_velocity.y(), .3);
  EXPECT_DOUBLE_EQ(task.publish_marker_rate(), 2.);
  EXPECT_FALSE(task.debug());
}

TEST_F(StabilizeLimitsTest, EachLowerAndUpperLimitProducesInwardCorrection)
{
  for (Eigen::Index axis = 0; axis < 2; ++axis) {
    for (const double sign : {-1.0, 1.0}) {
      task.reset();
      state.base_rpy.setZero();
      state.base_rpy(axis) = sign * radians(19);
      const auto out = task.update(state, backend);
      EXPECT_TRUE(out.active);
      EXPECT_NEAR(out.error(axis), -sign * radians(3), 1e-12);
      EXPECT_NEAR(out.desired_velocity(axis), -sign * 0.5 * radians(3), 1e-12);
      EXPECT_FALSE(out.jacobian.row(axis).isZero());
      EXPECT_TRUE(out.jacobian.row(1-axis).isZero());
      EXPECT_TRUE(task.build_status().active);
    }
  }
}

TEST_F(StabilizeLimitsTest, HysteresisPersistsUntilReleaseAndDisableOrResetClearsIt)
{
  EXPECT_TRUE(at(18).active);
  EXPECT_TRUE(at(17).active);
  EXPECT_FALSE(at(15).active);
  EXPECT_FALSE(at(17).active);
  EXPECT_TRUE(at(-18).active);
  EXPECT_TRUE(at(-17).active);
  EXPECT_FALSE(at(-15).active);
  EXPECT_FALSE(at(-17).active);
  EXPECT_TRUE(at(19).active);
  task.reset();
  EXPECT_FALSE(at(17).active);
  EXPECT_TRUE(at(19).active);
  task.set_enabled(false);
  EXPECT_FALSE(task.build_status().active);
  EXPECT_FALSE(at(19).active);
  task.set_enabled(true);
  EXPECT_FALSE(at(17).active);
}

TEST_F(StabilizeLimitsTest, OnlyRollConstrainsItsOwnRateAndLeavesOtherCommandsFree)
{
  const auto out = at(19);
  Eigen::VectorXd desired = Eigen::VectorXd::Constant(model->total_dofs(), 0.1);
  const auto command = solve_with_lower_task(out, desired);
  EXPECT_NEAR(command.generalized_velocity(3), out.desired_velocity(0), 1e-9);
  for (Eigen::Index dof = 0; dof < desired.size(); ++dof) {
    if (dof != 3) {EXPECT_NEAR(command.generalized_velocity(dof), desired(dof), 1e-9);}
  }
}

TEST_F(StabilizeLimitsTest, BothAxesActivateAndYawRotatesJacobianWithoutChangingBounds)
{
  const auto first = at(19, -19);
  const auto rotated = at(19, -19, 90);
  EXPECT_TRUE(rotated.active);
  EXPECT_TRUE(first.error.isApprox(rotated.error));
  EXPECT_TRUE(first.desired_velocity.isApprox(rotated.desired_velocity));
  EXPECT_NEAR(rotated.jacobian(0,4), 1 / std::cos(radians(-19)), 1e-12);
  EXPECT_NEAR(rotated.jacobian(1,3), -1, 1e-12);
  EXPECT_TRUE(rotated.jacobian.col(5).isZero());
  EXPECT_TRUE(rotated.jacobian.leftCols(3).isZero());
  EXPECT_TRUE(rotated.jacobian.rightCols(2).isZero());
}

TEST_F(StabilizeLimitsTest, ReadsDifferentAxisBoundsGainsMarginsAndSaturationFromParameters)
{
  auto parameters = limits_parameters();
  parameters["lower_limits"] = rclcpp::Parameter("lower_limits", std::vector<double>{radians(-30), radians(-10)});
  parameters["upper_limits"] = rclcpp::Parameter("upper_limits", std::vector<double>{radians(30), radians(15)});
  parameters["alpha"] = rclcpp::Parameter("alpha", radians(1));
  parameters["delta"] = rclcpp::Parameter("delta", radians(3));
  parameters["gain_scalar"] = rclcpp::Parameter("gain_scalar", 2.0);
  parameters["max_repulsive_velocity"] = rclcpp::Parameter("max_repulsive_velocity", 0.01);
  task.configure("limits", "plugin", parameters, TaskContext{model});
  EXPECT_FALSE(at(25, 13).active);
  const auto out = at(30, -10);
  EXPECT_TRUE(out.active);
  EXPECT_NEAR(out.error(0), radians(-3), 1e-12);
  EXPECT_NEAR(out.error(1), radians(3), 1e-12);
  EXPECT_DOUBLE_EQ(out.desired_velocity(0), -0.01);
  EXPECT_DOUBLE_EQ(out.desired_velocity(1), 0.01);
}

TEST_F(StabilizeLimitsTest, HandlesReorderedBaseColumns)
{
  model->configure("world", "base", "left", "right", {"left_joint"}, {"right_joint"}, {5,1,4,0,3,2});
  task.configure("limits", "plugin", limits_parameters(), TaskContext{model});
  const auto out = at(19, -19);
  EXPECT_NEAR(out.jacobian(0,4), 1 / std::cos(radians(-19)), 1e-12);
  EXPECT_DOUBLE_EQ(out.jacobian(1,2), 1);
  for (const Eigen::Index column : {0,1,3,5,6,7}) {EXPECT_TRUE(out.jacobian.col(column).isZero());}
}

TEST_F(StabilizeLimitsTest, InvalidNavigationAndSingularityProduceFiniteInactiveOutputs)
{
  EXPECT_TRUE(at(19).active);
  state.navigation_valid = false;
  EXPECT_EQ(task.update(state, backend).status_message, "waiting_for_navigation");
  state.navigation_valid = true;
  EXPECT_FALSE(at(17).active);
  state.base_rpy.x() = std::numeric_limits<double>::quiet_NaN();
  const auto invalid = task.update(state, backend);
  EXPECT_FALSE(invalid.active);
  EXPECT_TRUE(invalid.jacobian.allFinite());
  EXPECT_EQ(invalid.status_message, "invalid_navigation");
  const auto singular = at(0, 90);
  EXPECT_FALSE(singular.active);
  EXPECT_TRUE(singular.desired_velocity.isZero());
  EXPECT_EQ(singular.status_message, "roll/pitch_singularity");
}

TEST_F(StabilizeLimitsTest, RejectsMalformedConfigurationAndPoseTargets)
{
  const std::vector<rclcpp::Parameter> invalid = {
    rclcpp::Parameter("lower_limits", std::vector<double>{0}),
    rclcpp::Parameter("upper_limits", std::vector<double>{0,0}),
    rclcpp::Parameter("upper_limits", std::vector<double>{radians(20), radians(90)}),
    rclcpp::Parameter("lower_limits", std::vector<double>{std::numeric_limits<double>::quiet_NaN(), radians(-20)}),
    rclcpp::Parameter("alpha", -0.1),
    rclcpp::Parameter("delta", radians(2)),
    rclcpp::Parameter("delta", radians(21)),
    rclcpp::Parameter("eps", 1.0),
    rclcpp::Parameter("gain_scalar", -1.0),
    rclcpp::Parameter("max_repulsive_velocity", 0.0),
    rclcpp::Parameter("max_repulsive_velocity", std::numeric_limits<double>::infinity())};
  for (const auto & parameter : invalid) {
    auto parameters = limits_parameters();
    parameters[parameter.get_name()] = parameter;
    EXPECT_THROW(task.configure("limits", "plugin", parameters, TaskContext{model}), std::invalid_argument);
  }
  EXPECT_FALSE(task.set_pose_goal(geometry_msgs::msg::PoseStamped{}));
}

TEST_F(StabilizeLimitsTest, PluginLoadsThroughTheSharedTaskRegistryWithoutPoseInterfaces)
{
  pluginlib::ClassLoader<TaskBase> loader("task_priority_kinematic_control", "task_priority_kinematic_control::TaskBase");
  const auto classes = loader.getDeclaredClasses();
  const std::string plugin = "task_priority_kinematic_control/StabilizeLimitsTask";
  ASSERT_NE(std::find(classes.begin(), classes.end(), plugin), classes.end());
  auto instance = loader.createSharedInstance(plugin);
  ASSERT_TRUE(std::dynamic_pointer_cast<StabilizeLimitsTask>(instance));
  EXPECT_FALSE(std::dynamic_pointer_cast<PoseTaskCommon>(instance));
  EXPECT_FALSE(std::dynamic_pointer_cast<FramePoseTask>(instance));
  instance->configure("limits", plugin, limits_parameters(), TaskContext{model});
  EXPECT_EQ(instance->priority(), 2U);
}

}  // namespace task_priority_kinematic_control
