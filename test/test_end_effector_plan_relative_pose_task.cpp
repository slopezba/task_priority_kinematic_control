#include "task_priority_kinematic_control/tasks/end_effector_plan_relative_pose_task.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

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

std::map<std::string, rclcpp::Parameter> task_parameters(double planning_timeout = 5.0)
{
  return {
    {"plugin", rclcpp::Parameter("plugin", "task_priority_kinematic_control/EndEffectorPlanRelativePoseTask")},
    {"enabled", rclcpp::Parameter("enabled", true)},
    {"priority", rclcpp::Parameter("priority", 1.0)},
    {"group", rclcpp::Parameter("group", "bimanual")},
    {"reference_frame", rclcpp::Parameter("reference_frame", "left_tip")},
    {"controlled_frame", rclcpp::Parameter("controlled_frame", "right_tip")},
    {"gain", rclcpp::Parameter("gain", std::vector<double>{2.0, 2.0, 2.0, 1.0, 1.0, 1.0})},
    {"activation", rclcpp::Parameter("activation", std::vector<double>{1.0, 0.0, 0.0, 0.0, 0.0, 0.0})},
    {"lower_limits", rclcpp::Parameter("lower_limits", std::vector<double>{-10.0, -10.0})},
    {"upper_limits", rclcpp::Parameter("upper_limits", std::vector<double>{10.0, 10.0})},
    {"planning_timeout", rclcpp::Parameter("planning_timeout", planning_timeout)},
    {"move_gain", rclcpp::Parameter("move_gain", 1.0)},
    {"move_goal_tolerance", rclcpp::Parameter("move_goal_tolerance", 0.05)},
    {"blend_duration", rclcpp::Parameter("blend_duration", 0.0)},
    {"max_joint_velocity", rclcpp::Parameter("max_joint_velocity", 10.0)},
    {"min_sigma", rclcpp::Parameter("min_sigma", 0.0)},
    {"min_joint_margin", rclcpp::Parameter("min_joint_margin", 0.0)},
    {"min_collision_margin", rclcpp::Parameter("min_collision_margin", 0.0)},
    {"nullspace_probe_steps", rclcpp::Parameter("nullspace_probe_steps", 2.0)},
  };
}

std::map<std::string, rclcpp::Parameter> impossible_planning_parameters()
{
  auto params = task_parameters();
  params["min_sigma"] = rclcpp::Parameter("min_sigma", 100.0);
  return params;
}

MockBackend backend()
{
  MockBackend out;

  FrameState left;
  left.pose = Eigen::Isometry3d::Identity();
  left.jacobian = Eigen::MatrixXd::Zero(6, 2);
  left.jacobian(0, 0) = 1.0;

  FrameState right;
  right.pose = Eigen::Isometry3d::Identity();
  right.pose.translation() = Eigen::Vector3d(1.0, 0.0, 0.0);
  right.jacobian = Eigen::MatrixXd::Zero(6, 2);
  right.jacobian(0, 1) = 1.0;

  out.frames_["left_tip"] = left;
  out.frames_["right_tip"] = right;
  return out;
}

WholeBodyState state(const std::vector<double> & joints)
{
  WholeBodyState out;
  out.joints_valid = true;
  out.joint_positions = Eigen::VectorXd::Zero(joints.size());
  out.joint_velocities = Eigen::VectorXd::Zero(joints.size());
  for (size_t i = 0; i < joints.size(); ++i) {
    out.joint_positions(static_cast<Eigen::Index>(i)) = joints[i];
  }
  return out;
}

geometry_msgs::msg::PoseStamped relative_goal(double x)
{
  geometry_msgs::msg::PoseStamped goal;
  goal.pose.position.x = x;
  goal.pose.orientation.w = 1.0;
  return goal;
}

}  // namespace

TEST(EndEffectorPlanRelativePoseTask, ReportsPoseTargetType)
{
  EndEffectorPlanRelativePoseTask task;
  task.configure(
    "ee_plan_relative_pose",
    "task_priority_kinematic_control/EndEffectorPlanRelativePoseTask",
    task_parameters(),
    TaskContext{model()});

  const auto status = task.build_status();

  EXPECT_EQ(status.id, "ee_plan_relative_pose");
  EXPECT_EQ(status.target_type, "pose");
  EXPECT_EQ(status.joint_names.size(), 2U);
}

TEST(EndEffectorPlanRelativePoseTask, PoseGoalStartsNonBlockingPlanning)
{
  EndEffectorPlanRelativePoseTask task;
  task.configure(
    "ee_plan_relative_pose",
    "task_priority_kinematic_control/EndEffectorPlanRelativePoseTask",
    task_parameters(),
    TaskContext{model()});
  ASSERT_TRUE(task.set_pose_goal(relative_goal(2.0)));

  const auto output = task.update(state({0.0, 0.0}), backend());

  EXPECT_FALSE(output.active);
  EXPECT_EQ(output.status_message, "planning");
}

TEST(EndEffectorPlanRelativePoseTask, PlannerProducesInternalJointObjective)
{
  EndEffectorPlanRelativePoseTask task;
  task.configure(
    "ee_plan_relative_pose",
    "task_priority_kinematic_control/EndEffectorPlanRelativePoseTask",
    task_parameters(),
    TaskContext{model()});
  ASSERT_TRUE(task.set_pose_goal(relative_goal(2.0)));

  MockBackend mock_backend = backend();
  WholeBodyState current_state = state({0.0, 0.0});
  (void)task.update(current_state, mock_backend);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  const auto output = task.update(current_state, mock_backend);

  ASSERT_TRUE(output.active);
  ASSERT_EQ(output.jacobian.rows(), 2);
  ASSERT_EQ(output.jacobian.cols(), 2);
  EXPECT_NEAR(output.jacobian(0, 0), 1.0, 1e-9);
  EXPECT_NEAR(output.jacobian(1, 1), 1.0, 1e-9);
  EXPECT_EQ(output.status_message, "moving_to_manifold");
}

TEST(EndEffectorPlanRelativePoseTask, PlanningTimeoutReturnsIdleStatus)
{
  EndEffectorPlanRelativePoseTask task;
  task.configure(
    "ee_plan_relative_pose",
    "task_priority_kinematic_control/EndEffectorPlanRelativePoseTask",
    task_parameters(0.0),
    TaskContext{model()});
  ASSERT_TRUE(task.set_pose_goal(relative_goal(2.0)));

  MockBackend mock_backend = backend();
  WholeBodyState current_state = state({0.0, 0.0});
  (void)task.update(current_state, mock_backend);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  const auto output = task.update(current_state, mock_backend);
  const auto status = task.build_status();

  EXPECT_FALSE(output.active);
  EXPECT_NE(status.status_message.find("planning_timeout"), std::string::npos);
}

TEST(EndEffectorPlanRelativePoseTask, ReachesServoRelativePoseAfterMoveAndBlend)
{
  EndEffectorPlanRelativePoseTask task;
  task.configure(
    "ee_plan_relative_pose",
    "task_priority_kinematic_control/EndEffectorPlanRelativePoseTask",
    task_parameters(),
    TaskContext{model()});
  ASSERT_TRUE(task.set_pose_goal(relative_goal(1.0)));

  MockBackend mock_backend = backend();
  WholeBodyState current_state = state({0.0, 0.0});
  (void)task.update(current_state, mock_backend);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  (void)task.update(current_state, mock_backend);
  std::this_thread::sleep_for(std::chrono::milliseconds(120));
  (void)task.update(current_state, mock_backend);
  const auto output = task.update(current_state, mock_backend);

  EXPECT_TRUE(output.active);
  const auto status = task.build_status();
  EXPECT_EQ(status.status_message, "servo_relative_pose");
}

TEST(EndEffectorPlanRelativePoseTask, FallsBackToServoWhenNoManifoldCandidateIsValid)
{
  EndEffectorPlanRelativePoseTask task;
  task.configure(
    "ee_plan_relative_pose",
    "task_priority_kinematic_control/EndEffectorPlanRelativePoseTask",
    impossible_planning_parameters(),
    TaskContext{model()});
  ASSERT_TRUE(task.set_pose_goal(relative_goal(2.0)));

  MockBackend mock_backend = backend();
  WholeBodyState current_state = state({0.0, 0.0});
  (void)task.update(current_state, mock_backend);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  const auto output = task.update(current_state, mock_backend);

  EXPECT_TRUE(output.active);
  EXPECT_EQ(output.status_message, "tracking_relative_pose");
  EXPECT_TRUE(output.has_frame_pose);
  const auto status = task.build_status();
  EXPECT_EQ(status.status_message, "servo_relative_pose");
}

}  // namespace task_priority_kinematic_control
