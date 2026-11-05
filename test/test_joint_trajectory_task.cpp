#include "task_priority_kinematic_control/tasks/joint_trajectory_task.hpp"

#include <gtest/gtest.h>
#include <array>
#include <limits>

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

class JointTrajectoryInterpolationTest : public ::testing::Test
{
protected:
  void configure(JointTrajectoryTask & task, const std::string & interpolation = "linear")
  {
    auto model = std::make_shared<WholeBodyModel>();
    model->configure("world", "base", "left_tip", "right_tip", {"joint_a", "joint_b"}, {}, {});
    auto parameters = task_parameters();
    parameters.emplace("interpolation_method", rclcpp::Parameter("interpolation_method", interpolation));
    task.configure("trajectory", "task_priority_kinematic_control/JointTrajectoryTask", parameters,
      TaskContext{model});
  }

  std::array<double, 3> sample(const JointTrajectoryTask & task, double time)
  {
    const auto value = task.sample_at_locked(time);
    return {value.position(0), value.velocity(0), value.acceleration(0)};
  }

  void advance(JointTrajectoryTask & task, double seconds)
  {
    std::lock_guard<std::mutex> lock(task.mutex_);
    task.start_time_ -= std::chrono::duration_cast<JointTrajectoryTask::SteadyClock::duration>(
      std::chrono::duration<double>(seconds));
  }

  trajectory_msgs::msg::JointTrajectory smooth_trajectory()
  {
    auto trajectory = valid_trajectory();
    trajectory.points[0].positions = {0.0, 0.0};
    trajectory.points[1].positions = {1.0, 1.0};
    for (auto & point : trajectory.points) {
      point.velocities = {0.0, 0.0};
      point.accelerations = {0.0, 0.0};
    }
    return trajectory;
  }
};

TEST_F(JointTrajectoryInterpolationTest, DisabledTaskRejectsWithoutKeepingSamples)
{
  JointTrajectoryTask task;
  configure(task);
  ASSERT_TRUE(task.set_enabled(false));
  std::string message;
  EXPECT_FALSE(task.set_joint_trajectory(valid_trajectory(), message));
  EXPECT_FALSE(task.joint_trajectory_status().accepted);
  ASSERT_TRUE(task.set_enabled(true));
  EXPECT_FALSE(task.joint_trajectory_status().active);
  EXPECT_FALSE(task.joint_trajectory_status().accepted);
}

TEST_F(JointTrajectoryInterpolationTest, DisableAndReenableCannotResumeOldTrajectory)
{
  JointTrajectoryTask task;
  configure(task);
  std::string message;
  ASSERT_TRUE(task.set_joint_trajectory(valid_trajectory(), message));
  ASSERT_TRUE(task.set_enabled(false));
  ASSERT_TRUE(task.set_enabled(true));
  EXPECT_TRUE(task.joint_trajectory_status().canceled);
  EXPECT_FALSE(task.joint_trajectory_status().accepted);
  EXPECT_FALSE(task.joint_trajectory_status().active);
  ASSERT_TRUE(task.set_joint_trajectory(valid_trajectory(), message));
  EXPECT_TRUE(task.joint_trajectory_status().active);
  EXPECT_FALSE(task.joint_trajectory_status().canceled);
}

TEST_F(JointTrajectoryInterpolationTest, CancelAndResetClearStoredTrajectory)
{
  JointTrajectoryTask task;
  configure(task);
  std::string message;
  ASSERT_TRUE(task.set_joint_trajectory(valid_trajectory(), message));
  ASSERT_TRUE(task.cancel_joint_trajectory(message));
  EXPECT_FALSE(task.joint_trajectory_status().accepted);
  EXPECT_TRUE(task.joint_trajectory_status().canceled);
  ASSERT_TRUE(task.set_joint_trajectory(valid_trajectory(), message));
  task.reset();
  EXPECT_FALSE(task.joint_trajectory_status().accepted);
  EXPECT_FALSE(task.joint_trajectory_status().active);
  EXPECT_TRUE(task.joint_trajectory_status().canceled);
}

TEST_F(JointTrajectoryInterpolationTest, DisableClearsSuccessfulHoldingTrajectory)
{
  JointTrajectoryTask task;
  configure(task);
  auto trajectory = valid_trajectory();
  trajectory.points.resize(1);
  std::string message;
  ASSERT_TRUE(task.set_joint_trajectory(trajectory, message));
  WholeBodyState state;
  state.joints_valid = true;
  state.joint_positions = Eigen::Vector2d(1.0, -1.0);
  MockBackend backend;
  EXPECT_TRUE(task.update(state, backend).active);
  ASSERT_TRUE(task.joint_trajectory_status().succeeded);
  ASSERT_TRUE(task.set_enabled(false));
  ASSERT_TRUE(task.set_enabled(true));
  EXPECT_FALSE(task.joint_trajectory_status().succeeded);
  EXPECT_FALSE(task.joint_trajectory_status().accepted);
  EXPECT_FALSE(task.update(state, backend).active);
}

TEST_F(JointTrajectoryInterpolationTest, SuccessfulHoldingHasNoLaterTimeoutAndCanBeReplaced)
{
  JointTrajectoryTask task;
  configure(task);
  auto trajectory = smooth_trajectory();
  std::string message;
  ASSERT_TRUE(task.set_joint_trajectory(trajectory, message));
  WholeBodyState state;
  state.joints_valid = true;
  state.joint_positions = Eigen::Vector2d::Ones();
  MockBackend backend;
  advance(task, 1.05);  // Finish within the original goal tolerance timeout.
  ASSERT_TRUE(task.update(state, backend).active);
  ASSERT_TRUE(task.joint_trajectory_status().succeeded);
  advance(task, 10.0);
  EXPECT_TRUE(task.update(state, backend).active);
  EXPECT_TRUE(task.joint_trajectory_status().succeeded);
  EXPECT_FALSE(task.joint_trajectory_status().timed_out);

  // Holding continues to correct drift rather than relinquishing ownership.
  state.joint_positions = Eigen::Vector2d::Constant(0.9);
  const auto correction = task.update(state, backend);
  EXPECT_TRUE(correction.active);
  EXPECT_NEAR(correction.desired_velocity(0), 0.2, 1e-12);
  EXPECT_FALSE(task.joint_trajectory_status().timed_out);

  // A new goal starts its own timeline and replaces the former held target.
  trajectory.points[0].positions = {0.25, 0.25};
  trajectory.points[1].positions = {0.5, 0.5};
  ASSERT_TRUE(task.set_joint_trajectory(trajectory, message));
  EXPECT_FALSE(task.joint_trajectory_status().succeeded);
  EXPECT_TRUE(task.joint_trajectory_status().active);
  EXPECT_NEAR(task.current_target()[0], 0.25, 1e-12);
  ASSERT_TRUE(task.cancel_joint_trajectory(message));
  EXPECT_FALSE(task.update(state, backend).active);
  EXPECT_FALSE(task.joint_trajectory_status().accepted);
}

TEST_F(JointTrajectoryInterpolationTest, AnUnreachedGoalStillTimesOut)
{
  JointTrajectoryTask task;
  configure(task);
  std::string message;
  ASSERT_TRUE(task.set_joint_trajectory(smooth_trajectory(), message));
  WholeBodyState state;
  state.joints_valid = true;
  state.joint_positions = Eigen::Vector2d::Zero();
  MockBackend backend;
  advance(task, 2.0);
  EXPECT_FALSE(task.update(state, backend).active);
  EXPECT_TRUE(task.joint_trajectory_status().timed_out);
  EXPECT_FALSE(task.joint_trajectory_status().succeeded);
  EXPECT_FALSE(task.joint_trajectory_status().active);
  ASSERT_TRUE(task.cancel_joint_trajectory(message));
  EXPECT_FALSE(task.joint_trajectory_status().accepted);
  EXPECT_FALSE(task.joint_trajectory_status().timed_out);
  EXPECT_TRUE(task.joint_trajectory_status().canceled);
  EXPECT_FALSE(task.update(state, backend).active);
  ASSERT_TRUE(task.set_enabled(false));
  ASSERT_TRUE(task.set_enabled(true));
  EXPECT_FALSE(task.joint_trajectory_status().accepted);
  EXPECT_FALSE(task.update(state, backend).active);
}

TEST_F(JointTrajectoryInterpolationTest, LinearRemainsDefault)
{
  JointTrajectoryTask task;
  configure(task);
  std::string message;
  ASSERT_TRUE(task.set_joint_trajectory(smooth_trajectory(), message));
  EXPECT_DOUBLE_EQ(sample(task, 0.25)[0], 0.25);
}

TEST_F(JointTrajectoryInterpolationTest, QuinticMatchesSmoothPolynomialAndJointOrder)
{
  JointTrajectoryTask task;
  configure(task, "quintic");
  auto trajectory = smooth_trajectory();
  trajectory.points[1].positions = {2.0, 1.0};
  std::string message;
  ASSERT_TRUE(task.set_joint_trajectory(trajectory, message));
  const auto quarter = sample(task, 0.25);
  EXPECT_NEAR(quarter[0], 0.103515625, 1e-12);
  EXPECT_NEAR(quarter[1], 1.0546875, 1e-12);
  EXPECT_NEAR(quarter[2], 5.625, 1e-12);
  EXPECT_NEAR(sample(task, 0.5)[0], 0.5, 1e-12);
  EXPECT_NEAR(sample(task, 0.0)[1], 0.0, 1e-12);
  EXPECT_NEAR(sample(task, 1.0)[0], 1.0, 1e-12);
}

TEST_F(JointTrajectoryInterpolationTest, QuinticRejectsIncompleteAndNonFiniteDerivatives)
{
  JointTrajectoryTask task;
  configure(task, "quintic");
  std::string message;
  EXPECT_FALSE(task.set_joint_trajectory(valid_trajectory(), message));
  auto trajectory = smooth_trajectory();
  trajectory.points[0].accelerations[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(task.set_joint_trajectory(trajectory, message));
  EXPECT_FALSE(task.joint_trajectory_status().accepted);
}

TEST_F(JointTrajectoryInterpolationTest, QuinticMatchesNonzeroEndpointDerivatives)
{
  JointTrajectoryTask task;
  configure(task, "quintic");
  auto trajectory = smooth_trajectory();
  trajectory.points[0].velocities = {0.1, 0.2};
  trajectory.points[0].accelerations = {0.3, 0.4};
  trajectory.points[1].velocities = {0.5, 0.6};
  trajectory.points[1].accelerations = {0.7, 0.8};
  std::string message;
  ASSERT_TRUE(task.set_joint_trajectory(trajectory, message));
  EXPECT_NEAR(sample(task, 1e-8)[1], 0.2, 1e-7);
  EXPECT_NEAR(sample(task, 1e-8)[2], 0.4, 1e-6);
  EXPECT_NEAR(sample(task, 1.0 - 1e-8)[1], 0.6, 1e-7);
  EXPECT_NEAR(sample(task, 1.0 - 1e-8)[2], 0.8, 1e-6);
}

}  // namespace task_priority_kinematic_control
