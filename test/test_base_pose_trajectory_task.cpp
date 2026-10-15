#include "task_priority_kinematic_control/tasks/base_pose_trajectory_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_trajectory_task.hpp"
#include <gtest/gtest.h>
#include <rcl/time.h>

namespace task_priority_kinematic_control
{
namespace
{
class EmptyBackend : public KinematicsBackend
{
public:
  void configure(const WholeBodyModel &, const std::string &, const rclcpp::Logger &) override {}
  void update(const WholeBodyState &) override {}
  FrameState get_frame_state(const std::string &) const override {return {};}
  Eigen::Isometry3d get_relative_transform(const std::string &, const std::string &) const override
  {return Eigen::Isometry3d::Identity();}
  const std::vector<CollisionCapsule> & collision_capsules() const override {return capsules_;}
  std::string name() const override {return "empty";}
private:
  std::vector<CollisionCapsule> capsules_;
};

class BaseTrajectoryTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    model = std::make_shared<WholeBodyModel>();
    model->configure("world_ned", "base", "left_tip", "right_tip", {"left_joint"}, {}, {0,1,2,3,4,5});
    clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
    ASSERT_EQ(rcl_enable_ros_time_override(clock->get_clock_handle()), RCL_RET_OK);
    set_time(1.0);
    task.configure("base_pose_trajectory", "task_priority_kinematic_control/BasePoseTrajectoryTask",
      {{"enabled", rclcpp::Parameter("enabled", true)}}, TaskContext{model, {}, clock});
    state.navigation_valid = state.joints_valid = state.ros_time_valid = true;
    state.joint_positions = Eigen::VectorXd::Zero(1);
  }
  void set_time(double seconds)
  {
    const int64_t ns = static_cast<int64_t>(seconds * 1e9);
    ASSERT_EQ(rcl_set_ros_time_override(clock->get_clock_handle(), ns), RCL_RET_OK);
    state.ros_time_ns = ns;
  }
  trajectory_msgs::msg::MultiDOFJointTrajectory trajectory(double final_x = 0.1)
  {
    using namespace sura_manipulation_actions;
    trajectory_msgs::msg::MultiDOFJointTrajectory result;
    result.header.frame_id = "world_ned"; result.joint_names = {"floating_base"};
    BaseCoordinates begin = BaseCoordinates::Zero(), end = begin;
    end(0) = final_x;
    builtin_interfaces::msg::Duration zero, final;
    final.sec = 4;
    result.points = {baseTrajectoryPoint(begin, begin, begin, zero),
      baseTrajectoryPoint(end, begin, begin, final)};
    return result;
  }
  std::shared_ptr<WholeBodyModel> model;
  rclcpp::Clock::SharedPtr clock;
  WholeBodyState state;
  BasePoseTrajectoryTask task;
  EmptyBackend backend;
};

TEST_F(BaseTrajectoryTest, SharedFutureStampAndPausedRosTime)
{
  auto base = trajectory();
  base.header.stamp.sec = 2;
  std::string error;
  ASSERT_TRUE(task.set_base_trajectory(base, std::vector<double>(6, 0.01), 0.5, error)) << error;
  JointTrajectoryTask joints;
  joints.configure("left_joint_trajectory", "task_priority_kinematic_control/JointTrajectoryTask",
    {{"joint_names", rclcpp::Parameter("joint_names", std::vector<std::string>{"left_joint"})},
      {"interpolation_method", rclcpp::Parameter("interpolation_method", "quintic")}},
    TaskContext{model, {}, clock});
  trajectory_msgs::msg::JointTrajectory arm;
  arm.header.stamp = base.header.stamp; arm.joint_names = {"left_joint"};
  for (const auto & p : base.points) {
    trajectory_msgs::msg::JointTrajectoryPoint q;
    q.time_from_start = p.time_from_start; q.positions = {p.transforms.front().translation.x};
    q.velocities = {0.0}; q.accelerations = {0.0}; arm.points.push_back(q);
  }
  ASSERT_TRUE(joints.set_joint_trajectory(arm, error));
  EXPECT_EQ(task.update(state, backend).status_message, "waiting_for_start");
  EXPECT_EQ(joints.update(state, backend).status_message, "waiting_for_start");
  set_time(4.0);  // Both sample the middle of the exact same four-second interval.
  task.update(state, backend); joints.update(state, backend);
  EXPECT_NEAR(task.current_target()[0], 0.05, 1e-12);
  EXPECT_NEAR(joints.current_target()[0], 0.05, 1e-12);
  const auto reference = task.current_target();
  task.update(state, backend);
  EXPECT_EQ(task.current_target(), reference);
}

TEST_F(BaseTrajectoryTest, LimitsApplyToFeedbackAndAcceleration)
{
  auto base = trajectory(1.0);
  base.points.front() = base.points.back();
  base.points.front().time_from_start = builtin_interfaces::msg::Duration();
  task.set_motion_limits(std::vector<double>(6, 0.1), std::vector<double>(6, 0.2));
  std::string error;
  ASSERT_TRUE(task.set_base_trajectory(base, std::vector<double>(6, 0.01), 0.5, error));
  task.update(state, backend);
  set_time(1.05);
  const auto limited = task.update(state, backend);
  EXPECT_NEAR(limited.desired_velocity(0), 0.01, 1e-9);
  set_time(1.55);
  EXPECT_LE(std::abs(task.update(state, backend).desired_velocity(0)), 0.1 + 1e-12);
}

TEST_F(BaseTrajectoryTest, CancellationHoldsMeasuredAndDisableCannotResume)
{
  std::string error;
  ASSERT_TRUE(task.set_base_trajectory(trajectory(), std::vector<double>(6, 0.01), 0.5, error));
  set_time(2.0); state.base_position.x() = 0.025;
  task.hold_measured(state);
  const auto hold = task.update(state, backend);
  EXPECT_TRUE(hold.active);
  EXPECT_TRUE(hold.desired_velocity.isZero());
  EXPECT_NEAR(task.current_target()[0], 0.025, 1e-12);
  task.set_enabled(false); task.set_enabled(true);
  EXPECT_FALSE(task.base_trajectory_status().accepted);
  EXPECT_FALSE(task.update(state, backend).active);
}

TEST_F(BaseTrajectoryTest, RejectsInvalidFrameAndQuaternionAndOldStamp)
{
  std::string error;
  auto base = trajectory(); base.header.frame_id = "base";
  EXPECT_FALSE(task.set_base_trajectory(base, std::vector<double>(6, 0.01), 0.5, error));
  base = trajectory(); base.points.front().transforms.front().rotation.w = 0.0;
  EXPECT_FALSE(task.set_base_trajectory(base, std::vector<double>(6, 0.01), 0.5, error));
  base = trajectory(); base.header.stamp.nanosec = 1;
  EXPECT_FALSE(task.set_base_trajectory(base, std::vector<double>(6, 0.01), 0.5, error));
}

TEST_F(BaseTrajectoryTest, FirstControllerCycleCanPredateAdmissionClock)
{
  std::string error;
  set_time(1.05);
  auto base = trajectory(); base.header.stamp.sec = 2;
  ASSERT_TRUE(task.set_base_trajectory(base, std::vector<double>(6, 0.01), 0.5, error));
  JointTrajectoryTask joints;
  joints.configure("left_joint_trajectory", "task_priority_kinematic_control/JointTrajectoryTask",
    {{"joint_names", rclcpp::Parameter("joint_names", std::vector<std::string>{"left_joint"})}},
    TaskContext{model, {}, clock});
  trajectory_msgs::msg::JointTrajectory arm;
  arm.header.stamp = base.header.stamp; arm.joint_names = {"left_joint"};
  trajectory_msgs::msg::JointTrajectoryPoint point; point.positions = {0.0};
  arm.points.push_back(point); point.positions = {0.1}; point.time_from_start.sec = 4; arm.points.push_back(point);
  ASSERT_TRUE(joints.set_joint_trajectory(arm, error));
  // controller_manager sampled this cycle's time before admission acquired
  // the host mutex. This is not a clock rewind between two control cycles.
  state.ros_time_ns = 1040000000LL;
  EXPECT_TRUE(task.update(state, backend).active);
  EXPECT_TRUE(joints.update(state, backend).active);
  EXPECT_FALSE(task.base_trajectory_status().timed_out);
  state.ros_time_ns = 1020000000LL;
  EXPECT_FALSE(task.update(state, backend).active);
  EXPECT_FALSE(joints.update(state, backend).active);
  EXPECT_TRUE(task.base_trajectory_status().timed_out);
}
}
}  // namespace task_priority_kinematic_control
