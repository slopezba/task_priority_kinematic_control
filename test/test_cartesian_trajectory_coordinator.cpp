#include "task_priority_kinematic_control/core/cartesian_trajectory_coordinator.hpp"
#include <gtest/gtest.h>
#include <rcl/time.h>
#include <rclcpp/rclcpp.hpp>
#include <algorithm>

namespace task_priority_kinematic_control
{
namespace
{
class SimpleWholeBodyBackend : public KinematicsBackend
{
public:
  explicit SimpleWholeBodyBackend(std::shared_ptr<WholeBodyModel> model) : model_(std::move(model)) {}
  void configure(const WholeBodyModel &, const std::string &, const rclcpp::Logger &) override {}
  void update(const WholeBodyState & state) override {state_ = state;}
  FrameState get_frame_state(const std::string & frame) const override {return compute_frame_state(state_, frame);}
  FrameState compute_frame_state(const WholeBodyState & state, const std::string & frame) const override
  {
    FrameState result;
    sura_manipulation_actions::BaseCoordinates base;
    base << state.base_position, state.base_rpy;
    result.pose = sura_manipulation_actions::basePose(base);
    result.jacobian = Eigen::MatrixXd::Zero(6, model_->total_dofs());
    if (frame == "left_tip") {result.pose.translation().x() += state.joint_positions(0);}
    else if (frame == "right_tip") {result.pose.translation().x() += state.joint_positions(model_->left_arm_dofs());}
    else if (frame != "base") {throw std::runtime_error("unknown frame");}
    return result;
  }
  Eigen::Isometry3d get_relative_transform(const std::string & from, const std::string & to) const override
  {return get_frame_state(from).pose.inverse() * get_frame_state(to).pose;}
  const std::vector<CollisionCapsule> & collision_capsules() const override {return capsules_;}
  std::string name() const override {return "simple";}
private:
  std::shared_ptr<WholeBodyModel> model_;
  WholeBodyState state_;
  std::vector<CollisionCapsule> capsules_;
};

class CoordinatedExecutionTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}
  virtual std::vector<std::string> left_names() const {return {"left_joint"};}
  virtual std::vector<std::string> right_names() const {return {"right_joint"};}
  void SetUp() override
  {
    node = std::make_shared<rclcpp::Node>("cartesian_coordinator_test");
    model = std::make_shared<WholeBodyModel>();
    model->configure("world_ned", "base", "left_tip", "right_tip", left_names(), right_names(), {0,1,2,3,4,5});
    clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
    ASSERT_EQ(rcl_enable_ros_time_override(clock->get_clock_handle()), RCL_RET_OK);
    set_time(1.0);
    node->declare_parameter("task_ids", std::vector<std::string>{"base_pose_trajectory", "left_joint_trajectory",
      "right_joint_trajectory", "left_arm_nominal", "right_arm_nominal", "base_pose", "stabilize_limits"});
    declare_task("base_pose_trajectory", "BasePoseTrajectoryTask", false, {});
    declare_task("left_joint_trajectory", "JointTrajectoryTask", false, left_names());
    declare_task("right_joint_trajectory", "JointTrajectoryTask", false, right_names());
    declare_task("left_arm_nominal", "JointNominalTask", false, left_names());
    declare_task("right_arm_nominal", "JointNominalTask", false, right_names());
    declare_task("base_pose", "FramePoseTask", true, {});
    node->declare_parameter("tasks.base_pose.frame_id", std::string("base"));
    declare_task("stabilize_limits", "StabilizeLimitsTask", true, {});
    node->declare_parameter("tasks.stabilize_limits.lower_limits", std::vector<double>{-0.35,-0.35});
    node->declare_parameter("tasks.stabilize_limits.upper_limits", std::vector<double>{0.35,0.35});
    manager = std::make_unique<TaskManager>(node->get_node_parameters_interface(), node->get_logger());
    manager->configure(TaskContext{model, {}, clock});
    backend = std::make_unique<SimpleWholeBodyBackend>(model);
    state.navigation_valid = state.joints_valid = state.ros_time_valid = true;
    state.joint_positions = Eigen::VectorXd::Zero(model->left_arm_dofs() + model->right_arm_dofs());
    state.joint_velocities = Eigen::VectorXd::Zero(state.joint_positions.size());
    backend->update(state);
    coordinator.configure(*manager, model, clock, CartesianTrajectoryCoordinator::Settings());
    manager->set_mutation_guard([this](const std::string & id, std::string & message) {
      return coordinator.allow_mutation(id, state, *backend, message);
    });
  }
  void declare_task(const std::string & id, const std::string & plugin, bool enabled,
    const std::vector<std::string> & names)
  {
    node->declare_parameter("tasks."+id+".plugin", "task_priority_kinematic_control/"+plugin);
    node->declare_parameter("tasks."+id+".enabled", enabled);
    node->declare_parameter("tasks."+id+".priority", id == "stabilize_limits" ? 0.0 : 3.0);
    if (!names.empty()) {node->declare_parameter("tasks."+id+".joint_names", names);}
  }
  void set_time(double seconds)
  {
    const int64_t ns = static_cast<int64_t>(seconds * 1e9);
    ASSERT_EQ(rcl_set_ros_time_override(clock->get_clock_handle(), ns), RCL_RET_OK);
    state.ros_time_ns = ns;
  }
  std::shared_ptr<TaskBase> find(const std::string & id)
  {for (auto task : manager->tasks()) {if (task->id() == id) {return task;}} return {};}
  CartesianTrajectoryCoordinator::Action::Goal goal()
  {
    using namespace sura_manipulation_actions;
    CartesianTrajectoryCoordinator::Action::Goal g;
    g.arm = "custom_left"; g.tip_frame = "left_tip";
    g.available_dofs = {true,false,false,false,false,false,true};
    g.base_reference_pose.header.frame_id = g.target_pose.header.frame_id = g.tolerance_reference_pose.header.frame_id = "world_ned";
    g.base_reference_pose.pose.orientation.w = g.target_pose.pose.orientation.w = g.tolerance_reference_pose.pose.orientation.w = 1.0;
    g.target_pose.pose.position.x = 0.3;
    g.base_trajectory.header.frame_id = "world_ned"; g.base_trajectory.header.stamp.sec = 2;
    g.base_trajectory.joint_names = {"floating_base"};
    BaseCoordinates start = BaseCoordinates::Zero(), end = start;
    end(0) = 0.1;
    builtin_interfaces::msg::Duration zero, final; final.sec = 4;
    g.base_trajectory.points = {baseTrajectoryPoint(start, start, start, zero), baseTrajectoryPoint(end, start, start, final)};
    g.joint_trajectory.header.stamp = g.base_trajectory.header.stamp;
    g.joint_trajectory.joint_names = {"left_joint"};
    trajectory_msgs::msg::JointTrajectoryPoint first, last;
    first.positions = {0.0}; first.velocities = {0.0}; first.accelerations = {0.0};
    last = first; last.positions = {0.2}; last.time_from_start = final;
    g.joint_trajectory.points = {first,last};
    g.base_limits.max_velocity = std::vector<double>(6, 0.2);
    g.base_limits.max_acceleration = std::vector<double>(6, 0.3);
    g.tracking_tolerances.base_path = g.tracking_tolerances.base_goal = std::vector<double>(6, 0.01);
    g.tracking_tolerances.joint_path = g.tracking_tolerances.joint_goal = {0.01};
    g.tracking_tolerances.tip_path = g.tracking_tolerances.tip_goal = std::vector<double>(6, 0.01);
    g.tracking_tolerances.goal_time_tolerance.nanosec = 500000000U;
    return g;
  }
  void update()
  {
    backend->update(state);
    manager->update_all_into(state, *backend);
    if (coordinator.update(state, *backend)) {manager->update_all_into(state, *backend);}
  }
  std::shared_ptr<rclcpp::Node> node;
  std::shared_ptr<WholeBodyModel> model;
  rclcpp::Clock::SharedPtr clock;
  std::unique_ptr<TaskManager> manager;
  std::unique_ptr<SimpleWholeBodyBackend> backend;
  WholeBodyState state;
  CartesianTrajectoryCoordinator coordinator;
};

class JointOrderExecutionTest : public CoordinatedExecutionTest
{
protected:
  std::vector<std::string> left_names() const override
  {return {"left_b", "left_c", "left_d", "left_e"};}
  std::vector<std::string> right_names() const override
  {return {"right_b", "right_c", "right_d", "right_e"};}

  CartesianTrajectoryCoordinator::Action::Goal reversed_goal(bool right = false)
  {
    auto g = goal();
    auto names = right ? right_names() : left_names();
    std::reverse(names.begin(), names.end());
    g.arm = right ? "custom_right" : "custom_left";
    g.tip_frame = right ? "right_tip" : "left_tip";
    g.joint_trajectory.joint_names = names;
    // Only b is movable; all values below use the incoming e,d,c,b order.
    g.available_dofs = {true,false,false,false,false,false,false,false,false,true};
    auto & first = g.joint_trajectory.points.front();
    first.positions = {0.4, 0.3, 0.2, 0.1};
    first.velocities = first.accelerations = std::vector<double>(4, 0.0);
    auto & last = g.joint_trajectory.points.back();
    last.positions = {0.4, 0.3, 0.2, 0.3};
    last.velocities = last.accelerations = std::vector<double>(4, 0.0);
    g.tracking_tolerances.joint_path = {0.001, 0.002, 0.003, 0.04};
    g.tracking_tolerances.joint_goal = {0.001, 0.002, 0.003, 0.02};
    g.tracking_tolerances.tip_path.assign(6, 1.0);
    g.tracking_tolerances.tip_goal.assign(6, 0.05);
    g.target_pose.pose.position.x = 0.4;
    const size_t offset = right ? model->left_arm_dofs() : 0;
    for (size_t i = 0; i < 4; ++i) {state.joint_positions(offset + i) = 0.1 * (i + 1);}
    return g;
  }
};

TEST_F(JointOrderExecutionTest, AcceptsReversedLeftOrderAndPreservesMaskTolerancesAndFeedback)
{
  const auto g = reversed_goal();
  std::string message;
  ASSERT_TRUE(coordinator.validate_goal(g, state, *backend, message)) << message;
  ASSERT_TRUE(coordinator.install(g, state, *backend, message)) << message;
  auto task = std::dynamic_pointer_cast<JointTrajectoryTask>(find("left_joint_trajectory"));
  Eigen::VectorXd sample;
  ASSERT_TRUE(task->sample_trajectory(4.0, sample));
  ASSERT_EQ(sample.size(), 4);
  EXPECT_NEAR(sample(0), 0.3, 1e-12);
  EXPECT_NEAR(sample(1), 0.2, 1e-12);
  EXPECT_NEAR(sample(2), 0.3, 1e-12);
  EXPECT_NEAR(sample(3), 0.4, 1e-12);
  // b may lag by 0.03; this would exceed e's tolerance if not remapped.
  set_time(4.0); state.base_position.x() = 0.05; state.joint_positions(0) = 0.17;
  update();
  EXPECT_TRUE(coordinator.status().running) << coordinator.status().message;
  EXPECT_NEAR(coordinator.status().joint_error(0), 0.0, 1e-12);  // e first in feedback.
  EXPECT_NEAR(coordinator.status().joint_error(3), 0.03, 1e-12); // b last in feedback.
  set_time(6.0); state.base_position.x() = 0.1; state.joint_positions(0) = 0.285;
  update();
  EXPECT_TRUE(coordinator.status().success) << coordinator.status().message;
  EXPECT_EQ(g.joint_trajectory.joint_names.front(), "left_e");
}

TEST_F(JointOrderExecutionTest, AcceptsReversedRightOrder)
{
  auto g = reversed_goal(true);
  std::string message;
  ASSERT_TRUE(coordinator.install(g, state, *backend, message)) << message;
  EXPECT_TRUE(find("right_joint_trajectory")->enabled());
  EXPECT_TRUE(find("left_arm_nominal")->enabled());
  set_time(6.0); state.base_position.x() = 0.1; state.joint_positions(4) = 0.3;
  update();
  EXPECT_TRUE(coordinator.status().success) << coordinator.status().message;
}

TEST_F(JointOrderExecutionTest, ReordersVelocitiesAndAccelerationsWithTheirPositions)
{
  auto g = reversed_goal();
  std::fill(g.available_dofs.begin() + 6, g.available_dofs.end(), true);
  auto & first = g.joint_trajectory.points.front();
  auto & last = g.joint_trajectory.points.back();
  first.velocities = {0.004, 0.003, 0.002, 0.001};
  last.velocities = {0.008, 0.006, 0.004, 0.002};
  first.accelerations = {0.0004, 0.0003, 0.0002, 0.0001};
  last.accelerations = {0.0008, 0.0006, 0.0004, 0.0002};
  std::string message;
  ASSERT_TRUE(coordinator.install(g, state, *backend, message)) << message;
  auto task = std::dynamic_pointer_cast<JointTrajectoryTask>(find("left_joint_trajectory"));
  Eigen::VectorXd sample;
  ASSERT_TRUE(task->sample_trajectory(2.0, sample));
  for (size_t i = 0; i < 4; ++i) {
    const size_t incoming = 3 - i;
    // Quintic Hermite interpolation at half of a four-second interval.
    const double expected = 0.5 * (first.positions[incoming] + last.positions[incoming]) +
      0.625 * (first.velocities[incoming] - last.velocities[incoming]) +
      0.25 * (first.accelerations[incoming] + last.accelerations[incoming]);
    EXPECT_NEAR(sample(i), expected, 1e-12);
  }
}

TEST_F(JointOrderExecutionTest, RejectsMalformedJointListsAndArraysWithoutInstallingTasks)
{
  const auto valid = reversed_goal();
  std::vector<CartesianTrajectoryCoordinator::Action::Goal> invalid(10, valid);
  invalid[0].joint_trajectory.joint_names[0] = "left_b";  // Duplicate, missing e.
  invalid[1].joint_trajectory.joint_names[0] = "unknown";
  invalid[2].joint_trajectory.points.front().positions.pop_back();
  invalid[3].joint_trajectory.points.back().velocities.pop_back();
  invalid[4].joint_trajectory.points.back().accelerations.clear();
  invalid[5].tracking_tolerances.joint_path.pop_back();
  invalid[6].available_dofs.pop_back();
  invalid[7].joint_trajectory.joint_names.pop_back();
  invalid[8].tracking_tolerances.joint_goal.pop_back();
  invalid[9].joint_trajectory.points.front().effort = {1.0};
  for (const auto & g : invalid) {
    std::string message;
    EXPECT_FALSE(coordinator.install(g, state, *backend, message));
    EXPECT_FALSE(message.empty());
    EXPECT_TRUE(find("base_pose")->enabled());
    EXPECT_FALSE(find("base_pose_trajectory")->enabled());
    EXPECT_FALSE(coordinator.owns_tasks());
  }
}

TEST_F(CoordinatedExecutionTest, EmptyBoundsAndExplicitZeroTolerancesAreValid)
{
  auto g = goal();
  g.tracking_tolerances.base_goal.assign(6, 0.0);
  g.tracking_tolerances.joint_goal.assign(1, 0.0);
  std::string message;
  EXPECT_TRUE(coordinator.validate_goal(g, state, *backend, message)) << message;
}

TEST_F(CoordinatedExecutionTest, RejectsDisabledAxisBeforeChangingAnyTask)
{
  auto g = goal(); g.available_dofs[0] = false;
  std::string message;
  EXPECT_FALSE(coordinator.install(g, state, *backend, message));
  EXPECT_TRUE(find("base_pose")->enabled());
  EXPECT_FALSE(find("base_pose_trajectory")->enabled());
  EXPECT_FALSE(coordinator.owns_tasks());
}

TEST_F(CoordinatedExecutionTest, KeepsSafetyIndependentAndHoldsOtherArmJoints)
{
  auto g = goal(); g.available_dofs[3] = true;
  auto final = sura_manipulation_actions::BaseCoordinates::Zero().eval();
  final(0) = 0.1; final(3) = 0.4;  // Outside the independent stabilize_limits guard.
  g.base_trajectory.points.back() = sura_manipulation_actions::baseTrajectoryPoint(final,
    sura_manipulation_actions::BaseCoordinates::Zero(), sura_manipulation_actions::BaseCoordinates::Zero(),
    g.base_trajectory.points.back().time_from_start);
  state.joint_positions(1) = 0.37;
  std::string message;
  ASSERT_TRUE(coordinator.install(g, state, *backend, message)) << message;
  EXPECT_TRUE(find("stabilize_limits")->enabled());
  EXPECT_FALSE(find("base_pose")->enabled());
  EXPECT_TRUE(find("right_arm_nominal")->enabled());
  EXPECT_NEAR(find("right_arm_nominal")->current_target()[0], 0.37, 1e-12);
  EXPECT_FALSE(manager->set_task_enabled("base_pose", true, message));
}

TEST_F(CoordinatedExecutionTest, SuccessKeepsReservationAndReleaseCapturesMeasuredPose)
{
  std::string message;
  ASSERT_TRUE(coordinator.install(goal(), state, *backend, message)) << message;
  set_time(6.0); state.base_position.x() = 0.1; state.joint_positions(0) = 0.2;
  update();
  EXPECT_TRUE(coordinator.status().success);
  EXPECT_TRUE(coordinator.status().holding);
  EXPECT_TRUE(coordinator.reserved("base_pose"));
  ASSERT_TRUE(coordinator.release(state, *backend, message));
  EXPECT_FALSE(coordinator.owns_tasks());
  EXPECT_TRUE(find("base_pose")->enabled());
  EXPECT_NEAR(find("base_pose")->current_target()[0], 0.1, 1e-12);
  EXPECT_FALSE(find("base_pose_trajectory")->enabled());
}

TEST_F(CoordinatedExecutionTest, ToolDeviationAbortsBothAndHoldsMeasuredState)
{
  auto g = goal(); g.tracking_tolerances.base_path.assign(6, 1.0);
  g.tracking_tolerances.joint_path.assign(1, 1.0);
  std::string message;
  ASSERT_TRUE(coordinator.install(g, state, *backend, message)) << message;
  set_time(4.0);  // Middle reference x=0.15; the measured tool is still at x=0.
  update();
  EXPECT_TRUE(coordinator.status().complete);
  EXPECT_FALSE(coordinator.status().success);
  EXPECT_TRUE(coordinator.status().holding);
  EXPECT_NEAR(find("base_pose_trajectory")->current_target()[0], 0.0, 1e-12);
  EXPECT_NEAR(find("left_joint_trajectory")->current_target()[0], 0.0, 1e-12);
}

TEST_F(CoordinatedExecutionTest, SafetyCommandExceedingGoalCapAbortsWithoutClippingSafety)
{
  std::string message;
  ASSERT_TRUE(coordinator.install(goal(), state, *backend, message)) << message;
  set_time(1.1);
  WholeBodyCommand command;
  command.generalized_velocity = Eigen::VectorXd::Zero(model->total_dofs());
  command.generalized_velocity(0) = 0.25;  // A higher-priority guard can override the tracking objective.
  EXPECT_TRUE(coordinator.observe_command(command, state, *backend));
  EXPECT_FALSE(coordinator.status().success);
  EXPECT_TRUE(coordinator.status().holding);
  EXPECT_TRUE(find("stabilize_limits")->enabled());
  EXPECT_DOUBLE_EQ(command.generalized_velocity(0), 0.25);
}

TEST_F(CoordinatedExecutionTest, ArrivalAfterGoalDeadlineCannotReportSuccess)
{
  std::string message;
  ASSERT_TRUE(coordinator.install(goal(), state, *backend, message)) << message;
  set_time(6.6); state.base_position.x() = 0.1; state.joint_positions(0) = 0.2;
  update();
  EXPECT_TRUE(coordinator.status().complete);
  EXPECT_FALSE(coordinator.status().success);
  EXPECT_TRUE(coordinator.status().holding);
}

TEST_F(CoordinatedExecutionTest, MotionOfOtherArmAbortsEvenWhenSelectedToolIsOnPath)
{
  std::string message;
  ASSERT_TRUE(coordinator.install(goal(), state, *backend, message)) << message;
  set_time(4.0); state.base_position.x() = 0.05; state.joint_positions(0) = 0.1;
  state.joint_positions(1) = 0.06;
  update();
  EXPECT_TRUE(coordinator.status().complete);
  EXPECT_FALSE(coordinator.status().success);
  EXPECT_TRUE(coordinator.status().holding);
  EXPECT_NEAR(find("right_arm_nominal")->current_target()[0], 0.06, 1e-12);
  EXPECT_TRUE(find("stabilize_limits")->enabled());
}
}
}  // namespace task_priority_kinematic_control
