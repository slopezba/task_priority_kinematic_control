#include "task_priority_kinematic_control/core/nominal_trajectory.hpp"
#include "task_priority_kinematic_control/kinematics/kdl_kinematics_backend.hpp"
#include "task_priority_kinematic_control/tasks/frame_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effector_position_task.hpp"
#include "task_priority_kinematic_control/tasks/frame_position_norm_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/frame_orientation_norm_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_nominal_task.hpp"
#include "task_priority_kinematic_control/tasks/robot_nominal_task.hpp"
#include "task_priority_kinematic_control/tasks/self_collision_avoidance_task.hpp"
#include "task_priority_kinematic_control/tasks/stabilize_limits_task.hpp"
#include "sura_manipulation_actions/cartesian_pose_path.hpp"
#include "sura_manipulation_actions/whole_body_nominal.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <thread>

namespace task_priority_kinematic_control
{
namespace
{
constexpr char robot[] = R"(
<robot name="nominal_fixture">
  <link name="base"/><link name="left"/><link name="right"/>
  <link name="left_end"/><link name="right_end"/>
  <joint name="l" type="prismatic"><parent link="base"/><child link="left"/>
    <axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="10" velocity="1"/></joint>
  <joint name="r" type="prismatic"><parent link="base"/><child link="right"/>
    <origin xyz="0.09 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="10" velocity="1"/></joint>
  <joint name="le" type="fixed"><parent link="left"/><child link="left_end"/><origin xyz="0 0 0.1"/></joint>
  <joint name="re" type="fixed"><parent link="right"/><child link="right_end"/><origin xyz="0 0 0.1"/></joint>
</robot>)";

class RobotNominalTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    model = std::make_shared<WholeBodyModel>();
    model->configure("world", "base", "left", "right", {"l"}, {"r"}, {0, 1, 2, 3, 4, 5});
    backend = std::make_shared<KDLKinematicsBackend>();
    backend->configure(*model, robot, rclcpp::get_logger("robot_nominal_test"));
    state.joints_valid = state.navigation_valid = state.ros_time_valid = true;
    state.ros_time_ns = 1000000000LL;
    state.joint_positions = state.joint_velocities = Eigen::VectorXd::Zero(2);
    backend->update(state);
    solver.configure(model->total_dofs());
    solver.set_velocity_limits(Eigen::VectorXd::Constant(model->total_dofs(), 1.0));
    context = TaskContext{model, backend};
  }
  std::shared_ptr<RobotNominalTask> nominal_task(const std::vector<std::string> & sync = {})
  {
    auto task = std::make_shared<RobotNominalTask>();
    task->configure("robot_nominal", "task_priority_kinematic_control/RobotNominalTask", {
      {"gain", rclcpp::Parameter("gain", std::vector<double>(8, 1.0))},
      {"joint_command_timeout", rclcpp::Parameter("joint_command_timeout", 0.05)},
      {"sync_tasks", rclcpp::Parameter("sync_tasks", sync)}}, context);
    return task;
  }
  std::shared_ptr<WholeBodyModel> model;
  KinematicsBackendPtr backend;
  HierarchySolver solver;
  WholeBodyState state;
  TaskContext context;
};

TEST_F(RobotNominalTest, StreamsRowsAndExpires)
{
  auto task = nominal_task();
  EXPECT_FALSE(task->update(state, *backend).active);
  RobotNominalTask::Coordinates base;
  base << 0.1, -0.2, 0.3, 0.0, 0.0, 0.4;
  RobotNominalTask::Coordinates rates = RobotNominalTask::Coordinates::Zero();
  rates(0) = 0.05;
  std::string message;
  ASSERT_TRUE(task->set_reference(base, rates, Eigen::Vector2d(0.5, -0.5), Eigen::Vector2d(0.1, 0.0), message))
    << message;
  const auto out = task->update(state, *backend);
  ASSERT_TRUE(out.active);
  // Roll and pitch are inactive by default: x, y, z, yaw and both joints.
  ASSERT_EQ(out.jacobian.rows(), 6);
  EXPECT_NEAR(out.error(0), 0.1, 1e-12);
  EXPECT_NEAR(out.error(3), 0.4, 1e-12);
  EXPECT_NEAR(out.error(4), 0.5, 1e-12);
  EXPECT_NEAR(out.feedforward(0), 0.05, 1e-12);
  EXPECT_NEAR(out.feedforward(4), 0.1, 1e-12);
  EXPECT_DOUBLE_EQ(out.jacobian(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(out.jacobian(4, 6), 1.0);
  EXPECT_DOUBLE_EQ(out.jacobian(5, 7), 1.0);
  std::this_thread::sleep_for(std::chrono::milliseconds(80));
  EXPECT_FALSE(task->update(state, *backend).active);
  EXPECT_FALSE(task->has_live_reference());
}

TEST_F(RobotNominalTest, ReferenceSynchronizesHolds)
{
  auto base_hold = std::make_shared<FramePoseTask>();
  base_hold->configure("base_pose", "FramePoseTask", {
    {"frame_id", rclcpp::Parameter("frame_id", "base")}}, context);
  auto arm_hold = std::make_shared<JointNominalTask>();
  arm_hold->configure("left_arm_nominal", "JointNominalTask", {
    {"joint_names", rclcpp::Parameter("joint_names", std::vector<std::string>{"l"})}}, context);
  auto task = nominal_task({"base_pose", "left_arm_nominal"});
  std::vector<std::shared_ptr<TaskBase>> tasks{base_hold, arm_hold, task};
  sura_manipulation_actions::msg::RobotNominalReference reference;
  reference.header.frame_id = "world";
  reference.base_pose.position.x = 0.3;
  reference.base_pose.orientation.w = 1.0;
  reference.joint_names = {"r", "l"};
  reference.positions = {0.2, 0.7};
  reference.velocities = {0.0, 0.0};
  std::string message;
  ASSERT_TRUE(apply_robot_nominal_reference(tasks, *task, reference, message)) << message;
  EXPECT_NEAR(arm_hold->current_target().at(0), 0.7, 1e-12);
  EXPECT_NEAR(base_hold->current_target().at(0), 0.3, 1e-12);
  EXPECT_NEAR(task->joint_target()(1), 0.2, 1e-12);
  reference.joint_names = {"l"};
  reference.positions = {0.1};
  reference.velocities = {0.0};
  EXPECT_FALSE(apply_robot_nominal_reference(tasks, *task, reference, message));
}

TEST_F(RobotNominalTest, SamplerBuildsConsistentRestToRestMessages)
{
  std::vector<double> times{0.0, 0.5, 1.0};
  std::vector<sura_manipulation_actions::BaseCoordinates> base(3, sura_manipulation_actions::BaseCoordinates::Zero());
  base[1](0) = 0.1; base[2](0) = 0.2; base[2](5) = 0.3;
  std::vector<Eigen::VectorXd> joints{Eigen::Vector2d(0, 0), Eigen::Vector2d(0.2, 0), Eigen::Vector2d(0.4, 0.1)};
  trajectory_msgs::msg::JointTrajectory joint_msg;
  trajectory_msgs::msg::MultiDOFJointTrajectory base_msg;
  std::string error;
  ASSERT_TRUE(sura_manipulation_actions::buildWholeBodyNominal(times, base, joints, {"l", "r"}, "world",
    joint_msg, base_msg, error)) << error;
  sura_manipulation_actions::WholeBodyNominal nominal;
  ASSERT_TRUE(nominal.init(joint_msg, base_msg, error)) << error;
  sura_manipulation_actions::WholeBodyNominalSample sample;
  ASSERT_TRUE(nominal.sample(0.5, sample, error)) << error;
  EXPECT_NEAR(sample.base(0), 0.1, 1e-9);
  EXPECT_NEAR(sample.joints(0), 0.2, 1e-9);
  EXPECT_NEAR(sample.base_rates(0), 0.2, 1e-9);
  ASSERT_TRUE(nominal.sample(2.0, sample, error));
  EXPECT_NEAR(sample.base(5), 0.3, 1e-9);
  EXPECT_NEAR(sample.joint_velocities.norm(), 0.0, 1e-12);
}

class NominalPlannerTest : public RobotNominalTest
{
protected:
  void SetUp() override
  {
    RobotNominalTest::SetUp();
    auto limits = std::make_shared<JointLimitsTask>();
    limits->configure("joint_limits", "JointLimitsTask", {
      {"lower_limits", rclcpp::Parameter("lower_limits", std::vector<double>{-1, -1})},
      {"upper_limits", rclcpp::Parameter("upper_limits", std::vector<double>{1, 1})}}, context);
    auto level = std::make_shared<FramePoseTask>();
    level->configure("stabilize", "FramePoseTask", {
      {"frame_id", rclcpp::Parameter("frame_id", "base")},
      {"roll_pitch_control", rclcpp::Parameter("roll_pitch_control", true)},
      {"activation", rclcpp::Parameter("activation", std::vector<int64_t>{0, 0, 0, 1, 1, 0})}}, context);
    pose = std::make_shared<FramePoseTask>();
    pose->configure("left_pose", "FramePoseTask", {
      {"frame_id", rclcpp::Parameter("frame_id", "left")},
      {"gain", rclcpp::Parameter("gain", std::vector<double>(6, 2.0))}}, context);
    auto base_hold = std::make_shared<FramePoseTask>();
    base_hold->configure("base_pose", "FramePoseTask", {
      {"frame_id", rclcpp::Parameter("frame_id", "base")},
      {"activation", rclcpp::Parameter("activation", std::vector<int64_t>{1, 1, 1, 0, 0, 1})}}, context);
    for (const auto & task : std::vector<std::shared_ptr<TaskBase>>{limits, level, pose, base_hold}) {
      task->update(state, *backend);
    }
    robot_nominal = nominal_task({"base_pose"});
    tasks = {limits, level, pose, base_hold, robot_nominal};
  }
  Preparation::Goal request(double displacement, double duration)
  {
    const auto start = backend->get_frame_state("left").pose;
    auto target = start;
    target.translation().z() += displacement;
    target.translation().x() += 0.5 * displacement;
    sura_manipulation_actions::CartesianPosePath path;
    std::string reason;
    EXPECT_TRUE(path.init(start, target, 1, 1, 1, 1, 0.01, 0.05, 10000, reason)) << reason;
    Preparation::Goal out;
    out.trajectory = path.descriptor("world");
    out.trajectory.duration = duration;
    out.task_id = "left_pose"; out.tip_frame = "left";
    out.tolerance_orientation.w = 1.0;
    out.path_tolerance = {0.02, 0.02, 0.02, 0.3, 0.3, 0.3};
    out.goal_tolerance = {0.003, 0.003, 0.003, 0.03, 0.03, 0.03};
    out.goal_time_tolerance = 1.0;
    out.robot_description = robot;
    out.robot_description_semantic = R"(<robot name="nominal_fixture">
      <virtual_joint name="root" type="floating" parent_frame="world" child_link="base"/></robot>)";
    out.scene.robot_model_name = "nominal_fixture";
    out.scene.robot_state.joint_state.name = {"l", "r"};
    out.scene.robot_state.joint_state.position = {0.0, 0.0};
    out.full_trajectory = true;
    return out;
  }
  StpikSnapshot snapshot() {return StpikTaskAccess::capture(state, model, backend, solver, tasks);}
  std::shared_ptr<FramePoseTask> pose;
  std::shared_ptr<RobotNominalTask> robot_nominal;
  std::vector<std::shared_ptr<TaskBase>> tasks;
};

TEST(NominalOptionsTest, EqualityToleranceDefaultsToTwoCentimetresAndRejectsInvalidValues)
{
  NominalOptions options;
  EXPECT_DOUBLE_EQ(options.equality_tolerance, 0.02);
  std::string reason;
  EXPECT_TRUE(options.valid(reason)) << reason;
  for (double tolerance : {0.0, -0.01, std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN()})
  {
    options.equality_tolerance = tolerance;
    EXPECT_FALSE(options.valid(reason));
    EXPECT_NE(reason.find("equality tolerance"), std::string::npos);
  }
}

TEST_F(NominalPlannerTest, EqualityMarginAcceptsFifteenMillimetresAndCanBeTightened)
{
  // A static tool path and a second equality disagree by 15 mm in Y.
  // Zero gain keeps the rollout static, isolating nominal feasibility from tracking.
  auto equality = std::make_shared<EndEffectorPositionTask>();
  equality->configure("other_equality", "EndEffectorPositionTask", {
    {"frame_id", rclcpp::Parameter("frame_id", "left")},
    {"auto_update_target_pose", rclcpp::Parameter("auto_update_target_pose", false)},
    {"default_goal", rclcpp::Parameter("default_goal", std::vector<double>{0.0, 0.015, 0.0})},
    {"gain", rclcpp::Parameter("gain", std::vector<double>{0.0, 0.0, 0.0})}}, context);
  equality->update(state, *backend);
  tasks.insert(tasks.begin() + 2, equality);
  StpikOptions simulation;
  simulation.dt = 0.02;
  PostureOptions posture;
  NominalOptions options;
  options.budget = 10;
  options.global_fallback = false;
  const auto goal = request(0.0, 1.0);
  const auto plan = optimize_nominal_trajectory(snapshot(), goal, simulation, posture, options,
    [] {return false;}, {}, nullptr, true);
  EXPECT_EQ(plan.validation.outcome, plan.validation.SUCCESS) << plan.validation.message;
  options.equality_tolerance = 0.001;
  try {
    optimize_nominal_trajectory(snapshot(), goal, simulation, posture, options,
      [] {return false;}, {}, nullptr, true);
    FAIL() << "A 1 mm equality margin cannot accommodate a 15 mm disagreement";
  } catch (const std::runtime_error & error) {
    EXPECT_NE(std::string(error.what()).find("constraint violation"), std::string::npos) << error.what();
  }
}

TEST_F(NominalPlannerTest, HierarchyValidationUsesEqualityMarginAndPreservesSetBasedLimit)
{
  auto equality = std::make_shared<EndEffectorPositionTask>();
  equality->configure("other_equality", "EndEffectorPositionTask", {
    {"frame_id", rclcpp::Parameter("frame_id", "left")}}, context);
  tasks.insert(tasks.begin() + 2, equality);
  const auto computations = [&] {
      backend->update(state);
      std::vector<TaskComputation> out;
      for (const auto & task : tasks) {out.push_back(task->update(state, *backend));}
      return out;
    };
  auto relaxed = validate_posture_hierarchy(tasks, "left_pose", 1e-6, 0.02);
  auto strict = validate_posture_hierarchy(tasks, "left_pose", 1e-6, 0.001);
  relaxed(computations());
  strict(computations());
  state.base_position.y() = 0.015;
  auto shifted = computations();
  EXPECT_NO_THROW(relaxed(shifted));
  EXPECT_THROW(strict(shifted), std::runtime_error);

  state.base_position.y() = 0.0;
  backend->update(state);
  auto region = std::make_shared<FramePositionNormLimitsTask>();
  region->configure("tight_region", "FramePositionNormLimitsTask", {
    {"frame_id", rclcpp::Parameter("frame_id", "left")},
    {"axes", rclcpp::Parameter("axes", std::vector<int64_t>{0, 1, 0})},
    {"threshold", rclcpp::Parameter("threshold", std::vector<double>{0.005})}}, context);
  tasks.insert(tasks.begin(), region);
  auto set_based = validate_posture_hierarchy(tasks, "left_pose", 1e-6, 0.02);
  set_based(computations());
  state.base_position.y() = 0.015;
  shifted = computations();
  EXPECT_THROW(set_based(shifted), std::runtime_error);
}

TEST_F(NominalPlannerTest, OptimizedNominalFollowsTheToolPathAndValidates)
{
  StpikOptions simulation;
  simulation.dt = 0.02;
  simulation.wall_timeout = 60;
  PostureOptions posture;
  NominalOptions options;
  options.knot_dt = 0.2;
  options.budget = 60;
  const auto goal = request(0.05, 2.0);
  std::vector<std::string> stages;
  const auto plan = optimize_nominal_trajectory(snapshot(), goal, simulation, posture, options,
    [] {return false;}, [&](const std::string & stage) {stages.push_back(stage);});
  EXPECT_EQ(plan.validation.outcome, plan.validation.SUCCESS) << plan.validation.message;
  EXPECT_EQ(plan.method, "stpik_rollout+sqp");
  ASSERT_EQ(plan.joints.points.size(), plan.base.points.size());
  ASSERT_GE(plan.knots.size(), 3u);
  EXPECT_EQ(plan.joints.joint_names, model->all_joint_names());
  // Every knot keeps the tool on the timed path.
  sura_manipulation_actions::CartesianPosePath path;
  std::string reason;
  ASSERT_TRUE(path.initTimed(goal.trajectory, reason)) << reason;
  for (size_t k = 0; k < plan.knots.size(); ++k) {
    WholeBodyState s = state;
    s.base_position = plan.knots[k].head<3>(); s.base_rpy = plan.knots[k].segment<3>(3);
    s.joint_positions = plan.knots[k].tail(2);
    const auto tip = backend->compute_frame_state(s, "left").pose;
    EXPECT_LT((tip.translation() - path.sample(plan.times[k]).translation()).norm(), 0.02) << "knot " << k;
    EXPECT_NEAR(s.base_rpy(0), 0.0, 0.01);
    EXPECT_NEAR(s.base_rpy(1), 0.0, 0.01);
  }
  // The STPIK checker accepts the same nominal as an execution request.
  const auto check = StpikCheck::run(snapshot(), nominal_tracking_goal(goal, plan), simulation);
  EXPECT_EQ(check.outcome, check.SUCCESS) << check.code << ": " << check.message;
  EXPECT_FALSE(stages.empty());
}

TEST_F(NominalPlannerTest, WarmStartRefinesAPreviousPlan)
{
  StpikOptions simulation;
  simulation.dt = 0.02;
  simulation.wall_timeout = 60;
  PostureOptions posture;
  NominalOptions options;
  options.knot_dt = 0.25;
  options.budget = 60;
  options.global_layers = 4;
  options.global_samples = 8;
  const auto goal = request(0.04, 2.0);
  const auto seeded = optimize_nominal_trajectory(snapshot(), goal, simulation, posture, options,
    [] {return false;}, {});
  // Warm starting from a previous plan uses the same machinery from another state.
  const auto warm = optimize_nominal_trajectory(snapshot(), goal, simulation, posture, options,
    [] {return false;}, {}, &seeded);
  EXPECT_EQ(warm.validation.outcome, warm.validation.SUCCESS) << warm.validation.message;
  EXPECT_NE(warm.method.find("warm_start"), std::string::npos) << warm.method;
}

TEST_F(NominalPlannerTest, LayeredSearchProducesAValidNominal)
{
  StpikOptions simulation;
  simulation.dt = 0.02;
  simulation.wall_timeout = 60;
  PostureOptions posture;
  NominalOptions options;
  options.knot_dt = 0.2;
  options.budget = 60;
  options.rollout_seed = false;
  options.global_layers = 5;
  options.global_samples = 12;
  const auto plan = optimize_nominal_trajectory(snapshot(), request(0.05, 2.0), simulation, posture,
    options, [] {return false;}, {});
  EXPECT_EQ(plan.validation.outcome, plan.validation.SUCCESS) << plan.validation.message;
  EXPECT_EQ(plan.method.rfind("layered_dp", 0), 0u) << plan.method;
}

TEST_F(NominalPlannerTest, FreeStartOptimizesTheInitialPostureWithTheTrajectory)
{
  // Posture preparation needs the safety tasks and holds used during repositioning.
  auto tilt = std::make_shared<StabilizeLimitsTask>();
  tilt->configure("stabilize_limits", "StabilizeLimitsTask", {
    {"lower_limits", rclcpp::Parameter("lower_limits", std::vector<double>{-0.5, -0.5})},
    {"upper_limits", rclcpp::Parameter("upper_limits", std::vector<double>{0.5, 0.5})}}, context);
  auto collision = std::make_shared<SelfCollisionAvoidanceTask>();
  collision->configure("self_collision_avoidance", "SelfCollisionAvoidanceTask", {
    {"capsule_names", rclcpp::Parameter("capsule_names", std::vector<std::string>{"a", "b"})},
    {"capsules.a.start_frame", rclcpp::Parameter("capsules.a.start_frame", "left")},
    {"capsules.a.end_frame", rclcpp::Parameter("capsules.a.end_frame", "left_end")},
    {"capsules.b.start_frame", rclcpp::Parameter("capsules.b.start_frame", "right")},
    {"capsules.b.end_frame", rclcpp::Parameter("capsules.b.end_frame", "right_end")},
    {"capsules.a.radius", rclcpp::Parameter("capsules.a.radius", 0.005)},
    {"capsules.b.radius", rclcpp::Parameter("capsules.b.radius", 0.005)},
    {"safe_distance", rclcpp::Parameter("safe_distance", 0.005)},
    {"activation_distance", rclcpp::Parameter("activation_distance", 0.01)},
    {"release_distance", rclcpp::Parameter("release_distance", 0.015)}}, context);
  // Start with the left joint inside joint_buffer (outside the joint-limit activation zone,
  // where an active limit would fix its velocity); the base can take over along x.
  state.joint_positions(0) = 0.86;
  backend->update(state);
  for (const auto & task : tasks) {task->update(state, *backend);}
  collision->update(state, *backend);
  tilt->update(state, *backend);
  tasks.insert(tasks.begin(), {tilt, collision});
  StpikOptions simulation;
  simulation.dt = 0.02;
  simulation.wall_timeout = 60;
  PostureOptions posture;
  posture.duration = 4.0;
  posture.settle = 3.0;
  NominalOptions options;
  options.knot_dt = 0.2;
  options.budget = 60;
  const auto goal = request(0.03, 2.0);
  std::vector<std::string> stages;
  const auto plan = optimize_nominal_trajectory(snapshot(), goal, simulation, posture, options,
    [] {return false;}, [&](const std::string & stage) {stages.push_back(stage);}, nullptr, true);
  std::string trace;
  for (const auto & stage : stages) {trace += stage + "\n";}
  EXPECT_EQ(plan.validation.outcome, plan.validation.SUCCESS) << plan.validation.message;
  ASSERT_FALSE(plan.knots.empty());
  // The proposal moves the joint away from the limit and keeps the tool at the path start.
  EXPECT_LT(plan.knots.front()(6), 0.85) << plan.method << "\n" << trace;
  WholeBodyState start = state;
  start.base_position = plan.knots.front().head<3>();
  start.base_rpy = plan.knots.front().segment<3>(3);
  start.joint_positions = plan.knots.front().tail(2);
  const auto tip = backend->compute_frame_state(start, "left").pose;
  const auto expected = backend->compute_frame_state(state, "left").pose;
  EXPECT_LT((tip.translation() - expected.translation()).norm(), 0.006);
}

TEST_F(NominalPlannerTest, PreventiveNormMarginExistsBeforeActivationAndHasCorrectGradient)
{
  FramePositionNormLimitsTask norm;
  norm.configure("right_region", "FramePositionNormLimitsTask", {
    {"frame_id", rclcpp::Parameter("frame_id", "right")},
    {"axes", rclcpp::Parameter("axes", std::vector<int64_t>{0, 1, 0})},
    {"threshold", rclcpp::Parameter("threshold", std::vector<double>{0.1})},
    {"activation_margin", rclcpp::Parameter("activation_margin", std::vector<double>{0.02})},
    {"release_margin", rclcpp::Parameter("release_margin", std::vector<double>{0.04})}}, context);
  norm.update(state, *backend);  // Capture center.
  state.joint_positions(1) = 0.07;
  backend->update(state);
  EXPECT_FALSE(norm.update(state, *backend).active);
  const auto margins = StpikTaskAccess::preventive_margins(norm, state);
  ASSERT_EQ(margins.size(), 1u);
  EXPECT_NEAR(margins[0].deficit, 0.5, 1e-9);
  EXPECT_NEAR(margins[0].gradient(7), 50.0, 1e-9);
  state.joint_positions(1) += 1e-5;
  backend->update(state);
  norm.update(state, *backend);
  EXPECT_NEAR((StpikTaskAccess::preventive_margins(norm, state)[0].deficit -
    margins[0].deficit) / 1e-5, margins[0].gradient(7), 1e-6);
  norm.set_enabled(false);
  EXPECT_TRUE(StpikTaskAccess::preventive_margins(norm, state).empty());
}

TEST_F(NominalPlannerTest, AlwaysActiveNormIsNotAnActivationAvoidanceObjective)
{
  FramePositionNormLimitsTask norm;
  norm.configure("right_region", "FramePositionNormLimitsTask", {
    {"frame_id", rclcpp::Parameter("frame_id", "right")},
    {"always_active", rclcpp::Parameter("always_active", true)}}, context);
  norm.update(state, *backend);
  EXPECT_TRUE(StpikTaskAccess::preventive_margins(norm, state).empty());
}

TEST_F(NominalPlannerTest, InactiveCollisionPairsHavePreventiveGradients)
{
  SelfCollisionAvoidanceTask collision;
  collision.configure("self_collision", "SelfCollisionAvoidanceTask", {
    {"capsule_names", rclcpp::Parameter("capsule_names", std::vector<std::string>{"a", "b"})},
    {"capsules.a.start_frame", rclcpp::Parameter("capsules.a.start_frame", "left")},
    {"capsules.a.end_frame", rclcpp::Parameter("capsules.a.end_frame", "left_end")},
    {"capsules.b.start_frame", rclcpp::Parameter("capsules.b.start_frame", "right")},
    {"capsules.b.end_frame", rclcpp::Parameter("capsules.b.end_frame", "right_end")},
    {"capsules.a.radius", rclcpp::Parameter("capsules.a.radius", 0.005)},
    {"capsules.b.radius", rclcpp::Parameter("capsules.b.radius", 0.005)},
    {"safe_distance", rclcpp::Parameter("safe_distance", 0.005)},
    {"activation_distance", rclcpp::Parameter("activation_distance", 0.075)},
    {"release_distance", rclcpp::Parameter("release_distance", 0.10)}}, context);
  EXPECT_FALSE(collision.update(state, *backend).active);
  const auto margins = StpikTaskAccess::preventive_margins(collision, state);
  ASSERT_EQ(margins.size(), 1u);
  EXPECT_NEAR(margins[0].deficit, 0.8, 1e-9);  // Clearance is 0.08 m.
  EXPECT_NEAR(margins[0].gradient(6), 40.0, 1e-9);
  state.joint_positions(0) += 1e-5;
  backend->update(state);
  collision.update(state, *backend);
  EXPECT_NEAR((StpikTaskAccess::preventive_margins(collision, state)[0].deficit -
    margins[0].deficit) / 1e-5, margins[0].gradient(6), 1e-6);
}

TEST_F(NominalPlannerTest, PreventionWinsOverLargeSecondarySmoothnessCost)
{
  // The left joint is safe and inactive, but inside the preventive release margin.
  state.joint_positions(0) = 0.88;
  backend->update(state);
  for (const auto & task : tasks) {task->update(state, *backend);}
  StpikOptions simulation;
  simulation.dt = 0.02;
  simulation.wall_timeout = 60;
  PostureOptions posture;
  posture.joints = 0;  // Only the new preventive objective pushes away from the limit.
  NominalOptions options;
  options.budget = 60;
  options.iterations = 16;
  options.smooth_weight = 1000;  // Smoothness cannot buy deterioration of prevention.
  const auto plan = optimize_nominal_trajectory(snapshot(), request(0.02, 2.0), simulation,
    posture, options, [] {return false;}, {});
  EXPECT_EQ(plan.validation.outcome, plan.validation.SUCCESS) << plan.validation.message;
  // Fixed start is excluded, so every optimizable knot can regain the release margin.
  EXPECT_LT(plan.preventive_cost, 1e-5);
  for (size_t k = 1; k < plan.knots.size(); ++k) {EXPECT_LE(plan.knots[k](6), 0.8502);}
}

TEST_F(NominalPlannerTest, OtherArmMovesAwayFromNormActivationBeforeItBecomesActive)
{
  auto norm = std::make_shared<FramePositionNormLimitsTask>();
  norm->configure("right_region", "FramePositionNormLimitsTask", {
    {"frame_id", rclcpp::Parameter("frame_id", "right")},
    {"axes", rclcpp::Parameter("axes", std::vector<int64_t>{0, 1, 0})},
    {"threshold", rclcpp::Parameter("threshold", std::vector<double>{0.1})},
    {"activation_margin", rclcpp::Parameter("activation_margin", std::vector<double>{0.02})},
    {"release_margin", rclcpp::Parameter("release_margin", std::vector<double>{0.04})}}, context);
  norm->update(state, *backend);  // Region centered at r=0.
  state.joint_positions(1) = 0.07;  // Inactive but inside the preventive band.
  backend->update(state);
  for (const auto & task : tasks) {task->update(state, *backend);}
  EXPECT_FALSE(norm->update(state, *backend).active);
  tasks.insert(tasks.begin(), norm);
  StpikOptions simulation;
  simulation.dt = 0.02;
  simulation.wall_timeout = 60;
  PostureOptions posture;
  NominalOptions options;
  options.budget = 60;
  options.iterations = 16;
  options.smooth_weight = 1000;
  const auto plan = optimize_nominal_trajectory(snapshot(), request(0.02, 2.0), simulation,
    posture, options, [] {return false;}, {});
  EXPECT_EQ(plan.validation.outcome, plan.validation.SUCCESS) << plan.validation.message;
  EXPECT_LT(plan.preventive_cost, 1e-5);
  // Verify actual frame error; base motion can also contribute to prevention.
  for (size_t k = 1; k < plan.knots.size(); ++k) {
    auto sample = state;
    sample.base_position = plan.knots[k].head<3>();
    sample.base_rpy = plan.knots[k].segment<3>(3);
    sample.joint_positions = plan.knots[k].tail(2);
    EXPECT_LE(std::abs(backend->compute_frame_state(sample, "right").pose.translation().y()), 0.0602);
  }
  EXPECT_LT(plan.knots.back()(7), 0.065);
}

TEST_F(NominalPlannerTest, OrientationNormPreventionUsesAngularGradient)
{
  FrameOrientationNormLimitsTask norm;
  norm.configure("base_region", "FrameOrientationNormLimitsTask", {
    {"frame_id", rclcpp::Parameter("frame_id", "base")},
    {"axes", rclcpp::Parameter("axes", std::vector<int64_t>{0, 0, 1})},
    {"threshold", rclcpp::Parameter("threshold", std::vector<double>{0.1})},
    {"activation_margin", rclcpp::Parameter("activation_margin", std::vector<double>{0.02})},
    {"release_margin", rclcpp::Parameter("release_margin", std::vector<double>{0.04})}}, context);
  norm.update(state, *backend);
  state.base_rpy.z() = 0.07;
  backend->update(state);
  EXPECT_FALSE(norm.update(state, *backend).active);
  const auto margins = StpikTaskAccess::preventive_margins(norm, state);
  ASSERT_EQ(margins.size(), 1u);
  EXPECT_NEAR(margins[0].deficit, 0.5, 1e-9);
  EXPECT_NEAR(margins[0].gradient(5), 50.0, 1e-9);
}

TEST_F(NominalPlannerTest, CheckRejectsNominalWithoutRobotNominalTask)
{
  StpikOptions simulation;
  simulation.dt = 0.02;
  PostureOptions posture;
  NominalOptions options;
  const auto goal = request(0.03, 1.0);
  const auto plan = optimize_nominal_trajectory(snapshot(), goal, simulation, posture, options,
    [] {return false;}, {});
  tasks.pop_back();
  const auto check = StpikCheck::run(snapshot(), nominal_tracking_goal(goal, plan), simulation);
  EXPECT_EQ(check.code, "NOMINAL_TASK_UNAVAILABLE");
  EXPECT_THROW(optimize_nominal_trajectory(snapshot(), goal, simulation, posture, options,
    [] {return false;}, {}), std::runtime_error);
}

}  // namespace
}  // namespace task_priority_kinematic_control
