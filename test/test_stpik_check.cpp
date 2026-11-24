#include "task_priority_kinematic_control/core/stpik_check.hpp"
#include "task_priority_kinematic_control/core/initial_posture.hpp"
#include "task_priority_kinematic_control/kinematics/kdl_kinematics_backend.hpp"
#include "task_priority_kinematic_control/tasks/frame_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/frame_position_norm_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_nominal_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/stabilize_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/self_collision_avoidance_task.hpp"
#include "sura_manipulation_actions/cartesian_pose_path.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <iterator>
#include <limits>

namespace task_priority_kinematic_control
{
namespace
{
using Params = std::map<std::string, rclcpp::Parameter>;
constexpr char robot[] = R"(
<robot name="stpik_fixture">
  <link name="base"/><link name="left"/><link name="right"/>
  <link name="left_end"/><link name="right_end"/>
  <joint name="l" type="prismatic"><parent link="base"/><child link="left"/>
    <axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="10" velocity="0.001"/></joint>
  <joint name="r" type="prismatic"><parent link="base"/><child link="right"/>
    <origin xyz="0.09 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="10" velocity="0.001"/></joint>
  <joint name="le" type="fixed"><parent link="left"/><child link="left_end"/><origin xyz="0 0 0.1"/></joint>
  <joint name="re" type="fixed"><parent link="right"/><child link="right_end"/><origin xyz="0 0 0.1"/></joint>
</robot>)";

class StpikTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    model = std::make_shared<WholeBodyModel>();
    model->configure("world", "base", "left", "right", {"l"}, {"r"}, {0,1,2,3,4,5});
    backend = std::make_shared<KDLKinematicsBackend>();
    backend->configure(*model, robot, rclcpp::get_logger("stpik_test"));
    state.joints_valid = state.navigation_valid = state.ros_time_valid = true;
    state.ros_time_ns = 1000000000LL;
    state.joint_positions = state.joint_velocities = Eigen::VectorXd::Zero(2);
    backend->update(state);
    solver.configure(model->total_dofs());
    solver.set_velocity_limits(Eigen::VectorXd::Constant(model->total_dofs(), 1.0));
    pose = std::make_shared<FramePoseTask>();
    pose->configure("left_pose", "task_priority_kinematic_control/FramePoseTask", {
      {"frame_id", rclcpp::Parameter("frame_id", "left")},
      {"gain", rclcpp::Parameter("gain", std::vector<double>(6, 2.0))}}, TaskContext{model, backend});
    pose->update(state, *backend);  // Capture the same initialized real target.
    tasks = {pose};
    options.wall_timeout = 60.0;
  }
  StpikAction::Goal goal(double displacement = 0.03, double angle = 0.1, double duration = 2.0)
  {
    auto start = backend->get_frame_state("left").pose;
    auto target = start;
    target.translation().z() += displacement;
    target.linear() = Eigen::AngleAxisd(angle, Eigen::Vector3d(1,2,3).normalized()).toRotationMatrix() * start.rotation();
    sura_manipulation_actions::CartesianPosePath path;
    std::string reason;
    EXPECT_TRUE(path.init(start, target, 1,1,1,1,0.01,0.05,10000,reason)) << reason;
    StpikAction::Goal out;
    out.trajectory = path.descriptor("world"); out.trajectory.duration = duration;
    out.task_id = "left_pose"; out.tip_frame = "left";
    out.tolerance_orientation.w = 1.0;
    out.path_tolerance = {0.02,0.02,0.02,0.3,0.3,0.3};
    out.goal_tolerance = {0.002,0.002,0.002,0.02,0.02,0.02};
    out.goal_time_tolerance = 0.5; return out;
  }
  StpikSnapshot snapshot() {return StpikTaskAccess::capture(state, model, backend, solver, tasks);}
  StpikAction::Result run(const StpikAction::Goal & input) {return StpikCheck::run(snapshot(), input, options);}
  void block_motion()
  {
    auto base = std::make_shared<FramePoseTask>();
    base->configure("hold_base", "FramePoseTask", {}, TaskContext{model, backend});
    base->update(state, *backend);
    auto joints = std::make_shared<JointNominalTask>();
    joints->configure("hold_joints", "JointNominalTask", {}, TaskContext{model, backend});
    tasks = {base, joints, pose};
  }
  void collision(double radius, const std::string & id = "collision")
  {
    auto task = std::make_shared<SelfCollisionAvoidanceTask>();
    task->configure(id, "SelfCollisionAvoidanceTask", {
      {"capsule_names", rclcpp::Parameter("capsule_names", std::vector<std::string>{"a","b"})},
      {"capsules.a.start_frame", rclcpp::Parameter("capsules.a.start_frame", "left")},
      {"capsules.a.end_frame", rclcpp::Parameter("capsules.a.end_frame", "left_end")},
      {"capsules.b.start_frame", rclcpp::Parameter("capsules.b.start_frame", "right")},
      {"capsules.b.end_frame", rclcpp::Parameter("capsules.b.end_frame", "right_end")},
      {"capsules.a.radius", rclcpp::Parameter("capsules.a.radius", radius)},
      {"capsules.b.radius", rclcpp::Parameter("capsules.b.radius", radius)},
      {"safe_distance", rclcpp::Parameter("safe_distance", 0.02)},
      {"activation_distance", rclcpp::Parameter("activation_distance", 0.06)},
      {"release_distance", rclcpp::Parameter("release_distance", 0.08)}}, TaskContext{model, backend});
    tasks.insert(tasks.begin(), task);
  }
  std::shared_ptr<WholeBodyModel> model;
  KinematicsBackendPtr backend;
  HierarchySolver solver;
  WholeBodyState state;
  std::shared_ptr<FramePoseTask> pose;
  std::vector<std::shared_ptr<TaskBase>> tasks;
  StpikOptions options;
};

TEST(PostureCostTest, ProximityActivationAndZeroActivationDistance)
{
  EXPECT_DOUBLE_EQ(posture_limit_cost(0.3, 0.1, 0.2, false, 100), 0);
  EXPECT_NEAR(posture_limit_cost(0.15, 0.1, 0.2, false, 100), 0.25, 1e-12);
  EXPECT_DOUBLE_EQ(posture_limit_cost(0.1, 0.1, 0.2, false, 100), 1);
  EXPECT_DOUBLE_EQ(posture_limit_cost(0.1, 0.1, 0.2, true, 100), 100);
  EXPECT_DOUBLE_EQ(posture_limit_cost(0.05, 0.1, 0.2, true, 100), 125);
  EXPECT_DOUBLE_EQ(posture_limit_cost(0.15, 0.1, 0.2, true, 100), 100);
  EXPECT_DOUBLE_EQ(posture_limit_cost(0, 0.1, 0.2, true, 100), 200);
  EXPECT_DOUBLE_EQ(posture_limit_cost(0, 0, 0.2, true, 100), 100);
  EXPECT_DOUBLE_EQ(posture_limit_cost(std::numeric_limits<double>::infinity(), 0.1, 0.2, false, 100), 0);
}

TEST_F(StpikTest, JointPostureCostUsesBothBoundariesAndTheActualHysteresis)
{
  JointLimitsTask limits;
  limits.configure("joint_limits", "JointLimitsTask", {
    {"alpha", rclcpp::Parameter("alpha", 0.1)},
    {"delta", rclcpp::Parameter("delta", 0.2)},
    {"lower_limits", rclcpp::Parameter("lower_limits", std::vector<double>{-1,-1})},
    {"upper_limits", rclcpp::Parameter("upper_limits", std::vector<double>{1,1})}},
    TaskContext{model, backend});
  const auto at = [&](double q) {
      state.joint_positions(1) = q;
      backend->update(state);
      limits.update(state, *backend);
      return StpikTaskAccess::posture_safety_costs(limits, state, 100).joints;
    };
  EXPECT_DOUBLE_EQ(at(0), 0);
  EXPECT_NEAR(at(0.85), 0.25, 1e-12);
  EXPECT_GE(at(0.95), 100);
  EXPECT_DOUBLE_EQ(at(0.85), 100);  // Latched until the release boundary.
  EXPECT_DOUBLE_EQ(at(0.79), 0);
  EXPECT_GE(at(-0.95), 100);
  StpikAction::Result safety;
  EXPECT_TRUE(StpikTaskAccess::safety(limits, state, 1e-6, safety));
  at(-1.01);
  EXPECT_FALSE(StpikTaskAccess::safety(limits, state, 1e-6, safety));
  limits.set_enabled(false);
  EXPECT_DOUBLE_EQ(StpikTaskAccess::posture_safety_costs(limits, state, 100).joints, 0);
}

TEST_F(StpikTest, StabilizePostureCostUsesItsOwnActivationAndReleaseMargins)
{
  StabilizeLimitsTask limits;
  limits.configure("stabilize_limits", "StabilizeLimitsTask", {
    {"alpha", rclcpp::Parameter("alpha", 0.05)},
    {"delta", rclcpp::Parameter("delta", 0.1)},
    {"lower_limits", rclcpp::Parameter("lower_limits", std::vector<double>{-0.5,-0.5})},
    {"upper_limits", rclcpp::Parameter("upper_limits", std::vector<double>{0.5,0.5})}},
    TaskContext{model, backend});
  const auto at = [&](double roll, double pitch) {
      state.base_rpy << roll, pitch, 0;
      limits.update(state, *backend);
      return StpikTaskAccess::posture_safety_costs(limits, state, 100).stabilize;
    };
  EXPECT_DOUBLE_EQ(at(0, 0), 0);
  EXPECT_NEAR(at(0.425, 0), 0.25, 1e-12);
  EXPECT_GE(at(0.475, 0), 100);
  EXPECT_DOUBLE_EQ(at(0.425, 0), 100);
  EXPECT_DOUBLE_EQ(at(0.39, 0), 0);
  EXPECT_GE(at(0, -0.475), 100);
  StpikAction::Result safety;
  EXPECT_TRUE(StpikTaskAccess::safety(limits, state, 1e-6, safety));
  at(0, -0.51);
  EXPECT_FALSE(StpikTaskAccess::safety(limits, state, 1e-6, safety));
}

TEST_F(StpikTest, CollisionPostureCostDistinguishesProximityActivationAndUnsafeClearance)
{
  collision(0.01);
  const auto task = tasks.front();
  const auto at = [&](double q) {
      state.joint_positions(0) = q;
      backend->update(state);
      task->update(state, *backend);
      return StpikTaskAccess::posture_safety_costs(*task, state, 100).self_collision;
    };
  EXPECT_NEAR(at(0), 0.25, 1e-12);
  EXPECT_GE(at(0.02), 100);
  EXPECT_DOUBLE_EQ(at(0), 100);
  EXPECT_DOUBLE_EQ(at(-0.02), 0);
  StpikAction::Result safety;
  EXPECT_TRUE(StpikTaskAccess::safety(*task, state, 1e-6, safety));
  at(0.06);
  EXPECT_FALSE(StpikTaskAccess::safety(*task, state, 1e-6, safety));
  task->set_enabled(false);
  EXPECT_DOUBLE_EQ(StpikTaskAccess::posture_safety_costs(*task, state, 100).self_collision, 0);
}

TEST_F(StpikTest, TranslationAndOrientationUseEachRuntimeSolverWithoutChangingLiveObjects)
{
  for (auto method : {SolverMethod::kDls, SolverMethod::kPinv, SolverMethod::kSvd}) {
    solver.set_method(method);
    const auto before = snapshot();
    const auto target = pose->current_target();
    const auto result = run(goal());
    EXPECT_EQ(result.outcome, StpikAction::Result::SUCCESS) << result.code << ": " << result.message;
    EXPECT_EQ(pose->current_target(), target);
    EXPECT_TRUE(state.joint_positions.isZero());
    EXPECT_TRUE(backend->get_frame_state("left").pose.isApprox(Eigen::Isometry3d::Identity()));
    std::string reason;
    EXPECT_TRUE(StpikTaskAccess::compatible(before, state, *model, backend.get(), solver, tasks, options, reason)) << reason;
  }
}

TEST_F(StpikTest, MaskedAxesAreIgnoredForPathGoalAndRotatedToleranceFrame)
{
  pose->configure("left_pose", "FramePoseTask", {
    {"frame_id", rclcpp::Parameter("frame_id", "left")},
    {"activation", rclcpp::Parameter("activation", std::vector<int64_t>{1,1,0,0,0,0})}},
    TaskContext{model, backend});
  pose->update(state, *backend);
  auto input = goal(0.08, 0.6, 3.0);
  const Eigen::Quaterniond axes(Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitY()));
  input.tolerance_orientation.x = axes.x(); input.tolerance_orientation.y = axes.y();
  input.tolerance_orientation.z = axes.z(); input.tolerance_orientation.w = axes.w();
  const auto result = run(input);
  EXPECT_EQ(result.outcome, StpikAction::Result::SUCCESS) << result.code << ": " << result.message;
  // Raw diagnostics still show the intentionally uncontrolled deviation.
  EXPECT_GT(std::abs(result.tracking_error[0]), 0.04);
  EXPECT_GT(std::abs(result.tracking_error[2]), 0.04);
  EXPECT_GT(std::abs(result.tracking_error[4]), 0.1);

  state.base_position.x() = 0.1;
  backend->update(state);
  EXPECT_EQ(run(input).code, "INITIAL_STATE_MISMATCH");  // Active X must still be checked.
}

TEST_F(StpikTest, ZeroActivationMaskDoesNotApproveAnInactiveTask)
{
  pose->configure("left_pose", "FramePoseTask", {
    {"frame_id", rclcpp::Parameter("frame_id", "left")},
    {"activation", rclcpp::Parameter("activation", std::vector<int64_t>(6, 0))}},
    TaskContext{model, backend});
  EXPECT_EQ(run(goal()).code, "POSE_TASK_INACTIVE");
}

TEST_F(StpikTest, PostureSearchImprovesTheIdleArmMarginWithTheToolHeld)
{
  state.joint_positions(1) = 0.94;
  backend->update(state);
  const TaskContext context{model, backend};
  auto limits = std::make_shared<JointLimitsTask>();
  limits->configure("joint_limits", "JointLimitsTask", {
    {"alpha", rclcpp::Parameter("alpha", 0.02)},
    {"delta", rclcpp::Parameter("delta", 0.12)},
    {"lower_limits", rclcpp::Parameter("lower_limits", std::vector<double>{-1,-1})},
    {"upper_limits", rclcpp::Parameter("upper_limits", std::vector<double>{1,1})}}, context);
  tasks.insert(tasks.begin(), limits);
  collision(0.01, "self_collision_avoidance");
  auto tilt_limits = std::make_shared<StabilizeLimitsTask>();
  tilt_limits->configure("stabilize_limits", "StabilizeLimitsTask", {
    {"lower_limits", rclcpp::Parameter("lower_limits", std::vector<double>{-0.5,-0.5})},
    {"upper_limits", rclcpp::Parameter("upper_limits", std::vector<double>{0.5,0.5})}}, context);
  tasks.insert(tasks.begin(), tilt_limits);
  for (const std::string id : {"stabilize", "base_pose"}) {
    auto task = std::make_shared<FramePoseTask>();
    const bool level = id == "stabilize";
    task->configure(id, "FramePoseTask", {
      {"frame_id", rclcpp::Parameter("frame_id", "base")},
      {"roll_pitch_control", rclcpp::Parameter("roll_pitch_control", level)},
      {"activation", rclcpp::Parameter("activation", level ?
        std::vector<int64_t>{0,0,0,1,1,0} : std::vector<int64_t>{1,1,1,0,0,1})}}, context);
    tasks.push_back(task);
  }
  for (const std::string joint : {"l", "r"}) {
    auto task = std::make_shared<JointNominalTask>();
    task->configure(joint == "l" ? "left_arm_nominal" : "right_arm_nominal", "JointNominalTask", {
      {"joint_names", rclcpp::Parameter("joint_names", std::vector<std::string>{joint})},
      {"gain", rclcpp::Parameter("gain", std::vector<double>{4})}}, context);
    task->set_enabled(false);
    tasks.push_back(task);
  }
  const auto path = goal(0,0,0.3);
  Preparation::Goal request;
  request.task_id = path.task_id;
  request.tip_frame = path.tip_frame;
  request.trajectory = path.trajectory;
  request.path_tolerance = path.path_tolerance;
  request.goal_tolerance = path.goal_tolerance;
  request.tolerance_orientation = path.tolerance_orientation;
  request.robot_description = robot;
  request.robot_description_semantic = R"(<robot name="stpik_fixture">
    <virtual_joint name="root" type="floating" parent_frame="world" child_link="base"/></robot>)";
  request.scene.robot_model_name = "stpik_fixture";
  request.scene.robot_state.joint_state.name = {"l", "r"};
  request.scene.robot_state.joint_state.position = {state.joint_positions(0), state.joint_positions(1)};
  PostureOptions search;
  search.duration = 0.6;
  search.settle = 1.5;
  search.budget = 8;
  options.dt = 0.05;
  unsigned evaluations = 0;
  const auto result = optimize_posture(snapshot(), request, options, search,
    []{return false;}, [&](unsigned n){evaluations = n;});
  EXPECT_GT(evaluations, 1u);
  EXPECT_TRUE(result.summary.baseline_feasible);
  EXPECT_LT(result.summary.cost, result.summary.baseline_cost);
  EXPECT_LT(result.target.joint_positions(1), 0.94);
  EXPECT_DOUBLE_EQ(state.joint_positions(1), 0.94);
  EXPECT_FALSE(tasks.back()->enabled());

  // Disabling the joint cost removes the incentive to reconfigure this safe,
  // stationary fixture. The old fixed joint-buffer penalty must not remain.
  auto movement_only = search;
  movement_only.joints = movement_only.tilt = movement_only.stabilize = 0;
  movement_only.self_collision = movement_only.clearance = 0;
  const auto unchanged = optimize_posture(snapshot(), request, options, movement_only,
    []{return false;}, [](unsigned){});
  EXPECT_NEAR(unchanged.summary.cost, 0.0, 1e-12);
  EXPECT_FALSE(unchanged.move);

  auto norm = std::make_shared<FramePositionNormLimitsTask>();
  norm->configure("right_position_norm", "FramePositionNormLimitsTask", {
    {"frame_id", rclcpp::Parameter("frame_id", "right")},
    {"threshold", rclcpp::Parameter("threshold", std::vector<double>{0.06})},
    {"gain", rclcpp::Parameter("gain", std::vector<double>{4.0})}}, context);
  norm->update(state, *backend);
  tasks.insert(std::find(tasks.begin(), tasks.end(), pose), norm);
  const auto norm_target = norm->current_target();
  const auto bounded = optimize_posture(snapshot(), request, options, search,
    []{return false;}, [](unsigned){});
  EXPECT_LT(bounded.summary.cost, bounded.summary.baseline_cost);
  const auto start_right = backend->compute_frame_state(state, "right").pose;
  const auto bounded_right = backend->compute_frame_state(bounded.target, "right").pose;
  const double movement = (bounded_right.translation() - start_right.translation()).norm();
  EXPECT_GT(movement, 1e-3);
  EXPECT_LE(movement, 0.06 + 1e-6);
  EXPECT_EQ(norm->current_target(), norm_target);

  auto right_pose = std::make_shared<FramePoseTask>();
  right_pose->configure("right_pose", "FramePoseTask", {
    {"frame_id", rclcpp::Parameter("frame_id", "right")}}, context);
  right_pose->update(state, *backend);
  *std::find(tasks.begin(), tasks.end(), norm) = right_pose;
  const auto right_target = right_pose->current_target();
  const auto held = optimize_posture(snapshot(), request, options, search,
    []{return false;}, [](unsigned){});
  EXPECT_TRUE(held.summary.baseline_feasible);
  EXPECT_TRUE(backend->compute_frame_state(held.target, "right").pose.isApprox(start_right, 1e-3));
  EXPECT_EQ(right_pose->current_target(), right_target);

  // Selecting the right arm changes which task's reference may be written.
  auto isolated = snapshot();
  auto right_request = request;
  right_request.task_id = "right_pose";
  right_request.tip_frame = "right";
  right_request.trajectory.start_pose.position.x = start_right.translation().x();
  right_request.trajectory.start_pose.position.y = start_right.translation().y();
  right_request.trajectory.target_pose = right_request.trajectory.start_pose;
  auto & isolated_tasks = isolated.tasks;
  const auto isolated_right = std::find_if(isolated_tasks.begin(), isolated_tasks.end(),
    [](const auto & task) {return task->id() == "right_pose";});
  ASSERT_NE(isolated_right, isolated_tasks.end());
  const auto isolated_left = std::next(isolated_right);
  ASSERT_EQ((*isolated_left)->id(), "left_pose");
  std::iter_swap(isolated_right, isolated_left);
  const auto left_target = (*isolated_right)->current_target();
  const auto from_right = optimize_posture(
    StpikTaskAccess::capture(state, model, backend, solver, isolated_tasks),
    right_request, options, search, []{return false;}, [](unsigned){});
  EXPECT_TRUE(from_right.summary.baseline_feasible);
  const auto right_commands = prepare_tasks(isolated_tasks, *isolated.model, right_request);
  EXPECT_FALSE(right_commands.empty());
  EXPECT_EQ((*isolated_right)->current_target(), left_target);
  EXPECT_TRUE((*isolated_right)->enabled());
  std::string stale_reason;
  auto reordered = tasks;
  const auto left_in_order = std::find(reordered.begin(), reordered.end(), pose);
  std::iter_swap(left_in_order, std::prev(left_in_order));
  EXPECT_FALSE(StpikTaskAccess::compatible(isolated, state, *model, backend.get(),
    solver, reordered, options, stale_reason));
  EXPECT_NE(stale_reason.find("hierarchy"), std::string::npos);

  // An exhausted wall budget is separate from an operator cancel request.
  search.budget = 1e-12;
  try {
    optimize_posture(snapshot(), request, options, search,
      []{return false;}, [](unsigned){});
    FAIL() << "Expected the search deadline to interrupt validation";
  } catch (const std::runtime_error & error) {
    const std::string detail = error.what();
    EXPECT_NE(detail.find("time budget exhausted"), std::string::npos) << detail;
    EXPECT_NE(detail.find("evaluations_completed=0"), std::string::npos) << detail;
    EXPECT_NE(detail.find("initial_posture={evaluation interrupted"), std::string::npos) << detail;
    EXPECT_EQ(detail.find("STPIK canceled"), std::string::npos) << detail;
  }
  EXPECT_THROW(optimize_posture(snapshot(), request, options, search,
    []{return true;}, [](unsigned){}), std::runtime_error);

}

TEST_F(StpikTest, PostureHierarchyReportsNormViolationAtTheSimulationStep)
{
  auto norm = std::make_shared<FramePositionNormLimitsTask>();
  norm->configure("position_region", "FramePositionNormLimitsTask", {
    {"frame_id", rclcpp::Parameter("frame_id", "left")},
    {"threshold", rclcpp::Parameter("threshold", std::vector<double>{0.03})},
    {"axes", rclcpp::Parameter("axes", std::vector<bool>{true, false, false})}},
    TaskContext{model, backend});
  tasks.insert(tasks.begin(), norm);
  auto sim = snapshot();
  auto check = goal(0.0, 0.0, 1.0);
  // Let a coarse simulation step cross the region before its repulsive row can
  // activate. The selected tool still tracks within its own path tolerance.
  options.dt = 0.25;
  check.trajectory.target_pose.position.x = 0.14;
  check.path_tolerance.fill(0.2);
  check.goal_tolerance.fill(0.2);
  const auto validate = validate_posture_hierarchy(sim.tasks, check.task_id, options.numerical_tolerance);
  const auto result = StpikCheck::run(std::move(sim), check, options, {}, {}, {}, {}, validate);
  EXPECT_EQ(result.outcome, StpikAction::Result::FAIL);
  EXPECT_GT(result.time_from_start, 0.0);
  EXPECT_LT(result.time_from_start, check.trajectory.duration);
  EXPECT_NE(result.message.find("position_region exceeded norm limit"), std::string::npos)
    << result.code << ": " << result.message;
}

TEST_F(StpikTest, PostureRecheckDetectsObstacleInsidePathWithClearEndpoints)
{
  auto limits = std::make_shared<JointLimitsTask>();
  limits->configure("joint_limits", "JointLimitsTask", {
    {"lower_limits", rclcpp::Parameter("lower_limits", std::vector<double>{-1,-1})},
    {"upper_limits", rclcpp::Parameter("upper_limits", std::vector<double>{1,1})}}, TaskContext{model, backend});
  tasks.insert(tasks.begin(), limits);
  PosturePlan plan;
  plan.snapshot = snapshot();
  plan.target = state;
  plan.simulation = options;
  auto & request = plan.request;
  const auto path = goal(0.08, 0, 2);
  request.task_id = path.task_id;
  request.tip_frame = path.tip_frame;
  request.trajectory = path.trajectory;
  request.path_tolerance = path.path_tolerance;
  request.goal_tolerance = path.goal_tolerance;
  request.goal_time_tolerance = path.goal_time_tolerance;
  request.tolerance_orientation = path.tolerance_orientation;
  request.robot_description = robot;
  const std::string left_link = "<link name=\"left\"/>";
  request.robot_description.replace(request.robot_description.find(left_link), left_link.size(),
    R"(<link name="left"><collision><geometry><sphere radius="0.004"/></geometry></collision></link>)");
  request.robot_description_semantic = R"(<robot name="stpik_fixture">
    <virtual_joint name="root" type="floating" parent_frame="world" child_link="base"/></robot>)";
  request.scene.robot_model_name = "stpik_fixture";
  request.scene.robot_state.joint_state.name = {"l", "r"};
  request.scene.robot_state.joint_state.position = {state.joint_positions(0), state.joint_positions(1)};
  ASSERT_NO_THROW(recheck_posture(plan, request, []{return false;}));
  moveit_msgs::msg::CollisionObject obstacle;
  obstacle.id = "middle_of_path";
  obstacle.header.frame_id = "world";
  obstacle.pose.orientation.w = 1.0;
  shape_msgs::msg::SolidPrimitive sphere;
  sphere.type = sphere.SPHERE;
  sphere.dimensions = {0.008};
  geometry_msgs::msg::Pose location;
  location.position.z = 0.04;
  location.orientation.w = 1.0;
  obstacle.primitives = {sphere};
  obstacle.primitive_poses = {location};
  request.scene.world.collision_objects.push_back(obstacle);
  try {
    recheck_posture(plan, request, []{return false;});
    FAIL() << "Accepted a path crossing the obstacle";
  } catch (const std::runtime_error & error) {
    const std::string diagnostic = error.what();
    EXPECT_NE(diagnostic.find("source=MoveIt"), std::string::npos) << diagnostic;
    EXPECT_NE(diagnostic.find("type=robot_world"), std::string::npos) << diagnostic;
    EXPECT_NE(diagnostic.find("middle_of_path"), std::string::npos) << diagnostic;
    EXPECT_NE(diagnostic.find("left"), std::string::npos) << diagnostic;
    EXPECT_NE(diagnostic.find("penetration="), std::string::npos) << diagnostic;
  }
  EXPECT_TRUE(state.joint_positions.isZero());

  request.scene.world.collision_objects.clear();
  const std::string right_link = "<link name=\"right\"/>";
  request.robot_description.replace(request.robot_description.find(right_link), right_link.size(),
    R"(<link name="right"><collision><geometry><sphere radius="0.1"/></geometry></collision></link>)");
  try {
    recheck_posture(plan, request, []{return false;});
    FAIL() << "Accepted colliding robot links";
  } catch (const std::runtime_error & error) {
    const std::string diagnostic = error.what();
    EXPECT_NE(diagnostic.find("source=MoveIt"), std::string::npos) << diagnostic;
    EXPECT_NE(diagnostic.find("type=self_collision"), std::string::npos) << diagnostic;
    EXPECT_NE(diagnostic.find("left <-> right"), std::string::npos) << diagnostic;
    EXPECT_EQ(diagnostic.find("source=task"), std::string::npos) << diagnostic;
  }
}

TEST_F(StpikTest, UsesRuntimeVelocityCapsAndAllowsOrdinarySaturation)
{
  auto nominal = std::make_shared<JointNominalTask>();
  nominal->configure("right_nominal", "JointNominalTask", {
    {"joint_names", rclcpp::Parameter("joint_names", std::vector<std::string>{"r"})},
    {"target", rclcpp::Parameter("target", std::vector<double>{1.0})}}, TaskContext{model, backend});
  tasks.push_back(nominal);
  Eigen::VectorXd limits = Eigen::VectorXd::Ones(8); limits(7) = 0.01;
  solver.set_velocity_limits(limits);
  const auto result = run(goal());
  EXPECT_EQ(result.outcome, StpikAction::Result::SUCCESS) << result.message;
  EXPECT_NEAR(result.maximum_velocity_ratio, 1.0, 1e-12);
  // URDF caps are 0.001; the runtime right-joint cap is authoritative.
  EXPECT_GT(result.joint_positions.at(1), 0.001);
}

TEST_F(StpikTest, OptionalStricterVelocityLimitsRejectTheUnmodifiedRuntimeCommand)
{
  options.velocity_limits.assign(8, 1e-5);
  const auto result = run(goal());
  EXPECT_EQ(result.code, "VELOCITY_LIMIT");
  EXPECT_GT(result.maximum_velocity_ratio, 1.0);
}

TEST_F(StpikTest, RejectsInitialJointLimitViolationAndPreservesDiagnosticConfiguration)
{
  auto limits = std::make_shared<JointLimitsTask>();
  limits->configure("joint_limits", "JointLimitsTask", {
    {"lower_limits", rclcpp::Parameter("lower_limits", std::vector<double>{-1,-1})},
    {"upper_limits", rclcpp::Parameter("upper_limits", std::vector<double>{1,1})}}, TaskContext{model, backend});
  tasks.insert(tasks.begin(), limits);
  state.joint_positions(1) = 1.2; backend->update(state);
  const auto result = run(goal());
  EXPECT_EQ(result.code, "JOINT_LIMIT"); EXPECT_EQ(result.constraint, "r");
  EXPECT_EQ(result.time_from_start, 0.0); EXPECT_EQ(result.joint_positions.at(1), 1.2);
}

TEST_F(StpikTest, CollisionActivationIsAllowedButUnsafeClearanceIsRejected)
{
  collision(0.02);
  auto result = run(goal(0.01,0,2));
  EXPECT_EQ(result.outcome, StpikAction::Result::SUCCESS) << result.message;
  EXPECT_NEAR(result.minimum_clearance, 0.05, 1e-6);
  tasks.erase(tasks.begin()); collision(0.05);
  result = run(goal(0.01,0,2));
  EXPECT_EQ(result.code, "COLLISION_CLEARANCE");
  EXPECT_LT(result.minimum_clearance, 0.02);
  EXPECT_FALSE(result.constraint.empty());
  EXPECT_NE(result.message.find("source=task"), std::string::npos) << result.message;
  EXPECT_NE(result.message.find("task=collision"), std::string::npos) << result.message;
  EXPECT_NE(result.message.find("pair=a,b"), std::string::npos) << result.message;
  EXPECT_NE(result.message.find("safe_distance=0.02"), std::string::npos) << result.message;
  EXPECT_EQ(result.message.find("source=MoveIt"), std::string::npos) << result.message;

  tasks.erase(tasks.begin()); collision(0.02);
  state.joint_positions(0) = 0.09;  // Coincident capsule axes: invalid collision geometry.
  backend->update(state);
  result = run(goal(0,0,0));
  EXPECT_EQ(result.code, "COLLISION_GEOMETRY");
  EXPECT_NE(result.message.find("source=task"), std::string::npos) << result.message;
  EXPECT_NE(result.message.find("pair=a,b"), std::string::npos) << result.message;
  EXPECT_NE(result.message.find("degenerate_pairs=1"), std::string::npos) << result.message;
}

TEST_F(StpikTest, RankLossWithoutRequestedMovementIsAllowed)
{
  block_motion();
  auto input = goal(0,0,0);
  const auto result = run(input);
  EXPECT_EQ(result.outcome, StpikAction::Result::SUCCESS) << result.message;
  EXPECT_LT(result.minimum_singular_value, options.singular_value_tolerance);
}

TEST_F(StpikTest, BlockedProgressAndUnreachedGoalHaveDifferentReasons)
{
  block_motion();
  auto input = goal(0.01,0,5);
  auto result = run(input);
  EXPECT_EQ(result.code, "STALLED_SINGULARITY");
  EXPECT_EQ(result.constraint, "position");
  EXPECT_NE(result.message.find("position_stalled=true"), std::string::npos);
  EXPECT_NE(result.message.find("orientation_stalled=false"), std::string::npos);
  EXPECT_NE(result.message.find("orientation{requested=false"), std::string::npos);
  EXPECT_NE(result.message.find("measured_delta_m="), std::string::npos);
  input = goal(0.001,0,0.05); input.goal_time_tolerance = 0.05;
  input.goal_tolerance.fill(1e-12);
  result = run(input);
  EXPECT_EQ(result.code, "GOAL_SINGULARITY");
  EXPECT_NEAR(result.time_from_start, 0.1, 1e-9);
}

TEST_F(StpikTest, SlowConvergenceNearGoalUsesAdaptivePositionAndOrientationProgress)
{
  std::string message;
  ASSERT_TRUE(pose->set_gain(std::vector<double>(6, 0.002), message));
  auto input = goal(0, 0, 8);
  input.trajectory.start_pose.position.z = input.trajectory.target_pose.position.z = 0.003;
  input.trajectory.start_pose.orientation.x = input.trajectory.target_pose.orientation.x = std::sin(0.0015);
  input.trajectory.start_pose.orientation.w = input.trajectory.target_pose.orientation.w = std::cos(0.0015);
  input.goal_tolerance.fill(0.00299);
  input.goal_tolerance[2] = 0.00297;
  const auto result = run(input);
  EXPECT_EQ(result.outcome, StpikAction::Result::SUCCESS) << result.code << ": " << result.message;
  EXPECT_LT(std::abs(result.tracking_error[2]), input.goal_tolerance[2]);
  EXPECT_LT(std::abs(result.tracking_error[3]), input.goal_tolerance[3]);
}

TEST_F(StpikTest, ReferenceMotionInsideGoalToleranceDoesNotRequireProgress)
{
  block_motion();
  const auto result = run(goal(0.001, 0.005, 8));
  EXPECT_EQ(result.outcome, StpikAction::Result::SUCCESS) << result.code << ": " << result.message;
}

TEST_F(StpikTest, BlockedErrorOutsideToleranceStillRequiresSustainedProgress)
{
  block_motion();
  auto input = goal(0, 0, 8);
  input.trajectory.start_pose.position.z = input.trajectory.target_pose.position.z = 0.003;
  input.goal_tolerance.fill(0.00299);
  const auto result = run(input);
  EXPECT_EQ(result.code, "STALLED_SINGULARITY") << result.message;
  EXPECT_GE(result.time_from_start, 2 * options.stall_window);
  EXPECT_LT(result.time_from_start, 3 * options.stall_window);
  EXPECT_NE(result.message.find("error_outside_tolerance_m="), std::string::npos);
}

TEST_F(StpikTest, UsesGoalTrackingTolerancesWithoutAdditionalPositionOrOrientationCaps)
{
  block_motion(); options.stall_window = 10;
  auto input = goal(0.03, 1.0, 2.0);
  input.path_tolerance = {0.02, 0.02, 0.21, 1.1, 1.1, 1.1};
  input.goal_tolerance = input.path_tolerance;
  const auto result = run(input);
  EXPECT_EQ(result.outcome, StpikAction::Result::SUCCESS) << result.code << ": " << result.message;
  EXPECT_NEAR(result.tracking_error[2], 0.03, 1e-9);
  EXPECT_GT(std::abs(result.tracking_error[5]), 0.3);
}

TEST_F(StpikTest, StillEnforcesStrictGoalPathTolerance)
{
  block_motion(); options.stall_window = 10;
  auto input = goal(0.005, 0, 2);
  input.path_tolerance[2] = 0.001;
  const auto result = run(input);
  EXPECT_EQ(result.code, "TRACKING_SINGULARITY");
  EXPECT_GT(result.tracking_error[2], 0.001);
}

TEST_F(StpikTest, StallDiagnosticDistinguishesOrientationAndCombinedStalls)
{
  block_motion();
  auto result = run(goal(0, 0.2, 8));
  EXPECT_EQ(result.code, "STALLED_SINGULARITY");
  EXPECT_EQ(result.constraint, "orientation");
  EXPECT_NE(result.message.find("position_stalled=false"), std::string::npos);
  EXPECT_NE(result.message.find("orientation_stalled=true"), std::string::npos);
  EXPECT_NE(result.message.find("position{requested=false"), std::string::npos);
  EXPECT_NE(result.message.find("reference_delta_rad="), std::string::npos);
  result = run(goal(0.01, 0.2, 8));
  EXPECT_EQ(result.constraint, "position_and_orientation");
  EXPECT_NE(result.message.find("position_stalled=true"), std::string::npos);
  EXPECT_NE(result.message.find("orientation_stalled=true"), std::string::npos);
}

TEST_F(StpikTest, OrientationTrackingIsCheckedInTheFrozenToleranceAxes)
{
  block_motion(); options.stall_window = 10;
  auto input = goal(0,0.8,2);
  input.path_tolerance = {0.02,0.02,0.02,0.01,0.01,0.01};
  input.tolerance_orientation.w = std::cos(0.35);
  input.tolerance_orientation.z = std::sin(0.35);
  const auto result = run(input);
  EXPECT_EQ(result.code, "TRACKING_SINGULARITY");
  EXPECT_GT(Eigen::Map<const Eigen::Vector3d>(result.tracking_error.data()+3).norm(), 0.01);
}

TEST_F(StpikTest, CancelsAndTimesOutWithoutDeclaringThePathInfeasible)
{
  const auto input = goal();
  auto result = StpikCheck::run(snapshot(), input, options, []() {return true;});
  EXPECT_EQ(result.outcome, StpikAction::Result::CANCELED);
  options.wall_timeout = 1e-12;
  result = run(input); EXPECT_EQ(result.outcome, StpikAction::Result::UNAVAILABLE);
  EXPECT_EQ(result.code, "CHECK_TIMEOUT");
}

TEST_F(StpikTest, DetectsEffectiveGainAndStateChangesEvenWhenParametersDidNotChange)
{
  auto captured = snapshot(); std::string reason;
  ASSERT_TRUE(pose->set_gain(std::vector<double>(6, 0.5), reason));
  EXPECT_FALSE(StpikTaskAccess::compatible(captured, state, *model, backend.get(), solver, tasks, options, reason));
  captured = snapshot(); state.joint_positions(1) += 0.02;
  EXPECT_FALSE(StpikTaskAccess::compatible(captured, state, *model, backend.get(), solver, tasks, options, reason));
}

TEST_F(StpikTest, SnapshotFeedforwardExpiryUsesSimulatedTime)
{
  auto clock = std::make_shared<std::chrono::steady_clock::time_point>(std::chrono::steady_clock::now());
  auto task = std::make_shared<FramePoseTask>();
  task->configure("other_pose", "FramePoseTask", {
    {"feedforward_timeout", rclcpp::Parameter("feedforward_timeout", 0.1)}},
    TaskContext{model, backend, {}, [clock]() {return *clock;}});
  PoseVelocity velocity = PoseVelocity::Zero(); velocity(0) = 0.1;
  ASSERT_TRUE(task->set_feedforward(velocity, task->feedforward_epoch()));
  tasks.push_back(task);
  auto captured = snapshot();
  auto isolated = std::dynamic_pointer_cast<PoseTaskCommon>(captured.tasks.back());
  auto simulated = std::make_shared<std::chrono::steady_clock::time_point>(*clock);
  StpikTaskAccess::bind(*isolated, TaskContext{captured.model, captured.backend, {}, [simulated]() {return *simulated;}});
  EXPECT_NEAR(isolated->update(state, *captured.backend).feedforward(0), 0.1, 1e-12);
  *simulated += std::chrono::seconds(1);
  EXPECT_TRUE(isolated->update(state, *captured.backend).feedforward.isZero());
  EXPECT_NEAR(task->update(state, *backend).feedforward(0), 0.1, 1e-12);
}

TEST_F(StpikTest, RequiresFreshCaptureWhenAnotherTimedInputCanExpireDuringValidation)
{
  auto other = std::make_shared<FramePoseTask>();
  other->configure("other_pose", "FramePoseTask", {
    {"feedforward_timeout", rclcpp::Parameter("feedforward_timeout", 10.0)}}, TaskContext{model, backend});
  PoseVelocity velocity = PoseVelocity::Zero(); velocity(0) = 0.1;
  ASSERT_TRUE(other->set_feedforward(velocity, other->feedforward_epoch())); tasks.push_back(other);
  auto captured = snapshot(); std::string reason;
  EXPECT_FALSE(StpikTaskAccess::compatible(captured, state, *model, backend.get(), solver, tasks, options, reason));
  EXPECT_NE(reason.find("timed pose feedforward"), std::string::npos);
  other->clear_feedforward(); captured = snapshot();
  EXPECT_TRUE(StpikTaskAccess::compatible(captured, state, *model, backend.get(), solver, tasks, options, reason)) << reason;
}

class UnknownTask : public TaskBaseCommon
{
public:
  void configure(const std::string & id, const std::string & plugin, const Params & params,
    const TaskContext & context) override {configure_common(id, plugin, params, context);}
  TaskComputation update(const WholeBodyState &, const KinematicsBackend &) override
  {
    TaskComputation computation;
    return computation;
  }
};
TEST_F(StpikTest, NeverSilentlyOmitsAnUnknownEnabledTask)
{
  auto unknown = std::make_shared<UnknownTask>();
  unknown->configure("external", "external/plugin", {}, TaskContext{model, backend});
  tasks.push_back(unknown);
  EXPECT_EQ(run(goal()).code, "UNSUPPORTED_ACTIVE_TASK");
  unknown->set_enabled(false);
  EXPECT_EQ(run(goal()).outcome, StpikAction::Result::SUCCESS);
}

TEST_F(StpikTest, RejectsBadInputsAndAnIncompatibleStartingPose)
{
  auto input = goal(); input.trajectory.start_pose.position.x = 0.2;
  EXPECT_EQ(run(input).code, "INITIAL_STATE_MISMATCH");
  input = goal(); input.trajectory.duration = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(run(input).code, "INVALID_TRAJECTORY");
  options.dt = 0;
  EXPECT_EQ(run(goal()).code, "INVALID_OPTIONS");
}
}  // namespace
}  // namespace task_priority_kinematic_control
