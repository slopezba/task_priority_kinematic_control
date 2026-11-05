#include "task_priority_kinematic_control/core/stpik_check.hpp"
#include "task_priority_kinematic_control/kinematics/kdl_kinematics_backend.hpp"
#include "task_priority_kinematic_control/tasks/frame_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_nominal_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/self_collision_avoidance_task.hpp"
#include "sura_manipulation_actions/cartesian_pose_path.hpp"
#include <gtest/gtest.h>
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
  void collision(double radius)
  {
    auto task = std::make_shared<SelfCollisionAvoidanceTask>();
    task->configure("collision", "SelfCollisionAvoidanceTask", {
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
  auto result = run(goal(0, 0.02, 5));
  EXPECT_EQ(result.code, "STALLED_SINGULARITY");
  EXPECT_EQ(result.constraint, "orientation");
  EXPECT_NE(result.message.find("position_stalled=false"), std::string::npos);
  EXPECT_NE(result.message.find("orientation_stalled=true"), std::string::npos);
  EXPECT_NE(result.message.find("position{requested=false"), std::string::npos);
  EXPECT_NE(result.message.find("reference_delta_rad="), std::string::npos);
  result = run(goal(0.01, 0.02, 5));
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
