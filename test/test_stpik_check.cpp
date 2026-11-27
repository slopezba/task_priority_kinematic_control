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
#include "task_priority_kinematic_control/core/local_prediction.hpp"
#include "task_priority_kinematic_control/tasks/robot_nominal_task.hpp"
#include "sura_manipulation_actions/whole_body_nominal.hpp"
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <limits>
#include "task_priority_kinematic_control/core/stpik_check_io.hpp"
#include "task_priority_kinematic_control/tasks/end_effectors_facing_task.hpp"

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
  StpikAction::Goal local_goal(bool moving)
  {
    auto request = goal(moving ? 0.05 : 0,0,2);
    auto nominal = std::make_shared<RobotNominalTask>();
    nominal->configure("robot_nominal","RobotNominalTask",{
      {"gain",rclcpp::Parameter("gain",std::vector<double>(8,0.0))}
    },TaskContext{model,backend});
    tasks.push_back(nominal);
    auto & p = request.prediction;
    p.robot_description = robot;
    // The original checker fixture deliberately has a 1 mm/s URDF limit.
    // Use a 1 m/s fixture for these paired trials, with identical caps in both arms.
    const std::string slow = "velocity=\"0.001\"";
    for (auto i=p.robot_description.find(slow); i!=std::string::npos; i=p.robot_description.find(slow)) {
      p.robot_description.replace(i,slow.size(),"velocity=\"1\"");
    }
    p.robot_description_semantic = R"(<robot name="stpik_fixture"><virtual_joint
      name="root" type="floating" parent_frame="world" child_link="base"/></robot>)";
    p.scene.robot_model_name = "stpik_fixture";
    p.scene.robot_state.joint_state.name = {"l","r"};
    p.scene.robot_state.joint_state.position = {state.joint_positions(0),state.joint_positions(1)};
    p.global_progress = {0,1};
    p.start_progress = p.maximum_progress = moving ? 0.0 : 0.4;
    p.progress_rate_scale = moving ? 1.0 : 0.0;
    p.horizon = 0.2; p.budget = 10; p.improvement = 0;
    p.recovery_threshold = 0.02; p.gain = 2.5;
    options.dt = 0.025;
    std::vector<sura_manipulation_actions::BaseCoordinates> base(2,sura_manipulation_actions::BaseCoordinates::Zero());
    base.back()(2) = moving ? 0.05 : 0;
    std::string reason;
    EXPECT_TRUE(sura_manipulation_actions::buildWholeBodyNominal({0,2},base,
      {Eigen::Vector2d::Zero(),Eigen::Vector2d::Zero()},model->all_joint_names(),"world",
      request.nominal_joints,request.nominal_base,reason)) << reason;
    return request;
  }
  std::pair<StpikAction::Result,StpikAction::Result> compare(StpikAction::Goal request)
  {
    const auto start = snapshot();
    auto trial = [&](bool explore) {
      request.prediction.explore = explore;
      const auto report = predict_local_segment(start,request,options);
      const auto name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
      std::cout << "LOCAL_TRIAL," << name << "," << explore << "," << report.code << ","
                << report.configuration_error << "," << report.hierarchy_rank << ","
                << report.candidates << "," << report.compute_seconds << std::endl;
      if (const char * path = std::getenv("SURA_LOCAL_PLANNER_CSV")) {
        std::ofstream csv(path,std::ios::app);
        for (const auto & step : report.trace) {
          csv << name << ',' << explore << ',' << report.code << ',' << report.compute_seconds << ','
              << step.elapsed << ',' << step.progress << ',' << step.confirmed_progress << ','
              << step.configuration_error << ',' << step.hierarchy_rank << ',' << step.clearance;
          for (double e : step.cartesian_error) {csv << ',' << e;}
          csv << ',';
          for (const auto & id : step.active_constraints) {csv << id << ' ';}
          csv << '\n';
        }
      }
      return report;
    };
    auto baseline = trial(false);
    return {baseline,trial(true)};
  }
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

// Facing fixture: both tools on the world X axis; the left prismatic joint changes the distance.
class FacingStpikTest : public StpikTest
{
protected:
  void SetUp() override
  {
    StpikTest::SetUp();
    facing = std::make_shared<EndEffectorsFacingTask>();
    facing->configure("facing", "EndEffectorsFacingTask", {
      {"enabled", rclcpp::Parameter("enabled", true)},
      {"priority", rclcpp::Parameter("priority", 1.0)},
      {"left_frame", rclcpp::Parameter("left_frame", "left")},
      {"right_frame", rclcpp::Parameter("right_frame", "right")},
      {"left_pointing_axis", rclcpp::Parameter("left_pointing_axis", std::vector<double>{1, 0, 0})},
      {"right_pointing_axis", rclcpp::Parameter("right_pointing_axis", std::vector<double>{-1, 0, 0})},
      {"gain", rclcpp::Parameter("gain", std::vector<double>(6, 2.0))}}, TaskContext{model, backend});
    nominal = std::make_shared<RobotNominalTask>();
    nominal->configure("robot_nominal", "RobotNominalTask", {
      {"enabled", rclcpp::Parameter("enabled", true)},
      {"priority", rclcpp::Parameter("priority", 2.0)},
      {"gain", rclcpp::Parameter("gain", std::vector<double>(8, 1.0))}},
      TaskContext{model, backend});
    tasks = {facing, nominal};
  }
  StpikAction::Goal facing_goal()
  {
    StpikAction::Goal out;
    out.trajectory.header.frame_id = "world";
    out.task_id = "facing";
    out.nominal_activation = {false, false, false, false, false, false, true, true};
    out.nominal_base.header.frame_id = "world";
    out.nominal_base.joint_names = {"floating_base"};
    out.facing_path.left_goal_axis.x = 1.0;
    out.facing_path.right_goal_axis.x = -1.0;
    for (int i = 0; i <= 1; ++i) {
      sura_manipulation_actions::msg::FacingPathPoint point;
      point.progress = i;
      point.left_pointing_axis.x = 1.0;
      point.right_pointing_axis.x = -1.0;
      point.roll_angle = std::acos(-1.0);
      point.distance = 0.09 + 0.03 * i;
      out.facing_path.points.push_back(point);
      trajectory_msgs::msg::JointTrajectoryPoint joint;
      joint.positions = {-0.03 * i, 0.0};
      joint.time_from_start.sec = 2 * i;
      out.nominal_joints.points.push_back(joint);
      // A base pose far from the measured one: the mask must keep it from moving the base.
      sura_manipulation_actions::BaseCoordinates base = sura_manipulation_actions::BaseCoordinates::Zero();
      base(0) = 0.5;
      out.nominal_base.points.push_back(sura_manipulation_actions::baseTrajectoryPoint(
        base, sura_manipulation_actions::BaseCoordinates::Zero(),
        sura_manipulation_actions::BaseCoordinates::Zero(), joint.time_from_start));
    }
    out.nominal_joints.joint_names = {"l", "r"};
    out.nominal_progress = {0.0, 1.0};
    out.path_tolerance = {0.05, 0.05, 0.05, 0.05, 0.05, 0.005};
    out.goal_tolerance = {0.01, 0.01, 0.01, 0.01, 0.01, 0.001};
    out.goal_time_tolerance = 1.0;
    return out;
  }
  std::shared_ptr<EndEffectorsFacingTask> facing;
  std::shared_ptr<RobotNominalTask> nominal;
};

TEST_F(FacingStpikTest, SimulatesTheFacingPathWithTheMaskedRobotNominal)
{
  const auto result = run(facing_goal());
  EXPECT_EQ(result.outcome, StpikAction::Result::SUCCESS) << result.code << ": " << result.message;
  ASSERT_EQ(result.joint_positions.size(), 2u);
  EXPECT_NEAR(result.joint_positions[0], -0.03, 0.002);
  EXPECT_NEAR(result.base_pose.position.x, 0.0, 1e-9);
  // The live tasks are untouched by the isolated simulation.
  EXPECT_FALSE(nominal->has_live_reference());
}

TEST_F(FacingStpikTest, RejectsUnreachableOrIncompleteHierarchies)
{
  auto request = facing_goal();
  request.facing_path.points.back().roll_angle = std::acos(-1.0) - 0.5;
  auto result = run(request);
  EXPECT_EQ(result.outcome, StpikAction::Result::FAIL);
  // The roll rows are unactuated here, so the report may name the singularity.
  EXPECT_NE(result.code.find("TRACKING"), std::string::npos) << result.message;
  request = facing_goal();
  request.nominal_progress = {0.0};
  EXPECT_EQ(run(request).code, "INVALID_TRAJECTORY");
  nominal->set_enabled(false);
  EXPECT_EQ(run(facing_goal()).code, "NOMINAL_TASK_UNAVAILABLE");
}

TEST_F(FacingStpikTest, ReadinessChecksEnabledTasksFramesAndPriorities)
{
  sura_manipulation_actions::srv::CheckFacingExecution::Request request;
  request.facing_task = "facing"; request.nominal_task = "robot_nominal";
  request.left_frame = "left"; request.right_frame = "right";
  request.min_distance = 0.02;
  request.disabled_tasks = {"left_pose"};
  std::string message;
  EXPECT_TRUE(StpikCheckIO::facing_ready(tasks, request, message)) << message;
  auto with_pose = tasks;
  with_pose.push_back(pose);
  EXPECT_FALSE(StpikCheckIO::facing_ready(with_pose, request, message));
  EXPECT_NE(message.find("left_pose"), std::string::npos);
  nominal->set_priority(1);
  EXPECT_FALSE(StpikCheckIO::facing_ready(tasks, request, message));
  nominal->set_priority(2);
  request.left_frame = "left_end";
  EXPECT_FALSE(StpikCheckIO::facing_ready(tasks, request, message));
}

TEST_F(StpikTest, LocalFrozenNominalIsRetained)
{
  auto request = local_goal(false);
  auto [nominal,local] = compare(request);
  ASSERT_EQ(local.outcome,local.SUCCESS) << local.message;
  ASSERT_EQ(nominal.outcome,nominal.SUCCESS) << nominal.message;
  EXPECT_EQ(local.candidates,1u);
  ASSERT_EQ(local.explored_paths.size(), 1u);
  EXPECT_TRUE(local.explored_paths.front().rejoined);
  EXPECT_EQ(local.explored_paths.front().poses.size(), 1u);
  EXPECT_EQ(local.segment,nominal.segment);
  for (const auto & point : local.trace) {EXPECT_DOUBLE_EQ(point.progress,0.4);}
}

TEST_F(StpikTest, LocalFrozenRecoveryWithIdenticalPerturbation)
{
  state.joint_positions(1) = 0.06;
  auto request = local_goal(false);
  request.prediction.weights = {1,1,8,0,0,0.001};
  auto [nominal,local] = compare(request);
  ASSERT_EQ(local.outcome,local.SUCCESS) << local.message;
  ASSERT_EQ(nominal.outcome,nominal.SUCCESS) << nominal.message;
  ASSERT_FALSE(nominal.trace.empty());
  EXPECT_NEAR(nominal.configuration_error,nominal.trace.front().configuration_error,1e-6);
  EXPECT_LT(local.configuration_error,nominal.configuration_error);
  EXPECT_LT(local.segment_cost,local.continuation_cost);
  EXPECT_GT(local.candidates,1u);
  ASSERT_EQ(local.explored_paths.size(), local.candidates);
  for (const auto & candidate : local.explored_paths) {
    if (!candidate.poses.empty()) {
      EXPECT_EQ(candidate.poses.front(), local.trace.front().tip_pose);
    }
  }
  for (const auto & point : local.trace) {EXPECT_DOUBLE_EQ(point.progress,0.4);}
  EXPECT_DOUBLE_EQ(state.joint_positions(1),0.06);
}

TEST_F(StpikTest, LocalFrozenReconnectsAndMaintainsTolerance)
{
  state.joint_positions(1) = 0.06;
  auto request = local_goal(false);
  request.prediction.weights = {1,1,8,0,0,0.001};
  const auto disturbed = state;
  for (bool explore : {false,true}) {
    state = disturbed;
    request.prediction.explore = explore;
    size_t consecutive = 0;
    for (int i = 0; i < 14; ++i) {
      auto result = predict_local_segment(snapshot(),request,options);
      ASSERT_EQ(result.outcome,result.SUCCESS) << result.message;
      ASSERT_EQ(result.joint_positions.size(),2u);
      state.joint_positions = Eigen::Map<const Eigen::Vector2d>(result.joint_positions.data());
      const auto & base = result.base_pose;
      state.base_position << base.position.x,base.position.y,base.position.z;
      const auto & q = base.orientation;
      state.base_rpy = Eigen::Quaterniond(q.w,q.x,q.y,q.z).toRotationMatrix().eulerAngles(2,1,0).reverse();
      consecutive = result.configuration_error <= request.prediction.recovery_threshold ? consecutive+1 : 0;
    }
    if (explore) {EXPECT_GE(consecutive,2u);} else {EXPECT_EQ(consecutive,0u);}
  }
}

TEST_F(StpikTest, LocalMovingPathChecksBothArms)
{
  auto request = local_goal(true);
  auto [nominal,local] = compare(request);
  ASSERT_EQ(local.outcome,local.SUCCESS) << local.message;
  EXPECT_EQ(nominal.outcome,nominal.SUCCESS);
  EXPECT_GT(local.segment.progress.back(),local.segment.progress.front());
  ASSERT_GT(local.trace.size(), 1u);
  const auto & start = request.trajectory.start_pose.position;
  const auto & first = local.trace.front().tip_pose.position;
  EXPECT_NEAR(first.x, start.x, 1e-9);
  EXPECT_NEAR(first.y, start.y, 1e-9);
  EXPECT_NEAR(first.z, start.z, 1e-9);
  EXPECT_GT(local.trace.back().tip_pose.position.z, first.z);
  EXPECT_EQ(local.segment.joints.joint_names,model->all_joint_names());
}

TEST_F(StpikTest, LocalVisualizationExtendsPastAcceptedSegment)
{
  auto request = local_goal(false);
  request.prediction.explore = false;
  state.base_position.z() = 0.015;
  options.snapshot_position_tolerance = 0.05;
  options.rejoin_position_tolerance = 0.004;
  const auto result = predict_local_segment(snapshot(), request, options);
  ASSERT_EQ(result.outcome, result.SUCCESS) << result.message;
  ASSERT_FALSE(result.trace.empty());
  ASSERT_EQ(result.explored_paths.size(), 1u);
  const auto & explored = result.explored_paths.front();
  ASSERT_TRUE(explored.rejoined);
  ASSERT_GT(explored.poses.size(), result.trace.size());
  EXPECT_EQ(explored.poses.front(), result.trace.front().tip_pose);
  EXPECT_DOUBLE_EQ(result.trace.back().elapsed, request.prediction.horizon);
  const double target_z = request.trajectory.start_pose.position.z;
  EXPECT_GT(std::abs(result.trace.back().tip_pose.position.z - target_z),
    options.rejoin_position_tolerance);
  EXPECT_NEAR(explored.poses.back().position.z, target_z, options.rejoin_position_tolerance);
}

TEST_F(StpikTest, LocalActivatedRestrictionAndNoLocalSolution)
{
  collision(0.01);
  auto request = local_goal(false);
  state.joint_positions(0) = 0.02;
  auto [baseline,limited] = compare(request);
  ASSERT_EQ(limited.outcome,limited.SUCCESS) << limited.message;
  EXPECT_NE(std::find(limited.active_constraints.begin(),limited.active_constraints.end(),"collision"),
    limited.active_constraints.end());
  state.joint_positions(0) = 0.06;
  auto [nominal,local] = compare(request);
  EXPECT_EQ(nominal.code,"NO_LOCAL_CONTINUATION");
  EXPECT_EQ(local.code,"NO_LOCAL_CONTINUATION") << local.message;
  EXPECT_TRUE(local.segment.progress.empty());
}

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
