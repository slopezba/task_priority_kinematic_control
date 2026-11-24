#pragma once
#include "task_priority_kinematic_control/core/stpik_check.hpp"
#include "sura_manipulation_actions/action/prepare_cartesian_posture.hpp"

namespace task_priority_kinematic_control
{
using Preparation = sura_manipulation_actions::action::PrepareCartesianPosture;
using PostureSummary = sura_manipulation_actions::msg::PosturePreparationSummary;
using TaskList = std::vector<std::shared_ptr<TaskBase>>;
struct PostureOptions
{
  double budget = 30, radius = 0.15, duration = 6, settle = 4;
  double movement = 0.05, tilt = 1, joints = 1, clearance = 1;
  double joint_buffer = 0.20, collision_buffer = 0.06;
  double position_tolerance = 0.01, orientation_tolerance = 0.035, joint_tolerance = 0.035;
  double stabilize = 1, self_collision = 1, active_penalty = 100;
  double obstacle_activation_distance = 0.02;
  // Nominal planning overrides this with posture.nominal.equality_tolerance.
  double equality_tolerance = 1e-3;
};
struct PosturePlan
{
  StpikSnapshot snapshot;
  Preparation::Goal request;
  WholeBodyState target;
  PostureSummary summary;
  PostureOptions options;
  StpikOptions simulation;
  bool move = false;
};
// Tasks that apply the posture. With a RobotNominalTask below `selected`: robot_nominal
// first, then its sync_tasks holds below `selected`. Otherwise the legacy holds
// (base_pose, left_arm_nominal, right_arm_nominal) below `selected`.
TaskList posture_command_tasks(const TaskList &, const std::string & selected);
TaskList prepare_tasks(const TaskList &, const WholeBodyModel &, const Preparation::Goal &);
// Quintic reference from start to target at `fraction` of the move. With robot_nominal,
// `duration` > 0 adds the velocity feedforward of the profile.
void posture_reference(
  const TaskList &, const WholeBodyModel &, const WholeBodyState & start,
  const WholeBodyState & target, double fraction, double duration = 0.0);
// Restore the enabled state and activation of the posture tasks from `original`.
void restore_posture_commands(const TaskList & commands, const TaskList & original);
bool posture_reached(const WholeBodyState &, const WholeBodyState &, const PostureOptions &);
bool tip_within(const Eigen::Isometry3d &, const Preparation::Goal &);
StpikCheck::ValidateTasks validate_posture_hierarchy(
  const TaskList &, const std::string & selected, double numerical_tolerance,
  double equality_tolerance = 1e-3);
PosturePlan optimize_posture(
  StpikSnapshot snapshot, const Preparation::Goal &,
  const StpikOptions &, const PostureOptions &, const StpikCheck::Canceled &,
  const std::function<void(unsigned)> & feedback);
// Cartesian tracking request equivalent to a posture preparation goal.
StpikAction::Goal posture_tracking_goal(const Preparation::Goal &);
// Isolated snapshot at the planned posture, with the posture holds already at it,
// as the controller is after a successful repositioning.
StpikSnapshot posture_snapshot(const PosturePlan &);
// Simulate only the repositioning from the snapshot to p.target (tool held at the
// path start, posture holds driven, safety and scene checked), not the tool path.
void check_repositioning(
  const PosturePlan &, const Preparation::Goal & fresh_scene, const StpikCheck::Canceled &);
void recheck_posture(
  const PosturePlan &, const Preparation::Goal & fresh_scene,
  const StpikCheck::Canceled &);
}
