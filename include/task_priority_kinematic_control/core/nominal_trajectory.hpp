#pragma once
#include "task_priority_kinematic_control/core/initial_posture.hpp"
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>
#include <functional>
#include <string>
#include <vector>

namespace task_priority_kinematic_control
{
// Whole-body nominal trajectory planning for the robot_nominal task.
//
// Coordinates per knot: [base x, y, z, roll, pitch, yaw (ZYX, world), model joints].
// 1. Seed: offline STPIK rollout of the current hierarchy (what the controller does).
// 2. SQP (OSQP) over all knots: tool pose on the timed path, linearized higher tasks
//    (hard above the tool task, soft between it and robot_nominal), joint/tilt bounds,
//    MoveIt clearance and velocity limits between knots; smoothness/margin/tilt costs.
//    Feasible SQP steps minimize activation-margin deficits first, then preserve
//    each achieved deficit while optimizing the secondary costs.
// 3. Global fallback when the seed or the SQP fail: layered sampling of the self-motion
//    manifold at time layers + dynamic programming, refined again by the SQP.
// 4. Mandatory STPIK validation with robot_nominal streamed as at runtime.
struct NominalOptions
{
  double budget = 30.0;
  double knot_dt = 0.25;
  double max_knots = 60;
  double iterations = 8;
  double trust_position = 0.08;
  double trust_angle = 0.15;
  double trust_joint = 0.3;
  double smooth_weight = 1.0;
  double soft_task_weight = 50.0;
  // Per equality row, in task units: metres for position, radians for angles.
  double equality_tolerance = 0.02;
  double velocity_fraction = 0.8;
  double global_layers = 16;
  double global_samples = 32;
  double global_seed = 7;
  bool global_fallback = true;
  // false skips the rollout seed and forces the layered search (testing/benchmarks).
  bool rollout_seed = true;
  bool valid(std::string & reason) const;
};

struct NominalPlan
{
  std::vector<double> times;
  std::vector<Eigen::VectorXd> knots;
  trajectory_msgs::msg::JointTrajectory joints;
  trajectory_msgs::msg::MultiDOFJointTrajectory base;
  std::string method;
  double cost = 0.0;
  double preventive_cost = 0.0;  // Higher priority than cost; normalized squared deficits.
  StpikAction::Result validation;
};

using NominalFeedback = std::function<void(const std::string &)>;

// `start` is an isolated snapshot of the controller. With `free_start` the first knot
// (the initial posture) is optimized together with the trajectory: its tool pose is the
// path start, moving away from the measured posture costs posture.movement_weight, the
// move fits posture.duration, and the repositioning is simulated before validating the
// path from that posture (knots[0] is the proposal). Otherwise the path starts at the
// snapshot state. `warm` (optional, implies a fixed start) is a previous plan to refine.
NominalPlan optimize_nominal_trajectory(
  const StpikSnapshot & start, const Preparation::Goal & goal, const StpikOptions & simulation,
  const PostureOptions & posture, const NominalOptions & options,
  const StpikCheck::Canceled & canceled, const NominalFeedback & feedback,
  const NominalPlan * warm = nullptr, bool free_start = false);

// Tracking request of `goal` carrying the nominal for robot_nominal.
StpikAction::Goal nominal_tracking_goal(const Preparation::Goal & goal, const NominalPlan & plan);
}  // namespace task_priority_kinematic_control
