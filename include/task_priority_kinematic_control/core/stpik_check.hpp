#pragma once

#include "task_priority_kinematic_control/core/hierarchy_solver.hpp"
#include "task_priority_kinematic_control/core/task_manager.hpp"
#include "sura_manipulation_actions/action/check_cartesian_trajectory.hpp"
#include <chrono>
#include <functional>

namespace task_priority_kinematic_control
{
using StpikAction = sura_manipulation_actions::action::CheckCartesianTrajectory;

struct StpikOptions
{
  double dt = 0.01;
  double stall_window = 1.0;
  // Maximum progress requirements per window. Near the goal, use 10% of the
  // error outside per-axis goal tolerances, with numerical_tolerance as a floor.
  double min_position_progress = 0.0001;
  double min_orientation_progress = 0.001;
  double numerical_tolerance = 1e-6;
  double singular_value_tolerance = 1e-6;
  double wall_timeout = 30.0;
  // Optional stricter limits in generalized controller order. Empty uses runtime caps.
  std::vector<double> velocity_limits;
  double snapshot_joint_tolerance = 0.001;
  double snapshot_position_tolerance = 0.005;
  double snapshot_orientation_tolerance = 0.01;
  double rejoin_position_tolerance = 0.005;
  double rejoin_orientation_tolerance = 0.01;
  bool valid(std::string & reason) const;
  std::string configuration_key() const;
};

// Owning, isolated snapshot. No node, publishers, hardware handles or shared tasks.
struct StpikSnapshot
{
  WholeBodyState initial_state;
  std::shared_ptr<WholeBodyModel> model;
  KinematicsBackendPtr backend;
  HierarchySolver solver;
  std::vector<std::shared_ptr<TaskBase>> tasks;
  std::chrono::steady_clock::time_point captured;
  std::string configuration;
  std::string streamed_pose;
  std::string unavailable_reason;
  const KinematicsBackend * source_backend = nullptr;  // Identity only; never dereferenced.
};

// Error coordinates and their derivative with respect to task-space motion.
// Limits apply per axis, or to the Euclidean norm for a radial region.
struct PostureRegion
{
  Eigen::VectorXd error;
  Eigen::VectorXd limit;
  Eigen::MatrixXd jacobian;
  bool radial = false;
  bool always_active = false;
};

// Signed normalized deficit to the release margin: <= 0 is preventive-safe,
// 1 is the activation boundary. Gradient uses generalized velocity coordinates.
struct PreventiveMargin
{
  double deficit = 0.0;
  Eigen::RowVectorXd gradient;
};

struct PostureSafetyCosts
{
  double joints = 0.0;
  double stabilize = 0.0;
  double self_collision = 0.0;
};

// Distance to the hard safety boundary, activation distance, and preventive distance.
// Activation follows the task's latch (including hysteresis), not just distance.
double posture_limit_cost(
  double distance, double activation, double preventive, bool active, double active_penalty);

class StpikTaskAccess
{
public:
  // Caller serializes with the controller's task/state mutex. Never configures plugins.
  static StpikSnapshot capture(const WholeBodyState &, const std::shared_ptr<WholeBodyModel> &,
    const KinematicsBackendPtr &, const HierarchySolver &,
    const std::vector<std::shared_ptr<TaskBase>> &, const std::string & streamed_pose = {});
  static bool compatible(const StpikSnapshot &, const WholeBodyState &, const WholeBodyModel &,
    const KinematicsBackend *,
    const HierarchySolver &, const std::vector<std::shared_ptr<TaskBase>> &,
    const StpikOptions &, std::string &);
  static bool joint_bounds(const std::vector<std::shared_ptr<TaskBase>> & tasks, Eigen::VectorXd & lower, Eigen::VectorXd & upper);
  // Roll/pitch bounds of the first enabled StabilizeLimitsTask.
  static bool tilt_bounds(const std::vector<std::shared_ptr<TaskBase>> & tasks,
    Eigen::Vector2d & lower, Eigen::Vector2d & upper);
  static void bind(TaskBase &, const TaskContext &);
  // Called only on cloned tasks after their update; reads the same safety metrics.
  static bool safety(const TaskBase &, const WholeBodyState &, double tolerance,
    StpikAction::Result &);
  // Reads the full configured region after update, including inactive rows.
  static bool posture_region(const TaskBase &, PostureRegion &, bool linearize = false);
  static std::vector<PreventiveMargin> preventive_margins(
    const TaskBase &, const WholeBodyState &);
  static PostureSafetyCosts posture_safety_costs(
    const TaskBase &, const WholeBodyState &, double active_penalty);

private:
  static std::shared_ptr<TaskBase> copy(const TaskBase &, std::string &);
  static std::string key(const TaskBase &, bool streamed);
  static std::string configuration(const WholeBodyModel &, const HierarchySolver &,
    const std::vector<std::shared_ptr<TaskBase>> &, const std::string & streamed_pose);
};

class StpikCheck
{
public:
  using Canceled = std::function<bool()>;
  using Progress = std::function<void(double, double)>;
  using ValidateTasks = std::function<void(const std::vector<TaskComputation> &)>;
  struct Rollout
  {
    // Exact geometric reference and spatial velocity at local execution time.
    std::function<void(double, Eigen::Isometry3d &, Eigen::Matrix<double, 6, 1> &)> target;
    HierarchySolver::Prefix prefix;
    std::function<void(double, const WholeBodyState &, const std::vector<TaskComputation> &,
      const Eigen::Matrix<double, 6, 1> &, const HierarchySolver::Prefix &, WholeBodyCommand &)> command;
    std::function<void(double, const Eigen::Isometry3d &)> observe_pose;
    std::function<bool(double, const Eigen::Isometry3d &)> stop_when;
  };
  // Consumes its private snapshot. Integration is purely kinematic, never hardware IO.
  static StpikAction::Result run(StpikSnapshot, const StpikAction::Goal &,
    const StpikOptions &, const Canceled & canceled = {}, const Progress & progress = {},
    const std::function<void(const WholeBodyState &, double)> & observe = {},
    const std::function<void(const WholeBodyState &, double)> & reference = {},
    const ValidateTasks & validate_tasks = {}, Rollout * rollout = nullptr);
};
}  // namespace task_priority_kinematic_control
