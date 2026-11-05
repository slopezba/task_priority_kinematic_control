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
  std::string unavailable_reason;
  const KinematicsBackend * source_backend = nullptr;  // Identity only; never dereferenced.
};

class StpikTaskAccess
{
public:
  // Caller serializes with the controller's task/state mutex. Never configures plugins.
  static StpikSnapshot capture(const WholeBodyState &, const std::shared_ptr<WholeBodyModel> &,
    const KinematicsBackendPtr &, const HierarchySolver &,
    const std::vector<std::shared_ptr<TaskBase>> &);
  static bool compatible(const StpikSnapshot &, const WholeBodyState &, const WholeBodyModel &,
    const KinematicsBackend *,
    const HierarchySolver &, const std::vector<std::shared_ptr<TaskBase>> &,
    const StpikOptions &, std::string &);
  static void bind(TaskBase &, const TaskContext &);
  // Called only on cloned tasks after their update; reads the same safety metrics.
  static bool safety(const TaskBase &, const WholeBodyState &, double tolerance,
    StpikAction::Result &);

private:
  static std::shared_ptr<TaskBase> copy(const TaskBase &, std::string &);
  static std::string key(const TaskBase &);
  static std::string configuration(const WholeBodyModel &, const HierarchySolver &,
    const std::vector<std::shared_ptr<TaskBase>> &);
};

class StpikCheck
{
public:
  using Canceled = std::function<bool()>;
  using Progress = std::function<void(double, double)>;
  // Consumes its private snapshot. Integration is purely kinematic, never hardware IO.
  static StpikAction::Result run(StpikSnapshot, const StpikAction::Goal &,
    const StpikOptions &, const Canceled & canceled = {}, const Progress & progress = {});
};
}  // namespace task_priority_kinematic_control
