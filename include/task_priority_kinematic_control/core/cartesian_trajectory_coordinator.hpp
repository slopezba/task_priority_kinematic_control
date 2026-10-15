#pragma once

#include "task_priority_kinematic_control/core/task_manager.hpp"
#include "task_priority_kinematic_control/tasks/base_pose_trajectory_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_trajectory_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_nominal_task.hpp"
#include "sura_manipulation_actions/action/follow_cartesian_trajectory.hpp"

#include <set>

namespace task_priority_kinematic_control
{

// All calls are serialized by the host's task mutex, including admission,
// manual target changes, controller updates and lifecycle transitions.
class CartesianTrajectoryCoordinator
{
public:
  using Action = sura_manipulation_actions::action::FollowCartesianTrajectory;
  struct Settings
  {
    std::string base_task = "base_pose_trajectory";
    std::string left_joint_task = "left_joint_trajectory";
    std::string right_joint_task = "right_joint_trajectory";
    std::string left_hold_task = "left_arm_nominal";
    std::string right_hold_task = "right_arm_nominal";
    std::vector<std::string> safety_tasks = {
      "joint_limits", "self_collision_avoidance", "stabilize_limits"};
    double other_arm_tolerance = 0.05;
  };
  struct Status
  {
    bool running = false;
    bool complete = false;
    bool success = false;
    bool canceled = false;
    bool holding = false;
    float progress = 0.0F;
    std::string message = "idle";
    Eigen::Matrix<double, 6, 1> base_error = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::VectorXd joint_error;
    Eigen::Matrix<double, 6, 1> tip_error = Eigen::Matrix<double, 6, 1>::Zero();
  };
  void configure(TaskManager &, const std::shared_ptr<WholeBodyModel> &,
    const rclcpp::Clock::SharedPtr &, const Settings &);
  bool validate_goal(const Action::Goal &, const WholeBodyState &,
    const KinematicsBackend &, std::string &) const;
  bool install(const Action::Goal &, const WholeBodyState &,
    const KinematicsBackend &, std::string &);
  // Called after task update. true asks the host to recompute after a stop/hold.
  bool update(const WholeBodyState &, const KinematicsBackend &);
  bool observe_command(const WholeBodyCommand &, const WholeBodyState &,
    const KinematicsBackend &);
  void cancel_to_measured(const WholeBodyState &, const std::string &, bool canceled = true);
  bool release(const WholeBodyState &, const KinematicsBackend &, std::string &);
  void shutdown(const WholeBodyState &, const KinematicsBackend &, const std::string &);
  bool allow_mutation(const std::string &, const WholeBodyState &,
    const KinematicsBackend &, std::string &);
  bool reserved(const std::string &) const;
  bool owns_tasks() const {return !snapshots_.empty();}
  const Status & status() const {return status_;}

private:
  bool normalize_goal(const Action::Goal &, Action::Goal &, std::string &) const;
  bool validate_ordered_goal(const Action::Goal &, const WholeBodyState &,
    const KinematicsBackend &, std::string &) const;
  struct Snapshot
  {
    std::shared_ptr<TaskBase> task;
    bool enabled;
    uint32_t priority;
    std::vector<bool> joint_activation;
  };
  std::shared_ptr<TaskBase> find_task(const std::string &) const;
  void install_measured_hold(const WholeBodyState &);
  TaskManager * manager_ = nullptr;
  std::shared_ptr<WholeBodyModel> model_;
  rclcpp::Clock::SharedPtr clock_;
  Settings settings_;
  std::vector<Snapshot> snapshots_;
  std::shared_ptr<BasePoseTrajectoryTask> base_task_;
  std::shared_ptr<JointTrajectoryTask> joint_task_;
  std::shared_ptr<JointNominalTask> other_hold_task_;
  Action::Goal goal_;
  std::vector<sura_manipulation_actions::BaseTrajectoryKnot> base_knots_;
  std::vector<std::string> task_joint_names_;
  // Controller-order index -> original goal index, for action feedback.
  std::vector<size_t> feedback_indices_;
  WholeBodyState reference_state_;
  int64_t start_ns_ = 0;
  int64_t previous_ns_ = 0;
  double duration_ = 0.0;
  Eigen::Matrix<double, 6, 1> lower_, upper_;
  Eigen::Matrix<double, 6, 1> previous_solution_ = Eigen::Matrix<double, 6, 1>::Zero();
  int64_t previous_solution_ns_ = 0;
  Status status_;
};

}  // namespace task_priority_kinematic_control
