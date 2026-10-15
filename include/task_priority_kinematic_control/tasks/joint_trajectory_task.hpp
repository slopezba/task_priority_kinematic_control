#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"

#include <chrono>
#include <mutex>

namespace task_priority_kinematic_control
{

class JointTrajectoryTask : public TaskBaseCommon
{
public:
  void configure(
    const std::string & id,
    const std::string & plugin_name,
    const std::map<std::string, rclcpp::Parameter> & parameters,
    const TaskContext & context) override;

  TaskComputation update(
    const WholeBodyState & state,
    const KinematicsBackend & backend) override;

  bool set_joint_trajectory(
    const trajectory_msgs::msg::JointTrajectory & trajectory,
    std::string & message) override;
  bool cancel_joint_trajectory(std::string & message) override;
  JointTrajectoryTaskStatus joint_trajectory_status() const override;
  bool set_gain(const std::vector<double> & gain, std::string & message) override;
  bool set_enabled(bool enabled) override;
  bool enabled() const override;
  void reset() override;
  msg::TaskStatus build_status() const override;
  std::vector<double> current_target() const override;

  // The coordinated executor validates before changing any task and applies
  // goal-local tolerances without changing ROS parameters.
  bool validate_trajectory(const trajectory_msgs::msg::JointTrajectory &, std::string &) const;
  bool sample_trajectory(double elapsed, Eigen::VectorXd & position) const;
  void set_execution_tolerances(const std::vector<double> & tolerance, double timeout);
  void restore_execution_defaults();

private:
  using SteadyClock = std::chrono::steady_clock;

  struct Sample
  {
    Eigen::VectorXd position;
    Eigen::VectorXd velocity;
    Eigen::VectorXd acceleration;
    bool has_velocity = false;
    double time_from_start = 0.0;
  };

  bool validate_and_map_trajectory(
    const trajectory_msgs::msg::JointTrajectory & trajectory,
    std::vector<Sample> & samples,
    std::string & message) const;
  Sample sample_at_locked(double elapsed) const;
  double trajectory_duration_locked() const;
  void clear_trajectory_locked(const std::string & status, bool canceled);

  friend class JointTrajectoryInterpolationTest;

  std::vector<std::string> joint_names_;
  Eigen::VectorXd gains_;
  Eigen::VectorXd goal_tolerance_;
  double trajectory_timeout_{0.5};
  bool hold_last_point_{true};
  std::string interpolation_method_{"linear"};

  mutable std::mutex mutex_;
  std::vector<Sample> samples_;
  SteadyClock::time_point start_time_;
  int64_t start_time_ros_ns_{0};
  int64_t last_ros_time_ns_{0};
  bool use_ros_time_{false};
  bool ros_cycle_seen_{false};
  Eigen::VectorXd configured_goal_tolerance_;
  double configured_trajectory_timeout_{0.5};
  std::string configured_interpolation_method_{"linear"};
  bool configured_hold_last_point_{true};
  Eigen::VectorXd current_target_;
  Eigen::VectorXd last_error_;
  double final_error_norm_{0.0};
  bool trajectory_active_{false};
  bool succeeded_{false};
  bool canceled_{false};
  bool timed_out_{false};
  std::string status_message_{"idle"};
};

}  // namespace task_priority_kinematic_control
