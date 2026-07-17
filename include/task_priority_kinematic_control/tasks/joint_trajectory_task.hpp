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
  void reset() override;
  msg::TaskStatus build_status() const override;
  std::vector<double> current_target() const override;

private:
  using SteadyClock = std::chrono::steady_clock;

  struct Sample
  {
    Eigen::VectorXd position;
    Eigen::VectorXd velocity;
    bool has_velocity = false;
    double time_from_start = 0.0;
  };

  bool validate_and_map_trajectory(
    const trajectory_msgs::msg::JointTrajectory & trajectory,
    std::vector<Sample> & samples,
    std::string & message) const;
  Sample sample_at_locked(double elapsed) const;
  double trajectory_duration_locked() const;

  std::vector<std::string> joint_names_;
  Eigen::VectorXd gains_;
  Eigen::VectorXd goal_tolerance_;
  double trajectory_timeout_{0.5};
  bool hold_last_point_{true};

  mutable std::mutex mutex_;
  std::vector<Sample> samples_;
  SteadyClock::time_point start_time_;
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
