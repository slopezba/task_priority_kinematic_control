#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"
#include <cstdint>
#include <mutex>
#include <chrono>

namespace task_priority_kinematic_control
{
using PoseVelocity = Eigen::Matrix<double, 6, 1>;

// Epochs reject callbacks spanning a disable/reset. Expiry is opt-in.
class PoseTaskCommon : public TaskBaseCommon
{
public:
  uint64_t feedforward_epoch() const;
  bool set_feedforward(const PoseVelocity & velocity, uint64_t epoch);
  void clear_feedforward();
  // Used when returning ownership after trajectory execution: capture the
  // current frame even when auto_update_target_pose is disabled.
  void capture_measured_target(const Eigen::Isometry3d & pose);
  bool set_feedforward_option(const std::string & field, const rclcpp::Parameter & parameter,
    std::string & message);
  bool set_enabled(bool enabled) override;
  void reset() override;
  std::vector<double> current_target() const override;

protected:
  void configure_pose_feedforward(const std::map<std::string, rclcpp::Parameter> & parameters);
  PoseVelocity pose_feedforward() const;
  void update_pose_target(const Eigen::Isometry3d & pose, TaskComputation & computation,
    const PoseVelocity & activation = PoseVelocity::Ones());

private:
  mutable std::mutex feedforward_mutex_;
  PoseVelocity feedforward_ = PoseVelocity::Zero();
  uint64_t feedforward_epoch_ = 0;
  bool accept_feedforward_ = false;
  bool auto_update_target_pose_ = true;
  double feedforward_velocity_threshold_ = 0.001;
  bool initialize_target_ = true;
  double feedforward_timeout_ = 0.0;
  std::chrono::steady_clock::time_point feedforward_time_;
};
}  // namespace task_priority_kinematic_control
