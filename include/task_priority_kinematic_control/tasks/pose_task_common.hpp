#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"
#include <cstdint>
#include <mutex>
#include <chrono>
#include <optional>

namespace task_priority_kinematic_control
{
using PoseVelocity = Eigen::Matrix<double, 6, 1>;

// Shared pose helpers; namespaced to avoid clashing with file-local copies.
namespace pose_math
{
// Axis-angle vector of target * current^-1 (world-frame rotation error).
Eigen::Vector3d quaternion_error(const Eigen::Quaterniond & target, const Eigen::Quaterniond & current);
// World pose of the base from navigation position and ZYX roll/pitch/yaw.
Eigen::Isometry3d base_pose_from_state(const WholeBodyState & state);
// 6 x total_dofs Jacobian selecting the active base DoFs.
Eigen::MatrixXd base_jacobian_for_model(const WholeBodyModel & model);
// Moves a frame state to the controlled point `frame * offset` (rigid offset in the frame's own
// axes): pose becomes pose * offset, J_lin -= skew(R * t_offset) * J_ang and J_ang is unchanged.
void apply_tool_offset(FrameState & frame, const Eigen::Isometry3d & offset);
// Frame state of `frame` as `task` controls it: shifted to the offset point when `task` is a pose task
// with an active (enabled and fresh) offset, the plain frame state otherwise. Used where a frame of a
// planned/checked pose path must be compared with the task's own controlled point.
FrameState compute_tip_state(const KinematicsBackend & backend, const WholeBodyState & state,
  const std::string & frame, const TaskBase * task);
}  // namespace pose_math

// Epochs reject callbacks spanning a disable/reset. Expiry is opt-in.
class PoseTaskCommon : public TaskBaseCommon
{
  friend class StpikTaskAccess;
public:
  PoseTaskCommon() = default;
  PoseTaskCommon(const PoseTaskCommon & other);
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

  // Controlled-point offset: the task frame shifted by T_frame_offset (e.g. the tip of a zip tie held
  // by the tool). `use_offset` is explicit and changeable at runtime; it never turns on by itself
  // when data arrives. All of these are thread-safe.
  bool use_offset() const;
  // True when an offset has been received and is not older than offset_timeout.
  bool offset_ready() const;
  // Frame the received offset must be expressed in (the task frame), "" if unrestricted.
  std::string offset_frame() const;
  // Stores a received offset; false (and no change) if it is not a finite pose.
  bool set_offset(const Eigen::Isometry3d & offset);
  // Fails (no change) when the task has no offset support or no fresh offset to enable.
  bool set_use_offset(bool use, std::string & message);
  // The offset when use_offset is on and it is fresh; nothing otherwise.
  std::optional<Eigen::Isometry3d> active_offset() const;

protected:
  void configure_pose_feedforward(const std::map<std::string, rclcpp::Parameter> & parameters);
  // Reads use_offset / offset_timeout. `frame_id` is where offsets are expressed; `supported` is false
  // for tasks that cannot shift their point (e.g. the base frame).
  void configure_offset(const std::map<std::string, rclcpp::Parameter> & parameters,
    const std::string & frame_id, bool supported);
  // Shifts `frame` to the controlled point when use_offset is on. Returns false (frame untouched)
  // when it is on but no fresh offset is available; `message` then says so.
  bool apply_controlled_offset(FrameState & frame, std::string & message) const;
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

  bool offset_fresh_locked() const;
  mutable std::mutex offset_mutex_;
  bool offset_supported_ = false;
  bool use_offset_ = false;
  bool offset_valid_ = false;
  Eigen::Isometry3d offset_ = Eigen::Isometry3d::Identity();
  std::chrono::steady_clock::time_point offset_time_;
  double offset_timeout_ = 0.5;
  std::string offset_frame_;
  // Set when the controlled point changes: the target is recaptured from the new point, even with
  // auto_update_target_pose off, so the arm does not jump by the offset.
  bool force_capture_target_ = false;
};
}  // namespace task_priority_kinematic_control
