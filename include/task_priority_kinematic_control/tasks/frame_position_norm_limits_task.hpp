#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"
#include "task_priority_kinematic_control/core/snapshot_buffer.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <string>

namespace task_priority_kinematic_control
{

// Debug visualization sample of the allowed position region (world frame).
struct FramePositionNormSnapshot
{
  Eigen::Vector3d target_position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond target_orientation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d frame_position = Eigen::Vector3d::Zero();
  double norm = 0.0;        // sigma = ||S e||
  double limit = 0.0;       // radius
  double alpha = 0.0;       // activation margin
  double delta = 0.0;       // release margin
  std::array<bool, 3> axes{{true, true, true}};
  bool active = false;
  bool always_active = false;
  bool enabled = false;
  bool valid = false;
  bool visualization_enabled = false;
  uint32_t visualization_state = 0;
  int64_t timestamp_ns = 0;
  std::chrono::steady_clock::time_point captured{};
};

// Lock-free channel between the control loop and the marker observer. A copy
// (e.g. STPIK snapshots) keeps the debug flag but gets its own fresh buffer.
struct FramePositionNormDebugChannel
{
  FramePositionNormDebugChannel() = default;
  FramePositionNormDebugChannel(const FramePositionNormDebugChannel & other)
  : visibility(other.visibility.load(std::memory_order_relaxed)) {}
  FramePositionNormDebugChannel & operator=(const FramePositionNormDebugChannel & other)
  {
    visibility.store(other.visibility.load(std::memory_order_relaxed), std::memory_order_relaxed);
    return *this;
  }
  std::atomic<uint32_t> visibility{0};
  SnapshotBuffer<FramePositionNormSnapshot> buffer;
};

// One-row position-norm task for any frame (end effector, base or other).
// sigma = || S * R_t^T (p_target - p) ||, with S = diag(axes) in the target axes:
//   axes [x y z] -> sphere of radius `threshold`, [x y] -> cylinder (z free),
//   [z] -> slab |e_z| <= threshold.
// Set-based (default): the row only activates when sigma >= threshold - activation_margin,
// pushes sigma back to threshold - release_margin and releases with hysteresis.
// always_active: the row is always active and drives sigma to zero.
class FramePositionNormLimitsTask : public TaskBaseCommon
{
  friend class StpikTaskAccess;
public:
  void configure(
    const std::string & id, const std::string & plugin_name,
    const std::map<std::string, rclcpp::Parameter> & parameters,
    const TaskContext & context) override;

  TaskComputation update(
    const WholeBodyState & state, const KinematicsBackend & backend) override;
  void prepare_computation(TaskComputation & out) const override;
  void update_into(
    const WholeBodyState & state, const KinematicsBackend & backend,
    TaskComputation & out) override;

  bool set_pose_goal(const geometry_msgs::msg::PoseStamped & goal) override;
  bool set_gain(const std::vector<double> & gain, std::string & message) override;
  bool set_gain_scalar(double gain, std::string & message) override;
  bool set_enabled(bool enabled) override;
  void reset() override;
  msg::TaskStatus build_status() const override;
  std::vector<double> current_target() const override;

  bool row_active() const {return row_active_;}
  double norm() const {return last_norm_;}

  // Debug markers.
  void observe_command(const WholeBodyCommand &, int64_t timestamp_ns) override;
  uint32_t marker_visibility_state() const
  {return debug_channel_.visibility.load(std::memory_order_relaxed);}
  bool debug() const {return (marker_visibility_state() & 1U) != 0;}
  void set_debug(bool enabled);
  double publish_marker_rate() const {return publish_marker_rate_;}
  const std::string & world_frame() const {return context_.model->world_frame();}
  const FramePositionNormSnapshot * consume_snapshot() {return debug_channel_.buffer.consume();}

private:
  void record_status(TaskComputation & out, const char * message);

  std::string frame_id_;
  bool is_base_frame_ = false;
  std::array<bool, 3> axes_{{true, true, true}};
  Eigen::Index dofs_ = 0;
  double limit_ = 0.05;
  double alpha_ = 0.01;
  double delta_ = 0.02;
  double gain_ = 1.0;
  double eps_ = 1e-4;
  double max_velocity_ = 0.1;
  bool always_active_ = false;

  // Target in world coordinates; captured from the measured pose when unset.
  bool has_target_ = false;
  Eigen::Vector3d target_position_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond target_orientation_ = Eigen::Quaterniond::Identity();

  bool row_active_ = false;  // set-based hysteresis state

  FrameState frame_;
  Eigen::RowVectorXd row_;

  bool last_active_ = false;
  std::string last_status_ = "disabled";
  double last_norm_ = 0.0;
  double last_error_ = 0.0;
  double last_velocity_ = 0.0;

  double publish_marker_rate_ = 2.0;
  int64_t snapshot_period_ns_ = 500000000;
  int64_t last_snapshot_ns_ = 0;
  uint32_t last_snapshot_visibility_ = 0;
  bool has_snapshot_ = false;
  bool marker_valid_ = false;
  Eigen::Vector3d marker_frame_position_ = Eigen::Vector3d::Zero();
  FramePositionNormDebugChannel debug_channel_;
};

}  // namespace task_priority_kinematic_control
