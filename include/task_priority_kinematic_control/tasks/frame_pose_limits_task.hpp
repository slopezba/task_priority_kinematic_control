#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"
#include "task_priority_kinematic_control/core/snapshot_buffer.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>

namespace task_priority_kinematic_control
{

// Debug visualization sample of the allowed region (world frame).
struct FramePoseLimitsSnapshot
{
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  Eigen::Vector3d target_position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond target_orientation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d frame_position = Eigen::Vector3d::Zero();
  Vector6d error = Vector6d::Zero();  // target - current, target-frame axes
  Vector6d threshold = Vector6d::Zero();
  Vector6d alpha = Vector6d::Zero();
  Vector6d delta = Vector6d::Zero();
  std::array<bool, 6> axes{{false, false, false, false, false, false}};
  std::array<bool, 6> active{{false, false, false, false, false, false}};
  bool enabled = false;
  bool valid = false;
  bool visualization_enabled = false;
  uint32_t visualization_state = 0;
  int64_t timestamp_ns = 0;
  std::chrono::steady_clock::time_point captured{};
};

// Lock-free channel between the control loop and the marker observer. A copy
// (e.g. STPIK snapshots) keeps the debug flag but gets its own fresh buffer.
struct FramePoseLimitsDebugChannel
{
  FramePoseLimitsDebugChannel() = default;
  FramePoseLimitsDebugChannel(const FramePoseLimitsDebugChannel & other)
  : visibility(other.visibility.load(std::memory_order_relaxed)) {}
  FramePoseLimitsDebugChannel & operator=(const FramePoseLimitsDebugChannel & other)
  {
    visibility.store(other.visibility.load(std::memory_order_relaxed), std::memory_order_relaxed);
    return *this;
  }
  std::atomic<uint32_t> visibility{0};
  SnapshotBuffer<FramePoseLimitsSnapshot> buffer;
};

// Set-based (inequality) pose task. The frame may move freely inside a box of
// +/- threshold around a target pose, expressed in the target frame axes
// (x, y, z, rx, ry, rz). An axis only takes degrees of freedom when its error
// reaches threshold - activation_margin; it then pushes the error back to
// threshold - release_margin and releases with hysteresis, like JointLimitsTask.
// Inactive axes keep structural zero rows so lower priorities can use them.
class FramePoseLimitsTask : public TaskBaseCommon
{
  friend class StpikTaskAccess;
public:
  using Vector6d = Eigen::Matrix<double, 6, 1>;

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

  const std::array<bool, 6> & axes() const {return axes_;}
  const std::array<bool, 6> & axis_active() const {return axis_active_;}
  Eigen::Index rows() const {return rows_;}

  // Debug markers: allowed box (position) and rings (orientation).
  void observe_command(const WholeBodyCommand &, int64_t timestamp_ns) override;
  uint32_t marker_visibility_state() const
  {return debug_channel_.visibility.load(std::memory_order_relaxed);}
  bool debug() const {return (marker_visibility_state() & 1U) != 0;}
  void set_debug(bool enabled);
  double publish_marker_rate() const {return publish_marker_rate_;}
  const std::string & world_frame() const {return context_.model->world_frame();}
  const FramePoseLimitsSnapshot * consume_snapshot() {return debug_channel_.buffer.consume();}

private:
  void clear_hysteresis();
  void record_status(TaskComputation & out, const char * message);

  std::string frame_id_;
  bool is_base_frame_ = false;
  std::array<bool, 6> axes_{{true, true, true, true, true, true}};
  std::array<Eigen::Index, 6> axis_row_{{0, 1, 2, 3, 4, 5}};
  Eigen::Index rows_ = 6;
  Eigen::Index dofs_ = 0;
  Vector6d threshold_ = Vector6d::Constant(0.05);
  Vector6d alpha_ = Vector6d::Constant(0.01);
  Vector6d delta_ = Vector6d::Constant(0.02);
  Vector6d gains_ = Vector6d::Ones();
  double eps_ = 1e-4;
  double max_repulsive_velocity_ = 0.1;

  // Target in world coordinates; captured from the measured pose when unset.
  bool has_target_ = false;
  Eigen::Vector3d target_position_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond target_orientation_ = Eigen::Quaterniond::Identity();

  // Hysteresis state per axis: active flag and the side (+1/-1) being held.
  std::array<bool, 6> axis_active_{{false, false, false, false, false, false}};
  std::array<double, 6> axis_side_{{0, 0, 0, 0, 0, 0}};

  // Preallocated work buffers for the allocation-free update path.
  FrameState frame_;
  Eigen::MatrixXd local_jacobian_;

  bool last_active_ = false;
  std::string last_status_ = "disabled";
  Vector6d last_error_ = Vector6d::Zero();
  Vector6d last_velocity_ = Vector6d::Zero();

  // Debug visualization state (written by the control loop only).
  double publish_marker_rate_ = 2.0;
  int64_t snapshot_period_ns_ = 500000000;
  int64_t last_snapshot_ns_ = 0;
  uint32_t last_snapshot_visibility_ = 0;
  bool has_snapshot_ = false;
  bool marker_valid_ = false;
  Eigen::Vector3d marker_frame_position_ = Eigen::Vector3d::Zero();
  FramePoseLimitsDebugChannel debug_channel_;
};

}  // namespace task_priority_kinematic_control
