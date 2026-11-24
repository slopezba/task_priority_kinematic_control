#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"
#include "task_priority_kinematic_control/core/snapshot_buffer.hpp"

#include <atomic>
#include <chrono>
#include <string>
#include <vector>

namespace task_priority_kinematic_control
{

// Debug visualization sample: position manipulability ellipsoid at the tip (world frame).
struct ArmManipulabilitySnapshot
{
  Eigen::Vector3d tip_position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();  // principal axes of Jp Jp^T
  Eigen::Vector3d singular_values = Eigen::Vector3d::Zero();         // along local x, y, z
  double w = 0.0;
  double w_min = 0.0;
  double w_des = 0.0;
  double ellipsoid_scale = 1.0;
  bool active = false;
  bool enabled = false;
  bool valid = false;
  bool visualization_enabled = false;
  uint32_t visualization_state = 0;
  int64_t timestamp_ns = 0;
  std::chrono::steady_clock::time_point captured{};
};

// Lock-free channel between the control loop and the marker observer. A copy
// (e.g. STPIK snapshots) keeps the debug flag but gets its own fresh buffer.
struct ArmManipulabilityDebugChannel
{
  ArmManipulabilityDebugChannel() = default;
  ArmManipulabilityDebugChannel(const ArmManipulabilityDebugChannel & other)
  : visibility(other.visibility.load(std::memory_order_relaxed)) {}
  ArmManipulabilityDebugChannel & operator=(const ArmManipulabilityDebugChannel & other)
  {
    visibility.store(other.visibility.load(std::memory_order_relaxed), std::memory_order_relaxed);
    return *this;
  }
  std::atomic<uint32_t> visibility{0};
  SnapshotBuffer<ArmManipulabilitySnapshot> buffer;
};

// One-row, set-based task that raises the position manipulability of one arm:
//   w(q_a) = sqrt(det(Jp Jp^T)), Jp = linear rows of the tip Jacobian on the arm columns.
// The row is the normalized gradient n = grad(w)/||grad(w)|| (arm columns only, finite
// differences through KinematicsBackend::compute_frame_state). It activates when w < w_min,
// releases when w >= w_des and then requests min(k (w_des - w)/w_des, max_velocity) along n.
class ArmManipulabilityTask : public TaskBaseCommon
{
  friend class StpikTaskAccess;

public:
  void configure(
    const std::string & id,
    const std::string & plugin_name,
    const std::map<std::string, rclcpp::Parameter> & parameters,
    const TaskContext & context) override;

  TaskComputation update(
    const WholeBodyState & state,
    const KinematicsBackend & backend) override;
  void prepare_computation(TaskComputation & out) const override;
  void update_into(
    const WholeBodyState & state, const KinematicsBackend & backend, TaskComputation & out) override;

  bool set_gain(const std::vector<double> & gain, std::string & message) override;
  bool set_gain_scalar(double gain, std::string & message) override;
  void reset() override;
  msg::TaskStatus build_status() const override;

  // Position manipulability of the arm columns of a full 6xN Jacobian.
  double manipulability(const Eigen::MatrixXd & jacobian) const;
  double last_manipulability() const {return last_w_;}

  // Debug markers (manipulability ellipsoid coloured by state).
  void observe_command(const WholeBodyCommand &, int64_t timestamp_ns) override;
  uint32_t marker_visibility_state() const
  {return debug_channel_.visibility.load(std::memory_order_relaxed);}
  bool debug() const {return (marker_visibility_state() & 1U) != 0;}
  void set_debug(bool enabled);
  double publish_marker_rate() const {return publish_marker_rate_;}
  const std::string & world_frame() const {return context_.model->world_frame();}
  const ArmManipulabilitySnapshot * consume_snapshot() {return debug_channel_.buffer.consume();}

private:
  void record_status(TaskComputation & out, const char * message);
  bool compute_gradient(const WholeBodyState & state, const KinematicsBackend & backend);
  void capture_marker_geometry();

  std::string frame_id_;
  std::vector<std::string> joint_names_;
  std::vector<Eigen::Index> joint_indices_;  // into WholeBodyState::joint_positions
  std::vector<Eigen::Index> columns_;        // into the generalized velocity
  Eigen::Index dofs_ = 0;

  double w_min_ = 0.0025;
  double w_des_ = 0.0045;
  double gain_ = 1.0;
  double max_velocity_ = 0.3;
  double fd_step_ = 1e-4;
  double gradient_epsilon_ = 1e-9;

  bool row_active_ = false;
  bool last_active_ = false;
  double last_w_ = 0.0;
  double last_velocity_ = 0.0;
  std::string last_status_ = "idle";

  FrameState frame_;
  WholeBodyState perturbed_;
  Eigen::VectorXd gradient_;

  double publish_marker_rate_ = 2.0;
  double ellipsoid_scale_ = 0.5;
  int64_t snapshot_period_ns_ = 500000000;
  int64_t last_snapshot_ns_ = 0;
  uint32_t last_snapshot_visibility_ = 0;
  bool has_snapshot_ = false;
  bool marker_valid_ = false;
  Eigen::Vector3d marker_tip_position_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond marker_orientation_ = Eigen::Quaterniond::Identity();
  Eigen::Vector3d marker_singular_values_ = Eigen::Vector3d::Zero();
  ArmManipulabilityDebugChannel debug_channel_;
};

}  // namespace task_priority_kinematic_control
