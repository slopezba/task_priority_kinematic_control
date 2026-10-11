#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"
#include "task_priority_kinematic_control/core/snapshot_buffer.hpp"
#include <atomic>

namespace task_priority_kinematic_control
{

struct JointLimitPose
{
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();  // Joint axis expressed in the model base frame.
  bool active = false;
  bool valid = false;
};

struct JointLimitsSnapshot
{
  std::vector<JointLimitPose> joints;
  bool enabled = false;
  bool visualization_enabled = false;
  uint32_t visualization_state = 0;
  int64_t timestamp_ns = 0;
};

class JointLimitsTask : public TaskBaseCommon
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

  void prepare_computation(TaskComputation & out) const override;
  // The caller prepares out once; update_into never resizes it.
  void update_into(
    const WholeBodyState & state, const KinematicsBackend & backend, TaskComputation & out) override;
  void observe_command(const WholeBodyCommand &, int64_t timestamp_ns) override;

  uint32_t marker_visibility_state() const {return marker_visibility_.load(std::memory_order_relaxed);}
  bool publish_limits_marker() const {return (marker_visibility_state() & 1U) != 0;}
  void set_publish_limits_marker(bool enabled)
  {
    const uint32_t previous = marker_visibility_state();
    if (((previous & 1U) != 0) == enabled) {return;}
    // One serialized configuration writer. Epoch prevents redisplaying a pre-toggle snapshot.
    marker_visibility_.store(((previous + 2U) & ~1U) | (enabled ? 1U : 0U), std::memory_order_relaxed);
  }
  double limits_marker_publish_rate() const {return limits_marker_publish_rate_;}
  const std::vector<std::string> & joint_names() const {return joint_names_;}
  const std::string & base_frame() const {return context_.model->base_frame();}
  const JointLimitsSnapshot * consume_snapshot() {return snapshots_.consume();}

  bool set_gain_scalar(double gain, std::string & message) override;

private:
  void capture_markers(const WholeBodyState & state, const KinematicsBackend & backend);
  double alpha_ = 0.1;
  double delta_ = 0.15;
  double eps_ = 1e-4;
  double gain_ = 1.0;
  Eigen::VectorXd lower_limits_;
  Eigen::VectorXd upper_limits_;
  std::vector<bool> lower_active_;
  std::vector<bool> upper_active_;
  std::vector<std::string> joint_names_;
  std::vector<std::string> joint_frames_;
  Eigen::Index joint_count_ = 0;
  Eigen::Index dofs_ = 0;
  Eigen::Index base_dofs_ = 0;
  double limits_marker_publish_rate_ = 1.0;
  std::atomic<uint32_t> marker_visibility_{0};
  FrameState base_marker_frame_;
  FrameState joint_marker_frame_;
  SnapshotBuffer<JointLimitsSnapshot> snapshots_;
};

}  // namespace task_priority_kinematic_control
