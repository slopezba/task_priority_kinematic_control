#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"
#include "task_priority_kinematic_control/core/snapshot_buffer.hpp"

#include <array>
#include <atomic>
#include <chrono>

namespace task_priority_kinematic_control
{

struct StabilizeLimitsSnapshot
{
  Eigen::Vector2d angles = Eigen::Vector2d::Zero();
  Eigen::Vector2d lower = Eigen::Vector2d::Zero();
  Eigen::Vector2d upper = Eigen::Vector2d::Zero();
  std::array<bool, 2> active{{false, false}};
  double alpha = 0;
  double delta = 0;
  bool enabled = false;
  bool valid = false;
  bool visualization_enabled = false;
  uint32_t visualization_state = 0;
  int64_t timestamp_ns = 0;
  std::chrono::steady_clock::time_point captured{};
};

// Set-based bounds on base Euler roll/pitch; no pose target or feedforward.
class StabilizeLimitsTask : public TaskBaseCommon
{
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

  bool set_enabled(bool enabled) override;
  void reset() override;
  bool set_pose_goal(const geometry_msgs::msg::PoseStamped &) override {return false;}
  bool set_gain_scalar(double gain, std::string & message) override;
  msg::TaskStatus build_status() const override;
  void observe_command(const WholeBodyCommand &, int64_t timestamp_ns) override;
  uint32_t marker_visibility_state() const {return marker_visibility_.load(std::memory_order_relaxed);}
  bool debug() const {return (marker_visibility_state() & 1U) != 0;}
  void set_debug(bool enabled);
  double publish_marker_rate() const {return publish_marker_rate_;}
  const std::string & base_frame() const {return context_.model->base_frame();}
  const StabilizeLimitsSnapshot * consume_snapshot() {return snapshots_.consume();}

private:
  void record_status(TaskComputation & out, const char * message, const WholeBodyState & state);

  Eigen::Vector2d lower_limits_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d upper_limits_ = Eigen::Vector2d::Zero();
  std::array<bool, 2> lower_active_{{false, false}};
  std::array<bool, 2> upper_active_{{false, false}};
  std::array<Eigen::Index, 2> angular_columns_{{-1, -1}};
  Eigen::Index dofs_ = 0;
  double alpha_ = 0.06981317;
  double delta_ = 0.13962634;
  double eps_ = 0.0001;
  double gain_ = 2.0;
  double max_repulsive_velocity_ = 0.3;
  bool last_active_ = false;
  std::string last_status_ = "disabled";
  Eigen::Vector2d last_error_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d last_velocity_ = Eigen::Vector2d::Zero();
  double publish_marker_rate_ = 2.0;
  int64_t snapshot_period_ns_ = 500000000;
  int64_t last_snapshot_ns_ = 0;
  uint32_t last_snapshot_visibility_ = 0;
  bool has_snapshot_ = false;
  bool marker_valid_ = false;
  Eigen::Vector2d marker_angles_ = Eigen::Vector2d::Zero();
  std::array<bool, 2> marker_active_{{false, false}};
  std::atomic<uint32_t> marker_visibility_{0};
  SnapshotBuffer<StabilizeLimitsSnapshot> snapshots_;
};

}  // namespace task_priority_kinematic_control
