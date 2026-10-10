#pragma once
#include "task_priority_kinematic_control/tasks/task_base.hpp"
#include "task_priority_kinematic_control/core/snapshot_buffer.hpp"
#include <array>
#include <atomic>
#include <limits>

namespace task_priority_kinematic_control
{
struct ManualCapsule
{
  std::string name;
  std::string start_frame;
  std::string end_frame;
  double radius = 0.0;
  size_t start_index = 0;
  size_t end_index = 0;
};
struct CapsulePose
{
  Eigen::Vector3d a = Eigen::Vector3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  bool highlighted = false;
  bool valid = false;
};
struct ManualPlane
{
  std::string name;
  std::string reference_frame;
  int normal_axis = 0;
  double allowed_sign = 1.0;
  double position = 0.0;
  Eigen::Vector2d bounds_min = Eigen::Vector2d::Zero();
  Eigen::Vector2d bounds_max = Eigen::Vector2d::Zero();
  size_t reference_index = 0;
};
struct PlanePose
{
  std::array<Eigen::Vector3d, 4> corners{
    Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
    Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  bool highlighted = false;
  bool valid = false;
};
struct CollisionMetrics
{
  size_t active_pairs = 0;
  size_t active_joint_count = 0;
  size_t zero_jacobian_pairs = 0;
  size_t degenerate_pairs = 0;
  size_t closest_pair = std::numeric_limits<size_t>::max();
  size_t fault_pair = std::numeric_limits<size_t>::max();
  double min_clearance = std::numeric_limits<double>::infinity();
  double max_velocity_deficit = 0.0;
  bool enabled = false;
  bool stop_arms = false;
  bool invalid_geometry = false;
};
struct CollisionSnapshot
{
  std::vector<CapsulePose> capsules;
  std::vector<PlanePose> planes;
  CollisionMetrics metrics;
  int64_t timestamp_ns = 0;
};

class SelfCollisionAvoidanceTask : public TaskBaseCommon
{
public:
  void configure(
    const std::string & id, const std::string & plugin_name,
    const std::map<std::string, rclcpp::Parameter> & parameters,
    const TaskContext & context) override;
  TaskComputation update(const WholeBodyState &, const KinematicsBackend &) override;
  void prepare_computation(TaskComputation & out) const override;
  void update_into(const WholeBodyState &, const KinematicsBackend &, TaskComputation &) override;
  void observe_command(const WholeBodyCommand & command, int64_t timestamp_ns) override;
  bool set_enabled(bool enabled) override;
  void reset() override;
  bool set_gain_scalar(double gain, std::string & message) override;
  msg::TaskStatus build_status() const override;
  const std::vector<ManualCapsule> & capsules() const {return capsules_;}
  const std::vector<std::array<size_t, 2>> & checked_pairs() const {return pairs_;}
  const std::vector<ManualPlane> & planes() const {return planes_;}
  // {capsule index, plane index}; rows follow the capsule/capsule rows.
  const std::vector<std::array<size_t, 2>> & checked_plane_pairs() const {return plane_pairs_;}
  std::string collision_pair_name(size_t row) const;
  size_t ignored_pair_count() const {return ignored_pair_count_;}
  const CollisionSnapshot * consume_snapshot() {return snapshots_.consume();}
  bool publish_capsules() const {return publish_capsules_.load(std::memory_order_relaxed);}
  void set_publish_capsules(bool enabled) {publish_capsules_.store(enabled, std::memory_order_relaxed);}
  double capsule_publish_rate() const {return capsule_publish_rate_;}
  double segment_length_epsilon() const {return segment_length_epsilon_;}
  const std::string & base_frame() const {return context_.model->base_frame();}
private:
  struct EndpointState
  {
    FrameState world;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::MatrixXd jacobian;
  };
  size_t add_frame(const std::string & frame);
  void configure_pair_columns(const KinematicsBackend & backend);
  bool update_pair_activation(size_t row, double clearance);
  void invalid(TaskComputation & out);
  std::vector<ManualCapsule> capsules_;
  std::vector<std::array<size_t, 2>> pairs_;
  std::vector<ManualPlane> planes_;
  std::vector<std::array<size_t, 2>> plane_pairs_;
  std::vector<std::string> frames_;
  std::vector<EndpointState> endpoints_;
  // Fixed candidate columns and persistent latches, in the same order as task rows.
  std::vector<std::vector<Eigen::Index>> pair_columns_;
  std::vector<unsigned char> pair_active_;
  std::vector<unsigned char> active_joints_;
  size_t ignored_pair_count_ = 0;
  double safe_distance_ = 0.05;
  double activation_distance_ = 0.12;
  double release_distance_ = 0.13;
  double eps_ = 0.0001;
  double gain_ = 1.0;
  double max_repulsive_velocity_ = 0.08;
  double parallel_epsilon_ = 1e-8;
  double segment_length_epsilon_ = 1e-9;
  double degenerate_axis_distance_ = 1e-6;
  double capsule_publish_rate_ = 1.0;
  std::atomic_bool publish_capsules_{true};
  Eigen::MatrixXd jacobian_;
  Eigen::MatrixXd point_difference_;
  Eigen::VectorXd desired_;
  Eigen::VectorXd errors_;
  CollisionMetrics last_;
  SnapshotBuffer<CollisionSnapshot> snapshots_;
};
}  // namespace task_priority_kinematic_control
