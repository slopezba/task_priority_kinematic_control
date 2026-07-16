#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"

#include <limits>
#include <set>

namespace task_priority_kinematic_control
{

class SelfCollisionAvoidanceTask : public TaskBaseCommon
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

  msg::TaskStatus build_status() const override;

private:
  struct WorldCapsule
  {
    CollisionCapsule capsule;
    FrameState frame;
    Eigen::Vector3d a = Eigen::Vector3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
  };

  bool included_link(const std::string & link_name) const;
  bool ignored_pair(const CollisionCapsule & a, const CollisionCapsule & b) const;
  Eigen::MatrixXd point_jacobian(
    const FrameState & frame,
    const Eigen::Vector3d & point_world) const;

  std::vector<std::string> include_link_substrings_;
  std::vector<std::string> exclude_link_substrings_;
  double safe_distance_ = 0.05;
  double activation_distance_ = 0.12;
  double gain_ = 1.0;
  double max_repulsive_velocity_ = 0.08;
  bool check_adjacent_links_ = false;
  mutable double last_min_distance_ = std::numeric_limits<double>::infinity();
  mutable size_t last_active_pairs_ = 0;
};

}  // namespace task_priority_kinematic_control
