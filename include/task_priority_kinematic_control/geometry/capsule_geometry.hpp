#pragma once
#include <Eigen/Core>

namespace task_priority_kinematic_control
{
struct CapsuleDistance
{
  double s = 0.0;
  double t = 0.0;
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  Eigen::Vector3d q = Eigen::Vector3d::Zero();
  double axis_distance = 0.0;
  double clearance = 0.0;
  bool valid = false;
};

// Scale-independent parallel tolerance; segment_length_epsilon is in metres.
// Ties use the first candidate, giving a deterministic (not necessarily smooth) minimum.
CapsuleDistance capsule_distance(
  const Eigen::Vector3d & a, const Eigen::Vector3d & b,
  const Eigen::Vector3d & c, const Eigen::Vector3d & d,
  double radius_a, double radius_b,
  double parallel_epsilon = 1e-8, double segment_length_epsilon = 1e-9);
}  // namespace task_priority_kinematic_control
