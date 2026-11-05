#pragma once
#include <Eigen/Core>
#include <array>

namespace task_priority_kinematic_control
{
struct PlaneDistance
{
  double s = 0.0;
  double clearance = 0.0;
  Eigen::Vector3d gradient_a = Eigen::Vector3d::Zero();
  Eigen::Vector3d gradient_b = Eigen::Vector3d::Zero();
  bool valid = false;
  bool intersects = false;
};

// Tangential axes in ascending coordinate order: X -> YZ, Y -> XZ, Z -> XY.
std::array<int, 2> plane_tangent_axes(int normal_axis);

// One-sided normal clearance over the segment clipped to the rectangle expanded
// by the capsule radius. Conservative at edges/corners; not Euclidean distance.
// Gradients include the selected clipping boundary's derivative. Ties are
// deterministic; switching boundaries and entering/leaving the rectangle are nonsmooth.
PlaneDistance capsule_plane_distance(
  const Eigen::Vector3d & a, const Eigen::Vector3d & b, double radius,
  int normal_axis, double allowed_sign, double position,
  const Eigen::Vector2d & bounds_min, const Eigen::Vector2d & bounds_max);
}  // namespace task_priority_kinematic_control
