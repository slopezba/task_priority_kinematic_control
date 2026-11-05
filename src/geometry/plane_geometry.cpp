#include "task_priority_kinematic_control/geometry/plane_geometry.hpp"
#include <cmath>
#include <limits>
#include <utility>

namespace task_priority_kinematic_control
{
std::array<int, 2> plane_tangent_axes(int normal_axis)
{
  if (normal_axis == 0) {return {1, 2};}
  if (normal_axis == 1) {return {0, 2};}
  return {0, 1};
}

PlaneDistance capsule_plane_distance(
  const Eigen::Vector3d & a, const Eigen::Vector3d & b, double radius,
  int normal_axis, double allowed_sign, double position,
  const Eigen::Vector2d & bounds_min, const Eigen::Vector2d & bounds_max)
{
  PlaneDistance out;
  if (!a.allFinite() || !b.allFinite() || !bounds_min.allFinite() ||
    !bounds_max.allFinite() || !std::isfinite(radius) || radius <= 0 ||
    !std::isfinite(position) || normal_axis < 0 || normal_axis > 2 ||
    (allowed_sign != 1.0 && allowed_sign != -1.0) ||
    (bounds_min.array() >= bounds_max.array()).any()) {return out;}

  // Extended precision avoids cancellation/overflow in clipping very short or
  // long segments. A parallel coordinate is handled without division.
  const auto axes = plane_tangent_axes(normal_axis);
  long double lower = 0, upper = 1;
  int lower_axis = -1, upper_axis = -1;
  for (size_t i = 0; i < axes.size(); ++i) {
    const int axis = axes[i];
    const long double start = a(axis);
    const long double delta = static_cast<long double>(b(axis)) - start;
    const long double minimum = static_cast<long double>(bounds_min(i)) - radius;
    const long double maximum = static_cast<long double>(bounds_max(i)) + radius;
    if (delta == 0) {
      if (start < minimum || start > maximum) {out.valid = true; return out;}
      continue;
    }
    long double entry = (minimum - start) / delta;
    long double exit = (maximum - start) / delta;
    if (entry > exit) {std::swap(entry, exit);}
    if (entry > lower) {lower = entry; lower_axis = axis;}
    if (exit < upper) {upper = exit; upper_axis = axis;}
    if (lower > upper) {out.valid = true; return out;}
  }
  const long double normal_delta = static_cast<long double>(b(normal_axis)) - a(normal_axis);
  const bool use_lower = allowed_sign * normal_delta >= 0;
  const long double s = use_lower ? lower : upper;
  const int boundary_axis = use_lower ? lower_axis : upper_axis;
  const long double clearance = allowed_sign *
    ((1 - s) * a(normal_axis) + s * b(normal_axis) - position) - radius;
  const long double limit = std::numeric_limits<double>::max();
  if (!std::isfinite(clearance) || std::abs(clearance) > limit) {return out;}
  out.s = static_cast<double>(s);
  out.clearance = static_cast<double>(clearance);
  out.gradient_a(normal_axis) = static_cast<double>(allowed_sign * (1 - s));
  out.gradient_b(normal_axis) = static_cast<double>(allowed_sign * s);
  if (boundary_axis >= 0) {
    const long double tangent_delta = static_cast<long double>(b(boundary_axis)) - a(boundary_axis);
    // ds = -((1-s) da_boundary + s db_boundary) / tangent_delta.
    const long double factor = -allowed_sign * normal_delta / tangent_delta;
    const long double gradient_a = factor * (1 - s), gradient_b = factor * s;
    if (!std::isfinite(gradient_a) || !std::isfinite(gradient_b) ||
      std::abs(gradient_a) > limit || std::abs(gradient_b) > limit) {return out;}
    out.gradient_a(boundary_axis) = static_cast<double>(gradient_a);
    out.gradient_b(boundary_axis) = static_cast<double>(gradient_b);
  }
  out.valid = std::isfinite(out.s) && std::isfinite(out.clearance) &&
    out.gradient_a.allFinite() && out.gradient_b.allFinite();
  out.intersects = out.valid;
  return out;
}
}  // namespace task_priority_kinematic_control
