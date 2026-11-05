#include "task_priority_kinematic_control/geometry/capsule_geometry.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>

namespace task_priority_kinematic_control
{
CapsuleDistance capsule_distance(
  const Eigen::Vector3d & a0, const Eigen::Vector3d & b0,
  const Eigen::Vector3d & c0, const Eigen::Vector3d & d0,
  double ra, double rb, double parallel_epsilon, double length_epsilon)
{
  CapsuleDistance out;
  if (!a0.allFinite() || !b0.allFinite() || !c0.allFinite() || !d0.allFinite() ||
    !std::isfinite(ra) || !std::isfinite(rb) || ra < 0 || rb < 0 ||
    !std::isfinite(parallel_epsilon) || parallel_epsilon <= 0 || parallel_epsilon >= 1 ||
    !std::isfinite(length_epsilon) || length_epsilon <= 0)
  {
    return out;
  }
  // Extended precision avoids length-dependent overflow/cancellation in the dot products.
  using Vector = Eigen::Matrix<long double, 3, 1>;
  const Vector a = a0.cast<long double>();
  const Vector c = c0.cast<long double>();
  const Vector u = b0.cast<long double>() - a;
  const Vector v = d0.cast<long double>() - c;
  const Vector w = a - c;
  const long double aa = u.dot(u), bb = u.dot(v), cc = v.dot(v);
  const long double dd = u.dot(w), ee = v.dot(w);
  const long double eps2 = static_cast<long double>(length_epsilon) * length_epsilon;
  auto clamp = [](long double x) {return std::clamp(x, 0.0L, 1.0L);};
  long double best = std::numeric_limits<long double>::infinity();
  long double best_s = 0, best_t = 0;
  auto consider = [&](long double s, long double t) {
      const long double squared = (w + s * u - t * v).squaredNorm();
      if (squared < best) {best = squared; best_s = s; best_t = t;}
    };
  if (aa <= eps2 && cc <= eps2) {
    consider(0, 0);
  } else if (aa <= eps2) {
    consider(0, clamp(ee / cc));
  } else if (cc <= eps2) {
    consider(clamp(-dd / aa), 0);
  } else {
    // The cross product evaluates ac-b² more accurately for nearly parallel axes.
    const long double denominator = u.cross(v).squaredNorm();
    const long double parallel_measure = denominator / (aa * cc);
    if (parallel_measure > parallel_epsilon) {
      const long double s = (bb * ee - cc * dd) / denominator;
      const long double t = (aa * ee - bb * dd) / denominator;
      if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {consider(s, t);}
    }
    consider(0, clamp(ee / cc));
    consider(1, clamp((ee + bb) / cc));
    consider(clamp(-dd / aa), 0);
    consider(clamp((bb - dd) / aa), 1);
  }
  out.s = static_cast<double>(best_s);
  out.t = static_cast<double>(best_t);
  out.p = (a + best_s * u).cast<double>();
  out.q = (c + best_t * v).cast<double>();
  const long double distance = std::sqrt(std::max(0.0L, best));
  // Saturate only unrepresentable extreme distances; ordinary robot geometry is unaffected.
  const long double max_double = std::numeric_limits<double>::max();
  out.axis_distance = static_cast<double>(std::min(distance, max_double));
  out.clearance = static_cast<double>(std::clamp(distance - ra - rb, -max_double, max_double));
  out.valid = out.p.allFinite() && out.q.allFinite() && std::isfinite(out.axis_distance) &&
    std::isfinite(out.clearance);
  return out;
}
}  // namespace task_priority_kinematic_control
