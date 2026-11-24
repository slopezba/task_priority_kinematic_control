#include "task_priority_kinematic_control/core/capsule_observer.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>

namespace task_priority_kinematic_control
{
namespace
{
using Marker = visualization_msgs::msg::Marker;

constexpr int kConeSegments = 32;
constexpr int kArcPoints = 33;
constexpr double kConeLength = 0.12;   // m, cone and current-axis arrow
constexpr double kGaugeRadius = 0.09;  // m, angle gauge
constexpr double kYawRadius = 0.08;    // m, yaw ring

std_msgs::msg::ColorRGBA rgba(float r, float g, float b, float a)
{
  std_msgs::msg::ColorRGBA c;
  c.r = r; c.g = g; c.b = b; c.a = a;
  return c;
}

geometry_msgs::msg::Point to_point(const Eigen::Vector3d & p)
{
  geometry_msgs::msg::Point point;
  point.x = p.x(); point.y = p.y(); point.z = p.z();
  return point;
}

std_msgs::msg::ColorRGBA state_color(const FrameOrientationNormSnapshot & s, float alpha)
{
  if (s.active) {return rgba(1.0f, 0.0f, 0.0f, alpha);}
  if (s.norm >= s.limit - s.delta) {return rgba(1.0f, 0.55f, 0.0f, alpha);}
  return rgba(0.0f, 1.0f, 0.0f, alpha);
}

Marker base_marker(
  const std::string & frame, const std::string & ns, int id, int type,
  const builtin_interfaces::msg::Time & stamp, double lifetime)
{
  Marker m;
  m.header.frame_id = frame;
  m.header.stamp = stamp;
  m.ns = ns;
  m.id = id;
  m.type = type;
  m.action = Marker::DELETE;
  m.pose.orientation.w = 1.0;
  m.scale.x = m.scale.y = m.scale.z = 1.0;
  m.lifetime.sec = static_cast<int32_t>(lifetime);
  m.lifetime.nanosec = static_cast<uint32_t>((lifetime - m.lifetime.sec) * 1e9);
  return m;
}

// Point on an arc of `radius` around `centre` in the plane (u, v); angle 0 is along u.
geometry_msgs::msg::Point arc_point(
  const Eigen::Vector3d & centre, const Eigen::Vector3d & u, const Eigen::Vector3d & v,
  double angle, double radius)
{
  return to_point(centre + radius * (std::cos(angle) * u + std::sin(angle) * v));
}

void fill_arc(Marker & m, const Eigen::Vector3d & centre, const Eigen::Vector3d & u,
  const Eigen::Vector3d & v, double start, double end, double radius)
{
  m.points.resize(kArcPoints);
  for (int k = 0; k < kArcPoints; ++k) {
    const double t = static_cast<double>(k) / (kArcPoints - 1);
    m.points[static_cast<size_t>(k)] = arc_point(centre, u, v, start + (end - start) * t, radius);
  }
}

// Unit vector at polar angle `half_angle` from `axis`, azimuth `phi` around it.
Eigen::Vector3d cone_direction(const Eigen::Vector3d & axis, const Eigen::Vector3d & p,
  const Eigen::Vector3d & q, double half_angle, double phi)
{
  return std::cos(half_angle) * axis +
         std::sin(half_angle) * (std::cos(phi) * p + std::sin(phi) * q);
}

}  // namespace

// Marker ids per namespace:
//   cone/0 translucent cone (half angle = threshold) around the target z axis,
//   cone/1 activation circle (threshold - activation_margin), cone/2 current z axis arrow
//     (only when rx or ry are part of the norm);
//   yaw/0..3 yaw ring (only for axes [rz]): allowed arc, limit ticks, activation ticks, arrow;
//   gauge/0..3 angle gauge in the plane normal to the error axis: allowed arc 0..threshold,
//     limit tick, activation tick, arrow at the current angle.
visualization_msgs::msg::MarkerArray frame_orientation_norm_markers(
  const FrameOrientationNormSnapshot & s, const std::string & frame,
  const std::string & task_id, double publish_rate)
{
  const double lifetime = std::min(2147483646.0,
      std::max(2.0, 3.0 / std::max(publish_rate, 1e-9)));
  const auto stamp = static_cast<builtin_interfaces::msg::Time>(rclcpp::Time(s.timestamp_ns));
  const std::string prefix = task_id + "/limits/";

  Marker cone[3] = {
    base_marker(frame, prefix + "cone", 0, Marker::TRIANGLE_LIST, stamp, lifetime),
    base_marker(frame, prefix + "cone", 1, Marker::LINE_STRIP, stamp, lifetime),
    base_marker(frame, prefix + "cone", 2, Marker::ARROW, stamp, lifetime)};
  Marker yaw[4] = {
    base_marker(frame, prefix + "yaw", 0, Marker::LINE_STRIP, stamp, lifetime),
    base_marker(frame, prefix + "yaw", 1, Marker::LINE_LIST, stamp, lifetime),
    base_marker(frame, prefix + "yaw", 2, Marker::LINE_LIST, stamp, lifetime),
    base_marker(frame, prefix + "yaw", 3, Marker::ARROW, stamp, lifetime)};
  Marker gauge[4] = {
    base_marker(frame, prefix + "gauge", 0, Marker::LINE_STRIP, stamp, lifetime),
    base_marker(frame, prefix + "gauge", 1, Marker::LINE_LIST, stamp, lifetime),
    base_marker(frame, prefix + "gauge", 2, Marker::LINE_LIST, stamp, lifetime),
    base_marker(frame, prefix + "gauge", 3, Marker::ARROW, stamp, lifetime)};

  const bool usable = s.visualization_enabled && s.valid && s.enabled &&
    s.target_orientation.coeffs().allFinite() && s.frame_orientation.coeffs().allFinite() &&
    s.frame_position.allFinite() && s.error.allFinite() && std::isfinite(s.norm) &&
    std::isfinite(s.limit) && std::isfinite(s.alpha) && s.limit > 0 && s.limit < M_PI;

  if (usable) {
    const Eigen::Matrix3d rt = s.target_orientation.normalized().toRotationMatrix();
    const Eigen::Matrix3d rc = s.frame_orientation.normalized().toRotationMatrix();
    const Eigen::Vector3d & c = s.frame_position;
    const double limit = s.limit;
    const double inner = std::max(0.0, s.limit - s.alpha);
    const auto color = state_color(s, 1.0f);

    if (s.axes[0] || s.axes[1]) {
      // Tilt cone around the target z axis (exact for axes [rx ry]).
      const Eigen::Vector3d axis = rt.col(2);
      const Eigen::Vector3d p = rt.col(0);
      const Eigen::Vector3d q = rt.col(1);
      auto & body = cone[0];
      body.action = Marker::ADD;
      body.color = state_color(s, 0.18f);
      body.points.reserve(3 * kConeSegments);
      for (int k = 0; k < kConeSegments; ++k) {
        const double phi0 = 2.0 * M_PI * k / kConeSegments;
        const double phi1 = 2.0 * M_PI * (k + 1) / kConeSegments;
        body.points.push_back(to_point(c));
        body.points.push_back(to_point(c + kConeLength * cone_direction(axis, p, q, limit, phi0)));
        body.points.push_back(to_point(c + kConeLength * cone_direction(axis, p, q, limit, phi1)));
      }
      auto & ring = cone[1];
      ring.action = Marker::ADD;
      ring.scale.x = 0.003;
      ring.color = rgba(1.0f, 0.55f, 0.0f, 0.9f);
      ring.points.resize(kConeSegments + 1);
      for (int k = 0; k <= kConeSegments; ++k) {
        const double phi = 2.0 * M_PI * k / kConeSegments;
        ring.points[static_cast<size_t>(k)] =
          to_point(c + kConeLength * cone_direction(axis, p, q, inner, phi));
      }
      auto & arrow = cone[2];
      arrow.action = Marker::ADD;
      arrow.scale.x = 0.006; arrow.scale.y = 0.014; arrow.scale.z = 0.02;
      arrow.color = color;
      arrow.points = {to_point(c), to_point(c + kConeLength * rc.col(2))};
    } else {
      // Yaw only: ring in the target xy plane, like the frame_pose_limits rings.
      const Eigen::Vector3d u = rt.col(0);
      const Eigen::Vector3d v = rt.col(1);
      yaw[0].action = Marker::ADD;
      yaw[0].scale.x = 0.006;
      yaw[0].color = color;
      yaw[0].color.a = 0.7f;
      fill_arc(yaw[0], c, u, v, -limit, limit, kYawRadius);
      for (int id = 1; id <= 2; ++id) {
        const double a = id == 1 ? limit : inner;
        yaw[id].action = Marker::ADD;
        yaw[id].scale.x = 0.004;
        yaw[id].color = id == 1 ? rgba(1.0f, 0.0f, 0.0f, 1.0f) : rgba(1.0f, 0.55f, 0.0f, 1.0f);
        yaw[id].points = {
          arc_point(c, u, v, -a, kYawRadius - 0.012), arc_point(c, u, v, -a, kYawRadius + 0.012),
          arc_point(c, u, v, a, kYawRadius - 0.012), arc_point(c, u, v, a, kYawRadius + 0.012)};
      }
      // e = target - current, so the current yaw offset is -e_z.
      yaw[3].action = Marker::ADD;
      yaw[3].scale.x = 0.006; yaw[3].scale.y = 0.014; yaw[3].scale.z = 0.02;
      yaw[3].color = color;
      yaw[3].points = {to_point(c), arc_point(c, u, v, -s.error.z(), kYawRadius - 0.02)};
    }

    // Angle gauge: 0..threshold in the plane normal to the current error axis.
    if (s.norm > 1e-6) {
      const Eigen::Vector3d axis = (rt * s.error).normalized();
      const Eigen::Vector3d u = axis.unitOrthogonal();
      const Eigen::Vector3d v = axis.cross(u);
      gauge[0].action = Marker::ADD;
      gauge[0].scale.x = 0.005;
      gauge[0].color = color;
      gauge[0].color.a = 0.6f;
      fill_arc(gauge[0], c, u, v, 0.0, limit, kGaugeRadius);
      for (int id = 1; id <= 2; ++id) {
        const double a = id == 1 ? limit : inner;
        gauge[id].action = Marker::ADD;
        gauge[id].scale.x = 0.004;
        gauge[id].color = id == 1 ? rgba(1.0f, 0.0f, 0.0f, 1.0f) : rgba(1.0f, 0.55f, 0.0f, 1.0f);
        gauge[id].points = {arc_point(c, u, v, a, kGaugeRadius - 0.012),
          arc_point(c, u, v, a, kGaugeRadius + 0.012)};
      }
      gauge[3].action = Marker::ADD;
      gauge[3].scale.x = 0.005; gauge[3].scale.y = 0.012; gauge[3].scale.z = 0.018;
      gauge[3].color = color;
      gauge[3].points = {to_point(c), arc_point(c, u, v, std::min(s.norm, M_PI), kGaugeRadius - 0.02)};
    }
  }

  visualization_msgs::msg::MarkerArray array;
  array.markers.reserve(11);
  for (auto & m : cone) {array.markers.push_back(std::move(m));}
  for (auto & m : yaw) {array.markers.push_back(std::move(m));}
  for (auto & m : gauge) {array.markers.push_back(std::move(m));}
  return array;
}

visualization_msgs::msg::MarkerArray delete_frame_orientation_norm_markers(
  const std::string & frame, const std::string & task_id)
{
  return frame_orientation_norm_markers(FrameOrientationNormSnapshot{}, frame, task_id, 1.0);
}

}  // namespace task_priority_kinematic_control
