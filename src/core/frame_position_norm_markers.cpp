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

// Half length drawn along axes that are not part of the norm (free axes).
constexpr double kFreeHalfExtent = 0.25;
constexpr int kCirclePoints = 40;

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

// 0 = inside, 1 = inside the release band (close to the bound), 2 = row active.
std_msgs::msg::ColorRGBA state_color(const FramePositionNormSnapshot & s, float alpha)
{
  if (s.active) {return rgba(1.0f, 0.0f, 0.0f, alpha);}
  if (s.norm >= s.limit - s.delta) {return rgba(1.0f, 0.55f, 0.0f, alpha);}
  return rgba(0.0f, 1.0f, 0.0f, alpha);
}

Marker base_marker(
  const std::string & frame, const std::string & ns, int type,
  const builtin_interfaces::msg::Time & stamp, double lifetime)
{
  Marker m;
  m.header.frame_id = frame;
  m.header.stamp = stamp;
  m.ns = ns;
  m.id = 0;
  m.type = type;
  m.action = Marker::DELETE;
  m.pose.orientation.w = 1.0;
  m.scale.x = m.scale.y = m.scale.z = 1.0;
  m.lifetime.sec = static_cast<int32_t>(lifetime);
  m.lifetime.nanosec = static_cast<uint32_t>((lifetime - m.lifetime.sec) * 1e9);
  return m;
}

void set_orientation(Marker & m, const Eigen::Quaterniond & q)
{
  m.pose.orientation.x = q.x(); m.pose.orientation.y = q.y();
  m.pose.orientation.z = q.z(); m.pose.orientation.w = q.w();
}

// Closed circle as LINE_LIST segments, in the marker frame.
void add_circle(Marker & m, const Eigen::Vector3d & centre, const Eigen::Vector3d & u,
  const Eigen::Vector3d & v, double radius)
{
  for (int k = 0; k < kCirclePoints; ++k) {
    const double a0 = 2.0 * M_PI * k / kCirclePoints;
    const double a1 = 2.0 * M_PI * (k + 1) / kCirclePoints;
    m.points.push_back(to_point(centre + radius * (std::cos(a0) * u + std::sin(a0) * v)));
    m.points.push_back(to_point(centre + radius * (std::cos(a1) * u + std::sin(a1) * v)));
  }
}

// Closed rectangle in the plane (u, v) as LINE_LIST segments.
void add_rectangle(Marker & m, const Eigen::Vector3d & centre, const Eigen::Vector3d & u,
  const Eigen::Vector3d & v, double half_u, double half_v)
{
  const Eigen::Vector3d c[4] = {
    centre - half_u * u - half_v * v, centre + half_u * u - half_v * v,
    centre + half_u * u + half_v * v, centre - half_u * u + half_v * v};
  for (int k = 0; k < 4; ++k) {
    m.points.push_back(to_point(c[k]));
    m.points.push_back(to_point(c[(k + 1) % 4]));
  }
}

}  // namespace

visualization_msgs::msg::MarkerArray frame_position_norm_markers(
  const FramePositionNormSnapshot & s, const std::string & frame,
  const std::string & task_id, double publish_rate)
{
  const double lifetime = std::min(2147483646.0,
      std::max(2.0, 3.0 / std::max(publish_rate, 1e-9)));
  const auto stamp = static_cast<builtin_interfaces::msg::Time>(rclcpp::Time(s.timestamp_ns));
  const std::string prefix = task_id + "/limits/";

  auto region = base_marker(frame, prefix + "region", Marker::SPHERE, stamp, lifetime);
  auto activation = base_marker(frame, prefix + "activation", Marker::LINE_LIST, stamp, lifetime);
  auto current = base_marker(frame, prefix + "frame", Marker::SPHERE, stamp, lifetime);
  auto link = base_marker(frame, prefix + "link", Marker::LINE_LIST, stamp, lifetime);

  const bool usable = s.visualization_enabled && s.valid && s.enabled &&
    s.target_position.allFinite() && s.target_orientation.coeffs().allFinite() &&
    s.frame_position.allFinite() && std::isfinite(s.norm) && std::isfinite(s.limit) &&
    std::isfinite(s.alpha) && std::isfinite(s.delta) && s.limit > 0;

  if (usable) {
    const Eigen::Quaterniond q = s.target_orientation.normalized();
    const int count = static_cast<int>(s.axes[0]) + static_cast<int>(s.axes[1]) +
      static_cast<int>(s.axes[2]);
    const double r = s.limit;
    const double ra = std::max(0.0, s.limit - s.alpha);
    const Eigen::Vector3d ex = Eigen::Vector3d::UnitX();
    const Eigen::Vector3d ey = Eigen::Vector3d::UnitY();
    const Eigen::Vector3d ez = Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d units[3] = {ex, ey, ez};

    // Region and activation outline are drawn in the target pose (local axes).
    region.action = activation.action = Marker::ADD;
    region.pose.position = to_point(s.target_position);
    activation.pose.position = to_point(s.target_position);
    set_orientation(activation, q);
    region.color = state_color(s, 0.18f);
    activation.color = rgba(1.0f, 0.55f, 0.0f, 0.8f);
    activation.scale.x = 0.002;

    if (count == 3) {
      // Sphere ||e|| <= r.
      region.type = Marker::SPHERE;
      set_orientation(region, q);
      region.scale.x = region.scale.y = region.scale.z = 2 * r;
      add_circle(activation, Eigen::Vector3d::Zero(), ex, ey, ra);
      add_circle(activation, Eigen::Vector3d::Zero(), ey, ez, ra);
      add_circle(activation, Eigen::Vector3d::Zero(), ez, ex, ra);
    } else if (count == 2) {
      // Infinite cylinder along the free axis (drawn with a finite length).
      const size_t free_axis = !s.axes[0] ? 0 : (!s.axes[1] ? 1 : 2);
      const Eigen::Vector3d & axis = units[free_axis];
      const Eigen::Vector3d & u = units[(free_axis + 1) % 3];
      const Eigen::Vector3d & v = units[(free_axis + 2) % 3];
      region.type = Marker::CYLINDER;  // marker cylinder axis is local z
      set_orientation(region, q * Eigen::Quaterniond::FromTwoVectors(ez, axis));
      region.scale.x = region.scale.y = 2 * r;
      region.scale.z = 2 * kFreeHalfExtent;
      for (const double offset : {-kFreeHalfExtent, 0.0, kFreeHalfExtent}) {
        add_circle(activation, offset * axis, u, v, ra);
      }
    } else {
      // Slab |e_k| <= r between two planes normal to the constrained axis.
      const size_t axis_index = s.axes[0] ? 0 : (s.axes[1] ? 1 : 2);
      const Eigen::Vector3d & n = units[axis_index];
      const Eigen::Vector3d & u = units[(axis_index + 1) % 3];
      const Eigen::Vector3d & v = units[(axis_index + 2) % 3];
      region.type = Marker::CUBE;
      set_orientation(region, q);
      Eigen::Vector3d scale = Eigen::Vector3d::Constant(2 * kFreeHalfExtent);
      scale(static_cast<Eigen::Index>(axis_index)) = 2 * r;
      region.scale.x = scale.x(); region.scale.y = scale.y(); region.scale.z = scale.z();
      add_rectangle(activation, ra * n, u, v, kFreeHalfExtent, kFreeHalfExtent);
      add_rectangle(activation, -ra * n, u, v, kFreeHalfExtent, kFreeHalfExtent);
    }

    current.action = Marker::ADD;
    current.pose.position = to_point(s.frame_position);
    current.scale.x = current.scale.y = current.scale.z = 0.02;
    current.color = state_color(s, 1.0f);

    link.action = Marker::ADD;
    link.scale.x = 0.003;
    link.color = state_color(s, 0.9f);
    link.points = {to_point(s.target_position), to_point(s.frame_position)};
  }

  visualization_msgs::msg::MarkerArray array;
  array.markers.reserve(4);
  array.markers.push_back(std::move(region));
  array.markers.push_back(std::move(activation));
  array.markers.push_back(std::move(current));
  array.markers.push_back(std::move(link));
  return array;
}

visualization_msgs::msg::MarkerArray delete_frame_position_norm_markers(
  const std::string & frame, const std::string & task_id)
{
  return frame_position_norm_markers(FramePositionNormSnapshot{}, frame, task_id, 1.0);
}

}  // namespace task_priority_kinematic_control
