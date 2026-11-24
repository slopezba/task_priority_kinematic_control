#include "task_priority_kinematic_control/core/capsule_observer.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>

namespace task_priority_kinematic_control
{
namespace
{
using Marker = visualization_msgs::msg::Marker;

// Half extent drawn for position axes that are not constrained (free axis).
constexpr double kFreeHalfExtent = 0.25;
constexpr size_t kArcPoints = 33;
constexpr double kRingRadius = 0.07;
constexpr double kArrowLength = 0.06;
constexpr size_t kRingMarkers = 6;
constexpr const char * kRotationNames[3] = {"rx", "ry", "rz"};

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

// 0 = inside, 1 = inside release band (close to the bound), 2 = task holding the axis.
int axis_level(const FramePoseLimitsSnapshot & s, size_t axis)
{
  const auto i = static_cast<Eigen::Index>(axis);
  if (s.active[axis]) {return 2;}
  return std::abs(s.error(i)) >= s.threshold(i) - s.delta(i) ? 1 : 0;
}

std_msgs::msg::ColorRGBA level_color(int level, float alpha)
{
  if (level >= 2) {return rgba(1.0f, 0.0f, 0.0f, alpha);}
  if (level == 1) {return rgba(1.0f, 0.55f, 0.0f, alpha);}
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

void set_pose(Marker & m, const Eigen::Vector3d & p, const Eigen::Quaterniond & q)
{
  m.pose.position = to_point(p);
  m.pose.orientation.x = q.x(); m.pose.orientation.y = q.y();
  m.pose.orientation.z = q.z(); m.pose.orientation.w = q.w();
}

// Twelve box edges (LINE_LIST) with the given half extents, in the marker pose frame.
void box_edges(Marker & m, const Eigen::Vector3d & half)
{
  m.points.clear();
  m.points.reserve(24);
  auto corner = [&](int i) {
      return Eigen::Vector3d((i & 1) ? half.x() : -half.x(),
               (i & 2) ? half.y() : -half.y(), (i & 4) ? half.z() : -half.z());
    };
  for (int i = 0; i < 8; ++i) {
    for (int bit : {1, 2, 4}) {
      if (!(i & bit)) {
        m.points.push_back(to_point(corner(i)));
        m.points.push_back(to_point(corner(i | bit)));
      }
    }
  }
}

}  // namespace

visualization_msgs::msg::MarkerArray frame_pose_limits_markers(
  const FramePoseLimitsSnapshot & s, const std::string & frame,
  const std::string & task_id, double publish_rate)
{
  const double lifetime = std::min(2147483646.0,
      std::max(2.0, 3.0 / std::max(publish_rate, 1e-9)));
  const auto stamp = static_cast<builtin_interfaces::msg::Time>(rclcpp::Time(s.timestamp_ns));
  const std::string prefix = task_id + "/limits/";

  visualization_msgs::msg::MarkerArray array;
  auto box = base_marker(frame, prefix + "box", 0, Marker::CUBE, stamp, lifetime);
  auto edges = base_marker(frame, prefix + "box_edges", 0, Marker::LINE_LIST, stamp, lifetime);
  auto activation = base_marker(frame, prefix + "activation", 0, Marker::LINE_LIST, stamp, lifetime);
  auto current = base_marker(frame, prefix + "frame", 0, Marker::SPHERE, stamp, lifetime);
  std::array<std::array<Marker, kRingMarkers>, 3> rings;
  for (size_t r = 0; r < 3; ++r) {
    for (size_t id = 0; id < kRingMarkers; ++id) {
      const int type = id <= 2 ? Marker::LINE_STRIP : (id <= 4 ? Marker::LINE_LIST : Marker::ARROW);
      rings[r][id] = base_marker(frame, prefix + kRotationNames[r], static_cast<int>(id), type,
          stamp, lifetime);
    }
  }

  const bool usable = s.visualization_enabled && s.valid && s.enabled &&
    s.target_position.allFinite() && s.target_orientation.coeffs().allFinite() &&
    s.frame_position.allFinite() && s.error.allFinite() && s.threshold.allFinite() &&
    s.alpha.allFinite() && s.delta.allFinite();

  if (usable) {
    const Eigen::Quaterniond q = s.target_orientation.normalized();
    const Eigen::Matrix3d rotation = q.toRotationMatrix();

    // Position: box of +/- threshold around the target, in the target axes.
    bool any_position = false;
    int position_level = 0;
    Eigen::Vector3d half_limit, half_activation;
    for (size_t axis = 0; axis < 3; ++axis) {
      const auto i = static_cast<Eigen::Index>(axis);
      if (s.axes[axis]) {
        any_position = true;
        position_level = std::max(position_level, axis_level(s, axis));
        half_limit(i) = s.threshold(i);
        half_activation(i) = s.threshold(i) - s.alpha(i);
      } else {
        half_limit(i) = half_activation(i) = kFreeHalfExtent;
      }
    }
    if (any_position) {
      box.action = edges.action = activation.action = Marker::ADD;
      set_pose(box, s.target_position, q);
      set_pose(edges, s.target_position, q);
      set_pose(activation, s.target_position, q);
      box.scale.x = 2 * half_limit.x(); box.scale.y = 2 * half_limit.y();
      box.scale.z = 2 * half_limit.z();
      box.color = level_color(position_level, 0.15f);
      edges.scale.x = 0.004;
      edges.color = level_color(position_level, 0.9f);
      box_edges(edges, half_limit);
      activation.scale.x = 0.002;
      activation.color = rgba(1.0f, 0.55f, 0.0f, 0.6f);
      box_edges(activation, half_activation);
    }
    current.action = Marker::ADD;
    current.pose.position = to_point(s.frame_position);
    current.scale.x = current.scale.y = current.scale.z = 0.02;
    current.color = level_color(position_level, 1.0f);

    // Orientation: one ring per constrained rotation axis, centred on the frame.
    // Ring in the plane (u, v) normal to target axis k; angle 0 points along u.
    for (size_t r = 0; r < 3; ++r) {
      const size_t axis = 3 + r;
      if (!s.axes[axis]) {continue;}
      const auto i = static_cast<Eigen::Index>(axis);
      const Eigen::Vector3d u = rotation.col(static_cast<Eigen::Index>((r + 1) % 3));
      const Eigen::Vector3d v = rotation.col(static_cast<Eigen::Index>((r + 2) % 3));
      const Eigen::Vector3d & c = s.frame_position;
      auto ray = [&](double angle, double radius) {
          return to_point(c + radius * (std::cos(angle) * u + std::sin(angle) * v));
        };
      const double limit = s.threshold(i);
      const double inner = s.threshold(i) - s.alpha(i);
      const auto color = level_color(axis_level(s, axis), 1.0f);
      for (size_t id = 0; id < kRingMarkers; ++id) {
        auto & m = rings[r][id];
        m.action = Marker::ADD;
        m.color = color;
      }
      // 0: full allowed arc; 1/2: activation bands next to each bound.
      const std::array<std::pair<double, double>, 3> arcs{{
        {-limit, limit}, {-limit, -inner}, {inner, limit}}};
      for (size_t id = 0; id < 3; ++id) {
        auto & m = rings[r][id];
        m.scale.x = id == 0 ? 0.006 : 0.010;
        m.color.a = id == 0 ? 0.7f : 0.45f;
        m.points.resize(kArcPoints);
        for (size_t k = 0; k < kArcPoints; ++k) {
          const double t = static_cast<double>(k) / (kArcPoints - 1);
          m.points[k] = ray(arcs[id].first + (arcs[id].second - arcs[id].first) * t, kRingRadius);
        }
      }
      // 3: ticks at the bounds (red); 4: ticks where the task activates (orange).
      for (size_t id = 3; id <= 4; ++id) {
        auto & m = rings[r][id];
        const double a = id == 3 ? limit : inner;
        m.scale.x = 0.004;
        m.color = id == 3 ? rgba(1.0f, 0.0f, 0.0f, 1.0f) : rgba(1.0f, 0.55f, 0.0f, 1.0f);
        m.points = {ray(-a, kRingRadius - 0.012), ray(-a, kRingRadius + 0.012),
          ray(a, kRingRadius - 0.012), ray(a, kRingRadius + 0.012)};
      }
      // 5: arrow at the current orientation offset from the target (-error).
      auto & arrow = rings[r][5];
      arrow.scale.x = 0.006; arrow.scale.y = 0.014; arrow.scale.z = 0.02;
      arrow.points = {to_point(c), ray(-s.error(i), kArrowLength)};
    }
  }

  array.markers.reserve(4 + 3 * kRingMarkers);
  array.markers.push_back(std::move(box));
  array.markers.push_back(std::move(edges));
  array.markers.push_back(std::move(activation));
  array.markers.push_back(std::move(current));
  for (auto & ring : rings) {
    for (auto & m : ring) {array.markers.push_back(std::move(m));}
  }
  return array;
}

visualization_msgs::msg::MarkerArray delete_frame_pose_limits_markers(
  const std::string & frame, const std::string & task_id)
{
  return frame_pose_limits_markers(FramePoseLimitsSnapshot{}, frame, task_id, 1.0);
}

}  // namespace task_priority_kinematic_control
