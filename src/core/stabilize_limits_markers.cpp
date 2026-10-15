#include "task_priority_kinematic_control/core/capsule_observer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>

namespace task_priority_kinematic_control
{
namespace
{
using Marker = visualization_msgs::msg::Marker;
constexpr size_t markers_per_axis = 7;
constexpr size_t arc_points = 33;
constexpr double arc_radius = .18;
constexpr double arrow_length = .16;

Eigen::Vector3d center(size_t axis)
{return Eigen::Vector3d(0, axis == 0 ? 0 : .22, axis == 0 ? -.15 : -.18);}

geometry_msgs::msg::Point point(const Eigen::Vector3d & p)
{
  geometry_msgs::msg::Point result;
  result.x = p.x(); result.y = p.y(); result.z = p.z();
  return result;
}

geometry_msgs::msg::Point ray(size_t axis, double angle, double radius)
{
  return point(center(axis) + (axis == 0 ?
    Eigen::Vector3d(0, radius * std::sin(angle), -radius * std::cos(angle)) :
    Eigen::Vector3d(-radius * std::sin(angle), 0, -radius * std::cos(angle))));
}

std_msgs::msg::ColorRGBA color(float r, float g, float b)
{
  std_msgs::msg::ColorRGBA result;
  result.r = r; result.g = g; result.b = b; result.a = 1;
  return result;
}
}  // namespace

StabilizeLimitsMarkerRenderer::StabilizeLimitsMarkerRenderer(
  const std::string & frame, const std::string & task_id, double publish_rate)
{
  if (!std::isfinite(publish_rate) || publish_rate <= 0) {
    throw std::invalid_argument("Marker rate must be finite and positive");
  }
  markers_.markers.resize(2 * markers_per_axis);
  const double lifetime = std::min(2147483646.0, std::max(2.0, 3.0 / publish_rate));
  for (size_t axis = 0; axis < 2; ++axis) {
    const std::string ns = task_id + "/limits/" + (axis == 0 ? "roll" : "pitch");
    for (size_t id = 0; id < markers_per_axis; ++id) {
      auto & m = markers_.markers[axis * markers_per_axis + id];
      m.header.frame_id = frame;
      m.ns = ns;
      m.id = static_cast<int>(id);
      m.action = Marker::DELETE;
      m.pose.orientation.w = 1;
      m.frame_locked = true;
      m.lifetime.sec = static_cast<int32_t>(lifetime);
      m.lifetime.nanosec = static_cast<uint32_t>((lifetime - m.lifetime.sec) * 1e9);
      if (id <= 2) {
        m.type = Marker::LINE_STRIP; m.scale.x = .008;
        m.points.resize(arc_points);
      } else if (id <= 4) {
        m.type = Marker::LINE_LIST; m.scale.x = .007;
        m.points.resize(4);
      } else if (id == 5) {
        m.type = Marker::ARROW; m.scale.x = .010; m.scale.y = .022; m.scale.z = .035;
        m.points.resize(2);
        m.points[0] = point(center(axis));
      } else {
        // Delete the text IDs used by the previous renderer. No text is created.
        m.type = Marker::LINE_LIST;
      }
    }
  }
}

void StabilizeLimitsMarkerRenderer::prepare_geometry(const StabilizeLimitsSnapshot & snapshot)
{
  if (geometry_ready_ && (lower_.array() == snapshot.lower.array()).all() &&
    (upper_.array() == snapshot.upper.array()).all() && alpha_ == snapshot.alpha && delta_ == snapshot.delta) {return;}
  for (size_t axis = 0; axis < 2; ++axis) {
    const double lower = snapshot.lower(static_cast<Eigen::Index>(axis));
    const double upper = snapshot.upper(static_cast<Eigen::Index>(axis));
    for (size_t id = 0; id < 3; ++id) {
      auto & points = markers_.markers[axis * markers_per_axis + id].points;
      const double start = id == 2 ? upper - snapshot.alpha : lower;
      const double end = id == 1 ? lower + snapshot.alpha : upper;
      for (size_t sample = 0; sample < arc_points; ++sample) {
        points[sample] = ray(axis, start + (end-start) * sample / (arc_points-1), arc_radius);
      }
    }
    for (size_t id = 3; id <= 4; ++id) {
      auto & points = markers_.markers[axis * markers_per_axis + id].points;
      const double start = id == 3 ? lower : lower + snapshot.alpha;
      const double end = id == 3 ? upper : upper - snapshot.alpha;
      points[0] = ray(axis, start, .17); points[1] = ray(axis, start, .20);
      points[2] = ray(axis, end, .17); points[3] = ray(axis, end, .20);
    }
  }
  lower_ = snapshot.lower; upper_ = snapshot.upper;
  alpha_ = snapshot.alpha; delta_ = snapshot.delta;
  geometry_ready_ = true;
}

const visualization_msgs::msg::MarkerArray & StabilizeLimitsMarkerRenderer::update(
  const StabilizeLimitsSnapshot & snapshot)
{
  if (!snapshot.visualization_enabled || !snapshot.valid || !snapshot.angles.allFinite() ||
    !snapshot.lower.allFinite() || !snapshot.upper.allFinite() ||
    !std::isfinite(snapshot.alpha) || !std::isfinite(snapshot.delta) ||
    (snapshot.upper.array() <= snapshot.lower.array()).any()) {return clear();}
  prepare_geometry(snapshot);
  const auto stamp = static_cast<builtin_interfaces::msg::Time>(rclcpp::Time(snapshot.timestamp_ns));
  for (size_t axis = 0; axis < 2; ++axis) {
    const Eigen::Index index = static_cast<Eigen::Index>(axis);
    const double angle = snapshot.angles(index);
    const double margin = std::min(angle - snapshot.lower(index), snapshot.upper(index) - angle);
    const auto state_color = !snapshot.enabled ? color(.5f,.5f,.5f) :
      (snapshot.active[axis] ? color(1,0,0) : (margin <= snapshot.delta ? color(1,.55f,0) : color(0,1,0)));
    for (size_t id = 0; id < markers_per_axis; ++id) {
      auto & m = markers_.markers[axis * markers_per_axis + id];
      m.header.stamp = stamp;
      m.action = id == 6 ? Marker::DELETE : Marker::ADD;
      m.color = state_color;
      if (snapshot.enabled && id == 3) {m.color = color(1,0,0);}
      if (snapshot.enabled && id == 4) {m.color = color(1,.55f,0);}
      if (id == 0) {m.color.a = .7f;}
      if (id == 1 || id == 2) {m.color.a = .45f;}
    }
    markers_.markers[axis * markers_per_axis + 5].points[1] = ray(axis, angle, arrow_length);
  }
  return markers_;
}

const visualization_msgs::msg::MarkerArray & StabilizeLimitsMarkerRenderer::clear()
{
  for (auto & m : markers_.markers) {m.action = Marker::DELETE;}
  return markers_;
}

}  // namespace task_priority_kinematic_control
