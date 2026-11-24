#include "task_priority_kinematic_control/core/capsule_observer.hpp"

#include <algorithm>
#include <cmath>

namespace task_priority_kinematic_control
{
namespace
{
using Marker = visualization_msgs::msg::Marker;

// Keeps a flattened (near-singular) ellipsoid visible as a thin disc.
constexpr double kMinDiameter = 0.003;

std_msgs::msg::ColorRGBA rgba(float r, float g, float b, float a)
{
  std_msgs::msg::ColorRGBA c;
  c.r = r; c.g = g; c.b = b; c.a = a;
  return c;
}

// Red: row active or below w_min; yellow: below w_des; green: well conditioned.
std_msgs::msg::ColorRGBA state_color(const ArmManipulabilitySnapshot & s, float alpha)
{
  if (s.active || s.w < s.w_min) {return rgba(1.0f, 0.0f, 0.0f, alpha);}
  if (s.w < s.w_des) {return rgba(1.0f, 0.9f, 0.0f, alpha);}
  return rgba(0.0f, 1.0f, 0.0f, alpha);
}

}  // namespace

visualization_msgs::msg::MarkerArray arm_manipulability_markers(
  const ArmManipulabilitySnapshot & s, const std::string & frame,
  const std::string & task_id, double publish_rate)
{
  const double lifetime = std::min(2147483646.0,
      std::max(2.0, 3.0 / std::max(publish_rate, 1e-9)));

  Marker ellipsoid;
  ellipsoid.header.frame_id = frame;
  ellipsoid.header.stamp = static_cast<builtin_interfaces::msg::Time>(rclcpp::Time(s.timestamp_ns));
  ellipsoid.ns = task_id + "/manipulability/ellipsoid";
  ellipsoid.id = 0;
  ellipsoid.type = Marker::SPHERE;
  ellipsoid.action = Marker::DELETE;
  ellipsoid.pose.orientation.w = 1.0;
  ellipsoid.scale.x = ellipsoid.scale.y = ellipsoid.scale.z = 1.0;
  ellipsoid.lifetime.sec = static_cast<int32_t>(lifetime);
  ellipsoid.lifetime.nanosec = static_cast<uint32_t>((lifetime - ellipsoid.lifetime.sec) * 1e9);

  const bool usable = s.visualization_enabled && s.valid && s.enabled &&
    s.tip_position.allFinite() && s.orientation.coeffs().allFinite() &&
    s.singular_values.allFinite() && std::isfinite(s.w) && std::isfinite(s.ellipsoid_scale) &&
    s.ellipsoid_scale > 0.0;
  if (usable) {
    const Eigen::Quaterniond q = s.orientation.normalized();
    ellipsoid.action = Marker::ADD;
    ellipsoid.pose.position.x = s.tip_position.x();
    ellipsoid.pose.position.y = s.tip_position.y();
    ellipsoid.pose.position.z = s.tip_position.z();
    ellipsoid.pose.orientation.x = q.x();
    ellipsoid.pose.orientation.y = q.y();
    ellipsoid.pose.orientation.z = q.z();
    ellipsoid.pose.orientation.w = q.w();
    ellipsoid.scale.x = std::max(kMinDiameter, 2.0 * s.ellipsoid_scale * s.singular_values.x());
    ellipsoid.scale.y = std::max(kMinDiameter, 2.0 * s.ellipsoid_scale * s.singular_values.y());
    ellipsoid.scale.z = std::max(kMinDiameter, 2.0 * s.ellipsoid_scale * s.singular_values.z());
    ellipsoid.color = state_color(s, 0.45f);
  }

  visualization_msgs::msg::MarkerArray array;
  array.markers.push_back(std::move(ellipsoid));
  return array;
}

visualization_msgs::msg::MarkerArray delete_arm_manipulability_markers(
  const std::string & frame, const std::string & task_id)
{
  return arm_manipulability_markers(ArmManipulabilitySnapshot{}, frame, task_id, 1.0);
}

}  // namespace task_priority_kinematic_control
