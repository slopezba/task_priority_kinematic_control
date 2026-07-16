#include "task_priority_kinematic_control/tasks/self_collision_avoidance_task.hpp"

#include <pluginlib/class_list_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace task_priority_kinematic_control
{

namespace
{

struct ClosestSegmentPoints
{
  Eigen::Vector3d a = Eigen::Vector3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  double distance = 0.0;
};

double clamp01(double value)
{
  return std::clamp(value, 0.0, 1.0);
}

ClosestSegmentPoints closest_points_on_segments(
  const Eigen::Vector3d & p1,
  const Eigen::Vector3d & q1,
  const Eigen::Vector3d & p2,
  const Eigen::Vector3d & q2)
{
  constexpr double eps = 1e-12;
  const Eigen::Vector3d d1 = q1 - p1;
  const Eigen::Vector3d d2 = q2 - p2;
  const Eigen::Vector3d r = p1 - p2;
  const double a = d1.dot(d1);
  const double e = d2.dot(d2);
  const double f = d2.dot(r);

  double s = 0.0;
  double t = 0.0;
  if (a <= eps && e <= eps) {
    s = 0.0;
    t = 0.0;
  } else if (a <= eps) {
    s = 0.0;
    t = clamp01(f / e);
  } else {
    const double c = d1.dot(r);
    if (e <= eps) {
      t = 0.0;
      s = clamp01(-c / a);
    } else {
      const double b = d1.dot(d2);
      const double denom = a * e - b * b;
      if (denom != 0.0) {
        s = clamp01((b * f - c * e) / denom);
      }
      t = (b * s + f) / e;
      if (t < 0.0) {
        t = 0.0;
        s = clamp01(-c / a);
      } else if (t > 1.0) {
        t = 1.0;
        s = clamp01((b - c) / a);
      }
    }
  }

  ClosestSegmentPoints out;
  out.a = p1 + d1 * s;
  out.b = p2 + d2 * t;
  out.distance = (out.a - out.b).norm();
  return out;
}

bool contains_any(const std::string & value, const std::vector<std::string> & needles)
{
  for (const auto & needle : needles) {
    if (!needle.empty() && value.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

}  // namespace

void SelfCollisionAvoidanceTask::configure(
  const std::string & id,
  const std::string & plugin_name,
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const TaskContext & context)
{
  configure_common(id, plugin_name, parameters, context);
  safe_distance_ = get_double_param(parameters, "safe_distance", 0.05);
  activation_distance_ = get_double_param(parameters, "activation_distance", 0.12);
  gain_ = get_double_param(parameters, "gain_scalar", 1.0);
  max_repulsive_velocity_ = get_double_param(parameters, "max_repulsive_velocity", 0.08);
  check_adjacent_links_ = get_bool_param(parameters, "check_adjacent_links", false);
  include_link_substrings_ = get_string_array_param(
    parameters,
    "include_link_substrings",
    {"base_link", "alpha_left", "alpha_right"});
  exclude_link_substrings_ = get_string_array_param(parameters, "exclude_link_substrings", {});
  if (activation_distance_ < safe_distance_) {
    activation_distance_ = safe_distance_;
  }
}

TaskComputation SelfCollisionAvoidanceTask::update(
  const WholeBodyState &,
  const KinematicsBackend & backend)
{
  TaskComputation computation;
  computation.active = false;
  if (!enabled_) {
    computation.status_message = "disabled";
    return computation;
  }

  std::vector<WorldCapsule> capsules;
  const auto & source_capsules = backend.collision_capsules();
  capsules.reserve(source_capsules.size());
  for (const auto & capsule : source_capsules) {
    if (!included_link(capsule.link_name)) {
      continue;
    }
    try {
      WorldCapsule world;
      world.capsule = capsule;
      world.frame = backend.get_frame_state(capsule.link_name);
      world.a = world.frame.pose * capsule.local_a;
      world.b = world.frame.pose * capsule.local_b;
      capsules.push_back(world);
    } catch (const std::exception &) {
      continue;
    }
  }

  std::vector<Eigen::RowVectorXd> rows;
  std::vector<double> desired;
  std::vector<double> errors;
  last_min_distance_ = std::numeric_limits<double>::infinity();
  last_active_pairs_ = 0;

  for (size_t i = 0; i < capsules.size(); ++i) {
    for (size_t j = i + 1; j < capsules.size(); ++j) {
      if (ignored_pair(capsules[i].capsule, capsules[j].capsule)) {
        continue;
      }

      const auto closest = closest_points_on_segments(
        capsules[i].a, capsules[i].b, capsules[j].a, capsules[j].b);
      const double surface_distance =
        closest.distance - capsules[i].capsule.radius - capsules[j].capsule.radius;
      last_min_distance_ = std::min(last_min_distance_, surface_distance);
      if (surface_distance >= activation_distance_) {
        continue;
      }

      Eigen::Vector3d normal = closest.a - closest.b;
      if (normal.squaredNorm() < 1e-12) {
        normal = capsules[i].frame.pose.translation() - capsules[j].frame.pose.translation();
      }
      if (normal.squaredNorm() < 1e-12) {
        normal = Eigen::Vector3d::UnitZ();
      }
      normal.normalize();

      const Eigen::MatrixXd jacobian_a = point_jacobian(capsules[i].frame, closest.a);
      const Eigen::MatrixXd jacobian_b = point_jacobian(capsules[j].frame, closest.b);
      rows.push_back(normal.transpose() * (jacobian_a - jacobian_b));
      const double command = std::clamp(
        gain_ * (activation_distance_ - surface_distance),
        0.0,
        max_repulsive_velocity_);
      desired.push_back(command);
      errors.push_back(std::max(0.0, safe_distance_ - surface_distance));
      ++last_active_pairs_;
    }
  }

  if (rows.empty()) {
    computation.status_message = "clear";
    computation.error = Eigen::VectorXd::Zero(1);
    computation.desired_velocity = Eigen::VectorXd::Zero(1);
    computation.error(0) = std::isfinite(last_min_distance_) ? last_min_distance_ : 0.0;
    return computation;
  }

  computation.active = true;
  computation.jacobian = Eigen::MatrixXd::Zero(
    static_cast<Eigen::Index>(rows.size()),
    static_cast<Eigen::Index>(context_.model->total_dofs()));
  computation.desired_velocity = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(rows.size()));
  computation.error = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(rows.size()));
  for (size_t i = 0; i < rows.size(); ++i) {
    computation.jacobian.row(static_cast<Eigen::Index>(i)) = rows[i];
    computation.desired_velocity(static_cast<Eigen::Index>(i)) = desired[i];
    computation.error(static_cast<Eigen::Index>(i)) = errors[i];
  }
  computation.status_message = "repelling_self_collision";
  return computation;
}

msg::TaskStatus SelfCollisionAvoidanceTask::build_status() const
{
  auto status = TaskBaseCommon::build_status();
  std::ostringstream message;
  message << (enabled_ ? "min_distance=" : "disabled min_distance=");
  if (std::isfinite(last_min_distance_)) {
    message << last_min_distance_;
  } else {
    message << "unknown";
  }
  message << " active_pairs=" << last_active_pairs_;
  status.status_message = message.str();
  status.active = enabled_ && last_active_pairs_ > 0;
  return status;
}

bool SelfCollisionAvoidanceTask::included_link(const std::string & link_name) const
{
  if (!include_link_substrings_.empty() && !contains_any(link_name, include_link_substrings_)) {
    return false;
  }
  return !contains_any(link_name, exclude_link_substrings_);
}

bool SelfCollisionAvoidanceTask::ignored_pair(
  const CollisionCapsule & a,
  const CollisionCapsule & b) const
{
  if (a.link_name == b.link_name) {
    return true;
  }
  if (!check_adjacent_links_ &&
    (a.parent_link_name == b.link_name || b.parent_link_name == a.link_name))
  {
    return true;
  }
  const bool a_base = a.link_name == context_.model->base_frame();
  const bool b_base = b.link_name == context_.model->base_frame();
  if (a_base && b_base) {
    return true;
  }
  return false;
}

Eigen::MatrixXd SelfCollisionAvoidanceTask::point_jacobian(
  const FrameState & frame,
  const Eigen::Vector3d & point_world) const
{
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(3, frame.jacobian.cols());
  if (frame.jacobian.rows() < 6) {
    return out;
  }
  const Eigen::Vector3d r = point_world - frame.pose.translation();
  out = frame.jacobian.topRows(3) - skew(r) * frame.jacobian.bottomRows(3);
  return out;
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::SelfCollisionAvoidanceTask,
  task_priority_kinematic_control::TaskBase)
