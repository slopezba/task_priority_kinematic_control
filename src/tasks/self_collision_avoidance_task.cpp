#include "task_priority_kinematic_control/tasks/self_collision_avoidance_task.hpp"
#include "task_priority_kinematic_control/geometry/capsule_geometry.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <set>
#include <sstream>
#include <stdexcept>

namespace task_priority_kinematic_control
{
size_t SelfCollisionAvoidanceTask::add_frame(const std::string & frame)
{
  const auto it = std::find(frames_.begin(), frames_.end(), frame);
  if (it != frames_.end()) {return static_cast<size_t>(it - frames_.begin());}
  frames_.push_back(frame);
  return frames_.size() - 1;
}

void SelfCollisionAvoidanceTask::configure(
  const std::string & id, const std::string & plugin,
  const std::map<std::string, rclcpp::Parameter> & params, const TaskContext & context)
{
  configure_common(id, plugin, params, context);
  const auto fail = [&](const std::string & reason) {
      throw std::invalid_argument("Task '" + id + "': " + reason);
    };
  const auto kinematics = context.backend.lock();
  if (!context.model || !kinematics) {fail("a configured model and backend are required");}
  safe_distance_ = get_double_param(params, "safe_distance", 0.05);
  activation_distance_ = get_double_param(params, "activation_distance", 0.12);
  gain_ = get_double_param(params, "gain_scalar", 1.0);
  max_repulsive_velocity_ = get_double_param(params, "max_repulsive_velocity", 0.08);
  parallel_epsilon_ = get_double_param(params, "parallel_epsilon", 1e-8);
  segment_length_epsilon_ = get_double_param(params, "segment_length_epsilon", 1e-9);
  degenerate_axis_distance_ = get_double_param(params, "degenerate_axis_distance", 1e-6);
  capsule_publish_rate_ = get_double_param(params, "capsule_publish_rate", 1.0);
  publish_capsules_.store(get_bool_param(params, "publish_capsules", true));
  for (const auto value : {safe_distance_, activation_distance_, gain_, max_repulsive_velocity_,
      parallel_epsilon_, segment_length_epsilon_, degenerate_axis_distance_, capsule_publish_rate_}) {
    if (!std::isfinite(value)) {fail("distances, gains, tolerances and rate must be finite");}
  }
  if (safe_distance_ < 0 || activation_distance_ < safe_distance_ || gain_ < 0 ||
    max_repulsive_velocity_ < 0) {fail("require activation_distance >= safe_distance >= 0 and nonnegative gains/velocity");}
  if (parallel_epsilon_ <= 0 || parallel_epsilon_ >= 1 || segment_length_epsilon_ <= 0 ||
    degenerate_axis_distance_ <= 0) {fail("invalid geometry tolerance (parallel_epsilon must be between 0 and 1)");}
  if (publish_capsules() && capsule_publish_rate_ <= 0) {fail("capsule_publish_rate must be positive when enabled");}
  capsules_.clear(); pairs_.clear(); frames_.clear(); endpoints_.clear();
  add_frame(context.model->base_frame());
  const auto names = get_string_array_param(params, "capsule_names", {});
  if (names.empty()) {fail("capsule_names cannot be empty; automatic URDF capsules are no longer supported");}
  std::set<std::string> unique;
  for (const auto & name : names) {
    if (name.empty() || !std::all_of(name.begin(), name.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '_';
      }) || !unique.insert(name).second) {fail("invalid or duplicate capsule name: " + name);}
    const std::string prefix = "capsules." + name + ".";
    for (const auto & field : {"start_frame", "end_frame", "radius"}) {
      if (!params.count(prefix + field)) {fail("missing parameter " + prefix + field);}
    }
    ManualCapsule capsule;
    capsule.name = name;
    capsule.start_frame = get_string_param(params, prefix + "start_frame", "");
    capsule.end_frame = get_string_param(params, prefix + "end_frame", "");
    capsule.radius = get_double_param(params, prefix + "radius", 0);
    if (capsule.start_frame.empty() || capsule.end_frame.empty()) {fail("empty endpoint frame for " + name);}
    if (!std::isfinite(capsule.radius) || capsule.radius <= 0) {fail("radius must be positive and finite for " + name);}
    capsule.start_index = add_frame(capsule.start_frame);
    capsule.end_index = add_frame(capsule.end_frame);
    capsules_.push_back(std::move(capsule));
  }
  for (const auto & parameter : params) {
    if (parameter.first.rfind("capsules.", 0) != 0) {continue;}
    const auto dot = parameter.first.find('.', 9);
    const auto name = parameter.first.substr(9, dot - 9);
    const auto field = dot == std::string::npos ? std::string{} : parameter.first.substr(dot + 1);
    if (!unique.count(name) || (field != "start_frame" && field != "end_frame" && field != "radius")) {
      fail("unknown capsule definition field: " + parameter.first);
    }
  }
  std::set<std::array<size_t, 2>> ignored;
  for (const auto & entry : get_string_array_param(params, "ignored_collision_pairs", {})) {
    const auto comma = entry.find(',');
    if (comma == std::string::npos || entry.find(',', comma + 1) != std::string::npos) {
      fail("ignored pair must contain exactly two comma-separated capsule names: " + entry);
    }
    const auto first = std::find(names.begin(), names.end(), entry.substr(0, comma));
    const auto second = std::find(names.begin(), names.end(), entry.substr(comma + 1));
    if (first == names.end() || second == names.end()) {fail("unknown capsule in ignored pair: " + entry);}
    if (first == second) {fail("ignored pair references the same capsule twice: " + entry);}
    size_t a = static_cast<size_t>(first - names.begin()), b = static_cast<size_t>(second - names.begin());
    if (a > b) {std::swap(a, b);}
    ignored.insert({a, b});
  }
  ignored_pair_count_ = ignored.size();
  for (size_t i = 0; i < capsules_.size(); ++i) {
    for (size_t j = i + 1; j < capsules_.size(); ++j) {
      if (!ignored.count({i, j})) {pairs_.push_back({i, j});}
    }
  }
  // The caller seeds the existing backend cache before configuring tasks. Exact frame names only.
  endpoints_.resize(frames_.size());
  const Eigen::Index dofs = static_cast<Eigen::Index>(context.model->total_dofs());
  for (size_t i = 0; i < frames_.size(); ++i) {
    auto & endpoint = endpoints_[i];
    endpoint.world.jacobian.setZero(6, dofs);
    endpoint.jacobian.setZero(3, dofs);
    if (!kinematics->read_frame_state(frames_[i], endpoint.world) ||
      endpoint.world.jacobian.rows() != 6 || endpoint.world.jacobian.cols() != dofs ||
      !endpoint.world.jacobian.allFinite() || !endpoint.world.pose.matrix().allFinite()) {
      fail("frame cannot be evaluated with a finite 6 x model_dofs Jacobian: " + frames_[i]);
    }
  }
  jacobian_.setZero(pairs_.size(), dofs);
  point_difference_.setZero(3, dofs);
  desired_.setZero(pairs_.size()); errors_.setZero(pairs_.size());
  snapshots_.initialize([&](CollisionSnapshot & snapshot) {snapshot.capsules.resize(capsules_.size());});
  last_ = CollisionMetrics{};
}

void SelfCollisionAvoidanceTask::prepare_computation(TaskComputation & out) const
{
  out.jacobian.setZero(jacobian_.rows(), jacobian_.cols());
  out.desired_velocity.setZero(desired_.size()); out.error.setZero(errors_.size());
  out.status_message.reserve(64);
}

TaskComputation SelfCollisionAvoidanceTask::update(const WholeBodyState & state, const KinematicsBackend & backend)
{
  TaskComputation out;
  prepare_computation(out);
  update_into(state, backend, out);
  return out;
}

void SelfCollisionAvoidanceTask::invalid(TaskComputation & out)
{
  last_.invalid_geometry = true; last_.stop_arms = true;
  jacobian_.setZero(); desired_.setZero(); errors_.setZero();
  out.jacobian.setZero(); out.desired_velocity.setZero(); out.error.setZero();
  out.active = false; out.stop_arm_motion = true;
  out.status_message = "invalid_collision_geometry";
  // Never display stale/partial capsule locations as valid geometry.
  for (auto & capsule : snapshots_.writable().capsules) {capsule.valid = false;}
}

void SelfCollisionAvoidanceTask::update_into(
  const WholeBodyState &, const KinematicsBackend & backend, TaskComputation & out)
{
  last_ = CollisionMetrics{}; last_.enabled = enabled_;
  jacobian_.setZero(); desired_.setZero(); errors_.setZero();
  out.active = false; out.stop_arm_motion = false;
  out.jacobian.setZero(); out.desired_velocity.setZero(); out.error.setZero();
  for (auto & capsule : snapshots_.writable().capsules) {capsule = CapsulePose{};}
  if (!enabled_) {out.status_message = "disabled"; return;}
  for (size_t i = 0; i < frames_.size(); ++i) {
    auto & frame = endpoints_[i].world;
    if (!backend.read_frame_state(frames_[i], frame) || frame.jacobian.rows() != 6 ||
      frame.jacobian.cols() != jacobian_.cols() || !frame.jacobian.allFinite() ||
      !frame.pose.matrix().allFinite()) {invalid(out); return;}
  }
  const auto & base = endpoints_[0].world;
  const Eigen::Matrix3d rotation = base.pose.rotation().transpose();
  for (auto & endpoint : endpoints_) {
    const Eigen::Vector3d r = endpoint.world.pose.translation() - base.pose.translation();
    endpoint.position = rotation * r;
    for (Eigen::Index col = 0; col < jacobian_.cols(); ++col) {
      // Moving-reference derivative: R^T(v_frame - v_base - omega_base x r).
      endpoint.jacobian.col(col) = rotation * (
        endpoint.world.jacobian.col(col).head<3>() - base.jacobian.col(col).head<3>() +
        r.cross(base.jacobian.col(col).tail<3>()));
    }
    if (!endpoint.position.allFinite() || !endpoint.jacobian.allFinite()) {invalid(out); return;}
  }
  auto & snapshot = snapshots_.writable();
  for (size_t i = 0; i < capsules_.size(); ++i) {
    snapshot.capsules[i].a = endpoints_[capsules_[i].start_index].position;
    snapshot.capsules[i].b = endpoints_[capsules_[i].end_index].position;
    snapshot.capsules[i].valid = true;
  }
  for (size_t i = 0; i < pairs_.size(); ++i) {
    const auto a = pairs_[i][0], b = pairs_[i][1];
    const auto & ca = capsules_[a]; const auto & cb = capsules_[b];
    const auto & pa = snapshot.capsules[a]; const auto & pb = snapshot.capsules[b];
    const auto distance = capsule_distance(pa.a, pa.b, pb.a, pb.b,
      ca.radius, cb.radius, parallel_epsilon_, segment_length_epsilon_);
    if (!distance.valid) {last_.fault_pair = i; invalid(out); return;}
    if (distance.clearance < last_.min_clearance) {
      last_.min_clearance = distance.clearance; last_.closest_pair = i;
    }
    if (distance.axis_distance <= degenerate_axis_distance_) {
      ++last_.degenerate_pairs; last_.stop_arms = true;
      if (last_.fault_pair == std::numeric_limits<size_t>::max()) {last_.fault_pair = i;}
      snapshot.capsules[a].highlighted = snapshot.capsules[b].highlighted = true;
      continue;
    }
    if (distance.clearance >= activation_distance_) {continue;}
    const Eigen::Vector3d normal = (distance.q - distance.p) / distance.axis_distance;
    // Closest points can interpolate endpoints on different rigid links.
    point_difference_.noalias() = (1 - distance.t) * endpoints_[cb.start_index].jacobian;
    point_difference_.noalias() += distance.t * endpoints_[cb.end_index].jacobian;
    point_difference_.noalias() -= (1 - distance.s) * endpoints_[ca.start_index].jacobian;
    point_difference_.noalias() -= distance.s * endpoints_[ca.end_index].jacobian;
    // Scalar columns avoid temporary packing for a strided Eigen matrix row.
    for (Eigen::Index col = 0; col < jacobian_.cols(); ++col) {
      jacobian_(static_cast<Eigen::Index>(i), col) = normal.dot(point_difference_.col(col));
    }
    // A rigid motion of the whole robot cannot resolve a self collision.
    jacobian_.row(i).head(context_.model->base_dofs()).setZero();
    desired_(i) = std::clamp(gain_ * (activation_distance_ - distance.clearance), 0.0, max_repulsive_velocity_);
    errors_(i) = std::max(0.0, safe_distance_ - distance.clearance);
    ++last_.active_pairs;
    snapshot.capsules[a].highlighted = snapshot.capsules[b].highlighted = true;
  }
  if (!jacobian_.allFinite() || !desired_.allFinite() || !errors_.allFinite()) {invalid(out); return;}
  out.jacobian = jacobian_; out.desired_velocity = desired_; out.error = errors_;
  out.active = last_.active_pairs > 0;
  out.stop_arm_motion = last_.stop_arms;
  out.status_message = last_.stop_arms ? "degenerate_collision_geometry" :
    (out.active ? "repelling_self_collision" : "clear");
}

void SelfCollisionAvoidanceTask::observe_command(const WholeBodyCommand & command, int64_t timestamp_ns)
{
  last_.max_velocity_deficit = 0.0;
  if (command.generalized_velocity.size() == jacobian_.cols()) {
    for (Eigen::Index i = 0; i < desired_.size(); ++i) {
      if (desired_(i) > 0) {
        const double achieved = jacobian_.row(i).dot(command.generalized_velocity);
        last_.max_velocity_deficit = std::max(last_.max_velocity_deficit, desired_(i) - achieved);
      }
    }
  }
  auto & snapshot = snapshots_.writable();
  snapshot.metrics = last_; snapshot.timestamp_ns = timestamp_ns;
  snapshots_.publish();
}

bool SelfCollisionAvoidanceTask::set_gain_scalar(double gain, std::string & message)
{
  if (!std::isfinite(gain) || gain < 0) {message = "gain_scalar must be finite and nonnegative"; return false;}
  gain_ = gain; message = "Collision avoidance gain updated"; return true;
}

msg::TaskStatus SelfCollisionAvoidanceTask::build_status() const
{
  auto status = TaskBaseCommon::build_status();
  status.active = enabled_ && (last_.active_pairs > 0 || last_.stop_arms);
  std::ostringstream text;
  text << "capsules=" << capsules_.size() << " checked_pairs=" << pairs_.size()
       << " ignored_pairs=" << ignored_pair_count_ << " min_clearance=";
  if (std::isfinite(last_.min_clearance)) {text << last_.min_clearance;} else {text << "unknown";}
  if (last_.closest_pair < pairs_.size()) {
    const auto & pair = pairs_[last_.closest_pair];
    text << " closest_pair=" << capsules_[pair[0]].name << ',' << capsules_[pair[1]].name;
  }
  text << " active_pairs=" << last_.active_pairs << " degenerate_pairs=" << last_.degenerate_pairs
       << " stop_arms=" << last_.stop_arms << " invalid_geometry=" << last_.invalid_geometry
       << " max_velocity_deficit=" << last_.max_velocity_deficit;
  status.status_message = text.str();
  status.error.resize(errors_.size()); status.command.resize(desired_.size());
  for (Eigen::Index i = 0; i < errors_.size(); ++i) {
    status.error[static_cast<size_t>(i)] = errors_(i);
    status.command[static_cast<size_t>(i)] = desired_(i);
  }
  return status;
}
}  // namespace task_priority_kinematic_control
PLUGINLIB_EXPORT_CLASS(task_priority_kinematic_control::SelfCollisionAvoidanceTask, task_priority_kinematic_control::TaskBase)
