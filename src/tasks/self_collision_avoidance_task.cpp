#include "task_priority_kinematic_control/tasks/self_collision_avoidance_task.hpp"
#include "task_priority_kinematic_control/geometry/capsule_geometry.hpp"
#include "task_priority_kinematic_control/geometry/plane_geometry.hpp"
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
  reset();
  const auto fail = [&](const std::string & reason) {
      throw std::invalid_argument("Task '" + id + "': " + reason);
    };
  const auto kinematics = context.backend.lock();
  if (!context.model || !kinematics) {fail("a configured model and backend are required");}
  safe_distance_ = get_double_param(params, "safe_distance", 0.05);
  activation_distance_ = get_double_param(params, "activation_distance", 0.12);
  release_distance_ = get_double_param(params, "release_distance", activation_distance_ + 0.01);
  eps_ = get_double_param(params, "eps", 0.0001);
  gain_ = get_double_param(params, "gain_scalar", 1.0);
  max_repulsive_velocity_ = get_double_param(params, "max_repulsive_velocity", 0.08);
  parallel_epsilon_ = get_double_param(params, "parallel_epsilon", 1e-8);
  segment_length_epsilon_ = get_double_param(params, "segment_length_epsilon", 1e-9);
  degenerate_axis_distance_ = get_double_param(params, "degenerate_axis_distance", 1e-6);
  capsule_publish_rate_ = get_double_param(params, "capsule_publish_rate", 1.0);
  publish_capsules_.store(get_bool_param(params, "publish_capsules", true));
  for (const auto value : {safe_distance_, activation_distance_, release_distance_, eps_, gain_, max_repulsive_velocity_,
      parallel_epsilon_, segment_length_epsilon_, degenerate_axis_distance_, capsule_publish_rate_}) {
    if (!std::isfinite(value)) {fail("distances, gains, tolerances and rate must be finite");}
  }
  if (safe_distance_ < 0 || activation_distance_ < safe_distance_ || gain_ < 0 ||
    max_repulsive_velocity_ < 0) {fail("require activation_distance >= safe_distance >= 0 and nonnegative gains/velocity");}
  if (release_distance_ <= activation_distance_ || eps_ <= 0 ||
    eps_ >= release_distance_ - activation_distance_) {
    fail("require release_distance > activation_distance and 0 < eps < release_distance - activation_distance");
  }
  if (parallel_epsilon_ <= 0 || parallel_epsilon_ >= 1 || segment_length_epsilon_ <= 0 ||
    degenerate_axis_distance_ <= 0) {fail("invalid geometry tolerance (parallel_epsilon must be between 0 and 1)");}
  if (publish_capsules() && capsule_publish_rate_ <= 0) {fail("capsule_publish_rate must be positive when enabled");}
  capsules_.clear(); pairs_.clear(); planes_.clear(); plane_pairs_.clear();
  frames_.clear(); endpoints_.clear();
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
  const auto plane_names = get_string_array_param(params, "plane_names", {});
  std::set<std::string> unique_planes;
  const std::set<std::string> plane_fields{
    "reference_frame", "normal_axis", "position", "allowed_side",
    "bounds_min", "bounds_max", "checked_capsules"};
  for (const auto & name : plane_names) {
    if (name.empty() || !std::all_of(name.begin(), name.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '_';
      }) || !unique_planes.insert(name).second) {fail("invalid or duplicate plane name: " + name);}
    const std::string prefix = "planes." + name + ".";
    for (const auto & field : {"normal_axis", "position", "allowed_side", "bounds_min", "bounds_max"}) {
      if (!params.count(prefix + field)) {fail("missing parameter " + prefix + field);}
    }
    ManualPlane plane;
    plane.name = name;
    plane.reference_frame = get_string_param(params, prefix + "reference_frame", context.model->base_frame());
    // ROS parameter declaration uses an empty default to avoid any robot-specific frame.
    if (plane.reference_frame.empty()) {plane.reference_frame = context.model->base_frame();}
    const auto axis = get_string_param(params, prefix + "normal_axis", "");
    if (axis != "x" && axis != "y" && axis != "z") {fail("normal_axis must be x, y or z for " + name);}
    plane.normal_axis = axis == "x" ? 0 : (axis == "y" ? 1 : 2);
    const auto side = get_string_param(params, prefix + "allowed_side", "");
    if (side != "positive" && side != "negative") {fail("allowed_side must be positive or negative for " + name);}
    plane.allowed_sign = side == "positive" ? 1.0 : -1.0;
    plane.position = get_double_param(params, prefix + "position", 0);
    const auto minimum = get_double_array_param(params, prefix + "bounds_min", {});
    const auto maximum = get_double_array_param(params, prefix + "bounds_max", {});
    if (minimum.size() != 2 || maximum.size() != 2) {fail("plane bounds must each contain two doubles for " + name);}
    plane.bounds_min = Eigen::Vector2d(minimum[0], minimum[1]);
    plane.bounds_max = Eigen::Vector2d(maximum[0], maximum[1]);
    if (!std::isfinite(plane.position) || !plane.bounds_min.allFinite() ||
      !plane.bounds_max.allFinite() || (plane.bounds_min.array() >= plane.bounds_max.array()).any() ||
      !(plane.bounds_max - plane.bounds_min).allFinite()) {fail("invalid finite rectangle bounds/position for " + name);}
    plane.reference_index = add_frame(plane.reference_frame);
    const size_t plane_index = planes_.size();
    planes_.push_back(std::move(plane));
    const auto selected = get_string_array_param(params, prefix + "checked_capsules", {});
    if (selected.empty()) {
      for (size_t i = 0; i < capsules_.size(); ++i) {plane_pairs_.push_back({i, plane_index});}
    } else {
      std::set<std::string> selected_unique;
      for (const auto & capsule_name : selected) {
        const auto it = std::find(names.begin(), names.end(), capsule_name);
        if (it == names.end() || !selected_unique.insert(capsule_name).second) {
          fail("unknown or duplicate checked capsule for " + name + ": " + capsule_name);
        }
        plane_pairs_.push_back({static_cast<size_t>(it - names.begin()), plane_index});
      }
    }
  }
  for (const auto & parameter : params) {
    if (parameter.first.rfind("planes.", 0) != 0) {continue;}
    const auto dot = parameter.first.find('.', 7);
    const auto name = parameter.first.substr(7, dot - 7);
    const auto field = dot == std::string::npos ? std::string{} : parameter.first.substr(dot + 1);
    if (!unique_planes.count(name) || !plane_fields.count(field)) {
      fail("unknown plane definition field: " + parameter.first);
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
  const auto rows = static_cast<Eigen::Index>(pairs_.size() + plane_pairs_.size());
  jacobian_.setZero(rows, dofs);
  point_difference_.setZero(3, dofs);
  desired_.setZero(rows); errors_.setZero(rows);
  configure_pair_columns(*kinematics);
  pair_active_.assign(static_cast<size_t>(rows), 0);
  active_joints_.assign(context.model->total_dofs() - context.model->base_dofs(), 0);
  snapshots_.initialize([&](CollisionSnapshot & snapshot) {
      snapshot.capsules.resize(capsules_.size()); snapshot.planes.resize(planes_.size());
    });
  last_ = CollisionMetrics{};
}

void SelfCollisionAvoidanceTask::configure_pair_columns(const KinematicsBackend & backend)
{
  const size_t joint_count = context_.model->all_joint_names().size();
  const size_t base_dofs = context_.model->base_dofs();
  std::vector<std::vector<size_t>> dependencies(frames_.size());
  std::vector<bool> certified(frames_.size());
  for (size_t i = 0; i < frames_.size(); ++i) {
    certified[i] = backend.frame_joint_dependencies(frames_[i], dependencies[i]);
    for (const size_t joint : dependencies[i]) {
      if (joint >= joint_count) {
        throw std::invalid_argument("Task '" + id_ + "': out-of-model joint dependency for " + frames_[i]);
      }
    }
    std::sort(dependencies[i].begin(), dependencies[i].end());
    dependencies[i].erase(std::unique(dependencies[i].begin(), dependencies[i].end()), dependencies[i].end());
  }
  pair_columns_.clear();
  pair_columns_.reserve(pairs_.size() + plane_pairs_.size());
  const auto append = [&](const std::vector<size_t> & frames) {
      bool supported = true;
      std::vector<size_t> occurrences(joint_count, 0);
      for (const size_t frame : frames) {
        supported = supported && certified[frame];
        for (const size_t joint : dependencies[frame]) {++occurrences[joint];}
      }
      pair_columns_.emplace_back();
      auto & columns = pair_columns_.back();
      for (size_t joint = 0; joint < joint_count; ++joint) {
        // Shared ancestors move every endpoint/reference rigidly together.
        if (!supported || (occurrences[joint] > 0 && occurrences[joint] < frames.size())) {
          columns.push_back(static_cast<Eigen::Index>(base_dofs + joint));
        }
      }
    };
  for (const auto & pair : pairs_) {
    const auto & a = capsules_[pair[0]];
    const auto & b = capsules_[pair[1]];
    append({a.start_index, a.end_index, b.start_index, b.end_index});
  }
  for (const auto & pair : plane_pairs_) {
    const auto & capsule = capsules_[pair[0]];
    append({capsule.start_index, capsule.end_index, planes_[pair[1]].reference_index});
  }
}

bool SelfCollisionAvoidanceTask::update_pair_activation(size_t row, double clearance)
{
  if (pair_active_[row]) {
    if (clearance >= release_distance_ - eps_) {pair_active_[row] = 0;}
  } else if (clearance < activation_distance_) {
    pair_active_[row] = 1;
  }
  return pair_active_[row] != 0;
}

bool SelfCollisionAvoidanceTask::set_enabled(bool enabled)
{
  const bool result = TaskBaseCommon::set_enabled(enabled);
  reset();
  return result;
}

void SelfCollisionAvoidanceTask::reset()
{
  std::fill(pair_active_.begin(), pair_active_.end(), 0);
  std::fill(active_joints_.begin(), active_joints_.end(), 0);
  jacobian_.setZero(); desired_.setZero(); errors_.setZero();
  last_ = CollisionMetrics{};
  for (auto & capsule : snapshots_.writable().capsules) {capsule = CapsulePose{};}
  for (auto & plane : snapshots_.writable().planes) {plane = PlanePose{};}
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
  std::fill(pair_active_.begin(), pair_active_.end(), 0);
  last_.active_pairs = 0; last_.active_joint_count = 0; last_.zero_jacobian_pairs = 0;
  last_.invalid_geometry = true; last_.stop_arms = true;
  jacobian_.setZero(); desired_.setZero(); errors_.setZero();
  out.jacobian.setZero(); out.desired_velocity.setZero(); out.error.setZero();
  out.active = false; out.stop_arm_motion = true;
  out.status_message = "invalid_collision_geometry";
  // Never display stale/partial capsule locations as valid geometry.
  for (auto & capsule : snapshots_.writable().capsules) {capsule.valid = false;}
  for (auto & plane : snapshots_.writable().planes) {plane.valid = false;}
}

void SelfCollisionAvoidanceTask::update_into(
  const WholeBodyState &, const KinematicsBackend & backend, TaskComputation & out)
{
  last_ = CollisionMetrics{}; last_.enabled = enabled_;
  jacobian_.setZero(); desired_.setZero(); errors_.setZero();
  out.active = false; out.stop_arm_motion = false;
  out.jacobian.setZero(); out.desired_velocity.setZero(); out.error.setZero();
  for (auto & capsule : snapshots_.writable().capsules) {capsule = CapsulePose{};}
  for (auto & plane : snapshots_.writable().planes) {plane = PlanePose{};}
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
  for (size_t i = 0; i < planes_.size(); ++i) {
    const auto & plane = planes_[i];
    const auto & reference = endpoints_[plane.reference_index].world.pose;
    const Eigen::Matrix3d orientation = rotation * reference.rotation();
    const auto axes = plane_tangent_axes(plane.normal_axis);
    auto & pose = snapshot.planes[i];
    for (size_t corner = 0; corner < pose.corners.size(); ++corner) {
      Eigen::Vector3d local = Eigen::Vector3d::Zero();
      local(plane.normal_axis) = plane.position;
      local(axes[0]) = (corner == 1 || corner == 2) ? plane.bounds_max(0) : plane.bounds_min(0);
      local(axes[1]) = corner >= 2 ? plane.bounds_max(1) : plane.bounds_min(1);
      pose.corners[corner] = endpoints_[plane.reference_index].position + orientation * local;
      if (!pose.corners[corner].allFinite()) {invalid(out); return;}
    }
    pose.valid = true;
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
      pair_active_[i] = 0;
      ++last_.degenerate_pairs; last_.stop_arms = true;
      if (last_.fault_pair == std::numeric_limits<size_t>::max()) {last_.fault_pair = i;}
      snapshot.capsules[a].highlighted = snapshot.capsules[b].highlighted = true;
      continue;
    }
    if (!update_pair_activation(i, distance.clearance)) {continue;}
    const Eigen::Vector3d normal = (distance.q - distance.p) / distance.axis_distance;
    // Closest points can interpolate endpoints on different rigid links.
    point_difference_.noalias() = (1 - distance.t) * endpoints_[cb.start_index].jacobian;
    point_difference_.noalias() += distance.t * endpoints_[cb.end_index].jacobian;
    point_difference_.noalias() -= (1 - distance.s) * endpoints_[ca.start_index].jacobian;
    point_difference_.noalias() -= distance.s * endpoints_[ca.end_index].jacobian;
    // Scalar columns avoid temporary packing for a strided Eigen matrix row.
    for (const Eigen::Index col : pair_columns_[i]) {
      jacobian_(static_cast<Eigen::Index>(i), col) = normal.dot(point_difference_.col(col));
    }
    desired_(i) = std::clamp(gain_ * (release_distance_ - distance.clearance), 0.0, max_repulsive_velocity_);
    errors_(i) = std::max(0.0, safe_distance_ - distance.clearance);
    ++last_.active_pairs;
    snapshot.capsules[a].highlighted = snapshot.capsules[b].highlighted = true;
  }
  for (size_t i = 0; i < plane_pairs_.size(); ++i) {
    const auto capsule_index = plane_pairs_[i][0], plane_index = plane_pairs_[i][1];
    const auto & capsule = capsules_[capsule_index];
    const auto & plane = planes_[plane_index];
    const auto & reference = endpoints_[plane.reference_index].world;
    const auto & start = endpoints_[capsule.start_index].world;
    const auto & end = endpoints_[capsule.end_index].world;
    const Eigen::Matrix3d to_local = reference.pose.rotation().transpose();
    const Eigen::Vector3d ra = start.pose.translation() - reference.pose.translation();
    const Eigen::Vector3d rb = end.pose.translation() - reference.pose.translation();
    const auto distance = capsule_plane_distance(to_local * ra, to_local * rb, capsule.radius,
      plane.normal_axis, plane.allowed_sign, plane.position, plane.bounds_min, plane.bounds_max);
    const auto row = static_cast<Eigen::Index>(pairs_.size() + i);
    if (!distance.valid) {last_.fault_pair = static_cast<size_t>(row); invalid(out); return;}
    if (!distance.intersects) {pair_active_[static_cast<size_t>(row)] = 0; continue;}
    if (distance.clearance < last_.min_clearance) {
      last_.min_clearance = distance.clearance; last_.closest_pair = static_cast<size_t>(row);
    }
    if (!update_pair_activation(static_cast<size_t>(row), distance.clearance)) {continue;}
    for (const Eigen::Index col : pair_columns_[static_cast<size_t>(row)]) {
      // Relative point derivatives include rotation of the plane's reference frame.
      const Eigen::Vector3d ja = to_local * (start.jacobian.col(col).head<3>() -
        reference.jacobian.col(col).head<3>() + ra.cross(reference.jacobian.col(col).tail<3>()));
      const Eigen::Vector3d jb = to_local * (end.jacobian.col(col).head<3>() -
        reference.jacobian.col(col).head<3>() + rb.cross(reference.jacobian.col(col).tail<3>()));
      jacobian_(row, col) = distance.gradient_a.dot(ja) + distance.gradient_b.dot(jb);
    }
    desired_(row) = std::clamp(gain_ * (release_distance_ - distance.clearance), 0.0, max_repulsive_velocity_);
    errors_(row) = std::max(0.0, safe_distance_ - distance.clearance);
    ++last_.active_pairs;
    snapshot.capsules[capsule_index].highlighted = true;
    snapshot.planes[plane_index].highlighted = true;
  }
  if (!jacobian_.allFinite() || !desired_.allFinite() || !errors_.allFinite()) {invalid(out); return;}
  std::fill(active_joints_.begin(), active_joints_.end(), 0);
  for (size_t row = 0; row < pair_active_.size(); ++row) {
    if (!pair_active_[row]) {continue;}
    bool nonzero = false;
    for (const Eigen::Index col : pair_columns_[row]) {
      if (jacobian_(static_cast<Eigen::Index>(row), col) != 0.0) {
        nonzero = true;
        active_joints_[static_cast<size_t>(col) - context_.model->base_dofs()] = 1;
      }
    }
    if (!nonzero) {++last_.zero_jacobian_pairs;}
  }
  last_.active_joint_count = static_cast<size_t>(std::count(active_joints_.begin(), active_joints_.end(), 1));
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

std::string SelfCollisionAvoidanceTask::collision_pair_name(size_t row) const
{
  if (row < pairs_.size()) {
    const auto & pair = pairs_[row];
    return capsules_[pair[0]].name + ',' + capsules_[pair[1]].name;
  }
  row -= pairs_.size();
  if (row < plane_pairs_.size()) {
    const auto & pair = plane_pairs_[row];
    return capsules_[pair[0]].name + ",plane:" + planes_[pair[1]].name;
  }
  return "unknown";
}

msg::TaskStatus SelfCollisionAvoidanceTask::build_status() const
{
  auto status = TaskBaseCommon::build_status();
  status.active = enabled_ && (last_.active_pairs > 0 || last_.stop_arms);
  std::ostringstream text;
  text << "capsules=" << capsules_.size() << " checked_pairs=" << pairs_.size()
       << " planes=" << planes_.size() << " checked_plane_pairs=" << plane_pairs_.size()
       << " ignored_pairs=" << ignored_pair_count_ << " min_clearance=";
  if (std::isfinite(last_.min_clearance)) {text << last_.min_clearance;} else {text << "unknown";}
  if (last_.closest_pair < pairs_.size() + plane_pairs_.size()) {
    text << " closest_pair=" << collision_pair_name(last_.closest_pair);
  }
  text << " active_pairs=" << last_.active_pairs << " degenerate_pairs=" << last_.degenerate_pairs
       << " active_joint_count=" << last_.active_joint_count << " zero_jacobian_pairs=" << last_.zero_jacobian_pairs
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
