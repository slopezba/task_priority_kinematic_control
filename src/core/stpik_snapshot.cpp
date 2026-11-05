#include "task_priority_kinematic_control/core/stpik_check.hpp"
#include "task_priority_kinematic_control/tasks/frame_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effector_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effector_position_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effector_orientation_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effectors_relative_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_nominal_task.hpp"
#include "task_priority_kinematic_control/tasks/base_yaw_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/stabilize_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/self_collision_avoidance_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_trajectory_task.hpp"
#include "task_priority_kinematic_control/tasks/base_pose_trajectory_task.hpp"

#include <iomanip>
#include <sstream>
#include <typeinfo>
#include <limits>
#include <cmath>
#include <algorithm>

namespace task_priority_kinematic_control
{
namespace
{
template<class T> void append(std::ostream & out, const T & x) {out << x << ';';}
void append(std::ostream & out, const std::string & x) {out << x.size() << ':' << x << ';';}
void append(std::ostream &, const ManualCapsule &);
void append(std::ostream &, const ManualPlane &);
template<class T, size_t N> void append(std::ostream & out, const std::array<T, N> & xs);
template<class T> void append(std::ostream & out, const std::vector<T> & xs)
{out << xs.size() << '['; for (const auto & x : xs) {append(out, static_cast<T>(x));} out << ']';}
template<class T, size_t N> void append(std::ostream & out, const std::array<T, N> & xs)
{for (const auto & x : xs) {append(out, x);}}
void append(std::ostream & out, const Eigen::Quaterniond & q) {out << q.coeffs().transpose() << ';';}
void append(std::ostream & out, const Eigen::Isometry3d & p) {out << p.matrix() << ';';}
void append(std::ostream & out, const std::chrono::steady_clock::time_point & t)
{out << t.time_since_epoch().count() << ';';}
void append(std::ostream & out, const geometry_msgs::msg::Pose & p)
{out << p.position.x << ',' << p.position.y << ',' << p.position.z << ',' << p.orientation.x
     << ',' << p.orientation.y << ',' << p.orientation.z << ',' << p.orientation.w << ';';}
void append(std::ostream & out, const ManualCapsule & c)
{append(out, c.name); append(out, c.start_frame); append(out, c.end_frame); append(out, c.radius);}
void append(std::ostream & out, const ManualPlane & p)
{append(out, p.name); append(out, p.reference_frame); append(out, p.normal_axis);
 append(out, p.allowed_sign); append(out, p.position); append(out, p.bounds_min); append(out, p.bounds_max);}
Eigen::Matrix3d rotation(const WholeBodyState & s)
{return (Eigen::AngleAxisd(s.base_rpy.z(), Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(s.base_rpy.y(), Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(s.base_rpy.x(), Eigen::Vector3d::UnitX())).toRotationMatrix();}
}

std::shared_ptr<TaskBase> StpikTaskAccess::copy(const TaskBase & task, std::string & error)
{
  if (typeid(task) == typeid(FramePoseTask)) {
    const auto & source = static_cast<const FramePoseTask &>(task);
    auto out = std::make_shared<FramePoseTask>(source);
    return out;
  }
  if (typeid(task) == typeid(EndEffectorPoseTask)) {
    const auto & source = static_cast<const EndEffectorPoseTask &>(task);
    auto out = std::make_shared<EndEffectorPoseTask>(source);
    return out;
  }
  if (typeid(task) == typeid(EndEffectorPositionTask)) {
    const auto & source = static_cast<const EndEffectorPositionTask &>(task);
    auto out = std::make_shared<EndEffectorPositionTask>(source);
    return out;
  }
  if (typeid(task) == typeid(EndEffectorOrientationTask)) {
    const auto & source = static_cast<const EndEffectorOrientationTask &>(task);
    auto out = std::make_shared<EndEffectorOrientationTask>(source);
    return out;
  }
  if (typeid(task) == typeid(EndEffectorsRelativePoseTask)) {
    const auto & source = static_cast<const EndEffectorsRelativePoseTask &>(task);
    auto out = std::make_shared<EndEffectorsRelativePoseTask>(source);
    return out;
  }
  if (typeid(task) == typeid(JointNominalTask)) {
    const auto & source = static_cast<const JointNominalTask &>(task);
    auto out = std::make_shared<JointNominalTask>(source);
    return out;
  }
  if (typeid(task) == typeid(BaseYawTask)) {
    const auto & source = static_cast<const BaseYawTask &>(task);
    auto out = std::make_shared<BaseYawTask>(source);
    return out;
  }
  if (typeid(task) == typeid(JointLimitsTask)) {
    const auto & source = static_cast<const JointLimitsTask &>(task);
    auto out = std::make_shared<JointLimitsTask>();
    static_cast<TaskBaseCommon &>(*out) = static_cast<const TaskBaseCommon &>(source);
    out->alpha_ = source.alpha_;
    out->delta_ = source.delta_;
    out->eps_ = source.eps_;
    out->gain_ = source.gain_;
    out->lower_limits_ = source.lower_limits_;
    out->upper_limits_ = source.upper_limits_;
    out->lower_active_ = source.lower_active_;
    out->upper_active_ = source.upper_active_;
    out->joint_names_ = source.joint_names_;
    out->joint_frames_ = source.joint_frames_;
    out->joint_count_ = source.joint_count_;
    out->dofs_ = source.dofs_;
    out->base_dofs_ = source.base_dofs_;
    out->limits_marker_publish_rate_ = source.limits_marker_publish_rate_;
    return out;
  }
  if (typeid(task) == typeid(StabilizeLimitsTask)) {
    const auto & source = static_cast<const StabilizeLimitsTask &>(task);
    auto out = std::make_shared<StabilizeLimitsTask>();
    static_cast<TaskBaseCommon &>(*out) = static_cast<const TaskBaseCommon &>(source);
    out->lower_limits_ = source.lower_limits_;
    out->upper_limits_ = source.upper_limits_;
    out->lower_active_ = source.lower_active_;
    out->upper_active_ = source.upper_active_;
    out->angular_columns_ = source.angular_columns_;
    out->dofs_ = source.dofs_;
    out->alpha_ = source.alpha_;
    out->delta_ = source.delta_;
    out->eps_ = source.eps_;
    out->gain_ = source.gain_;
    out->max_repulsive_velocity_ = source.max_repulsive_velocity_;
    return out;
  }
  if (typeid(task) == typeid(SelfCollisionAvoidanceTask)) {
    const auto & source = static_cast<const SelfCollisionAvoidanceTask &>(task);
    auto out = std::make_shared<SelfCollisionAvoidanceTask>();
    static_cast<TaskBaseCommon &>(*out) = static_cast<const TaskBaseCommon &>(source);
    out->capsules_ = source.capsules_;
    out->pairs_ = source.pairs_;
    out->planes_ = source.planes_;
    out->plane_pairs_ = source.plane_pairs_;
    out->frames_ = source.frames_;
    out->endpoints_ = source.endpoints_;
    out->pair_columns_ = source.pair_columns_;
    out->pair_active_ = source.pair_active_;
    out->active_joints_ = source.active_joints_;
    out->ignored_pair_count_ = source.ignored_pair_count_;
    out->safe_distance_ = source.safe_distance_;
    out->activation_distance_ = source.activation_distance_;
    out->release_distance_ = source.release_distance_;
    out->eps_ = source.eps_;
    out->gain_ = source.gain_;
    out->max_repulsive_velocity_ = source.max_repulsive_velocity_;
    out->parallel_epsilon_ = source.parallel_epsilon_;
    out->segment_length_epsilon_ = source.segment_length_epsilon_;
    out->degenerate_axis_distance_ = source.degenerate_axis_distance_;
    out->capsule_publish_rate_ = source.capsule_publish_rate_;
    out->jacobian_ = source.jacobian_;
    out->point_difference_ = source.point_difference_;
    out->desired_ = source.desired_;
    out->errors_ = source.errors_;
    out->last_ = source.last_;
    out->snapshots_.initialize([&](CollisionSnapshot & snapshot) {
      snapshot.capsules.resize(out->capsules_.size()); snapshot.planes.resize(out->planes_.size());
    });
    return out;
  }
  if (typeid(task) == typeid(JointTrajectoryTask)) {
    const auto & source = static_cast<const JointTrajectoryTask &>(task);
    std::scoped_lock lock(source.mutex_);
    if (source.enabled_ && source.trajectory_active_ && !source.succeeded_) {
      error = "ACTIVE_TASK_BUSY: " + source.id_; return {};
    }
    auto out = std::make_shared<JointTrajectoryTask>();
    static_cast<TaskBaseCommon &>(*out) = static_cast<const TaskBaseCommon &>(source);
    out->joint_names_ = source.joint_names_;
    out->gains_ = source.gains_;
    out->goal_tolerance_ = source.goal_tolerance_;
    out->trajectory_timeout_ = source.trajectory_timeout_;
    out->hold_last_point_ = source.hold_last_point_;
    out->interpolation_method_ = source.interpolation_method_;
    out->samples_ = source.samples_;
    out->start_time_ = source.start_time_;
    out->start_time_ros_ns_ = source.start_time_ros_ns_;
    out->last_ros_time_ns_ = source.last_ros_time_ns_;
    out->use_ros_time_ = source.use_ros_time_;
    out->ros_cycle_seen_ = source.ros_cycle_seen_;
    out->configured_goal_tolerance_ = source.configured_goal_tolerance_;
    out->configured_trajectory_timeout_ = source.configured_trajectory_timeout_;
    out->configured_interpolation_method_ = source.configured_interpolation_method_;
    out->configured_hold_last_point_ = source.configured_hold_last_point_;
    out->current_target_ = source.current_target_;
    out->last_error_ = source.last_error_;
    out->final_error_norm_ = source.final_error_norm_;
    out->trajectory_active_ = source.trajectory_active_;
    out->succeeded_ = source.succeeded_;
    out->canceled_ = source.canceled_;
    out->timed_out_ = source.timed_out_;
    out->status_message_ = source.status_message_;
    return out;
  }
  if (typeid(task) == typeid(BasePoseTrajectoryTask)) {
    const auto & source = static_cast<const BasePoseTrajectoryTask &>(task);
    std::scoped_lock lock(source.mutex_);
    if (source.enabled_ && source.installed_ && !source.holding_ && !source.succeeded_ && !source.failed_) {
      error = "ACTIVE_TASK_BUSY: " + source.id_; return {};
    }
    auto out = std::make_shared<BasePoseTrajectoryTask>();
    static_cast<TaskBaseCommon &>(*out) = static_cast<const TaskBaseCommon &>(source);
    out->trajectory_ = source.trajectory_;
    out->knots_ = source.knots_;
    out->gains_ = source.gains_;
    out->tolerance_ = source.tolerance_;
    out->target_ = source.target_;
    out->last_error_ = source.last_error_;
    out->max_velocity_ = source.max_velocity_;
    out->max_acceleration_ = source.max_acceleration_;
    out->previous_command_ = source.previous_command_;
    out->command_time_ns_ = source.command_time_ns_;
    out->start_ns_ = source.start_ns_;
    out->previous_ns_ = source.previous_ns_;
    out->timeout_ = source.timeout_;
    out->installed_ = source.installed_;
    out->succeeded_ = source.succeeded_;
    out->failed_ = source.failed_;
    out->holding_ = source.holding_;
    out->ros_cycle_seen_ = source.ros_cycle_seen_;
    out->message_ = source.message_;
    return out;
  }
  if (task.enabled()) {error = "UNSUPPORTED_ACTIVE_TASK: " + task.id() + " (" + task.plugin_name() + ")";}
  return {};
}

void StpikTaskAccess::bind(TaskBase & task, const TaskContext & context)
{
  auto * common = dynamic_cast<TaskBaseCommon *>(&task);
  if (!common) {throw std::runtime_error("Unsupported isolated task context");}
  common->context_ = context;
}

std::string StpikTaskAccess::key(const TaskBase & task)
{
  std::ostringstream out;
  out << std::setprecision(17);
  append(out, task.id()); append(out, task.plugin_name()); append(out, task.priority());
  append(out, task.enabled()); append(out, task.group());
  // Disabled tasks do not contribute to the hierarchy; enabling one changes this key.
  if (!task.enabled()) {return out.str();}
  const auto * frame = dynamic_cast<const FramePoseTask *>(&task);
  // Roll/pitch control derives its display pose from measured position/yaw each
  // cycle. Its actual reference is tilt_target_, captured separately below.
  if (const auto * common = dynamic_cast<const TaskBaseCommon *>(&task);
    common && !(frame && frame->roll_pitch_control_)) {
    append(out, common->has_pose_goal_); append(out, common->pose_goal_.header.frame_id);
    if (common->has_pose_goal_) {append(out, common->pose_goal_.pose);}
  }
  if (const auto * pose = dynamic_cast<const PoseTaskCommon *>(&task)) {
    std::scoped_lock lock(pose->feedforward_mutex_);
    append(out, pose->feedforward_); append(out, pose->feedforward_epoch_);
    append(out, pose->accept_feedforward_); append(out, pose->auto_update_target_pose_);
    append(out, pose->feedforward_velocity_threshold_); append(out, pose->initialize_target_);
    append(out, pose->feedforward_timeout_);
    if (pose->feedforward_.norm() > 0.0) {append(out, pose->feedforward_time_);}
  }
  if (typeid(task) == typeid(FramePoseTask)) {
    const auto & source = static_cast<const FramePoseTask &>(task);
    append(out, source.frame_id_);
    append(out, source.activation_);
    append(out, source.roll_pitch_control_);
    append(out, source.allow_roll_pitch_);
    append(out, source.tilt_initialized_);
    append(out, source.tilt_moving_);
    append(out, source.tilt_target_);
    append(out, source.tilt_threshold_);
    append(out, source.gains_);
  }
  if (typeid(task) == typeid(EndEffectorPoseTask)) {
    const auto & source = static_cast<const EndEffectorPoseTask &>(task);
    append(out, source.frame_id_);
    append(out, source.gains_);
  }
  if (typeid(task) == typeid(EndEffectorPositionTask)) {
    const auto & source = static_cast<const EndEffectorPositionTask &>(task);
    append(out, source.frame_id_);
    append(out, source.default_goal_);
    append(out, source.gains_);
  }
  if (typeid(task) == typeid(EndEffectorOrientationTask)) {
    const auto & source = static_cast<const EndEffectorOrientationTask &>(task);
    append(out, source.frame_id_);
    append(out, source.gains_);
    append(out, source.default_goal_);
  }
  if (typeid(task) == typeid(EndEffectorsRelativePoseTask)) {
    const auto & source = static_cast<const EndEffectorsRelativePoseTask &>(task);
    append(out, source.reference_frame_);
    append(out, source.controlled_frame_);
    append(out, source.gains_);
    append(out, source.activation_);
    append(out, source.target_relative_pose_);
    append(out, source.has_target_relative_pose_);
  }
  if (typeid(task) == typeid(JointNominalTask)) {
    const auto & source = static_cast<const JointNominalTask &>(task);
    append(out, source.joint_names_);
    append(out, source.target_);
    append(out, source.gains_);
    append(out, source.joint_activation_);
    append(out, source.feedforward_);
    append(out, source.command_timeout_);
    if (source.feedforward_.norm() > 0.0) {append(out, source.command_time_);}
  }
  if (typeid(task) == typeid(BaseYawTask)) {
    const auto & source = static_cast<const BaseYawTask &>(task);
    append(out, source.target_yaw_);
    append(out, source.gain_);
  }
  if (typeid(task) == typeid(JointLimitsTask)) {
    const auto & source = static_cast<const JointLimitsTask &>(task);
    append(out, source.alpha_);
    append(out, source.delta_);
    append(out, source.eps_);
    append(out, source.gain_);
    append(out, source.lower_limits_);
    append(out, source.upper_limits_);
    append(out, source.lower_active_);
    append(out, source.upper_active_);
    append(out, source.joint_names_);
    append(out, source.joint_frames_);
    append(out, source.joint_count_);
    append(out, source.dofs_);
    append(out, source.base_dofs_);
    append(out, source.limits_marker_publish_rate_);
  }
  if (typeid(task) == typeid(StabilizeLimitsTask)) {
    const auto & source = static_cast<const StabilizeLimitsTask &>(task);
    append(out, source.lower_limits_);
    append(out, source.upper_limits_);
    append(out, source.lower_active_);
    append(out, source.upper_active_);
    append(out, source.angular_columns_);
    append(out, source.dofs_);
    append(out, source.alpha_);
    append(out, source.delta_);
    append(out, source.eps_);
    append(out, source.gain_);
    append(out, source.max_repulsive_velocity_);
  }
  if (typeid(task) == typeid(SelfCollisionAvoidanceTask)) {
    const auto & source = static_cast<const SelfCollisionAvoidanceTask &>(task);
    append(out, source.capsules_);
    append(out, source.pairs_);
    append(out, source.planes_);
    append(out, source.plane_pairs_);
    append(out, source.frames_);
    append(out, source.pair_columns_);
    append(out, source.pair_active_);
    append(out, source.safe_distance_);
    append(out, source.activation_distance_);
    append(out, source.release_distance_);
    append(out, source.eps_);
    append(out, source.gain_);
    append(out, source.max_repulsive_velocity_);
    append(out, source.parallel_epsilon_);
    append(out, source.segment_length_epsilon_);
    append(out, source.degenerate_axis_distance_);
  }
  if (typeid(task) == typeid(JointTrajectoryTask)) {
    const auto & source = static_cast<const JointTrajectoryTask &>(task);
    std::scoped_lock lock(source.mutex_);
    append(out, source.joint_names_);
    append(out, source.gains_);
    append(out, source.goal_tolerance_);
    append(out, source.trajectory_timeout_);
    append(out, source.hold_last_point_);
    append(out, source.interpolation_method_);
    append(out, source.start_time_);
    append(out, source.start_time_ros_ns_);
    append(out, source.use_ros_time_);
    append(out, source.trajectory_active_);
    append(out, source.succeeded_);
    append(out, source.canceled_);
    append(out, source.timed_out_);
    for (const auto & sample : source.samples_) {
      append(out, sample.position); append(out, sample.velocity); append(out, sample.acceleration);
      append(out, sample.has_velocity); append(out, sample.time_from_start);
    }
  }
  if (typeid(task) == typeid(BasePoseTrajectoryTask)) {
    const auto & source = static_cast<const BasePoseTrajectoryTask &>(task);
    std::scoped_lock lock(source.mutex_);
    append(out, source.gains_);
    append(out, source.tolerance_);
    append(out, source.target_);
    append(out, source.max_velocity_);
    append(out, source.max_acceleration_);
    append(out, source.start_ns_);
    append(out, source.timeout_);
    append(out, source.installed_);
    append(out, source.succeeded_);
    append(out, source.failed_);
    append(out, source.holding_);
    append(out, source.previous_command_);
  }
  return out.str();
}

std::string StpikTaskAccess::configuration(const WholeBodyModel & model,
  const HierarchySolver & solver, const std::vector<std::shared_ptr<TaskBase>> & tasks)
{
  std::ostringstream out;
  append(out, model.world_frame()); append(out, model.base_frame());
  append(out, model.left_tip_frame()); append(out, model.right_tip_frame());
  append(out, model.all_joint_names()); append(out, model.active_base_dofs());
  append(out, solver.configuration_key());
  for (const auto & task : tasks) {append(out, key(*task));}
  return out.str();
}

StpikSnapshot StpikTaskAccess::capture(const WholeBodyState & state,
  const std::shared_ptr<WholeBodyModel> & model, const KinematicsBackendPtr & backend,
  const HierarchySolver & solver, const std::vector<std::shared_ptr<TaskBase>> & tasks)
{
  StpikSnapshot out;
  out.initial_state = state;
  out.captured = std::chrono::steady_clock::now();
  if (!model || !backend) {out.unavailable_reason = "Kinematics/model unavailable"; return out;}
  out.model = std::make_shared<WholeBodyModel>(*model);
  out.source_backend = backend.get();
  out.solver = solver;
  out.configuration = configuration(*model, solver, tasks);
  out.backend = backend->clone();
  if (!out.backend || out.backend.get() == backend.get()) {
    out.unavailable_reason = "Backend does not support an independent offline clone"; return out;
  }
  for (const auto & task : tasks) {
    auto isolated = copy(*task, out.unavailable_reason);
    if (!out.unavailable_reason.empty()) {break;}
    if (isolated) {
      bind(*isolated, TaskContext{out.model, out.backend, {}, {}});
      out.tasks.push_back(std::move(isolated));
    }
  }
  return out;
}

bool StpikTaskAccess::compatible(const StpikSnapshot & snapshot, const WholeBodyState & state,
  const WholeBodyModel & model, const KinematicsBackend * backend, const HierarchySolver & solver,
  const std::vector<std::shared_ptr<TaskBase>> & tasks, const StpikOptions & options,
  std::string & reason)
{
  if (snapshot.source_backend != backend) {reason = "Kinematics backend changed since validation"; return false;}
  if (snapshot.configuration != configuration(model, solver, tasks)) {
    reason = "Effective task hierarchy, targets, gains, safety latches or solver changed"; return false;
  }
  // A finite external input lifetime cannot be shifted by time spent checking.
  // Once it has expired, a new snapshot simulates zero from the correct start.
  // Returning unavailable here avoids approving a simulation with an earlier expiry.
  for (const auto & task : tasks) {
    if (!task->enabled()) {continue;}
    if (const auto * pose = dynamic_cast<const PoseTaskCommon *>(task.get())) {
      std::scoped_lock lock(pose->feedforward_mutex_);
      if (pose->feedforward_timeout_ > 0.0 && pose->feedforward_.norm() > 0.0 &&
        std::chrono::duration<double>(snapshot.captured - pose->feedforward_time_).count() <
        pose->feedforward_timeout_)
      {reason = "A timed pose feedforward was live at snapshot capture; stop its sender and retry after expiry"; return false;}
    }
    if (const auto * nominal = dynamic_cast<const JointNominalTask *>(task.get())) {
      if (nominal->feedforward_.norm() > 0.0 &&
        std::chrono::duration<double>(snapshot.captured - nominal->command_time_).count() < nominal->command_timeout_)
      {reason = "A timed joint feedforward was live at snapshot capture; stop its sender and retry after expiry"; return false;}
    }
  }
  const auto & initial = snapshot.initial_state;
  if (!state.navigation_valid || !state.joints_valid || !state.base_position.allFinite() ||
    !state.base_rpy.allFinite() || !state.joint_positions.allFinite() ||
    initial.joint_positions.size() != state.joint_positions.size() ||
    (state.joint_positions.size() > 0 &&
    (initial.joint_positions - state.joint_positions).cwiseAbs().maxCoeff() > options.snapshot_joint_tolerance) ||
    (initial.base_position - state.base_position).norm() > options.snapshot_position_tolerance ||
    Eigen::AngleAxisd(rotation(initial) * rotation(state).transpose()).angle() > options.snapshot_orientation_tolerance)
  {reason = "Robot configuration changed since STPIK snapshot"; return false;}
  StpikAction::Result safety_result;
  for (const auto & task : tasks) {
    if (!safety(*task, state, options.numerical_tolerance, safety_result)) {
      reason = "Current safety state changed: " + safety_result.code + "/" + safety_result.constraint;
      return false;
    }
  }
  return true;
}

bool StpikTaskAccess::safety(const TaskBase & task, const WholeBodyState & state,
  double tolerance, StpikAction::Result & result)
{
  if (!task.enabled()) {return true;}
  result.task_id = task.id();
  if (const auto * frame = dynamic_cast<const FramePoseTask *>(&task)) {
    if (frame->roll_pitch_control_ && std::abs(std::cos(state.base_rpy.y())) < 1e-4) {
      result.code = "CONSTRAINT_SINGULARITY"; result.constraint = "roll_pitch_control"; return false;
    }
  }
  if (const auto * limits = dynamic_cast<const JointLimitsTask *>(&task)) {
    if (limits->lower_limits_.size() != state.joint_positions.size() ||
      limits->upper_limits_.size() != state.joint_positions.size())
    {result.code = "INVALID_JOINT_LIMITS"; result.constraint = "limits/state dimensions"; return false;}
    for (Eigen::Index i = 0; i < limits->lower_limits_.size(); ++i) {
      if (!std::isfinite(limits->lower_limits_(i)) || !std::isfinite(limits->upper_limits_(i)))
      {result.code = "INVALID_JOINT_LIMITS"; return false;}
      if (limits->upper_limits_(i) <= limits->lower_limits_(i)) {continue;}
      if (state.joint_positions(i) < limits->lower_limits_(i) - tolerance ||
        state.joint_positions(i) > limits->upper_limits_(i) + tolerance)
      {result.code = "JOINT_LIMIT"; result.constraint = limits->joint_names_.at(i); return false;}
    }
  }
  if (const auto * limits = dynamic_cast<const StabilizeLimitsTask *>(&task)) {
    for (Eigen::Index i = 0; i < 2; ++i) {
      if (state.base_rpy(i) < limits->lower_limits_(i) - tolerance ||
        state.base_rpy(i) > limits->upper_limits_(i) + tolerance)
      {result.code = "BASE_TILT_LIMIT"; result.constraint = i == 0 ? "roll" : "pitch"; return false;}
    }
  }
  if (const auto * collision = dynamic_cast<const SelfCollisionAvoidanceTask *>(&task)) {
    const auto & metric = collision->last_;
    result.minimum_clearance = std::min(result.minimum_clearance, metric.min_clearance);
    if (metric.invalid_geometry || metric.degenerate_pairs > 0 || metric.stop_arms) {
      result.code = "COLLISION_GEOMETRY";
      result.constraint = collision->collision_pair_name(metric.fault_pair);
      return false;
    }
    if (metric.min_clearance < collision->safe_distance_ - tolerance) {
      result.code = "COLLISION_CLEARANCE";
      result.constraint = collision->collision_pair_name(metric.closest_pair); return false;
    }
  }
  result.task_id.clear();
  return true;
}
}  // namespace task_priority_kinematic_control
