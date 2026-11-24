#include "task_priority_kinematic_control/core/stpik_check.hpp"
#include <type_traits>
#include "task_priority_kinematic_control/tasks/frame_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/frame_pose_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/frame_position_norm_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/frame_orientation_norm_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/arm_manipulability_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effector_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effector_position_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effector_orientation_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effectors_relative_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effectors_facing_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_nominal_task.hpp"
#include "task_priority_kinematic_control/tasks/robot_nominal_task.hpp"
#include "task_priority_kinematic_control/tasks/base_yaw_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/stabilize_limits_task.hpp"
#include "task_priority_kinematic_control/tasks/self_collision_avoidance_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_trajectory_task.hpp"
#include "task_priority_kinematic_control/tasks/base_pose_trajectory_task.hpp"

#include "task_priority_kinematic_control/geometry/capsule_geometry.hpp"
#include "task_priority_kinematic_control/geometry/plane_geometry.hpp"

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
  if (typeid(task) == typeid(FramePoseLimitsTask)) {
    const auto & source = static_cast<const FramePoseLimitsTask &>(task);
    return std::make_shared<FramePoseLimitsTask>(source);
  }
  if (typeid(task) == typeid(FramePositionNormLimitsTask)) {
    const auto & source = static_cast<const FramePositionNormLimitsTask &>(task);
    return std::make_shared<FramePositionNormLimitsTask>(source);
  }
  if (typeid(task) == typeid(FrameOrientationNormLimitsTask)) {
    const auto & source = static_cast<const FrameOrientationNormLimitsTask &>(task);
    return std::make_shared<FrameOrientationNormLimitsTask>(source);
  }
  if (typeid(task) == typeid(ArmManipulabilityTask)) {
    const auto & source = static_cast<const ArmManipulabilityTask &>(task);
    return std::make_shared<ArmManipulabilityTask>(source);
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
  if (typeid(task) == typeid(EndEffectorsFacingTask)) {
    const auto & source = static_cast<const EndEffectorsFacingTask &>(task);
    auto out = std::make_shared<EndEffectorsFacingTask>(source);
    return out;
  }
  if (typeid(task) == typeid(JointNominalTask)) {
    const auto & source = static_cast<const JointNominalTask &>(task);
    auto out = std::make_shared<JointNominalTask>(source);
    return out;
  }
  if (typeid(task) == typeid(RobotNominalTask)) {
    return std::make_shared<RobotNominalTask>(static_cast<const RobotNominalTask &>(task));
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
  if (typeid(task) == typeid(FramePoseLimitsTask)) {
    const auto & source = static_cast<const FramePoseLimitsTask &>(task);
    append(out, source.frame_id_);
    append(out, source.axes_);
    append(out, source.threshold_.transpose());
    append(out, source.alpha_.transpose());
    append(out, source.delta_.transpose());
    append(out, source.gains_.transpose());
    append(out, source.eps_);
    append(out, source.max_repulsive_velocity_);
    append(out, source.has_target_);
    if (source.has_target_) {
      append(out, source.target_position_.transpose());
      append(out, source.target_orientation_);
    }
    append(out, source.axis_active_);
    append(out, source.axis_side_);
  }
  auto append_norm_task = [&out](const auto & source) {
      append(out, source.frame_id_);
      append(out, source.axes_);
      append(out, source.limit_);
      append(out, source.alpha_);
      append(out, source.delta_);
      append(out, source.gain_);
      append(out, source.eps_);
      append(out, source.max_velocity_);
      append(out, source.always_active_);
      append(out, source.has_target_);
      if (source.has_target_) {
        append(out, source.target_position_.transpose());
        append(out, source.target_orientation_);
      }
      append(out, source.row_active_);
    };
  if (typeid(task) == typeid(FramePositionNormLimitsTask)) {
    append_norm_task(static_cast<const FramePositionNormLimitsTask &>(task));
  }
  if (typeid(task) == typeid(FrameOrientationNormLimitsTask)) {
    append_norm_task(static_cast<const FrameOrientationNormLimitsTask &>(task));
  }
  if (typeid(task) == typeid(ArmManipulabilityTask)) {
    const auto & source = static_cast<const ArmManipulabilityTask &>(task);
    append(out, source.frame_id_);
    append(out, source.joint_names_);
    append(out, source.w_min_);
    append(out, source.w_des_);
    append(out, source.gain_);
    append(out, source.max_velocity_);
    append(out, source.fd_step_);
    append(out, source.gradient_epsilon_);
    append(out, source.row_active_);
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
  if (typeid(task) == typeid(EndEffectorsFacingTask)) {
    const auto & source = static_cast<const EndEffectorsFacingTask &>(task);
    append(out, source.left_frame_);
    append(out, source.right_frame_);
    append(out, source.left_pointing_axis_.transpose());
    append(out, source.right_pointing_axis_.transpose());
    append(out, source.target_roll_);
    append(out, source.target_distance_);
    append(out, source.capture_roll_pending_);
    append(out, source.capture_distance_pending_);
    append(out, source.min_distance_);
    append(out, source.gains_.transpose());
    append(out, source.activation_.transpose());
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
  if (typeid(task) == typeid(RobotNominalTask)) {
    const auto & source = static_cast<const RobotNominalTask &>(task);
    append(out, source.joint_names_);
    append(out, source.sync_tasks_);
    append(out, source.gains_);
    append(out, source.activation_);
    append(out, source.timeout_);
    append(out, source.has_reference_);
    if (source.has_reference_) {
      append(out, source.base_target_); append(out, source.base_rates_);
      append(out, source.joint_target_); append(out, source.joint_velocity_);
      append(out, source.reference_time_);
    }
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
    if (const auto * robot = dynamic_cast<const RobotNominalTask *>(task.get())) {
      if (robot->has_reference_ &&
        std::chrono::duration<double>(snapshot.captured - robot->reference_time_).count() <= robot->timeout_)
      {reason = "A robot_nominal reference stream was live at snapshot capture; stop its sender and retry after expiry"; return false;}
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

bool StpikTaskAccess::joint_bounds(const std::vector<std::shared_ptr<TaskBase>> & tasks,
  Eigen::VectorXd & lower, Eigen::VectorXd & upper)
{
  for (const auto & task : tasks) {
    auto limits = std::dynamic_pointer_cast<JointLimitsTask>(task);
    if (limits && limits->enabled()) {
      lower = limits->lower_limits_; upper = limits->upper_limits_;
      return lower.size() == upper.size() && lower.allFinite() && upper.allFinite() &&
             (upper.array() > lower.array()).all();
    }
  }
  return false;
}

bool StpikTaskAccess::tilt_bounds(const std::vector<std::shared_ptr<TaskBase>> & tasks,
  Eigen::Vector2d & lower, Eigen::Vector2d & upper)
{
  for (const auto & task : tasks) {
    auto limits = std::dynamic_pointer_cast<StabilizeLimitsTask>(task);
    if (limits && limits->enabled() && limits->lower_limits_.size() >= 2 && limits->upper_limits_.size() >= 2) {
      lower = limits->lower_limits_.head<2>(); upper = limits->upper_limits_.head<2>();
      return lower.allFinite() && upper.allFinite() && (upper.array() > lower.array()).all();
    }
  }
  return false;
}

bool StpikTaskAccess::posture_region(const TaskBase & task, PostureRegion & region, bool linearize)
{
  if (const auto * pose = dynamic_cast<const FramePoseLimitsTask *>(&task)) {
    if (!pose->marker_valid_) {throw std::runtime_error("Unavailable pose region: " + task.id());}
    region.error.resize(pose->rows_);
    region.limit.resize(pose->rows_);
    if (linearize) {region.jacobian.resize(pose->rows_, pose->dofs_);}
    for (size_t axis = 0; axis < pose->axes_.size(); ++axis) {
      if (!pose->axes_[axis]) {continue;}
      const auto row = pose->axis_row_[axis];
      region.error(row) = pose->last_error_(axis);
      region.limit(row) = pose->threshold_(axis);
      if (linearize) {region.jacobian.row(row) = pose->local_jacobian_.row(axis);}
    }
    return true;
  }
  const auto norm_region = [&](const auto & norm, Eigen::Index offset) {
      if (!norm.marker_valid_) {throw std::runtime_error("Unavailable norm region: " + task.id());}
      region.radial = true;
      region.always_active = norm.always_active_;
      region.limit = Eigen::VectorXd::Constant(1, norm.limit_);
      if (!linearize) {
        region.error = Eigen::VectorXd::Constant(1, norm.last_norm_);
        return;
      }
      const Eigen::Matrix3d world_to_target = norm.target_orientation_.toRotationMatrix().transpose();
      if constexpr (std::is_same_v<std::decay_t<decltype(norm)>, FramePositionNormLimitsTask>) {
        region.error = world_to_target * (norm.target_position_ - norm.frame_.pose.translation());
      } else {
        region.error = norm.marker_error_;
      }
      region.jacobian = world_to_target * norm.frame_.jacobian.middleRows(offset, 3);
      for (size_t axis = 0; axis < norm.axes_.size(); ++axis) {
        if (norm.axes_[axis]) {continue;}
        region.error(axis) = 0.0;
        region.jacobian.row(axis).setZero();
      }
    };
  if (const auto * position = dynamic_cast<const FramePositionNormLimitsTask *>(&task)) {
    norm_region(*position, 0);
    return true;
  }
  if (const auto * orientation = dynamic_cast<const FrameOrientationNormLimitsTask *>(&task)) {
    norm_region(*orientation, 3);
    return true;
  }
  return false;
}

std::vector<PreventiveMargin> StpikTaskAccess::preventive_margins(
  const TaskBase & task, const WholeBodyState & state)
{
  std::vector<PreventiveMargin> out;
  if (!task.enabled()) {return out;}
  const auto add = [&](double distance, double activation, double release,
      const Eigen::RowVectorXd & gradient) {
      const double width = std::max(release - activation, 1e-6);
      out.push_back({(release - distance) / width, -gradient / width});
    };
  if (const auto * limits = dynamic_cast<const JointLimitsTask *>(&task)) {
    for (Eigen::Index j = 0; j < limits->lower_limits_.size(); ++j) {
      if (limits->upper_limits_(j) <= limits->lower_limits_(j)) {continue;}
      Eigen::RowVectorXd g = Eigen::RowVectorXd::Zero(limits->dofs_);
      g(limits->base_dofs_ + j) = 1.0;
      add(state.joint_positions(j) - limits->lower_limits_(j), limits->alpha_, limits->delta_, g);
      add(limits->upper_limits_(j) - state.joint_positions(j), limits->alpha_, limits->delta_, -g);
    }
  }
  if (const auto * limits = dynamic_cast<const StabilizeLimitsTask *>(&task)) {
    const double cp = std::cos(state.base_rpy.y()), yaw = state.base_rpy.z();
    if (std::abs(cp) < 1e-4) {return out;}  // Already rejected by safety().
    Eigen::Matrix2d map;
    map << std::cos(yaw) / cp, std::sin(yaw) / cp, -std::sin(yaw), std::cos(yaw);
    for (int axis = 0; axis < 2; ++axis) {
      Eigen::RowVectorXd g = Eigen::RowVectorXd::Zero(limits->dofs_);
      for (size_t c = 0; c < limits->angular_columns_.size(); ++c) {
        if (limits->angular_columns_[c] >= 0) {g(limits->angular_columns_[c]) = map(axis, c);}
      }
      add(state.base_rpy(axis) - limits->lower_limits_(axis), limits->alpha_, limits->delta_, g);
      add(limits->upper_limits_(axis) - state.base_rpy(axis), limits->alpha_, limits->delta_, -g);
    }
  }
  const auto norm_margin = [&](const auto & norm) {
      if (norm.always_active_ || !norm.marker_valid_) {return;}
      PostureRegion region;
      posture_region(task, region, true);
      const double sigma = region.error.norm();
      Eigen::RowVectorXd g = Eigen::RowVectorXd::Zero(region.jacobian.cols());
      if (sigma > 1e-9) {g = (region.error / sigma).transpose() * region.jacobian;}
      // d(limit - ||error||)/dt = direction * J.
      add(norm.limit_ - sigma, norm.alpha_, norm.delta_, g);
    };
  if (const auto * norm = dynamic_cast<const FramePositionNormLimitsTask *>(&task)) {norm_margin(*norm);}
  if (const auto * norm = dynamic_cast<const FrameOrientationNormLimitsTask *>(&task)) {norm_margin(*norm);}
  if (const auto * pose = dynamic_cast<const FramePoseLimitsTask *>(&task)) {
    if (pose->marker_valid_) {
      for (size_t axis = 0; axis < pose->axes_.size(); ++axis) {
        if (!pose->axes_[axis]) {continue;}
        const auto g = pose->local_jacobian_.row(axis).eval();
        add(pose->threshold_(axis) - pose->last_error_(axis), pose->alpha_(axis), pose->delta_(axis), g);
        add(pose->threshold_(axis) + pose->last_error_(axis), pose->alpha_(axis), pose->delta_(axis), -g);
      }
    }
  }
  if (const auto * collision = dynamic_cast<const SelfCollisionAvoidanceTask *>(&task)) {
    // Compute every pair, including inactive rows whose runtime Jacobian is zero.
    for (const auto & pair : collision->pairs_) {
      const auto & a = collision->capsules_[pair[0]], & b = collision->capsules_[pair[1]];
      const auto & a0 = collision->endpoints_[a.start_index], & a1 = collision->endpoints_[a.end_index];
      const auto & b0 = collision->endpoints_[b.start_index], & b1 = collision->endpoints_[b.end_index];
      const auto d = capsule_distance(a0.position, a1.position, b0.position, b1.position,
        a.radius, b.radius, collision->parallel_epsilon_, collision->segment_length_epsilon_);
      if (!d.valid || d.axis_distance <= collision->degenerate_axis_distance_) {continue;}
      const Eigen::RowVectorXd g = ((d.q - d.p) / d.axis_distance).transpose() *
        ((1 - d.t) * b0.jacobian + d.t * b1.jacobian - (1 - d.s) * a0.jacobian - d.s * a1.jacobian);
      add(d.clearance, collision->activation_distance_, collision->release_distance_, g);
    }
    for (const auto & pair : collision->plane_pairs_) {
      const auto & capsule = collision->capsules_[pair[0]];
      const auto & plane = collision->planes_[pair[1]];
      const auto & ref = collision->endpoints_[plane.reference_index].world;
      const auto & a = collision->endpoints_[capsule.start_index].world;
      const auto & b = collision->endpoints_[capsule.end_index].world;
      const Eigen::Matrix3d rotation = ref.pose.rotation().transpose();
      const Eigen::Vector3d ra = a.pose.translation() - ref.pose.translation();
      const Eigen::Vector3d rb = b.pose.translation() - ref.pose.translation();
      const auto d = capsule_plane_distance(rotation * ra, rotation * rb, capsule.radius,
        plane.normal_axis, plane.allowed_sign, plane.position, plane.bounds_min, plane.bounds_max);
      if (!d.valid || !d.intersects) {continue;}
      Eigen::RowVectorXd g = Eigen::RowVectorXd::Zero(collision->jacobian_.cols());
      for (Eigen::Index c = 0; c < g.size(); ++c) {
        const Eigen::Vector3d ja = rotation * (a.jacobian.col(c).head<3>() -
          ref.jacobian.col(c).head<3>() + ra.cross(ref.jacobian.col(c).tail<3>()));
        const Eigen::Vector3d jb = rotation * (b.jacobian.col(c).head<3>() -
          ref.jacobian.col(c).head<3>() + rb.cross(ref.jacobian.col(c).tail<3>()));
        g(c) = d.gradient_a.dot(ja) + d.gradient_b.dot(jb);
      }
      add(d.clearance, collision->activation_distance_, collision->release_distance_, g);
    }
  }
  return out;
}

double posture_limit_cost(
  double distance, double activation, double preventive, bool active, double active_penalty)
{
  if (active) {
    const double depth = std::clamp(
      (activation - distance) / std::max(activation, 1e-12), 0.0, 1.0);
    return active_penalty * (1.0 + depth * depth);
  }
  const double proximity = std::clamp(
    (preventive - distance) / std::max(preventive - activation, 1e-12), 0.0, 1.0);
  return proximity * proximity;
}

PostureSafetyCosts StpikTaskAccess::posture_safety_costs(
  const TaskBase & task, const WholeBodyState & state, double active_penalty)
{
  PostureSafetyCosts costs;
  if (!task.enabled()) {return costs;}
  const auto bounded_cost = [&](double value, double lower, double upper,
      double alpha, double delta, bool lower_active, bool upper_active) {
      return std::max(
        posture_limit_cost(value - lower, alpha, delta, lower_active, active_penalty),
        posture_limit_cost(upper - value, alpha, delta, upper_active, active_penalty));
    };
  if (const auto * limits = dynamic_cast<const JointLimitsTask *>(&task)) {
    for (Eigen::Index i = 0; i < limits->lower_limits_.size(); ++i) {
      if (limits->upper_limits_(i) <= limits->lower_limits_(i)) {continue;}
      costs.joints = std::max(costs.joints, bounded_cost(
        state.joint_positions(i), limits->lower_limits_(i), limits->upper_limits_(i),
        limits->alpha_, limits->delta_, limits->lower_active_[i], limits->upper_active_[i]));
    }
  }
  if (const auto * limits = dynamic_cast<const StabilizeLimitsTask *>(&task)) {
    for (Eigen::Index i = 0; i < 2; ++i) {
      costs.stabilize = std::max(costs.stabilize, bounded_cost(
        state.base_rpy(i), limits->lower_limits_(i), limits->upper_limits_(i),
        limits->alpha_, limits->delta_, limits->lower_active_[i], limits->upper_active_[i]));
    }
  }
  if (const auto * collision = dynamic_cast<const SelfCollisionAvoidanceTask *>(&task)) {
    costs.self_collision = posture_limit_cost(
      collision->last_.min_clearance - collision->safe_distance_,
      collision->activation_distance_ - collision->safe_distance_,
      collision->release_distance_ - collision->safe_distance_,
      collision->last_.active_pairs > 0, active_penalty);
  }
  return costs;
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
    const auto reject = [&](const std::string & code, size_t pair) {
        result.code = code;
        result.constraint = collision->collision_pair_name(pair);
        std::ostringstream detail;
        detail << "Collision task safety check failed; source=task; task=" << task.id()
               << "; pair=" << result.constraint
               << "; min_clearance=" << metric.min_clearance << " m"
               << "; safe_distance=" << collision->safe_distance_ << " m"
               << "; numerical_tolerance=" << tolerance
               << "; invalid_geometry=" << metric.invalid_geometry
               << "; degenerate_pairs=" << metric.degenerate_pairs
               << "; stop_arms=" << metric.stop_arms;
        result.message = detail.str();
        return false;
      };
    if (metric.invalid_geometry || metric.degenerate_pairs > 0 || metric.stop_arms) {
      return reject("COLLISION_GEOMETRY", metric.fault_pair);
    }
    if (metric.min_clearance < collision->safe_distance_ - tolerance) {
      return reject("COLLISION_CLEARANCE", metric.closest_pair);
    }
  }
  result.task_id.clear();
  return true;
}
}  // namespace task_priority_kinematic_control
