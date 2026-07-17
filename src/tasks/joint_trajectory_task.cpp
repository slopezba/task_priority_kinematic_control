#include "task_priority_kinematic_control/tasks/joint_trajectory_task.hpp"

#include <pluginlib/class_list_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace task_priority_kinematic_control
{
namespace
{
double duration_to_seconds(const builtin_interfaces::msg::Duration & duration)
{
  return static_cast<double>(duration.sec) + static_cast<double>(duration.nanosec) * 1.0e-9;
}

bool finite_vector(const std::vector<double> & values)
{
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value);
  });
}
}  // namespace

void JointTrajectoryTask::configure(
  const std::string & id,
  const std::string & plugin_name,
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const TaskContext & context)
{
  configure_common(id, plugin_name, parameters, context);
  joint_names_ = get_string_array_param(parameters, "joint_names", context.model->all_joint_names());
  const auto gains = get_double_array_param(parameters, "gain", std::vector<double>(joint_names_.size(), 1.0));
  const auto tolerance =
    get_double_array_param(parameters, "goal_tolerance", std::vector<double>(joint_names_.size(), 0.05));
  trajectory_timeout_ = get_double_param(parameters, "trajectory_timeout", 0.5);
  hold_last_point_ = get_bool_param(parameters, "hold_last_point", true);

  gains_ = Eigen::VectorXd::Constant(joint_names_.size(), 1.0);
  goal_tolerance_ = Eigen::VectorXd::Constant(joint_names_.size(), 0.05);
  current_target_ = Eigen::VectorXd::Zero(joint_names_.size());
  last_error_ = Eigen::VectorXd::Zero(joint_names_.size());
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    if (i < gains.size()) {
      gains_(static_cast<Eigen::Index>(i)) = gains[i];
    }
    if (i < tolerance.size()) {
      goal_tolerance_(static_cast<Eigen::Index>(i)) = tolerance[i];
    }
  }
}

TaskComputation JointTrajectoryTask::update(
  const WholeBodyState & state,
  const KinematicsBackend &)
{
  std::lock_guard<std::mutex> lock(mutex_);

  TaskComputation computation;
  computation.active = enabled_ && trajectory_active_ && state.joints_valid;
  if (!enabled_) {
    computation.status_message = "disabled";
    return computation;
  }
  if (!trajectory_active_) {
    computation.status_message = status_message_;
    return computation;
  }
  if (!state.joints_valid || state.joint_positions.size() == 0) {
    computation.status_message = "waiting_for_joints";
    return computation;
  }

  const auto elapsed_duration = SteadyClock::now() - start_time_;
  const double elapsed = std::chrono::duration<double>(elapsed_duration).count();
  const double duration = trajectory_duration_locked();
  if (elapsed > duration + trajectory_timeout_) {
    trajectory_active_ = false;
    timed_out_ = true;
    succeeded_ = false;
    status_message_ = "trajectory_timeout";
    computation.status_message = status_message_;
    return computation;
  }

  const auto sample = sample_at_locked(elapsed);
  current_target_ = sample.position;

  computation.jacobian = Eigen::MatrixXd::Zero(joint_names_.size(), context_.model->total_dofs());
  computation.error = Eigen::VectorXd::Zero(joint_names_.size());
  computation.desired_velocity = Eigen::VectorXd::Zero(joint_names_.size());
  for (size_t row = 0; row < joint_names_.size(); ++row) {
    const int joint_idx = context_.model->joint_index(joint_names_[row]);
    if (joint_idx < 0 || joint_idx >= state.joint_positions.size()) {
      continue;
    }
    const auto r = static_cast<Eigen::Index>(row);
    const Eigen::Index col = static_cast<Eigen::Index>(context_.model->base_dofs() + joint_idx);
    computation.jacobian(r, col) = 1.0;
    computation.error(r) = sample.position(r) - state.joint_positions(joint_idx);
    const double feedforward = sample.has_velocity ? sample.velocity(r) : 0.0;
    computation.desired_velocity(r) = feedforward + gains_(r) * computation.error(r);
  }

  last_error_ = computation.error;
  final_error_norm_ = last_error_.norm();
  bool within_tolerance = true;
  for (Eigen::Index i = 0; i < last_error_.size(); ++i) {
    if (std::abs(last_error_(i)) > goal_tolerance_(i)) {
      within_tolerance = false;
      break;
    }
  }

  if (elapsed >= duration && within_tolerance) {
    succeeded_ = true;
    status_message_ = hold_last_point_ ? "holding" : "succeeded";
    if (!hold_last_point_) {
      trajectory_active_ = false;
      computation.active = false;
    }
  } else {
    status_message_ = "tracking";
  }
  computation.status_message = status_message_;
  return computation;
}

bool JointTrajectoryTask::set_joint_trajectory(
  const trajectory_msgs::msg::JointTrajectory & trajectory,
  std::string & message)
{
  std::vector<Sample> samples;
  if (!validate_and_map_trajectory(trajectory, samples, message)) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  samples_ = std::move(samples);
  start_time_ = SteadyClock::now();
  current_target_ = samples_.front().position;
  last_error_ = Eigen::VectorXd::Zero(joint_names_.size());
  final_error_norm_ = std::numeric_limits<double>::infinity();
  trajectory_active_ = true;
  succeeded_ = false;
  canceled_ = false;
  timed_out_ = false;
  status_message_ = "tracking";
  message = "Joint trajectory accepted";
  return true;
}

bool JointTrajectoryTask::cancel_joint_trajectory(std::string & message)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (!trajectory_active_ && !succeeded_) {
    message = "No active joint trajectory";
    return false;
  }
  trajectory_active_ = false;
  canceled_ = true;
  succeeded_ = false;
  status_message_ = "canceled";
  message = "Joint trajectory canceled";
  return true;
}

JointTrajectoryTaskStatus JointTrajectoryTask::joint_trajectory_status() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  JointTrajectoryTaskStatus status;
  status.accepted = !samples_.empty();
  status.active = trajectory_active_ && !succeeded_ && !canceled_ && !timed_out_;
  status.succeeded = succeeded_;
  status.canceled = canceled_;
  status.timed_out = timed_out_;
  status.final_error_norm = final_error_norm_;
  status.message = status_message_;
  return status;
}

bool JointTrajectoryTask::set_gain(const std::vector<double> & gain, std::string & message)
{
  if (gain.size() != joint_names_.size()) {
    message = "Joint trajectory gain size does not match configured joint_names size";
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < gain.size(); ++i) {
    if (gain[i] < 0.0) {
      message = "Joint trajectory gain values must be non-negative";
      return false;
    }
    gains_(static_cast<Eigen::Index>(i)) = gain[i];
  }
  message = "Joint trajectory gain updated";
  return true;
}

void JointTrajectoryTask::reset()
{
  std::lock_guard<std::mutex> lock(mutex_);
  trajectory_active_ = false;
  succeeded_ = false;
  canceled_ = false;
  timed_out_ = false;
  status_message_ = "idle";
}

msg::TaskStatus JointTrajectoryTask::build_status() const
{
  auto status = TaskBaseCommon::build_status();
  std::lock_guard<std::mutex> lock(mutex_);
  status.active = enabled_ && trajectory_active_;
  status.status_message = status_message_;
  status.target_type = "joint_trajectory";
  status.joint_names = joint_names_;
  return status;
}

std::vector<double> JointTrajectoryTask::current_target() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<double> target(static_cast<size_t>(current_target_.size()), 0.0);
  for (Eigen::Index i = 0; i < current_target_.size(); ++i) {
    target[static_cast<size_t>(i)] = current_target_(i);
  }
  return target;
}

bool JointTrajectoryTask::validate_and_map_trajectory(
  const trajectory_msgs::msg::JointTrajectory & trajectory,
  std::vector<Sample> & samples,
  std::string & message) const
{
  if (trajectory.points.empty()) {
    message = "Joint trajectory has no points";
    return false;
  }
  if (trajectory.joint_names.size() != joint_names_.size()) {
    message = "Joint trajectory joint_names size does not match task joint_names size";
    return false;
  }

  std::unordered_map<std::string, size_t> incoming_index;
  for (size_t i = 0; i < trajectory.joint_names.size(); ++i) {
    incoming_index[trajectory.joint_names[i]] = i;
  }
  std::vector<size_t> order;
  order.reserve(joint_names_.size());
  for (const auto & joint_name : joint_names_) {
    const auto it = incoming_index.find(joint_name);
    if (it == incoming_index.end()) {
      message = "Joint trajectory is missing joint '" + joint_name + "'";
      return false;
    }
    order.push_back(it->second);
  }

  samples.clear();
  samples.reserve(trajectory.points.size());
  double previous_time = -std::numeric_limits<double>::infinity();
  for (const auto & point : trajectory.points) {
    if (point.positions.size() != trajectory.joint_names.size()) {
      message = "Joint trajectory point positions size does not match joint_names size";
      return false;
    }
    if (!point.velocities.empty() && point.velocities.size() != trajectory.joint_names.size()) {
      message = "Joint trajectory point velocities size does not match joint_names size";
      return false;
    }
    if (!finite_vector(point.positions) || (!point.velocities.empty() && !finite_vector(point.velocities))) {
      message = "Joint trajectory contains non-finite values";
      return false;
    }

    const double time_from_start = duration_to_seconds(point.time_from_start);
    if (time_from_start < 0.0 || time_from_start <= previous_time) {
      message = "Joint trajectory time_from_start values must be strictly increasing and non-negative";
      return false;
    }
    previous_time = time_from_start;

    Sample sample;
    sample.position = Eigen::VectorXd::Zero(joint_names_.size());
    sample.velocity = Eigen::VectorXd::Zero(joint_names_.size());
    sample.has_velocity = !point.velocities.empty();
    sample.time_from_start = time_from_start;
    for (size_t i = 0; i < order.size(); ++i) {
      sample.position(static_cast<Eigen::Index>(i)) = point.positions[order[i]];
      if (sample.has_velocity) {
        sample.velocity(static_cast<Eigen::Index>(i)) = point.velocities[order[i]];
      }
    }
    samples.push_back(sample);
  }
  return true;
}

JointTrajectoryTask::Sample JointTrajectoryTask::sample_at_locked(double elapsed) const
{
  if (samples_.size() == 1 || elapsed <= samples_.front().time_from_start) {
    return samples_.front();
  }
  if (elapsed >= samples_.back().time_from_start) {
    Sample sample = samples_.back();
    sample.velocity.setZero();
    sample.has_velocity = true;
    return sample;
  }

  for (size_t i = 1; i < samples_.size(); ++i) {
    const auto & next = samples_[i];
    if (elapsed > next.time_from_start) {
      continue;
    }
    const auto & prev = samples_[i - 1];
    const double span = next.time_from_start - prev.time_from_start;
    const double alpha = span > 0.0 ? (elapsed - prev.time_from_start) / span : 1.0;

    Sample sample;
    sample.position = prev.position + alpha * (next.position - prev.position);
    if (prev.has_velocity && next.has_velocity) {
      sample.velocity = prev.velocity + alpha * (next.velocity - prev.velocity);
      sample.has_velocity = true;
    } else {
      sample.velocity = (next.position - prev.position) / span;
      sample.has_velocity = true;
    }
    sample.time_from_start = elapsed;
    return sample;
  }
  return samples_.back();
}

double JointTrajectoryTask::trajectory_duration_locked() const
{
  return samples_.empty() ? 0.0 : samples_.back().time_from_start;
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::JointTrajectoryTask,
  task_priority_kinematic_control::TaskBase)
