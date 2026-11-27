#include "task_priority_kinematic_control/tasks/robot_nominal_task.hpp"
#include "task_priority_kinematic_control/tasks/frame_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_nominal_task.hpp"
#include "sura_manipulation_actions/base_trajectory_sampling.hpp"

#include <pluginlib/class_list_macros.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace task_priority_kinematic_control
{
namespace
{
constexpr const char * kBaseNames[6] = {"base_x", "base_y", "base_z", "base_roll", "base_pitch", "base_yaw"};

std::vector<bool> activation_parameter(
  const std::map<std::string, rclcpp::Parameter> & parameters, std::vector<bool> activation)
{
  const auto it = parameters.find("activation");
  if (it == parameters.end()) {return activation;}
  std::vector<double> values;
  switch (it->second.get_type()) {
    case rclcpp::ParameterType::PARAMETER_BOOL_ARRAY:
      for (bool value : it->second.as_bool_array()) {values.push_back(value ? 1.0 : 0.0);}
      break;
    case rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY:
      for (auto value : it->second.as_integer_array()) {values.push_back(static_cast<double>(value));}
      break;
    case rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY:
      values = it->second.as_double_array();
      break;
    default:
      throw std::invalid_argument("robot_nominal activation must be an array");
  }
  if (values.empty()) {return activation;}
  if (values.size() != activation.size()) {
    throw std::invalid_argument("robot_nominal activation needs 6 base axes plus every arm joint");
  }
  for (size_t i = 0; i < values.size(); ++i) {activation[i] = values[i] != 0.0;}
  return activation;
}
}  // namespace

void RobotNominalTask::configure(
  const std::string & id, const std::string & plugin_name,
  const std::map<std::string, rclcpp::Parameter> & parameters, const TaskContext & context)
{
  configure_common(id, plugin_name, parameters, context);
  joint_names_ = context.model->all_joint_names();
  const size_t count = rows();
  // Roll and pitch are left to the leveling task by default.
  std::vector<bool> activation(count, true);
  activation[3] = activation[4] = false;
  activation_ = activation_parameter(parameters, activation);
  const auto gain = get_double_array_param(parameters, "gain", {});
  gains_ = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(count), 0.5);
  std::string message;
  if (!gain.empty() && !set_gain(gain, message)) {throw std::invalid_argument(message);}
  timeout_ = get_double_param(parameters, "joint_command_timeout", 0.5);
  if (!std::isfinite(timeout_) || timeout_ <= 0.0) {
    throw std::invalid_argument("joint_command_timeout must be finite and positive");
  }
  sync_tasks_ = get_string_array_param(parameters, "sync_tasks", {});
  joint_target_ = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(joint_names_.size()));
  joint_velocity_ = joint_target_;
  has_reference_ = false;
  message_ = "idle";
}

bool RobotNominalTask::live() const
{
  return has_reference_ &&
         std::chrono::duration<double>(steady_now() - reference_time_).count() <= timeout_;
}

bool RobotNominalTask::has_live_reference() const {return enabled_ && live();}

TaskComputation RobotNominalTask::update(const WholeBodyState & state, const KinematicsBackend &)
{
  TaskComputation out;
  if (!enabled_) {out.status_message = message_ = "disabled"; return out;}
  if (!live()) {
    if (has_reference_) {has_reference_ = false; message_ = "reference_expired";}
    else if (message_ != "reference_expired") {message_ = "idle";}
    out.status_message = message_;
    return out;
  }
  if (!state.navigation_valid || !state.joints_valid || !state.base_position.allFinite() ||
    !state.base_rpy.allFinite() ||
    state.joint_positions.size() != static_cast<Eigen::Index>(joint_names_.size()))
  {out.status_message = "waiting_for_state"; return out;}
  Coordinates current;
  current << state.base_position, state.base_rpy;
  if (std::abs(std::cos(current(4))) < 1e-4) {out.status_message = "base RPY singularity"; return out;}
  const Eigen::Matrix3d inverse_map = sura_manipulation_actions::baseAngularVelocityMap(current).inverse();
  const auto & axes = context_.model->active_base_dofs();
  std::vector<size_t> active;
  for (size_t row = 0; row < rows(); ++row) {
    if (!activation_[row] || (!reference_mask_.empty() && !reference_mask_[row])) {continue;}
    if (row < 6 && std::find(axes.begin(), axes.end(), static_cast<int>(row)) == axes.end()) {continue;}
    active.push_back(row);
  }
  if (active.empty()) {out.status_message = "no_active_rows"; return out;}
  const auto dofs = static_cast<Eigen::Index>(context_.model->total_dofs());
  const auto base_dofs = static_cast<Eigen::Index>(context_.model->base_dofs());
  const auto count = static_cast<Eigen::Index>(active.size());
  out.jacobian = Eigen::MatrixXd::Zero(count, dofs);
  out.error = Eigen::VectorXd::Zero(count);
  out.feedforward = Eigen::VectorXd::Zero(count);
  out.desired_velocity = Eigen::VectorXd::Zero(count);
  for (Eigen::Index r = 0; r < count; ++r) {
    const size_t row = active[static_cast<size_t>(r)];
    if (row < 6) {
      double error = base_target_(static_cast<Eigen::Index>(row)) - current(static_cast<Eigen::Index>(row));
      if (row >= 3) {error = std::atan2(std::sin(error), std::cos(error));}
      out.error(r) = error;
      out.feedforward(r) = base_rates_(static_cast<Eigen::Index>(row));
      for (size_t column = 0; column < axes.size(); ++column) {
        const int axis = axes[column];
        if (row < 3 && axis == static_cast<int>(row)) {out.jacobian(r, static_cast<Eigen::Index>(column)) = 1.0;}
        if (row >= 3 && axis >= 3) {
          out.jacobian(r, static_cast<Eigen::Index>(column)) =
            inverse_map(static_cast<Eigen::Index>(row - 3), axis - 3);
        }
      }
    } else {
      const auto j = static_cast<Eigen::Index>(row - 6);
      out.error(r) = joint_target_(j) - state.joint_positions(j);
      out.feedforward(r) = joint_velocity_(j);
      out.jacobian(r, base_dofs + j) = 1.0;
    }
    out.desired_velocity(r) = out.feedforward(r) + gains_(static_cast<Eigen::Index>(row)) * out.error(r);
  }
  out.active = true;
  out.has_frame_pose = true;
  out.frame_id = context_.model->base_frame();
  out.frame_pose = sura_manipulation_actions::basePose(current);
  out.status_message = message_ = "tracking";
  return out;
}

bool RobotNominalTask::set_reference(const Coordinates & base, const Coordinates & base_rates,
  const Eigen::VectorXd & joints, const Eigen::VectorXd & joint_velocities, std::string & message,
  const std::vector<bool> & mask)
{
  if (!mask.empty() && mask.size() != rows()) {
    message = "robot_nominal reference mask needs 6 base rows plus every joint"; return false;
  }
  if (!base.allFinite() || !base_rates.allFinite() || !joints.allFinite() || !joint_velocities.allFinite() ||
    joints.size() != static_cast<Eigen::Index>(joint_names_.size()) || joint_velocities.size() != joints.size())
  {message = "robot_nominal reference needs finite base and every model joint"; return false;}
  if (std::abs(std::cos(base(4))) < 1e-4) {message = "robot_nominal base pitch is singular"; return false;}
  base_target_ = base;
  base_rates_ = enabled_ ? base_rates : Coordinates::Zero();
  joint_target_ = joints;
  joint_velocity_ = enabled_ ? joint_velocities : Eigen::VectorXd::Zero(joints.size());
  reference_mask_ = mask;
  has_reference_ = true;
  reference_time_ = steady_now();
  message = "robot_nominal reference updated";
  return true;
}

bool RobotNominalTask::set_reference(
  const sura_manipulation_actions::msg::RobotNominalReference & reference, std::string & message,
  bool zero_velocity)
{
  if (!reference.header.frame_id.empty() && reference.header.frame_id != context_.model->world_frame()) {
    message = "robot_nominal reference must use the controller world frame"; return false;
  }
  const size_t count = joint_names_.size();
  if (reference.joint_names.size() != count || reference.positions.size() != count ||
    (!reference.velocities.empty() && reference.velocities.size() != count))
  {message = "robot_nominal reference needs every model joint"; return false;}
  Eigen::VectorXd joints(static_cast<Eigen::Index>(count)), velocity = Eigen::VectorXd::Zero(joints.size());
  for (size_t i = 0; i < count; ++i) {
    const auto found = std::find(reference.joint_names.begin(), reference.joint_names.end(), joint_names_[i]);
    if (found == reference.joint_names.end() ||
      std::count(reference.joint_names.begin(), reference.joint_names.end(), joint_names_[i]) != 1)
    {message = "Unknown, duplicate or missing robot_nominal joint: " + joint_names_[i]; return false;}
    const auto index = static_cast<size_t>(found - reference.joint_names.begin());
    joints(static_cast<Eigen::Index>(i)) = reference.positions[index];
    if (!reference.velocities.empty() && !zero_velocity) {
      velocity(static_cast<Eigen::Index>(i)) = reference.velocities[index];
    }
  }
  const auto & q = reference.base_pose.orientation;
  Eigen::Quaterniond rotation(q.w, q.x, q.y, q.z);
  if (!rotation.coeffs().allFinite() || std::abs(rotation.norm() - 1.0) > 1e-3) {
    message = "robot_nominal base quaternion must be normalized"; return false;
  }
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() << reference.base_pose.position.x, reference.base_pose.position.y,
    reference.base_pose.position.z;
  pose.linear() = rotation.normalized().toRotationMatrix();
  // Keep the yaw branch continuous with the previous reference.
  const Coordinates base = has_reference_ ?
    sura_manipulation_actions::baseCoordinates(pose, base_target_) :
    sura_manipulation_actions::baseCoordinates(pose);
  Coordinates rates = Coordinates::Zero();
  if (!zero_velocity) {
    rates.head<3>() << reference.base_twist.linear.x, reference.base_twist.linear.y, reference.base_twist.linear.z;
    if (std::abs(std::cos(base(4))) < 1e-4) {message = "robot_nominal base pitch is singular"; return false;}
    rates.tail<3>() = sura_manipulation_actions::baseAngularVelocityMap(base).inverse() *
      Eigen::Vector3d(reference.base_twist.angular.x, reference.base_twist.angular.y,
      reference.base_twist.angular.z);
  }
  return set_reference(base, rates, joints, velocity, message, reference.activation);
}

bool RobotNominalTask::base_masked() const
{
  return !reference_mask_.empty() &&
         std::none_of(reference_mask_.begin(), reference_mask_.begin() + 6, [](bool value) {return value;});
}

void RobotNominalTask::clear_reference()
{
  has_reference_ = false;
  reference_mask_.clear();
  base_rates_.setZero();
  joint_velocity_.setZero();
  message_ = enabled_ ? "idle" : "disabled";
}

bool RobotNominalTask::set_gain(const std::vector<double> & gain, std::string & message)
{
  if (gain.size() != rows() || !std::all_of(gain.begin(), gain.end(),
    [](double value) {return std::isfinite(value) && value >= 0.0;}))
  {message = "robot_nominal gain needs 6 base + every joint finite nonnegative values"; return false;}
  gains_ = Eigen::Map<const Eigen::VectorXd>(gain.data(), static_cast<Eigen::Index>(gain.size()));
  message = "robot_nominal gain updated";
  return true;
}

bool RobotNominalTask::set_joint_activation(const std::vector<bool> & activation, std::string & message)
{
  if (activation.size() != rows()) {
    message = "robot_nominal activation needs 6 base axes plus every joint"; return false;
  }
  activation_ = activation;
  message = "robot_nominal activation updated";
  return true;
}

bool RobotNominalTask::set_enabled(bool enabled)
{
  if (enabled != enabled_) {clear_reference();}
  const bool result = TaskBaseCommon::set_enabled(enabled);
  message_ = enabled_ ? "idle" : "disabled";
  return result;
}

void RobotNominalTask::reset()
{
  TaskBaseCommon::reset();
  clear_reference();
}

msg::TaskStatus RobotNominalTask::build_status() const
{
  auto status = TaskBaseCommon::build_status();
  status.active = enabled_ && live();
  status.status_message = message_;
  status.target_type = "robot_nominal";
  status.joint_names.assign(std::begin(kBaseNames), std::end(kBaseNames));
  status.joint_names.insert(status.joint_names.end(), joint_names_.begin(), joint_names_.end());
  status.joint_activation = activation_;
  return status;
}

std::vector<double> RobotNominalTask::current_target() const
{
  std::vector<double> out(base_target_.data(), base_target_.data() + 6);
  out.insert(out.end(), joint_target_.data(), joint_target_.data() + joint_target_.size());
  return out;
}

std::shared_ptr<RobotNominalTask> find_robot_nominal_task(
  const std::vector<std::shared_ptr<TaskBase>> & tasks)
{
  for (const auto & task : tasks) {
    if (auto nominal = std::dynamic_pointer_cast<RobotNominalTask>(task)) {return nominal;}
  }
  return {};
}

namespace
{
bool sync_holds(const std::vector<std::shared_ptr<TaskBase>> & tasks, const RobotNominalTask & task,
  const WholeBodyModel & model, const RobotNominalTask::Coordinates & base,
  const Eigen::VectorXd & joints, std::string & message)
{
  for (const auto & id : task.sync_tasks()) {
    const auto found = std::find_if(tasks.begin(), tasks.end(),
      [&](const auto & candidate) {return candidate->id() == id;});
    if (found == tasks.end()) {continue;}  // Optional holds may be absent in some setups.
    if (auto frame = std::dynamic_pointer_cast<FramePoseTask>(*found)) {
      // A reference without base rows leaves the base hold where it was.
      if (task.base_masked()) {continue;}
      if (frame->roll_pitch_control() || frame->frame_id() != model.base_frame()) {
        message = "robot_nominal can only synchronize base FramePoseTask holds: " + id; return false;
      }
      frame->capture_measured_target(sura_manipulation_actions::basePose(base));
    } else if (auto nominal = std::dynamic_pointer_cast<JointNominalTask>(*found)) {
      std::vector<double> values;
      for (const auto & name : nominal->build_status().joint_names) {
        const int index = model.joint_index(name);
        if (index < 0 || index >= joints.size()) {message = "Unknown synchronized joint: " + name; return false;}
        values.push_back(joints(index));
      }
      if (!nominal->set_joint_target(values, message)) {return false;}
    } else {
      message = "robot_nominal sync task must be a FramePoseTask or JointNominalTask: " + id; return false;
    }
  }
  return true;
}
}  // namespace

bool apply_robot_nominal_reference(
  const std::vector<std::shared_ptr<TaskBase>> & tasks, RobotNominalTask & task,
  const RobotNominalTask::Coordinates & base, const RobotNominalTask::Coordinates & base_rates,
  const Eigen::VectorXd & joints, const Eigen::VectorXd & joint_velocities, std::string & message,
  const std::vector<bool> & mask)
{
  if (!task.set_reference(base, base_rates, joints, joint_velocities, message, mask)) {return false;}
  return sync_holds(tasks, task, task.model(), base, joints, message);
}

bool apply_robot_nominal_reference(
  const std::vector<std::shared_ptr<TaskBase>> & tasks, RobotNominalTask & task,
  const sura_manipulation_actions::msg::RobotNominalReference & reference, std::string & message,
  bool zero_velocity)
{
  if (!task.set_reference(reference, message, zero_velocity)) {return false;}
  return sync_holds(tasks, task, task.model(), task.base_target(), task.joint_target(), message);
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(task_priority_kinematic_control::RobotNominalTask,
  task_priority_kinematic_control::TaskBase)
