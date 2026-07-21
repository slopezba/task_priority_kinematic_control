#include "task_priority_kinematic_control/core/task_priority_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "pluginlib/class_list_macros.hpp"

namespace task_priority_kinematic_control
{

namespace
{

const char * base_dof_name(int dof)
{
  switch (dof) {
    case 0:
      return "base.linear.x";
    case 1:
      return "base.linear.y";
    case 2:
      return "base.linear.z";
    case 3:
      return "base.angular.x";
    case 4:
      return "base.angular.y";
    case 5:
      return "base.angular.z";
    default:
      return "base.unknown";
  }
}

template<typename T>
void auto_declare_if_missing(controller_interface::ControllerInterface * controller,
  const std::string & name, const T & default_value)
{
  if (!controller->get_node()->has_parameter(name)) {
    controller->get_node()->declare_parameter<T>(name, default_value);
  }
}

SolverMethod parse_solver_method(const std::string & name)
{
  if (name == "pinv") {
    return SolverMethod::kPinv;
  }
  if (name == "svd") {
    return SolverMethod::kSvd;
  }
  return SolverMethod::kDls;
}

bool is_known_solver_method(const std::string & name)
{
  return name == "dls" || name == "pinv" || name == "svd";
}

std::string solver_method_name(SolverMethod method)
{
  switch (method) {
    case SolverMethod::kPinv:
      return "pinv";
    case SolverMethod::kSvd:
      return "svd";
    case SolverMethod::kDls:
    default:
      return "dls";
  }
}

geometry_msgs::msg::Pose pose_msg_from_isometry(const Eigen::Isometry3d & pose)
{
  geometry_msgs::msg::Pose msg;
  msg.position.x = pose.translation().x();
  msg.position.y = pose.translation().y();
  msg.position.z = pose.translation().z();
  const Eigen::Quaterniond q(pose.rotation());
  msg.orientation.x = q.x();
  msg.orientation.y = q.y();
  msg.orientation.z = q.z();
  msg.orientation.w = q.w();
  return msg;
}

std::vector<double> vector_from_eigen(const Eigen::VectorXd & vector)
{
  std::vector<double> out(static_cast<size_t>(vector.size()), 0.0);
  for (Eigen::Index i = 0; i < vector.size(); ++i) {
    out[static_cast<size_t>(i)] = vector(i);
  }
  return out;
}

bool split_task_parameter_name(
  const std::string & name,
  std::string & task_id,
  std::string & field)
{
  const std::string prefix = "tasks.";
  if (name.rfind(prefix, 0) != 0) {
    return false;
  }

  const auto field_separator = name.rfind('.');
  if (field_separator == std::string::npos || field_separator <= prefix.size()) {
    return false;
  }

  task_id = name.substr(prefix.size(), field_separator - prefix.size());
  field = name.substr(field_separator + 1);
  return !task_id.empty() && !field.empty();
}

bool values_are_non_negative(const std::vector<double> & values)
{
  return std::all_of(values.begin(), values.end(), [](double value) {
    return value >= 0.0;
  });
}

class ParameterUpdateGuard
{
public:
  explicit ParameterUpdateGuard(std::atomic_bool & pending)
  : pending_(pending)
  {
    pending_.store(true, std::memory_order_release);
  }

  ~ParameterUpdateGuard()
  {
    pending_.store(false, std::memory_order_release);
  }

private:
  std::atomic_bool & pending_;
};

std::vector<int> parse_active_base_dofs(
  const std::vector<int64_t> & param_values,
  const rclcpp::Logger & logger)
{
  std::ostringstream values_stream;
  values_stream << "[";

  std::vector<int> parsed_values;
  parsed_values.reserve(param_values.size());
  for (size_t i = 0; i < param_values.size(); ++i) {
    const auto value = param_values[i];
    if (i > 0) {
      values_stream << ", ";
    }
    values_stream << value;

    if (value < 0 || value > 5) {
      throw std::runtime_error(
        "Invalid active_base_dofs parameter. Expected values in [0, 5], got " +
        std::to_string(value) + " at index " + std::to_string(i));
    }
    parsed_values.push_back(static_cast<int>(value));
  }

  values_stream << "]";
  RCLCPP_INFO(
    logger,
    "Raw active_base_dofs parameter: %s",
    values_stream.str().c_str());

  return parsed_values;
}

bool validate_positive_vector_parameter(
  const std::string & name,
  const std::vector<double> & values,
  size_t expected_size,
  std::string & reason)
{
  if (values.size() != expected_size) {
    reason = name + " must contain " + std::to_string(expected_size) +
      " values, got " + std::to_string(values.size());
    return false;
  }
  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i] <= 0.0 || !std::isfinite(values[i])) {
      reason = name + " values must be finite and positive; invalid value at index " +
        std::to_string(i);
      return false;
    }
  }
  return true;
}

Eigen::VectorXd eigen_vector_from_std(const std::vector<double> & values)
{
  Eigen::VectorXd vector(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    vector(static_cast<Eigen::Index>(i)) = values[i];
  }
  return vector;
}

void append_joint_velocity_interfaces(
  const controller_interface::ControllerInterface * controller,
  const std::string & parameter_name,
  std::vector<std::string> & interface_names)
{
  if (!controller->get_node()->has_parameter(parameter_name)) {
    return;
  }

  std::vector<std::string> joints;
  if (!controller->get_node()->get_parameter(parameter_name, joints)) {
    return;
  }

  for (const auto & joint : joints) {
    if (joint.empty()) {
      RCLCPP_WARN(
        controller->get_node()->get_logger(),
        "Ignoring empty joint name in '%s'",
        parameter_name.c_str());
      continue;
    }

    std::string interface_name;
    interface_name.reserve(joint.size() + std::string("/velocity").size());
    interface_name = joint;
    interface_name += "/velocity";
    interface_names.push_back(interface_name);
  }
}

void append_joint_state_interfaces(
  const controller_interface::ControllerInterface * controller,
  const std::string & parameter_name,
  const std::string & interface_type,
  std::vector<std::string> & interface_names)
{
  if (!controller->get_node()->has_parameter(parameter_name)) {
    return;
  }

  std::vector<std::string> joints;
  if (!controller->get_node()->get_parameter(parameter_name, joints)) {
    return;
  }

  for (const auto & joint : joints) {
    if (joint.empty()) {
      RCLCPP_WARN(
        controller->get_node()->get_logger(),
        "Ignoring empty joint name in '%s'",
        parameter_name.c_str());
      continue;
    }

    interface_names.push_back(joint + "/" + interface_type);
  }
}

}  // namespace

controller_interface::CallbackReturn TaskPriorityController::on_init()
{
  try {
    auto_declare_if_missing(this, "backend_plugin",
      std::string("task_priority_kinematic_control/KDLKinematicsBackend"));
    auto_declare_if_missing(this, "robot_description", std::string(""));
    auto_declare_if_missing(this, "robot_description_source_node",
      std::string("/robot_state_publisher"));
    auto_declare_if_missing(this, "robot_description_wait_timeout_sec", 3.0);
    auto_declare_if_missing(this, "world_frame", std::string("world_ned"));
    auto_declare_if_missing(this, "base_frame", std::string("cirtesub/base_link"));
    auto_declare_if_missing(this, "left_tip_frame", std::string("cirtesub/alpha_left/ee_base_link"));
    auto_declare_if_missing(this, "right_tip_frame", std::string("cirtesub/alpha_right/ee_base_link"));
    auto_declare_if_missing(this, "left_arm_joints", std::vector<std::string>{});
    auto_declare_if_missing(this, "right_arm_joints", std::vector<std::string>{});
    auto_declare_if_missing(this, "active_base_dofs", std::vector<int64_t>{0, 1, 2, 3, 4, 5});
    auto_declare_if_missing(this, "solver_method", std::string("dls"));
    auto_declare_if_missing(this, "dls_lambda", 0.05);
    auto_declare_if_missing(this, "dof_weights", std::vector<double>{});
    auto_declare_if_missing(this, "velocity_limits", std::vector<double>{});
    auto_declare_if_missing(this, "navigator_topic", std::string("/cirtesub/navigator/navigation"));
    auto_declare_if_missing(this, "body_velocity_controller_name", std::string("body_velocity"));
    auto_declare_if_missing(this, "task_ids", std::vector<std::string>{});
    declare_task_parameters();
  } catch (const std::exception & ex) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception in on_init: %s", ex.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

void TaskPriorityController::declare_task_parameters()
{
  const auto task_ids = get_node()->get_parameter("task_ids").as_string_array();
  for (const auto & task_id : task_ids) {
    const std::string prefix = "tasks." + task_id + ".";
    auto_declare_if_missing(this, prefix + "plugin", std::string(""));
    auto_declare_if_missing(this, prefix + "enabled", true);
    auto_declare_if_missing(this, prefix + "priority", 0.0);
    auto_declare_if_missing(this, prefix + "group", std::string("default"));
    auto_declare_if_missing(this, prefix + "frame_id", std::string(""));
    auto_declare_if_missing(this, prefix + "gain", std::vector<double>{});
    auto_declare_if_missing(this, prefix + "lower_limits", std::vector<double>{});
    auto_declare_if_missing(this, prefix + "upper_limits", std::vector<double>{});
    auto_declare_if_missing(this, prefix + "margin", 0.1);
    auto_declare_if_missing(this, prefix + "alpha", 0.1);
    auto_declare_if_missing(this, prefix + "delta", 0.15);
    auto_declare_if_missing(this, prefix + "eps", 1e-4);
    auto_declare_if_missing(this, prefix + "gain_scalar", 0.5);
    auto_declare_if_missing(this, prefix + "safe_distance", 0.05);
    auto_declare_if_missing(this, prefix + "activation_distance", 0.12);
    auto_declare_if_missing(this, prefix + "max_repulsive_velocity", 0.08);
    auto_declare_if_missing(this, prefix + "check_adjacent_links", false);
    auto_declare_if_missing(this, prefix + "include_link_substrings", std::vector<std::string>{});
    auto_declare_if_missing(this, prefix + "exclude_link_substrings", std::vector<std::string>{});
    auto_declare_if_missing(this, prefix + "joint_names", std::vector<std::string>{});
    auto_declare_if_missing(this, prefix + "target", std::vector<double>{});
    auto_declare_if_missing(this, prefix + "activation", std::vector<int64_t>{});
    auto_declare_if_missing(this, prefix + "target_yaw", 0.0);
    auto_declare_if_missing(this, prefix + "goal_tolerance", std::vector<double>{});
    auto_declare_if_missing(this, prefix + "trajectory_timeout", 0.5);
    auto_declare_if_missing(this, prefix + "hold_last_point", true);
  }
}

controller_interface::InterfaceConfiguration
TaskPriorityController::command_interface_configuration() const
{
  const std::string body_velocity_name = get_node()->has_parameter("body_velocity_controller_name") ?
    get_node()->get_parameter("body_velocity_controller_name").as_string() :
    std::string("body_velocity");
  std::vector<std::string> names = {
    body_velocity_name + "/linear.x",
    body_velocity_name + "/linear.y",
    body_velocity_name + "/linear.z",
    body_velocity_name + "/angular.x",
    body_velocity_name + "/angular.y",
    body_velocity_name + "/angular.z"
  };

  try {
    append_joint_velocity_interfaces(this, "left_arm_joints", names);
    append_joint_velocity_interfaces(this, "right_arm_joints", names);
  } catch (const std::exception & ex) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "Failed to build TaskPriorityController command interface list: %s",
      ex.what());
  }

  return {
    controller_interface::interface_configuration_type::INDIVIDUAL,
    names
  };
}

controller_interface::InterfaceConfiguration
TaskPriorityController::state_interface_configuration() const
{
  std::vector<std::string> names;
  append_joint_state_interfaces(this, "left_arm_joints", "position", names);
  append_joint_state_interfaces(this, "right_arm_joints", "position", names);
  append_joint_state_interfaces(this, "left_arm_joints", "velocity", names);
  append_joint_state_interfaces(this, "right_arm_joints", "velocity", names);

  return {
    controller_interface::interface_configuration_type::INDIVIDUAL,
    names
  };
}

controller_interface::CallbackReturn TaskPriorityController::on_configure(
  const rclcpp_lifecycle::State &)
{
  model_ = std::make_shared<WholeBodyModel>();
  task_manager_ = std::make_unique<TaskManager>(
    get_node()->get_node_parameters_interface(), get_node()->get_logger());
  backend_plugin_name_ = get_node()->get_parameter("backend_plugin").as_string();
  body_velocity_controller_name_ =
    get_node()->get_parameter("body_velocity_controller_name").as_string();
  left_arm_joints_ = get_node()->get_parameter("left_arm_joints").as_string_array();
  right_arm_joints_ = get_node()->get_parameter("right_arm_joints").as_string_array();
  task_ids_ = get_node()->get_parameter("task_ids").as_string_array();

  active_base_dofs_ = parse_active_base_dofs(
    get_node()->get_parameter("active_base_dofs").as_integer_array(),
    get_node()->get_logger());

  configure_model();
  backend_loader_ = std::make_unique<pluginlib::ClassLoader<KinematicsBackend>>(
    "task_priority_kinematic_control", "task_priority_kinematic_control::KinematicsBackend");
  rebuild_backend();
  configure_solver();
  param_callback_handle_ = get_node()->add_on_set_parameters_callback(
    std::bind(&TaskPriorityController::on_parameters_set, this, std::placeholders::_1));
  refresh_task_manager();
  RCLCPP_INFO(
    get_node()->get_logger(),
    "TaskPriorityController frames: world='%s' base='%s' left_tip='%s' right_tip='%s'",
    get_node()->get_parameter("world_frame").as_string().c_str(),
    get_node()->get_parameter("base_frame").as_string().c_str(),
    get_node()->get_parameter("left_tip_frame").as_string().c_str(),
    get_node()->get_parameter("right_tip_frame").as_string().c_str());
  {
    std::ostringstream joints_stream;
    joints_stream << "\n  Base DOFs:";
    for (size_t i = 0; i < active_base_dofs_.size(); ++i) {
      joints_stream << "\n    [" << i << "] " << base_dof_name(active_base_dofs_[i]) <<
        " (index " << active_base_dofs_[i] << ")";
    }
    joints_stream << "\n  Arm joints:";
    const auto all_joints = model_->all_joint_names();
    for (size_t i = 0; i < all_joints.size(); ++i) {
      joints_stream << "\n    [" << (active_base_dofs_.size() + i) << "] " << all_joints[i];
    }
    RCLCPP_INFO(
      get_node()->get_logger(),
      "TaskPriorityController generalized velocity order:%s",
      joints_stream.str().c_str());
  }

  const auto navigator_topic = get_node()->get_parameter("navigator_topic").as_string();
  navigator_sub_ = get_node()->create_subscription<NavigatorMsg>(
    navigator_topic,
    rclcpp::SystemDefaultsQoS(),
    [this](const NavigatorMsg::SharedPtr msg)
    {
      navigator_buffer_.writeFromNonRT(msg);
    });
  configure_external_interfaces();

  RCLCPP_INFO(get_node()->get_logger(), "Configured TaskPriorityController");
  return controller_interface::CallbackReturn::SUCCESS;
}

void TaskPriorityController::configure_model()
{
  model_->configure(
    get_node()->get_parameter("world_frame").as_string(),
    get_node()->get_parameter("base_frame").as_string(),
    get_node()->get_parameter("left_tip_frame").as_string(),
    get_node()->get_parameter("right_tip_frame").as_string(),
    left_arm_joints_,
    right_arm_joints_,
    active_base_dofs_);
}

std::string TaskPriorityController::resolve_robot_description()
{
  const std::string local = get_node()->get_parameter("robot_description").as_string();
  if (!local.empty()) {
    return local;
  }

  const std::string source_node =
    get_node()->get_parameter("robot_description_source_node").as_string();
  const double timeout_sec = std::max(
    0.1, get_node()->get_parameter("robot_description_wait_timeout_sec").as_double());

  remote_param_client_ = std::make_shared<rclcpp::SyncParametersClient>(get_node(), source_node);
  if (!remote_param_client_->wait_for_service(std::chrono::duration<double>(timeout_sec))) {
    throw std::runtime_error(
      "robot_description is empty locally and parameter service for '" + source_node + "' is not available");
  }

  const auto parameters = remote_param_client_->get_parameters({"robot_description"});
  if (parameters.empty()) {
    throw std::runtime_error("Failed to read robot_description from '" + source_node + "'");
  }

  const std::string value = parameters.front().as_string();
  if (value.empty()) {
    throw std::runtime_error("Node '" + source_node + "' returned an empty robot_description");
  }

  RCLCPP_INFO(
    get_node()->get_logger(),
    "Loaded robot_description at runtime from '%s'",
    source_node.c_str());
  return value;
}

void TaskPriorityController::rebuild_backend()
{
  backend_ = backend_loader_->createSharedInstance(backend_plugin_name_);
  backend_->configure(*model_, resolve_robot_description(), get_node()->get_logger());
}

void TaskPriorityController::configure_solver()
{
  solver_.configure(model_->total_dofs());
  solver_.set_method(parse_solver_method(get_node()->get_parameter("solver_method").as_string()));
  solver_.set_damping(get_node()->get_parameter("dls_lambda").as_double());

  const auto weights = get_node()->get_parameter("dof_weights").as_double_array();
  if (!weights.empty()) {
    std::string reason;
    if (!validate_positive_vector_parameter("dof_weights", weights, model_->total_dofs(), reason)) {
      RCLCPP_WARN(
        get_node()->get_logger(),
        "Ignoring invalid initial dof_weights: %s",
        reason.c_str());
    } else {
      solver_.set_weights(eigen_vector_from_std(weights));
    }
  }

  const auto limits = get_node()->get_parameter("velocity_limits").as_double_array();
  if (!limits.empty()) {
    std::string reason;
    if (!validate_positive_vector_parameter("velocity_limits", limits, model_->total_dofs(), reason)) {
      RCLCPP_WARN(
        get_node()->get_logger(),
        "Ignoring invalid initial velocity_limits: %s",
        reason.c_str());
    } else {
      solver_.set_velocity_limits(eigen_vector_from_std(limits));
    }
  }
}

void TaskPriorityController::refresh_task_manager()
{
  task_manager_ = std::make_unique<TaskManager>(
    get_node()->get_node_parameters_interface(), get_node()->get_logger());
  task_manager_->configure(TaskContext{model_});
}

bool TaskPriorityController::apply_runtime_tuning_command(
  RuntimeTuningCommand & command,
  std::string & message)
{
  if (command.type == RuntimeTuningCommand::Type::Solver) {
    if (!is_known_solver_method(command.solver_method)) {
      message = "Unknown solver_method: " + command.solver_method;
      return false;
    }
    if (command.dls_lambda <= 0.0) {
      message = "dls_lambda must be positive";
      return false;
    }
    if (command.update_dof_weights) {
      std::string reason;
      if (!validate_positive_vector_parameter(
          "dof_weights", command.dof_weights, model_->total_dofs(), reason))
      {
        message = reason;
        return false;
      }
      solver_.set_weights(eigen_vector_from_std(command.dof_weights));
    }
    solver_.set_method(parse_solver_method(command.solver_method));
    solver_.set_damping(command.dls_lambda);
    message = "Solver config updated";
    return true;
  }

  for (const auto & update : command.gain_updates) {
    std::string update_message;
    if (update.field == "gain") {
      if (!values_are_non_negative(update.values)) {
        message = "tasks." + update.task_id + ".gain values must be non-negative";
        return false;
      }
      if (!task_manager_->set_task_gain(update.task_id, update.values, update_message)) {
        message = update_message;
        return false;
      }
    } else if (update.field == "gain_scalar") {
      if (update.values.size() != 1U) {
        message = "tasks." + update.task_id + ".gain_scalar must contain one value";
        return false;
      }
      if (update.values.front() < 0.0) {
        message = "tasks." + update.task_id + ".gain_scalar must be non-negative";
        return false;
      }
      if (!task_manager_->set_task_gain_scalar(update.task_id, update.values.front(), update_message)) {
        message = update_message;
        return false;
      }
    } else {
      message = "Unsupported gain field: " + update.field;
      return false;
    }
  }

  message = "Task gains updated";
  return true;
}

void TaskPriorityController::process_pending_runtime_tuning()
{
  std::deque<std::shared_ptr<RuntimeTuningCommand>> commands;
  {
    std::scoped_lock lock(runtime_tuning_mutex_);
    commands.swap(pending_runtime_tuning_);
  }

  if (commands.empty()) {
    return;
  }

  std::scoped_lock task_lock(task_mutex_);
  for (const auto & command : commands) {
    {
      std::scoped_lock command_lock(command->mutex);
      if (command->canceled) {
        continue;
      }
    }

    std::string message;
    const bool success = apply_runtime_tuning_command(*command, message);
    {
      std::scoped_lock command_lock(command->mutex);
      command->success = success;
      command->message = message;
      command->completed = true;
    }
    command->cv.notify_all();
  }
}

bool TaskPriorityController::enqueue_runtime_tuning_command(
  const std::shared_ptr<RuntimeTuningCommand> & command,
  std::string & message)
{
  {
    std::scoped_lock lock(runtime_tuning_mutex_);
    pending_runtime_tuning_.push_back(command);
  }

  std::unique_lock<std::mutex> lock(command->mutex);
  const bool completed = command->cv.wait_for(
    lock,
    std::chrono::seconds(2),
    [&command]() { return command->completed; });
  if (!completed) {
    command->canceled = true;
    message = "Timed out waiting for controller update";
    return false;
  }

  message = command->message;
  return command->success;
}

rcl_interfaces::msg::SetParametersResult TaskPriorityController::on_parameters_set(
  const std::vector<rclcpp::Parameter> & parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;

  ParameterUpdateGuard pending(parameter_update_pending_);
  std::scoped_lock lock(task_mutex_);
  for (const auto & parameter : parameters) {
    if (parameter.get_name() == "dls_lambda") {
      if (parameter.as_double() <= 0.0) {
        result.successful = false;
        result.reason = "dls_lambda must be positive";
        return result;
      }
      solver_.set_damping(parameter.as_double());
    } else if (parameter.get_name() == "solver_method") {
      solver_.set_method(parse_solver_method(parameter.as_string()));
    } else if (parameter.get_name() == "dof_weights") {
      const auto values = parameter.as_double_array();
      std::string reason;
      if (!validate_positive_vector_parameter("dof_weights", values, model_->total_dofs(), reason)) {
        RCLCPP_WARN(
          get_node()->get_logger(),
          "Rejecting invalid dof_weights parameter update: %s",
          reason.c_str());
        result.successful = false;
        result.reason = reason;
        return result;
      }
      solver_.set_weights(eigen_vector_from_std(values));
    } else if (parameter.get_name() == "velocity_limits") {
      const auto values = parameter.as_double_array();
      std::string reason;
      if (!validate_positive_vector_parameter("velocity_limits", values, model_->total_dofs(), reason)) {
        RCLCPP_WARN(
          get_node()->get_logger(),
          "Rejecting invalid velocity_limits parameter update: %s",
          reason.c_str());
        result.successful = false;
        result.reason = reason;
        return result;
      }
      solver_.set_velocity_limits(eigen_vector_from_std(values));
    } else {
      std::string task_id;
      std::string field;
      if (!split_task_parameter_name(parameter.get_name(), task_id, field)) {
        continue;
      }

      std::string message;
      if (field == "gain") {
        const auto values = parameter.as_double_array();
        if (!values_are_non_negative(values)) {
          result.successful = false;
          result.reason = parameter.get_name() + " values must be non-negative";
          return result;
        }
        if (!task_manager_->set_task_gain(task_id, values, message)) {
          result.successful = false;
          result.reason = message;
          return result;
        }
      } else if (field == "gain_scalar") {
        const double value = parameter.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = parameter.get_name() + " must be non-negative";
          return result;
        }
        if (!task_manager_->set_task_gain_scalar(task_id, value, message)) {
          result.successful = false;
          result.reason = message;
          return result;
        }
      }
    }
  }

  return result;
}

void TaskPriorityController::configure_external_interfaces()
{
  hierarchy_state_pub_ = get_node()->create_publisher<msg::HierarchyState>(
    "/hierarchy_state", rclcpp::SystemDefaultsQoS());
  controller_output_pub_ = get_node()->create_publisher<msg::ControllerOutput>(
    "task_priority/output", rclcpp::SystemDefaultsQoS());

  list_tasks_srv_ = get_node()->create_service<srv::ListTasks>(
    "/list_tasks",
    [this](
      const std::shared_ptr<srv::ListTasks::Request>,
      std::shared_ptr<srv::ListTasks::Response> response)
    {
      std::scoped_lock lock(task_mutex_);
      response->tasks = task_manager_->get_task_statuses();
      response->backend_name = backend_ ? backend_->name() : backend_plugin_name_;
      response->solver_method = solver_method_name(
        parse_solver_method(get_node()->get_parameter("solver_method").as_string()));
      response->success = true;
      response->message = "ok";
    });

  set_task_enabled_srv_ = get_node()->create_service<srv::SetTaskEnabled>(
    "/set_task_enabled",
    [this](
      const std::shared_ptr<srv::SetTaskEnabled::Request> request,
      std::shared_ptr<srv::SetTaskEnabled::Response> response)
    {
      std::scoped_lock lock(task_mutex_);
      response->success = task_manager_->set_task_enabled(request->task_id, request->enabled, response->message);
    });

  set_task_joint_activation_srv_ = get_node()->create_service<srv::SetTaskJointActivation>(
    "/set_task_joint_activation",
    [this](
      const std::shared_ptr<srv::SetTaskJointActivation::Request> request,
      std::shared_ptr<srv::SetTaskJointActivation::Response> response)
    {
      std::scoped_lock lock(task_mutex_);
      response->success = task_manager_->set_task_joint_activation(
        request->task_id,
        request->joint_activation,
        response->message);
    });

  set_task_disabled_srv_ = get_node()->create_service<srv::SetTaskDisabled>(
    "/set_task_disabled",
    [this](
      const std::shared_ptr<srv::SetTaskDisabled::Request> request,
      std::shared_ptr<srv::SetTaskDisabled::Response> response)
    {
      std::scoped_lock lock(task_mutex_);
      response->success = task_manager_->set_task_enabled(request->task_id, false, response->message);
    });

  set_solver_config_srv_ = get_node()->create_service<srv::SetSolverConfig>(
    "/set_solver_config",
    [this](
      const std::shared_ptr<srv::SetSolverConfig::Request> request,
      std::shared_ptr<srv::SetSolverConfig::Response> response)
    {
      auto command = std::make_shared<RuntimeTuningCommand>();
      command->type = RuntimeTuningCommand::Type::Solver;
      command->solver_method = request->solver_method;
      command->dls_lambda = request->dls_lambda;
      command->update_dof_weights = request->update_dof_weights;
      command->dof_weights = request->dof_weights;
      response->success = enqueue_runtime_tuning_command(command, response->message);
    });

  set_task_gains_srv_ = get_node()->create_service<srv::SetTaskGains>(
    "/set_task_gains",
    [this](
      const std::shared_ptr<srv::SetTaskGains::Request> request,
      std::shared_ptr<srv::SetTaskGains::Response> response)
    {
      auto command = std::make_shared<RuntimeTuningCommand>();
      command->type = RuntimeTuningCommand::Type::Gains;
      command->gain_updates.reserve(request->updates.size());
      for (const auto & update : request->updates) {
        RuntimeGainUpdate runtime_update;
        runtime_update.task_id = update.task_id;
        runtime_update.field = update.field;
        runtime_update.values = update.values;
        command->gain_updates.push_back(runtime_update);
      }
      response->success = enqueue_runtime_tuning_command(command, response->message);
    });

  reorder_tasks_srv_ = get_node()->create_service<srv::ReorderTasks>(
    "/reorder_tasks",
    [this](
      const std::shared_ptr<srv::ReorderTasks::Request> request,
      std::shared_ptr<srv::ReorderTasks::Response> response)
    {
      std::scoped_lock lock(task_mutex_);
      response->success = task_manager_->reorder_tasks(request->ordered_task_ids, response->message);
    });

  stop_srv_ = get_node()->create_service<std_srvs::srv::Trigger>(
    "/stop_task_priority",
    [this](
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
      std::scoped_lock lock(task_mutex_);
      task_manager_->disable_all_tasks();
      reset_commands();
      publish_zero_controller_output(get_node()->now());
      response->success = true;
      response->message = "Task priority stopped: all tasks disabled and outputs set to zero";
    });

  task_target_subs_.clear();
  task_joint_target_subs_.clear();
  task_joint_trajectory_subs_.clear();
  task_joint_trajectory_action_servers_.clear();
  task_state_pubs_.clear();
  for (const auto & task : task_manager_->tasks()) {
    const auto & task_id = task->id();
    task_state_pubs_[task_id] = get_node()->create_publisher<msg::TaskState>(
      "task_priority/tasks/" + task_id + "/state", rclcpp::SystemDefaultsQoS());

    if (task->plugin_name().find("PoseTask") == std::string::npos) {
      if (task->plugin_name().find("JointNominalTask") != std::string::npos) {
        const auto joint_target_callback = [this, task_id](const Float64MultiArray::SharedPtr msg)
        {
          if (!msg) {
            return;
          }
          std::string message;
          std::scoped_lock lock(task_mutex_);
          if (!task_manager_->set_task_joint_target(task_id, msg->data, message)) {
            RCLCPP_WARN(
              get_node()->get_logger(),
              "Rejected joint target for task '%s': %s",
              task_id.c_str(),
              message.c_str());
          }
        };

        task_joint_target_subs_[task_id] = get_node()->create_subscription<Float64MultiArray>(
          "task_priority/tasks/" + task_id + "/joint_target",
          rclcpp::SystemDefaultsQoS(),
          joint_target_callback);
      } else if (task->plugin_name().find("JointTrajectoryTask") != std::string::npos) {
        const auto trajectory_topic = "task_priority/tasks/" + task_id + "/joint_trajectory";
        task_joint_trajectory_subs_[task_id] = get_node()->create_subscription<JointTrajectory>(
          trajectory_topic,
          rclcpp::SystemDefaultsQoS(),
          [this, task_id](const JointTrajectory::SharedPtr msg)
          {
            if (!msg) {
              return;
            }
            std::string message;
            std::scoped_lock lock(task_mutex_);
            if (!task_manager_->set_task_joint_trajectory(task_id, *msg, message)) {
              RCLCPP_WARN(
                get_node()->get_logger(),
                "Rejected joint trajectory for task '%s': %s",
                task_id.c_str(),
                message.c_str());
            }
          });

        const auto action_name = "task_priority/tasks/" + task_id + "/follow_joint_trajectory";
        task_joint_trajectory_action_servers_[task_id] =
          rclcpp_action::create_server<FollowJointTrajectory>(
          get_node(),
          action_name,
          [](const rclcpp_action::GoalUUID &,
          std::shared_ptr<const FollowJointTrajectory::Goal>)
          {
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
          },
          [this, task_id](const std::shared_ptr<FollowJointTrajectoryGoalHandle>)
          {
            std::string message;
            std::scoped_lock lock(task_mutex_);
            task_manager_->cancel_task_joint_trajectory(task_id, message);
            return rclcpp_action::CancelResponse::ACCEPT;
          },
          [this, task_id](const std::shared_ptr<FollowJointTrajectoryGoalHandle> goal_handle)
          {
            std::thread(
              &TaskPriorityController::execute_task_trajectory_goal,
              this,
              task_id,
              goal_handle).detach();
          });
      }
      continue;
    }

    const auto pose_goal_callback = [this, task_id](const PoseStamped::SharedPtr msg)
    {
      if (!msg) {
        return;
      }
      std::string message;
      std::scoped_lock lock(task_mutex_);
      if (!task_manager_->set_task_pose_goal(task_id, *msg, message)) {
        RCLCPP_WARN(
          get_node()->get_logger(),
          "Rejected pose goal for task '%s': %s",
          task_id.c_str(),
          message.c_str());
      }
    };

    task_target_subs_[task_id] = get_node()->create_subscription<PoseStamped>(
      "task_priority/tasks/" + task_id + "/target",
      rclcpp::SystemDefaultsQoS(),
      pose_goal_callback);
  }
}

void TaskPriorityController::publish_hierarchy_state(const rclcpp::Time & time)
{
  if (!hierarchy_state_pub_) {
    return;
  }

  msg::HierarchyState state_msg;
  state_msg.header.stamp = time;
  state_msg.header.frame_id = get_node()->get_parameter("world_frame").as_string();
  state_msg.backend_name = backend_ ? backend_->name() : backend_plugin_name_;
  state_msg.solver_method = solver_method_name(
    parse_solver_method(get_node()->get_parameter("solver_method").as_string()));
  state_msg.ready = true;
  state_msg.tasks = task_manager_->get_task_statuses();
  hierarchy_state_pub_->publish(state_msg);
}

void TaskPriorityController::publish_task_states(const std::vector<TaskComputation> & task_outputs)
{
  const auto tasks = task_manager_->tasks();
  const auto count = std::min(tasks.size(), task_outputs.size());
  for (size_t i = 0; i < count; ++i) {
    const auto pub_it = task_state_pubs_.find(tasks[i]->id());
    if (pub_it == task_state_pubs_.end()) {
      continue;
    }

    msg::TaskState state_msg;
    state_msg.id = tasks[i]->id();
    state_msg.type = tasks[i]->plugin_name();
    state_msg.enabled = tasks[i]->enabled();
    state_msg.active = task_outputs[i].active;
    state_msg.has_frame_pose = task_outputs[i].has_frame_pose;
    state_msg.frame_id = task_outputs[i].frame_id;
    if (task_outputs[i].has_frame_pose) {
      state_msg.frame_pose = pose_msg_from_isometry(task_outputs[i].frame_pose);
    }
    state_msg.feedforward.clear();

    const auto & error = task_outputs[i].error;
    state_msg.error.resize(static_cast<size_t>(error.size()), 0.0);
    for (Eigen::Index j = 0; j < error.size(); ++j) {
      state_msg.error[static_cast<size_t>(j)] = error(j);
    }

    const auto & velocity = task_outputs[i].desired_velocity;
    state_msg.velocity.resize(static_cast<size_t>(velocity.size()), 0.0);
    for (Eigen::Index j = 0; j < velocity.size(); ++j) {
      state_msg.velocity[static_cast<size_t>(j)] = velocity(j);
    }

    state_msg.target = tasks[i]->current_target();
    pub_it->second->publish(state_msg);
  }
}

void TaskPriorityController::publish_controller_output(
  const rclcpp::Time & time,
  const WholeBodyCommand & command)
{
  if (!controller_output_pub_) {
    return;
  }

  msg::ControllerOutput output_msg;
  output_msg.header.stamp = time;
  output_msg.header.frame_id = get_node()->get_parameter("world_frame").as_string();

  for (const auto dof : active_base_dofs_) {
    output_msg.generalized_velocity_names.push_back(base_dof_name(dof));
  }
  for (const auto & joint : left_arm_joints_) {
    output_msg.generalized_velocity_names.push_back(joint);
  }
  for (const auto & joint : right_arm_joints_) {
    output_msg.generalized_velocity_names.push_back(joint);
  }
  output_msg.generalized_velocity = vector_from_eigen(command.generalized_velocity);

  output_msg.base_velocity_names = {
    "base.linear.x",
    "base.linear.y",
    "base.linear.z",
    "base.angular.x",
    "base.angular.y",
    "base.angular.z"
  };
  output_msg.base_velocity.assign(6, 0.0);
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(active_base_dofs_.size()) &&
       i < command.generalized_velocity.size(); ++i) {
    const int dof = active_base_dofs_[static_cast<size_t>(i)];
    if (dof >= 0 && dof < 6) {
      output_msg.base_velocity[static_cast<size_t>(dof)] = command.generalized_velocity(i);
    }
  }

  output_msg.left_joint_names = left_arm_joints_;
  output_msg.left_arm_velocity.assign(left_arm_joints_.size(), 0.0);
  const Eigen::Index left_offset = static_cast<Eigen::Index>(model_->left_offset());
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(left_arm_joints_.size()) &&
       left_offset + i < command.generalized_velocity.size(); ++i) {
    output_msg.left_arm_velocity[static_cast<size_t>(i)] =
      command.generalized_velocity(left_offset + i);
  }

  output_msg.right_joint_names = right_arm_joints_;
  output_msg.right_arm_velocity.assign(right_arm_joints_.size(), 0.0);
  const Eigen::Index right_offset = static_cast<Eigen::Index>(model_->right_offset());
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(right_arm_joints_.size()) &&
       right_offset + i < command.generalized_velocity.size(); ++i) {
    output_msg.right_arm_velocity[static_cast<size_t>(i)] =
      command.generalized_velocity(right_offset + i);
  }

  controller_output_pub_->publish(output_msg);
}

controller_interface::CallbackReturn TaskPriorityController::on_activate(
  const rclcpp_lifecycle::State &)
{
  reset_commands();
  publish_zero_controller_output(get_node()->now());
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn TaskPriorityController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  reset_commands();
  publish_zero_controller_output(get_node()->now());
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn TaskPriorityController::on_cleanup(
  const rclcpp_lifecycle::State &)
{
  reset_commands();
  publish_zero_controller_output(get_node()->now());
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn TaskPriorityController::on_shutdown(
  const rclcpp_lifecycle::State &)
{
  reset_commands();
  publish_zero_controller_output(get_node()->now());
  return controller_interface::CallbackReturn::SUCCESS;
}

void TaskPriorityController::reset_commands()
{
  for (auto & interface : command_interfaces_) {
    interface.set_value(0.0);
  }
}

void TaskPriorityController::publish_zero_controller_output(const rclcpp::Time & time)
{
  if (!model_) {
    return;
  }

  WholeBodyCommand command;
  command.generalized_velocity = Eigen::VectorXd::Zero(model_->total_dofs());
  publish_controller_output(time, command);
}

void TaskPriorityController::execute_task_trajectory_goal(
  const std::string & task_id,
  const std::shared_ptr<FollowJointTrajectoryGoalHandle> goal_handle)
{
  auto result = std::make_shared<FollowJointTrajectory::Result>();
  std::string message;
  {
    std::scoped_lock lock(task_mutex_);
    if (!task_manager_->set_task_joint_trajectory(task_id, goal_handle->get_goal()->trajectory, message)) {
      result->error_code = FollowJointTrajectory::Result::INVALID_GOAL;
      result->error_string = message;
      goal_handle->abort(result);
      return;
    }
  }

  rclcpp::Rate rate(50.0);
  while (rclcpp::ok()) {
    if (goal_handle->is_canceling()) {
      std::scoped_lock lock(task_mutex_);
      task_manager_->cancel_task_joint_trajectory(task_id, message);
      result->error_code = FollowJointTrajectory::Result::SUCCESSFUL;
      result->error_string = "Joint trajectory task canceled";
      goal_handle->canceled(result);
      return;
    }

    JointTrajectoryTaskStatus status;
    {
      std::scoped_lock lock(task_mutex_);
      status = task_manager_->get_task_joint_trajectory_status(task_id);
    }

    if (status.succeeded) {
      result->error_code = FollowJointTrajectory::Result::SUCCESSFUL;
      result->error_string = status.message;
      goal_handle->succeed(result);
      return;
    }
    if (status.canceled) {
      result->error_code = FollowJointTrajectory::Result::SUCCESSFUL;
      result->error_string = status.message;
      goal_handle->canceled(result);
      return;
    }
    if (status.timed_out) {
      result->error_code = FollowJointTrajectory::Result::GOAL_TOLERANCE_VIOLATED;
      result->error_string = status.message;
      goal_handle->abort(result);
      return;
    }

    rate.sleep();
  }

  result->error_code = FollowJointTrajectory::Result::INVALID_GOAL;
  result->error_string = "ROS shutdown during joint trajectory task execution";
  goal_handle->abort(result);
}

controller_interface::return_type TaskPriorityController::update(
  const rclcpp::Time & time,
  const rclcpp::Duration &)
{
  const size_t expected_command_interface_count =
    6U + left_arm_joints_.size() + right_arm_joints_.size();
  if (command_interfaces_.size() != expected_command_interface_count) {
    RCLCPP_ERROR_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 2000,
      "Unexpected command interface count: %zu (expected %zu)",
      command_interfaces_.size(),
      expected_command_interface_count);
    return controller_interface::return_type::ERROR;
  }

  process_pending_runtime_tuning();

  auto navigator_msg = navigator_buffer_.readFromRT();
  if (!navigator_msg || !(*navigator_msg)) {
    reset_commands();
    return controller_interface::return_type::OK;
  }

  state_.base_position = Eigen::Vector3d(
    (*navigator_msg)->position.position.x,
    (*navigator_msg)->position.position.y,
    (*navigator_msg)->position.position.z);
  state_.base_rpy = Eigen::Vector3d((*navigator_msg)->rpy.x, (*navigator_msg)->rpy.y, (*navigator_msg)->rpy.z);
  state_.navigation_valid = true;

  const auto all_joints = model_->all_joint_names();
  const Eigen::Index joint_count = static_cast<Eigen::Index>(all_joints.size());
  const size_t expected_state_interface_count = all_joints.size() * 2U;
  if (state_interfaces_.size() != expected_state_interface_count) {
    RCLCPP_ERROR_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 2000,
      "Unexpected state interface count: %zu (expected %zu)",
      state_interfaces_.size(),
      expected_state_interface_count);
    reset_commands();
    return controller_interface::return_type::ERROR;
  }

  state_.joint_positions = Eigen::VectorXd::Zero(joint_count);
  state_.joint_velocities = Eigen::VectorXd::Zero(joint_count);
  for (size_t i = 0; i < all_joints.size(); ++i) {
    const double position = state_interfaces_[i].get_value();
    const double velocity = state_interfaces_[all_joints.size() + i].get_value();
    if (!std::isfinite(position) || !std::isfinite(velocity)) {
      RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 2000,
        "State interface has non-finite value for joint '%s'",
        all_joints[i].c_str());
      reset_commands();
      return controller_interface::return_type::OK;
    }
    state_.joint_positions(static_cast<Eigen::Index>(i)) = position;
    state_.joint_velocities(static_cast<Eigen::Index>(i)) = velocity;
  }
  state_.joints_valid = true;

  WholeBodyCommand command;
  {
    if (parameter_update_pending_.load(std::memory_order_acquire)) {
      return controller_interface::return_type::OK;
    }

    std::scoped_lock lock(task_mutex_);
    backend_->update(state_);
    const auto task_outputs = task_manager_->update_all(state_, *backend_);
    command = solver_.solve(task_outputs);
    publish_hierarchy_state(time);
    publish_task_states(task_outputs);
  }

  Eigen::Matrix<double, 6, 1> base_command = Eigen::Matrix<double, 6, 1>::Zero();
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(active_base_dofs_.size()) &&
       i < command.generalized_velocity.size(); ++i) {
    const int dof = active_base_dofs_[static_cast<size_t>(i)];
    if (dof >= 0 && dof < 6) {
      base_command(dof) = command.generalized_velocity(i);
    }
  }

  for (Eigen::Index i = 0; i < 6; ++i) {
    command_interfaces_[i].set_value(base_command(i));
  }

  const Eigen::Index left_offset = static_cast<Eigen::Index>(model_->left_offset());
  const Eigen::Index right_offset = static_cast<Eigen::Index>(model_->right_offset());
  size_t command_interface_index = 6U;
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(left_arm_joints_.size()); ++i) {
    command_interfaces_[command_interface_index].set_value(
      command.generalized_velocity(left_offset + i));
    ++command_interface_index;
  }
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(right_arm_joints_.size()); ++i) {
    command_interfaces_[command_interface_index].set_value(
      command.generalized_velocity(right_offset + i));
    ++command_interface_index;
  }

  publish_controller_output(time, command);

  return controller_interface::return_type::OK;
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::TaskPriorityController,
  controller_interface::ControllerInterface)
