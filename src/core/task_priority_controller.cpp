#include "task_priority_kinematic_control/core/task_priority_controller.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

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
    auto_declare_if_missing(this, "left_tip_frame", std::string("cirtesub/alpha_left/standard_jaws_tool"));
    auto_declare_if_missing(this, "right_tip_frame", std::string("cirtesub/alpha_right/standard_jaws_tool"));
    auto_declare_if_missing(this, "left_arm_joints", std::vector<std::string>{});
    auto_declare_if_missing(this, "right_arm_joints", std::vector<std::string>{});
    auto_declare_if_missing(this, "active_base_dofs", std::vector<int64_t>{0, 1, 2, 3, 4, 5});
    auto_declare_if_missing(this, "solver_method", std::string("dls"));
    auto_declare_if_missing(this, "dls_lambda", 0.05);
    auto_declare_if_missing(this, "dof_weights", std::vector<double>{});
    auto_declare_if_missing(this, "velocity_limits", std::vector<double>{});
    auto_declare_if_missing(this, "navigator_topic", std::string("/cirtesub/navigator/navigation"));
    auto_declare_if_missing(this, "body_velocity_controller_name", std::string("body_velocity_controller"));
    auto_declare_if_missing(
      this, "left_arm_command_topic",
      std::string("/alpha_left_forward_velocity_controller/commands"));
    auto_declare_if_missing(
      this, "right_arm_command_topic",
      std::string("/alpha_right_forward_velocity_controller/commands"));
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
    auto_declare_if_missing(this, prefix + "gain_scalar", 0.5);
    auto_declare_if_missing(this, prefix + "joint_names", std::vector<std::string>{});
    auto_declare_if_missing(this, prefix + "target", std::vector<double>{});
    auto_declare_if_missing(this, prefix + "target_yaw", 0.0);
  }
}

controller_interface::InterfaceConfiguration
TaskPriorityController::command_interface_configuration() const
{
  const std::string body_velocity_name = get_node()->has_parameter("body_velocity_controller_name") ?
    get_node()->get_parameter("body_velocity_controller_name").as_string() :
    std::string("body_velocity_controller");
  std::vector<std::string> names = {
    body_velocity_name + "/linear.x",
    body_velocity_name + "/linear.y",
    body_velocity_name + "/linear.z",
    body_velocity_name + "/angular.x",
    body_velocity_name + "/angular.y",
    body_velocity_name + "/angular.z"
  };

  return {
    controller_interface::interface_configuration_type::INDIVIDUAL,
    names
  };
}

controller_interface::InterfaceConfiguration
TaskPriorityController::state_interface_configuration() const
{
  const auto left_joints = get_node()->has_parameter("left_arm_joints") ?
    get_node()->get_parameter("left_arm_joints").as_string_array() :
    std::vector<std::string>{};
  const auto right_joints = get_node()->has_parameter("right_arm_joints") ?
    get_node()->get_parameter("right_arm_joints").as_string_array() :
    std::vector<std::string>{};
  std::vector<std::string> names;
  names.reserve((left_joints.size() + right_joints.size()) * 2U);
  for (const auto & joint : left_joints) {
    names.push_back(joint + "/position");
    names.push_back(joint + "/velocity");
  }
  for (const auto & joint : right_joints) {
    names.push_back(joint + "/position");
    names.push_back(joint + "/velocity");
  }

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
  left_arm_command_topic_ = get_node()->get_parameter("left_arm_command_topic").as_string();
  right_arm_command_topic_ = get_node()->get_parameter("right_arm_command_topic").as_string();
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
  left_arm_command_pub_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
    left_arm_command_topic_, rclcpp::SystemDefaultsQoS());
  right_arm_command_pub_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
    right_arm_command_topic_, rclcpp::SystemDefaultsQoS());
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
    Eigen::VectorXd w(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
      w(static_cast<Eigen::Index>(i)) = weights[i];
    }
    solver_.set_weights(w);
  }

  const auto limits = get_node()->get_parameter("velocity_limits").as_double_array();
  if (!limits.empty()) {
    Eigen::VectorXd v(limits.size());
    for (size_t i = 0; i < limits.size(); ++i) {
      v(static_cast<Eigen::Index>(i)) = limits[i];
    }
    solver_.set_velocity_limits(v);
  }
}

void TaskPriorityController::refresh_task_manager()
{
  task_manager_ = std::make_unique<TaskManager>(
    get_node()->get_node_parameters_interface(), get_node()->get_logger());
  task_manager_->configure(TaskContext{model_});
}

void TaskPriorityController::configure_external_interfaces()
{
  hierarchy_state_pub_ = get_node()->create_publisher<msg::HierarchyState>(
    "/hierarchy_state", rclcpp::SystemDefaultsQoS());

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

  set_task_disabled_srv_ = get_node()->create_service<srv::SetTaskDisabled>(
    "/set_task_disabled",
    [this](
      const std::shared_ptr<srv::SetTaskDisabled::Request> request,
      std::shared_ptr<srv::SetTaskDisabled::Response> response)
    {
      std::scoped_lock lock(task_mutex_);
      response->success = task_manager_->set_task_enabled(request->task_id, false, response->message);
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

  task_target_subs_.clear();
  task_joint_target_subs_.clear();
  task_state_pubs_.clear();
  for (const auto & task : task_manager_->tasks()) {
    const auto & task_id = task->id();
    task_state_pubs_[task_id] = get_node()->create_publisher<msg::TaskState>(
      "tasks/" + task_id + "/state", rclcpp::SystemDefaultsQoS());

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
          "/task_priority_controller/tasks/" + task_id + "/joint_target",
          rclcpp::SystemDefaultsQoS(),
          joint_target_callback);
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
      "/task_priority_controller/tasks/" + task_id + "/target",
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

controller_interface::CallbackReturn TaskPriorityController::on_activate(
  const rclcpp_lifecycle::State &)
{
  reset_commands();
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn TaskPriorityController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  reset_commands();
  return controller_interface::CallbackReturn::SUCCESS;
}

void TaskPriorityController::reset_commands()
{
  for (auto & interface : command_interfaces_) {
    interface.set_value(0.0);
  }

  if (left_arm_command_pub_) {
    std_msgs::msg::Float64MultiArray left_msg;
    left_msg.data.assign(left_arm_joints_.size(), 0.0);
    left_arm_command_pub_->publish(left_msg);
  }
  if (right_arm_command_pub_) {
    std_msgs::msg::Float64MultiArray right_msg;
    right_msg.data.assign(right_arm_joints_.size(), 0.0);
    right_arm_command_pub_->publish(right_msg);
  }
}

controller_interface::return_type TaskPriorityController::update(
  const rclcpp::Time & time,
  const rclcpp::Duration &)
{
  if (command_interfaces_.size() != 6U) {
    RCLCPP_ERROR_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 2000,
      "Unexpected command interface count: %zu", command_interfaces_.size());
    return controller_interface::return_type::ERROR;
  }

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

  const Eigen::Index joint_count = static_cast<Eigen::Index>(left_arm_joints_.size() + right_arm_joints_.size());
  state_.joint_positions = Eigen::VectorXd::Zero(joint_count);
  state_.joint_velocities = Eigen::VectorXd::Zero(joint_count);
  for (Eigen::Index i = 0; i < joint_count; ++i) {
    const Eigen::Index state_offset = i * 2;
    state_.joint_positions(i) = state_interfaces_[state_offset].get_value();
    state_.joint_velocities(i) = state_interfaces_[state_offset + 1].get_value();
  }
  state_.joints_valid = true;

  WholeBodyCommand command;
  {
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
  std_msgs::msg::Float64MultiArray left_msg;
  left_msg.data.resize(left_arm_joints_.size(), 0.0);
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(left_arm_joints_.size()); ++i) {
    left_msg.data[static_cast<size_t>(i)] = command.generalized_velocity(left_offset + i);
  }
  std_msgs::msg::Float64MultiArray right_msg;
  right_msg.data.resize(right_arm_joints_.size(), 0.0);
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(right_arm_joints_.size()); ++i) {
    right_msg.data[static_cast<size_t>(i)] = command.generalized_velocity(right_offset + i);
  }

  if (left_arm_command_pub_) {
    left_arm_command_pub_->publish(left_msg);
  }
  if (right_arm_command_pub_) {
    right_arm_command_pub_->publish(right_msg);
  }

  return controller_interface::return_type::OK;
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::TaskPriorityController,
  controller_interface::ControllerInterface)
